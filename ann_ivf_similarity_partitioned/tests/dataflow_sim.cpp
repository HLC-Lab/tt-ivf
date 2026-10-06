#include "shim/api/dataflow/dataflow_api.h"
#include "../partition_protocol.hpp"
#include "../fine_tile_layout.hpp"
#include <iostream>
#include <future>
#include <set>
#define IVF_TRANSPORT_SIM
#ifndef SIM_RELAY
#define SIM_RELAY 1
#endif
#ifndef SIM_MASK
#define SIM_MASK 0
#endif
#ifndef SIM_SEPARATE
#define SIM_SEPARATE 0
#endif
#ifndef SIM_WIDTH
#define SIM_WIDTH 4
#endif
#ifndef SIM_DEPTH
#define SIM_DEPTH 2
#endif
#ifndef SIM_INPUT
#define SIM_INPUT 2
#endif
#ifndef SIM_STAGGER
#define SIM_STAGGER 1
#endif
#ifndef SIM_DIRECT_PREFETCH
#define SIM_DIRECT_PREFETCH 1
#endif
#ifndef SIM_SHORT_LISTS
#define SIM_SHORT_LISTS 0
#endif
constexpr uint32_t vector_bytes=SIM_WIDTH*2048, first_pages=17, second_pages=9;
constexpr uint32_t receiver_compile[] = {SIM_WIDTH,SIM_RELAY,SIM_MASK,SIM_INPUT,SIM_DIRECT_PREFETCH};
#define get_compile_time_arg_val(i) receiver_compile[i]
#define kernel_main receiver_main
#include "../kernels/dataflow/worker_receiver.cpp"
#undef kernel_main
#undef get_compile_time_arg_val
constexpr uint32_t leader_compile[] = {SIM_WIDTH,SIM_RELAY,SIM_STAGGER,SIM_DEPTH};
#define get_compile_time_arg_val(i) leader_compile[i]
#define kernel_main leader_main
#include "../kernels/dataflow/leader_reader.cpp"
#undef kernel_main
#undef get_compile_time_arg_val
#define kernel_main writer_main
#include "../kernels/dataflow/worker_writer.cpp"
#undef kernel_main
#define kernel_main gather_main
#include "../kernels/dataflow/leader_gather_writer.cpp"
#undef kernel_main
using namespace ivf_partitioned;
void require(bool ok,const char* what) { if(!ok) throw std::runtime_error(what); }
void cb_setup(uint32_t core,uint32_t id,uint32_t pages,uint32_t bytes) {
    // Keep CBs disjoint from the largest raw leader staging region.
    uint32_t base=(core>=2 && core<6?64:320)*1024;
    for (const auto& [unused,existing] : sim_cores[core]->cbs) base=std::max(base,existing->base+existing->bytes*existing->capacity);
    require(base+pages*bytes<=sim_cores[core]->memory.size()*4,"Simulated CB exceeds L1");
    auto cb=std::make_unique<SimCB>(); cb->base=base; cb->bytes=bytes; cb->capacity=pages;
    sim_cores[core]->cbs.emplace(id,std::move(cb));
}
Record consume_record() {
    cb_wait_front(tt::CB::c_intermed5,1); Record r;
    std::memcpy(&r,ivf_sim_l1(get_read_ptr(tt::CB::c_intermed5)),sizeof(r));
    cb_pop_front(tt::CB::c_intermed5,1); return r;
}
void worker_compute_sim(uint32_t batches) {
    for(uint32_t b=0;b<batches;++b) {
        const auto h=consume_record();
        cb_wait_front(tt::CB::c_in0,SIM_WIDTH);
        const auto* query=ivf_sim_l1(get_read_ptr(tt::CB::c_in0));
        require(std::memcmp(query,sim_dram.at(1001).bytes.data()+h.word[0]*vector_bytes,vector_bytes)==0,"Query broadcast corrupted or wrong batch");
        for(uint32_t t=0;t<h.word[2];++t) {
            const auto list=consume_record();
            for(uint32_t page=0;page<list.word[2];++page) {
                cb_wait_front(tt::CB::c_in1,SIM_WIDTH);
                require(std::memcmp(ivf_sim_l1(get_read_ptr(tt::CB::c_in1)),sim_dram.at(1002).bytes.data()+(list.word[1]+page)*vector_bytes,vector_bytes)==0,"Candidate transfer corrupted/reordered");
                // Actual compute releases vectors after matmul, then retains
                // the ID page through local top-k. The rings free separately.
                cb_pop_front(tt::CB::c_in1,SIM_WIDTH);
                cb_wait_front(tt::CB::c_in2,1);
                require(std::memcmp(ivf_sim_l1(get_read_ptr(tt::CB::c_in2)),sim_dram.at(1003).bytes.data()+(list.word[1]+page)*4096,4096)==0,"Int32 ID transfer corrupted/reordered");
                if constexpr(SIM_MASK==1) if(needs_mask(page,list.word[2],list.word[4])) {
                    cb_wait_front(tt::CB::c_in3,1);
                    auto* mask=reinterpret_cast<uint16_t*>(ivf_sim_l1(get_read_ptr(tt::CB::c_in3)));
                    for(uint32_t q=0;q<32;++q) for(uint32_t lane=0;lane<32;++lane)
                        require(mask[fine_tile_offset(q,lane)]==(lane<list.word[4]?0:invalid_score_bf16),"Additive mask is misaligned");
                    cb_pop_front(tt::CB::c_in3,1);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(sim_state.core%2?100:20));
                cb_pop_front(tt::CB::c_in2,1);
            }
        }
        cb_pop_front(tt::CB::c_in0,SIM_WIDTH);
        cb_reserve_back(tt::CB::c_out0,1); cb_reserve_back(tt::CB::c_out1,1);
        // Include the global batch in both payloads to detect stale generations.
        const uint8_t result=sim_state.core+h.word[0]*8;
        std::memset(ivf_sim_l1(get_write_ptr(tt::CB::c_out0)),result,2048);
        std::memset(ivf_sim_l1(get_write_ptr(tt::CB::c_out1)),result,4096);
        cb_push_back(tt::CB::c_out0,1); cb_push_back(tt::CB::c_out1,1);
    }
}
void aggregate_sim(uint32_t batches) {
    for(uint32_t b=0;b<batches;++b) {
        const auto h=consume_record();
        std::set<uint8_t> seen;
        for(uint32_t w=0;w<h.word[2];++w) {
            cb_wait_front(tt::CB::c_in0,1); cb_wait_front(tt::CB::c_in1,1);
            const auto value=ivf_sim_l1(get_read_ptr(tt::CB::c_in0))[0];
            const uint8_t first_worker=h.word[0]==1?4:2;
            require(value==first_worker+h.word[0]*8 || value==first_worker+1+h.word[0]*8,"Gather read an unwritten or stale partial result");
            require(seen.insert(value).second,"Gather consumed one worker twice");
            require(ivf_sim_l1(get_read_ptr(tt::CB::c_in1))[0]==value,"Partial value and ID generations disagree");
            cb_pop_front(tt::CB::c_in0,1); cb_pop_front(tt::CB::c_in1,1);
        }
        cb_reserve_back(tt::CB::c_out0,1); cb_reserve_back(tt::CB::c_out1,1);
        std::memset(ivf_sim_l1(get_write_ptr(tt::CB::c_out0)),h.word[0]+20,2048);
        std::memset(ivf_sim_l1(get_write_ptr(tt::CB::c_out1)),h.word[0]+30,4096);
        cb_push_back(tt::CB::c_out0,1); cb_push_back(tt::CB::c_out1,1);
    }
}
int main() {
    try {
        constexpr uint32_t empty_core=SIM_SEPARATE?8:6;
        for(uint32_t i=0;i<=empty_core;++i) sim_cores.push_back(std::make_unique<SimCore>());
        for(uint32_t i=0;i<empty_core;++i) {
            const bool worker=i>=2 && i<6;
            cb_setup(i,tt::CB::c_intermed5,4,32);
            cb_setup(i,tt::CB::c_in0,worker?2*SIM_WIDTH:2,2048);
            cb_setup(i,tt::CB::c_in1,worker?SIM_INPUT*SIM_WIDTH:2,worker?2048:4096);
            if(worker) { cb_setup(i,tt::CB::c_in2,SIM_INPUT,4096); cb_setup(i,tt::CB::c_in3,SIM_INPUT,2048); }
            cb_setup(i,tt::CB::c_out0,2,2048); cb_setup(i,tt::CB::c_out1,2,4096);
        }
        sim_dram[1001]={vector_bytes,std::vector<uint8_t>(3*vector_bytes)};
        sim_dram[1002]={vector_bytes,std::vector<uint8_t>((first_pages+second_pages)*vector_bytes)};
        sim_dram[1003]={4096,std::vector<uint8_t>((first_pages+second_pages)*4096)};
        for(uint32_t p=0;p<3;++p) std::fill_n(sim_dram[1001].bytes.begin()+p*vector_bytes,vector_bytes,p+7);
        for(uint32_t p=0;p<first_pages+second_pages;++p) { std::fill_n(sim_dram[1002].bytes.begin()+p*vector_bytes,vector_bytes,p+70); std::fill_n(sim_dram[1003].bytes.begin()+p*4096,4096,p+100); }
        for(uint32_t id : {1004u,1005u}) sim_dram[id]={id==1004?2048u:4096u,std::vector<uint8_t>(3*4*(id==1004?2048:4096),0)};
        sim_dram[1006]={2048,std::vector<uint8_t>(3*2048,0)};
        sim_dram[1007]={4096,std::vector<uint8_t>(3*4096,0)};
        std::vector<Record> records;
        uint32_t leader_first[2],worker_first[2][2];
        const std::vector<uint32_t> queues[2]={{2,0},{1}};
        std::vector<Record> lists[2];
        if constexpr(SIM_SHORT_LISTS) {
            for(uint32_t w=0;w<2;++w) {
                const uint32_t pages=w?second_pages:first_pages;
                for(uint32_t page=0;page<pages;++page)
                    lists[w].push_back(Record{{10+w*32+page,(w?first_pages:0)+page,1,page%2?32u:1u,page%2?32u:1u}});
            }
        } else {
            // Preserve a long list that fills all 15 transaction IDs, then
            // drain into a one-page partial list; worker 1 has two full lists.
            lists[0]={Record{{10,0,16,512,32}},Record{{11,16,1,1,1}}};
            lists[1]={Record{{12,first_pages,5,160,32}},Record{{13,first_pages+5,4,128,32}}};
        }
        // Every script starts at the end of a cache page. A list transition
        // forces a normal transaction-ID-zero script read after pending DMA.
        auto align_script=[&] {
            while(records.size()%records_per_page!=records_per_page-2) records.push_back(Record{});
        };
        for(uint32_t p=0;p<2;++p) {
            align_script();
            leader_first[p]=records.size();
            for(auto bid:queues[p]) {
                const bool empty=bid==1;
                const uint32_t tasks=lists[0].size()+(empty?0:lists[1].size());
                records.push_back(Record{{bid,32,tasks,first_pages+(empty?0:second_pages)}});
                for(uint32_t w=0;w<(empty?1u:2u);++w)
                    for(const auto& list:lists[w])
                        records.push_back(Record{{w,list.word[0],list.word[1],list.word[2],list.word[3]}});
            }
            for(uint32_t w=0;w<2;++w) {
                align_script();
                worker_first[p][w]=records.size();
                for(auto bid:queues[p]) {
                    const bool empty=bid==1 && w==1;
                    records.push_back(Record{{bid,32,empty?0u:static_cast<uint32_t>(lists[w].size()),empty?0u:(w?second_pages:first_pages)}});
                    if(!empty) for(const auto& list:lists[w]) records.push_back(list);
                }
            }
        }
        sim_dram[1000]={4096,std::vector<uint8_t>(((records.size()*32+4095)/4096)*4096,0)};
        std::memcpy(sim_dram[1000].bytes.data(),records.data(),records.size()*32);
        std::vector<std::thread> threads;
        std::atomic<uint32_t> finished=0;
        std::mutex errors_mutex; std::vector<std::string> errors;
        auto start=[&](uint32_t core,std::vector<uint32_t> args,auto work) {
            threads.emplace_back([&,core,args=std::move(args),work]{
                sim_state={core,args};
                try {
                    work();
                    require(sim_reads.empty() && sim_writes.empty(),"Kernel returned with pending DMA");
                } catch(const std::exception& e) {
                    std::lock_guard lock(errors_mutex);
                    errors.push_back("Core "+std::to_string(core)+": "+e.what());
                }
                ++finished;
            });
        };
        constexpr uint32_t ctrl=1024;
        for(uint32_t p=0;p<2;++p) {
            const uint32_t count=queues[p].size();
            const uint32_t agg=SIM_SEPARATE?6+p:p;
            // Partition 0 uses multicast; partition 1 remains unicast.
            std::vector<uint32_t> args={1000,leader_first[p],count,ctrl,1001,1002,1003,2,0,12,6,p,first_pages+second_pages,p==0,2+p*2,0,3+p*2,0,0};
            for(uint32_t w=0;w<2;++w) { args.push_back(2+p*2+w); args.push_back(0); }
            for(uint32_t bank=0;bank<12;++bank) args.push_back(bank/2);
            start(p,args,leader_main);
            args={1000,leader_first[p],count,ctrl,1004,1005,4,2,1006,1007,p,0,ctrl,UINT32_MAX,p*2,p*2+1};
            start(agg,args,gather_main);
            start(agg,{},[count]{aggregate_sim(count);});
            for(uint32_t w=0;w<2;++w) {
                const uint32_t core=2+p*2+w;
                args={1000,worker_first[p][w],count,ctrl,p,0,ctrl,w,0,1002,1003,0};
                start(core,args,[]{ sim_track_direct_reads=!SIM_RELAY && SIM_DIRECT_PREFETCH; receiver_main(); });
                args={count,p*2+w,4,1004,1005,agg,0,ctrl,w,ctrl,1000,worker_first[p][w],0};
                start(core,args,writer_main);
                start(core,{},[count]{worker_compute_sim(count);});
            }
        }
        // Include a leader with an empty queue. It must return without waiting.
        start(empty_core,{1000,0,0,ctrl,1001,1002,1003,1,0,12,6,2,2,0,0,0,0,0,UINT32_MAX,2,0},leader_main);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(finished<threads.size() && std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if(finished<threads.size()) {
            std::lock_guard lock(errors_mutex);
            for(const auto& error:errors) std::cerr<<error<<'\n';
            std::cerr<<"Actual dataflow kernels deadlocked\n";
            std::_Exit(2);
        }
        for(auto& thread:threads) thread.join();
        for(const auto& error:errors) std::cerr<<error<<'\n';
        require(errors.empty(),"Dataflow simulation failed");
        for (uint32_t core=0;core<=empty_core;++core) {
            const bool leader=core<2, worker=core>=2 && core<6;
            const uint64_t query_bytes=leader?queues[core].size()*vector_bytes:0;
            uint64_t pages=0;
            if (leader || worker) {
                const uint32_t partition=leader?core:(core-2)/2;
                for (auto batch : queues[partition]) {
                    if (leader || core%2==0) pages+=first_pages;
                    if (batch!=1 && (leader || core%2==1)) pages+=second_pages;
                }
            }
            require(sim_dram_read_bytes[{core,1001}]==query_bytes,"Only leaders must read query payloads exactly once per batch");
            const bool payload_reader=SIM_RELAY?leader:worker;
            require(sim_dram_read_bytes[{core,1002}]==(payload_reader?pages*vector_bytes:0),
                    "Candidate vector DRAM reads came from the wrong role or were duplicated/missing");
            require(sim_dram_read_bytes[{core,1003}]==(payload_reader?pages*4096:0),
                    "Candidate ID DRAM reads came from the wrong role or were duplicated/missing");
        }
        for(const auto& core:sim_cores) for(const auto& [unused,cb]:core->cbs)
            require(cb->count==0,"Unconsumed pages or unexpected mask tiles remain");
        if constexpr(SIM_RELAY && SIM_DEPTH>1) require(sim_partial_barriers>0,"Per-ID barriers were never exercised with other pending reads");
        if constexpr(!SIM_RELAY && SIM_DIRECT_PREFETCH) {
            constexpr uint32_t expected=SIM_SHORT_LISTS?1:(SIM_INPUT<15?SIM_INPUT:15);
            require(sim_direct_max_tags==expected,"Direct reader never filled its configured read window");
            if constexpr(SIM_INPUT>1 && !SIM_SHORT_LISTS) require(sim_direct_partial_barriers>0,"Direct reader drained every transaction at each barrier");
        }
        if constexpr(SIM_COMPLETE_AHEAD) require(sim_out_of_order_completions>0,"Out-of-order DMA completion was not exercised");
        for(uint32_t b=0;b<3;++b) {
            require(sim_dram[1006].bytes[b*2048]==b+20,"Final value written at wrong global batch offset");
            require(sim_dram[1007].bytes[b*4096]==b+30,"Final index written at wrong global batch offset");
        }
        std::cout<<"Actual dataflow kernels passed: relay="<<SIM_RELAY<<", mask="<<SIM_MASK<<", separate="<<SIM_SEPARATE
                 <<", width="<<SIM_WIDTH<<", relay/input depths="<<SIM_DEPTH<<'/'<<SIM_INPUT
                 <<", direct prefetch="<<SIM_DIRECT_PREFETCH<<", short lists="<<SIM_SHORT_LISTS
                 <<", payload-reader roles and exact bytes, deferred DMA, wrapped rings/scripts, unequal queues, empty worker/partition, reordered global batches.\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
