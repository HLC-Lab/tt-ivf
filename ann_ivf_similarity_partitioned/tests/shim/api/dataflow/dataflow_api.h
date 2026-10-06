#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#define tt_l1_ptr
#define ASSERT(x) do { if (!(x)) throw std::runtime_error("Kernel assertion: " #x); } while (false)
constexpr uint32_t NOC_MAX_BURST_SIZE = 8192;
namespace tt { namespace CB {
enum : uint32_t { c_in0=0, c_in1=1, c_in2=2, c_in3=3, c_out0=16, c_out1=17, c_intermed5=29 };
}}
struct SimCB { uint32_t base, bytes, capacity, write=0, read=0, count=0; std::mutex mutex; std::condition_variable changed; };
struct SimCore { std::vector<uint32_t> memory=std::vector<uint32_t>(512*1024/4); std::map<uint32_t,std::unique_ptr<SimCB>> cbs; };
struct SimDRAM { uint32_t page; std::vector<uint8_t> bytes; };
struct SimState { uint32_t core; std::vector<uint32_t> args; };
inline std::vector<std::unique_ptr<SimCore>> sim_cores;
inline std::map<uint32_t,SimDRAM> sim_dram;
// Attribute payload reads to the issuing core, including split NoC packets.
// This checks the transport role rather than inferring it from output scores.
inline std::mutex sim_read_accounting_mutex;
inline std::map<std::pair<uint32_t,uint32_t>,uint64_t> sim_dram_read_bytes;
inline thread_local SimState sim_state;
inline thread_local uint64_t sim_read_source;
inline thread_local uint32_t sim_read_length;
inline thread_local uint32_t sim_transaction_id = 0;
inline thread_local bool sim_track_direct_reads = false;
struct SimRead { uint64_t source; uint32_t dest, bytes, id; };
struct SimWrite { uint32_t source; uint64_t dest; uint32_t bytes; };
inline thread_local std::vector<SimRead> sim_reads;
inline thread_local std::vector<SimWrite> sim_writes;
inline std::atomic<uint32_t> sim_partial_barriers = 0;
inline std::atomic<uint32_t> sim_direct_max_tags = 0;
inline std::atomic<uint32_t> sim_direct_partial_barriers = 0;
inline std::atomic<uint32_t> sim_out_of_order_completions = 0;
#ifndef SIM_COMPLETE_AHEAD
#define SIM_COMPLETE_AHEAD 0
#endif
inline uint8_t* ivf_sim_l1(uint32_t address) { return reinterpret_cast<uint8_t*>(sim_cores.at(sim_state.core)->memory.data())+address; }
inline uint32_t get_arg_val_raw(uint32_t i) { return sim_state.args.at(i); }
template<class T> T get_arg_val(uint32_t i) { return static_cast<T>(get_arg_val_raw(i)); }
inline uint32_t get_semaphore(uint32_t) { return 32; }
inline void invalidate_l1_cache() { std::atomic_thread_fence(std::memory_order_seq_cst); std::this_thread::yield(); }
inline uint64_t get_noc_addr(uint32_t x, uint32_t, uint32_t addr) { return uint64_t(x+1)<<32 | addr; }
inline uint64_t get_noc_multicast_addr(uint32_t xmax, uint32_t ymax, uint32_t xmin, uint32_t ymin, uint32_t addr) {
    if (xmax < xmin || ymin != 0 || ymax != 0) throw std::runtime_error("NoC 1 multicast bounds are wrong");
    return uint64_t(xmin) << 48 | uint64_t(xmax) << 40 | addr;
}
inline uint8_t* sim_address(uint64_t address) {
    uint32_t space=address>>32, offset=address;
    if (space >= 1000) return sim_dram.at(space).bytes.data()+offset;
    return reinterpret_cast<uint8_t*>(sim_cores.at(space-1)->memory.data())+offset;
}
template<bool Dram> struct InterleavedAddrGen {
    uint32_t bank_base_address, page_size;
    uint64_t get_noc_addr(uint32_t page) const { return uint64_t(bank_base_address)<<32 | uint64_t(page)*page_size; }
};
inline void noc_async_read(uint64_t source,uint32_t dest,uint32_t bytes) {
    ASSERT(uint32_t(source)%32 == dest%32);
    ASSERT(bytes && dest+bytes <= sim_cores.at(sim_state.core)->memory.size()*4);
    if ((source>>32)>=1000) {
        ASSERT(uint32_t(source)+bytes <= sim_dram.at(source>>32).bytes.size());
        std::lock_guard lock(sim_read_accounting_mutex);
        sim_dram_read_bytes[{sim_state.core,uint32_t(source>>32)}] += bytes;
    }
    for (const auto& read : sim_reads)
        ASSERT(dest+bytes<=read.dest || dest>=read.dest+read.bytes);
    for (const auto& [unused,ptr] : sim_cores.at(sim_state.core)->cbs) {
        auto& cb=*ptr;
        std::lock_guard lock(cb.mutex);
        for (uint32_t i=0;i<cb.count;++i) {
            const uint32_t occupied=cb.base+((cb.read+i)%cb.capacity)*cb.bytes;
            ASSERT(dest+bytes<=occupied || dest>=occupied+cb.bytes);
        }
    }
    // Pending slots remain visibly invalid until their DMA completes.
    std::memset(ivf_sim_l1(dest),0xdd,bytes);
    sim_reads.push_back({source,dest,bytes,sim_transaction_id});
    if (sim_track_direct_reads) {
        uint32_t tags=0;
        for (const auto& read : sim_reads) if (read.id) tags|=1u<<read.id;
        uint32_t count=0;
        for (;tags;tags&=tags-1) ++count;
        uint32_t previous=sim_direct_max_tags.load();
        while (previous<count && !sim_direct_max_tags.compare_exchange_weak(previous,count)) {}
    }
}
inline void noc_async_write(uint32_t source,uint64_t dest,uint32_t bytes) {
    ASSERT(source%16 == uint32_t(dest)%16);
    sim_writes.push_back({source,dest,bytes});
}
inline void sim_finish_reads(uint32_t id, bool all) {
    // Optionally complete a later page before the requested oldest page.
    // Remaining tags stay poisoned until a subsequent barrier.
    uint32_t ahead=id;
    if constexpr (SIM_COMPLETE_AHEAD) {
        if (!all && !sim_reads.empty()) ahead=sim_reads.back().id;
        if (ahead!=id) ++sim_out_of_order_completions;
    }
    for (size_t i=sim_reads.size(); i>0; --i) {
        const auto read=sim_reads[i-1];
        // Reverse packet order also tests that IDs may arrive before vectors.
        if (!all && read.id!=id && read.id!=ahead) continue;
        std::memcpy(ivf_sim_l1(read.dest),sim_address(read.source),read.bytes);
        sim_reads.erase(sim_reads.begin()+i-1);
    }
    if (!all && !sim_reads.empty()) {
        ++sim_partial_barriers;
        if (sim_track_direct_reads) ++sim_direct_partial_barriers;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
inline void noc_async_read_barrier() { sim_finish_reads(0,true); }
inline void noc_async_write_barrier() {
    for (auto i=sim_writes.rbegin(); i!=sim_writes.rend(); ++i) {
        if (i->bytes==4 && (i->dest>>32)<1000) {
            uint32_t value;
            std::memcpy(&value,ivf_sim_l1(i->source),4);
            __atomic_store_n(reinterpret_cast<uint32_t*>(sim_address(i->dest)),value,__ATOMIC_RELEASE);
        } else std::memcpy(sim_address(i->dest),ivf_sim_l1(i->source),i->bytes);
    }
    sim_writes.clear();
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
inline void noc_async_atomic_barrier() { std::atomic_thread_fence(std::memory_order_seq_cst); }
template<class G> void noc_async_read_page(uint32_t page,const G& gen,uint32_t dest) {
    ASSERT(sim_transaction_id==0);
    if (sim_track_direct_reads) ASSERT(sim_reads.empty()); // List/query boundaries must drain.
    noc_async_read(gen.get_noc_addr(page),dest,gen.page_size);
}
template<class G> void noc_async_write_page(uint32_t page,const G& gen,uint32_t source) { noc_async_write(source,gen.get_noc_addr(page),gen.page_size); }
inline void noc_async_write_multicast(uint32_t source,uint64_t dest,uint32_t bytes,uint32_t workers) {
    const uint32_t xmin=dest>>48, xmax=(dest>>40)&0xffu;
    if (xmax-xmin+1 != workers) throw std::runtime_error("Multicast recipient count is wrong");
    for(uint32_t x=xmin;x<=xmax;++x) noc_async_write(source,get_noc_addr(x,0,uint32_t(dest)),bytes);
}
inline void noc_async_read_one_packet_set_state(uint64_t src,uint32_t bytes) { sim_read_source=src; sim_read_length=bytes; }
inline void noc_async_read_set_trid(uint32_t id) { ASSERT(id<=15); sim_transaction_id=id; }
inline void noc_async_read_one_packet_with_state_with_trid(uint32_t base,uint32_t offset,uint32_t dest,uint32_t id) {
    ASSERT(id==sim_transaction_id && sim_read_length<=NOC_MAX_BURST_SIZE);
    noc_async_read((sim_read_source&0xffffffff00000000ull)|(base+offset),dest,sim_read_length);
}
inline void noc_async_read_barrier_with_trid(uint32_t id) { sim_finish_reads(id,false); }
inline void noc_semaphore_inc(uint64_t dest,uint32_t value) {
    ASSERT(sim_writes.empty());
    __atomic_fetch_add(reinterpret_cast<uint32_t*>(sim_address(dest)),value,__ATOMIC_SEQ_CST);
}
inline void noc_semaphore_wait_min(volatile uint32_t* p,uint32_t n) {
    while (__atomic_load_n(const_cast<uint32_t*>(p),__ATOMIC_SEQ_CST)<n) std::this_thread::yield();
}
inline SimCB& sim_cb(uint32_t id) { return *sim_cores.at(sim_state.core)->cbs.at(id); }
inline bool cb_pages_reservable_at_back(uint32_t id,uint32_t pages) {
    auto& cb=sim_cb(id); std::lock_guard lock(cb.mutex);
    ASSERT(pages && pages<=cb.capacity);
    return cb.capacity-cb.count>=pages;
}
inline void cb_reserve_back(uint32_t id,uint32_t pages) {
    auto& cb=sim_cb(id); std::unique_lock lock(cb.mutex);
    ASSERT(pages && pages<=cb.capacity);
    if(!cb.changed.wait_for(lock,std::chrono::seconds(8),[&]{return cb.capacity-cb.count>=pages;})) throw std::runtime_error("CB reserve timeout");
}
inline void cb_wait_front(uint32_t id,uint32_t pages) {
    auto& cb=sim_cb(id); std::unique_lock lock(cb.mutex);
    if(!cb.changed.wait_for(lock,std::chrono::seconds(8),[&]{return cb.count>=pages;})) throw std::runtime_error("CB wait timeout");
}
inline uint32_t get_write_ptr(uint32_t id) { auto& cb=sim_cb(id); std::lock_guard lock(cb.mutex); return cb.base+cb.write*cb.bytes; }
inline uint32_t get_read_ptr(uint32_t id) { auto& cb=sim_cb(id); std::lock_guard lock(cb.mutex); return cb.base+cb.read*cb.bytes; }
inline void cb_push_back(uint32_t id,uint32_t pages) {
    auto& cb=sim_cb(id);
    { std::lock_guard lock(cb.mutex);
      ASSERT(pages && cb.count+pages<=cb.capacity && cb.write+pages<=cb.capacity);
      const uint32_t start=cb.base+cb.write*cb.bytes, end=start+pages*cb.bytes;
      for (const auto& read : sim_reads) ASSERT(end<=read.dest || start>=read.dest+read.bytes);
      cb.write+=pages;
      if (cb.write==cb.capacity) cb.write=0;
      cb.count+=pages;
    }
    cb.changed.notify_all();
}
inline void cb_pop_front(uint32_t id,uint32_t pages) {
    auto& cb=sim_cb(id);
    { std::lock_guard lock(cb.mutex);
      ASSERT(pages && cb.count>=pages && cb.read+pages<=cb.capacity);
      cb.read+=pages;
      if (cb.read==cb.capacity) cb.read=0;
      cb.count-=pages;
    }
    cb.changed.notify_all();
}
