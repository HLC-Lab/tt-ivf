// Compile the actual worker kernel against a CB/format lifecycle model.
// Compare sampled and unsampled API sequences; this is not a math emulator.
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "../partition_protocol.hpp"
#define UNPACK(...) do { __VA_ARGS__ } while (false)
#define PACK(...) do { __VA_ARGS__ } while (false)
#define IVF_SCOPE_TEST
namespace tt::CB {
enum : uint32_t { c_in0=0, c_in1=1, c_in2=2, c_in3=3, c_out0=16, c_out1=17,
                 c_intermed0=24, c_intermed1=25, c_intermed3=27, c_intermed4=28, c_intermed5=29 };
}
using namespace ivf_partitioned;
void require(bool ok) { if (!ok) throw std::runtime_error("Worker scope/CB/format lifecycle error"); }
constexpr uint32_t width=4;
constexpr uint32_t get_compile_time_arg_val(uint32_t i) { return i ? SIM_MASK : width; }
uint32_t profile_batch=UINT32_MAX, metadata_cursor=0;
template<class T> T get_arg_val(uint32_t i) { return i ? profile_batch : 3; }
std::vector<Record> records;
enum class Format { BF16, Int32 };
Format source_format=Format::Int32, pack_format=Format::Int32;
std::array<Format,4> destination;
std::array<bool,4> loaded{};
std::array<uint32_t,32> queued{};
alignas(16) std::array<std::array<uint32_t,1024>,32> l1{};
bool acquired=false;
uint32_t sorts=0, multiplies=0, adds=0, tail_addresses=0, scope_depth=0;
std::map<std::string,uint32_t> scopes;
std::vector<uint64_t> operations;
struct ScopeRecord {
    explicit ScopeRecord(const char* name) { ++scopes[name]; ++scope_depth; }
    ~ScopeRecord() { --scope_depth; }
};
void op(uint32_t kind, uint32_t a=0, uint32_t b=0) { operations.push_back((uint64_t(kind)<<32)|(a<<16)|b); }
Format format(uint32_t cb) { return cb==2 || cb==17 || cb==28 || cb==29 ? Format::Int32 : Format::BF16; }
void reconfig_data_format_srca(uint32_t cb) { source_format=format(cb); op(1,cb); }
void reconfig_data_format_srca(uint32_t old, uint32_t cb) {
    if (format(old)!=format(cb)) reconfig_data_format_srca(cb);
}
void pack_reconfig_data_format(uint32_t cb) { pack_format=format(cb); op(2,cb); }
void copy_tile_to_dst_init_short(uint32_t cb) { require(source_format==format(cb)); op(3,cb); }
void copy_tile_to_dst_init_short_with_dt(uint32_t old, uint32_t cb) {
    reconfig_data_format_srca(old,cb); copy_tile_to_dst_init_short(cb);
}
void copy_tile(uint32_t cb, uint32_t, uint32_t slot) {
    require(acquired && source_format==format(cb) && queued[cb]);
    destination[slot]=format(cb); loaded[slot]=true; op(4,cb,slot);
}
void transpose_wh_init_short(uint32_t cb) { require(source_format==format(cb)); op(5,cb); }
void transpose_wh_tile(uint32_t cb, uint32_t, uint32_t slot) {
    require(acquired && source_format==format(cb) && queued[cb]);
    destination[slot]=format(cb); loaded[slot]=true; op(6,cb,slot);
}
uintptr_t get_tile_address(uint32_t cb, uint32_t) {
    require((cb==27 || cb==28) ? !queued[cb] : queued[cb]>0);
    if (cb==24) ++tail_addresses;
    op(7,cb); return reinterpret_cast<uintptr_t>(l1[cb].data());
}
void cb_reserve_back(uint32_t cb, uint32_t count) { op(8,cb,count); }
void cb_wait_front(uint32_t cb, uint32_t count) { require(queued[cb]>=count); op(9,cb,count); }
void cb_push_back(uint32_t cb, uint32_t count) {
    if (cb!=16 && cb!=17) queued[cb]+=count; // The writer consumes output immediately.
    op(10,cb,count);
}
void cb_pop_front(uint32_t cb, uint32_t count) {
    require(queued[cb]>=count); queued[cb]-=count;
    if (cb==29) metadata_cursor+=count;
    op(11,cb,count);
}
uint32_t read_tile_value(uint32_t cb, uint32_t, uint32_t element) {
    require(cb==29 && queued[cb] && metadata_cursor<records.size());
    op(12,cb,element); return records[metadata_cursor].word[element];
}
void pack_tile(uint32_t slot, uint32_t cb) {
    require(acquired && loaded[slot] && pack_format==format(cb) && destination[slot]==format(cb));
    l1[cb].fill(0x12341234); op(13,cb,slot);
}
void acquire_dst() { require(!acquired); acquired=true; loaded.fill(false); op(14); }
void release_dst() { require(acquired); acquired=false; op(15); }
void mm_init(uint32_t a, uint32_t b, uint32_t out) {
    require(!acquired && format(a)==Format::BF16 && format(b)==Format::BF16);
    source_format=format(b); pack_format=format(out); op(16);
}
void mm_init_short(uint32_t a, uint32_t b) {
    require(!acquired && format(a)==Format::BF16 && source_format==format(b)); op(17);
}
void matmul_tiles(uint32_t a, uint32_t b, uint32_t ka, uint32_t kb, uint32_t slot) {
    require(acquired && source_format==Format::BF16 && queued[a]>=width && queued[b]>=width && ka==kb);
    destination[slot]=Format::BF16; loaded[slot]=true; ++multiplies; op(18,ka,slot);
}
void topk_tile_init() { op(19); }
void topk_local_sort(uint32_t, uint32_t, uint32_t) {
    require(acquired && source_format==Format::BF16);
    for (auto present:loaded) require(present);
    require(destination[0]==Format::BF16 && destination[1]==Format::BF16
            && destination[2]==Format::Int32 && destination[3]==Format::Int32);
    ++sorts; op(20);
}
void add_tiles_init(uint32_t a, uint32_t b) {
    require(!acquired && format(a)==Format::BF16 && format(b)==Format::BF16);
    source_format=Format::BF16; op(21);
}
void add_tiles(uint32_t a, uint32_t b, uint32_t, uint32_t, uint32_t slot) {
    require(acquired && queued[a] && queued[b]);
    destination[slot]=Format::BF16; loaded[slot]=true; ++adds; op(22);
}
#include "../kernels/compute/worker_compute.cpp"

void run(uint32_t selected) {
    profile_batch=selected; metadata_cursor=sorts=multiplies=adds=tail_addresses=scope_depth=0;
    acquired=false; queued.fill(0); l1={}; scopes.clear(); operations.clear();
    source_format=pack_format=Format::Int32;
    queued[0]=3*width; queued[1]=43*width; queued[2]=43;
    queued[3]=SIM_MASK==1 ? 19 : 0;
    queued[29]=records.size();
    kernel_main();
    require(!acquired && !scope_depth && metadata_cursor==records.size());
    for (auto count:queued) require(count==0);
    require(sorts==43 && multiplies==43*width);
    require(adds==(SIM_MASK==1 ? 19u : 0u));
    require(tail_addresses==(SIM_MASK==0 ? 19u : 0u));
    require(source_format==Format::Int32 && pack_format==Format::Int32);
}
int main() {
    // Reordered IDs, >16 pages per worker, sampled/unsampled partial tails,
    // many one-page tails (maximum scope budget), and an empty worker batch.
    const uint32_t tail=SIM_MASK==2 ? 32 : 17;
    records={Record{{2,32,3}},Record{{10,0,5,0,tail}},Record{{11,5,20,0,32}},Record{{12,25,1,0,tail}},Record{{0,32,17}}};
    for (uint32_t i=0;i<17;++i) records.push_back(Record{{i,26+i,1,0,tail}});
    records.push_back(Record{{1,16,0}});
    run(UINT32_MAX);
    require(scopes.empty());
    const auto baseline=operations;
    const auto baseline_l1=l1;
    for (uint32_t batch:{2u,0u,1u}) {
        run(batch);
        require(operations==baseline && l1==baseline_l1);
        const uint32_t pages=batch==1 ? 0 : 16;
        for (const auto* name:{"Worker input wait","Worker matmul","Worker score pack","Worker score and ID wait","Worker local topk"})
            require(scopes[name]==pages);
        require(scopes["Worker query wait"]==1 && scopes["Worker result pack"]==1);
        const uint32_t masks=SIM_MASK==2 || batch==1 ? 0 : (batch==0 ? 16 : 1);
        require(scopes["Worker tail mask"]==masks);
        uint32_t total=0;
        for (const auto& [name,count]:scopes) { (void)name; total+=count; }
        require(total==2+5*pages+masks && total<=98);
    }
    std::cout<<"Actual worker kernel: sampled/unsampled operation sequences and payloads agree; mode="<<SIM_MASK
             <<", all CBs consumed, tails preserved, <=98 scopes/RISC\n";
}
