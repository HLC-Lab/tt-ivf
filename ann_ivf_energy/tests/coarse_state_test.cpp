// Compile the actual coarse kernel against an API lifecycle/format model.
// This checks CB ownership and BF16/Int32 transitions, not Tensix arithmetic.
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#define UNPACK(...) do { __VA_ARGS__ } while (false)
#define PACK(...) do { __VA_ARGS__ } while (false)
namespace tt::CB { enum : uint32_t { c_intermed3 = 27, c_intermed4 = 28 }; }
constexpr std::array<uint32_t, 14> args{0, 1, 2, 25, 26, 16, 17, 16, 32, 1, 5, 4, 24, 2};
uint32_t batches = 3;
constexpr uint32_t get_compile_time_arg_val(uint32_t index) { return args[index]; }
template<class T> T get_arg_val(uint32_t) { return batches; }
enum class Format { BF16, Int32 };
Format source_format = Format::Int32, pack_format = Format::Int32;
std::array<Format, 4> destination;
std::array<bool, 4> loaded{};
std::array<uint32_t, 32> queued{};
alignas(16) std::array<std::array<uint32_t, 1024>, 32> l1{};
uint32_t full_inits = 0, short_inits = 0, sorts = 0, copies = 0, transposes = 0;
bool acquired = false;
void require(bool ok) { if (!ok) throw std::runtime_error("Coarse compute format/lifecycle error"); }
Format format(uint32_t cb) { return cb == 2 || cb == 26 || cb == 17 || cb == 28 ? Format::Int32 : Format::BF16; }
void reconfig_data_format_srca(uint32_t cb) { source_format = format(cb); }
void pack_reconfig_data_format(uint32_t cb) { pack_format = format(cb); }
void copy_tile_to_dst_init_short(uint32_t cb) { require(source_format == format(cb)); }
void copy_tile_to_dst_init_short_with_dt(uint32_t old_cb, uint32_t cb) {
    if (format(old_cb) != format(cb)) reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
}
void copy_tile(uint32_t cb, uint32_t, uint32_t slot) {
    require(acquired && source_format == format(cb) && queued[cb]);
    destination[slot] = format(cb); loaded[slot] = true; ++copies;
}
void transpose_wh_init_short(uint32_t cb) { require(source_format == format(cb)); }
void transpose_wh_tile(uint32_t cb, uint32_t, uint32_t slot) {
    require(acquired && source_format == format(cb) && queued[cb]);
    destination[slot] = format(cb); loaded[slot] = true; ++transposes;
}
uintptr_t get_tile_address(uint32_t cb, uint32_t) {
    require(!queued[cb]);  // initialize_champion is allowed only on an empty CB.
    return reinterpret_cast<uintptr_t>(l1[cb].data());
}
void cb_reserve_back(uint32_t, uint32_t) {}
void cb_wait_front(uint32_t cb, uint32_t count) { require(queued[cb] >= count); }
void cb_push_back(uint32_t cb, uint32_t count) {
    if (cb != 16 && cb != 17) queued[cb] += count;  // Output writer consumes its tiles.
}
void cb_pop_front(uint32_t cb, uint32_t count) { require(queued[cb] >= count); queued[cb] -= count; }
void pack_tile(uint32_t slot, uint32_t cb) {
    require(acquired && loaded[slot] && pack_format == format(cb) && destination[slot] == format(cb));
}
void acquire_dst() { require(!acquired); acquired = true; loaded.fill(false); }
void release_dst() { require(acquired); acquired = false; }
void mm_init(uint32_t a, uint32_t b, uint32_t out) {
    require(!acquired && format(a) == Format::BF16 && format(b) == Format::BF16);
    source_format = format(b); pack_format = format(out); ++full_inits;
}
void mm_init_short(uint32_t a, uint32_t b) {
    require(!acquired && format(a) == Format::BF16 && source_format == format(b)); ++short_inits;
}
void matmul_tiles(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t slot) {
    require(acquired && source_format == Format::BF16);
    destination[slot] = Format::BF16; loaded[slot] = true;
}
void topk_tile_init() {}
void topk_local_sort(uint32_t, uint32_t, uint32_t) {
    require(acquired && source_format == Format::BF16);
    for (auto present : loaded) require(present);
    require(destination[0] == Format::BF16 && destination[1] == Format::BF16
            && destination[2] == Format::Int32 && destination[3] == Format::Int32);
    ++sorts;
}
#include "../kernels/compute/compute_coarse.cpp"

int main() {
    for (uint32_t count : {0u, 1u, 3u, 5u}) {
        batches = count;
        full_inits = short_inits = sorts = copies = transposes = 0;
        queued.fill(0);
        source_format = pack_format = Format::Int32;
        queued[0] = batches * args[13];
        queued[1] = batches * args[7] * args[13];
        queued[2] = batches * args[7];
        kernel_main();
        require(full_inits == 1 && short_inits == batches * args[7] && sorts == short_inits);
        require(copies == 2 * sorts && transposes == 2 * sorts + 2 * batches);
        require(!acquired);
        const auto final_format = batches ? Format::Int32 : Format::BF16;
        require(source_format == final_format && pack_format == final_format);
        for (auto queued_count : queued) require(queued_count == 0);
    }
    std::cout << "Actual coarse kernel format/lifecycle checks passed for 0, 1, 3 and 5 query batches per core\n";
}
