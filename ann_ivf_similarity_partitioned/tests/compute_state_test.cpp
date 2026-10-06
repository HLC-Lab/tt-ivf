// Exercise the real reduction helper across the Int32 -> BF16 batch boundary.
// This models API format state only; it does not emulate Tensix arithmetic.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#define UNPACK(...) do { __VA_ARGS__ } while (false)
#define PACK(...) do { __VA_ARGS__ } while (false)
namespace tt::CB { enum : uint32_t { c_intermed3=27, c_intermed4=28 }; }
enum class Format { BF16, Int32 };
constexpr uint32_t scores=0, indices=1, output_values=16, output_ids=17;
Format source_format=Format::BF16;
uint32_t copies=0, transposes=0;
Format format(uint32_t cb) {
    return cb==indices || cb==tt::CB::c_intermed4 || cb==output_ids ? Format::Int32 : Format::BF16;
}
void require(bool condition) { if (!condition) throw std::runtime_error("Incorrect compute source format"); }
void reconfig_data_format_srca(uint32_t cb) { source_format=format(cb); }
void copy_tile_to_dst_init_short(uint32_t cb) { require(source_format==format(cb)); }
void copy_tile_to_dst_init_short_with_dt(uint32_t old_cb, uint32_t new_cb) {
    // As in the real API, this updates formats only if the supplied formats
    // differ; it cannot correct a wrongly supplied old CB.
    if (format(old_cb)!=format(new_cb)) reconfig_data_format_srca(new_cb);
    copy_tile_to_dst_init_short(new_cb);
}
void copy_tile(uint32_t cb, uint32_t, uint32_t) { require(source_format==format(cb)); ++copies; }
void transpose_wh_init_short(uint32_t cb) { require(source_format==format(cb)); }
void transpose_wh_tile(uint32_t cb, uint32_t, uint32_t) { require(source_format==format(cb)); ++transposes; }
uint32_t get_tile_address(uint32_t, uint32_t) { return 0; }
void cb_reserve_back(uint32_t, uint32_t) {}
void cb_wait_front(uint32_t, uint32_t) {}
void cb_push_back(uint32_t, uint32_t) {}
void cb_pop_front(uint32_t, uint32_t) {}
void pack_reconfig_data_format(uint32_t) {}
void pack_tile(uint32_t, uint32_t) {}
void acquire_dst() {}
void release_dst() {}
void topk_tile_init() {}
void topk_local_sort(uint32_t, uint32_t, uint32_t) {}
#include "../kernels/compute/fine_common.hpp"
int main() {
    try {
        source_format=Format::Int32;
        bool rejected=false;
        try { copy_tile_to_dst_init_short_with_dt(scores,tt::CB::c_intermed3); }
        catch (const std::runtime_error&) { rejected=true; }
        require(rejected); // Demonstrates why the old batch transition failed.
        for (uint32_t batch=0; batch<3; ++batch) {
            for (uint32_t partial=0; partial<7; ++partial) combine_page(scores,indices);
            transpose_output(tt::CB::c_intermed3,output_values);
            transpose_output(tt::CB::c_intermed4,output_ids);
            require(source_format==Format::Int32);
        }
        require(copies==3*7*2 && transposes==3*(7*2+2));
        std::cout<<"Reduction source format restored across three Int32 output / BF16 input batch boundaries.\n";
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
