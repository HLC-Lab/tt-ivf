#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "fine_common.hpp"
using namespace ivf_partitioned;
inline void score_page(uint32_t width, bool sample) {
    constexpr uint32_t query = tt::CB::c_in0, input = tt::CB::c_in1, scores = tt::CB::c_intermed0;
    auto multiply = [&] {
        reconfig_data_format_srca(tt::CB::c_intermed4, input);
        mm_init_short(query, input);
        acquire_dst();
        for (uint32_t k = 0; k < width; ++k) matmul_tiles(query, input, k, k, 0);
    };
    if (sample) {
        DeviceZoneScopedN("Worker matmul"); multiply();
    } else multiply();
    auto pack_scores = [&] {
        cb_reserve_back(scores, 1);
        pack_reconfig_data_format(scores); pack_tile(0, scores); cb_push_back(scores, 1);
        release_dst(); cb_pop_front(input, width);
    };
    if (sample) {
        DeviceZoneScopedN("Worker score pack"); pack_scores();
    } else pack_scores();
}
void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr auto mode = static_cast<MaskMode>(get_compile_time_arg_val(1));
    const uint32_t batches = get_arg_val<uint32_t>(0), profile_batch = get_arg_val<uint32_t>(1);
    constexpr uint32_t meta = tt::CB::c_intermed5, scores = tt::CB::c_intermed0;
    mm_init(tt::CB::c_in0, tt::CB::c_in1, scores);
    for (uint32_t b = 0; b < batches; ++b) {
        cb_wait_front(meta, 1);
        const uint32_t batch_id = read_tile_value(meta, 0, 0);
        const uint32_t tasks = read_tile_value(meta, 0, 2);
        cb_pop_front(meta, 1);
        const bool sample_batch = batch_id == profile_batch;
        if (sample_batch) {
            DeviceZoneScopedN("Worker query wait"); cb_wait_front(tt::CB::c_in0, width);
        } else cb_wait_front(tt::CB::c_in0, width);
        initialize_champion(tt::CB::c_intermed3, tt::CB::c_intermed4);
        uint32_t ordinal = 0;
        for (uint32_t task = 0; task < tasks; ++task) {
            cb_wait_front(meta, 1);
            const uint32_t pages = read_tile_value(meta, 0, 2);
            const uint32_t valid = read_tile_value(meta, 0, 4);
            cb_pop_front(meta, 1);
            for (uint32_t page = 0; page < pages; ++page, ++ordinal) {
                // Six zones per sampled page at most, plus two batch zones:
                // <= 98 scopes/RISC, below the 125-scope optional buffer.
                const bool sample = sample_batch && ordinal < 16;
                if (sample) {
                    DeviceZoneScopedN("Worker input wait"); cb_wait_front(tt::CB::c_in1, width);
                } else cb_wait_front(tt::CB::c_in1, width);
                score_page(width, sample);
                auto wait_result = [&] { cb_wait_front(scores, 1); cb_wait_front(tt::CB::c_in2, 1); };
                if (sample) {
                    DeviceZoneScopedN("Worker score and ID wait"); wait_result();
                } else wait_result();
                uint32_t result = scores;
                if constexpr (mode != MaskMode::Off) {
                    if (needs_mask(page, pages, valid)) {
                        auto apply_mask = [&] {
                            if constexpr (mode == MaskMode::Auto) {
                                mask_tail(scores, tt::CB::c_in2, valid);
                            } else {
                                cb_wait_front(tt::CB::c_in3, 1);
                                add_tiles_init(scores, tt::CB::c_in3);
                                acquire_dst(); add_tiles(scores, tt::CB::c_in3, 0, 0, 0);
                                cb_reserve_back(tt::CB::c_intermed1, 1);
                                pack_reconfig_data_format(tt::CB::c_intermed1); pack_tile(0, tt::CB::c_intermed1);
                                cb_push_back(tt::CB::c_intermed1, 1); release_dst();
                                cb_pop_front(scores, 1); cb_pop_front(tt::CB::c_in3, 1);
                                result = tt::CB::c_intermed1;
                            }
                        };
                        if (sample) {
                            DeviceZoneScopedN("Worker tail mask"); apply_mask();
                        } else apply_mask();
                    }
                }
                if (sample) {
                    DeviceZoneScopedN("Worker local topk"); combine_page(result, tt::CB::c_in2);
                } else combine_page(result, tt::CB::c_in2);
            }
        }
        cb_pop_front(tt::CB::c_in0, width);
        auto pack_result = [&] {
            transpose_output(tt::CB::c_intermed3, tt::CB::c_out0);
            transpose_output(tt::CB::c_intermed4, tt::CB::c_out1);
        };
        if (sample_batch) {
            DeviceZoneScopedN("Worker result pack"); pack_result();
        } else pack_result();
    }
}
