#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ivf_compute_l1.hpp"

inline void transpose_and_pack(uint32_t transposed_cb_index, uint32_t dest_cb_index) {
    reconfig_data_format_srca(transposed_cb_index);
    transpose_wh_init_short(transposed_cb_index);
    pack_reconfig_data_format(dest_cb_index);
    cb_wait_front(transposed_cb_index, 1);
    acquire_dst();
    cb_reserve_back(dest_cb_index, 1);
    transpose_wh_tile(transposed_cb_index, 0, 0);
    pack_tile(0, dest_cb_index);
    cb_push_back(dest_cb_index, 1);
    release_dst();
    cb_pop_front(transposed_cb_index, 1);
}

// Per-query merge: every batch of 32 original-order queries folds nprobe
// gathered input tiles (row layout; row q' = query q''s j-th partial or a
// sentinel row) with the batch-major final-sort reduction.
// Script (read-only, published by the reader, never popped):
//   [batches, nprobe, then per batch: batch, nprobe x 32 entries].
void kernel_main() {
    constexpr uint32_t cb_script = tt::CB::c_intermed5;
    cb_wait_front(cb_script, 1);
    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*)get_tile_address(cb_script, 0);
    const uint32_t num_batches = script[0];
    const uint32_t nprobe = script[1];

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in1 = tt::CB::c_in1;
    constexpr uint32_t champion_val_cb = tt::CB::c_intermed3;
    constexpr uint32_t champion_ind_cb = tt::CB::c_intermed4;
    constexpr uint32_t values_cb_index = tt::CB::c_out0;
    constexpr uint32_t output_ind_cb_index = tt::CB::c_out1;

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t index_dest_end = 3;
    constexpr int sort_descending = 0;
    constexpr int end_phase = 5;

    topk_tile_init();
    transpose_wh_init(cb_in0, values_cb_index);

    for (uint32_t b = 0; b < num_batches; b++) {
        DeviceZoneScopedN("Merge.Batch");
        // The champion CBs are empty between batches, as required here.
        initialize_champion(champion_val_cb, champion_ind_cb);
        for (uint32_t j = 0; j < nprobe; ++j) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            cb_wait_front(champion_val_cb, 1);
            cb_wait_front(champion_ind_cb, 1);

            acquire_dst();
            copy_tile_to_dst_init_short_with_dt(cb_in0, champion_val_cb);
            copy_tile(champion_val_cb, 0, input_dest_start);
            copy_tile_to_dst_init_short_with_dt(champion_val_cb, champion_ind_cb);
            copy_tile(champion_ind_cb, 0, index_dest_start);

            reconfig_data_format_srca(cb_in0);
            transpose_wh_init_short(cb_in0);
            transpose_wh_tile(cb_in0, 0, input_dest_end);
            reconfig_data_format_srca(cb_in1);
            transpose_wh_init_short(cb_in1);
            transpose_wh_tile(cb_in1, 0, index_dest_end);

            reconfig_data_format_srca(champion_val_cb);
            topk_tile_init();
            topk_local_sort(0, sort_descending, end_phase);

            cb_pop_front(champion_val_cb, 1);
            cb_pop_front(champion_ind_cb, 1);
            cb_reserve_back(champion_val_cb, 1);
            cb_reserve_back(champion_ind_cb, 1);
            pack_reconfig_data_format(champion_val_cb);
            pack_tile(input_dest_start, champion_val_cb);
            pack_reconfig_data_format(champion_ind_cb);
            pack_tile(index_dest_start, champion_ind_cb);
            cb_push_back(champion_val_cb, 1);
            cb_push_back(champion_ind_cb, 1);

            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
            release_dst();
        }
        transpose_and_pack(champion_val_cb, values_cb_index);
        transpose_and_pack(champion_ind_cb, output_ind_cb_index);
        reconfig_data_format_srca(cb_in0);
    }
}
