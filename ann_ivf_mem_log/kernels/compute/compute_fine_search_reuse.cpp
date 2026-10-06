#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ivf_compute_l1.hpp"

inline void transpose_and_pack(uint32_t transposed_cb_index, uint32_t dest_cb_index, uint32_t Kt) {
    reconfig_data_format_srca(transposed_cb_index);
    transpose_wh_init_short(transposed_cb_index);
    pack_reconfig_data_format(dest_cb_index);

    cb_wait_front(transposed_cb_index, Kt);
    for (uint32_t i = 0; i < Kt; ++i) {
        acquire_dst();
        cb_reserve_back(dest_cb_index, 1);
        transpose_wh_tile(transposed_cb_index, i, 0);
        pack_tile(0, dest_cb_index);
        cb_push_back(dest_cb_index, 1);
        release_dst();
    }
    cb_wait_front(transposed_cb_index, Kt);
    cb_pop_front(transposed_cb_index, Kt);
}

void kernel_main() {
    uint32_t query_tiles_width = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_script = tt::CB::c_intermed5;
    cb_wait_front(cb_script, 1);
    uint32_t script_addr = get_tile_address(cb_script, 0);
    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*) script_addr;

    uint32_t num_batches = script[0];
    uint32_t num_assigned = script[1];
    uint32_t num_mask_words = (num_batches + 31) / 32;

    constexpr uint32_t input_cb_index = tt::CB::c_in0;
    constexpr uint32_t in1_cb_index = tt::CB::c_in1;
    constexpr uint32_t index_cb_index = tt::CB::c_in2;
    constexpr uint32_t matmul_out_cb = tt::CB::c_intermed0;
    constexpr uint32_t champion_val_cb = tt::CB::c_intermed3;
    constexpr uint32_t champion_ind_cb = tt::CB::c_intermed4;
    constexpr uint32_t values_cb_index = tt::CB::c_out0;
    constexpr uint32_t output_ind_cb_index = tt::CB::c_out1;

    constexpr uint32_t Kt = 1;
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t index_dest_end = 3;

    constexpr int end_phase = 5;
    constexpr int sort_descending = 0;

    mm_init(input_cb_index, in1_cb_index, matmul_out_cb);

    for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
        // The reader pushes a query only when this worker has at least one
        // active, non-empty task for the batch. Pre-scan the script so the
        // compute kernel waits exactly when the reader produced a query.
        bool query_loaded = false;
        uint32_t query_scan_ptr = 2;
        for (uint32_t cluster_idx = 0; cluster_idx < num_assigned; cluster_idx++) {
            query_scan_ptr += 3;  // Cluster ID, DRAM page offset, actual count.
            uint32_t padded_count = script[query_scan_ptr++];
            uint32_t mask_ptr = query_scan_ptr;
            query_scan_ptr += num_mask_words;

            bool is_active =
                (script[mask_ptr + (batch_id / 32)] & (1u << (batch_id % 32))) != 0;
            if (is_active && padded_count > 0) {
                {
                    DeviceZoneScopedN("Query");
                    {
                        DeviceZoneScopedN("FineBubble.Compute.QueryWait");
                        cb_wait_front(input_cb_index, query_tiles_width);
                    }
                }
                query_loaded = true;
                break;
            }
        }


        initialize_champion(champion_val_cb, champion_ind_cb);

        uint32_t ptr = 2;
        for (uint32_t cluster_idx = 0; cluster_idx < num_assigned; cluster_idx++) {
            DeviceZoneScopedN("Cluster");
            ptr += 2; // cluster ID and DRAM page offset
            uint32_t actual_count = script[ptr++];
            uint32_t padded_count = script[ptr++];
            uint32_t mask_ptr = ptr;
            ptr += num_mask_words;

            bool is_active = (script[mask_ptr + (batch_id / 32)] & (1u << (batch_id % 32))) != 0;

            if (is_active && padded_count > 0) {
                uint32_t Wt = padded_count / 32;

                {
                    DeviceZoneScopedN("FineBubble.Compute.Cluster");
                    for (uint32_t wt = 0; wt < Wt; wt++) {
                        reconfig_data_format_srca(champion_ind_cb, in1_cb_index);
                        mm_init_short(input_cb_index, in1_cb_index);
                        cb_wait_front(in1_cb_index, query_tiles_width);

                        {
                            DeviceZoneScopedN("FineBubble.Compute.Work");
                            acquire_dst();
                            for (uint32_t k = 0; k < query_tiles_width; k++) {
                                matmul_tiles(input_cb_index, in1_cb_index, k, k, 0);
                            }
                            cb_reserve_back(matmul_out_cb, 1);
                            pack_reconfig_data_format(matmul_out_cb);
                            pack_tile(0, matmul_out_cb);
                            cb_push_back(matmul_out_cb, 1);
                            release_dst();
                            cb_pop_front(in1_cb_index, query_tiles_width);

                            cb_wait_front(matmul_out_cb, 1);
                            cb_wait_front(index_cb_index, 1);
                            cb_wait_front(champion_val_cb, 1);
                            cb_wait_front(champion_ind_cb, 1);

                            if (wt * 32 + 32 > actual_count) {
                                mask_fine_candidates(
                                    matmul_out_cb,
                                    index_cb_index,
                                    wt * 32,
                                    0xFFFFFFFFu,
                                    actual_count);
                            }
                            acquire_dst();
                            copy_tile_to_dst_init_short_with_dt(input_cb_index, champion_val_cb);
                            copy_tile(champion_val_cb, 0, input_dest_start);
                            copy_tile_to_dst_init_short_with_dt(champion_val_cb, champion_ind_cb);
                            copy_tile(champion_ind_cb, 0, index_dest_start);
                            reconfig_data_format_srca(matmul_out_cb);
                            transpose_wh_init_short(matmul_out_cb);
                            transpose_wh_tile(matmul_out_cb, 0, input_dest_end);
                            reconfig_data_format_srca(index_cb_index);
                            transpose_wh_init_short(index_cb_index);
                            transpose_wh_tile(index_cb_index, 0, index_dest_end);
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
                            release_dst();
                            cb_pop_front(matmul_out_cb, 1);
                            cb_pop_front(index_cb_index, 1);
                        }
                    }
                }
            }
        }

        if (query_loaded) {
            cb_pop_front(input_cb_index, query_tiles_width);
        }
        {
            transpose_and_pack(champion_val_cb, values_cb_index, Kt);
            transpose_and_pack(champion_ind_cb, output_ind_cb_index, Kt);
        }
    }
    cb_pop_front(cb_script, 1);
}
