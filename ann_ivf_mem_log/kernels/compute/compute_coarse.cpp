// tt_metal/programming_examples/ann_ivf_mem_log/kernels/compute/compute_coarse.cpp
#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"

void initialize_champion(uint32_t val_cb, uint32_t ind_cb) {
    cb_reserve_back(val_cb, 1);
    [[maybe_unused]] const auto val_addr = get_tile_address(val_cb, 0);
    PACK({
        volatile uint16_t* val_ptr = reinterpret_cast<volatile uint16_t*>(val_addr);
        for (uint32_t i = 0; i < 1024; i++) val_ptr[i] = 0xC61C; // -10000 BF16
    });
    cb_push_back(val_cb, 1);

    cb_reserve_back(ind_cb, 1);
    [[maybe_unused]] const auto ind_addr = get_tile_address(ind_cb, 0);
    PACK({
        volatile uint32_t* ind_ptr = reinterpret_cast<volatile uint32_t*>(ind_addr);
        for (uint32_t i = 0; i < 1024; i++) ind_ptr[i] = 0xFFFFFFFF;
    });
    cb_push_back(ind_cb, 1);
}

void transpose_and_pack(uint32_t transposed_cb_index, uint32_t dest_cb_index, uint32_t Kt) {
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
    cb_wait_front(transposed_cb_index, 1);
    cb_pop_front(transposed_cb_index, 1);
}


void kernel_main() {
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cluster_data = get_compile_time_arg_val(1);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t champion_val_cb = get_compile_time_arg_val(3);
    constexpr uint32_t champion_ind_cb = get_compile_time_arg_val(4);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(6);
    uint32_t Ht = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t Kt = get_compile_time_arg_val(9);
    constexpr uint32_t matmul_out_cb = get_compile_time_arg_val(12);
    constexpr uint32_t query_tiles_width = get_compile_time_arg_val(13);

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t index_dest_end = 3;

    constexpr int end_phase = 5;
    constexpr int sort_descending = 0;

    static_assert(Kt == 1, "Coarse search returns one top-32 tile per batch");
    // Full init resets destination synchronization; issue it once, rather
    // than resetting MATH/PACK while the other RISC may still use it.
    mm_init(input_cb_index, cb_cluster_data, matmul_out_cb);
    topk_tile_init();

    for (uint32_t ht = 0; ht < Ht; ++ht) {

        cb_wait_front(input_cb_index, query_tiles_width);

        // Initialize champion
        initialize_champion(champion_val_cb, champion_ind_cb);

        for (uint32_t wt = 0; wt < Wt; wt++) {
            reconfig_data_format_srca(cb_cluster_data);
            mm_init_short(input_cb_index, cb_cluster_data);
            acquire_dst();
            for (uint32_t k = 0; k < query_tiles_width; k++) {
                cb_wait_front(cb_cluster_data, 1);
                matmul_tiles(input_cb_index, cb_cluster_data, k, 0, 0);
                cb_pop_front(cb_cluster_data, 1);
            }
            cb_reserve_back(matmul_out_cb, 1);
            pack_reconfig_data_format(matmul_out_cb);
            pack_tile(0, matmul_out_cb);
            cb_push_back(matmul_out_cb, 1);
            release_dst();

            // Wait for champion and challenger
            cb_wait_front(champion_val_cb, 1);
            cb_wait_front(champion_ind_cb, 1);
            cb_wait_front(matmul_out_cb, 1);
            cb_wait_front(index_cb_index, 1);

            acquire_dst();

            // Load champion values to DST 0 and champion indices to DST 2
            reconfig_data_format_srca(champion_val_cb);
            copy_tile_to_dst_init_short(champion_val_cb);
            copy_tile(champion_val_cb, 0, input_dest_start);

            reconfig_data_format_srca(champion_ind_cb);
            copy_tile_to_dst_init_short(champion_ind_cb);
            copy_tile(champion_ind_cb, 0, index_dest_start);

            // Load and transpose challenger values to DST 1
            reconfig_data_format_srca(matmul_out_cb);
            transpose_wh_init_short(matmul_out_cb);
            transpose_wh_tile(matmul_out_cb, 0, input_dest_end);

            // Load and transpose challenger indices to DST 3
            reconfig_data_format_srca(index_cb_index);
            transpose_wh_init_short(index_cb_index);
            transpose_wh_tile(index_cb_index, 0, index_dest_end);

            // Sort (merging challenger into champion)
            reconfig_data_format_srca(champion_val_cb);
            topk_tile_init();
            topk_local_sort(0, sort_descending, end_phase);

            // Pop old champion, pack and push new champion
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

            cb_pop_front(matmul_out_cb, 1);
            cb_pop_front(index_cb_index, 1);
            release_dst();
        }

        // Save final champion results
        transpose_and_pack(champion_val_cb, values_cb_index, Kt);
        transpose_and_pack(champion_ind_cb, output_ind_cb_index, Kt);

        cb_pop_front(input_cb_index, query_tiles_width);
    }
}
