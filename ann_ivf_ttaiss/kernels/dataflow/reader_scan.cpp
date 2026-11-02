#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "../../fine_tile_layout.hpp"

// Cluster-major scan reader. Script (cluster_schedule.hpp):
//   [tasks] then per task [start_block, count, pages, n, (batch, slot) x n].
// For each task: assemble the n gathered 32-query tiles in one whole
// query-chunk reservation, then stream every list page exactly once.
// Gathered tiles are built here, not on the host: tile k's 32 query IDs come
// from a 128-byte page of the tile-query table (0xFFFFFFFF = unused row), and
// each query row is copied from the original tilized query buffer as two
// 32-byte face segments per 32-column tile (32-byte aligned in DRAM and L1).
void kernel_main() {
    const uint32_t query_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t dataset_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t indices_dram_addr = get_arg_val<uint32_t>(2);
    const uint32_t dim_padded = get_arg_val<uint32_t>(3);
    const uint32_t scripts_dram_addr = get_arg_val<uint32_t>(4);
    const uint32_t script_page_bytes = get_arg_val<uint32_t>(5);
    const uint32_t worker_idx = get_arg_val<uint32_t>(6);
    const uint32_t batch_chunk = get_arg_val<uint32_t>(7);
    const uint32_t tile_queries_addr = get_arg_val<uint32_t>(8);
    // Second pass only: the first pass's merged top-32 tiles (row layout, row =
    // query, column r = (r+1)-th best) at pages threshold_page_base + batch.
    // The threshold rank is a compile-time argument: 31 when the host re-ranks
    // the device top-32, k - 1 otherwise.
    constexpr uint32_t threshold_rank = get_compile_time_arg_val(0);
    static_assert(threshold_rank < 32, "The merged rows hold ranks 0..31");
    constexpr uint32_t threshold_half = threshold_rank / 16;  // aligned 16-column segment holding the rank
    const uint32_t threshold_mode = get_arg_val<uint32_t>(9);
    const uint32_t threshold_val_addr = get_arg_val<uint32_t>(10);
    const uint32_t threshold_page_base = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_query = tt::CB::c_in0;
    constexpr uint32_t cb_cluster_data = tt::CB::c_in1;
    constexpr uint32_t cb_cluster_indices = tt::CB::c_in2;
    constexpr uint32_t cb_script = tt::CB::c_intermed5;
    constexpr uint32_t cb_tile_queries = tt::CB::c_in3;  // reader scratch: batch_chunk pages of 32 IDs
    constexpr uint32_t cb_threshold = tt::CB::c_in4;
    constexpr uint32_t unused_row = 0xFFFFFFFFu;
    constexpr uint32_t unused_threshold = 0x7F7F;  // largest finite BF16: never merged

    const InterleavedAddrGen<true> s_script = {.bank_base_address = scripts_dram_addr, .page_size = script_page_bytes};
    // The script is published once and stays read-only for the whole program;
    // compute and writer wait on it and nobody pops it.
    cb_reserve_back(cb_script, 1);
    const uint32_t script_l1 = get_write_ptr(cb_script);
    noc_async_read_page(worker_idx, s_script, script_l1);
    noc_async_read_barrier();
    cb_push_back(cb_script, 1);
    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*)script_l1;

    const uint32_t query_tiles_width = dim_padded / 32;
    const uint32_t chunk_tiles = batch_chunk * query_tiles_width;
    const InterleavedAddrGen<true> s_query = {.bank_base_address = query_dram_addr, .page_size = 2048};
    const InterleavedAddrGen<true> s_dataset = {.bank_base_address = dataset_dram_addr, .page_size = dim_padded * 64};
    const InterleavedAddrGen<true> s_indices = {.bank_base_address = indices_dram_addr, .page_size = 4096};
    const InterleavedAddrGen<true> s_tile_queries = {.bank_base_address = tile_queries_addr, .page_size = 128};
    const InterleavedAddrGen<true> s_threshold = {.bank_base_address = threshold_val_addr, .page_size = 2048};
    // Reserved once and never pushed: private scratch for the ID pages.
    cb_reserve_back(cb_tile_queries, batch_chunk);
    const uint32_t ids_l1 = get_write_ptr(cb_tile_queries);

    const uint32_t num_tasks = script[0];
    uint32_t ptr = 1;
    for (uint32_t task = 0; task < num_tasks; task++) {
        const uint32_t start_block = script[ptr++];
        ptr++;  // actual vector count: compute masks the tail page
        const uint32_t pages = script[ptr++];
        const uint32_t pairs = script[ptr++];
        const uint32_t pair_ptr = ptr;
        ptr += 2 * pairs;
        {
            DeviceZoneScopedN("Scan.Query");
            // Always reserve a whole chunk: a variable-size multi-tile
            // reservation could straddle the CB end. Unused tail slots are
            // never read by compute.
            cb_reserve_back(cb_query, chunk_tiles);
            const uint32_t q_l1 = get_write_ptr(cb_query);
            for (uint32_t i = 0; i < pairs; i++) {
                const uint32_t tile = script[pair_ptr + 2 * i];
                noc_async_read(s_tile_queries.get_noc_addr(tile), ids_l1 + 128 * i, 128);
            }
            noc_async_read_barrier();  // IDs are read below by this RISC
            for (uint32_t i = 0; i < pairs; i++) {
                volatile tt_l1_ptr uint32_t* ids = (volatile tt_l1_ptr uint32_t*)(ids_l1 + 128 * i);
                for (uint32_t row = 0; row < 32; row++) {
                    const uint32_t q = ids[row];
                    for (uint32_t t = 0; t < query_tiles_width; t++) {
                        const uint32_t dst_tile = q_l1 + (i * query_tiles_width + t) * 2048;
                        for (uint32_t half = 0; half < 2; half++) {
                            const uint32_t dst = dst_tile + 2 * fine_tile_offset(row, 16 * half);
                            if (q == unused_row) {
                                volatile tt_l1_ptr uint32_t* zero = (volatile tt_l1_ptr uint32_t*)dst;
                                for (uint32_t w = 0; w < 8; w++) zero[w] = 0;  // 16 BF16 zeros
                            } else {
                                const uint32_t src_tile = (q / 32) * query_tiles_width + t;
                                noc_async_read(s_query.get_noc_addr(src_tile, 2 * fine_tile_offset(q % 32, 16 * half)),
                                               dst, 32);
                            }
                        }
                    }
                }
            }
            // Every row segment has landed before compute sees the chunk.
            noc_async_read_barrier();
            cb_push_back(cb_query, chunk_tiles);

            if (threshold_mode) {
                // Threshold tile of pair i: row r = the threshold-rank score of
                // the tile's query r in its first-pass row, repeated across the
                // row. The aligned 16-column segment holding that rank lands in
                // the same place of row r; the rank's value is then spread over
                // the row.
                cb_reserve_back(cb_threshold, batch_chunk);
                const uint32_t thr_l1 = get_write_ptr(cb_threshold);
                for (uint32_t i = 0; i < pairs; i++) {
                    volatile tt_l1_ptr uint32_t* ids = (volatile tt_l1_ptr uint32_t*)(ids_l1 + 128 * i);
                    for (uint32_t row = 0; row < 32; row++) {
                        const uint32_t q = ids[row];
                        if (q == unused_row) continue;
                        noc_async_read(s_threshold.get_noc_addr(threshold_page_base + q / 32,
                                                                2 * fine_tile_offset(q % 32, 16 * threshold_half)),
                                       thr_l1 + 2048 * i + 2 * fine_tile_offset(row, 16 * threshold_half), 32);
                    }
                }
                noc_async_read_barrier();
                for (uint32_t i = 0; i < pairs; i++) {
                    volatile tt_l1_ptr uint32_t* ids = (volatile tt_l1_ptr uint32_t*)(ids_l1 + 128 * i);
                    volatile tt_l1_ptr uint16_t* thr = (volatile tt_l1_ptr uint16_t*)(thr_l1 + 2048 * i);
                    for (uint32_t row = 0; row < 32; row++) {
                        const uint32_t value =
                            ids[row] == unused_row ? unused_threshold : thr[fine_tile_offset(row, threshold_rank)];
                        const uint32_t word = value | (value << 16);
                        for (uint32_t half = 0; half < 2; half++) {
                            volatile tt_l1_ptr uint32_t* out =
                                (volatile tt_l1_ptr uint32_t*)(thr_l1 + 2048 * i + 2 * fine_tile_offset(row, 16 * half));
                            for (uint32_t w = 0; w < 8; w++) out[w] = word;
                        }
                    }
                }
                cb_push_back(cb_threshold, batch_chunk);
            }
        }
        DeviceZoneScopedN("Scan.List");
        for (uint32_t p = 0; p < pages; p++) {
            cb_reserve_back(cb_cluster_data, query_tiles_width);
            cb_reserve_back(cb_cluster_indices, 1);
            const uint32_t d_l1 = get_write_ptr(cb_cluster_data);
            const uint32_t i_l1 = get_write_ptr(cb_cluster_indices);
            // Vector page and its ID tile complete together before publishing.
            noc_async_read_page(start_block + p, s_dataset, d_l1);
            noc_async_read_page(start_block + p, s_indices, i_l1);
            noc_async_read_barrier();
            cb_push_back(cb_cluster_data, query_tiles_width);
            cb_push_back(cb_cluster_indices, 1);
        }
    }
}
