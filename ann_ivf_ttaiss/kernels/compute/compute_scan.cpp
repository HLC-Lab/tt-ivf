#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
// reduce.h names these in its default template arguments.
#define REDUCE_OP PoolType::MAX
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "compute_kernel_api/reduce.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ivf_compute_l1.hpp"

// List scan. Optimizations that keep the scores bit-identical to the plain
// per-page matmul + top-k loop:
//  - batched matmul: per page, all chunk matmuls after one init, then merges;
//  - IDs once: each page's ID tile is transposed once and copied per batch;
//  - narrow mask: only padded lanes' scores get the sentinel (the host already
//    stores 0xFFFFFFFF IDs for padded vectors);
//
// Per-list query tiles (ann_ivf_ttaiss): a "batch" of the scan script
// is a gathered 32-query tile of one list. Partials are published in row
// layout (row r = query row r of the tile, columns = ranks) so the merge can
// gather each query's partial row from different tiles.
//
// Skipping hopeless merges (ann_ivf_ttaiss, threshold mode):
// the second pass gets one threshold tile per gathered tile (row r = the
// 32nd-best score its query already has from the first pass). Per page, all
// score tiles are checked first: max(score - threshold) over the tile, by an
// eltwise sub and a scalar max reduce; a pair whose maximum is not positive
// cannot change its query's final top-32 and skips the top-k merge.
//
// The champion ring always holds batch_chunk pairs (pairs of the task plus
// sentinel dummies), so it is full: the slot popped at the front is the slot
// the next push reserves. A skipped pair (and every dummy) rotates by pop and
// push alone, with no data movement; a merged pair packs its update into the
// slot it just read, after the unpack -> math -> pack dependency.

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

// Append one sentinel champion pair to the ring. initialize_champion() writes
// through the read pointer and is valid only on an empty CB, so the ring is
// filled by copying the resident sentinel tiles through DST and packing them
// at the ring's write pointer.
inline void push_sentinel_champion(
    uint32_t sentinel_val_cb, uint32_t sentinel_ind_cb, uint32_t champion_val_cb, uint32_t champion_ind_cb) {
    acquire_dst();
    reconfig_data_format_srca(sentinel_val_cb);
    copy_tile_to_dst_init_short(sentinel_val_cb);
    copy_tile(sentinel_val_cb, 0, 0);
    copy_tile_to_dst_init_short_with_dt(sentinel_val_cb, sentinel_ind_cb);
    copy_tile(sentinel_ind_cb, 0, 1);
    cb_reserve_back(champion_val_cb, 1);
    cb_reserve_back(champion_ind_cb, 1);
    pack_reconfig_data_format(champion_val_cb);
    pack_tile(0, champion_val_cb);
    pack_reconfig_data_format(champion_ind_cb);
    pack_tile(1, champion_ind_cb);
    cb_push_back(champion_val_cb, 1);
    cb_push_back(champion_ind_cb, 1);
    release_dst();
}

// Rotate the full ring by one pair without touching its data.
inline void rotate_champion(uint32_t champion_val_cb, uint32_t champion_ind_cb) {
    cb_wait_front(champion_val_cb, 1);
    cb_wait_front(champion_ind_cb, 1);
    cb_pop_front(champion_val_cb, 1);
    cb_pop_front(champion_ind_cb, 1);
    cb_reserve_back(champion_val_cb, 1);
    cb_reserve_back(champion_ind_cb, 1);
    cb_push_back(champion_val_cb, 1);
    cb_push_back(champion_ind_cb, 1);
}

// Bit i set: some row of pair i's score tile beats its threshold. The score
// tiles stay in the CB (consumed by the merge loop). The diff and flag CBs
// hold batch_chunk tiles, so neither phase waits on the other's consumer.
inline uint32_t page_merge_mask(
    uint32_t pairs, uint32_t score_cb, uint32_t threshold_cb, uint32_t scaler_cb, uint32_t diff_cb, uint32_t flag_cb) {
    reconfig_data_format_srca(score_cb);
    sub_tiles_init(score_cb, threshold_cb);
    pack_reconfig_data_format(diff_cb);
    cb_wait_front(score_cb, pairs);
    for (uint32_t i = 0; i < pairs; i++) {
        acquire_dst();
        sub_tiles(score_cb, threshold_cb, i, i, 0);
        cb_reserve_back(diff_cb, 1);
        pack_tile(0, diff_cb);
        cb_push_back(diff_cb, 1);
        release_dst();
    }
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(diff_cb, scaler_cb, flag_cb);
    pack_reconfig_data_format(flag_cb);
    for (uint32_t i = 0; i < pairs; i++) {
        cb_wait_front(diff_cb, 1);
        acquire_dst();
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(diff_cb, scaler_cb, 0, 0, 0);
        cb_reserve_back(flag_cb, 1);
        pack_tile(0, flag_cb);
        cb_push_back(flag_cb, 1);
        release_dst();
        cb_pop_front(diff_cb, 1);
    }
    reduce_uninit();
    // Flags are read at the CB front only: a window of `pairs` tiles may wrap
    // the CB end, and tile addresses past the front do not wrap.
    uint32_t mask = 0;
    for (uint32_t i = 0; i < pairs; i++) {
        cb_wait_front(flag_cb, 1);
        // Element (0, 0) is the low half-word: merge only on a positive BF16.
        const uint32_t bits = read_tile_value(flag_cb, 0, 0) & 0xFFFFu;
        if ((bits & 0x8000u) == 0 && bits != 0) mask |= 1u << i;
        cb_pop_front(flag_cb, 1);
    }
    return mask;
}

void kernel_main() {
    const uint32_t query_tiles_width = get_arg_val<uint32_t>(0);
    const uint32_t batch_chunk = get_arg_val<uint32_t>(1);
    const uint32_t threshold_mode = get_arg_val<uint32_t>(2);  // 1: second pass, batch_chunk <= 32
    const uint32_t chunk_tiles = batch_chunk * query_tiles_width;

    constexpr uint32_t cb_script = tt::CB::c_intermed5;
    cb_wait_front(cb_script, 1);
    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*)get_tile_address(cb_script, 0);

    constexpr uint32_t input_cb_index = tt::CB::c_in0;
    constexpr uint32_t in1_cb_index = tt::CB::c_in1;
    constexpr uint32_t index_cb_index = tt::CB::c_in2;
    constexpr uint32_t matmul_out_cb = tt::CB::c_intermed0;
    constexpr uint32_t ids_transposed_cb = tt::CB::c_intermed1;
    constexpr uint32_t champion_val_cb = tt::CB::c_intermed3;
    constexpr uint32_t champion_ind_cb = tt::CB::c_intermed4;
    constexpr uint32_t sentinel_val_cb = tt::CB::c_intermed6;
    constexpr uint32_t sentinel_ind_cb = tt::CB::c_intermed7;
    constexpr uint32_t values_cb_index = tt::CB::c_out0;
    constexpr uint32_t output_ind_cb_index = tt::CB::c_out1;
    constexpr uint32_t threshold_cb = tt::CB::c_in4;   // reader: one tile per gathered tile
    constexpr uint32_t scaler_cb = tt::CB::c_in5;      // 1.0 reduce scaler, resident
    constexpr uint32_t diff_cb = tt::CB::c_intermed2;  // score - threshold
    constexpr uint32_t flag_cb = tt::CB::c_out2;       // max of each diff tile

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t index_dest_end = 3;
    constexpr int end_phase = 5;
    constexpr int sort_descending = 0;

    // The sentinel CBs are empty here, which initialize_champion requires.
    // They stay resident (never popped) for the program's lifetime.
    initialize_champion(sentinel_val_cb, sentinel_ind_cb);
    cb_wait_front(sentinel_val_cb, 1);
    cb_wait_front(sentinel_ind_cb, 1);
    if (threshold_mode) {
        initialize_constant_tile(scaler_cb, 0x3F80);  // BF16 1.0
        cb_wait_front(scaler_cb, 1);
    }

    mm_init(input_cb_index, in1_cb_index, matmul_out_cb);

    const uint32_t num_tasks = script[0];
    uint32_t ptr = 1;
    for (uint32_t task = 0; task < num_tasks; task++) {
        ptr++;  // DRAM start block: reader only
        const uint32_t actual_count = script[ptr++];
        const uint32_t pages = script[ptr++];
        const uint32_t pairs = script[ptr++];
        ptr += 2 * pairs;

        cb_wait_front(input_cb_index, chunk_tiles);
        if (threshold_mode) cb_wait_front(threshold_cb, batch_chunk);
        for (uint32_t i = 0; i < batch_chunk; i++)  // full ring: pairs + dummies
            push_sentinel_champion(sentinel_val_cb, sentinel_ind_cb, champion_val_cb, champion_ind_cb);

        for (uint32_t wt = 0; wt < pages; wt++) {
            DeviceZoneScopedN("Scan.Page");
            cb_wait_front(in1_cb_index, query_tiles_width);
            cb_wait_front(index_cb_index, 1);
            const bool tail = wt * 32 + 32 > actual_count;
            const uint32_t valid_lanes = tail ? actual_count - wt * 32 : 32;

            {
                // Every batch merges the same transposed ID tile: build it
                // once, release the input page to the reader immediately.
                reconfig_data_format_srca(index_cb_index);
                transpose_wh_init_short(index_cb_index);
                pack_reconfig_data_format(ids_transposed_cb);
                acquire_dst();
                transpose_wh_tile(index_cb_index, 0, 0);
                cb_reserve_back(ids_transposed_cb, 1);
                pack_tile(0, ids_transposed_cb);
                cb_push_back(ids_transposed_cb, 1);
                release_dst();
                cb_pop_front(index_cb_index, 1);
                cb_wait_front(ids_transposed_cb, 1);
            }
            {
                // Phase A: one matmul init, one score tile per batch. The
                // score CB holds a whole chunk, so this never blocks.
                reconfig_data_format_srca(in1_cb_index);
                mm_init_short(input_cb_index, in1_cb_index);
                pack_reconfig_data_format(matmul_out_cb);
                for (uint32_t i = 0; i < pairs; i++) {
                    acquire_dst();
                    for (uint32_t k = 0; k < query_tiles_width; k++) {
                        matmul_tiles(input_cb_index, in1_cb_index, i * query_tiles_width + k, k, 0);
                    }
                    cb_reserve_back(matmul_out_cb, 1);
                    pack_tile(0, matmul_out_cb);
                    cb_push_back(matmul_out_cb, 1);
                    release_dst();
                }
                // Pad the page's score window to batch_chunk slots (no data):
                // every window then starts at the CB base and tiles 0..pairs-1
                // are contiguous, as page_merge_mask's indexed reads need.
                if (pairs < batch_chunk) {
                    cb_reserve_back(matmul_out_cb, batch_chunk - pairs);
                    cb_push_back(matmul_out_cb, batch_chunk - pairs);
                }
                cb_pop_front(in1_cb_index, query_tiles_width);
            }
            const uint32_t merge_mask = threshold_mode
                ? page_merge_mask(pairs, matmul_out_cb, threshold_cb, scaler_cb, diff_cb, flag_cb)
                : 0xFFFFFFFFu;
            // The full ring holds pair i at the front at step i. Each step
            // pops the front and pushes the update (or the unchanged pair)
            // at the back; after the dummies rotate too, the ring is back in
            // pair order.
            for (uint32_t i = 0; i < pairs; i++) {
                cb_wait_front(matmul_out_cb, 1);
                if (((merge_mask >> i) & 1u) == 0) {
                    rotate_champion(champion_val_cb, champion_ind_cb);
                    cb_pop_front(matmul_out_cb, 1);
                    continue;
                }
                cb_wait_front(champion_val_cb, 1);
                cb_wait_front(champion_ind_cb, 1);
                if (tail) mask_tail_scores(matmul_out_cb, valid_lanes);
                acquire_dst();
                copy_tile_to_dst_init_short_with_dt(input_cb_index, champion_val_cb);
                copy_tile(champion_val_cb, 0, input_dest_start);
                copy_tile_to_dst_init_short_with_dt(champion_val_cb, champion_ind_cb);
                copy_tile(champion_ind_cb, 0, index_dest_start);
                reconfig_data_format_srca(matmul_out_cb);
                transpose_wh_init_short(matmul_out_cb);
                transpose_wh_tile(matmul_out_cb, 0, input_dest_end);
                copy_tile_to_dst_init_short_with_dt(matmul_out_cb, ids_transposed_cb);
                copy_tile(ids_transposed_cb, 0, index_dest_end);
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
            }
            for (uint32_t i = pairs; i < batch_chunk; i++) rotate_champion(champion_val_cb, champion_ind_cb);
            if (pairs < batch_chunk) {  // the score window's padding slots
                cb_wait_front(matmul_out_cb, batch_chunk - pairs);
                cb_pop_front(matmul_out_cb, batch_chunk - pairs);
            }
            // Every batch of the chunk has merged this page's IDs.
            cb_pop_front(ids_transposed_cb, 1);
        }

        // Publish partials in pair order, in row layout; the writer maps
        // pair i to its slot (= gathered tile id).
        for (uint32_t i = 0; i < pairs; i++) {
            transpose_and_pack(champion_val_cb, values_cb_index);
            transpose_and_pack(champion_ind_cb, output_ind_cb_index);
        }
        if (pairs < batch_chunk) {  // drop the dummies: the ring is empty between tasks
            cb_wait_front(champion_val_cb, batch_chunk - pairs);
            cb_wait_front(champion_ind_cb, batch_chunk - pairs);
            cb_pop_front(champion_val_cb, batch_chunk - pairs);
            cb_pop_front(champion_ind_cb, batch_chunk - pairs);
        }
        cb_pop_front(input_cb_index, chunk_tiles);
        if (threshold_mode) cb_pop_front(threshold_cb, batch_chunk);
    }
}
