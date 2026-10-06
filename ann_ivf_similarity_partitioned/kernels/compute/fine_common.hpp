#pragma once
#include "../../partition_protocol.hpp"
#include "../../fine_tile_layout.hpp"
// Champions are empty here, so read and write pointers coincide.
// get_tile_address broadcasts the UNPACK pointer to every compute thread;
// only PACK initializes the payload before publishing it.
inline void initialize_champion(uint32_t val_cb, uint32_t ind_cb) {
    cb_reserve_back(val_cb, 1);
    [[maybe_unused]] const auto value_addr = get_tile_address(val_cb, 0);
    PACK({
        auto* values = reinterpret_cast<volatile uint16_t*>(value_addr);
        for (uint32_t i = 0; i < 1024; ++i) values[i] = ivf_partitioned::invalid_score_bf16;
    });
    cb_push_back(val_cb, 1);
    cb_reserve_back(ind_cb, 1);
    [[maybe_unused]] const auto index_addr = get_tile_address(ind_cb, 0);
    PACK({
        auto* indices = reinterpret_cast<volatile uint32_t*>(index_addr);
        for (uint32_t i = 0; i < 1024; ++i) indices[i] = ivf_partitioned::invalid_id;
    });
    cb_push_back(ind_cb, 1);
}
inline void mask_tail(uint32_t score_cb, uint32_t index_cb, uint32_t valid) {
    [[maybe_unused]] const auto score_addr = get_tile_address(score_cb, 0);
    [[maybe_unused]] const auto index_addr = get_tile_address(index_cb, 0);
    UNPACK({
        auto* scores = reinterpret_cast<volatile uint16_t*>(score_addr);
        auto* indices = reinterpret_cast<volatile uint32_t*>(index_addr);
        for (uint32_t q = 0; q < 32; ++q) for (uint32_t lane = valid; lane < 32; ++lane) {
            const uint32_t offset = fine_tile_offset(q, lane);
            scores[offset] = ivf_partitioned::invalid_score_bf16;
            indices[offset] = ivf_partitioned::invalid_id;
        }
    });
}
inline void transpose_output(uint32_t source, uint32_t target) {
    reconfig_data_format_srca(source);
    transpose_wh_init_short(source);
    pack_reconfig_data_format(target);
    cb_wait_front(source, 1);
    acquire_dst();
    cb_reserve_back(target, 1);
    transpose_wh_tile(source, 0, 0);
    pack_tile(0, target);
    cb_push_back(target, 1);
    release_dst();
    cb_pop_front(source, 1);
}
inline void combine_page(uint32_t scores, uint32_t indices) {
    constexpr uint32_t values = tt::CB::c_intermed3, ids = tt::CB::c_intermed4;
    cb_wait_front(scores, 1); cb_wait_front(indices, 1);
    cb_wait_front(values, 1); cb_wait_front(ids, 1);
    acquire_dst();
    // The previous batch's final ID transpose leaves SrcA in Int32. An
    // old/new conditional reconfig with two BF16 CBs would silently skip the
    // reset on the aggregator. Restore the actual format unconditionally.
    reconfig_data_format_srca(values);
    copy_tile_to_dst_init_short(values);
    copy_tile(values, 0, 0);
    copy_tile_to_dst_init_short_with_dt(values, ids);
    copy_tile(ids, 0, 2);
    reconfig_data_format_srca(scores); transpose_wh_init_short(scores); transpose_wh_tile(scores, 0, 1);
    reconfig_data_format_srca(indices); transpose_wh_init_short(indices); transpose_wh_tile(indices, 0, 3);
    reconfig_data_format_srca(values);
    topk_tile_init();
    topk_local_sort(0, 0, 5);
    cb_pop_front(values, 1); cb_pop_front(ids, 1);
    cb_reserve_back(values, 1); cb_reserve_back(ids, 1);
    pack_reconfig_data_format(values); pack_tile(0, values);
    pack_reconfig_data_format(ids); pack_tile(2, ids);
    cb_push_back(values, 1); cb_push_back(ids, 1);
    release_dst();
    cb_pop_front(scores, 1); cb_pop_front(indices, 1);
}
