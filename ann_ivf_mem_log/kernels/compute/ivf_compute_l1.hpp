#pragma once

#include <cstdint>

#include "../../fine_tile_layout.hpp"

// Include after compute_kernel_api.h. Compute code is compiled separately for
// UNPACK, MATH, and PACK; only the CB producer may initialize its L1 payload.
// Winner CBs are empty here, so their read and write pointers identify the
// same page. get_tile_address broadcasts that address to all three threads.
inline void initialize_champion(uint32_t val_cb, uint32_t ind_cb) {
    cb_reserve_back(val_cb, 1);
    [[maybe_unused]] const auto val_addr = get_tile_address(val_cb, 0);
    PACK({
        volatile uint16_t* values = reinterpret_cast<volatile uint16_t*>(val_addr);
        for (uint32_t i = 0; i < 1024; ++i) {
            values[i] = 0xC61C;
        }
    });
    cb_push_back(val_cb, 1);

    cb_reserve_back(ind_cb, 1);
    [[maybe_unused]] const auto ind_addr = get_tile_address(ind_cb, 0);
    PACK({
        volatile uint32_t* indices = reinterpret_cast<volatile uint32_t*>(ind_addr);
        for (uint32_t i = 0; i < 1024; ++i) {
            indices[i] = 0xFFFFFFFFu;
        }
    });
    cb_push_back(ind_cb, 1);
}

// UNPACK owns these input pages until it pops them. Mask invalid lanes before
// issuing the unpack instructions so padded zero vectors cannot beat valid
// candidates with negative cosine similarity.
inline void mask_fine_candidates(
    uint32_t score_cb,
    uint32_t index_cb,
    uint32_t page_slot,
    uint32_t query_mask,
    uint32_t valid_end_slot) {
    [[maybe_unused]] const auto score_addr = get_tile_address(score_cb, 0);
    [[maybe_unused]] const auto index_addr = get_tile_address(index_cb, 0);
    UNPACK({
        volatile uint16_t* scores = reinterpret_cast<volatile uint16_t*>(score_addr);
        volatile uint32_t* indices = reinterpret_cast<volatile uint32_t*>(index_addr);
        for (uint32_t q = 0; q < 32; ++q) {
            for (uint32_t lane = 0; lane < 32; ++lane) {
                if (fine_lane_valid(query_mask, q, page_slot, lane, valid_end_slot)) {
                    continue;
                }
                const uint32_t offset = fine_tile_offset(q, lane);
                scores[offset] = 0xC61C;
                indices[offset] = 0xFFFFFFFFu;
            }
        }
    });
}
