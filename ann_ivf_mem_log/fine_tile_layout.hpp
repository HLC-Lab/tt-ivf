#pragma once

#include <stdint.h>

// A 32x32 tile is stored as four 16x16 faces in row-major face order.
static inline constexpr uint32_t fine_tile_offset(uint32_t row, uint32_t column) {
    return ((row / 16) * 2 + column / 16) * 256 + (row % 16) * 16 + column % 16;
}

static inline constexpr bool fine_lane_valid(
    uint32_t query_mask,
    uint32_t query_row,
    uint32_t page_slot,
    uint32_t candidate_lane,
    uint32_t valid_end_slot) {
    return ((query_mask >> query_row) & 1u) != 0 && page_slot + candidate_lane < valid_end_slot;
}
