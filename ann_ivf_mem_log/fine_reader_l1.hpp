#pragma once

#include <cstdint>

namespace ivf_fine_reader {

constexpr uint64_t bf16_tile_bytes = 2048;
constexpr uint64_t index_tile_bytes = 4096;

constexpr uint64_t candidate_block_bytes(uint32_t dim) {
    const uint64_t tiles_per_block = (static_cast<uint64_t>(dim) + 31) / 32;
    return tiles_per_block * bf16_tile_bytes + index_tile_bytes;
}

constexpr uint64_t worker_cb_bytes(uint32_t dim, uint32_t prefetch_reader) {
    const uint64_t tiles_per_block = (static_cast<uint64_t>(dim) + 31) / 32;
    const uint64_t query_cb = 2 * tiles_per_block * bf16_tile_bytes;
    const uint64_t candidate_cbs = static_cast<uint64_t>(prefetch_reader) * candidate_block_bytes(dim);
    const uint64_t score_cb = 2 * bf16_tile_bytes;
    const uint64_t champion_cbs = 2 * bf16_tile_bytes + 2 * index_tile_bytes;
    const uint64_t output_cbs = 2 * bf16_tile_bytes + 2 * index_tile_bytes;
    const uint64_t script_cb = 8192;
    return query_cb + candidate_cbs + score_cb + champion_cbs + output_cbs + script_cb;
}

constexpr uint32_t max_prefetch_reader(uint32_t dim, uint64_t usable_l1_bytes) {
    const uint64_t fixed_bytes = worker_cb_bytes(dim, 0);
    if (dim == 0 || usable_l1_bytes < fixed_bytes) {
        return 0;
    }
    return static_cast<uint32_t>((usable_l1_bytes - fixed_bytes) / candidate_block_bytes(dim));
}

}  // namespace ivf_fine_reader
