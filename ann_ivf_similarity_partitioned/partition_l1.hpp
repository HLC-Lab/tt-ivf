#pragma once
#include <algorithm>
#include <cstdint>
#include "partition_protocol.hpp"
namespace ivf_partitioned {
constexpr uint64_t query_bytes(uint32_t dim) { return uint64_t((dim + 31) / 32) * 2048; }
constexpr uint64_t pair_bytes(uint32_t dim) { return query_bytes(dim) + 4096; }
constexpr uint64_t coarse_cb_bytes(uint32_t dim) { return 4 * query_bytes(dim) + 36 * 1024; }
constexpr uint64_t worker_cb_bytes(uint32_t dim, uint32_t depth, MaskMode mask) {
    return 2 * query_bytes(dim) + depth * pair_bytes(dim) + 8 * 2048 + 4 * 4096 +
           4 * record_bytes + (mask == MaskMode::Additive ? depth * 2048 : 0);
}
constexpr uint64_t leader_raw_bytes(uint32_t dim, uint32_t depth, uint32_t max_tasks) {
    return role_control_bytes + 2 * query_bytes(dim) + uint64_t(max_tasks + 1) * record_bytes + depth * pair_bytes(dim);
}
constexpr uint64_t aggregator_cb_bytes = 6 * 2048 + 6 * 4096 + 4 * record_bytes;
constexpr uint64_t working_set_cap = 512 * 1024;
} // namespace ivf_partitioned
