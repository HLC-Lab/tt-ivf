#pragma once
#include <stdint.h>

// Fixed wire layout shared by host, data movement and all three TRISC builds.
namespace ivf_partitioned {
constexpr uint32_t max_workers = 63;
constexpr uint32_t record_words = 8;
constexpr uint32_t record_bytes = 32;
constexpr uint32_t script_page_bytes = 4096;
constexpr uint32_t records_per_page = script_page_bytes / record_bytes;
constexpr uint32_t offer_offset = 256;
constexpr uint32_t offer_stride = 32;
constexpr uint32_t done_offset = 2304;
constexpr uint32_t batch_done_offset = 64;
constexpr uint32_t scratch_offset = 128;
constexpr uint32_t cache_offset = 8192;
constexpr uint32_t control_bytes = 12288; // Includes NCRISC script cache.
constexpr uint32_t role_control_bytes = 16384; // BRISC has another cache.
constexpr uint32_t kind_query = 0;
constexpr uint32_t kind_candidate = 1;
constexpr uint32_t invalid_id = 0xffffffffu;
constexpr uint16_t invalid_score_bf16 = 0xc47a; // -1000, exactly representable
enum class MaskMode : uint32_t { Auto = 0, Additive = 1, Off = 2 };
// Batch: [global_batch, valid_queries, list_tasks, candidate_pages, ...].
// Worker list: [list_id, start_page, page_count, actual_count, tail_lanes, ...].
// Leader list: [worker_in_partition, list_id, start_page, page_count, actual_count, ...].
struct Record { uint32_t word[record_words]{}; };
static_assert(sizeof(Record) == record_bytes);
constexpr uint32_t pages_for(uint32_t count) { return count / 32 + (count % 32 != 0); }
constexpr uint32_t tail_lanes(uint32_t count) { return count ? (count - 1) % 32 + 1 : 0; }
constexpr bool needs_mask(uint32_t page, uint32_t pages, uint32_t tail) {
    return pages && page == pages - 1 && tail < 32;
}
} // namespace ivf_partitioned
