#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include "../partition_schedule.hpp"
#include "../partition_l1.hpp"
#include "../fine_tile_layout.hpp"
using namespace ivf_partitioned;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        for (uint32_t count : {0u, 1u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u}) {
            uint32_t masked = 0;
            for (uint32_t page = 0; page < pages_for(count); ++page)
                masked += needs_mask(page, pages_for(count), tail_lanes(count));
            require(masked == uint32_t(count && count % 32), "Tail mask applied on a full page or omitted on padding");
        }
        std::set<uint32_t> offsets;
        for (uint32_t row = 0; row < 32; ++row) for (uint32_t lane = 0; lane < 32; ++lane)
            offsets.insert(fine_tile_offset(row, lane));
        require(offsets.size() == 1024 && *offsets.rbegin() == 1023, "Invalid face layout");
        for (uint32_t n : {1u, 16u, 32u, 33u, 64u, 320u, 10000u}) {
            Selections selections(n);
            for (uint32_t q = 0; q < n; ++q) selections[q] = {(q * 17) % 128, (q * 17 + 1) % 128};
            std::vector<uint32_t> pages(128);
            std::vector<List> lists(128);
            for (uint32_t id = 0; id < pages.size(); ++id) {
                pages[id] = id % 11 + 1; lists[id] = {id * 20, pages[id] * 32};
            }
            for (const std::string mode : {"none", "primary-list", "weighted-union"}) {
                const auto grouping = group_queries(selections, pages, mode, 1024, 128);
                const auto repeat = group_queries(selections, pages, mode, 1024, 128);
                require(grouping.packed_to_original == repeat.packed_to_original, "Grouping is not deterministic");
                require(grouping.grouped_pages <= grouping.original_pages, "Grouping fallback failed");
                std::set<uint32_t> seen(grouping.packed_to_original.begin(), grouping.packed_to_original.end());
                require(seen.size() == n && *seen.rbegin() == n - 1, "Permutation lost/duplicated a query");
                for (uint32_t q = 0; q < n; ++q)
                    require(grouping.packed_to_original[grouping.original_to_packed[q]] == q, "Inverse permutation is wrong");
                for (uint32_t count : {1u, 4u, 6u, 8u, 12u}) {
                    auto parts = make_partitions(8, 8, count, "rows", false);
                    auto batches = schedule_batches(selections, grouping, lists, parts, MaskMode::Auto);
                    std::set<uint32_t> assigned;
                    for (const auto& part : parts) for (auto batch : part.batches)
                        require(assigned.insert(batch).second, "Batch assigned to multiple leaders");
                    require(assigned.size() == (n + 31) / 32, "Leader queue missed a batch");
                    for (const auto& batch : batches) {
                        require(batch.valid_queries == std::min(32u, n - batch.id * 32), "Padded query row counted");
                        uint64_t total = 0;
                        std::set<uint32_t> tasks;
                        for (uint32_t w = 0; w < batch.worker_lists.size(); ++w) {
                            uint64_t load = 0;
                            for (auto id : batch.worker_lists[w]) {
                                require(tasks.insert(id).second, "List assigned twice inside a batch");
                                load += pages_for(lists[id].count);
                            }
                            require(load == batch.worker_pages[w], "Worker load does not match tasks"); total += load;
                        }
                        require(total == batch.pages && tasks.size() == batch.lists.size(), "Union and worker schedule disagree");
                    }
                }
            }
        }
        for (const std::string layout : {"rows", "columns", "compact"}) for (bool separate : {false, true}) {
            for (uint32_t count : {4u, 6u, 8u, 12u}) {
                const auto parts = make_partitions(8, 8, count, layout, separate);
                std::set<std::pair<uint32_t, uint32_t>> cores;
                for (const auto& part : parts) {
                    require(!part.workers.empty(), "Partition has no worker");
                    require(cores.emplace(part.leader.x, part.leader.y).second, "Leader core overlaps another role");
                    if (separate) require(cores.emplace(part.aggregator.x, part.aggregator.y).second, "Aggregator overlaps");
                    for (const auto core : part.workers) require(cores.emplace(core.x, core.y).second, "Worker overlaps");
                }
                require(cores.size() == 64, "Topology discarded cores");
            }
        }
        Selections selected{{0}, {1}};
        std::vector<List> lists{{0, 33}, {2, 0}};
        auto group = group_queries(selected, {2, 0}, "none", 0, 16);
        auto parts = make_partitions(8, 8, 8, "rows", false);
        bool rejected = false;
        try { schedule_batches(selected, group, lists, parts, MaskMode::Off); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Unsafe mask-off launch accepted");
        lists[0].count = 32;
        const auto safe = schedule_batches(selected, group, lists, parts, MaskMode::Off);
        require(safe[0].lists.size() == 1 && safe[0].pages == 1, "Empty list scheduled a dummy page");
        require(offer_offset + max_workers * offer_stride <= cache_offset, "Offer mailboxes overlap script cache");
        require(done_offset + max_workers * 32 <= cache_offset, "Done mailboxes overlap script cache");
        require(worker_cb_bytes(100, 2, MaskMode::Additive) - worker_cb_bytes(100, 2, MaskMode::Auto) == 4096,
                "Mask capacity absent from L1 budget");
        auto must_reject = [](auto operation) {
            bool failed = false;
            try { operation(); } catch (const std::invalid_argument&) { failed = true; }
            require(failed, "Malformed host input was accepted");
        };
        must_reject([] { make_partitions(0, 8, 1, "rows", false); });
        must_reject([] { make_partitions(UINT32_MAX, UINT32_MAX, 1, "rows", false); });
        must_reject([] { group_queries({{0, 0}}, {1}, "weighted-union", 0, 16); });
        must_reject([&] { std::vector<Partition> empty; schedule_batches(selected, group, lists, empty, MaskMode::Auto); });
        must_reject([&] {
            auto broken = group;
            broken.packed_to_original[1] = broken.packed_to_original[0];
            schedule_batches(selected, broken, lists, parts, MaskMode::Auto);
        });
        require(worker_cb_bytes(100, UINT32_MAX, MaskMode::Auto) > working_set_cap,
                "Large prefetch depth wrapped the L1 budget");
        std::cout << "Grouping, permutation, tail masks, whole-list schedules, queue ownership, topology and L1 tests passed.\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
