#pragma once
#include <array>
#include <limits>
#include <set>
#include "query_grouping.hpp"

namespace ivf_partitioned {
struct Core { uint32_t x, y; };
struct Partition { Core leader, aggregator; std::vector<Core> workers; std::vector<uint32_t> batches; };
struct List { uint32_t start_page, count; };
struct Batch {
    uint32_t id = 0, valid_queries = 0, partition = 0;
    std::vector<uint32_t> lists;
    std::vector<std::vector<uint32_t>> worker_lists;
    std::vector<uint64_t> worker_pages;
    uint64_t pages = 0, vectors = 0, own_vectors = 0;
};
inline std::vector<Partition> make_partitions(uint32_t width, uint32_t height, uint32_t count,
                                             const std::string& layout, bool separate) {
    if (layout != "rows" && layout != "columns" && layout != "compact") throw std::invalid_argument("Unknown partition layout");
    const uint32_t overhead = separate ? 2 : 1;
    const uint64_t core_count = uint64_t(width) * height;
    if (!width || !height || !count || core_count > 64 || count > core_count / (overhead + 1))
        throw std::invalid_argument("Partitions need a leader, optional aggregator, and at least one worker; at most 64 cores are supported");
    std::vector<Core> cores;
    if (layout == "columns") {
        for (uint32_t x = 0; x < width; ++x) for (uint32_t y = 0; y < height; ++y) cores.push_back({x, y});
    } else {
        for (uint32_t y = 0; y < height; ++y) for (uint32_t i = 0; i < width; ++i)
            cores.push_back({layout == "compact" && y % 2 ? width - 1 - i : i, y});
    }
    std::vector<Partition> result;
    size_t next = 0;
    for (uint32_t p = 0; p < count; ++p) {
        const uint32_t group_size = width * height / count + (p < width * height % count);
        Partition part;
        part.leader = cores[next++];
        part.aggregator = separate ? cores[next++] : part.leader;
        for (uint32_t w = overhead; w < group_size; ++w) part.workers.push_back(cores[next++]);
        result.push_back(std::move(part));
    }
    return result;
}
inline std::vector<Batch> schedule_batches(const Selections& selections, const Grouping& grouping,
                                          const std::vector<List>& lists, std::vector<Partition>& partitions,
                                          MaskMode mask) {
    if (partitions.empty() || grouping.packed_to_original.size() != selections.size())
        throw std::invalid_argument("Scheduling requires partitions and a complete query permutation");
    for (const auto& partition : partitions)
        if (partition.workers.empty()) throw std::invalid_argument("Scheduling requires at least one worker in every partition");
    std::vector<bool> seen(selections.size());
    for (auto query : grouping.packed_to_original) {
        if (query >= selections.size() || seen[query]) throw std::invalid_argument("Invalid query permutation");
        seen[query] = true;
    }
    std::vector<Batch> batches;
    std::vector<double> partition_load(partitions.size(), 0);
    for (auto& partition : partitions) partition.batches.clear();
    for (uint32_t base = 0; base < grouping.packed_to_original.size(); base += 32) {
        Batch b;
        b.id = base / 32;
        b.valid_queries = std::min<uint32_t>(32, grouping.packed_to_original.size() - base);
        std::set<uint32_t> ids;
        for (uint32_t q = 0; q < b.valid_queries; ++q) {
            for (auto id : selections[grouping.packed_to_original[base + q]]) {
                ids.insert(id);
                b.own_vectors += lists.at(id).count;
            }
        }
        for (auto id : ids) {
            const auto& list = lists.at(id);
            if (!list.count) continue;
            if (mask == MaskMode::Off && list.count % 32)
                throw std::invalid_argument("--candidate-mask off encountered padded list " + std::to_string(id) + " (" + std::to_string(list.count) + " vectors)");
            b.lists.push_back(id);
            b.pages += pages_for(list.count);
            b.vectors += list.count;
        }
        std::stable_sort(b.lists.begin(), b.lists.end(), [&](auto a, auto c) {
            return pages_for(lists[a].count) > pages_for(lists[c].count);
        });
        double best_finish = std::numeric_limits<double>::infinity();
        for (uint32_t p = 0; p < partitions.size(); ++p) {
            const auto n = partitions[p].workers.size();
            std::vector<uint64_t> loads(n, 0);
            for (auto id : b.lists) *std::min_element(loads.begin(), loads.end()) += pages_for(lists[id].count);
            const double finish = partition_load[p] + *std::max_element(loads.begin(), loads.end()) + 1;
            if (finish < best_finish) { best_finish = finish; b.partition = p; }
        }
        auto& part = partitions[b.partition];
        b.worker_lists.resize(part.workers.size());
        b.worker_pages.resize(part.workers.size());
        for (auto id : b.lists) {
            const auto w = std::min_element(b.worker_pages.begin(), b.worker_pages.end()) - b.worker_pages.begin();
            b.worker_lists[w].push_back(id);
            b.worker_pages[w] += pages_for(lists[id].count);
        }
        partition_load[b.partition] = best_finish;
        part.batches.push_back(b.id);
        batches.push_back(std::move(b));
    }
    return batches;
}
} // namespace ivf_partitioned
