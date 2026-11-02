#pragma once
// Per-list query tiles. For every non-empty inverted list, the queries that
// selected it (ascending query ID) are packed into 32-row "gathered tiles".
// The scan multiplies each list page only with that list's tiles, so a
// tile's rows are almost all queries that need the list, unlike 32-query
// batches whose union also scores lists that most rows did not select.
//
//   scan:  task = (list, <= batch_chunk of its tiles); slot = tile ID;
//          partial top-32 in row layout (row r = the tile's query row r).
//   merge: per 32 original-order queries, nprobe input tiles; row q' of input
//          tile j holds query q''s partial for its j-th selected list.
//
// Each query is compared only with its own nprobe lists (exact per-query
// IVF), so recall differs from the batch-union implementations.
#include <algorithm>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include "cluster_schedule.hpp"

namespace ivf_cluster_gather {

using ivf_cluster_major::ListLayout;
using ivf_cluster_major::ScanPair;
using ivf_cluster_major::ScanTask;

constexpr uint32_t sentinel_entry = 0xFFFFFFFFu;

struct GatherTile {
    uint32_t list = 0, first = 0, rows = 0;  // rows: requesters[first, first + rows)
};

struct GatherPlan {
    uint32_t queries = 0, nprobe = 0, batches = 0;
    std::vector<uint32_t> requesters;                  // per list, ascending query IDs
    std::vector<GatherTile> tiles;                     // tile ID = index, grouped by list
    std::vector<uint32_t> location;                    // [query * nprobe + j] = tile * 32 + row or sentinel
    std::vector<std::vector<ScanTask>> worker_tasks;   // pairs: batch = slot = tile ID
    std::vector<std::vector<uint32_t>> merge_batches;  // per merge core, ascending batch IDs
    uint64_t tasks = 0, pair_pages = 0, scan_pages = 0, useful_rows = 0;
    uint64_t min_worker_weight = 0, max_worker_weight = 0, total_worker_weight = 0;
};

inline GatherPlan plan_gather(const std::vector<std::vector<uint32_t>>& selections, uint32_t nprobe,
                              const std::vector<ListLayout>& lists, uint32_t batch_chunk,
                              uint32_t scan_workers, uint32_t merge_cores) {
    if (selections.empty() || !nprobe || !batch_chunk || !scan_workers || !merge_cores || lists.empty())
        throw std::invalid_argument("Gather plan needs queries, nprobe, a chunk size, cores and lists");
    const uint32_t nlist = uint32_t(lists.size());
    GatherPlan plan;
    plan.queries = uint32_t(selections.size());
    plan.nprobe = nprobe;
    plan.batches = (plan.queries + 31) / 32;
    if (uint64_t(plan.queries) * nprobe > UINT32_MAX / 2) throw std::invalid_argument("Too many query selections");

    // Requesters of every list in ascending query order (counting sort).
    std::vector<uint32_t> begin(nlist + 1, 0);
    for (const auto& row : selections) {
        if (row.size() != nprobe) throw std::invalid_argument("Every query needs exactly nprobe selected lists");
        for (auto id : row) {
            if (id >= nlist) throw std::invalid_argument("Coarse list ID outside index");
            ++begin[id + 1];
        }
    }
    for (uint32_t c = 0; c < nlist; ++c) begin[c + 1] += begin[c];
    plan.requesters.assign(begin[nlist], 0);
    std::vector<uint32_t> fill(begin.begin(), begin.end() - 1), seen(nlist, UINT32_MAX);
    for (uint32_t q = 0; q < plan.queries; ++q) {
        for (auto id : selections[q]) {
            if (seen[id] == q) throw std::invalid_argument("Duplicate coarse list selection");
            seen[id] = q;
            plan.requesters[fill[id]++] = q;
        }
    }

    // Tiles of each non-empty list, then every (query, j) location.
    std::vector<uint32_t> first_tile(nlist, UINT32_MAX);
    for (uint32_t c = 0; c < nlist; ++c) {
        if (lists[c].padded_count % 32 || lists[c].count > lists[c].padded_count)
            throw std::invalid_argument("Inverted list layout is not padded to 32-vector pages");
        const uint32_t requests = begin[c + 1] - begin[c];
        if (!lists[c].count || !requests) continue;  // empty lists are never read
        first_tile[c] = uint32_t(plan.tiles.size());
        for (uint32_t r = 0; r < requests; r += 32)
            plan.tiles.push_back({c, begin[c] + r, std::min(32u, requests - r)});
        plan.useful_rows += uint64_t(requests) * (lists[c].padded_count / 32);
    }
    if (plan.tiles.size() >= (uint64_t(1) << 27)) throw std::invalid_argument("Too many gathered tiles");
    plan.location.assign(size_t(plan.queries) * nprobe, sentinel_entry);
    std::vector<uint32_t> position(nlist, 0);
    for (uint32_t q = 0; q < plan.queries; ++q) {
        for (uint32_t j = 0; j < nprobe; ++j) {
            const uint32_t c = selections[q][j];
            if (first_tile[c] == UINT32_MAX) continue;  // empty list: sentinel row
            const uint32_t pos = position[c]++;          // q is the pos-th requester of c
            plan.location[size_t(q) * nprobe + j] = (first_tile[c] + pos / 32) * 32 + pos % 32;
        }
    }

    // Scan tasks: each list's tiles in near-equal chunks of <= batch_chunk.
    std::vector<ScanTask> tasks;
    for (uint32_t c = 0; c < nlist; ++c) {
        if (first_tile[c] == UINT32_MAX) continue;
        const uint32_t tiles = (begin[c + 1] - begin[c] + 31) / 32, pages = lists[c].padded_count / 32;
        const uint32_t chunks = (tiles + batch_chunk - 1) / batch_chunk;
        for (uint32_t k = 0, start = 0; k < chunks; ++k) {
            const uint32_t end = start + (tiles - start + (chunks - k) - 1) / (chunks - k);
            ScanTask task{c, lists[c].start_block, lists[c].count, pages, {}};
            for (uint32_t t = start; t < end; ++t) task.pairs.push_back({first_tile[c] + t, first_tile[c] + t});
            plan.pair_pages += uint64_t(pages) * task.pairs.size();
            plan.scan_pages += pages;
            tasks.push_back(std::move(task));
            start = end;
        }
    }
    plan.tasks = tasks.size();

    // Longest-processing-time-first placement on the scan cores.
    std::stable_sort(tasks.begin(), tasks.end(), [](const ScanTask& a, const ScanTask& b) {
        return std::tuple(b.weight(), b.pages, a.cid) < std::tuple(a.weight(), a.pages, b.cid);
    });
    using Load = std::pair<uint64_t, uint32_t>;
    std::priority_queue<Load, std::vector<Load>, std::greater<Load>> workers;
    for (uint32_t w = 0; w < scan_workers; ++w) workers.push({0, w});
    std::vector<uint64_t> worker_weight(scan_workers, 0);
    plan.worker_tasks.assign(scan_workers, {});
    for (auto& task : tasks) {
        const auto [load, w] = workers.top();
        workers.pop();
        worker_weight[w] = load + task.weight();
        workers.push({worker_weight[w], w});
        plan.worker_tasks[w].push_back(std::move(task));
    }
    for (auto weight : worker_weight) plan.total_worker_weight += weight;
    plan.min_worker_weight = *std::min_element(worker_weight.begin(), worker_weight.end());
    plan.max_worker_weight = *std::max_element(worker_weight.begin(), worker_weight.end());

    // Every batch has the same merge work (nprobe inputs): contiguous blocks.
    plan.merge_batches.assign(merge_cores, {});
    for (uint32_t b = 0; b < plan.batches; ++b)
        plan.merge_batches[uint64_t(b) * merge_cores / plan.batches].push_back(b);
    return plan;
}

// [batches, nprobe, then per batch: batch, entries j-major: (j, q') for
// q' = 0..31 = location of query 32 * batch + q''s j-th partial, or sentinel
// for padded queries and empty lists].
inline std::vector<uint32_t> encode_merge_script(const GatherPlan& plan, uint32_t merge_core) {
    const auto& batches = plan.merge_batches.at(merge_core);
    std::vector<uint32_t> script{uint32_t(batches.size()), plan.nprobe};
    script.reserve(2 + batches.size() * (1 + 32 * size_t(plan.nprobe)));
    for (auto b : batches) {
        script.push_back(b);
        for (uint32_t j = 0; j < plan.nprobe; ++j)
            for (uint32_t row = 0; row < 32; ++row) {
                const uint64_t q = uint64_t(b) * 32 + row;
                script.push_back(q < plan.queries ? plan.location[q * plan.nprobe + j] : sentinel_entry);
            }
    }
    return script;
}

// Two-pass search (merge skipping): the first `first_lists` selections of
// every query (coarse rank order) and the remaining ones.
inline std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> split_selections(
    const std::vector<std::vector<uint32_t>>& selections, uint32_t first_lists) {
    std::vector<std::vector<uint32_t>> first(selections.size()), rest(selections.size());
    for (size_t q = 0; q < selections.size(); ++q) {
        if (first_lists == 0 || first_lists >= selections[q].size())
            throw std::invalid_argument("The first pass needs between 1 and nprobe - 1 lists");
        first[q].assign(selections[q].begin(), selections[q].begin() + first_lists);
        rest[q].assign(selections[q].begin() + first_lists, selections[q].end());
    }
    return {std::move(first), std::move(rest)};
}

// Place a plan's partials at slots base + k: scan pair slots and merge
// locations move; pair.batch stays the local tile ID (the row of the pass's
// own tile-query table).
inline void shift_slots(GatherPlan& plan, uint32_t base) {
    if (uint64_t(base) + plan.tiles.size() >= (uint64_t(1) << 27)) throw std::invalid_argument("Too many partial slots");
    for (auto& tasks : plan.worker_tasks)
        for (auto& task : tasks)
            for (auto& pair : task.pairs) pair.slot += base;
    for (auto& entry : plan.location)
        if (entry != sentinel_entry) entry += base * 32;
}

// Upper bound on a pass's tile count before planning it: each list's tiles
// are ceil(requests / 32) <= requests / 32 + 1 over at most nlist lists.
inline uint64_t max_tiles(uint64_t queries, uint32_t lists_per_query, uint32_t nlist) {
    return (queries * lists_per_query + 31) / 32 + nlist;
}

// Final merge of the two-pass search, same layout as encode_merge_script with
// 1 + second.nprobe inputs: input 0 = the first pass's merged row of query q
// (slot merged_base + q / 32, row q % 32), inputs 1.. = the second pass's
// partials (second must already be shifted to its global tile IDs).
inline std::vector<uint32_t> encode_final_merge_script(const GatherPlan& second, uint32_t merged_base, uint32_t merge_core) {
    const auto& batches = second.merge_batches.at(merge_core);
    const uint32_t inputs = 1 + second.nprobe;
    std::vector<uint32_t> script{uint32_t(batches.size()), inputs};
    script.reserve(2 + batches.size() * (1 + 32 * size_t(inputs)));
    for (auto b : batches) {
        script.push_back(b);
        for (uint32_t row = 0; row < 32; ++row)
            script.push_back(uint64_t(b) * 32 + row < second.queries ? (merged_base + b) * 32 + row : sentinel_entry);
        for (uint32_t j = 0; j < second.nprobe; ++j)
            for (uint32_t row = 0; row < 32; ++row) {
                const uint64_t q = uint64_t(b) * 32 + row;
                script.push_back(q < second.queries ? second.location[q * second.nprobe + j] : sentinel_entry);
            }
    }
    return script;
}

// Merge-core L1: two input pairs, champion pair, output pair, the reader's
// sentinel tile pair and the script page.
constexpr uint64_t merge_cb_bytes(uint64_t script_bytes) {
    return 4 * (ivf_cluster_major::scan_l1::bf16_tile + ivf_cluster_major::scan_l1::index_tile) +
           (ivf_cluster_major::scan_l1::bf16_tile + ivf_cluster_major::scan_l1::index_tile) + script_bytes;
}

}  // namespace ivf_cluster_gather
