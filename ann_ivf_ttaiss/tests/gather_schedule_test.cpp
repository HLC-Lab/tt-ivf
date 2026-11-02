// Portable checks for the per-list query-tile plan.
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include "../gather_schedule.hpp"

using namespace ivf_cluster_gather;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void must_reject(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Malformed plan input was accepted");
}

void check(uint32_t queries, uint32_t nlist, uint32_t nprobe, uint32_t chunk, uint32_t cores, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<ListLayout> lists(nlist);
    for (uint32_t c = 0, start = 0; c < nlist; ++c) {
        const uint32_t count = rng() % 5 == 0 ? 0 : rng() % 150 + 1;
        const uint32_t blocks = std::max(1u, (count + 31) / 32);
        lists[c] = {start, count, blocks * 32};
        start += blocks;
    }
    std::vector<std::vector<uint32_t>> selections(queries);
    for (auto& row : selections) {
        std::set<uint32_t> chosen;
        while (chosen.size() < nprobe) chosen.insert(rng() % (rng() % 3 ? std::min(nlist, 2 * nprobe) : nlist));
        row.assign(chosen.begin(), chosen.end());
        std::shuffle(row.begin(), row.end(), rng);
    }
    const auto plan = plan_gather(selections, nprobe, lists, chunk, cores, cores);
    require(plan.batches == (queries + 31) / 32, "Wrong batch count");

    // Tiles: grouped by list, <= 32 ascending requesters, every request once.
    std::vector<uint32_t> requests(nlist, 0);
    for (const auto& row : selections) for (auto c : row) ++requests[c];
    uint64_t expected_pairs = 0, useful = 0;
    std::vector<uint32_t> tiles_of(nlist, 0);
    for (size_t t = 0; t < plan.tiles.size(); ++t) {
        const auto& tile = plan.tiles[t];
        require(tile.rows >= 1 && tile.rows <= 32 && lists[tile.list].count, "Bad tile rows or empty list");
        require(t == 0 || plan.tiles[t - 1].list <= tile.list, "Tiles are not grouped by list");
        for (uint32_t r = 1; r < tile.rows; ++r)
            require(plan.requesters[tile.first + r - 1] < plan.requesters[tile.first + r], "Rows not ascending");
        ++tiles_of[tile.list];
    }
    for (uint32_t c = 0; c < nlist; ++c) {
        if (!lists[c].count) { require(!tiles_of[c], "Empty list got tiles"); continue; }
        require(tiles_of[c] == (requests[c] + 31) / 32, "Tile count differs from ceil(requests / 32)");
        expected_pairs += uint64_t(tiles_of[c]) * (lists[c].padded_count / 32);
        useful += uint64_t(requests[c]) * (lists[c].padded_count / 32);
    }
    require(plan.pair_pages == expected_pairs && plan.useful_rows == useful, "Pair/useful page counts are wrong");

    // Locations: (q, j) -> the tile row that holds q for its j-th list.
    for (uint32_t q = 0; q < queries; ++q)
        for (uint32_t j = 0; j < nprobe; ++j) {
            const uint32_t entry = plan.location[size_t(q) * nprobe + j], c = selections[q][j];
            if (!lists[c].count) { require(entry == sentinel_entry, "Empty list must map to a sentinel"); continue; }
            const auto& tile = plan.tiles.at(entry / 32);
            require(tile.list == c && entry % 32 < tile.rows && plan.requesters[tile.first + entry % 32] == q,
                    "Location does not hold this query's row");
        }

    // Scan tasks: every tile exactly once, slot = tile, chunk <= G, one list.
    std::vector<uint32_t> scanned(plan.tiles.size(), 0);
    for (const auto& tasks : plan.worker_tasks)
        for (const auto& task : tasks) {
            require(!task.pairs.empty() && task.pairs.size() <= chunk, "Task chunk size out of range");
            require(task.start_block == lists[task.cid].start_block && task.count == lists[task.cid].count,
                    "Task list layout is wrong");
            for (const auto& pair : task.pairs) {
                require(pair.batch == pair.slot && plan.tiles.at(pair.batch).list == task.cid, "Pair is not its list's tile");
                ++scanned[pair.batch];
            }
        }
    for (auto count : scanned) require(count == 1, "A tile is scanned zero or several times");

    // Merge: every batch once; scripts carry each query's locations, padding sentinel.
    std::vector<uint32_t> merged(plan.batches, 0);
    for (uint32_t m = 0; m < cores; ++m) {
        const auto script = encode_merge_script(plan, m);
        require(script[0] == plan.merge_batches[m].size() && script[1] == nprobe, "Merge script header is wrong");
        size_t ptr = 2;
        for (uint32_t i = 0; i < script[0]; ++i) {
            const uint32_t b = script[ptr++];
            ++merged.at(b);
            for (uint32_t j = 0; j < nprobe; ++j)
                for (uint32_t row = 0; row < 32; ++row) {
                    const uint64_t q = uint64_t(b) * 32 + row;
                    const uint32_t expected = q < queries ? plan.location[q * nprobe + j] : sentinel_entry;
                    require(script[ptr++] == expected, "Merge script entry is wrong");
                }
        }
        require(ptr == script.size(), "Merge script has trailing words");
    }
    for (auto count : merged) require(count == 1, "A batch is merged zero or several times");
}

// Two-pass search: split, global tile IDs and the final merge script.
void check_two_pass(uint32_t queries, uint32_t nlist, uint32_t nprobe, uint32_t first, uint32_t cores, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<ListLayout> lists(nlist);
    for (uint32_t c = 0, start = 0; c < nlist; ++c) {
        const uint32_t count = rng() % 5 == 0 ? 0 : rng() % 150 + 1;
        const uint32_t blocks = std::max(1u, (count + 31) / 32);
        lists[c] = {start, count, blocks * 32};
        start += blocks;
    }
    std::vector<std::vector<uint32_t>> selections(queries);
    for (auto& row : selections) {
        std::set<uint32_t> chosen;
        while (chosen.size() < nprobe) chosen.insert(rng() % nlist);
        row.assign(chosen.begin(), chosen.end());
        std::shuffle(row.begin(), row.end(), rng);
    }
    const auto [head, rest] = split_selections(selections, first);
    for (uint32_t q = 0; q < queries; ++q) {
        require(head[q].size() == first && rest[q].size() == nprobe - first, "Split sizes are wrong");
        for (uint32_t j = 0; j < nprobe; ++j)
            require((j < first ? head[q][j] : rest[q][j - first]) == selections[q][j], "Split changed the rank order");
    }
    const auto plan1 = plan_gather(head, first, lists, 32, cores, cores);
    auto plan2 = plan_gather(rest, nprobe - first, lists, 32, cores, cores);
    const auto local = plan2;
    const uint32_t base = uint32_t(plan1.tiles.size());
    // Slot layout of the overlapped search: pass-1 tiles, merged rows, pass-2 tiles.
    const uint32_t merged_base = base;
    const uint32_t pass2_base = merged_base + plan1.batches;
    require(plan2.tiles.size() <= max_tiles(queries, nprobe - first, nlist) &&
            plan1.tiles.size() <= max_tiles(queries, first, nlist), "Tile bound is not an upper bound");
    shift_slots(plan2, pass2_base);
    // Every scan slot and location moved by exactly pass2_base; batches stay local.
    for (uint32_t w = 0; w < cores; ++w)
        for (size_t t = 0; t < plan2.worker_tasks[w].size(); ++t)
            for (size_t i = 0; i < plan2.worker_tasks[w][t].pairs.size(); ++i) {
                const auto& moved = plan2.worker_tasks[w][t].pairs[i];
                const auto& original = local.worker_tasks[w][t].pairs[i];
                require(moved.batch == original.batch && moved.slot == original.slot + pass2_base,
                        "Second-pass scan pair not moved to its global slot");
            }
    for (size_t e = 0; e < plan2.location.size(); ++e)
        require(local.location[e] == sentinel_entry ? plan2.location[e] == sentinel_entry
                                                    : plan2.location[e] == local.location[e] + 32 * pass2_base,
                "Second-pass location not moved to its global slot");
    // Final script: input 0 = the first-pass merged row, then the pass-2 rows.
    require(plan1.merge_batches == plan2.merge_batches, "The passes split merge batches differently");
    std::vector<uint32_t> seen(plan2.batches, 0);
    for (uint32_t core = 0; core < cores; ++core) {
        const auto script = encode_final_merge_script(plan2, merged_base, core);
        const uint32_t inputs = 1 + nprobe - first;
        require(script[0] == plan2.merge_batches[core].size() && script[1] == inputs, "Final script header is wrong");
        size_t ptr = 2;
        for (uint32_t b = 0; b < script[0]; ++b) {
            const uint32_t batch = script[ptr++];
            ++seen.at(batch);
            for (uint32_t j = 0; j < inputs; ++j)
                for (uint32_t row = 0; row < 32; ++row) {
                    const uint64_t q = uint64_t(batch) * 32 + row;
                    const uint32_t entry = script[ptr++];
                    if (q >= queries) {
                        require(entry == sentinel_entry, "Padded query row is not a sentinel");
                    } else if (j == 0) {
                        require(entry == (merged_base + batch) * 32 + row, "Input 0 is not the first-pass merged row");
                    } else {
                        require(entry == plan2.location[q * (nprobe - first) + j - 1], "Pass-2 input is not the query's partial");
                    }
                }
        }
        require(ptr == script.size(), "Final script has trailing words");
    }
    for (auto count : seen) require(count == 1, "A batch is not merged exactly once");
    must_reject([&] { split_selections(selections, 0); });
    must_reject([&] { split_selections(selections, nprobe); });
}

int main() {
    try {
        uint32_t seed = 5;
        for (uint32_t queries : {1u, 31u, 32u, 33u, 700u, 3001u})
            for (uint32_t nprobe : {1u, 4u, 16u})
                for (uint32_t chunk : {1u, 3u, 32u})
                    for (uint32_t cores : {1u, 64u})
                        check(queries, 64, nprobe, chunk, cores, seed++);
        for (uint32_t queries : {1u, 33u, 700u})
            for (uint32_t first : {1u, 3u})
                for (uint32_t cores : {1u, 7u})
                    check_two_pass(queries, 48, 5, first, cores, seed++);
        std::vector<ListLayout> lists{{0, 5, 32}, {1, 0, 32}};
        must_reject([&] { plan_gather({{0, 0}}, 2, lists, 32, 1, 1); });      // duplicate list
        must_reject([&] { plan_gather({{0, 7}}, 2, lists, 32, 1, 1); });      // ID outside index
        must_reject([&] { plan_gather({{0}}, 2, lists, 32, 1, 1); });         // short row
        const auto only_empty = plan_gather({{1}}, 1, lists, 32, 1, 1);
        require(only_empty.tiles.empty() && only_empty.location[0] == sentinel_entry, "Empty-list query must be sentinel");
        std::cout << "Gather tiles, locations, scan tasks, merge scripts and the two-pass split passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
