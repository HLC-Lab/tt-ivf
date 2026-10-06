#pragma once
#include <algorithm>
#include <cstdint>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "partition_protocol.hpp"

namespace ivf_partitioned {
using Selections = std::vector<std::vector<uint32_t>>;
struct Grouping {
    std::vector<uint32_t> packed_to_original, original_to_packed;
    uint64_t original_pages = 0, grouped_pages = 0;
    bool fell_back = false;
};
inline uint64_t union_cost(const Selections& selections, const std::vector<uint32_t>& order,
                           const std::vector<uint32_t>& pages) {
    uint64_t sum = 0;
    std::vector<bool> seen(pages.size());
    for (size_t base = 0; base < order.size(); base += 32) {
        std::fill(seen.begin(), seen.end(), false);
        for (size_t q = base; q < std::min(order.size(), base + 32); ++q)
            for (auto id : selections.at(order[q])) if (!seen.at(id)) { seen[id] = true; sum += pages[id]; }
    }
    return sum;
}
inline Grouping group_queries(const Selections& selections, const std::vector<uint32_t>& pages,
                              const std::string& mode, uint32_t window, uint32_t lookahead) {
    if (mode != "none" && mode != "primary-list" && mode != "weighted-union")
        throw std::invalid_argument("Unknown query grouping mode");
    if (!lookahead || (window && window % 32)) throw std::invalid_argument("Grouping window must be 0 or a multiple of 32; lookahead must be positive");
    for (const auto& selected : selections) {
        if (selected.empty()) throw std::invalid_argument("Empty coarse selection");
        auto distinct = selected;
        std::sort(distinct.begin(), distinct.end());
        if (std::adjacent_find(distinct.begin(), distinct.end()) != distinct.end())
            throw std::invalid_argument("Duplicate coarse list selection");
        for (auto id : selected) if (id >= pages.size()) throw std::invalid_argument("Coarse list ID outside index");
    }
    Grouping result;
    std::vector<uint32_t> original(selections.size());
    std::iota(original.begin(), original.end(), 0);
    result.original_pages = union_cost(selections, original, pages);
    const size_t step = window ? window : std::max<size_t>(1, selections.size());
    for (size_t begin = 0; begin < original.size(); begin += step) {
        std::deque<uint32_t> pending(original.begin() + begin, original.begin() + std::min(original.size(), begin + step));
        if (mode != "none") std::stable_sort(pending.begin(), pending.end(), [&](uint32_t a, uint32_t b) {
            if (selections[a] != selections[b]) return selections[a] < selections[b];
            return a < b;
        });
        std::vector<bool> available(selections.size(), false);
        std::vector<std::vector<uint32_t>> postings(pages.size());
        for (auto q : pending) {
            available[q] = true;
            if (mode == "weighted-union") for (auto id : selections[q]) postings[id].push_back(q);
        }
        std::vector<size_t> posting_front(pages.size());
        std::vector<uint32_t> stamp(selections.size());
        uint32_t generation = 0;
        auto discard_used = [&] { while (!pending.empty() && !available[pending.front()]) pending.pop_front(); };
        while (!pending.empty()) {
            std::vector<bool> present(pages.size());
            std::vector<uint32_t> union_ids;
            for (uint32_t lane = 0; lane < 32 && !pending.empty(); ++lane) {
                uint32_t best = pending.front();
                if (mode == "weighted-union" && lane) {
                    std::vector<uint32_t> candidates{pending.front()};
                    stamp[pending.front()] = ++generation;
                    std::stable_sort(union_ids.begin(), union_ids.end(), [&](auto a, auto b) { return pages[a] > pages[b]; });
                    uint32_t visits = 0;
                    for (auto id : union_ids) {
                        auto& front = posting_front[id];
                        while (front < postings[id].size() && !available[postings[id][front]]) ++front;
                        for (size_t i = front; i < postings[id].size() && visits < lookahead; ++i, ++visits) {
                            const auto q = postings[id][i];
                            if (available[q] && stamp[q] != generation) { stamp[q] = generation; candidates.push_back(q); }
                        }
                        if (visits == lookahead) break;
                    }
                    for (auto q : pending) {
                        if (visits++ >= lookahead) break;
                        if (available[q] && stamp[q] != generation) { stamp[q] = generation; candidates.push_back(q); }
                    }
                    uint64_t best_cost = UINT64_MAX;
                    for (auto q : candidates) {
                        uint64_t extra = 0;
                        for (auto id : selections[q]) if (!present[id]) extra += pages[id];
                        if (extra < best_cost) { best_cost = extra; best = q; }
                    }
                }
                result.packed_to_original.push_back(best);
                available[best] = false;
                for (auto id : selections[best]) if (!present[id]) { present[id] = true; union_ids.push_back(id); }
                discard_used();
            }
        }
    }
    result.grouped_pages = union_cost(selections, result.packed_to_original, pages);
    if (result.grouped_pages > result.original_pages) {
        result.packed_to_original = original;
        result.grouped_pages = result.original_pages;
        result.fell_back = true;
    }
    result.original_to_packed.resize(original.size());
    for (uint32_t p = 0; p < original.size(); ++p) result.original_to_packed[result.packed_to_original[p]] = p;
    return result;
}
} // namespace ivf_partitioned
