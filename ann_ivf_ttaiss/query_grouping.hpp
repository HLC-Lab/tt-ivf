#pragma once
// Host grouping/selection adapted from ann_ivf_similarity_partitioned.
#include <algorithm>
#include <cstdint>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ivf_cluster_major {
using Selections = std::vector<std::vector<uint32_t>>;
struct GroupingConfig {
    std::string mode = "none";
    uint32_t window = 0, lookahead = 256;
};
inline void validate_grouping_config(const GroupingConfig& config) {
    if (config.mode != "none" && config.mode != "primary-list" && config.mode != "weighted-union")
        throw std::invalid_argument("--query-grouping must be none, primary-list or weighted-union");
    if (config.window && config.window % 32)
        throw std::invalid_argument("--grouping-window must be 0 or a multiple of 32");
    if (!config.lookahead) throw std::invalid_argument("--grouping-lookahead must be positive");
}
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
inline Grouping group_queries_reference(const Selections& selections, const std::vector<uint32_t>& pages,
                              const std::string& mode, uint32_t window, uint32_t lookahead) {
    validate_grouping_config({mode, window, lookahead});
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

// Same output as group_queries_reference (the original heuristic), faster:
//  - the batch union stays sorted by pages (descending) through a stable merge
//    of the previous sorted union with the new IDs, instead of a stable_sort
//    every lane; stable_sort(sorted + appended) equals that merge;
//  - candidates are costed in collection order: the original takes the first
//    strictly cheapest one, and no cost is below 0, so the first zero-cost
//    candidate ends the search, and a partial sum >= the best cost is
//    abandoned. Lazily skipped posting fronts land on the same entries later
//    because a query never becomes available again.
inline Grouping group_queries(const Selections& selections, const std::vector<uint32_t>& pages,
                              const std::string& mode, uint32_t window, uint32_t lookahead) {
    if (mode != "weighted-union") return group_queries_reference(selections, pages, mode, window, lookahead);
    validate_grouping_config({mode, window, lookahead});
    // Same checks and messages as the reference, without sorting every row.
    std::vector<uint32_t> seen(pages.size(), 0);
    uint32_t row_mark = 0;
    for (const auto& selected : selections) {
        if (selected.empty()) throw std::invalid_argument("Empty coarse selection");
        if (std::all_of(selected.begin(), selected.end(), [&](uint32_t id) { return id < pages.size(); })) {
            ++row_mark;
            for (auto id : selected) {
                if (seen[id] == row_mark) throw std::invalid_argument("Duplicate coarse list selection");
                seen[id] = row_mark;
            }
        } else {
            auto distinct = selected;
            std::sort(distinct.begin(), distinct.end());
            if (std::adjacent_find(distinct.begin(), distinct.end()) != distinct.end())
                throw std::invalid_argument("Duplicate coarse list selection");
            throw std::invalid_argument("Coarse list ID outside index");
        }
    }
    // Contiguous selections: row q is ids[offsets[q] .. offsets[q + 1]).
    std::vector<uint32_t> offsets(selections.size() + 1, 0), ids;
    for (size_t q = 0; q < selections.size(); ++q) offsets[q + 1] = offsets[q] + uint32_t(selections[q].size());
    ids.reserve(offsets.back());
    for (const auto& row : selections) ids.insert(ids.end(), row.begin(), row.end());
    auto by_pages = [&](uint32_t a, uint32_t b) { return pages[a] > pages[b]; };

    Grouping result;
    std::vector<uint32_t> original(selections.size());
    std::iota(original.begin(), original.end(), 0);
    result.original_pages = union_cost(selections, original, pages);
    result.packed_to_original.reserve(selections.size());
    const size_t step = window ? window : std::max<size_t>(1, selections.size());
    std::vector<uint8_t> available(selections.size(), 0), present(pages.size(), 0);
    std::vector<uint32_t> stamp(selections.size(), 0), union_ids, fresh, merged;
    std::vector<size_t> posting_front(pages.size());
    uint32_t generation = 0;
    for (size_t begin = 0; begin < original.size(); begin += step) {
        std::deque<uint32_t> pending(original.begin() + begin, original.begin() + std::min(original.size(), begin + step));
        std::stable_sort(pending.begin(), pending.end(), [&](uint32_t a, uint32_t b) {
            if (selections[a] != selections[b]) return selections[a] < selections[b];
            return a < b;
        });
        std::vector<std::vector<uint32_t>> postings(pages.size());
        for (auto q : pending) {
            available[q] = 1;
            for (uint32_t j = offsets[q]; j < offsets[q + 1]; ++j) postings[ids[j]].push_back(q);
        }
        std::fill(posting_front.begin(), posting_front.end(), 0);
        auto discard_used = [&] { while (!pending.empty() && !available[pending.front()]) pending.pop_front(); };
        while (!pending.empty()) {
            for (auto id : union_ids) present[id] = 0;
            union_ids.clear();
            for (uint32_t lane = 0; lane < 32 && !pending.empty(); ++lane) {
                uint32_t best = pending.front();
                if (lane) {
                    stamp[best] = ++generation;
                    uint64_t best_cost = UINT64_MAX;
                    auto consider = [&](uint32_t q) {  // true: a zero-cost candidate was found
                        uint64_t extra = 0;
                        for (uint32_t j = offsets[q]; j < offsets[q + 1]; ++j) {
                            if (!present[ids[j]]) extra += pages[ids[j]];
                            if (extra >= best_cost) return false;
                        }
                        best_cost = extra;
                        best = q;
                        return extra == 0;
                    };
                    bool done = consider(best);
                    uint32_t visits = 0;
                    for (size_t u = 0; !done && u < union_ids.size(); ++u) {
                        const uint32_t id = union_ids[u];
                        auto& front = posting_front[id];
                        const auto& posting = postings[id];
                        while (front < posting.size() && !available[posting[front]]) ++front;
                        for (size_t i = front; i < posting.size() && visits < lookahead; ++i, ++visits) {
                            const uint32_t q = posting[i];
                            if (available[q] && stamp[q] != generation) {
                                stamp[q] = generation;
                                if (consider(q)) { done = true; break; }
                            }
                        }
                        if (visits == lookahead) break;
                    }
                    if (!done) {
                        for (auto q : pending) {
                            if (visits++ >= lookahead) break;
                            if (available[q] && stamp[q] != generation) {
                                stamp[q] = generation;
                                if (consider(q)) break;
                            }
                        }
                    }
                }
                result.packed_to_original.push_back(best);
                available[best] = 0;
                fresh.clear();
                for (uint32_t j = offsets[best]; j < offsets[best + 1]; ++j)
                    if (!present[ids[j]]) { present[ids[j]] = 1; fresh.push_back(ids[j]); }
                std::stable_sort(fresh.begin(), fresh.end(), by_pages);
                merged.resize(union_ids.size() + fresh.size());
                std::merge(union_ids.begin(), union_ids.end(), fresh.begin(), fresh.end(), merged.begin(), by_pages);
                union_ids.swap(merged);
                discard_used();
            }
        }
    }
    for (auto id : union_ids) present[id] = 0;
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

inline bool is_identity(const Grouping& grouping) {
    for (size_t packed = 0; packed < grouping.packed_to_original.size(); ++packed)
        if (grouping.packed_to_original[packed] != packed) return false;
    return true;
}

inline void validate_permutation(const Grouping& grouping, size_t queries) {
    if (grouping.packed_to_original.size() != queries || grouping.original_to_packed.size() != queries)
        throw std::invalid_argument("Query permutation has the wrong size");
    for (size_t packed = 0; packed < queries; ++packed) {
        const auto original = grouping.packed_to_original[packed];
        if (original >= queries || grouping.original_to_packed[original] != packed)
            throw std::invalid_argument("Query permutation is not a bijection");
    }
}

// Query rows are already normalized and converted to BF16. Copy those exact
// values; never normalize/recompute the coarse choices after reordering.
template<class T>
std::vector<T> pack_query_rows(const std::vector<T>& rows, uint32_t width, const Grouping& grouping) {
    const size_t queries = grouping.packed_to_original.size();
    validate_permutation(grouping, queries);
    if (!width || rows.size() != ((queries + 31) / 32 * 32) * width)
        throw std::invalid_argument("Query row buffer has the wrong padded shape");
    std::vector<T> packed(rows.size(), T{});
    for (size_t q = 0; q < queries; ++q)
        std::copy_n(rows.begin() + size_t(grouping.packed_to_original[q]) * width,
                    width, packed.begin() + q * width);
    return packed;
}

// The existing worker scripts contain one bit per packed batch, per list.
// Valid queries alone determine these masks; the last batch's zero padding
// must not add arbitrary coarse lists to the union.
inline std::vector<uint32_t> make_batch_masks(
    const Selections& selections, const Grouping& grouping, uint32_t nlist) {
    validate_permutation(grouping, selections.size());
    const size_t batches = (selections.size() + 31) / 32, words = (batches + 31) / 32;
    std::vector<uint32_t> masks(size_t(nlist) * words, 0);
    for (size_t packed = 0; packed < selections.size(); ++packed) {
        const size_t batch = packed / 32;
        for (auto id : selections[grouping.packed_to_original[packed]]) {
            if (id >= nlist) throw std::invalid_argument("Coarse list ID outside index");
            masks[size_t(id) * words + batch / 32] |= uint32_t(1) << (batch % 32);
        }
    }
    return masks;
}

// Fine output is row-major top-32 after untilization. Return k entries per
// original query, retaining the existing sentinel filtering/padding behavior.
template<class Score>
std::pair<std::vector<float>, std::vector<int64_t>> restore_topk(
    const std::vector<Score>& scores, const std::vector<uint32_t>& ids,
    const Grouping& grouping, uint32_t k) {
    const size_t queries = grouping.packed_to_original.size();
    validate_permutation(grouping, queries);
    const size_t expected = ((queries + 31) / 32 * 32) * 32;
    if (!k || k > 32 || scores.size() != expected || ids.size() != expected)
        throw std::invalid_argument("Fine result buffer has the wrong padded shape or k");
    std::vector<float> original_scores(queries * k, -10000.0f);
    std::vector<int64_t> original_ids(queries * k, -1);
    for (size_t packed = 0; packed < queries; ++packed) {
        const size_t original = grouping.packed_to_original[packed];
        uint32_t valid = 0;
        for (uint32_t rank = 0; rank < k; ++rank) {
            const uint32_t id = ids[packed * 32 + rank];
            if (id == UINT32_MAX) continue;
            original_scores[original * k + valid] = static_cast<float>(scores[packed * 32 + rank]);
            original_ids[original * k + valid++] = id;
        }
    }
    return {std::move(original_scores), std::move(original_ids)};
}
} // namespace ivf_cluster_major
