#pragma once
// Host FP32 re-ranking of the device top-32 candidates (--rerank). The device
// ranks with BF16 scores, which tie or swap neighbours that differ by less
// than one BF16 step; re-scoring the 32 returned candidates of every query
// with the L2-normalized FP32 vectors and keeping the best k restores most of
// that precision. Candidates outside the device top-32 are not considered.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include "index_build.hpp"

namespace ivf_rerank {

// queries: row q at queries + q * query_stride (first `dim` values used).
// candidates: queries x candidates_per_query IDs in device order, -1 = none.
// Returns k (score, id) per query, FP32 score descending; equal scores keep
// the device order; missing results are (-10000, -1) as in restore_topk.
inline std::pair<std::vector<float>, std::vector<int64_t>> rerank(
    const float* queries, size_t query_stride, size_t num_queries, uint32_t dim,
    const float* vectors, size_t num_vectors, const std::vector<int64_t>& candidates,
    uint32_t candidates_per_query, uint32_t k, unsigned threads) {
    if (!dim || !k || k > candidates_per_query || query_stride < dim ||
        candidates.size() != num_queries * candidates_per_query)
        throw std::invalid_argument("Re-ranking received inconsistent shapes");
    std::vector<float> scores(num_queries * k, -10000.0f);
    std::vector<int64_t> ids(num_queries * k, -1);
    ivf_index_build::parallel_chunks(num_queries, threads, [&](size_t begin, size_t end) {
        std::vector<std::pair<float, uint32_t>> scored;  // (score, device rank)
        for (size_t q = begin; q < end; ++q) {
            const float* query = queries + q * query_stride;
            scored.clear();
            for (uint32_t rank = 0; rank < candidates_per_query; ++rank) {
                const int64_t id = candidates[q * candidates_per_query + rank];
                if (id < 0) continue;
                if (uint64_t(id) >= num_vectors) throw std::runtime_error("Re-ranking candidate outside the dataset");
                const float* vector = vectors + size_t(id) * dim;
                float score = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) score += query[d] * vector[d];
                scored.emplace_back(score, rank);
            }
            std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t i = 0; i < std::min<size_t>(k, scored.size()); ++i) {
                scores[q * k + i] = scored[i].first;
                ids[q * k + i] = candidates[q * candidates_per_query + scored[i].second];
            }
        }
    });
    return {std::move(scores), std::move(ids)};
}

}  // namespace ivf_rerank
