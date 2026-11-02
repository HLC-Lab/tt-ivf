#pragma once
// Host grouping/selection adapted from ann_ivf_similarity_partitioned.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ivf_cluster_major {

struct CoarseRow {
    std::vector<uint32_t> selected;
    uint32_t expected = 32, distinct = 0, invalid_ids = 0, nonfinite_scores = 0, duplicate_ids = 0;
    bool complete() const { return distinct == expected && !invalid_ids && !nonfinite_scores && !duplicate_ids; }
    std::string description() const {
        std::ostringstream message;
        message << "distinct=" << distinct << "/" << expected << ", invalid_ids=" << invalid_ids
                << ", nonfinite_scores=" << nonfinite_scores << ", duplicate_ids=" << duplicate_ids;
        return message.str();
    }
};

// The device always produces its full top-K (K = 32, or 64 when nprobe > 32),
// including when N_probe is smaller. All K lanes must be valid and distinct;
// accepting a corrupt suffix at a small N_probe would hide the same problem
// that becomes visible at N_probe = K.
template <size_t K>
inline CoarseRow decode_coarse_row(const std::array<uint32_t, K>& ids,
                                   const std::array<float, K>& scores,
                                   uint32_t nlist, uint32_t nprobe) {
    static_assert(K == 32 || K == 64, "The coarse kernel returns top-32 or top-64");
    if (nlist < K || !nprobe || nprobe > K) throw std::invalid_argument("Invalid coarse selection configuration");
    CoarseRow row;
    row.expected = K;
    std::array<uint32_t, K> seen{};
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(K);
    for (uint32_t rank = 0; rank < K; ++rank) {
        const bool valid_id = ids[rank] < nlist, finite = std::isfinite(scores[rank]);
        row.invalid_ids += !valid_id;
        row.nonfinite_scores += !finite;
        if (!valid_id || !finite) continue;
        if (std::find(seen.begin(), seen.begin() + row.distinct, ids[rank]) != seen.begin() + row.distinct) {
            ++row.duplicate_ids;
        } else {
            seen[row.distinct++] = ids[rank];
            candidates.emplace_back(scores[rank], ids[rank]);
        }
    }
    if (!row.complete()) return row;
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    row.selected.reserve(nprobe);
    for (uint32_t rank = 0; rank < nprobe; ++rank) row.selected.push_back(candidates[rank].second);
    return row;
}

// Allocation-free decode with the same result as decode_coarse_row: returns
// true exactly when that row is complete, and then writes the same nprobe
// selected IDs. stamp (nlist entries, reused across rows) marks IDs seen in
// this row with `token`, which must differ from every earlier row's token.
// Rows already in non-increasing score order (the device sort) need no sort.
template <size_t K>
inline bool decode_coarse_row_fast(const std::array<uint32_t, K>& ids, const std::array<float, K>& scores,
                                   uint32_t nlist, uint32_t nprobe, uint32_t* selected,
                                   std::vector<uint32_t>& stamp, uint32_t token) {
    static_assert(K == 32 || K == 64, "The coarse kernel returns top-32 or top-64");
    if (nlist < K || !nprobe || nprobe > K || stamp.size() < nlist || !token)
        throw std::invalid_argument("Invalid coarse selection configuration");
    bool ordered = true;
    for (uint32_t rank = 0; rank < K; ++rank) {
        if (ids[rank] >= nlist || !std::isfinite(scores[rank]) || stamp[ids[rank]] == token) return false;
        stamp[ids[rank]] = token;
        ordered = ordered && (rank == 0 || !(scores[rank] > scores[rank - 1]));
    }
    if (ordered) {
        std::copy_n(ids.begin(), nprobe, selected);
        return true;
    }
    std::array<uint32_t, K> order;
    for (uint32_t rank = 0; rank < K; ++rank) order[rank] = rank;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return scores[a] > scores[b]; });
    for (uint32_t rank = 0; rank < nprobe; ++rank) selected[rank] = ids[order[rank]];
    return true;
}

// Device coarse top-K needed for nprobe (top-64 needs nlist >= 64).
inline uint32_t coarse_k(uint32_t nprobe) { return nprobe > 32 ? 64 : 32; }

}  // namespace ivf_cluster_major
