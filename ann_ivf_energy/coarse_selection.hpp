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

namespace ivf_energy {

struct CoarseRow {
    std::vector<uint32_t> selected;
    uint32_t distinct = 0, invalid_ids = 0, nonfinite_scores = 0, duplicate_ids = 0;
    bool complete() const { return distinct == 32 && !invalid_ids && !nonfinite_scores && !duplicate_ids; }
    std::string description() const {
        std::ostringstream message;
        message << "distinct=" << distinct << "/32, invalid_ids=" << invalid_ids
                << ", nonfinite_scores=" << nonfinite_scores << ", duplicate_ids=" << duplicate_ids;
        return message.str();
    }
};

// The device always produces top-32, including when N_probe is smaller. All
// 32 lanes must be valid and distinct; accepting a corrupt suffix at N_probe=16
// would hide the same problem that becomes visible at N_probe=32.
inline CoarseRow decode_coarse_row(const std::array<uint32_t, 32>& ids,
                                   const std::array<float, 32>& scores,
                                   uint32_t nlist, uint32_t nprobe) {
    if (nlist < 32 || !nprobe || nprobe > 32) throw std::invalid_argument("Invalid coarse selection configuration");
    CoarseRow row;
    std::array<uint32_t, 32> seen{};
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(32);
    for (uint32_t rank = 0; rank < 32; ++rank) {
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

}  // namespace ivf_energy
