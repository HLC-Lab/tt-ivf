#include "../coarse_selection.hpp"
#include <iostream>
#include <limits>

using namespace ivf_partitioned;

int main() {
    auto require = [](bool ok) { if (!ok) throw std::runtime_error("Coarse row validation failed"); };
    std::array<uint32_t, 32> ids;
    std::array<float, 32> scores;
    for (uint32_t i = 0; i < 32; ++i) { ids[i] = i; scores[i] = float(i) / 32; }
    for (uint32_t nprobe : {1u, 8u, 16u, 32u}) {
        auto row = decode_coarse_row(ids, scores, 512, nprobe);
        require(row.complete() && row.selected.size() == nprobe);
        for (uint32_t i = 0; i < nprobe; ++i) require(row.selected[i] == 31 - i);
        auto repeated = ids;
        repeated[31] = repeated[30];
        row = decode_coarse_row(repeated, scores, 512, nprobe);
        require(!row.complete() && row.distinct == 31 && row.duplicate_ids == 1 && row.selected.empty());
    }
    // BF16 ties are valid: equal scores must not imply duplicate list IDs.
    scores.fill(0);
    auto row = decode_coarse_row(ids, scores, 32, 32);
    require(row.complete());
    for (uint32_t i = 0; i < 32; ++i) require(row.selected[i] == ids[i]);
    ids[31] = UINT32_MAX;
    row = decode_coarse_row(ids, scores, 512, 32);
    require(!row.complete() && row.invalid_ids == 1 && row.distinct == 31);
    ids[31] = 31;
    scores[31] = std::numeric_limits<float>::quiet_NaN();
    row = decode_coarse_row(ids, scores, 512, 32);
    require(!row.complete() && row.nonfinite_scores == 1 && row.distinct == 31);
    require(row.description().find("nonfinite_scores=1") != std::string::npos);
    std::cout << "Coarse top-32 validation accepts score ties and rejects corrupt suffixes at every N_probe\n";
}
