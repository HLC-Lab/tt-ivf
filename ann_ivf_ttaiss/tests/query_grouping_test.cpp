// Portable integration checks for the host path used by search_all.
#include <array>
#include <cmath>
#include <iostream>
#include <functional>
#include <numeric>
#include <limits>
#include <random>
#include <set>
#include "../query_grouping.hpp"
#include "../coarse_selection.hpp"

using namespace ivf_cluster_major;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void must_reject(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Malformed grouping input was accepted");
}

void check_pipeline(uint32_t count, const std::string& mode, uint32_t window) {
    constexpr uint32_t nlist = 128, width = 128, k = 3;
    Selections selections(count);
    std::vector<uint32_t> pages(nlist);
    for (uint32_t id = 0; id < nlist; ++id) pages[id] = id % 7 ? id % 13 + 2 : 0;
    for (uint32_t q = 0; q < count; ++q)
        selections[q] = {(q * 17) % nlist, (q * 17 + 3) % nlist};
    const auto grouping = group_queries(selections, pages, mode, window, 256);
    const auto repeat = group_queries(selections, pages, mode, window, 256);
    require(grouping.packed_to_original == repeat.packed_to_original, "Grouping is not deterministic");
    require(grouping.grouped_pages <= grouping.original_pages, "Grouping increased page reads");
    validate_permutation(grouping, count);
    require(mode != "none" || is_identity(grouping), "Baseline reordered queries");
    if (window) for (uint32_t q = 0; q < count; ++q)
        require(q / window == grouping.packed_to_original[q] / window, "Query crossed its grouping window");

    // Reconstruct batch unions independently from the packed query mapping.
    // This covers the partial batch and the mask-word transition at batch 32.
    const uint32_t batches = (count + 31) / 32, words = (batches + 31) / 32;
    const auto masks = make_batch_masks(selections, grouping, nlist);
    require(masks.size() == nlist * words, "Batch mask shape is wrong");
    uint64_t actual_pages = 0;
    for (uint32_t batch = 0; batch < batches; ++batch) {
        std::set<uint32_t> expected;
        for (uint32_t q = batch * 32; q < std::min(count, (batch + 1) * 32); ++q)
            for (auto id : selections[grouping.packed_to_original[q]]) expected.insert(id);
        for (uint32_t id = 0; id < nlist; ++id) {
            const bool active = (masks[id * words + batch / 32] >> (batch % 32)) & 1;
            require(active == bool(expected.count(id)), "Fine script mask differs from the packed batch union");
            if (active) actual_pages += pages[id];
        }
    }
    require(actual_pages == grouping.grouped_pages, "Fine script page count differs from grouping cost");
    if (batches % 32) for (uint32_t id = 0; id < nlist; ++id)
        require((masks[id * words + words - 1] >> (batches % 32)) == 0, "Padded batch bit was activated");

    // Stand-ins for normalized BF16 bit patterns: retain exact row payloads,
    // zero the padded queries, and verify results return to the original rows.
    const uint32_t padded = batches * 32;
    std::vector<uint32_t> rows(size_t(padded) * width, UINT32_MAX);
    for (uint32_t q = 0; q < count; ++q) for (uint32_t d = 0; d < width; ++d)
        rows[size_t(q) * width + d] = q * width + d;
    const auto packed = pack_query_rows(rows, width, grouping);
    for (uint32_t q = 0; q < count; ++q) for (uint32_t d = 0; d < width; ++d)
        require(packed[size_t(q) * width + d] == rows[size_t(grouping.packed_to_original[q]) * width + d],
                "Query row changed during reordering");
    for (size_t i = size_t(count) * width; i < packed.size(); ++i)
        require(packed[i] == 0, "Padded query row was not zeroed");
    std::vector<float> scores(size_t(padded) * 32, 999.0f);
    std::vector<uint32_t> ids(size_t(padded) * 32, 123);
    for (uint32_t q = 0; q < count; ++q) {
        const auto original = packed[size_t(q) * width] / width;
        for (uint32_t rank = 0; rank < k; ++rank) {
            ids[q * 32 + rank] = rank == 1 ? UINT32_MAX : original * 100 + rank;
            scores[q * 32 + rank] = float(original * 100 + rank);
        }
    }
    const auto restored = restore_topk(scores, ids, grouping, k);
    require(restored.first.size() == count * k && restored.second.size() == count * k, "Wrong result shape");
    for (uint32_t q = 0; q < count; ++q) {
        require(restored.second[q * k] == q * 100 && restored.second[q * k + 1] == q * 100 + 2,
                "Result IDs were assigned to the wrong original query");
        require(restored.first[q * k] == float(q * 100) && restored.first[q * k + 1] == float(q * 100 + 2),
                "Result scores were assigned to the wrong original query");
        require(restored.second[q * k + 2] == -1 && restored.first[q * k + 2] == -10000.0f,
                "Invalid result padding changed");
    }
}

// The faster weighted-union implementation must reproduce the reference
// permutation exactly, including small lookahead budgets, windows, repeated
// selections, empty-page lists and partial batches.
void check_fast_matches_reference() {
    std::mt19937 rng(23);
    for (uint32_t trial = 0; trial < 120; ++trial) {
        const uint32_t nlist = 8 + rng() % 120, nprobe = 1 + rng() % std::min(8u, nlist), count = 1 + rng() % 700;
        std::vector<uint32_t> pages(nlist);
        for (auto& p : pages) p = rng() % 5 ? 1 + rng() % 9 : 0;
        Selections selections(count);
        for (auto& row : selections) {
            std::set<uint32_t> chosen;
            const uint32_t hot = rng() % 4;  // skew toward few lists to create ties and full coverage
            while (chosen.size() < nprobe) chosen.insert(hot && rng() % 2 ? rng() % std::min(nlist, nprobe + 3) : rng() % nlist);
            row.assign(chosen.begin(), chosen.end());
            std::shuffle(row.begin(), row.end(), rng);
        }
        const uint32_t window = (trial % 3 == 0) ? 32 * (1 + rng() % 4) : 0;
        const uint32_t lookahead = std::vector<uint32_t>{1, 4, 31, 256}[trial % 4];
        const auto slow = group_queries_reference(selections, pages, "weighted-union", window, lookahead);
        const auto fast = group_queries(selections, pages, "weighted-union", window, lookahead);
        require(slow.packed_to_original == fast.packed_to_original && slow.original_to_packed == fast.original_to_packed &&
                    slow.original_pages == fast.original_pages && slow.grouped_pages == fast.grouped_pages &&
                    slow.fell_back == fast.fell_back,
                "Fast weighted-union grouping differs from the reference");
    }
    must_reject([] { group_queries({{0, 0}}, {1}, "weighted-union", 0, 256); });
    must_reject([] { group_queries({{0, 5}}, {1}, "weighted-union", 0, 256); });
    must_reject([] { group_queries({{}}, {1}, "weighted-union", 0, 256); });
}

int main() {
    try {
        check_fast_matches_reference();
        for (auto count : {1u, 16u, 32u, 33u, 64u, 65u, 1025u, 10000u})
            for (const std::string mode : {"none", "primary-list", "weighted-union"})
                for (auto window : {0u, 64u}) check_pipeline(count, mode, window);

        Selections interleaved(96);
        for (uint32_t q = 0; q < interleaved.size(); ++q) interleaved[q] = {q % 3};
        const auto reduced = group_queries(interleaved, {2, 10, 30}, "weighted-union", 0, 256);
        require(reduced.original_pages == 126 && reduced.grouped_pages == 42,
                "Grouping did not reduce repeated scans in an interleaved workload");

        // Find an adversarial primary-list sort: the fallback must retain the
        // identity order instead of publishing a more expensive permutation.
        std::mt19937 rng(17);
        bool fallback_seen = false;
        for (uint32_t trial = 0; trial < 256 && !fallback_seen; ++trial) {
            Selections selections(65);
            for (auto& row : selections) {
                uint32_t a = rng() % 64, b = (a + 1 + rng() % 63) % 64;
                row = {a, b};
            }
            const auto group = group_queries(selections, std::vector<uint32_t>(64, 1), "primary-list", 0, 256);
            if (group.fell_back) {
                fallback_seen = true;
                require(is_identity(group) && group.grouped_pages == group.original_pages, "Fallback broke identity");
            }
        }
        require(fallback_seen, "Fallback fixture did not exercise a regressing grouping");

        std::array<uint32_t, 32> ids;
        std::array<float, 32> scores;
        for (uint32_t rank = 0; rank < 32; ++rank) { ids[rank] = rank; scores[rank] = -float(rank / 2); }
        for (uint32_t nprobe : {1u, 16u, 32u}) {
            const auto row = decode_coarse_row(ids, scores, 512, nprobe);
            require(row.complete() && row.selected.size() == nprobe, "Valid device coarse result was rejected");
            for (uint32_t rank = 0; rank < nprobe; ++rank)
                require(row.selected[rank] == ids[rank], "Equal-score device rank order was changed");
        }
        ids[31] = ids[0];
        require(!decode_coarse_row(ids, scores, 512, 16).complete(), "Corrupt coarse suffix was accepted");
        ids[31] = UINT32_MAX;
        require(!decode_coarse_row(ids, scores, 512, 16).complete(), "Invalid coarse ID was accepted");
        ids[31] = 31; scores[31] = std::numeric_limits<float>::quiet_NaN();
        require(!decode_coarse_row(ids, scores, 512, 16).complete(), "Nonfinite coarse score was accepted");
        // Top-64 rows (nprobe > 32): all 64 lanes are validated; nprobe best kept.
        std::array<uint32_t, 64> ids64;
        std::array<float, 64> scores64;
        for (uint32_t rank = 0; rank < 64; ++rank) { ids64[rank] = 100 + rank; scores64[rank] = float(rank % 7); }
        const auto row64 = decode_coarse_row(ids64, scores64, 512, 64);
        require(row64.complete() && row64.selected.size() == 64 && row64.expected == 64, "Valid top-64 row was rejected");
        for (uint32_t rank = 1; rank < 64; ++rank)
            require(scores64[row64.selected[rank - 1] - 100] >= scores64[row64.selected[rank] - 100], "Top-64 not ordered");
        require(decode_coarse_row(ids64, scores64, 512, 40).selected.size() == 40, "Top-64 row must keep nprobe lists");
        ids64[63] = ids64[0];
        require(!decode_coarse_row(ids64, scores64, 512, 40).complete(), "Duplicate in the second top-32 half was accepted");
        must_reject([&] { decode_coarse_row(ids64, scores64, 32, 40); });
        // The allocation-free decode matches decode_coarse_row on random rows:
        // ordered, unordered with ties, duplicates, invalid IDs and NaNs.
        {
            std::mt19937 fast_rng(77);
            std::vector<uint32_t> stamp(512, 0);
            uint32_t token = 0;
            auto compare = [&]<size_t K>(std::array<uint32_t, K>& row_ids, std::array<float, K>& row_scores) {
                for (uint32_t nprobe : {1u, 7u, uint32_t(K)}) {
                    const auto reference = decode_coarse_row(row_ids, row_scores, 512, nprobe);
                    std::array<uint32_t, K> selected{};
                    const bool ok = decode_coarse_row_fast(row_ids, row_scores, 512, nprobe, selected.data(), stamp, ++token);
                    require(ok == reference.complete(), "Fast coarse decode disagrees on validity");
                    if (ok)
                        for (uint32_t rank = 0; rank < nprobe; ++rank)
                            require(selected[rank] == reference.selected[rank], "Fast coarse decode selected other lists");
                }
            };
            for (uint32_t trial = 0; trial < 4000; ++trial) {
                auto fill = [&]<size_t K>(std::array<uint32_t, K>& row_ids, std::array<float, K>& row_scores) {
                    std::vector<uint32_t> pool(512);
                    std::iota(pool.begin(), pool.end(), 0u);
                    std::shuffle(pool.begin(), pool.end(), fast_rng);
                    for (uint32_t rank = 0; rank < K; ++rank) {
                        row_ids[rank] = pool[rank];
                        row_scores[rank] = float(fast_rng() % 9) / 4.0f;
                    }
                    if (trial % 2) std::sort(row_scores.begin(), row_scores.end(), std::greater<float>());
                    switch (trial % 7) {
                        case 3: row_ids[fast_rng() % K] = row_ids[fast_rng() % K]; break;
                        case 4: row_ids[fast_rng() % K] = 512 + fast_rng() % 4; break;
                        case 5: row_scores[fast_rng() % K] = std::numeric_limits<float>::quiet_NaN(); break;
                        default: break;
                    }
                };
                std::array<uint32_t, 32> row_ids32; std::array<float, 32> row_scores32;
                fill(row_ids32, row_scores32);
                compare(row_ids32, row_scores32);
                std::array<uint32_t, 64> row_ids64; std::array<float, 64> row_scores64;
                fill(row_ids64, row_scores64);
                compare(row_ids64, row_scores64);
            }
            std::array<uint32_t, 32> one{};
            must_reject([&] { decode_coarse_row_fast(ids, scores, 512, 16, one.data(), stamp, 0); });  // token 0
        }
        require(coarse_k(32) == 32 && coarse_k(33) == 64 && coarse_k(64) == 64, "Coarse top-K selection is wrong");
        must_reject([] { group_queries({{0, 0}}, {1}, "weighted-union", 0, 256); });
        must_reject([] { group_queries({{}}, {1}, "weighted-union", 0, 256); });
        must_reject([] { validate_grouping_config({"bad", 0, 256}); });
        must_reject([] { validate_grouping_config({"none", 33, 256}); });
        must_reject([] { validate_grouping_config({"none", 0, 0}); });
        auto broken = reduced;
        broken.packed_to_original[0] = broken.packed_to_original[1];
        must_reject([&] { make_batch_masks(interleaved, broken, 3); });
        must_reject([&] { pack_query_rows(std::vector<uint16_t>(1), 128, reduced); });
        must_reject([&] { restore_topk(std::vector<float>(1), std::vector<uint32_t>(1), reduced, 10); });
        std::cout << "Grouping, coarse selection, packed batch masks, exact query rows and original-order results passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
