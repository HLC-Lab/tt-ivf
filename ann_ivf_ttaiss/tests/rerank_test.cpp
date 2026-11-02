// FP32 re-ranking of the device top-32 must equal a brute-force stable sort.
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "../rerank.hpp"

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main() {
    try {
        std::mt19937 rng(3);
        std::normal_distribution<float> normal;
        const uint32_t dim = 100, stride = 128, queries = 257, vectors = 5000, per_query = 32;
        std::vector<float> data(size_t(vectors) * dim), rows(size_t(queries) * stride, 0.0f);
        for (auto& v : data) v = normal(rng);
        for (uint32_t q = 0; q < queries; ++q)
            for (uint32_t d = 0; d < dim; ++d) rows[size_t(q) * stride + d] = normal(rng);
        // Vector 7 duplicates vector 3: exact ties must keep the device order.
        std::copy_n(data.begin() + 3 * dim, dim, data.begin() + 7 * dim);
        std::vector<int64_t> candidates(size_t(queries) * per_query);
        for (uint32_t q = 0; q < queries; ++q)
            for (uint32_t r = 0; r < per_query; ++r)
                candidates[q * per_query + r] = (q % 5 == 0 && r >= 6) ? -1 : int64_t(rng() % vectors);
        candidates[0] = 7; candidates[1] = 3;  // query 0: tie at device ranks 0 and 1
        for (uint32_t k : {1u, 10u, 32u})
            for (unsigned threads : {1u, 7u}) {
                const auto [scores, ids] = ivf_rerank::rerank(rows.data(), stride, queries, dim, data.data(), vectors,
                                                              candidates, per_query, k, threads);
                for (uint32_t q = 0; q < queries; ++q) {
                    std::vector<std::pair<float, uint32_t>> expected;
                    for (uint32_t r = 0; r < per_query; ++r) {
                        const int64_t id = candidates[q * per_query + r];
                        if (id < 0) continue;
                        float s = 0.0f;
                        for (uint32_t d = 0; d < dim; ++d) s += rows[size_t(q) * stride + d] * data[size_t(id) * dim + d];
                        expected.emplace_back(s, r);
                    }
                    std::stable_sort(expected.begin(), expected.end(), [](auto& a, auto& b) { return a.first > b.first; });
                    for (uint32_t i = 0; i < k; ++i) {
                        if (i < expected.size()) {
                            require(scores[q * k + i] == expected[i].first &&
                                    ids[q * k + i] == candidates[q * per_query + expected[i].second],
                                    "Re-ranked order differs from the brute-force stable sort");
                        } else {
                            require(scores[q * k + i] == -10000.0f && ids[q * k + i] == -1, "Missing result not padded");
                        }
                    }
                }
                if (k >= 2) {  // tie: device rank 0 (id 7) stays ahead of rank 1 (id 3)
                    const auto first = std::find(ids.begin(), ids.begin() + k, 7), second = std::find(ids.begin(), ids.begin() + k, 3);
                    require(first < second || second == ids.begin() + k, "Equal scores did not keep the device order");
                }
            }
        bool rejected = false;
        try { ivf_rerank::rerank(rows.data(), stride, queries, dim, data.data(), vectors, candidates, per_query, 33, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "k above the candidate count was accepted");
        std::cout << "FP32 re-ranking matches a brute-force stable sort\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
