// The fast index-build kernels must reproduce the reference bit for bit.
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "../index_build.hpp"

using namespace ivf_index_build;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main() {
    try {
        std::mt19937 rng(11);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (uint32_t dim : {1u, 7u, 100u, 128u})
            for (uint32_t nlist : {1u, 3u, 32u, 512u})
                for (unsigned threads : {1u, 5u}) {
                    const size_t rows = 997;
                    std::vector<float> data(rows * dim), centroids(size_t(nlist) * dim);
                    for (auto& value : data) value = normal(rng);
                    for (auto& value : centroids) value = normal(rng);
                    // Duplicate centroids force exact score ties: the first wins.
                    if (nlist > 2) std::copy_n(centroids.begin(), dim, centroids.begin() + 2 * dim);
                    // A zero row keeps its zeros (norm 0 is not divided).
                    std::fill_n(data.begin(), dim, 0.0f);
                    auto reference = data;
                    for (size_t i = 0; i < rows; ++i) normalize_row(reference.data() + i * dim, dim);
                    normalize_rows(data.data(), rows, dim, threads);
                    require(std::memcmp(reference.data(), data.data(), data.size() * sizeof(float)) == 0,
                            "Parallel normalization changed a value");
                    normalize_rows(centroids.data(), nlist, dim, 1);
                    // Make vector 1 equal to centroid 0 so its best score ties centroid 2.
                    if (nlist > 2) std::copy_n(centroids.begin(), dim, data.begin() + dim);
                    const auto slow = assign_reference(data.data(), rows, centroids.data(), nlist, dim, threads);
                    const auto fast = assign_fast(data.data(), rows, centroids.data(), nlist, dim, threads);
                    require(slow == fast, "Vectorized assignment differs from the reference");
                    if (nlist > 2) require(fast[1] == 0, "Tied centroids must keep the first maximum");
                }
        std::vector<int> touched(1000, 0);
        parallel_items(touched.size(), 7, [&](size_t i) { ++touched[i]; });
        for (int count : touched) require(count == 1, "parallel_items must visit every item once");
        require(fnv1a("a", 1) != fnv1a("b", 1), "Digest must depend on content");
        std::cout << "Index-build normalization and assignment match the reference exactly\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
