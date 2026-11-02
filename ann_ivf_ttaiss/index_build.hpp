#pragma once
// Host index-build kernels for ann_ivf_ttaiss. The "fast" versions
// produce bit-identical results to the original single-threaded/scalar code
// ("reference"): every per-row and per-score floating-point operation keeps
// its order, and only independent rows, centroids or lists run in parallel.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ivf_index_build {

inline unsigned worker_threads() {
    const unsigned threads = std::thread::hardware_concurrency();
    return threads ? threads : 4;
}

// Calls body(begin, end) on contiguous chunks of [0, count) in parallel.
template <class Body>
void parallel_chunks(size_t count, unsigned threads, Body body) {
    threads = std::max(1u, std::min<unsigned>(threads, count ? unsigned(std::min<size_t>(count, threads)) : 1u));
    const size_t chunk = (count + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        const size_t begin = size_t(t) * chunk, end = std::min(count, begin + chunk);
        if (begin < end) pool.emplace_back(body, begin, end);
    }
    for (auto& thread : pool) thread.join();
}

// Calls body(i) for i in [0, count) with dynamic load balancing (uneven work).
template <class Body>
void parallel_items(size_t count, unsigned threads, Body body) {
    std::atomic<size_t> next{0};
    parallel_chunks(threads, threads, [&](size_t, size_t) {
        for (size_t i = next++; i < count; i = next++) body(i);
    });
}

// Same arithmetic as the original normalize_l2 lambda, row by row.
inline void normalize_row(float* row, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) sum += row[d] * row[d];
    const float norm = std::sqrt(sum);
    if (norm > 0) {
        for (uint32_t d = 0; d < dim; ++d) row[d] /= norm;
    }
}

inline void normalize_rows(float* data, size_t rows, uint32_t dim, unsigned threads) {
    parallel_chunks(rows, threads, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) normalize_row(data + i * dim, dim);
    });
}

// Original assignment: per vector, scalar dot product with every centroid in
// dimension order; the first centroid with the strictly largest score wins.
inline std::vector<uint32_t> assign_reference(const float* data, size_t rows, const float* centroids,
                                              uint32_t nlist, uint32_t dim, unsigned threads) {
    std::vector<uint32_t> best(rows);
    parallel_chunks(rows, threads, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            float max_score = -1e20f;
            uint32_t best_cluster = 0;
            for (uint32_t c = 0; c < nlist; ++c) {
                float score = 0;
                for (uint32_t d = 0; d < dim; ++d) score += data[i * dim + d] * centroids[c * dim + d];
                if (score > max_score) { max_score = score; best_cluster = c; }
            }
            best[i] = best_cluster;
        }
    });
    return best;
}

// Same scores, vectorized across centroids: each score still accumulates its
// dimensions in order (score[c] += x[d] * centroid[c][d] for d = 0, 1, ...),
// so every FP32 score and the strict first-maximum choice are unchanged.
inline std::vector<uint32_t> assign_fast(const float* data, size_t rows, const float* centroids,
                                         uint32_t nlist, uint32_t dim, unsigned threads) {
    std::vector<float> transposed(size_t(dim) * nlist);
    for (uint32_t c = 0; c < nlist; ++c)
        for (uint32_t d = 0; d < dim; ++d) transposed[size_t(d) * nlist + c] = centroids[size_t(c) * dim + d];
    std::vector<uint32_t> best(rows);
    parallel_chunks(rows, threads, [&](size_t begin, size_t end) {
        std::vector<float> scores(nlist);
        float* __restrict s = scores.data();
        const float* __restrict ct = transposed.data();
        for (size_t i = begin; i < end; ++i) {
            const float* x = data + i * dim;
            std::fill(scores.begin(), scores.end(), 0.0f);
            for (uint32_t d = 0; d < dim; ++d) {
                const float xd = x[d];
                const float* __restrict column = ct + size_t(d) * nlist;
                for (uint32_t c = 0; c < nlist; ++c) s[c] += xd * column[c];
            }
            float max_score = -1e20f;
            uint32_t best_cluster = 0;
            for (uint32_t c = 0; c < nlist; ++c)
                if (s[c] > max_score) { max_score = s[c]; best_cluster = c; }
            best[i] = best_cluster;
        }
    });
    return best;
}

// FNV-1a over raw bytes, for comparing index builds.
inline uint64_t fnv1a(const void* bytes, size_t size, uint64_t hash = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(bytes);
    for (size_t i = 0; i < size; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}

}  // namespace ivf_index_build
