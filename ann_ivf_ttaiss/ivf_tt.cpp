// ann_ivf_ttaiss/ivf_tt.cpp
#include <thread>

#include "ivf_tt.hpp"
#include "cluster_schedule.hpp"
#include "index_build.hpp"
#include "rerank.hpp"
#include "gather_schedule.hpp"
#include "fine_tile_layout.hpp"
#include <numeric>
#include "coarse_selection.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tilize_utils.hpp>

// cluster_schedule.hpp stores fidelities as these MathFidelity values.
static_assert(static_cast<uint32_t>(MathFidelity::LoFi) == 0 && static_cast<uint32_t>(MathFidelity::HiFi2) == 2 &&
              static_cast<uint32_t>(MathFidelity::HiFi3) == 3 && static_cast<uint32_t>(MathFidelity::HiFi4) == 4);

inline uint16_t float_to_bfloat16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return (uint16_t)(i >> 16);
}

namespace {

template <typename Clock>
double elapsed_us(
    const std::chrono::time_point<Clock>& start, const std::chrono::time_point<Clock>& end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

}  // namespace

IVF_TT::IVF_TT(
    int device_id [[maybe_unused]],
    uint32_t dim,
    uint32_t nlist,
    uint32_t nprobe,
    const std::string& memory_log_path,
    uint32_t prefetch_reader,
    uint32_t batch_chunk,
    ivf_cluster_major::GroupingConfig query_grouping,
    RunOptions options) :
    dim_(dim),
    nlist_(nlist),
    nprobe_(nprobe),
    prefetch_reader_(prefetch_reader),
    batch_chunk_(batch_chunk),
    options_(options),
    query_grouping_(std::move(query_grouping)),
    memory_logger_(std::make_unique<MemoryTransferLogger>(memory_log_path)) {
    ivf_cluster_major::validate_grouping_config(query_grouping_);
    if (dim_ == 0) {
        throw std::invalid_argument("dim must be greater than zero");
    }
    if (nlist_ < 32 || nlist_ % 32 != 0) {
        throw std::invalid_argument("nlist must be at least 32 and divisible by 32");
    }
    if (nprobe_ == 0 || nprobe_ > 64 || (nprobe_ > 32 && nlist_ < 64)) {
        throw std::invalid_argument("nprobe must be 1..64 (coarse top-64 above 32 needs nlist >= 64)");
    }
    if (prefetch_reader_ == 0) {
        throw std::invalid_argument("--prefetch-reader must be greater than zero");
    }
    if (batch_chunk_ == 0) {
        throw std::invalid_argument("--batch-chunk must be greater than zero");
    }
    mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    const auto* device = mesh_device_->get_device(0);
    const uint64_t l1_start = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint64_t l1_end = device->l1_size_per_core();
    if (l1_start >= l1_end) {
        throw std::runtime_error("No allocatable worker L1 is available for fine-search circular buffers");
    }
    usable_l1_bytes_ = l1_end - l1_start;
    // The script page is sized per search; check the fixed part here so an
    // oversized chunk/prefetch is rejected before indexing.
    const uint64_t fixed_l1 = ivf_cluster_major::scan_l1::worker_cb_bytes(dim_, prefetch_reader_, batch_chunk_, 64);
    if (fixed_l1 > usable_l1_bytes_) {
        throw std::invalid_argument(
            "--batch-chunk=" + std::to_string(batch_chunk_) + " with --prefetch-reader=" +
            std::to_string(prefetch_reader_) + " needs " + std::to_string(fixed_l1 / 1024) +
            " KiB of worker L1; usable worker L1 is " + std::to_string(usable_l1_bytes_ / 1024) + " KiB");
    }
    std::cout << "[Host] Cluster-major scan: batch chunk=" << batch_chunk_
              << ", prefetch=" << prefetch_reader_ << " pages; worker CBs without script="
              << fixed_l1 / 1024 << " KiB; usable worker L1=" << usable_l1_bytes_ / 1024 << " KiB; fine/coarse fidelity="
              << ivf_cluster_major::fidelity_name(options_.fine_fidelity) << "/"
              << ivf_cluster_major::fidelity_name(options_.coarse_fidelity) << std::endl;
    auto grid = mesh_device_->compute_with_storage_grid_size();
    all_cores_ = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
}

IVF_TT::~IVF_TT() {}

void IVF_TT::set_memory_log_context(const std::string& context) {
    memory_logger_->set_context(context);
}

void IVF_TT::flush_memory_log() {
    memory_logger_->flush();
}

void IVF_TT::create_index_reference(const std::vector<float>& dataset_vectors, const std::string& centroids_path) {
    if (dataset_vectors.empty() || dataset_vectors.size() % dim_ != 0) {
        throw std::invalid_argument("Dataset must contain a non-zero whole number of vectors");
    }

    size_t num_vectors = dataset_vectors.size() / dim_;
    std::ifstream file(centroids_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open centroids file: " + centroids_path);

    uint32_t header[2];
    file.read(reinterpret_cast<char*>(header), 8);
    uint32_t n_centroids = header[0];
    uint32_t d_centroids = header[1];

    if (n_centroids != nlist_) {
        throw std::runtime_error(
            "Centroid count mismatch: file has " + std::to_string(n_centroids) +
            ", expected " + std::to_string(nlist_));
    }
    if (d_centroids != dim_) {
        throw std::runtime_error("Centroid dimension mismatch!");
    }

    centroids_host_.resize(n_centroids * dim_);
    file.read(reinterpret_cast<char*>(centroids_host_.data()), n_centroids * dim_ * sizeof(float));

    auto normalize_l2 = [](float* data, uint32_t num_vectors, uint32_t dim) {
        for (uint32_t i = 0; i < num_vectors; ++i) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                sum += data[i * dim + d] * data[i * dim + d];
            }
            float norm = std::sqrt(sum);
            if (norm > 0) {
                for (uint32_t d = 0; d < dim; ++d) {
                    data[i * dim + d] /= norm;
                }
            }
        }
    };

    normalize_l2(centroids_host_.data(), nlist_, dim_);

    std::vector<float> dataset_norm = dataset_vectors;
    num_vectors = dataset_vectors.size() / dim_;
    normalize_l2(dataset_norm.data(), num_vectors, dim_);
    if (options_.rerank) {  // re-ranking input only
        rerank_vectors_.reset(new float[dataset_norm.size()]);
        std::copy(dataset_norm.begin(), dataset_norm.end(), rerank_vectors_.get());
        rerank_vector_count_ = num_vectors;
    }


    std::cout << "[IVF_TT] Assigning " << num_vectors << " vectors to " << nlist_ << " clusters (Multi-threaded)..." << std::endl;
    std::vector<std::vector<uint32_t>> cluster_assignments(nlist_);

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<uint32_t> best_clusters(num_vectors);

    auto worker = [&](uint32_t start_idx, uint32_t end_idx) {
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            float max_score = -1e20f;
            uint32_t best_cluster = 0;
            for (uint32_t c = 0; c < nlist_; ++c) {
                float score = 0;
                for (uint32_t d = 0; d < dim_; ++d) {
                    score += dataset_norm[i * dim_ + d] * centroids_host_[c * dim_ + d];
                }
                if (score > max_score) { max_score = score; best_cluster = c; }
            }
            best_clusters[i] = best_cluster;
        }
    };

    uint32_t chunk_size = (num_vectors + num_threads - 1) / num_threads;
    for (uint32_t t = 0; t < num_threads; ++t) {
        uint32_t start_idx = t * chunk_size;
        uint32_t end_idx = std::min(start_idx + chunk_size, static_cast<uint32_t>(num_vectors));
        if (start_idx < end_idx) {
            threads.emplace_back(worker, start_idx, end_idx);
        }
    }

    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    for (uint32_t i = 0; i < num_vectors; ++i) {
        cluster_assignments[best_clusters[i]].push_back(i);
    }
    vector_lists_ = best_clusters;  // validation export only

    uint32_t padded_dim = (dim_ + 31) / 32 * 32;
    uint32_t current_page_offset = 0;
    dataset_host_padded_.clear();
    index_map_host_padded_.clear();



    for (uint32_t i = 0; i < nlist_; ++i) {
        uint32_t count = cluster_assignments[i].size();
        uint32_t cluster_blocks = (count + 31) / 32;
        if (cluster_blocks < 1) cluster_blocks = 1;
        uint32_t padded_count = cluster_blocks * 32;

        cluster_toc_[i] = {current_page_offset, count, padded_count, current_page_offset};

        std::vector<bfloat16> cluster_flat;
        for (uint32_t d = 0; d < padded_dim; ++d) {
            for (uint32_t v = 0; v < padded_count; ++v) {
                if (v < count) {
                    uint32_t idx = cluster_assignments[i][v];
                    if (d < dim_) cluster_flat.push_back(bfloat16(dataset_norm[idx * dim_ + d]));
                    else cluster_flat.push_back(bfloat16(0.0f));
                } else cluster_flat.push_back(bfloat16(0.0f));
            }
        }
        std::vector<bfloat16> cluster_tilized = tilize_nfaces(cluster_flat, padded_dim, padded_count);

        uint32_t Ht = padded_dim / 32;
        uint32_t Wt = cluster_blocks;
        std::vector<bfloat16> cluster_tilized_col_major(Ht * Wt * 1024);
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                std::copy(
                    cluster_tilized.begin() + (h * Wt + w) * 1024,
                    cluster_tilized.begin() + (h * Wt + w + 1) * 1024,
                    cluster_tilized_col_major.begin() + (w * Ht + h) * 1024
                );
            }
        }
        dataset_host_padded_.insert(dataset_host_padded_.end(), cluster_tilized_col_major.begin(), cluster_tilized_col_major.end());

        std::vector<uint32_t> index_flat;
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t v = 0; v < padded_count; ++v) {
                uint32_t idx_val = (v < count) ? static_cast<uint32_t>(cluster_assignments[i][v]) : 0xFFFFFFFF;
                index_flat.push_back(idx_val);
            }
        }
        std::vector<uint32_t> index_tilized = tilize_nfaces(index_flat, 32, padded_count);
        index_map_host_padded_.insert(index_map_host_padded_.end(), index_tilized.begin(), index_tilized.end());

        current_page_offset += cluster_blocks;
    }
}


// Reads the centroid file and checks its [N][D] header (shared by both builds).
std::vector<float> IVF_TT::read_centroids(const std::string& centroids_path) const {
    std::ifstream file(centroids_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open centroids file: " + centroids_path);
    uint32_t header[2];
    file.read(reinterpret_cast<char*>(header), 8);
    if (header[0] != nlist_) {
        throw std::runtime_error(
            "Centroid count mismatch: file has " + std::to_string(header[0]) + ", expected " + std::to_string(nlist_));
    }
    if (header[1] != dim_) throw std::runtime_error("Centroid dimension mismatch!");
    std::vector<float> centroids(size_t(nlist_) * dim_);
    file.read(reinterpret_cast<char*>(centroids.data()), centroids.size() * sizeof(float));
    if (!file) throw std::runtime_error("Centroid file is truncated: " + centroids_path);
    return centroids;
}

void IVF_TT::create_index(const std::vector<float>& dataset_vectors, const std::string& centroids_path) {
    const auto start = std::chrono::steady_clock::now();
    if (options_.reference_index_build) {
        create_index_reference(dataset_vectors, centroids_path);
        std::cout << "[Index] reference build: " << elapsed_us(start, std::chrono::steady_clock::now()) / 1e6
                  << " s" << std::endl;
    } else {
        create_index_fast(dataset_vectors, centroids_path);
    }
    if (options_.index_digest) {
        // Byte-level identity check between --index-build reference and fast.
        uint64_t hash = ivf_index_build::fnv1a(
            dataset_host_padded_.data(), dataset_host_padded_.size() * sizeof(bfloat16));
        hash = ivf_index_build::fnv1a(
            index_map_host_padded_.data(), index_map_host_padded_.size() * sizeof(uint32_t), hash);
        for (uint32_t cid = 0; cid < nlist_; ++cid) {
            const auto& toc = cluster_toc_.at(cid);
            const uint32_t entry[4] = {toc.offset, toc.count, toc.padded_count, toc.start_block};
            hash = ivf_index_build::fnv1a(entry, sizeof(entry), hash);
        }
        std::cout << "[Index] digest=" << std::hex << hash << std::dec << " dataset_bytes="
                  << dataset_host_padded_.size() * sizeof(bfloat16) << " id_bytes="
                  << index_map_host_padded_.size() * sizeof(uint32_t) << std::endl;
    }
}

// Same index as create_index_reference, built in parallel: rows are
// normalized independently, assignment is vectorized across centroids with
// each score's dimension order unchanged, and every list is tiled by one
// thread directly into its precomputed page range.
void IVF_TT::create_index_fast(const std::vector<float>& dataset_vectors, const std::string& centroids_path) {
    if (dataset_vectors.empty() || dataset_vectors.size() % dim_ != 0) {
        throw std::invalid_argument("Dataset must contain a non-zero whole number of vectors");
    }
    const size_t num_vectors = dataset_vectors.size() / dim_;
    if (num_vectors > UINT32_MAX / dim_) throw std::invalid_argument("Dataset is too large for 32-bit indexing");
    const unsigned threads = ivf_index_build::worker_threads();
    auto phase = std::chrono::steady_clock::now();
    auto lap = [&phase] {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = elapsed_us(phase, now) / 1e6;
        phase = now;
        return seconds;
    };

    centroids_host_ = read_centroids(centroids_path);
    ivf_index_build::normalize_rows(centroids_host_.data(), nlist_, dim_, 1);
    // Uninitialized storage: each thread writes (and first-touches) its rows,
    // instead of a single-threaded zero fill of the whole copy.
    std::unique_ptr<float[]> dataset_norm(new float[dataset_vectors.size()]);
    ivf_index_build::parallel_chunks(num_vectors, threads, [&](size_t begin, size_t end) {
        std::copy(dataset_vectors.begin() + begin * dim_, dataset_vectors.begin() + end * dim_,
                  dataset_norm.get() + begin * dim_);
        for (size_t i = begin; i < end; ++i) ivf_index_build::normalize_row(dataset_norm.get() + i * dim_, dim_);
    });
    const double normalize_s = lap();

    const auto best_clusters = ivf_index_build::assign_fast(
        dataset_norm.get(), num_vectors, centroids_host_.data(), nlist_, dim_, threads);
    std::vector<std::vector<uint32_t>> cluster_assignments(nlist_);
    for (uint32_t i = 0; i < num_vectors; ++i) cluster_assignments[best_clusters[i]].push_back(i);
    vector_lists_ = best_clusters;  // validation export only
    const double assign_s = lap();

    // Page ranges first (an empty list keeps one placeholder page), then
    // every list fills its own range.
    const uint32_t padded_dim = (dim_ + 31) / 32 * 32;
    const uint32_t Ht = padded_dim / 32;
    std::vector<uint32_t> first_page(nlist_ + 1, 0);
    cluster_toc_.clear();
    for (uint32_t i = 0; i < nlist_; ++i) {
        const uint32_t count = cluster_assignments[i].size();
        const uint32_t cluster_blocks = std::max(1u, (count + 31) / 32);
        cluster_toc_[i] = {first_page[i], count, cluster_blocks * 32, first_page[i]};
        first_page[i + 1] = first_page[i] + cluster_blocks;
    }
    const size_t total_pages = first_page[nlist_];
    dataset_host_padded_.assign(total_pages * Ht * 1024, bfloat16(0.0f));
    index_map_host_padded_.assign(total_pages * 1024, 0);
    ivf_index_build::parallel_items(nlist_, threads, [&](size_t i) {
        const auto& members = cluster_assignments[i];
        const uint32_t count = members.size();
        const uint32_t Wt = first_page[i + 1] - first_page[i];
        const uint32_t padded_count = Wt * 32;
        std::vector<bfloat16> cluster_flat(size_t(padded_dim) * padded_count, bfloat16(0.0f));
        for (uint32_t d = 0; d < dim_; ++d) {
            for (uint32_t v = 0; v < count; ++v) {
                cluster_flat[size_t(d) * padded_count + v] = bfloat16(dataset_norm[size_t(members[v]) * dim_ + d]);
            }
        }
        const std::vector<bfloat16> cluster_tilized = tilize_nfaces(cluster_flat, padded_dim, padded_count);
        bfloat16* destination = dataset_host_padded_.data() + size_t(first_page[i]) * Ht * 1024;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                std::copy_n(cluster_tilized.begin() + (size_t(h) * Wt + w) * 1024, 1024,
                            destination + (size_t(w) * Ht + h) * 1024);
            }
        }
        std::vector<uint32_t> index_flat(size_t(32) * padded_count, 0xFFFFFFFFu);
        for (uint32_t r = 0; r < 32; ++r) std::copy(members.begin(), members.end(), index_flat.begin() + size_t(r) * padded_count);
        const std::vector<uint32_t> index_tilized = tilize_nfaces(index_flat, 32, padded_count);
        std::copy(index_tilized.begin(), index_tilized.end(),
                  index_map_host_padded_.begin() + size_t(first_page[i]) * 1024);
    });
    const double tile_s = lap();
    if (options_.rerank) {  // keep the normalized FP32 vectors for re-ranking
        rerank_vectors_ = std::move(dataset_norm);
        rerank_vector_count_ = num_vectors;
    }
    std::cout << "[Index] fast build: normalize " << normalize_s << " s, assign " << assign_s << " s, tile "
              << tile_s << " s (" << threads << " threads)" << std::endl;
}

void IVF_TT::load_to_device() {
    uint32_t padded_dim = (dim_ + 31) / 32 * 32;

    uint32_t dataset_page_size = padded_dim * 64;
    uint32_t index_page_size = 4096;

    distributed::DeviceLocalBufferConfig dataset_dram_config{.page_size = dataset_page_size, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig index_dram_config{.page_size = index_page_size, .buffer_type = BufferType::DRAM};


    distributed::DeviceLocalBufferConfig generic_dram_config_u32{.page_size = 4096, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig centroids_dram_config{.page_size = padded_dim * 64, .buffer_type = BufferType::DRAM};



    std::vector<bfloat16> centroids_host_bf16;
    centroids_host_bf16.reserve(padded_dim * nlist_);
    for (uint32_t d = 0; d < padded_dim; d++) {
        for (uint32_t i = 0; i < nlist_; i++) {
            if (d < dim_) centroids_host_bf16.push_back(bfloat16(centroids_host_[i * dim_ + d]));
            else centroids_host_bf16.push_back(bfloat16(0.0f));
        }
    }

    std::vector<bfloat16> centroids_bf_row_major = tilize_nfaces(centroids_host_bf16, padded_dim, nlist_);
    uint32_t Ht_centroids = padded_dim / 32;
    uint32_t Wt_centroids = nlist_ / 32;
    std::vector<bfloat16> centroids_bf(Ht_centroids * Wt_centroids * 1024);
    for (uint32_t w = 0; w < Wt_centroids; w++) {
        for (uint32_t h = 0; h < Ht_centroids; h++) {
            std::copy(
                centroids_bf_row_major.begin() + (h * Wt_centroids + w) * 1024,
                centroids_bf_row_major.begin() + (h * Wt_centroids + w + 1) * 1024,
                centroids_bf.begin() + (w * Ht_centroids + h) * 1024
            );
        }
    }

    centroids_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = centroids_bf.size() * 2}, centroids_dram_config, mesh_device_.get());
    auto transfer_start = std::chrono::steady_clock::now();
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), centroids_buffer_, centroids_bf, true);
    auto transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "host_to_device",
        "index_centroids",
        centroids_bf.size() * sizeof(bfloat16),
        elapsed_us(transfer_start, transfer_end),
        "blocking");

    std::vector<uint32_t> coarse_indices_flat(32 * nlist_);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < nlist_; c++) {
            coarse_indices_flat[r * nlist_ + c] = c;
        }
    }
    std::vector<uint32_t> coarse_indices = tilize_nfaces(coarse_indices_flat, 32, nlist_);

    coarse_indices_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = coarse_indices.size() * 4}, generic_dram_config_u32, mesh_device_.get());
    transfer_start = std::chrono::steady_clock::now();
    distributed::EnqueueWriteMeshBuffer(
        mesh_device_->mesh_command_queue(), coarse_indices_buffer_, coarse_indices, true);
    transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "host_to_device",
        "index_coarse_indices",
        coarse_indices.size() * sizeof(uint32_t),
        elapsed_us(transfer_start, transfer_end),
        "blocking");

    dataset_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = dataset_host_padded_.size() * 2}, dataset_dram_config, mesh_device_.get());
    transfer_start = std::chrono::steady_clock::now();
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), dataset_buffer_, dataset_host_padded_, true);
    transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "host_to_device",
        "index_dataset",
        dataset_host_padded_.size() * sizeof(bfloat16),
        elapsed_us(transfer_start, transfer_end),
        "blocking");

    indices_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = index_map_host_padded_.size() * 4}, index_dram_config, mesh_device_.get());
    transfer_start = std::chrono::steady_clock::now();
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), indices_buffer_, index_map_host_padded_, true);
    transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "host_to_device",
        "index_vector_ids",
        index_map_host_padded_.size() * sizeof(uint32_t),
        elapsed_us(transfer_start, transfer_end),
        "blocking");
}

// Coarse program, built once: its CBs and compile arguments do not depend on
// the query count; each search only sets runtime arguments.
void IVF_TT::build_coarse_program() {
    Program program{};

    tt::tt_metal::IDevice* device = mesh_device_->get_device(0);
    auto grid = device->compute_with_storage_grid_size();
    CoreCoord end_core = {grid.x - 1, grid.y - 1};
    CoreRange all_cores = CoreRange({0, 0}, end_core);

    uint32_t tile_size_bf16 = 2048;
    uint32_t tile_size_u32 = 4096;
    uint32_t query_tiles_width = (dim_ + 31) / 32;

    uint32_t Wt = nlist_ / 32;
    uint32_t logWt = (uint32_t)std::log2(Wt);
    // Top-32, or exact top-64 (two champion tiles) when nprobe > 32.
    uint32_t K = ivf_cluster_major::coarse_k(nprobe_);
    uint32_t Kt = (K + 31) / 32;
    uint32_t logk = (uint32_t)std::log2(K);

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * query_tiles_width * tile_size_bf16, {{CB::c_in0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in0, tile_size_bf16));
    CreateCircularBuffer(program, all_cores, CircularBufferConfig(8 * tile_size_bf16, {{CB::c_in1, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in1, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_u32, {{CB::c_in2, tt::DataFormat::Int32}}).set_page_size(CB::c_in2, tile_size_u32));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_intermed0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed0, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(Wt * tile_size_bf16, {{CB::c_intermed1, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed1, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(Wt * tile_size_u32, {{CB::c_intermed2, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed2, tile_size_u32));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * Kt * tile_size_bf16, {{CB::c_out0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_out0, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * Kt * tile_size_u32, {{CB::c_out1, tt::DataFormat::Int32}}).set_page_size(CB::c_out1, tile_size_u32));

    // Top-64 only: champion B (ranks 33..64) and the elements dropped from A.
    if (Kt == 2) {
        CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_intermed3, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed3, tile_size_bf16));
        CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_u32, {{CB::c_intermed4, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed4, tile_size_u32));
        CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_intermed5, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed5, tile_size_bf16));
        CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_u32, {{CB::c_intermed6, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed6, tile_size_u32));
    }

    std::vector<uint32_t> reader_compile_args = { CB::c_in0, CB::c_in1, CB::c_in2, query_tiles_width, Wt };
    reader_coarse_id_ = CreateKernel(program, "ann_ivf_ttaiss/kernels/dataflow/reader_coarse.cpp",
        all_cores, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_args});

    std::vector<uint32_t> writer_compile_args = { CB::c_out0, CB::c_out1 };
    writer_coarse_id_ = CreateKernel(program, "ann_ivf_ttaiss/kernels/dataflow/writer_coarse.cpp",
        all_cores, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_args});

    std::vector<uint32_t> compute_compile_args = {
        CB::c_in0, CB::c_in1, CB::c_in2, CB::c_intermed1, CB::c_intermed2,
        CB::c_out0, CB::c_out1, Wt, K, Kt, logk, logWt, CB::c_intermed0, query_tiles_width,
        CB::c_intermed3, CB::c_intermed4, CB::c_intermed5, CB::c_intermed6
    };
    compute_coarse_id_ = CreateKernel(program, "ann_ivf_ttaiss/kernels/compute/compute_coarse.cpp",
        all_cores, ComputeConfig{.math_fidelity = static_cast<MathFidelity>(options_.coarse_fidelity),
                                 .fp32_dest_acc_en = true, .compile_args = compute_compile_args});

    coarse_workload_.emplace();
    coarse_workload_->add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
}

namespace {
constexpr uint32_t tile_size_bf16 = 2048;
constexpr uint32_t tile_size_u32 = 4096;

CBHandle make_cb(Program& program, const CoreRangeSet& cores, uint32_t cb, uint64_t bytes, uint32_t page,
                 tt::DataFormat format) {
    return CreateCircularBuffer(program, cores,
        CircularBufferConfig(static_cast<uint32_t>(bytes), {{cb, format}}).set_page_size(cb, page));
}

// Writes/reads only the leading region a search uses of a (possibly larger,
// reused) buffer. Region offset and size must be whole pages.
template <class T>
void write_region(distributed::MeshCommandQueue& queue, const std::shared_ptr<distributed::MeshBuffer>& buffer,
                  const std::vector<T>& data, bool blocking) {
    const uint64_t bytes = data.size() * sizeof(T);
    if (bytes > buffer->size() || bytes % buffer->page_size())
        throw std::logic_error("Upload is not a whole-page region of its buffer");
    std::vector<distributed::ShardDataTransfer> transfers{
        distributed::ShardDataTransfer{distributed::MeshCoordinate(0, 0)}
            .host_data(const_cast<T*>(data.data()))
            .region(BufferRegion(0, bytes))};
    queue.enqueue_write_shards(buffer, transfers, blocking);
}
template <class T>
std::vector<T> read_region(distributed::MeshCommandQueue& queue, const std::shared_ptr<distributed::MeshBuffer>& buffer,
                           size_t count) {
    std::vector<T> data(count);
    const uint64_t bytes = count * sizeof(T);
    if (bytes > buffer->size() || bytes % buffer->page_size())
        throw std::logic_error("Readback is not a whole-page region of its buffer");
    std::vector<distributed::ShardDataTransfer> transfers{
        distributed::ShardDataTransfer{distributed::MeshCoordinate(0, 0)}.host_data(data.data()).region(
            BufferRegion(0, bytes))};
    queue.enqueue_read_shards(transfers, buffer, true);
    return data;
}
}  // namespace

bool IVF_TT::grow_buffer(std::shared_ptr<distributed::MeshBuffer>& buffer, uint64_t bytes, uint32_t page) {
    if (buffer && buffer->size() >= bytes && buffer->page_size() == page) return false;
    buffer.reset();  // free first, so the allocator can reuse the space
    buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = bytes},
        distributed::DeviceLocalBufferConfig{.page_size = page, .buffer_type = BufferType::DRAM},
        mesh_device_.get());
    return true;
}

// Scan program. CB capacities must match scan_l1::worker_cb_bytes. Both passes
// use the same program shape; the threshold CBs are idle in the first pass.
void IVF_TT::build_scan_stage(FineStage& stage, uint32_t threshold_rank) {
    const uint32_t query_tiles_width = (dim_ + 31) / 32;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto i32 = tt::DataFormat::Int32;
    Program program{};
    make_cb(program, all_cores_, CB::c_in0,
            ivf_cluster_major::scan_l1::query_cb_tiles(dim_, batch_chunk_) * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_in1, uint64_t(prefetch_reader_) * query_tiles_width * tile_size_bf16,
            tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_in2, uint64_t(prefetch_reader_) * tile_size_u32, tile_size_u32, i32);
    make_cb(program, all_cores_, CB::c_intermed0,
            ivf_cluster_major::scan_l1::score_cb_tiles(batch_chunk_) * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_intermed1, tile_size_u32, tile_size_u32, i32);
    make_cb(program, all_cores_, CB::c_intermed3, uint64_t(batch_chunk_) * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_intermed4, uint64_t(batch_chunk_) * tile_size_u32, tile_size_u32, i32);
    make_cb(program, all_cores_, CB::c_intermed6, tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_intermed7, tile_size_u32, tile_size_u32, i32);
    // Reader-private query IDs of the chunk's gathered tiles (scan_l1::tile_query_page).
    make_cb(program, all_cores_, CB::c_in3, uint64_t(batch_chunk_) * ivf_cluster_major::scan_l1::tile_query_page,
            static_cast<uint32_t>(ivf_cluster_major::scan_l1::tile_query_page), i32);
    // Merge skipping: threshold tiles (reader), reduce scaler, diffs, flags.
    make_cb(program, all_cores_, CB::c_in4, 2ull * batch_chunk_ * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_in5, tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_intermed2, uint64_t(batch_chunk_) * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_out2, uint64_t(batch_chunk_) * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_out0, 2 * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_out1, 2 * tile_size_u32, tile_size_u32, i32);
    make_cb(program, all_cores_, CB::c_intermed5, stage.script_bytes, static_cast<uint32_t>(stage.script_bytes), i32);
    // Compile-time threshold rank (second pass): each rank is its own binary.
    stage.reader = CreateKernel(program, "ann_ivf_ttaiss/kernels/dataflow/reader_scan.cpp",
        all_cores_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                                       .compile_args = {threshold_rank}});
    stage.threshold_rank = threshold_rank;
    stage.writer = CreateKernel(program, "ann_ivf_ttaiss/kernels/dataflow/writer_scan.cpp",
        all_cores_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    stage.compute = CreateKernel(program, "ann_ivf_ttaiss/kernels/compute/compute_scan.cpp",
        all_cores_, ComputeConfig{.math_fidelity = static_cast<MathFidelity>(options_.fine_fidelity),
                                  .fp32_dest_acc_en = true});
    stage.workload.emplace();
    stage.workload->add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
}

// Merge program on the same cores, ordered after its scan by the queue.
void IVF_TT::build_merge_stage(FineStage& stage) {
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto i32 = tt::DataFormat::Int32;
    Program program{};
    make_cb(program, all_cores_, CB::c_in0, 2 * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_in1, 2 * tile_size_u32, tile_size_u32, i32);
    make_cb(program, all_cores_, CB::c_intermed3, 2 * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_intermed4, 2 * tile_size_u32, tile_size_u32, i32);
    make_cb(program, all_cores_, CB::c_out0, 2 * tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_out1, 2 * tile_size_u32, tile_size_u32, i32);
    // Reader-private sentinel rows for missing partials (ivf_cluster_gather::merge_cb_bytes).
    make_cb(program, all_cores_, CB::c_intermed6, tile_size_bf16, tile_size_bf16, bf16);
    make_cb(program, all_cores_, CB::c_intermed7, tile_size_u32, tile_size_u32, i32);
    make_cb(program, all_cores_, CB::c_intermed5, stage.script_bytes, static_cast<uint32_t>(stage.script_bytes), i32);
    stage.reader = CreateKernel(program, "ann_ivf_ttaiss/kernels/dataflow/reader_merge.cpp",
        all_cores_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    stage.writer = CreateKernel(program, "ann_ivf_ttaiss/kernels/dataflow/writer_merge.cpp",
        all_cores_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    stage.compute = CreateKernel(program, "ann_ivf_ttaiss/kernels/compute/compute_merge.cpp",
        all_cores_, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    stage.workload.emplace();
    stage.workload->add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
}

std::pair<std::vector<float>, std::vector<int64_t>> IVF_TT::search_all(const float* query_data [[maybe_unused]], uint32_t total_queries, uint32_t k) {
    if (query_data == nullptr || total_queries == 0) {
        throw std::invalid_argument("search_all requires at least one query");
    }
    if (total_queries > 10000) {
        throw std::invalid_argument("ann_ivf_ttaiss currently supports at most 10000 queries per search");
    }
    if (k == 0 || k > 32) {
        throw std::invalid_argument("k must be between 1 and the hardware top-k limit of 32");
    }

    auto total_start = std::chrono::steady_clock::now();
    [[maybe_unused]] uint32_t padded_q = (total_queries + 31) / 32 * 32;
    (void)nprobe_;

    uint32_t num_batches = padded_q / 32;

    tt::tt_metal::IDevice* device = mesh_device_->get_device(0);
    auto grid = device->compute_with_storage_grid_size();
    // Scan and merge both use every core; there is no dedicated aggregator.
    const uint32_t num_cores = grid.x * grid.y;
    if (num_cores == 0) {
        throw std::runtime_error("ann_ivf_ttaiss requires at least one Tensix core");
    }


    uint32_t padded_dim = (dim_ + 31) / 32 * 32;
    distributed::DeviceLocalBufferConfig dram_bf16{.page_size = 2048, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig generic_dram_config_bf16{.page_size = 2048, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig generic_dram_config_u32{.page_size = 4096, .buffer_type = BufferType::DRAM};

    // Buffers and programs persist across searches and grow when needed.
    if (options_.ablation_rebuild) {  // ablation: everything per search, as before persistence
        fine_ = FinePrograms{};
        coarse_workload_.reset();
        coarse_query_buffer_.reset();
        coarse_out_val_buffer_.reset();
        coarse_out_ind_buffer_.reset();
    }
    const uint32_t Kt_coarse = ivf_cluster_major::coarse_k(nprobe_) / 32;
    grow_buffer(coarse_query_buffer_, uint64_t(padded_q) * padded_dim * 2, static_cast<uint32_t>(dram_bf16.page_size));
    grow_buffer(coarse_out_val_buffer_, uint64_t(num_batches) * Kt_coarse * 2048, static_cast<uint32_t>(generic_dram_config_bf16.page_size));
    grow_buffer(coarse_out_ind_buffer_, uint64_t(num_batches) * Kt_coarse * 4096, static_cast<uint32_t>(generic_dram_config_u32.page_size));
    if (!coarse_workload_) build_coarse_program();

    std::vector<float> query_vec(padded_q * padded_dim, 0.0f);
    auto transfer_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < total_queries; i++) {
        std::memcpy(query_vec.data() + i * padded_dim, query_data + i * dim_, dim_ * sizeof(float));
    }
    auto transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "cpu",
        "host_to_host",
        "query_row_staging_memcpy",
        static_cast<uint64_t>(total_queries) * dim_ * sizeof(float),
        elapsed_us(transfer_start, transfer_end),
        "synchronous",
        "Times the complete row-copy loop; destination rows include padded stride");
    // Rows are independent: L2 normalization (same per-row arithmetic as the
    // sequential loop), BF16 rounding and tiling run in parallel. Element
    // (q, d) goes to tile (q / 32) * (padded_dim / 32) + d / 32 at its face
    // offset, the layout tilize_nfaces produces.
    const uint32_t query_tiles_per_row = padded_dim / 32;
    std::vector<bfloat16> query_bf(size_t(padded_q) * padded_dim);
    ivf_index_build::parallel_chunks(padded_q, 8, [&](size_t begin, size_t end) {  // a few threads: spawn cost
        for (size_t i = begin; i < end; ++i) {
            float* row = query_vec.data() + i * padded_dim;
            float sum = 0.0f;
            for (uint32_t d = 0; d < padded_dim; ++d) sum += row[d] * row[d];
            const float norm = std::sqrt(sum);
            if (norm > 0) {
                for (uint32_t d = 0; d < padded_dim; ++d) row[d] /= norm;
            }
            bfloat16* tiles = query_bf.data() + (i / 32) * query_tiles_per_row * 1024;
            for (uint32_t d = 0; d < padded_dim; ++d)
                tiles[(d / 32) * 1024 + fine_tile_offset(uint32_t(i % 32), d % 32)] = bfloat16(row[d]);
        }
    });
    transfer_start = std::chrono::steady_clock::now();
    write_region(mesh_device_->mesh_command_queue(), coarse_query_buffer_, query_bf, true);
    transfer_end = std::chrono::steady_clock::now();
    const double query_h2d_us = elapsed_us(transfer_start, transfer_end);
    memory_logger_->record(
        "pcie",
        "host_to_device",
        "query_tiles",
        query_bf.size() * sizeof(bfloat16),
        query_h2d_us,
        "blocking",
        "Blocking mode ensures the timer covers completion rather than enqueue latency");

    auto t_coarse_start = std::chrono::steady_clock::now();
    // Fine search uses the per-query selections kept in coarse_selections_.
    (void)perform_coarse_search(query_data, total_queries);
    auto t_coarse_end = std::chrono::steady_clock::now();

    // Per-list query tiles: no query batches and no grouping. Every
    // non-empty list gets 32-row tiles of the queries that selected it.
    // Two passes (--first-pass-lists m): pass 1 scans each query's best m
    // lists and merges them into partial slots merged_base + batch; pass 2
    // scans the remaining lists with each query's pass-1 32nd-best score as a
    // per-row threshold and skips page merges that cannot reach the top-32;
    // the final merge combines the pass-1 row with the pass-2 partials.
    const auto plan_start = std::chrono::steady_clock::now();
    std::vector<ivf_cluster_major::ListLayout> lists(nlist_);
    for (uint32_t cid = 0; cid < nlist_; ++cid) {
        const auto& toc = cluster_toc_.at(cid);
        lists[cid] = {toc.start_block, toc.count, toc.padded_count};
    }
    const uint32_t first_lists = options_.first_pass_lists;
    const bool two_pass = first_lists > 0 && first_lists < nprobe_;
    const uint32_t query_tiles_width = padded_dim / 32;
    // Partial slots: single pass [tiles]; two passes [pass-1 tiles][pass-1
    // merged rows, one tile per batch][pass-2 tiles]. The pass-2 range is an
    // upper bound, so pass 1 is launched before pass 2 is planned and the
    // host plans pass 2 while the device runs pass 1.
    std::vector<ivf_cluster_gather::GatherPlan> plans;
    std::vector<std::vector<uint32_t>> rest;
    if (two_pass) {
        auto [first, second] = ivf_cluster_gather::split_selections(coarse_selections_, first_lists);
        plans.push_back(ivf_cluster_gather::plan_gather(first, first_lists, lists, batch_chunk_, num_cores, num_cores));
        rest = std::move(second);
    } else {
        plans.push_back(
            ivf_cluster_gather::plan_gather(coarse_selections_, nprobe_, lists, batch_chunk_, num_cores, num_cores));
    }
    const uint32_t merged_base = static_cast<uint32_t>(plans[0].tiles.size());
    const uint32_t pass2_base = merged_base + num_batches;
    const uint64_t slot_count = std::max<uint64_t>(
        1, two_pass ? pass2_base + ivf_cluster_gather::max_tiles(total_queries, nprobe_ - first_lists, nlist_)
                    : plans[0].tiles.size());
    if (slot_count >= (uint64_t(1) << 27)) throw std::runtime_error("Too many partial slots");

    // Scripts of a stage, one per core. A stage's DRAM page (and script CB)
    // has the largest size any search has needed so far; L1 must hold its CBs.
    using CoreScripts = std::vector<std::vector<uint32_t>>;
    auto encode_pass = [&](size_t p) {
        CoreScripts scan(num_cores), merge(num_cores);
        for (uint32_t w = 0; w < num_cores; ++w) {
            scan[w] = ivf_cluster_major::encode_scan_script(plans[p].worker_tasks[w], batch_chunk_);
            merge[w] = p == 1 ? ivf_cluster_gather::encode_final_merge_script(plans[1], merged_base, w)
                              : ivf_cluster_gather::encode_merge_script(plans[p], w);
        }
        return std::pair(std::move(scan), std::move(merge));
    };
    // Grows the stage's page to fit `scripts`; true when it grew (the program
    // must then be rebuilt with the larger script CB).
    auto fit_stage = [&](FineStage& stage, const CoreScripts& scripts, bool scan) {
        size_t words = 0;
        for (const auto& script : scripts) words = std::max(words, script.size());
        const uint64_t needed = ivf_cluster_major::script_page_bytes(words);
        if (needed <= stage.script_bytes) return false;
        const uint64_t l1 = scan ? ivf_cluster_major::scan_l1::worker_cb_bytes(dim_, prefetch_reader_, batch_chunk_, needed)
                                 : ivf_cluster_gather::merge_cb_bytes(needed);
        if (l1 > usable_l1_bytes_) {
            throw std::runtime_error(
                "Circular buffers need " + std::to_string(l1 / 1024) + " KiB (" + (scan ? "scan" : "merge") +
                ") but usable worker L1 is " + std::to_string(usable_l1_bytes_ / 1024) +
                " KiB; reduce --batch-chunk or --prefetch-reader");
        }
        stage.script_bytes = needed;
        return true;
    };
    auto flatten = [&](const CoreScripts& scripts, uint64_t page_bytes) {
        std::vector<uint32_t> flat(size_t(num_cores) * (page_bytes / sizeof(uint32_t)), 0);
        for (uint32_t w = 0; w < num_cores; ++w)
            std::copy(scripts[w].begin(), scripts[w].end(), flat.begin() + size_t(w) * (page_bytes / sizeof(uint32_t)));
        return flat;
    };
    // Tile-query table of a pass: the scan reader assembles every gathered
    // tile on the device from the original query tiles (already in DRAM), so
    // only 32 query IDs per tile are uploaded; unused rows are UINT32_MAX.
    auto tile_table = [&](const ivf_cluster_gather::GatherPlan& plan) {
        std::vector<uint32_t> table(std::max<size_t>(1, plan.tiles.size()) * 32, UINT32_MAX);
        for (size_t tile = 0; tile < plan.tiles.size(); ++tile)
            std::copy_n(plan.requesters.begin() + plan.tiles[tile].first, plan.tiles[tile].rows,
                        table.begin() + tile * 32);
        return table;
    };
    auto print_plan = [&](size_t p) {
        const auto& plan = plans[p];
        const double average = static_cast<double>(plan.total_worker_weight) / static_cast<double>(num_cores);
        std::cout << "[Host] Gather plan" << (two_pass ? (p == 0 ? " (pass 1)" : " (pass 2)") : "")
                  << ": tiles=" << plan.tiles.size() << ", tasks=" << plan.tasks
                  << ", page x tile pairs=" << plan.pair_pages << ", useful rows=" << std::fixed << std::setprecision(1)
                  << (plan.pair_pages ? 100.0 * plan.useful_rows / (32.0 * plan.pair_pages) : 0.0) << "%"
                  << ", list page reads=" << plan.scan_pages << ", script pages=" << fine_.stages[2 * p].script_bytes
                  << "/" << fine_.stages[2 * p + 1].script_bytes << " B, scan load[min/avg/max]=" << plan.min_worker_weight << "/" << average
                  << "/" << plan.max_worker_weight << std::defaultfloat << std::setprecision(6) << std::endl;
    };

    // Host data of non-blocking uploads stays alive until Finish.
    std::vector<std::vector<uint32_t>> pending_uploads;
    uint64_t gathered_bytes = 0, script_upload_bytes = 0;
    double gather_h2d_us = 0.0, fine_scripts_h2d_us = 0.0, programs_us = 0.0;
    uint32_t programs_built = 0;
    // Second-pass threshold: the 32nd-best first-pass score when the host
    // re-ranks the device top-32 (all 32 must be exact), else the k-th best
    // (only the top k are returned). A compile-time argument of the reader.
    const uint32_t threshold_rank = options_.rerank || options_.ablation_rank32 ? 31u : k - 1;
    // Uploads, programs and runtime arguments of pass p; enqueued without waiting.
    auto launch_pass = [&](size_t p, CoreScripts scan, CoreScripts merge, std::vector<uint32_t> table, bool blocking) {
        auto& scan_stage = fine_.stages[2 * p];
        auto& merge_stage = fine_.stages[2 * p + 1];
        const bool rebuild_scan = fit_stage(scan_stage, scan, true) || !scan_stage.workload ||
                                  scan_stage.threshold_rank != threshold_rank;
        const bool rebuild_merge = fit_stage(merge_stage, merge, false) || !merge_stage.workload;
        std::vector<uint32_t> scan_flat = flatten(scan, scan_stage.script_bytes);
        std::vector<uint32_t> merge_flat = flatten(merge, merge_stage.script_bytes);
        grow_buffer(fine_.tile_queries[p], table.size() * sizeof(uint32_t),
                    static_cast<uint32_t>(ivf_cluster_major::scan_l1::tile_query_page));
        grow_buffer(scan_stage.scripts, scan_flat.size() * sizeof(uint32_t), static_cast<uint32_t>(scan_stage.script_bytes));
        grow_buffer(merge_stage.scripts, merge_flat.size() * sizeof(uint32_t),
                    static_cast<uint32_t>(merge_stage.script_bytes));
        auto& queue = mesh_device_->mesh_command_queue();
        auto start = std::chrono::steady_clock::now();
        write_region(queue, fine_.tile_queries[p], table, blocking);
        auto end = std::chrono::steady_clock::now();
        gather_h2d_us += blocking ? elapsed_us(start, end) : 0.0;
        gathered_bytes += table.size() * sizeof(uint32_t);
        start = std::chrono::steady_clock::now();
        write_region(queue, scan_stage.scripts, scan_flat, blocking);
        write_region(queue, merge_stage.scripts, merge_flat, blocking);
        end = std::chrono::steady_clock::now();
        fine_scripts_h2d_us += blocking ? elapsed_us(start, end) : 0.0;
        script_upload_bytes += (scan_flat.size() + merge_flat.size()) * sizeof(uint32_t);

        const auto programs_start = std::chrono::steady_clock::now();
        if (rebuild_scan) { build_scan_stage(scan_stage, threshold_rank); ++programs_built; }
        if (rebuild_merge) { build_merge_stage(merge_stage); ++programs_built; }
        const bool second_pass = two_pass && p == 1;
        const bool first_pass_merge = two_pass && p == 0;
        Program& scan_program = scan_stage.workload->get_programs().begin()->second;
        Program& merge_program = merge_stage.workload->get_programs().begin()->second;
        // Core w (row-major logical order) runs scan script w and merge
        // script w. Idle cores receive empty scripts and still run the full
        // control flow.
        for (uint32_t y = 0, w = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++, w++) {
                const CoreCoord core = {x, y};
                // The scan assembles gathered tile `batch` (script pairs) from
                // the original query tiles, using that tile's row of the pass's
                // ID table; the second pass also reads the pass-1 thresholds.
                SetRuntimeArgs(scan_program, scan_stage.reader, core,
                    {coarse_query_buffer_->address(), dataset_buffer_->address(), indices_buffer_->address(),
                     padded_dim, scan_stage.scripts->address(), static_cast<uint32_t>(scan_stage.script_bytes), w,
                     batch_chunk_, fine_.tile_queries[p]->address(), second_pass ? 1u : 0u,
                     fine_.partial_val->address(), merged_base});
                SetRuntimeArgs(scan_program, scan_stage.writer, core,
                    {fine_.partial_val->address(), fine_.partial_ind->address()});
                SetRuntimeArgs(scan_program, scan_stage.compute, core,
                    {query_tiles_width, batch_chunk_, second_pass ? 1u : 0u});
                // The pass-1 merge writes its rows into the partial buffers.
                const std::vector<uint32_t> merge_args = {
                    fine_.partial_val->address(), fine_.partial_ind->address(),
                    first_pass_merge ? fine_.partial_val->address() : fine_.out_val->address(),
                    first_pass_merge ? fine_.partial_ind->address() : fine_.out_ind->address(),
                    merge_stage.scripts->address(), static_cast<uint32_t>(merge_stage.script_bytes), w,
                    first_pass_merge ? merged_base : 0u};
                SetRuntimeArgs(merge_program, merge_stage.reader, core, merge_args);
                SetRuntimeArgs(merge_program, merge_stage.writer, core, merge_args);
                SetRuntimeArgs(merge_program, merge_stage.compute, core, merge_args);
            }
        }
        programs_us += elapsed_us(programs_start, std::chrono::steady_clock::now());
        distributed::EnqueueMeshWorkload(queue, *scan_stage.workload, false);
        distributed::EnqueueMeshWorkload(queue, *merge_stage.workload, false);
        pending_uploads.push_back(std::move(table));
        pending_uploads.push_back(std::move(scan_flat));
        pending_uploads.push_back(std::move(merge_flat));
    };

    auto [scan0, merge0] = encode_pass(0);
    double gather_plan_us = elapsed_us(plan_start, std::chrono::steady_clock::now());
    auto tiles_start = std::chrono::steady_clock::now();
    auto table0 = tile_table(plans[0]);
    double gather_tiles_us = elapsed_us(tiles_start, std::chrono::steady_clock::now());

    grow_buffer(fine_.partial_val, slot_count * tile_size_bf16, tile_size_bf16);
    grow_buffer(fine_.partial_ind, slot_count * tile_size_u32, tile_size_u32);
    grow_buffer(fine_.out_val, uint64_t(padded_q) * 32 * 2, tile_size_bf16);
    grow_buffer(fine_.out_ind, uint64_t(padded_q) * 32 * 4, tile_size_u32);

    std::cout << "[Host] Enqueuing " << (two_pass ? 4 : 2) << " scan/merge workloads..." << std::endl;
    // Pass 1 (or the single pass): blocking uploads, timed as H2D.
    const auto pass1_start = std::chrono::steady_clock::now();
    launch_pass(0, std::move(scan0), std::move(merge0), std::move(table0), true);
    print_plan(0);
    // The device time starts with the first workload; the uploads and the
    // program building before it are host preparation.
    auto start_device = std::chrono::steady_clock::now();
    const double pass1_launch_us = elapsed_us(pass1_start, start_device);  // uploads + programs of pass 1
    double overlapped_prep_us = 0.0;
    if (two_pass) {
        // Planned while the device runs pass 1; non-blocking uploads.
        const auto overlap_start = std::chrono::steady_clock::now();
        const auto plan2_start = overlap_start;
        plans.push_back(
            ivf_cluster_gather::plan_gather(rest, nprobe_ - first_lists, lists, batch_chunk_, num_cores, num_cores));
        if (plans[1].tiles.size() > ivf_cluster_gather::max_tiles(total_queries, nprobe_ - first_lists, nlist_))
            throw std::logic_error("Second-pass tiles exceed their reserved partial slots");
        ivf_cluster_gather::shift_slots(plans[1], pass2_base);
        auto [scan1, merge1] = encode_pass(1);
        gather_plan_us += elapsed_us(plan2_start, std::chrono::steady_clock::now());
        tiles_start = std::chrono::steady_clock::now();
        auto table1 = tile_table(plans[1]);
        gather_tiles_us += elapsed_us(tiles_start, std::chrono::steady_clock::now());
        launch_pass(1, std::move(scan1), std::move(merge1), std::move(table1), false);
        overlapped_prep_us = elapsed_us(overlap_start, std::chrono::steady_clock::now());
    }
    uint64_t total_tiles = 0, total_tasks = 0, total_pair_pages = 0, total_scan_pages = 0, total_useful_rows = 0;
    for (const auto& plan : plans) {
        total_tiles += plan.tiles.size();
        total_tasks += plan.tasks;
        total_pair_pages += plan.pair_pages;
        total_scan_pages += plan.scan_pages;
        total_useful_rows += plan.useful_rows;
    }
    const uint32_t gathered_tiles = static_cast<uint32_t>(total_tiles);
    distributed::Finish(mesh_device_->mesh_command_queue());
    auto end_device = std::chrono::steady_clock::now();
    if (two_pass) print_plan(1);
    pending_uploads.clear();
    memory_logger_->record(
        "pcie", "host_to_device", "tile_query_ids", gathered_bytes, gather_h2d_us, "blocking",
        "32 query IDs per gathered tile (the scan reader assembles the tiles); pass-2 upload is non-blocking "
        "and overlaps pass 1; included in H2D stage");
    memory_logger_->record(
        "pcie", "host_to_device", "fine_worker_scripts", script_upload_bytes, fine_scripts_h2d_us, "blocking",
        "Scan and merge scripts of every pass; pass-2 uploads are non-blocking and overlap pass 1");
    double device_time_us = elapsed_us(start_device, end_device);
    auto out_val = fine_.out_val;
    auto out_ind = fine_.out_ind;
    if (options_.energy && !options_.rerank) {
        // Energy workload: the merge has written every result tile to DRAM;
        // skip readback, restoration and reporting (ann_ivf_energy contract).
        last_search_us_ = elapsed_us(total_start, std::chrono::steady_clock::now());
        return {};
    }

    auto start_output = end_device;
    std::vector<bfloat16> final_vals_raw_bf(num_batches * 1024);
    std::vector<uint32_t> final_inds_raw_u32(num_batches * 1024);

    transfer_start = std::chrono::steady_clock::now();
    final_vals_raw_bf = read_region<bfloat16>(mesh_device_->mesh_command_queue(), out_val, final_vals_raw_bf.size());
    transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie", "device_to_host", "fine_output_values", final_vals_raw_bf.size() * sizeof(bfloat16),
        elapsed_us(transfer_start, transfer_end), "blocking");

    transfer_start = std::chrono::steady_clock::now();
    final_inds_raw_u32 = read_region<uint32_t>(mesh_device_->mesh_command_queue(), out_ind, final_inds_raw_u32.size());
    transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie", "device_to_host", "fine_output_indices", final_inds_raw_u32.size() * sizeof(uint32_t),
        elapsed_us(transfer_start, transfer_end), "blocking");
    std::vector<bfloat16> final_vals_bf = untilize_nfaces(final_vals_raw_bf, 32 * num_batches, 32);
    std::vector<uint32_t> final_inds_u32 = untilize_nfaces(final_inds_raw_u32, 32 * num_batches, 32);

    // Output pages are in original query order (identity permutation).
    const auto restore_start = std::chrono::steady_clock::now();
    ivf_cluster_major::Grouping identity;
    identity.packed_to_original.resize(total_queries);
    std::iota(identity.packed_to_original.begin(), identity.packed_to_original.end(), 0u);
    identity.original_to_packed = identity.packed_to_original;
    auto results = ivf_cluster_major::restore_topk(final_vals_bf, final_inds_u32, identity, k);
    const double query_restore_us = elapsed_us(restore_start, std::chrono::steady_clock::now());
    double rerank_us = 0.0;
    if (options_.rerank) {
        // Re-score the device top-32 of every query in FP32, keep the best k.
        const auto rerank_start = std::chrono::steady_clock::now();
        const auto candidates = ivf_cluster_major::restore_topk(final_vals_bf, final_inds_u32, identity, 32);
        results = ivf_rerank::rerank(query_vec.data(), padded_dim, total_queries, dim_, rerank_vectors_.get(),
                                     rerank_vector_count_, candidates.second, 32, k, ivf_index_build::worker_threads());
        rerank_us = elapsed_us(rerank_start, std::chrono::steady_clock::now());
    }
    if (options_.energy) {
        // Energy workload with --rerank: the readback and FP32 re-ranking
        // produce the final results and are measured; no reporting, logging,
        // recall check or dump follows.
        last_search_us_ = elapsed_us(total_start, std::chrono::steady_clock::now());
        return results;
    }

    auto end_postprocess = std::chrono::steady_clock::now();
    const double h2d_us = query_h2d_us + gather_h2d_us + fine_scripts_h2d_us;
    const double cpu_coarse_config_us = std::max(
        0.0,
        elapsed_us(total_start, t_coarse_start) + coarse_pre_device_us_ - query_h2d_us);
    const double cpu_fine_prep_us = std::max(
        0.0,
        coarse_post_device_us_ + elapsed_us(t_coarse_end, start_device) -
        fine_scripts_h2d_us - gather_h2d_us);
    const double cpu_output_us = elapsed_us(start_output, end_postprocess);
    const double total_time_us = elapsed_us(total_start, end_postprocess);
    last_search_us_ = total_time_us;
    const double chrono_stage_sum_us =
        h2d_us + cpu_coarse_config_us + coarse_device_us_ +
        cpu_fine_prep_us + device_time_us + cpu_output_us;

    const uint64_t h2d_bytes = query_bf.size() * sizeof(bfloat16) + gathered_bytes + script_upload_bytes;
    memory_logger_->record(
        "chrono_stage", "stage", "h2d", h2d_bytes, h2d_us, "blocking_transfers");
    memory_logger_->record(
        "chrono_stage", "stage", "cpu_coarse_config", 0, cpu_coarse_config_us, "wall_clock");
    memory_logger_->record(
        "chrono_stage", "stage", "tt_coarse_search", 0, coarse_device_us_, "enqueue_plus_finish");
    memory_logger_->record(
        "chrono_stage", "stage", "cpu_fine_prep", 0, cpu_fine_prep_us, "wall_clock");
    memory_logger_->record(
        "chrono_stage", "stage", "tt_fine_search", 0, device_time_us, "enqueue_plus_finish");
    memory_logger_->record(
        "chrono_stage", "stage", "cpu_output", 0, cpu_output_us, "wall_clock");
    // Detail rows belong to the six stages above; plotters must not add them
    // again. Write diagnostic records only after the search timer stops.
    memory_logger_->record(
        "chrono_detail", "stage_detail", "gather_plan", 0, gather_plan_us, "wall_clock",
        "Requesters, tiles, locations, LPT placement and scripts; pass 1 in cpu_fine_prep, pass 2 overlaps the "
        "device (inside tt_fine_search)");
    memory_logger_->record(
        "chrono_detail", "stage_detail", "gather_query_tiles", 0, gather_tiles_us, "wall_clock",
        "Included in cpu_fine_prep; tile-query ID table (tiles are assembled on the device)");
    memory_logger_->record(
        "chrono_detail", "stage_detail", "coarse_decode", 0, coarse_post_device_us_, "wall_clock",
        "Included in cpu_fine_prep; coarse readback, untilize and validation of the selected lists");
    memory_logger_->record(
        "chrono_detail", "stage_detail", "fine_programs", 0, programs_us, "wall_clock",
        "Per-search scan/merge program creation and runtime arguments of every pass; the pass-2 part overlaps "
        "the device (inside tt_fine_search), the rest is in cpu_fine_prep");
    memory_logger_->record(
        "chrono_detail", "stage_detail", "pass2_host_prep_overlapped", 0, overlapped_prep_us, "wall_clock",
        "Included in tt_fine_search: pass-2 plan, scripts, table, uploads and programs while pass 1 runs");
    memory_logger_->record(
        "chrono_detail", "stage_detail", "query_restore", 0, query_restore_us, "wall_clock",
        "Included in cpu_output; return k results per original query");
    memory_logger_->record(
        "chrono_detail", "stage_detail", "rerank", 0, rerank_us, "wall_clock",
        options_.rerank ? "Included in cpu_output; FP32 re-ranking of the device top-32" : "Re-ranking disabled");
    memory_logger_->record(
        "ttaiss", "metric", "scan_plan", 0, 0, "host_count",
        "batch_chunk=" + std::to_string(batch_chunk_) +
        "; first_pass_lists=" + std::to_string(two_pass ? first_lists : 0) +
        "; tiles=" + std::to_string(gathered_tiles) +
        "; tasks=" + std::to_string(total_tasks) +
        "; pair_pages=" + std::to_string(total_pair_pages) +
        "; useful_row_pages=" + std::to_string(total_useful_rows) +
        "; list_page_reads=" + std::to_string(total_scan_pages) +
        "; fine_fidelity=" + std::string(ivf_cluster_major::fidelity_name(options_.fine_fidelity)) +
        "; coarse_fidelity=" + std::string(ivf_cluster_major::fidelity_name(options_.coarse_fidelity)) +
        "; scan_load_max=" + std::to_string(plans.back().max_worker_weight));

    std::cout << "=========================================" << std::endl;
    std::cout << "[Chrono] H2D:                      " << std::fixed << std::setprecision(3) << h2d_us / 1000.0 << " ms" << std::endl;
    std::cout << "[Chrono] CPU coarse search config: " << std::fixed << std::setprecision(3) << cpu_coarse_config_us / 1000.0 << " ms" << std::endl;
    std::cout << "[Chrono] TT coarse search:         " << std::fixed << std::setprecision(3) << coarse_device_us_ / 1000.0 << " ms" << std::endl;
    std::cout << "[Chrono] CPU fine search prep:     " << std::fixed << std::setprecision(3) << cpu_fine_prep_us / 1000.0 << " ms" << std::endl;
    std::cout << "  [Detail] Gather plan:            " << gather_plan_us / 1000.0 << " ms" << std::endl;
    std::cout << "  [Detail] Tile-query table:       " << gather_tiles_us / 1000.0 << " ms" << std::endl;
    std::cout << "  [Detail] Coarse decode:          " << coarse_post_device_us_ / 1000.0 << " ms" << std::endl;
    std::cout << "  [Detail] Pass-1 launch:          " << pass1_launch_us / 1000.0 << " ms (fine programs built: "
              << programs_built << ")" << std::endl;
    if (two_pass)
        std::cout << "  [Detail] Pass-2 host prep (overlapped with the device, in TT fine): "
                  << overlapped_prep_us / 1000.0 << " ms" << std::endl;
    std::cout << "[Chrono] TT fine scan+merge:       " << std::fixed << std::setprecision(3) << device_time_us / 1000.0 << " ms" << std::endl;
    std::cout << "[Chrono] CPU output:               " << std::fixed << std::setprecision(3) << cpu_output_us / 1000.0 << " ms" << std::endl;
    if (options_.rerank) std::cout << "  [Detail] FP32 re-ranking:       " << rerank_us / 1000.0 << " ms" << std::endl;
    std::cout << "[Chrono] Stage sum:                " << std::fixed << std::setprecision(3) << chrono_stage_sum_us / 1000.0 << " ms" << std::endl;
    std::cout << "[Chrono] End-to-end:               " << std::fixed << std::setprecision(3) << total_time_us / 1000.0 << " ms" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "[Gather] tiles=" << gathered_tiles << "; page x tile pairs=" << total_pair_pages
              << "; list page reads=" << total_scan_pages << "; batches=" << num_batches
              << "; valid queries=" << total_queries << std::endl;

    std::cout << "[Host] DEVICE EXECUTION COMPLETED SUCCESSFULLY!" << std::endl;

    return results;
}

std::vector<uint32_t> IVF_TT::perform_coarse_search(
    const float* query_batch [[maybe_unused]], uint32_t total_queries) {
    const auto coarse_function_start = std::chrono::steady_clock::now();
    coarse_pre_device_us_ = 0.0;
    coarse_device_us_ = 0.0;
    coarse_post_device_us_ = 0.0;

    Program& program = coarse_workload_->get_programs().begin()->second;
    // The coarse buffers may be larger than this search needs (they persist).
    const uint32_t Ht = (total_queries + 31) / 32;
    const uint32_t Kt = ivf_cluster_major::coarse_k(nprobe_) / 32;  // output tiles per batch
    if (!total_queries || coarse_query_buffer_->size() < uint64_t(Ht) * 32 * ((dim_ + 31) / 32 * 32) * 2 ||
        coarse_out_val_buffer_->size() < uint64_t(Ht) * Kt * 2048 || coarse_out_ind_buffer_->size() < uint64_t(Ht) * Kt * 4096)
        throw std::invalid_argument("Valid query count does not fit the coarse buffers");

    tt::tt_metal::IDevice* device = mesh_device_->get_device(0);
    auto grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = grid.x * grid.y;
    uint32_t Ht_per_core = (Ht + num_cores - 1) / num_cores;

    uint32_t start_batch = 0;
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            CoreCoord core = {x, y};
            uint32_t Ht_core = std::min(Ht_per_core, Ht > start_batch ? Ht - start_batch : 0);

            SetRuntimeArgs(program, reader_coarse_id_, core, {coarse_query_buffer_->address(), centroids_buffer_->address(), coarse_indices_buffer_->address(), Ht_core, start_batch});
            SetRuntimeArgs(program, writer_coarse_id_, core, {coarse_out_val_buffer_->address(), coarse_out_ind_buffer_->address(), Kt, Ht_core, start_batch});
            SetRuntimeArgs(program, compute_coarse_id_, core, {Ht_core, start_batch});

            start_batch += Ht_core;
        }
    }

    auto device_start = std::chrono::steady_clock::now();
    coarse_pre_device_us_ = elapsed_us(coarse_function_start, device_start);
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *coarse_workload_, false);
    distributed::Finish(mesh_device_->mesh_command_queue());
    auto device_end = std::chrono::steady_clock::now();
    coarse_device_us_ = elapsed_us(device_start, device_end);

    auto transfer_start = std::chrono::steady_clock::now();
    const std::vector<bfloat16> result_vals =
        read_region<bfloat16>(mesh_device_->mesh_command_queue(), coarse_out_val_buffer_, size_t(Ht) * Kt * 1024);
    auto transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "device_to_host",
        "coarse_output_values",
        result_vals.size() * sizeof(bfloat16),
        elapsed_us(transfer_start, transfer_end),
        "blocking");

    transfer_start = std::chrono::steady_clock::now();
    const std::vector<uint32_t> result_inds =
        read_region<uint32_t>(mesh_device_->mesh_command_queue(), coarse_out_ind_buffer_, size_t(Ht) * Kt * 1024);
    transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "device_to_host",
        "coarse_output_indices",
        result_inds.size() * sizeof(uint32_t),
        elapsed_us(transfer_start, transfer_end),
        "blocking");

    // Output tile (batch h, i) is page h * Kt + i: ranks 32i..32i+31.
    const size_t expected_coarse_elements = size_t(Ht) * Kt * 32 * 32;
    if (result_vals.size() != expected_coarse_elements || result_inds.size() != expected_coarse_elements)
        throw std::runtime_error("Device coarse output buffer size differs from the expected top-K tiles");
    // Element (q, rank) read in place from the tiled output (no untilize):
    // tile (q / 32) * Kt + rank / 32, face layout inside the tile.
    auto element = [&](uint32_t q, uint32_t rank) {
        return (size_t(q / 32) * Kt + rank / 32) * 1024 + fine_tile_offset(q % 32, rank % 32);
    };

    // Every query's row is validated and decoded independently on a few
    // threads (spawning one per core costs more than the decode). A failing
    // row is described by the reference decoder; the lowest query ID is reported.
    coarse_selections_.assign(total_queries, std::vector<uint32_t>(nprobe_));
    std::vector<uint8_t> failed(total_queries, 0);
    constexpr unsigned decode_threads = 8;
    ivf_index_build::parallel_chunks(total_queries, decode_threads, [&](size_t begin, size_t end) {
        std::vector<uint32_t> stamp(nlist_, 0);
        uint32_t token = 0;
        auto decode = [&]<size_t K>(uint32_t q) {
            std::array<uint32_t, K> ids;
            std::array<float, K> scores;
            for (uint32_t rank = 0; rank < K; ++rank) {
                ids[rank] = result_inds[element(q, rank)];
                scores[rank] = static_cast<float>(result_vals[element(q, rank)]);
            }
            return ivf_cluster_major::decode_coarse_row_fast(
                ids, scores, nlist_, nprobe_, coarse_selections_[q].data(), stamp, ++token);
        };
        for (size_t q = begin; q < end; ++q)
            failed[q] = !(Kt == 2 ? decode.template operator()<64>(uint32_t(q)) : decode.template operator()<32>(uint32_t(q)));
    });
    for (uint32_t q = 0; q < total_queries; ++q) {
        if (!failed[q]) continue;
        auto describe = [&]<size_t K>() {
            std::array<uint32_t, K> ids;
            std::array<float, K> scores;
            for (uint32_t rank = 0; rank < K; ++rank) {
                ids[rank] = result_inds[element(q, rank)];
                scores[rank] = static_cast<float>(result_vals[element(q, rank)]);
            }
            return ivf_cluster_major::decode_coarse_row(ids, scores, nlist_, nprobe_).description();
        };
        throw std::runtime_error("Device coarse output is invalid for query " + std::to_string(q) + ": " +
                                 (Kt == 2 ? describe.template operator()<64>() : describe.template operator()<32>()));
    }
    std::vector<uint32_t> top_clusters;
    top_clusters.reserve(size_t(total_queries) * nprobe_);
    for (const auto& row : coarse_selections_) top_clusters.insert(top_clusters.end(), row.begin(), row.end());

    const auto coarse_function_end = std::chrono::steady_clock::now();
    coarse_post_device_us_ = elapsed_us(device_end, coarse_function_end);
    return top_clusters;
}
