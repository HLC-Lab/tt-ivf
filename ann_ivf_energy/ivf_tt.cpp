// ann_ivf_energy/ivf_tt.cpp
#include <thread>

#include "ivf_tt.hpp"
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
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tilize_utils.hpp>

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
    FineResultStaging result_staging,
    uint32_t cluster_chunk_blocks,
    ivf_energy::GroupingConfig query_grouping) :
    dim_(dim),
    nlist_(nlist),
    nprobe_(nprobe),
    result_staging_(result_staging),
    query_grouping_(std::move(query_grouping)) {
    ivf_energy::validate_grouping_config(query_grouping_);
    if (cluster_chunk_blocks != 0) throw std::invalid_argument("Cluster chunking was removed; use --cluster-chunk-blocks 0 or omit it");
    if (dim_ == 0) {
        throw std::invalid_argument("dim must be greater than zero");
    }
    if (nlist_ < 32 || nlist_ % 32 != 0) {
        throw std::invalid_argument("nlist must be at least 32 and divisible by 32");
    }
    if (nprobe_ == 0 || nprobe_ > 32) {
        throw std::invalid_argument("nprobe must be between 1 and the hardware coarse top-k limit of 32");
    }
    mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    auto grid = mesh_device_->compute_with_storage_grid_size();
    if (grid.x * grid.y > 64) {
        throw std::runtime_error(
            "ann_ivf_energy supports at most 64 Tensix cores because its "
            "fine-search scripts reserve one aggregator entry and 63 worker entries");
    }
    all_cores_ = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    std::set<CoreRange> worker_cores_set;
    if (grid.x == 1 && grid.y == 1) {
        worker_cores_set.insert(CoreRange({0, 0}, {0, 0}));
    } else {
        if (grid.x > 1) worker_cores_set.insert(CoreRange({1, 0}, {grid.x - 1, 0}));
        if (grid.y > 1) worker_cores_set.insert(CoreRange({0, 1}, {grid.x - 1, grid.y - 1}));
    }
    worker_cores_ = CoreRangeSet(worker_cores_set);
}

IVF_TT::~IVF_TT() {}

void IVF_TT::create_index(const std::vector<float>& dataset_vectors, const std::string& centroids_path) {
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
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), centroids_buffer_, centroids_bf, true);

    std::vector<uint32_t> coarse_indices_flat(32 * nlist_);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < nlist_; c++) {
            coarse_indices_flat[r * nlist_ + c] = c;
        }
    }
    std::vector<uint32_t> coarse_indices = tilize_nfaces(coarse_indices_flat, 32, nlist_);

    coarse_indices_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = coarse_indices.size() * 4}, generic_dram_config_u32, mesh_device_.get());
    distributed::EnqueueWriteMeshBuffer(
        mesh_device_->mesh_command_queue(), coarse_indices_buffer_, coarse_indices, true);

    dataset_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = dataset_host_padded_.size() * 2}, dataset_dram_config, mesh_device_.get());
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), dataset_buffer_, dataset_host_padded_, true);

    indices_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = index_map_host_padded_.size() * 4}, index_dram_config, mesh_device_.get());
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), indices_buffer_, index_map_host_padded_, true);
}

void IVF_TT::setup_coarse_program(uint32_t num_queries) {
    (void)num_queries;
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
    uint32_t K = 32; // Hardware topK size
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

    std::vector<uint32_t> reader_compile_args = { CB::c_in0, CB::c_in1, CB::c_in2, query_tiles_width, Wt };
    reader_coarse_id_ = CreateKernel(program, "ann_ivf_energy/kernels/dataflow/reader_coarse.cpp",
        all_cores, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_args});

    std::vector<uint32_t> writer_compile_args = { CB::c_out0, CB::c_out1 };
    writer_coarse_id_ = CreateKernel(program, "ann_ivf_energy/kernels/dataflow/writer_coarse.cpp",
        all_cores, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_args});

    std::vector<uint32_t> compute_compile_args = {
        CB::c_in0, CB::c_in1, CB::c_in2, CB::c_intermed1, CB::c_intermed2,
        CB::c_out0, CB::c_out1, Wt, K, Kt, logk, logWt, CB::c_intermed0, query_tiles_width
    };
    compute_coarse_id_ = CreateKernel(program, "ann_ivf_energy/kernels/compute/compute_coarse.cpp",
        all_cores, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compute_compile_args});

    program_coarse_temp_ = std::move(program);
}


void IVF_TT::setup_fine_program() {
    Program program{};

    uint32_t tile_size_bf16 = 2048;
    uint32_t tile_size_u32 = 4096;
    uint32_t query_tiles_width = (dim_ + 31) / 32;

    // Worker CBs
    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * query_tiles_width * tile_size_bf16, {{CB::c_in0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in0, tile_size_bf16));
    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * query_tiles_width * tile_size_bf16, {{CB::c_in1, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in1, tile_size_bf16));
    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * tile_size_u32, {{CB::c_in2, tt::DataFormat::Int32}}).set_page_size(CB::c_in2, tile_size_u32));

    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_intermed0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed0, tile_size_bf16));
    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_intermed3, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed3, tile_size_bf16));
    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * tile_size_u32, {{CB::c_intermed4, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed4, tile_size_u32));

    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_out0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_out0, tile_size_bf16));
    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(2 * tile_size_u32, {{CB::c_out1, tt::DataFormat::Int32}}).set_page_size(CB::c_out1, tile_size_u32));

    // Script buffer
    CreateCircularBuffer(program, worker_cores_, CircularBufferConfig(8192, {{CB::c_intermed5, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed5, 8192));

    // Aggregator CBs (Core 0,0)
    CreateCircularBuffer(program, core_0_0_, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_in0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in0, tile_size_bf16));
    CreateCircularBuffer(program, core_0_0_, CircularBufferConfig(2 * tile_size_u32, {{CB::c_in1, tt::DataFormat::Int32}}).set_page_size(CB::c_in1, tile_size_u32));
    CreateCircularBuffer(program, core_0_0_, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_intermed3, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed3, tile_size_bf16));
    CreateCircularBuffer(program, core_0_0_, CircularBufferConfig(2 * tile_size_u32, {{CB::c_intermed4, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed4, tile_size_u32));
    CreateCircularBuffer(program, core_0_0_, CircularBufferConfig(2 * tile_size_u32, {{CB::c_intermed5, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed5, tile_size_u32));
    CreateCircularBuffer(program, core_0_0_, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_out0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_out0, tile_size_bf16));
    CreateCircularBuffer(program, core_0_0_, CircularBufferConfig(2 * tile_size_u32, {{CB::c_out1, tt::DataFormat::Int32}}).set_page_size(CB::c_out1, tile_size_u32));

    // Create Semaphores
    done_sem_addr_ = CreateSemaphore(program, core_0_0_, 0);
    ready_sem_addr_ = CreateSemaphore(program, worker_cores_, 0);

    constexpr uint32_t max_workers = 63;
    if (result_staging_ == FineResultStaging::Dram) {
        // Preserve the original, faster result handoff: each batch/worker has
        // its own page and pages are interleaved across the DRAM banks.
        constexpr uint32_t max_query_batches = 10000 / 32 + 1;
        const uint32_t slot_val_size =
            max_query_batches * max_workers * tile_size_bf16;
        const uint32_t slot_ind_size =
            max_query_batches * max_workers * tile_size_u32;
        distributed::DeviceLocalBufferConfig slot_val_config{
            .page_size = tile_size_bf16, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig slot_ind_config{
            .page_size = tile_size_u32, .buffer_type = BufferType::DRAM};

        result_slots_l1_buf_.reset();
        result_slot_values_dram_buf_.reset();
        result_slot_indices_dram_buf_.reset();
        result_slot_values_dram_buf_ = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = slot_val_size},
            slot_val_config,
            mesh_device_.get());
        result_slot_indices_dram_buf_ = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = slot_ind_size},
            slot_ind_config,
            mesh_device_.get());
        std::cout << "[Host] Fine-result staging: interleaved DRAM (default)"
                  << std::endl;
    } else {
        // Experimental path retained for A/B tests. One reusable slot per
        // worker is resident on the aggregator core. The ready semaphore
        // prevents batch B+1 from overwriting batch B.
        const uint32_t slot_val_size = max_workers * tile_size_bf16;
        const uint32_t slot_ind_size = max_workers * tile_size_u32;
        const uint32_t result_slots_size = slot_val_size + slot_ind_size;
        const CoreRangeSet aggregator_grid(
            std::set<CoreRange>{CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})});
        const ShardSpecBuffer result_slots_shard(
            aggregator_grid,
            {1, result_slots_size / sizeof(uint32_t)},
            ShardOrientation::ROW_MAJOR,
            {1, result_slots_size / sizeof(uint32_t)},
            {1, 1});

        result_slot_values_dram_buf_.reset();
        result_slot_indices_dram_buf_.reset();
        result_slots_l1_buf_.reset();
        result_slots_l1_buf_ = CreateBuffer(ShardedBufferConfig{
            .device = mesh_device_->get_device(0),
            .size = result_slots_size,
            .page_size = result_slots_size,
            .buffer_type = BufferType::L1,
            .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_parameters = result_slots_shard});
        AssignGlobalBufferToProgram(result_slots_l1_buf_, program);
        std::cout << "[Host] Fine-result staging: experimental core-(0,0) L1 ("
                  << result_slots_size / 1024 << " KiB)" << std::endl;
    }

    distributed::DeviceLocalBufferConfig scripts_dram_config{.page_size = 8192, .buffer_type = BufferType::DRAM};
    scripts_dram_buf_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 64 * 8192}, scripts_dram_config, mesh_device_.get());

    std::vector<uint32_t> compile_args = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    reader_fine_id_ = CreateKernel(program, "ann_ivf_energy/kernels/dataflow/reader_fine.cpp",
        worker_cores_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = compile_args});
    const std::string writer_fine_path =
        result_staging_ == FineResultStaging::Dram
            ? "ann_ivf_energy/kernels/dataflow/writer_fine.cpp"
            : "ann_ivf_energy/kernels/dataflow/writer_fine_core0_l1.cpp";
    writer_fine_id_ = CreateKernel(program, writer_fine_path,
        worker_cores_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args});
    compute_fine_id_ = CreateKernel(program, "ann_ivf_energy/kernels/compute/compute_fine_search_reuse.cpp",
        worker_cores_, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compile_args});

    const std::string reader_final_sort_path =
        result_staging_ == FineResultStaging::Dram
            ? "ann_ivf_energy/kernels/dataflow/reader_final_sort.cpp"
            : "ann_ivf_energy/kernels/dataflow/reader_final_sort_core0_l1.cpp";
    reader_final_sort_id_ = CreateKernel(program, reader_final_sort_path,
        core_0_0_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    writer_final_sort_id_ = CreateKernel(program, "ann_ivf_energy/kernels/dataflow/writer_final_sort.cpp",
        core_0_0_, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    compute_final_sort_id_ = CreateKernel(program, "ann_ivf_energy/kernels/compute/compute_final_sort.cpp",
        core_0_0_, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});

    program_fine_temp_ = std::move(program);
}

void IVF_TT::search_all(const float* query_data [[maybe_unused]], uint32_t total_queries, uint32_t k) {
    if (query_data == nullptr || total_queries == 0) {
        throw std::invalid_argument("search_all requires at least one query");
    }
    if (total_queries > 10000) {
        throw std::invalid_argument("ann_ivf_energy currently supports at most 10000 queries per search");
    }
    if (k == 0 || k > 32) {
        throw std::invalid_argument("k must be between 1 and the hardware top-k limit of 32");
    }

    [[maybe_unused]] uint32_t padded_q = (total_queries + 31) / 32 * 32;
    (void)nprobe_;

    uint32_t num_batches = padded_q / 32;

    tt::tt_metal::IDevice* device = mesh_device_->get_device(0);
    auto grid = device->compute_with_storage_grid_size();
    constexpr uint32_t max_workers = 63;
    uint32_t num_workers = std::min<uint32_t>(max_workers, grid.x * grid.y > 0 ? grid.x * grid.y - 1 : 0);
    if (num_workers == 0) {
        throw std::runtime_error("ann_ivf_energy fine search requires at least one worker core in addition to the aggregator core");
    }


    uint32_t padded_dim = (dim_ + 31) / 32 * 32;
    distributed::DeviceLocalBufferConfig dram_bf16{.page_size = 2048, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig generic_dram_config_bf16{.page_size = 2048, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig generic_dram_config_u32{.page_size = 4096, .buffer_type = BufferType::DRAM};

    coarse_query_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = padded_q * padded_dim * 2}, dram_bf16, mesh_device_.get());
    uint32_t Kt_coarse = 1;
    coarse_out_val_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = num_batches * Kt_coarse * 2048}, generic_dram_config_bf16, mesh_device_.get());
    coarse_out_ind_buffer_ = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = num_batches * Kt_coarse * 4096}, generic_dram_config_u32, mesh_device_.get());

    setup_coarse_program(padded_q);

    std::vector<float> query_vec(padded_q * padded_dim, 0.0f);
    for (uint32_t i = 0; i < total_queries; i++) {
        std::memcpy(query_vec.data() + i * padded_dim, query_data + i * dim_, dim_ * sizeof(float));
    }
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


    normalize_l2(query_vec.data(), padded_q, padded_dim);

    std::vector<bfloat16> query_vec_bf;
    for (float v : query_vec) query_vec_bf.push_back(bfloat16(v));

    std::vector<bfloat16> query_bf = tilize_nfaces(query_vec_bf, padded_q, padded_dim);
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), coarse_query_buffer_, query_bf, true);

    std::vector<uint32_t> top_clusters = perform_coarse_search(query_data, total_queries);

    const auto grouping_start = std::chrono::steady_clock::now();
    std::vector<uint32_t> list_pages(nlist_, 0);
    for (uint32_t cid = 0; cid < nlist_; ++cid) {
        const auto& toc = cluster_toc_.at(cid);
        if (toc.count) list_pages[cid] = toc.padded_count / 32;
    }
    const auto grouping = ivf_energy::group_queries(
        coarse_selections_, list_pages, query_grouping_.mode,
        query_grouping_.window, query_grouping_.lookahead);
    const double grouping_us = elapsed_us(grouping_start, std::chrono::steady_clock::now());
    // Device coarse search is complete. Reuse its query buffer for fine search.
    // Copy the same normalized BF16 rows; only their batch membership changes.
    if (!ivf_energy::is_identity(grouping)) {
        const auto packed_rows = ivf_energy::pack_query_rows(query_vec_bf, padded_dim, grouping);
        const auto packed_tiles = tilize_nfaces(packed_rows, padded_q, padded_dim);
        distributed::EnqueueWriteMeshBuffer(
            mesh_device_->mesh_command_queue(), coarse_query_buffer_, packed_tiles, true);
    }
    cluster_query_masks_ = ivf_energy::make_batch_masks(coarse_selections_, grouping, nlist_);
    std::cout << "[Grouping] " << query_grouping_.mode << ": union pages "
              << grouping.original_pages << " -> " << grouping.grouped_pages
              << "; batches=" << num_batches << "; valid queries=" << total_queries
              << "; grouping_ms=" << grouping_us / 1000.0 << std::endl;

    std::vector<std::vector<uint32_t>> worker_scripts(num_workers);
    std::vector<uint32_t> aggregator_script;
    aggregator_script.push_back(num_batches);

    std::vector<uint32_t> noc_x(64, 0);
    std::vector<uint32_t> noc_y(64, 0);

    for(uint32_t y = 0; y < grid.y; y++) {
        for(uint32_t x = 0; x < grid.x; x++) {
            uint32_t core_id = y * grid.x + x;
            if (core_id < 64) {
                CoreCoord core = {x, y};
                CoreCoord physical = device->worker_core_from_logical_core(core);
                noc_x[core_id] = physical.x;
                noc_y[core_id] = physical.y;
            }
        }
    }

    for(uint32_t i = 0; i < 64; i++) aggregator_script.push_back(noc_x[i]);
    for(uint32_t i = 0; i < 64; i++) aggregator_script.push_back(noc_y[i]);

    aggregator_script.push_back(num_workers);

    uint32_t num_mask_words = (num_batches + 31) / 32;
    std::unordered_set<uint32_t> unique_clusters(top_clusters.begin(), top_clusters.end());
    std::vector<uint32_t> clusters_to_search(unique_clusters.begin(), unique_clusters.end());

    struct ClusterTask {
        uint32_t cid;
        uint32_t start_block;
        uint32_t actual_count;
        uint32_t padded_count;
        uint32_t cluster_blocks;
        uint32_t active_batch_count;
        uint64_t weight;
        std::vector<uint32_t> active_mask;
    };

    std::vector<ClusterTask> cluster_tasks;
    cluster_tasks.reserve(clusters_to_search.size());

    for (uint32_t cid : clusters_to_search) {
        const auto& toc = cluster_toc_[cid];
        std::vector<uint32_t> active_mask(num_mask_words, 0);
        uint32_t active_batch_count = 0;

        for (uint32_t b = 0; b < num_batches; b++) {
            bool is_active = (cluster_query_masks_[cid * num_mask_words + (b / 32)] & (1u << (b % 32))) != 0;
            if (is_active) {
                active_mask[b / 32] |= (1u << (b % 32));
                active_batch_count++;
            }
        }

        if (active_batch_count == 0 || toc.count == 0 || toc.padded_count == 0) {
            continue;
        }

        const uint32_t cluster_blocks = toc.padded_count / 32;
        const uint64_t weight = static_cast<uint64_t>(cluster_blocks) * active_batch_count;
        cluster_tasks.push_back({cid, toc.start_block, toc.count, toc.padded_count,
                                 cluster_blocks, active_batch_count, weight, active_mask});
    }

    // Longest-processing-time-first (LPT) on whole inverted lists. Work is
    // estimated as vector blocks multiplied by active query batches, and the
    // placement objective minimizes the worst per-batch worker load first.
    std::sort(cluster_tasks.begin(), cluster_tasks.end(), [](const ClusterTask& a, const ClusterTask& b) {
        if (a.weight != b.weight) return a.weight > b.weight;
        if (a.cluster_blocks != b.cluster_blocks) return a.cluster_blocks > b.cluster_blocks;
        return a.cid < b.cid;
    });

    std::vector<std::vector<ClusterTask>> worker_tasks(num_workers);
    std::vector<uint64_t> worker_weights(num_workers, 0);
    std::vector<std::vector<uint64_t>> worker_batch_load(
        num_workers, std::vector<uint64_t>(num_batches, 0));

    for (auto& task : cluster_tasks) {
        uint32_t best_worker = 0;
        uint64_t best_peak = std::numeric_limits<uint64_t>::max();
        uint64_t best_total = std::numeric_limits<uint64_t>::max();

        for (uint32_t w = 0; w < num_workers; w++) {
            uint64_t candidate_peak = 0;
            for (uint32_t b = 0; b < num_batches; b++) {
                const bool active =
                    (task.active_mask[b / 32] & (1u << (b % 32))) != 0;
                const uint64_t candidate_load =
                    worker_batch_load[w][b] + (active ? task.cluster_blocks : 0);
                candidate_peak = std::max(candidate_peak, candidate_load);
            }
            const uint64_t candidate_total = worker_weights[w] + task.weight;
            if (candidate_peak < best_peak ||
                (candidate_peak == best_peak && candidate_total < best_total)) {
                best_worker = w;
                best_peak = candidate_peak;
                best_total = candidate_total;
            }
        }

        for (uint32_t b = 0; b < num_batches; b++) {
            if ((task.active_mask[b / 32] & (1u << (b % 32))) != 0) {
                worker_batch_load[best_worker][b] += task.cluster_blocks;
            }
        }
        worker_weights[best_worker] += task.weight;
        worker_tasks[best_worker].push_back(std::move(task));
    }

    const auto [min_worker_weight, max_worker_weight] =
        std::minmax_element(worker_weights.begin(), worker_weights.end());
    uint64_t total_worker_weight = 0;
    for (uint64_t weight : worker_weights) {
        total_worker_weight += weight;
    }
    const double average_worker_weight =
        static_cast<double>(total_worker_weight) / static_cast<double>(num_workers);
    const double max_to_average =
        average_worker_weight > 0.0 ? static_cast<double>(*max_worker_weight) / average_worker_weight : 0.0;
    if (total_worker_weight != grouping.grouped_pages)
        throw std::runtime_error("Scheduled pages differ from the grouped batch unions");
    std::cout << "[Host] Batch-aware LPT assignment: whole lists";
    std::cout
              << ", tasks=" << cluster_tasks.size()
              << ", workers=" << num_workers
              << ", load[min/avg/max]=" << *min_worker_weight << "/"
              << std::fixed << std::setprecision(1) << average_worker_weight << "/"
              << *max_worker_weight
              << ", max/avg=" << std::setprecision(3) << max_to_average << "x"
              << std::defaultfloat << std::setprecision(6) << std::endl;

    for (uint32_t w = 0; w < num_workers; w++) {
        worker_scripts[w].push_back(num_batches);
        uint32_t num_assigned = static_cast<uint32_t>(worker_tasks[w].size());
        worker_scripts[w].push_back(num_assigned);

        for (const auto& task : worker_tasks[w]) {
            worker_scripts[w].push_back(task.cid);
            worker_scripts[w].push_back(task.start_block);
            worker_scripts[w].push_back(task.actual_count);
            worker_scripts[w].push_back(task.padded_count);
            worker_scripts[w].insert(worker_scripts[w].end(), task.active_mask.begin(), task.active_mask.end());
        }
    }


    std::cout << "[Host] Packing scripts complete. Allocating L1 buffers..." << std::endl;
    uint32_t w_idx = 0;

    std::cout << "[Host] Checking if fine program needs setup..." << std::endl;
    setup_fine_program();

    std::vector<uint32_t> host_scripts(64 * 8192 / 4, 0);
    for (uint32_t w = 0; w < num_workers; w++) {
        if (worker_scripts[w].size() > 8192 / sizeof(uint32_t)) {
            throw std::runtime_error(
                "Worker " + std::to_string(w) + " script exceeds its 8192-byte DRAM page");
        }
        for (size_t i = 0; i < worker_scripts[w].size(); i++) {
            host_scripts[w * (8192 / 4) + i] = worker_scripts[w][i];
        }
    }
    distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), scripts_dram_buf_, host_scripts, true);

    std::cout << "[Host] Program is ready. Setting Runtime Arguments..." << std::endl;
    Program& program = program_fine_temp_.value();

    uint32_t result_slot_values_base = 0;
    uint32_t result_slot_indices_base = 0;
    if (result_staging_ == FineResultStaging::Dram) {
        result_slot_values_base = result_slot_values_dram_buf_->address();
        result_slot_indices_base = result_slot_indices_dram_buf_->address();
    } else {
        result_slot_values_base = result_slots_l1_buf_->address();
        result_slot_indices_base =
            result_slots_l1_buf_->address() + max_workers * 2048;
    }

    auto dummy_query = coarse_query_buffer_;


    CoreCoord core_0_physical = device->worker_core_from_logical_core({0, 0});
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            CoreCoord core = {x, y};
            if (x == 0 && y == 0) continue; // Aggregator
            if (w_idx >= num_workers) break;

            std::vector<uint32_t> reader_args = {
                dummy_query->address(),
                dataset_buffer_->address(),
                indices_buffer_->address(),
                padded_dim,
                scripts_dram_buf_->address(),
                w_idx
            };
            SetRuntimeArgs(program, reader_fine_id_, core, reader_args);

            std::vector<uint32_t> writer_args = {
                w_idx,
                (uint32_t)core_0_physical.x,
                (uint32_t)core_0_physical.y,
                result_slot_values_base,
                result_slot_indices_base,
                done_sem_addr_,
                ready_sem_addr_,
                num_batches
            };
            SetRuntimeArgs(program, writer_fine_id_, core, writer_args);

            std::vector<uint32_t> compute_args = {
                (dim_ + 31) / 32
            };
            SetRuntimeArgs(program, compute_fine_id_, core, compute_args);
            w_idx++;
        }
    }

    std::vector<uint32_t> agg_reader_args = {
        result_slot_values_base,
        result_slot_indices_base,
        done_sem_addr_,
        ready_sem_addr_
    };
    CoreCoord core_0 = {0, 0};
    agg_reader_args.insert(agg_reader_args.end(), aggregator_script.begin(), aggregator_script.end());
    SetRuntimeArgs(program, reader_final_sort_id_, core_0, agg_reader_args);

    distributed::DeviceLocalBufferConfig dram_val_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig dram_ind_config{.page_size = 4096, .buffer_type = BufferType::DRAM};
    auto dummy_val = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = padded_q * 32 * 2}, dram_val_config, mesh_device_.get());
    auto dummy_ind = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = padded_q * 32 * 4}, dram_ind_config, mesh_device_.get());

    std::vector<uint32_t> agg_writer_args = {
        dummy_val->address(),
        dummy_ind->address()
    };
    agg_writer_args.insert(agg_writer_args.end(), aggregator_script.begin(), aggregator_script.end());
    SetRuntimeArgs(program, writer_final_sort_id_, core_0, agg_writer_args);

    std::vector<uint32_t> agg_compute_args;
    agg_compute_args.insert(agg_compute_args.end(), aggregator_script.begin(), aggregator_script.end());
    SetRuntimeArgs(program, compute_final_sort_id_, core_0, agg_compute_args);

    distributed::MeshWorkload fine_workload;
    fine_workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0,0)), std::move(program_fine_temp_.value()));

    std::cout << "[Host] Enqueuing workload to the Mesh Command Queue..." << std::endl;
    auto start_device = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), fine_workload, false);

    std::cout << "[Host] Workload enqueued. Blocking on distributed::Finish() to wait for device..." << std::endl;
    distributed::Finish(mesh_device_->mesh_command_queue());
    auto end_device = std::chrono::steady_clock::now();

    const double device_time_us = elapsed_us(start_device, end_device);
    std::cout << "[Energy] Fine device execution: " << std::fixed << std::setprecision(3)
              << device_time_us / 1000.0 << " ms" << std::endl;
}

std::vector<uint32_t> IVF_TT::perform_coarse_search(const float* query_batch [[maybe_unused]], uint32_t total_queries) {
    Program& program = program_coarse_temp_.value();
    uint32_t Ht = coarse_query_buffer_->size() / (2 * ((dim_ + 31) / 32 * 32)) / 32;
    constexpr uint32_t coarse_k = 32;
    constexpr uint32_t Kt = 1;

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

    distributed::MeshWorkload coarse_workload;
    coarse_workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0,0)), std::move(program_coarse_temp_.value()));

    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), coarse_workload, false);
    distributed::Finish(mesh_device_->mesh_command_queue());

    std::vector<bfloat16> result_vals;
    std::vector<uint32_t> result_inds;

    distributed::EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), result_vals, coarse_out_val_buffer_, true);
    distributed::EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), result_inds, coarse_out_ind_buffer_, true);

    std::vector<uint32_t> utilzed_inds_u32 = untilize_nfaces(result_inds, 32 * Ht, coarse_k);
    std::vector<bfloat16> utilzed_vals_bf16 = untilize_nfaces(result_vals, 32 * Ht, coarse_k);

    std::vector<uint32_t> top_clusters;
    if (total_queries > Ht * 32 || utilzed_inds_u32.size() < static_cast<size_t>(Ht) * 1024 ||
        utilzed_vals_bf16.size() < static_cast<size_t>(Ht) * 1024)
        throw std::runtime_error("Coarse result buffer is smaller than the requested query count");
    coarse_selections_.clear();
    coarse_selections_.reserve(total_queries);
    // Padded query rows must never activate lists for the final partial batch.
    for (uint32_t q = 0; q < total_queries; ++q) {
        std::array<uint32_t, 32> ids;
        std::array<float, 32> scores;
        for (uint32_t rank = 0; rank < coarse_k; ++rank) {
            ids[rank] = utilzed_inds_u32[q * coarse_k + rank];
            scores[rank] = static_cast<float>(utilzed_vals_bf16[q * coarse_k + rank]);
        }
        auto decoded = ivf_energy::decode_coarse_row(ids, scores, nlist_, nprobe_);
        if (!decoded.complete())
            throw std::runtime_error("Invalid device coarse row " + std::to_string(q) + ": " + decoded.description());
        top_clusters.insert(top_clusters.end(), decoded.selected.begin(), decoded.selected.end());
        coarse_selections_.push_back(std::move(decoded.selected));
    }

    return top_clusters;
}
