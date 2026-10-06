// Index construction and coarse search are derived from ann_ivf_mem_log.
#include <thread>

#include "ivf_tt.hpp"
#include "partition_schedule.hpp"
#include "partition_l1.hpp"
#include "coarse_selection.hpp"
#include "tt-metalium/hal.hpp"
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
#include <filesystem>
#include <map>
#include <sstream>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

namespace {

template <typename Clock>
double elapsed_us(
    const std::chrono::time_point<Clock>& start, const std::chrono::time_point<Clock>& end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

}  // namespace

IVF_TT::IVF_TT(uint32_t dim, uint32_t nlist, uint32_t nprobe, const std::string& log,
               const PartitionConfig& config) :
    dim_(dim), nlist_(nlist), nprobe_(nprobe), config_(config),
    memory_logger_(std::make_unique<MemoryTransferLogger>(log)) {
    if (!dim || dim > 8192 || nlist < 32 || nlist % 32 || nlist > 2048 || !nprobe || nprobe > 32)
        throw std::invalid_argument("Require 1<=D<=8192, N_list a multiple of 32 in [32,2048], N_probe in [1,32]");
    if (!config.worker_input || !config.leader_prefetch || config.leader_prefetch > 15)
        throw std::invalid_argument("Worker input depth must be positive; leader prefetch must be in [1,15]");
    if (config.mask != ivf_partitioned::MaskMode::Auto && config.mask != ivf_partitioned::MaskMode::Additive &&
        config.mask != ivf_partitioned::MaskMode::Off)
        throw std::invalid_argument("Invalid candidate mask mode");
    if ((config.reader != "relay" && config.reader != "direct" && config.reader != "direct-serial") ||
        (config.bank_schedule != "fifo" && config.bank_schedule != "staggered") ||
        (config.aggregation != "leader" && config.aggregation != "separate") ||
        (config.query_broadcast != "auto" && config.query_broadcast != "unicast"))
        throw std::invalid_argument("Invalid reader, bank schedule, aggregation, or query broadcast mode");
    if (config.reader != "relay" && config.bank_schedule != "fifo")
        throw std::invalid_argument("Direct readers require --bank-schedule fifo; staggering is a relay scheduling option");
    mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    auto* device = mesh_device_->get_device(0);
    if (device->arch() != tt::ARCH::WORMHOLE_B0) throw std::runtime_error("This example requires Wormhole B0");
    const auto grid = device->compute_with_storage_grid_size();
    ivf_partitioned::make_partitions(grid.x, grid.y, config.partitions, config.layout, config.aggregation == "separate");
    const uint64_t l1_size = device->l1_size_per_core(), l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    if (l1_base >= l1_size) throw std::runtime_error("Device has no allocatable worker L1");
    const uint64_t available = l1_size - l1_base;
    if (ivf_partitioned::coarse_cb_bytes(dim_) > std::min(available, ivf_partitioned::working_set_cap))
        throw std::invalid_argument("Coarse-query working set exceeds allocatable L1 or the 512 KiB cap");
}
IVF_TT::~IVF_TT() { if (mesh_device_) mesh_device_->close(); }

void IVF_TT::create_index(const std::vector<float>& dataset_vectors, const std::string& centroids_path) {
    if (dataset_vectors.empty() || dataset_vectors.size() % dim_ != 0) {
        throw std::invalid_argument("Dataset must contain a non-zero whole number of vectors");
    }

    size_t num_vectors = dataset_vectors.size() / dim_;
    if (num_vectors >= ivf_partitioned::invalid_id) throw std::invalid_argument("Vector IDs must fit below the int32 sentinel");
    database_size_ = static_cast<uint32_t>(num_vectors);
    std::ifstream file(centroids_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open centroids file: " + centroids_path);

    uint32_t header[2]{};
    if (!file.read(reinterpret_cast<char*>(header), 8)) throw std::runtime_error("Truncated centroid header");
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
    if (!file.read(reinterpret_cast<char*>(centroids_host_.data()), n_centroids * dim_ * sizeof(float)))
        throw std::runtime_error("Truncated centroid payload");

    auto normalize_l2 = [](float* data, uint32_t num_vectors, uint32_t dim) {
        for (uint32_t i = 0; i < num_vectors; ++i) {
            double sum = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                if (!std::isfinite(data[size_t(i) * dim + d])) throw std::invalid_argument("Non-finite vector component");
                const double component = data[size_t(i) * dim + d];
                sum += component * component;
            }
            double norm = std::sqrt(sum);
            if (norm > 0) {
                for (uint32_t d = 0; d < dim; ++d) {
                    data[size_t(i) * dim + d] /= norm;
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
                    score += dataset_norm[size_t(i) * dim_ + d] * centroids_host_[c * dim_ + d];
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
        if (cluster_blocks < 1) cluster_blocks = 1; // Storage only: empty lists are never scheduled.
        uint32_t padded_count = cluster_blocks * 32;

        cluster_toc_[i] = {current_page_offset, count, padded_count, current_page_offset};

        std::vector<bfloat16> cluster_flat;
        for (uint32_t d = 0; d < padded_dim; ++d) {
            for (uint32_t v = 0; v < padded_count; ++v) {
                if (v < count) {
                    uint32_t idx = cluster_assignments[i][v];
                    if (d < dim_) cluster_flat.push_back(bfloat16(dataset_norm[size_t(idx) * dim_ + d]));
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
    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * query_tiles_width * tile_size_bf16, {{CB::c_in1, tt::DataFormat::Float16_b}}).set_page_size(CB::c_in1, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_u32, {{CB::c_in2, tt::DataFormat::Int32}}).set_page_size(CB::c_in2, tile_size_u32));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * tile_size_bf16, {{CB::c_intermed0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed0, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * Kt * tile_size_bf16, {{CB::c_intermed1, tt::DataFormat::Float16_b}}).set_page_size(CB::c_intermed1, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * Kt * tile_size_u32, {{CB::c_intermed2, tt::DataFormat::Int32}}).set_page_size(CB::c_intermed2, tile_size_u32));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * Kt * tile_size_bf16, {{CB::c_out0, tt::DataFormat::Float16_b}}).set_page_size(CB::c_out0, tile_size_bf16));

    CreateCircularBuffer(program, all_cores, CircularBufferConfig(2 * Kt * tile_size_u32, {{CB::c_out1, tt::DataFormat::Int32}}).set_page_size(CB::c_out1, tile_size_u32));

    std::vector<uint32_t> reader_compile_args = { CB::c_in0, CB::c_in1, CB::c_in2, query_tiles_width, Wt };
    reader_coarse_id_ = CreateKernel(program, "ann_ivf_similarity_partitioned/kernels/dataflow/reader_coarse.cpp",
        all_cores, DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_args});

    std::vector<uint32_t> writer_compile_args = { CB::c_out0, CB::c_out1 };
    writer_coarse_id_ = CreateKernel(program, "ann_ivf_similarity_partitioned/kernels/dataflow/writer_coarse.cpp",
        all_cores, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_args});

    std::vector<uint32_t> compute_compile_args = {
        CB::c_in0, CB::c_in1, CB::c_in2, CB::c_intermed1, CB::c_intermed2,
        CB::c_out0, CB::c_out1, Wt, K, Kt, logk, logWt, CB::c_intermed0, query_tiles_width
    };
    compute_coarse_id_ = CreateKernel(program, "ann_ivf_similarity_partitioned/kernels/compute/compute_coarse.cpp",
        all_cores, ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = compute_compile_args});

    program_coarse_temp_ = std::move(program);
}

ivf_partitioned::Selections IVF_TT::perform_coarse_search(uint32_t valid_queries) {
    const auto coarse_function_start = std::chrono::steady_clock::now();
    coarse_pre_device_us_ = 0.0;
    coarse_device_us_ = 0.0;

    Program& program = program_coarse_temp_.value();
    uint32_t Ht = coarse_query_buffer_->size() / (2 * ((dim_ + 31) / 32 * 32)) / 32;
    uint32_t Kt = (32 + 31) / 32;

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

    auto device_start = std::chrono::steady_clock::now();
    coarse_pre_device_us_ = elapsed_us(coarse_function_start, device_start);
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), coarse_workload, false);
    distributed::Finish(mesh_device_->mesh_command_queue());
    auto device_end = std::chrono::steady_clock::now();
    coarse_device_us_ = elapsed_us(device_start, device_end);

    std::vector<bfloat16> result_vals;
    std::vector<uint32_t> result_inds;

    auto transfer_start = std::chrono::steady_clock::now();
    distributed::EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), result_vals, coarse_out_val_buffer_, true);
    auto transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "device_to_host",
        "coarse_output_values",
        result_vals.size() * sizeof(bfloat16),
        elapsed_us(transfer_start, transfer_end),
        "blocking");

    transfer_start = std::chrono::steady_clock::now();
    distributed::EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), result_inds, coarse_out_ind_buffer_, true);
    transfer_end = std::chrono::steady_clock::now();
    memory_logger_->record(
        "pcie",
        "device_to_host",
        "coarse_output_indices",
        result_inds.size() * sizeof(uint32_t),
        elapsed_us(transfer_start, transfer_end),
        "blocking");

    const auto decode_start = std::chrono::steady_clock::now();
    last_timings_.coarse_readback_us = elapsed_us(device_end, decode_start);
    const size_t expected_coarse_elements = size_t(Ht) * 32 * 32;
    if (result_vals.size() != expected_coarse_elements || result_inds.size() != expected_coarse_elements)
        throw std::runtime_error("Device coarse output buffer size differs from the expected top-32 tiles");
    std::vector<uint32_t> utilzed_inds_u32 = untilize_nfaces(result_inds, 32 * Ht, 32);
    std::vector<bfloat16> utilzed_vals_bf16 = untilize_nfaces(result_vals, 32 * Ht, 32);

    ivf_partitioned::Selections selections(valid_queries);
    for (uint32_t q = 0; q < valid_queries; ++q) {
        std::array<uint32_t, 32> ids;
        std::array<float, 32> scores;
        for (uint32_t i = 0; i < 32; ++i) {
            ids[i] = utilzed_inds_u32[q * 32 + i];
            scores[i] = static_cast<float>(utilzed_vals_bf16[q * 32 + i]);
        }
        auto row = ivf_partitioned::decode_coarse_row(ids, scores, nlist_, nprobe_);
        if (!row.complete()) {
            // Error-only diagnostics; the failed invocation never publishes QPS.
            std::ostringstream message;
            const uint32_t coarse_batch = q / 32, core = coarse_batch / Ht_per_core;
            message << "Device coarse output is invalid: context=" << context_ << ", query=" << q
                    << ", batch=" << coarse_batch << ", lane=" << q % 32
                    << ", logical_core=(" << core % grid.x << ',' << core / grid.x << "), nprobe=" << nprobe_
                    << ", " << row.description();
            if (!config_.diagnostics.empty()) {
                std::filesystem::create_directories(config_.diagnostics);
                const auto path = std::filesystem::path(config_.diagnostics) / ("coarse_failure_" + context_ + ".csv");
                std::ofstream failure(path);
                if (!failure) throw std::runtime_error(message.str() + "; could not open " + path.string());
                failure << "context,nlist,nprobe,query,batch,lane,rank,index,score,valid_id,finite_score\n" << std::setprecision(9);
                for (uint32_t rank = 0; rank < 32; ++rank)
                    failure << context_ << ',' << nlist_ << ',' << nprobe_ << ',' << q << ',' << coarse_batch << ',' << q % 32
                            << ',' << rank << ',' << ids[rank] << ',' << scores[rank] << ',' << (ids[rank] < nlist_)
                            << ',' << std::isfinite(scores[rank]) << '\n';
                failure.close();
                if (!failure) throw std::runtime_error(message.str() + "; could not finish " + path.string());
                message << "; raw row: " << path;
            }
            message << "; raw IDs:";
            for (auto id : ids) message << ' ' << id;
            throw std::runtime_error(message.str());
        }
        selections[q] = std::move(row.selected);
    }

    last_timings_.coarse_decode_us = elapsed_us(decode_start, std::chrono::steady_clock::now());
    return selections;
}

std::pair<std::vector<float>, std::vector<int64_t>> IVF_TT::search_all(
    const float* query_data, uint32_t queries, uint32_t k) {
    using namespace ivf_partitioned;
    if (!query_data || !queries || queries > 10000 || !k || k > 32 || !database_size_)
        throw std::invalid_argument("Require an uploaded index, 1..10000 queries and k in [1,32]");
    auto* device = mesh_device_->get_device(0);
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t width = (dim_ + 31) / 32, padded_dim = width * 32;
    const uint32_t batch_count = (queries + 31) / 32, padded_queries = batch_count * 32;
    last_timings_ = {};
    const auto start = std::chrono::steady_clock::now();
    auto dram_buffer = [&](uint64_t bytes, uint32_t page) {
        return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{.page_size = page, .buffer_type = BufferType::DRAM}, mesh_device_.get());
    };
    auto write = [&](auto& buffer, const auto& payload, const char* name) {
        const auto begin = std::chrono::steady_clock::now();
        distributed::EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), buffer, payload, true);
        const auto end = std::chrono::steady_clock::now();
        memory_logger_->record("pcie", "host_to_device", name, payload.size() * sizeof(payload[0]), elapsed_us(begin, end), "blocking");
        return elapsed_us(begin, end);
    };
    const auto normalize_begin = std::chrono::steady_clock::now();
    std::vector<bfloat16> normalized(padded_queries * padded_dim, bfloat16(0.0f));
    for (uint32_t q = 0; q < queries; ++q) {
        double norm = 0;
        for (uint32_t d = 0; d < dim_; ++d) {
            const double value = query_data[q * dim_ + d];
            if (!std::isfinite(value)) throw std::invalid_argument("Non-finite query component");
            norm += value * value;
        }
        norm = std::sqrt(norm);
        for (uint32_t d = 0; d < dim_; ++d)
            normalized[q * padded_dim + d] = bfloat16(norm ? query_data[q * dim_ + d] / norm : 0.0);
    }
    last_timings_.query_normalize_us = elapsed_us(normalize_begin, std::chrono::steady_clock::now());
    coarse_query_buffer_ = dram_buffer(uint64_t(padded_queries) * padded_dim * 2, 2048);
    coarse_out_val_buffer_ = dram_buffer(uint64_t(batch_count) * 2048, 2048);
    coarse_out_ind_buffer_ = dram_buffer(uint64_t(batch_count) * 4096, 4096);
    const auto original_tiles = tilize_nfaces(normalized, padded_queries, padded_dim);
    const double coarse_h2d_us = write(coarse_query_buffer_, original_tiles, "coarse_query_tiles");
    last_timings_.coarse_query_h2d_us = coarse_h2d_us;
    double h2d_us = coarse_h2d_us;
    setup_coarse_program(padded_queries);
    const auto coarse_begin = std::chrono::steady_clock::now();
    const auto selections = perform_coarse_search(queries);
    const auto metadata_begin = std::chrono::steady_clock::now();
    std::vector<List> lists(nlist_);
    std::vector<uint32_t> list_pages(nlist_);
    for (uint32_t id = 0; id < nlist_; ++id) {
        const auto& toc = cluster_toc_.at(id);
        lists[id] = {toc.offset, toc.count}; list_pages[id] = pages_for(toc.count);
    }
    const auto grouping_begin = std::chrono::steady_clock::now();
    last_timings_.list_metadata_us = elapsed_us(metadata_begin, grouping_begin);
    const auto grouping = group_queries(selections, list_pages, config_.grouping,
        config_.grouping_window, config_.grouping_lookahead);
    const auto grouping_end = std::chrono::steady_clock::now();
    last_timings_.query_grouping_us = elapsed_us(grouping_begin, grouping_end);
    std::vector<bfloat16> packed(normalized.size(), bfloat16(0.0f));
    for (uint32_t q = 0; q < queries; ++q)
        std::copy_n(normalized.begin() + grouping.packed_to_original[q] * padded_dim,
                    padded_dim, packed.begin() + q * padded_dim);
    const auto packed_tiles = tilize_nfaces(packed, padded_queries, padded_dim);
    last_timings_.query_reorder_us = elapsed_us(grouping_end, std::chrono::steady_clock::now());
    auto fine_queries = dram_buffer(uint64_t(padded_queries) * padded_dim * 2, width * 2048);
    last_timings_.fine_query_h2d_us = write(fine_queries, packed_tiles, "grouped_fine_query_pages");
    h2d_us += last_timings_.fine_query_h2d_us;
    const auto schedule_begin = std::chrono::steady_clock::now();
    auto partitions = make_partitions(grid.x, grid.y, config_.partitions, config_.layout, config_.aggregation == "separate");
    const auto batches = schedule_batches(selections, grouping, lists, partitions, config_.mask);
    last_timings_.partition_schedule_us = elapsed_us(schedule_begin, std::chrono::steady_clock::now());
    uint32_t total_workers = 0, max_tasks = 0;
    std::vector<Core> worker_coords, leader_coords, aggregator_coords;
    std::vector<std::vector<uint32_t>> worker_slots(partitions.size());
    for (uint32_t p = 0; p < partitions.size(); ++p) {
        leader_coords.push_back(partitions[p].leader); aggregator_coords.push_back(partitions[p].aggregator);
        for (const auto core : partitions[p].workers) {
            worker_coords.push_back(core); worker_slots[p].push_back(total_workers++);
        }
    }
    for (const auto& b : batches) max_tasks = std::max<uint32_t>(max_tasks, b.lists.size());
    auto core_set = [](const std::vector<Core>& coords) {
        std::set<CoreRange> ranges;
        for (const auto core : coords) ranges.emplace(CoreCoord{core.x, core.y}, CoreCoord{core.x, core.y});
        return CoreRangeSet(ranges);
    };
    const auto workers_set = core_set(worker_coords), leaders_set = core_set(leader_coords), agg_set = core_set(aggregator_coords);
    const auto all_set = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint64_t l1_size = device->l1_size_per_core(), l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    if (l1_base >= l1_size) throw std::runtime_error("Device has no allocatable worker L1");
    const uint64_t available = l1_size - l1_base;
    const uint32_t relay_depth = config_.reader == "relay" ? config_.leader_prefetch : 0;
    const uint64_t leader_size = leader_raw_bytes(dim_, relay_depth, max_tasks);
    const uint64_t worker_size = role_control_bytes + worker_cb_bytes(dim_, config_.worker_input, config_.mask);
    const uint64_t leader_total = leader_size + (config_.aggregation == "leader" ? aggregator_cb_bytes : 0);
    const uint64_t aggregate_total = role_control_bytes + aggregator_cb_bytes;
    if (std::max({worker_size, leader_total, aggregate_total}) > std::min(available, working_set_cap))
        throw std::invalid_argument("Requested topology/prefetch exceeds the 512 KiB cap or runtime allocatable L1");
    std::cout << "[Host] Union pages " << grouping.original_pages << " -> " << grouping.grouped_pages
              << (grouping.fell_back ? " (identity fallback)" : "") << "; batches=" << batch_count
              << ", partitions=" << partitions.size() << ", workers=" << total_workers << '\n';
    std::cout << "[Host] Planned L1 bytes: worker=" << worker_size << ", leader=" << leader_total
              << ", available=" << available << "; masks=" << static_cast<uint32_t>(config_.mask) << '\n';

    // Queues are concatenated records; cache reads use 4 KiB interleaved pages.
    const auto descriptor_begin = std::chrono::steady_clock::now();
    std::vector<Record> records;
    std::vector<uint32_t> leader_start(partitions.size());
    std::vector<std::vector<uint32_t>> worker_start(partitions.size());
    for (uint32_t p = 0; p < partitions.size(); ++p) {
        leader_start[p] = records.size();
        for (auto bid : partitions[p].batches) {
            const auto& b = batches[bid];
            records.push_back(Record{{bid, b.valid_queries, static_cast<uint32_t>(b.lists.size()), static_cast<uint32_t>(b.pages), 0, 0, 0, 0}});
            for (uint32_t w = 0; w < b.worker_lists.size(); ++w) for (auto id : b.worker_lists[w])
                records.push_back(Record{{w, id, lists[id].start_page, pages_for(lists[id].count), lists[id].count, 0, 0, 0}});
        }
        for (uint32_t w = 0; w < partitions[p].workers.size(); ++w) {
            worker_start[p].push_back(records.size());
            for (auto bid : partitions[p].batches) {
                const auto& b = batches[bid];
                records.push_back(Record{{bid, b.valid_queries, static_cast<uint32_t>(b.worker_lists[w].size()), static_cast<uint32_t>(b.worker_pages[w]), 0, 0, 0, 0}});
                for (auto id : b.worker_lists[w])
                    records.push_back(Record{{id, lists[id].start_page, pages_for(lists[id].count), lists[id].count, tail_lanes(lists[id].count), 0, 0, 0}});
            }
        }
    }
    const size_t words_count = ((records.size() * record_words + 1023) / 1024) * 1024;
    std::vector<uint32_t> script_words(words_count, 0);
    for (size_t r = 0; r < records.size(); ++r) std::copy_n(records[r].word, record_words, script_words.begin() + r * record_words);
    last_timings_.descriptor_pack_us = elapsed_us(descriptor_begin, std::chrono::steady_clock::now());
    auto scripts = dram_buffer(script_words.size() * 4, script_page_bytes);
    last_timings_.descriptor_h2d_us = write(scripts, script_words, "partition_descriptors");
    h2d_us += last_timings_.descriptor_h2d_us;
    auto slot_values = dram_buffer(uint64_t(batch_count) * total_workers * 2048, 2048);
    auto slot_indices = dram_buffer(uint64_t(batch_count) * total_workers * 4096, 4096);
    auto final_values = dram_buffer(uint64_t(batch_count) * 2048, 2048);
    auto final_indices = dram_buffer(uint64_t(batch_count) * 4096, 4096);
    Program program{};
    std::vector<std::shared_ptr<Buffer>> l1_buffers;
    auto allocate_l1 = [&](const CoreRangeSet& cores, uint32_t per_core) {
        per_core = (per_core + 31) / 32 * 32;
        auto buffer = CreateBuffer(ShardedBufferConfig{
            .device = device, .size = uint64_t(cores.num_cores()) * per_core, .page_size = per_core,
            .buffer_type = BufferType::L1, .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_parameters = ShardSpecBuffer(cores, {1, per_core / 4}, ShardOrientation::ROW_MAJOR,
                                               {1, per_core / 4}, {static_cast<uint32_t>(cores.num_cores()), 1})});
        const auto begin = std::chrono::steady_clock::now();
        detail::WriteToBuffer(buffer, std::vector<uint32_t>(buffer->size() / 4, 0));
        const auto end = std::chrono::steady_clock::now();
        h2d_us += elapsed_us(begin, end);
        last_timings_.control_h2d_us += elapsed_us(begin, end);
        memory_logger_->record("pcie", "host_to_device", "partition_control_init", buffer->size(), elapsed_us(begin, end), "blocking");
        AssignGlobalBufferToProgram(buffer, program);
        l1_buffers.push_back(buffer);
        return buffer->address();
    };
    const uint32_t worker_ctrl = allocate_l1(workers_set, role_control_bytes);
    const uint32_t leader_ctrl = allocate_l1(leaders_set, leader_size);
    const uint32_t agg_ctrl = config_.aggregation == "leader" ? leader_ctrl : allocate_l1(agg_set, role_control_bytes);
    auto cb = [&](const CoreRangeSet& cores, uint32_t id, uint32_t pages, uint32_t bytes, DataFormat format) {
        CreateCircularBuffer(program, cores, CircularBufferConfig(pages * bytes, {{id, format}}).set_page_size(id, bytes));
    };
    cb(workers_set, CB::c_in0, 2 * width, 2048, DataFormat::Float16_b);
    cb(workers_set, CB::c_in1, config_.worker_input * width, 2048, DataFormat::Float16_b);
    cb(workers_set, CB::c_in2, config_.worker_input, 4096, DataFormat::Int32);
    if (config_.mask == MaskMode::Additive) cb(workers_set, CB::c_in3, config_.worker_input, 2048, DataFormat::Float16_b);
    cb(workers_set, CB::c_intermed0, 2, 2048, DataFormat::Float16_b);
    cb(workers_set, CB::c_intermed1, 2, 2048, DataFormat::Float16_b);
    for (const auto& cores : {workers_set, agg_set}) {
        cb(cores, CB::c_intermed3, 2, 2048, DataFormat::Float16_b);
        cb(cores, CB::c_intermed4, 2, 4096, DataFormat::Int32);
        cb(cores, CB::c_out0, 2, 2048, DataFormat::Float16_b);
        cb(cores, CB::c_out1, 2, 4096, DataFormat::Int32);
        cb(cores, CB::c_intermed5, 4, record_bytes, DataFormat::Int32);
    }
    cb(agg_set, CB::c_in0, 2, 2048, DataFormat::Float16_b);
    cb(agg_set, CB::c_in1, 2, 4096, DataFormat::Int32);
    const uint32_t ready = CreateSemaphore(program, all_set, 0);
    const std::string prefix = "ann_ivf_similarity_partitioned/kernels/";
    const auto receiver = CreateKernel(program, prefix + "dataflow/worker_receiver.cpp", workers_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
            .compile_args = {width, config_.reader == "relay", static_cast<uint32_t>(config_.mask),
                             config_.worker_input, config_.reader == "direct"}});
    const auto worker_writer = CreateKernel(program, prefix + "dataflow/worker_writer.cpp", workers_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const auto compute = CreateKernel(program, prefix + "compute/worker_compute.cpp", workers_set,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
            .compile_args = {width, static_cast<uint32_t>(config_.mask)}});
    const auto leader_reader = CreateKernel(program, prefix + "dataflow/leader_reader.cpp", leaders_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
            .compile_args = {width, config_.reader == "relay", config_.bank_schedule == "staggered", config_.leader_prefetch}});
    const auto gather = CreateKernel(program, prefix + "dataflow/leader_gather_writer.cpp", agg_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const auto aggregate = CreateKernel(program, prefix + "compute/leader_aggregate.cpp", agg_set,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});

    // Use the actual allocator bank -> DRAM view -> physical controller map.
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const uint32_t bank_count = device->allocator()->get_num_banks(BufferType::DRAM);
    std::vector<uint32_t> bank_controller(bank_count);
    std::map<size_t, uint32_t> controller_ids;
    for (uint32_t bank = 0; bank < bank_count; ++bank) {
        const auto view = device->allocator_impl()->get_dram_channel_from_bank_id(bank);
        const auto physical = soc.get_channel_for_dram_view(view);
        if (!controller_ids.count(physical)) controller_ids[physical] = controller_ids.size();
        bank_controller[bank] = controller_ids.at(physical);
    }
    std::cout << "[Host] DRAM banks=" << bank_count << ", physical controller groups=" << controller_ids.size() << '\n';
    auto physical_core = [&](Core c) {
        const auto coord = device->worker_core_from_logical_core(CoreCoord{c.x, c.y});
        // CoreCoord components are size_t in some UMD versions. Runtime
        // arguments and the partition protocol use uint32_t coordinates.
        if (!std::in_range<uint32_t>(coord.x) || !std::in_range<uint32_t>(coord.y))
            throw std::runtime_error("Physical core coordinates exceed the uint32_t protocol");
        return Core{static_cast<uint32_t>(coord.x), static_cast<uint32_t>(coord.y)};
    };
    for (uint32_t p = 0; p < partitions.size(); ++p) {
        const auto& part = partitions[p];
        const auto leader = physical_core(part.leader), agg = physical_core(part.aggregator);
        const uint32_t count = part.batches.size(), num_workers = part.workers.size();
        uint32_t xmin = UINT32_MAX, ymin = UINT32_MAX, xmax = 0, ymax = 0;
        std::set<std::pair<uint32_t, uint32_t>> physical_workers;
        for (const auto core : part.workers) {
            const auto coord = physical_core(core);
            xmin = std::min(xmin, coord.x); ymin = std::min(ymin, coord.y);
            xmax = std::max(xmax, coord.x); ymax = std::max(ymax, coord.y);
            physical_workers.emplace(coord.x, coord.y);
        }
        const bool multicast = config_.query_broadcast == "auto" &&
            uint64_t(xmax - xmin + 1) * (ymax - ymin + 1) == physical_workers.size() &&
            !(leader.x >= xmin && leader.x <= xmax && leader.y >= ymin && leader.y <= ymax);
        std::vector<uint32_t> args = {scripts->address(), leader_start[p], count, leader_ctrl,
            fine_queries->address(), dataset_buffer_->address(), indices_buffer_->address(), num_workers,
            ready, bank_count, static_cast<uint32_t>(controller_ids.size()), p, max_tasks,
            multicast, xmin, ymin, xmax, ymax, config_.profile_batch};
        for (const auto core : part.workers) { const auto coord = physical_core(core); args.push_back(coord.x); args.push_back(coord.y); }
        args.insert(args.end(), bank_controller.begin(), bank_controller.end());
        SetRuntimeArgs(program, leader_reader, CoreCoord{part.leader.x, part.leader.y}, args);
        args = {scripts->address(), leader_start[p], count, agg_ctrl, slot_values->address(), slot_indices->address(),
                total_workers, num_workers, final_values->address(), final_indices->address(), leader.x, leader.y, leader_ctrl, config_.profile_batch};
        args.insert(args.end(), worker_slots[p].begin(), worker_slots[p].end());
        SetRuntimeArgs(program, gather, CoreCoord{part.aggregator.x, part.aggregator.y}, args);
        SetRuntimeArgs(program, aggregate, CoreCoord{part.aggregator.x, part.aggregator.y}, {count, config_.profile_batch});
        for (uint32_t w = 0; w < num_workers; ++w) {
            const auto core = CoreCoord{part.workers[w].x, part.workers[w].y};
            SetRuntimeArgs(program, receiver, core, {scripts->address(), worker_start[p][w], count, worker_ctrl,
                leader.x, leader.y, leader_ctrl, w, ready, dataset_buffer_->address(), indices_buffer_->address(), config_.profile_batch});
            SetRuntimeArgs(program, worker_writer, core, {count, worker_slots[p][w], total_workers,
                slot_values->address(), slot_indices->address(), agg.x, agg.y, agg_ctrl, w, worker_ctrl,
                scripts->address(), worker_start[p][w], config_.profile_batch});
            SetRuntimeArgs(program, compute, core, {count, config_.profile_batch});
        }
        std::cout << "[Partition " << p << "] leader logical(" << part.leader.x << ',' << part.leader.y
                  << "), NoC(" << leader.x << ',' << leader.y << "), workers=" << num_workers
                  << ", batches=" << count << ", query broadcast=" << (multicast ? "multicast" : "unicast") << '\n';
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)), std::move(program));
    const auto fine_begin = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device_->mesh_command_queue());
    const auto fine_end = std::chrono::steady_clock::now();
    std::vector<bfloat16> raw_values;
    std::vector<uint32_t> raw_indices;
    distributed::EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), raw_values, final_values, true);
    distributed::EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(), raw_indices, final_indices, true);
    last_timings_.final_readback_us = elapsed_us(fine_end, std::chrono::steady_clock::now());
    const auto scores = untilize_nfaces(raw_values, padded_queries, 32);
    const auto ids = untilize_nfaces(raw_indices, padded_queries, 32);
    std::vector<float> result_scores(queries * k, -1000.0f);
    std::vector<int64_t> result_ids(queries * k, -1);
    for (uint32_t packed_q = 0; packed_q < queries; ++packed_q) {
        const uint32_t original = grouping.packed_to_original[packed_q];
        uint32_t valid = 0;
        for (uint32_t lane = 0; lane < 32 && valid < k; ++lane) {
            const uint32_t id = ids[packed_q * 32 + lane];
            if (id == invalid_id) continue;
            const float score = static_cast<float>(scores[packed_q * 32 + lane]);
            if (id >= database_size_ || !std::isfinite(score)) throw std::runtime_error("Invalid fine result ID/score");
            result_scores[original * k + valid] = score;
            result_ids[original * k + valid++] = id;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    last_elapsed_us_ = elapsed_us(start, end);
    const double fine_us = elapsed_us(fine_begin, fine_end);
    const double coarse_setup = elapsed_us(start, coarse_begin) + coarse_pre_device_us_ - coarse_h2d_us;
    const double fine_prep = elapsed_us(coarse_begin, fine_begin) - coarse_device_us_ - coarse_pre_device_us_ - (h2d_us - coarse_h2d_us);
    const double output_us = elapsed_us(fine_end, end);
    last_timings_.h2d_us = h2d_us;
    last_timings_.cpu_coarse_config_us = coarse_setup;
    last_timings_.coarse_runtime_setup_us = coarse_pre_device_us_;
    last_timings_.tt_coarse_search_us = coarse_device_us_;
    last_timings_.cpu_fine_prep_us = fine_prep;
    last_timings_.tt_fine_search_us = fine_us;
    last_timings_.cpu_output_us = output_us;
    last_timings_.pipeline_us = last_elapsed_us_;
    last_timings_.finish();
    // Printing, stage logging and diagnostics run after the pipeline timer.
    std::cout << std::fixed << std::setprecision(3)
              << "[Timing] " << context_ << ": details below are included in their parent stage\n";
    for (const auto& stage : timing_stages) {
        const std::string field_name(stage.name);
        memory_logger_->record("chrono_stage", "stage", field_name.substr(0, field_name.size() - 3),
                               0, last_timings_.*(stage.value), "wall_clock");
        std::cout << "[Chrono] " << stage.label << ": " << last_timings_.*(stage.value) / 1000.0 << " ms\n";
        for (const auto& detail : timing_details) {
            if (field_name != detail.parent) continue;
            memory_logger_->record("chrono_detail", "detail", detail.name, 0, last_timings_.*(detail.value),
                                   "wall_clock", "included_in=" + field_name);
            std::cout << "  [Detail] " << detail.label << ": " << last_timings_.*(detail.value) / 1000.0 << " ms\n";
        }
    }
    for (const auto& total : timing_totals)
        std::cout << "[Chrono] " << total.label << ": " << last_timings_.*(total.value) / 1000.0 << " ms\n";
    memory_logger_->record("chrono", "host", "pipeline", 0, last_elapsed_us_, "synchronous");
    std::vector<uint64_t> scanned_vectors;
    uint64_t scanned_sum = 0;
    for (const auto& batch : batches) {
        scanned_vectors.push_back(batch.vectors);
        scanned_sum += batch.vectors;
    }
    std::sort(scanned_vectors.begin(), scanned_vectors.end());
    auto percentile = [&](double fraction) {
        const double index = fraction * (scanned_vectors.size() - 1);
        const size_t lower = static_cast<size_t>(index), upper = std::min(lower + 1, scanned_vectors.size() - 1);
        return double(scanned_vectors[lower]) + (index - lower) * double(scanned_vectors[upper] - scanned_vectors[lower]);
    };
    const double percent_scale = 100.0 / database_size_;
    std::cout << "[Scan] " << context_ << ": dataset=" << database_size_ << " vectors; batch-union coverage (%)"
              << " mean=" << double(scanned_sum) / batches.size() * percent_scale
              << ", median=" << percentile(0.5) * percent_scale << ", p95=" << percentile(0.95) * percent_scale
              << ", max=" << scanned_vectors.back() * percent_scale << '\n';
    if (!config_.diagnostics.empty()) {
        std::filesystem::create_directories(config_.diagnostics);
        const auto root = std::filesystem::path(config_.diagnostics);
        std::ofstream output(root / ("batches_" + context_ + ".csv"));
        if (!output) throw std::runtime_error("Could not write grouping diagnostics");
        output << std::setprecision(12);
        output << "nlist,nprobe,queries,k,grouping,partitions,reader,bank_schedule,aggregation,mask,layout,query_broadcast,leader_prefetch,worker_input,grouping_window,grouping_lookahead,batch,valid_queries,partition,union_lists,union_pages,union_vectors,own_vectors_mean,extra_scan_factor,active_workers,worker_pages_max,worker_pages_mean,imbalance,database_vectors,scanned_fraction,scanned_percent,own_scanned_percent\n";
        for (const auto& b : batches) {
            const auto maximum = *std::max_element(b.worker_pages.begin(), b.worker_pages.end());
            const double mean = double(b.pages) / b.worker_pages.size();
            const double own = double(b.own_vectors) / b.valid_queries;
            output << nlist_ << ',' << nprobe_ << ',' << queries << ',' << k << ',' << config_.grouping << ',' << config_.partitions << ',' << config_.reader << ',' << config_.bank_schedule << ','
                   << config_.aggregation << ',' << static_cast<uint32_t>(config_.mask) << ',' << config_.layout << ',' << config_.query_broadcast << ','
                   << config_.leader_prefetch << ',' << config_.worker_input << ',' << config_.grouping_window << ',' << config_.grouping_lookahead << ','
                   << b.id << ',' << b.valid_queries << ',' << b.partition << ',' << b.lists.size() << ',' << b.pages << ',' << b.vectors << ',' << own << ','
                   << (own ? b.vectors / own : 0) << ',' << std::count_if(b.worker_pages.begin(), b.worker_pages.end(), [](auto n) { return n != 0; }) << ','
                   << maximum << ',' << mean << ',' << (mean ? maximum / mean : 0) << ',' << database_size_ << ','
                   << double(b.vectors) / database_size_ << ',' << double(b.vectors) * percent_scale << ',' << own * percent_scale << '\n';
        }
        std::ofstream permutation(root / ("queries_" + context_ + ".csv"));
        if (!permutation) throw std::runtime_error("Could not write query permutation diagnostics");
        permutation << "packed_query,original_query,selected_lists\n";
        for (uint32_t q = 0; q < queries; ++q) {
            permutation << q << ',' << grouping.packed_to_original[q] << ',';
            for (auto id : selections[grouping.packed_to_original[q]]) permutation << id << ' ';
            permutation << '\n';
        }
        std::ofstream sizes(root / "list_sizes.csv");
        if (!sizes) throw std::runtime_error("Could not write list-size diagnostics");
        sizes << "list,vectors,pages\n";
        for (uint32_t id = 0; id < nlist_; ++id) sizes << id << ',' << lists[id].count << ',' << list_pages[id] << '\n';
        output.close(); permutation.close(); sizes.close();
        if (!output || !permutation || !sizes) throw std::runtime_error("Could not finish writing grouping diagnostics");
        std::cout << "[Scan] Per-batch vectors and dataset coverage: " << root / ("batches_" + context_ + ".csv") << '\n';
    }
    return {std::move(result_scores), std::move(result_ids)};
}
