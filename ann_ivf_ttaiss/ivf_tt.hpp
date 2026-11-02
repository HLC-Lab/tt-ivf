// ann_ivf_ttaiss/ivf_tt.hpp
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <array>
#include <optional>
#include <unordered_map>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "memory_transfer_logger.hpp"
#include "query_grouping.hpp"
#include "cluster_schedule.hpp"

using namespace tt;
using namespace tt::tt_metal;

struct ClusterInfo {
    uint32_t offset;
    uint32_t count;
    uint32_t padded_count;
    uint32_t start_block;
};

struct RunOptions {
    uint32_t fine_fidelity = 4;    // tt::tt_metal::MathFidelity value; 4 = HiFi4
    uint32_t coarse_fidelity = 4;
    // Energy workload: no final-result readback/restore, no transfer log and
    // no per-search console report. The merge still writes results to DRAM.
    bool energy = false;
    bool reference_index_build = false;  // original single-threaded/scalar build
    bool index_digest = false;           // print a hash of the finished index
    bool rerank = false;                 // host FP32 re-ranking of the device top-32
    // Two-pass search with merge skipping: the first pass scans each query's
    // first_pass_lists best lists; 0 = single pass (no skipping).
    uint32_t first_pass_lists = 0;
    // Ablations (measurement only; both off by default):
    // ablation_rank32: skip against the 32nd-best score even without --rerank;
    // ablation_rebuild: rebuild and reallocate every program and buffer per search.
    bool ablation_rank32 = false;
    bool ablation_rebuild = false;
};

class IVF_TT {
public:
    IVF_TT(
        int device_id,
        uint32_t dim,
        uint32_t nlist = 1024,
        uint32_t nprobe = 32,
        const std::string& memory_log_path = "ann_ivf_ttaiss.csv",
        uint32_t prefetch_reader = 2,
        uint32_t batch_chunk = 32,
        ivf_cluster_major::GroupingConfig query_grouping = {},
        RunOptions options = {});
    ~IVF_TT();

    void create_index(const std::vector<float>& dataset_vectors, const std::string& centroids_path);
    void load_to_device();
    std::pair<std::vector<float>, std::vector<int64_t>> search_all(const float* query_data, uint32_t total_queries, uint32_t k);
    std::vector<uint32_t> perform_coarse_search(const float* query_batch, uint32_t total_queries = 32);
    void set_memory_log_context(const std::string& context);
    void flush_memory_log();
    double last_search_us() const { return last_search_us_; }
    // Validation exports: the last search's coarse selections (nprobe list IDs
    // per query, rank order) and every dataset vector's list.
    const ivf_cluster_major::Selections& last_selections() const { return coarse_selections_; }
    const std::vector<uint32_t>& vector_lists() const { return vector_lists_; }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    uint32_t dim_;
    uint32_t nlist_;
    uint32_t nprobe_;
    uint32_t prefetch_reader_;
    uint32_t batch_chunk_;
    uint64_t usable_l1_bytes_ = 0;
    RunOptions options_;
    // --rerank: L2-normalized FP32 dataset kept after the index build.
    std::unique_ptr<float[]> rerank_vectors_;
    size_t rerank_vector_count_ = 0;

    // Fine programs and DRAM buffers, kept across searches. Stages run in
    // queue order: scan, merge (single pass) or scan, merge, scan, merge (two
    // passes). A stage's script page grows to the largest script a search has
    // needed; its program is rebuilt only when the page grows (the script CB
    // capacity is part of the program) or, for scans, when the threshold rank
    // changes (a compile-time argument). Buffers grow the same way and are
    // written and read only over the region a search uses.
    struct FineStage {
        std::optional<distributed::MeshWorkload> workload;
        KernelHandle reader{}, writer{}, compute{};
        std::shared_ptr<distributed::MeshBuffer> scripts;  // one page per core
        uint64_t script_bytes = 0;                         // page size = script CB capacity
        uint32_t threshold_rank = UINT32_MAX;              // scans: compile-time argument built
    };
    struct FinePrograms {
        std::array<FineStage, 4> stages;
        // Per pass: [tile][32] query IDs, UINT32_MAX = unused row.
        std::array<std::shared_ptr<distributed::MeshBuffer>, 2> tile_queries;
        std::shared_ptr<distributed::MeshBuffer> partial_val, partial_ind, out_val, out_ind;
    } fine_;
    void build_scan_stage(FineStage& stage, uint32_t threshold_rank);
    void build_merge_stage(FineStage& stage);
    // Keep `buffer` while it holds `bytes` with this page size; otherwise
    // allocate exactly `bytes`. Returns true when it allocated.
    bool grow_buffer(std::shared_ptr<distributed::MeshBuffer>& buffer, uint64_t bytes, uint32_t page);
    ivf_cluster_major::GroupingConfig query_grouping_;
    ivf_cluster_major::Selections coarse_selections_;
    std::unique_ptr<MemoryTransferLogger> memory_logger_;
    double coarse_pre_device_us_ = 0.0;
    double coarse_device_us_ = 0.0;
    double coarse_post_device_us_ = 0.0;
    double last_search_us_ = 0.0;

    std::vector<float> centroids_host_;
    std::vector<bfloat16> dataset_host_padded_;
    std::vector<uint32_t> index_map_host_padded_;
    std::unordered_map<uint32_t, ClusterInfo> cluster_toc_;
    std::vector<uint32_t> cluster_query_masks_;
    std::vector<uint32_t> vector_lists_;

    std::shared_ptr<distributed::MeshBuffer> centroids_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_query_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_indices_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_out_val_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_out_ind_buffer_;
    std::shared_ptr<distributed::MeshBuffer> dataset_buffer_;
    std::shared_ptr<distributed::MeshBuffer> indices_buffer_;

    // Built once: the coarse program's CBs do not depend on the query count.
    std::optional<distributed::MeshWorkload> coarse_workload_;

    tt::tt_metal::KernelHandle reader_coarse_id_;
    tt::tt_metal::KernelHandle writer_coarse_id_;
    tt::tt_metal::KernelHandle compute_coarse_id_;

    CoreRangeSet all_cores_;

    void build_coarse_program();
    std::vector<float> read_centroids(const std::string& centroids_path) const;
    void create_index_reference(const std::vector<float>& dataset_vectors, const std::string& centroids_path);
    void create_index_fast(const std::vector<float>& dataset_vectors, const std::string& centroids_path);
};
