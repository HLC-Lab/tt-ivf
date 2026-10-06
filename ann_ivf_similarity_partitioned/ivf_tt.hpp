#pragma once
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "memory_transfer_logger.hpp"
#include "pipeline_timings.hpp"
#include "query_grouping.hpp"
using namespace tt;
using namespace tt::tt_metal;
struct ClusterInfo { uint32_t offset, count, padded_count, start_block; };
struct PartitionConfig {
    std::string grouping = "weighted-union", layout = "rows", reader = "relay", bank_schedule = "staggered";
    std::string aggregation = "leader", query_broadcast = "auto", diagnostics;
    uint32_t partitions = 8, grouping_window = 0, grouping_lookahead = 256;
    uint32_t leader_prefetch = 8, worker_input = 2;
    uint32_t profile_batch = UINT32_MAX;
    ivf_partitioned::MaskMode mask = ivf_partitioned::MaskMode::Auto;
};
class IVF_TT {
public:
    IVF_TT(uint32_t dim, uint32_t nlist, uint32_t nprobe, const std::string& log, const PartitionConfig& config);
    ~IVF_TT();
    void create_index(const std::vector<float>& vectors, const std::string& centroids);
    void load_to_device();
    std::pair<std::vector<float>, std::vector<int64_t>> search_all(const float*, uint32_t queries, uint32_t k);
    void set_memory_log_context(const std::string& context) { context_ = context; memory_logger_->set_context(context); }
    void flush_memory_log() { memory_logger_->flush(); }
    double last_elapsed_us() const { return last_elapsed_us_; }
    const ivf_partitioned::SearchTimings& last_timings() const { return last_timings_; }
private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    uint32_t dim_, nlist_, nprobe_, database_size_ = 0;
    PartitionConfig config_;
    std::string context_;
    std::unique_ptr<MemoryTransferLogger> memory_logger_;
    double coarse_pre_device_us_ = 0, coarse_device_us_ = 0;
    double last_elapsed_us_ = 0;
    ivf_partitioned::SearchTimings last_timings_;
    std::vector<float> centroids_host_;
    std::vector<bfloat16> dataset_host_padded_;
    std::vector<uint32_t> index_map_host_padded_;
    std::unordered_map<uint32_t, ClusterInfo> cluster_toc_;
    std::shared_ptr<distributed::MeshBuffer> centroids_buffer_, coarse_query_buffer_, coarse_indices_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_out_val_buffer_, coarse_out_ind_buffer_, dataset_buffer_, indices_buffer_;
    std::optional<Program> program_coarse_temp_;
    KernelHandle reader_coarse_id_, writer_coarse_id_, compute_coarse_id_;
    void setup_coarse_program(uint32_t queries);
    ivf_partitioned::Selections perform_coarse_search(uint32_t valid_queries);
};
