// ann_ivf_energy/ivf_tt.hpp
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "query_grouping.hpp"

using namespace tt;
using namespace tt::tt_metal;

struct ClusterInfo {
    uint32_t offset;
    uint32_t count;
    uint32_t padded_count;
    uint32_t start_block;
};

enum class FineResultStaging {
    Dram,
    Core0L1,
};

class IVF_TT {
public:
    IVF_TT(
        int device_id,
        uint32_t dim,
        uint32_t nlist = 1024,
        uint32_t nprobe = 32,
        FineResultStaging result_staging = FineResultStaging::Dram,
        uint32_t cluster_chunk_blocks = 0,
        ivf_energy::GroupingConfig query_grouping = {});
    ~IVF_TT();

    void create_index(const std::vector<float>& dataset_vectors, const std::string& centroids_path);
    void load_to_device();
    void search_all(const float* query_data, uint32_t total_queries, uint32_t k);
    std::vector<uint32_t> perform_coarse_search(const float* query_batch, uint32_t total_queries);

private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    uint32_t dim_;
    uint32_t nlist_;
    uint32_t nprobe_;
    FineResultStaging result_staging_;
    ivf_energy::GroupingConfig query_grouping_;
    ivf_energy::Selections coarse_selections_;

    std::vector<float> centroids_host_;
    std::vector<bfloat16> dataset_host_padded_;
    std::vector<uint32_t> index_map_host_padded_;
    std::unordered_map<uint32_t, ClusterInfo> cluster_toc_;
    std::vector<uint32_t> cluster_query_masks_;

    std::shared_ptr<distributed::MeshBuffer> centroids_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_query_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_indices_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_out_val_buffer_;
    std::shared_ptr<distributed::MeshBuffer> coarse_out_ind_buffer_;
    std::shared_ptr<distributed::MeshBuffer> dataset_buffer_;
    std::shared_ptr<distributed::MeshBuffer> indices_buffer_;

    std::optional<Program> program_coarse_temp_;
    std::optional<Program> program_fine_temp_;

    tt::tt_metal::KernelHandle reader_coarse_id_;
    tt::tt_metal::KernelHandle writer_coarse_id_;
    tt::tt_metal::KernelHandle compute_coarse_id_;
    tt::tt_metal::KernelHandle reader_fine_id_;
    tt::tt_metal::KernelHandle writer_fine_id_;
    tt::tt_metal::KernelHandle compute_fine_id_;
    tt::tt_metal::KernelHandle reader_final_sort_id_;
    tt::tt_metal::KernelHandle writer_final_sort_id_;
    tt::tt_metal::KernelHandle compute_final_sort_id_;

    uint32_t done_sem_addr_;
    uint32_t ready_sem_addr_;
    std::shared_ptr<Buffer> result_slots_l1_buf_;
    std::shared_ptr<distributed::MeshBuffer> result_slot_values_dram_buf_;
    std::shared_ptr<distributed::MeshBuffer> result_slot_indices_dram_buf_;
    std::shared_ptr<distributed::MeshBuffer> scripts_dram_buf_;

    CoreRangeSet all_cores_;
    CoreRangeSet worker_cores_;
    distributed::MeshWorkload workload_coarse_;
    CoreRange core_0_0_ = CoreRange({0,0}, {0,0});

    void setup_coarse_program(uint32_t num_queries);
    void setup_fine_program();

    template<typename T>
    std::vector<T> load_binary(const std::string& path, size_t elements) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Could not open file: " + path);

        // Skip the 8-byte [N][D] uint32 header
        file.seekg(8, std::ios::beg);

        std::vector<T> data(elements);
        file.read(reinterpret_cast<char*>(data.data()), elements * sizeof(T));
        return data;
    }
};
