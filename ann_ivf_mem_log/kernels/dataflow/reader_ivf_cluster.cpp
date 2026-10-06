#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t query_addr = get_arg_val<uint32_t>(0);
    uint32_t dataset_addr = get_arg_val<uint32_t>(1);
    uint32_t indices_addr = get_arg_val<uint32_t>(2);
    uint32_t args_addr  = get_arg_val<uint32_t>(3);
    uint32_t base_page  = get_arg_val<uint32_t>(4);
    uint32_t num_tasks  = get_arg_val<uint32_t>(5);
    uint32_t query_tiles_width = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_query = tt::CB::c_in0;
    constexpr uint32_t cb_cluster_data = tt::CB::c_in1;
    constexpr uint32_t cb_indices = tt::CB::c_in2;
    constexpr uint32_t cb_compute_args = tt::CB::c_in3;
    constexpr uint32_t cb_writer_args = tt::CB::c_in4;
    constexpr uint32_t cb_args = tt::CB::c_in5;

    uint32_t tile_bytes_q = get_tile_size(cb_query);
    uint32_t tile_bytes_c = get_tile_size(cb_cluster_data);
    uint32_t tile_bytes_i = get_tile_size(cb_indices);

    const InterleavedAddrGenFast<true> q_gen = {
        .bank_base_address = query_addr,
        .page_size = tile_bytes_q,
        .data_format = DataFormat::Float16_b
    };

    uint32_t dataset_page_size = query_tiles_width * 64;
    const InterleavedAddrGenFast<true> d_gen = {
        .bank_base_address = dataset_addr,
        .page_size = dataset_page_size,
        .data_format = DataFormat::Float16_b
    };

    const InterleavedAddrGenFast<true> idx_gen = {
        .bank_base_address = indices_addr,
        .page_size = 4096,
        .data_format = DataFormat::Int32
    };

    const InterleavedAddrGenFast<true> args_gen = {
        .bank_base_address = args_addr,
        .page_size = 4096,
        .data_format = DataFormat::Int32
    };

    uint32_t arg_page = base_page;
    uint32_t arg_offset = 1024; // Force read on first get_next_arg()
    volatile uint32_t* arg_ptr = nullptr;

    auto get_next_arg = [&]() -> uint32_t {
        if (arg_offset >= 1024) {
            if (arg_ptr != nullptr) {
                cb_push_back(cb_args, 1);
                cb_pop_front(cb_args, 1);
            }
            cb_reserve_back(cb_args, 1);
            arg_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_args));

            uint64_t noc_addr = args_gen.get_noc_addr(arg_page);
            noc_async_read(noc_addr, (uint32_t)arg_ptr, 4096);
            noc_async_read_barrier();

            arg_page++;
            arg_offset = 0;
        }
        return arg_ptr[arg_offset++];
    };

    for (uint32_t task = 0; task < num_tasks; task++) {
        uint32_t query_page_id = get_next_arg();
        uint32_t result_write_idx = get_next_arg();
        uint32_t total_blocks = get_next_arg(); // Used for debug/info
        uint32_t num_clusters = get_next_arg();


        // 1. Fetch Query Block
        cb_reserve_back(cb_query, query_tiles_width);
        uint32_t l1_addr_q = get_write_ptr(cb_query);
        for (uint32_t i = 0; i < query_tiles_width; ++i) {
            noc_async_read_tile(query_page_id * query_tiles_width + i, q_gen, l1_addr_q);
            l1_addr_q += tile_bytes_q;
        }
        noc_async_read_barrier();
        cb_push_back(cb_query, query_tiles_width);

        // 2. Decode cluster metadata for Compute
        cb_reserve_back(cb_compute_args, 1);
        volatile uint32_t* comp_args = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_compute_args));
        comp_args[0] = num_clusters;
        uint32_t comp_arg_idx = 1;

        // We must pre-read the Wt and query_mask arrays because compute kernel needs them
        // to start, but the cluster data loop reads from DRAM synchronously.
        // If we don't push cb_compute_args first, compute kernel hangs waiting for it,
        // while we hang trying to push cluster data into a full CB!
        // Fortunately, the DRAM array is contiguous: [start_page, Wt, query_mask] * num_clusters
        uint32_t current_arg_page = arg_page;
        uint32_t current_arg_offset = arg_offset;

        for (uint32_t c = 0; c < num_clusters; c++) {
            // skip start_page
            get_next_arg();
            comp_args[comp_arg_idx++] = get_next_arg(); // Wt
            comp_args[comp_arg_idx++] = get_next_arg(); // query_mask
        }
        cb_push_back(cb_compute_args, 1);

        // Reset pointers to re-read the start_page
        arg_page = current_arg_page;
        arg_offset = current_arg_offset;

        // 3. Push metadata to Writer
        cb_reserve_back(cb_writer_args, 1);
        volatile uint32_t* wr_args = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_writer_args));
        wr_args[0] = result_write_idx;
        cb_push_back(cb_writer_args, 1);

        // 4. Process Clusters dynamically
        for (uint32_t c = 0; c < num_clusters; c++) {
            uint32_t start_page = get_next_arg();
            uint32_t Wt = get_next_arg();
            uint32_t query_mask = get_next_arg();

            for (uint32_t w = 0; w < Wt; ++w) {
                cb_reserve_back(cb_cluster_data, query_tiles_width);
                uint32_t l1_addr_d = get_write_ptr(cb_cluster_data);
                noc_async_read(d_gen.get_noc_addr(start_page + w), l1_addr_d, dataset_page_size);

                cb_reserve_back(cb_indices, 1);
                uint32_t l1_addr_i = get_write_ptr(cb_indices);
                noc_async_read(idx_gen.get_noc_addr(start_page + w), l1_addr_i, 4096);

                noc_async_read_barrier();

                cb_push_back(cb_cluster_data, query_tiles_width);
                cb_push_back(cb_indices, 1);
            }
        }

    }
}
