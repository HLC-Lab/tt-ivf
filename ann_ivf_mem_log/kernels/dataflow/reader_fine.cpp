#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // Move this template to the exact lexical scope you want to measure, then uncomment it.
    // DeviceZoneScopedN("IVF_Custom_Profile_Zone");
    uint32_t query_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dataset_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t indices_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t dim_padded = get_arg_val<uint32_t>(3);

    uint32_t scripts_dram_addr = get_arg_val<uint32_t>(4);
    uint32_t worker_idx = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_query = tt::CB::c_in0;
    constexpr uint32_t cb_cluster_data = tt::CB::c_in1;
    constexpr uint32_t cb_cluster_indices = tt::CB::c_in2;
    constexpr uint32_t cb_script = tt::CB::c_intermed5;

    const InterleavedAddrGen<true> s_script = {
        .bank_base_address = scripts_dram_addr,
        .page_size = 8192 // 1 page per worker
    };

    cb_reserve_back(cb_script, 1);
    uint32_t script_l1 = get_write_ptr(cb_script);
    noc_async_read_page(worker_idx, s_script, script_l1);
    noc_async_read_barrier();
    cb_push_back(cb_script, 1);

    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*) script_l1;

    uint32_t query_tiles_width = dim_padded / 32;

    const InterleavedAddrGen<true> s_query = {
        .bank_base_address = query_dram_addr,
        .page_size = 2048
    };

    const InterleavedAddrGen<true> s_dataset = {
        .bank_base_address = dataset_dram_addr,
        .page_size = dim_padded * 64 // 1 col of 32 vectors
    };

    const InterleavedAddrGen<true> s_indices = {
        .bank_base_address = indices_dram_addr,
        .page_size = 4096 // 1 tile of indices
    };

    uint32_t num_batches = script[0];
    uint32_t num_assigned = script[1];
    uint32_t num_mask_words = (num_batches + 31) / 32;

    for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
        bool query_pushed = false;
        uint32_t ptr = 2;
        for (uint32_t cluster_idx = 0; cluster_idx < num_assigned; cluster_idx++) {
            ptr++; // cluster ID is not required by the data-movement kernel
            uint32_t offset = script[ptr++];
            ptr++; // actual vector count; padding controls the page reads
            uint32_t padded = script[ptr++];
            uint32_t pages = padded / 32;
            uint32_t mask_ptr = ptr;
            ptr += num_mask_words;

            bool is_active = (script[mask_ptr + (batch_id / 32)] & (1u << (batch_id % 32))) != 0;

            if (!is_active) {
                continue; // Push NOTHING!
            }

            // The query is identical for every cluster task assigned to this
            // worker in the current batch. Keep one copy in the CB until the
            // compute kernel has processed all active tasks.
            if (!query_pushed) {
                {
                    DeviceZoneScopedN("Query");
                    {
                        DeviceZoneScopedN("FineBubble.Reader.Query");
                        cb_reserve_back(cb_query, query_tiles_width);
                        uint32_t q_l1 = get_write_ptr(cb_query);
                        for (uint32_t t = 0; t < query_tiles_width; t++) {
                            noc_async_read_tile(batch_id * query_tiles_width + t, s_query, q_l1);
                            q_l1 += 2048;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_query, query_tiles_width);
                    }
                }
                query_pushed = true;
            }
            {
                DeviceZoneScopedN("Cluster");
                {
                    DeviceZoneScopedN("FineBubble.Reader.Cluster");
                    for (uint32_t p = 0; p < pages; p++) {
                        cb_reserve_back(cb_cluster_data, query_tiles_width);
                        cb_reserve_back(cb_cluster_indices, 1);

                        {
                            DeviceZoneScopedN("FineBubble.Reader.Fetch");
                            const uint32_t d_l1 = get_write_ptr(cb_cluster_data);
                            const uint32_t i_l1 = get_write_ptr(cb_cluster_indices);
                            noc_async_read_page(offset + p, s_dataset, d_l1);
                            noc_async_read_page(offset + p, s_indices, i_l1);
                            noc_async_read_barrier();
                            cb_push_back(cb_cluster_data, query_tiles_width);
                            cb_push_back(cb_cluster_indices, 1);
                        }
                    }
                }
            }
        }
    }
}
