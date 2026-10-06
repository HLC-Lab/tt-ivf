// tt_metal/programming_examples/ann_ivf_energy/kernels/dataflow/reader_coarse.cpp
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t c_addr = get_arg_val<uint32_t>(1);
    uint32_t i_addr = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t start_batch = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_query = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cluster = get_compile_time_arg_val(1);
    constexpr uint32_t cb_indices = get_compile_time_arg_val(2);
    constexpr uint32_t query_tiles_width = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);


    uint32_t tile_size_q = get_tile_size(cb_query);
    uint32_t tile_size_c = get_tile_size(cb_cluster);
    uint32_t tile_size_i = get_tile_size(cb_indices);

    const InterleavedAddrGenFast<true> q_gen = {
        .bank_base_address = q_addr,
        .page_size = tile_size_q,
        .data_format = DataFormat::Float16_b
    };

    const InterleavedAddrGen<true> c_gen = {
        .bank_base_address = c_addr,
        .page_size = query_tiles_width * 2048
    };

    const InterleavedAddrGenFast<true> i_gen = {
        .bank_base_address = i_addr,
        .page_size = tile_size_i,
        .data_format = DataFormat::Int32
    };

    for (uint32_t h = 0; h < Ht; h++) {

        cb_reserve_back(cb_query, query_tiles_width);
        uint32_t l1_addr_q = get_write_ptr(cb_query);
        if (h == 0) {
            for (uint32_t i = 0; i < query_tiles_width; i++) {
                noc_async_read_tile((start_batch + h) * query_tiles_width + i, q_gen, l1_addr_q);
                l1_addr_q += tile_size_q;
            }
            noc_async_read_barrier();
        } else {
            for (uint32_t i = 0; i < query_tiles_width; i++) {
                noc_async_read_tile((start_batch + h) * query_tiles_width + i, q_gen, l1_addr_q);
                l1_addr_q += tile_size_q;
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_query, query_tiles_width);

        // Prefetch column 0
        if (Wt > 0) {
            cb_reserve_back(cb_cluster, query_tiles_width);
            cb_reserve_back(cb_indices, 1);
            if (h == 0) {
                noc_async_read(c_gen.get_noc_addr(0), get_write_ptr(cb_cluster), query_tiles_width * 2048);
                noc_async_read_tile(0, i_gen, get_write_ptr(cb_indices));
                noc_async_read_barrier();
            } else {
                noc_async_read(c_gen.get_noc_addr(0), get_write_ptr(cb_cluster), query_tiles_width * 2048);
                noc_async_read_tile(0, i_gen, get_write_ptr(cb_indices));
            }
        }

        for (uint32_t w = 0; w < Wt; w++) {
            if (!(h == 0 && w == 0)) {
                noc_async_read_barrier();
            }
            cb_push_back(cb_cluster, query_tiles_width);
            cb_push_back(cb_indices, 1);

            if (w + 1 < Wt) {
                cb_reserve_back(cb_cluster, query_tiles_width);
                cb_reserve_back(cb_indices, 1);
                noc_async_read(c_gen.get_noc_addr(w + 1), get_write_ptr(cb_cluster), query_tiles_width * 2048);
                noc_async_read_tile(w + 1, i_gen, get_write_ptr(cb_indices));
            }
        }
    }
}
