// tt_metal/programming_examples/ann_ivf_mem_log/kernels/dataflow/writer_coarse.cpp
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_val_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_ind_addr = get_arg_val<uint32_t>(1);
    uint32_t output_tiles = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t start_batch = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0); // values
    constexpr uint32_t cb_out1 = get_compile_time_arg_val(1); // indices


    uint32_t tile_size_out0 = get_tile_size(cb_out0);
    uint32_t tile_size_out1 = get_tile_size(cb_out1);

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = dst_val_addr,
        .page_size = tile_size_out0,
        .data_format = DataFormat::Float16_b
    };

    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = dst_ind_addr,
        .page_size = tile_size_out1,
        .data_format = DataFormat::Int32
    };

    for (uint32_t h = 0; h < Ht; h++) {

        for (uint32_t i = 0; i < output_tiles; i++) {
            cb_wait_front(cb_out0, 1);
            uint32_t l1_read_addr_out0 = get_read_ptr(cb_out0);
            if (h == 0 && i == 0) {
                noc_async_write_tile((start_batch + h) * output_tiles + i, s0, l1_read_addr_out0);
                noc_async_write_barrier();
            } else {
                noc_async_write_tile((start_batch + h) * output_tiles + i, s0, l1_read_addr_out0);
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out0, 1);
        }
        for (uint32_t i = 0; i < output_tiles; i++) {
            cb_wait_front(cb_out1, 1);
            uint32_t l1_read_addr_out1 = get_read_ptr(cb_out1);
            if (h == 0 && i == 0) {
                noc_async_write_tile((start_batch + h) * output_tiles + i, s1, l1_read_addr_out1);
                noc_async_write_barrier();
            } else {
                noc_async_write_tile((start_batch + h) * output_tiles + i, s1, l1_read_addr_out1);
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out1, 1);
        }
    }
}
