#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t out_val_addr = get_arg_val<uint32_t>(0);
    uint32_t out_ind_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tasks = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_val = tt::CB::c_out0;
    constexpr uint32_t cb_ind = tt::CB::c_out1;
    constexpr uint32_t cb_writer_args = tt::CB::c_in4;

    uint32_t tile_bytes_val = get_tile_size(cb_val);
    uint32_t tile_bytes_ind = get_tile_size(cb_ind);

    const InterleavedAddrGenFast<true> s_val = {
        .bank_base_address = out_val_addr,
        .page_size = tile_bytes_val,
        .data_format = DataFormat::Float16_b
    };

    const InterleavedAddrGenFast<true> s_ind = {
        .bank_base_address = out_ind_addr,
        .page_size = tile_bytes_ind,
        .data_format = DataFormat::Int32
    };


    for (uint32_t task = 0; task < num_tasks; task++) {
        // Wait for metadata from Reader Kernel
        cb_wait_front(cb_writer_args, 1);
        uint32_t l1_ptr = get_read_ptr(cb_writer_args);
        volatile uint32_t* meta = reinterpret_cast<volatile uint32_t*>(l1_ptr);
        uint32_t result_write_idx = meta[0];
        cb_pop_front(cb_writer_args, 1);

        // Write Values Champion
        cb_wait_front(cb_val, 1);
        uint32_t l1_read_addr_val = get_read_ptr(cb_val);
        noc_async_write_tile(result_write_idx, s_val, l1_read_addr_val);
        noc_async_write_barrier();
        cb_pop_front(cb_val, 1);

        // Write Indices Champion
        cb_wait_front(cb_ind, 1);
        uint32_t l1_read_addr_ind = get_read_ptr(cb_ind);
        noc_async_write_tile(result_write_idx, s_ind, l1_read_addr_ind);
        noc_async_write_barrier();
        cb_pop_front(cb_ind, 1);
    }
}
