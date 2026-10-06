#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    uint32_t dram_val_out = get_arg_val<uint32_t>(0);
    uint32_t dram_ind_out = get_arg_val<uint32_t>(1);

    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*) get_arg_addr(2);
    uint32_t total_batches = script[0];
    uint32_t ptr = 1 + 64 * 2;

    constexpr uint32_t cb_out_val = tt::CB::c_out0;
    constexpr uint32_t cb_out_ind = tt::CB::c_out1;

    const InterleavedAddrGenFast<true> s_val = {
        .bank_base_address = dram_val_out,
        .page_size = 2048,
        .data_format = DataFormat::Float16_b
    };

    const InterleavedAddrGenFast<true> s_ind = {
        .bank_base_address = dram_ind_out,
        .page_size = 4096,
        .data_format = DataFormat::Int32
    };

    uint32_t num_active_workers = script[ptr];
    for (uint32_t b = 0; b < total_batches; b++) {
        if (num_active_workers == 0) {
            continue;
        }

        cb_wait_front(cb_out_val, 1);
        {
        DeviceZoneScopedN("Res");
        uint32_t val_addr = get_read_ptr(cb_out_val);
        if (b == 0) {
            noc_async_write_tile(b, s_val, val_addr);
            noc_async_write_barrier();
        } else {
            noc_async_write_tile(b, s_val, val_addr);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out_val, 1);

        cb_wait_front(cb_out_ind, 1);

        uint32_t ind_addr = get_read_ptr(cb_out_ind);
        if (b == 0) {
            noc_async_write_tile(b, s_ind, ind_addr);
            noc_async_write_barrier();
        } else {
            noc_async_write_tile(b, s_ind, ind_addr);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out_ind, 1);
        }
    }
}
