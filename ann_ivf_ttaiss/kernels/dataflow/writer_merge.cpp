#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// Writes each assigned batch's final top-32 tile pair to output page
// page_base + batch (original query order: queries 32 * batch ... + 31).
// The first pass of the two-pass search writes into the partial buffers
// (page_base = first merged slot); the final merge uses page_base 0.
void kernel_main() {
    const uint32_t out_val_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_ind_addr = get_arg_val<uint32_t>(3);
    const uint32_t page_base = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out_val = tt::CB::c_out0;
    constexpr uint32_t cb_out_ind = tt::CB::c_out1;
    constexpr uint32_t cb_script = tt::CB::c_intermed5;

    // Published by the reader; read-only, never popped.
    cb_wait_front(cb_script, 1);
    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_script);

    const InterleavedAddrGenFast<true> s_val = {
        .bank_base_address = out_val_addr, .page_size = 2048, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> s_ind = {
        .bank_base_address = out_ind_addr, .page_size = 4096, .data_format = DataFormat::Int32};

    const uint32_t num_batches = script[0];
    const uint32_t nprobe = script[1];
    for (uint32_t b = 0; b < num_batches; b++) {
        const uint32_t batch = script[2 + b * (1 + 32 * nprobe)];
        cb_wait_front(cb_out_val, 1);
        cb_wait_front(cb_out_ind, 1);
        DeviceZoneScopedN("Merge.Write");
        noc_async_write_tile(page_base + batch, s_val, get_read_ptr(cb_out_val));
        noc_async_write_tile(page_base + batch, s_ind, get_read_ptr(cb_out_ind));
        noc_async_write_barrier();
        cb_pop_front(cb_out_val, 1);
        cb_pop_front(cb_out_ind, 1);
    }
}
