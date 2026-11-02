#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// Writes one partial top-32 tile pair per (list, batch) pair to its DRAM slot.
void kernel_main() {
    const uint32_t partial_val_addr = get_arg_val<uint32_t>(0);
    const uint32_t partial_ind_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out0 = tt::CB::c_out0;
    constexpr uint32_t cb_out1 = tt::CB::c_out1;
    constexpr uint32_t cb_script = tt::CB::c_intermed5;

    // Published by the reader; read-only, never popped.
    cb_wait_front(cb_script, 1);
    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_script);

    const InterleavedAddrGenFast<true> partial_values = {
        .bank_base_address = partial_val_addr, .page_size = 2048, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> partial_indices = {
        .bank_base_address = partial_ind_addr, .page_size = 4096, .data_format = DataFormat::Int32};

    const uint32_t num_tasks = script[0];
    uint32_t ptr = 1;
    for (uint32_t task = 0; task < num_tasks; task++) {
        ptr += 3;  // start block, count, pages
        const uint32_t pairs = script[ptr++];
        for (uint32_t i = 0; i < pairs; i++) {
            const uint32_t slot = script[ptr + 2 * i + 1];
            cb_wait_front(cb_out0, 1);
            cb_wait_front(cb_out1, 1);
            DeviceZoneScopedN("Scan.Partial");
            noc_async_write_tile(slot, partial_values, get_read_ptr(cb_out0));
            noc_async_write_tile(slot, partial_indices, get_read_ptr(cb_out1));
            // The CB pages are reused by compute after the pop.
            noc_async_write_barrier();
            cb_pop_front(cb_out0, 1);
            cb_pop_front(cb_out1, 1);
        }
        ptr += 2 * pairs;
    }
}
