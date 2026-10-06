#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    uint32_t worker_idx = get_arg_val<uint32_t>(0);
    uint32_t core_0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t core_0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t slot_val_dram_base = get_arg_val<uint32_t>(3);
    uint32_t slot_ind_dram_base = get_arg_val<uint32_t>(4);
    uint32_t done_sem_addr = get_semaphore(get_arg_val<uint32_t>(5));
    uint32_t ready_sem_addr = get_semaphore(get_arg_val<uint32_t>(6));
    uint32_t num_batches = get_arg_val<uint32_t>(7);

    volatile tt_l1_ptr uint32_t* ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);
    noc_semaphore_set(ready_sem, 0);

    constexpr uint32_t cb_out0 = tt::CB::c_out0;
    constexpr uint32_t cb_out1 = tt::CB::c_out1;

    const InterleavedAddrGenFast<true> slot_values = {
        .bank_base_address = slot_val_dram_base,
        .page_size = 2048,
        .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> slot_indices = {
        .bank_base_address = slot_ind_dram_base,
        .page_size = 4096,
        .data_format = DataFormat::Int32};

    for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
        noc_semaphore_wait_min(ready_sem, batch_id);

        cb_wait_front(cb_out0, 1);
        cb_wait_front(cb_out1, 1);

        uint32_t val_src = get_read_ptr(cb_out0);
        uint32_t ind_src = get_read_ptr(cb_out1);


        {
            DeviceZoneScopedN("Send Res");
            noc_async_write_tile(batch_id * 63 + worker_idx, slot_values, val_src);
            noc_async_write_tile(batch_id * 63 + worker_idx, slot_indices, ind_src);
            noc_async_write_barrier();
        }

        {
            uint64_t sem_dest = get_noc_addr(core_0_noc_x, core_0_noc_y, done_sem_addr);
            noc_semaphore_inc(sem_dest, 1);
            cb_pop_front(cb_out0, 1);
            cb_pop_front(cb_out1, 1);
        }
    }
    noc_async_atomic_barrier();
}
