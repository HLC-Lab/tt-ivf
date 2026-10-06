#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t slot_val_l1_base = get_arg_val<uint32_t>(0);
    uint32_t slot_ind_l1_base = get_arg_val<uint32_t>(1);
    uint32_t done_sem_addr = get_semaphore(get_arg_val<uint32_t>(2));
    uint32_t ready_sem_addr = get_semaphore(get_arg_val<uint32_t>(3));

    volatile tt_l1_ptr uint32_t* script =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(4));
    volatile tt_l1_ptr uint32_t* done_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);
    // CreateSemaphore initializes this counter before launch. Resetting it
    // here can erase an increment from a worker that starts first.

    uint32_t total_batches = script[0];
    uint32_t ptr = 1 + 64 * 2;

    constexpr uint32_t cb_val_in = tt::CB::c_in0;
    constexpr uint32_t cb_ind_in = tt::CB::c_in1;
    constexpr uint32_t val_size = 2048;
    constexpr uint32_t ind_size = 4096;

    uint32_t expected_done_signals = 0;
    uint32_t num_active_workers = script[ptr];

    for (uint32_t batch_id = 0; batch_id < total_batches; batch_id++) {
        expected_done_signals += num_active_workers;
        {
            noc_semaphore_wait_min(done_sem, expected_done_signals);
        }

        for (uint32_t worker_idx = 0; worker_idx < num_active_workers; worker_idx++) {
            cb_reserve_back(cb_val_in, 1);
            cb_reserve_back(cb_ind_in, 1);

            uint32_t dest_val = get_write_ptr(cb_val_in);
            uint32_t dest_ind = get_write_ptr(cb_ind_in);
            const uint64_t val_src =
                get_noc_addr(slot_val_l1_base + worker_idx * val_size);
            const uint64_t ind_src =
                get_noc_addr(slot_ind_l1_base + worker_idx * ind_size);

            {
                if (batch_id == 0 && worker_idx == 0) {
                    noc_async_read(val_src, dest_val, val_size);
                    noc_async_read(ind_src, dest_ind, ind_size);
                    noc_async_read_barrier();
                } else {
                    noc_async_read(val_src, dest_val, val_size);
                    noc_async_read(ind_src, dest_ind, ind_size);
                    noc_async_read_barrier();
                }
            }

            {
                cb_push_back(cb_val_in, 1);
                cb_push_back(cb_ind_in, 1);
            }
        }

        for (uint32_t worker_idx = 0; worker_idx < num_active_workers; worker_idx++) {
            uint32_t worker_core = worker_idx + 1;
            uint32_t worker_noc_x = script[1 + worker_core];
            uint32_t worker_noc_y = script[1 + 64 + worker_core];

            {
                uint64_t sem_dest = get_noc_addr(worker_noc_x, worker_noc_y, ready_sem_addr);
                noc_semaphore_inc(sem_dest, 1);
            }
        }
    }
    noc_async_atomic_barrier();
}
