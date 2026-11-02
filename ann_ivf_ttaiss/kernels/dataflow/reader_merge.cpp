#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "../../fine_tile_layout.hpp"

// Per-query merge input assembly. Fixed runtime args: partial value/index
// buffers, out value/index buffers (writer only), merge-script buffer, its
// page size and this core's page. Script (gather_schedule.hpp):
//   [batches, nprobe, then per batch: batch, nprobe x 32 entries]
// Entry (j, q') = gathered tile * 32 + row holding query q''s j-th partial
// (row layout, columns = ranks), or 0xFFFFFFFF for a sentinel row. Input tile
// j of a batch gets entry (j, q') in its row q'. A tile row is two 16-element
// face segments: 32 bytes of BF16 values and 64 bytes of int32 IDs, each at a
// 32-byte-aligned offset in the DRAM page and in the destination CB page.
void kernel_main() {
    const uint32_t partial_val_addr = get_arg_val<uint32_t>(0);
    const uint32_t partial_ind_addr = get_arg_val<uint32_t>(1);
    const uint32_t script_addr = get_arg_val<uint32_t>(4);
    const uint32_t script_page_bytes = get_arg_val<uint32_t>(5);
    const uint32_t core_idx = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_val_in = tt::CB::c_in0;
    constexpr uint32_t cb_ind_in = tt::CB::c_in1;
    constexpr uint32_t cb_script = tt::CB::c_intermed5;
    constexpr uint32_t cb_sentinel_val = tt::CB::c_intermed6;
    constexpr uint32_t cb_sentinel_ind = tt::CB::c_intermed7;
    constexpr uint32_t sentinel_entry = 0xFFFFFFFFu;

    // Published once; compute and writer read it, nobody pops it.
    const InterleavedAddrGen<true> s_script = {.bank_base_address = script_addr, .page_size = script_page_bytes};
    cb_reserve_back(cb_script, 1);
    const uint32_t script_l1 = get_write_ptr(cb_script);
    noc_async_read_page(core_idx, s_script, script_l1);
    noc_async_read_barrier();
    cb_push_back(cb_script, 1);
    volatile tt_l1_ptr uint32_t* script = (volatile tt_l1_ptr uint32_t*)script_l1;

    // Reader-private sentinel tiles (-10000 BF16, 0xFFFFFFFF IDs): the source
    // of rows for missing partials. Reserved, filled by stores, never pushed.
    cb_reserve_back(cb_sentinel_val, 1);
    cb_reserve_back(cb_sentinel_ind, 1);
    const uint32_t sentinel_val = get_write_ptr(cb_sentinel_val);
    const uint32_t sentinel_ind = get_write_ptr(cb_sentinel_ind);
    volatile tt_l1_ptr uint16_t* sv = (volatile tt_l1_ptr uint16_t*)sentinel_val;
    volatile tt_l1_ptr uint32_t* si = (volatile tt_l1_ptr uint32_t*)sentinel_ind;
    for (uint32_t i = 0; i < 1024; ++i) {
        sv[i] = 0xC61C;
        si[i] = 0xFFFFFFFFu;
    }

    const InterleavedAddrGen<true> partial_values = {.bank_base_address = partial_val_addr, .page_size = 2048};
    const InterleavedAddrGen<true> partial_indices = {.bank_base_address = partial_ind_addr, .page_size = 4096};

    const uint32_t num_batches = script[0];
    const uint32_t nprobe = script[1];
    uint32_t ptr = 2;
    for (uint32_t b = 0; b < num_batches; b++) {
        ptr++;  // batch id: writer only
        for (uint32_t j = 0; j < nprobe; j++) {
            DeviceZoneScopedN("Merge.Gather");
            cb_reserve_back(cb_val_in, 1);
            cb_reserve_back(cb_ind_in, 1);
            const uint32_t dst_val = get_write_ptr(cb_val_in);
            const uint32_t dst_ind = get_write_ptr(cb_ind_in);
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t entry = script[ptr++];
                for (uint32_t half = 0; half < 2; half++) {
                    const uint32_t dst_offset = fine_tile_offset(row, 16 * half);
                    if (entry == sentinel_entry) {
                        // Every element of the sentinel tile is equal, so any
                        // segment works; use the same offset (local L1 read).
                        noc_async_read(get_noc_addr(sentinel_val + 2 * dst_offset), dst_val + 2 * dst_offset, 32);
                        noc_async_read(get_noc_addr(sentinel_ind + 4 * dst_offset), dst_ind + 4 * dst_offset, 64);
                    } else {
                        const uint32_t tile = entry >> 5;
                        const uint32_t src_offset = fine_tile_offset(entry & 31u, 16 * half);
                        noc_async_read(partial_values.get_noc_addr(tile, 2 * src_offset), dst_val + 2 * dst_offset, 32);
                        noc_async_read(partial_indices.get_noc_addr(tile, 4 * src_offset), dst_ind + 4 * dst_offset, 64);
                    }
                }
            }
            // All 32 rows of both tiles have landed before compute sees them.
            noc_async_read_barrier();
            cb_push_back(cb_val_in, 1);
            cb_push_back(cb_ind_in, 1);
        }
    }
}
