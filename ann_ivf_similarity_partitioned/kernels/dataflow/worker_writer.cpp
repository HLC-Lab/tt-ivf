#include "transport.hpp"
#include "tools/profiler/kernel_profiler.hpp"
using namespace ivf_transport;
void kernel_main() {
    const uint32_t batches = get_arg_val<uint32_t>(0), slot = get_arg_val<uint32_t>(1);
    const uint32_t total_workers = get_arg_val<uint32_t>(2), value_base = get_arg_val<uint32_t>(3);
    const uint32_t index_base = get_arg_val<uint32_t>(4), agg_x = get_arg_val<uint32_t>(5);
    const uint32_t agg_y = get_arg_val<uint32_t>(6), agg_ctrl = get_arg_val<uint32_t>(7);
    const uint32_t worker = get_arg_val<uint32_t>(8), local = get_arg_val<uint32_t>(9);
    const uint32_t script_base = get_arg_val<uint32_t>(10), first = get_arg_val<uint32_t>(11);
    const uint32_t profile_batch = get_arg_val<uint32_t>(12);
    const InterleavedAddrGen<true> values{.bank_base_address = value_base, .page_size = 2048};
    const InterleavedAddrGen<true> ids{.bank_base_address = index_base, .page_size = 4096};
    // BRISC has a separate 4 KiB script cache; never overwrite NCRISC's cache.
    ScriptReader script(script_base, local + control_bytes);
    uint32_t cursor = first;
    for (uint32_t b = 0; b < batches; ++b) {
        const Record header = script.read(cursor);
        cursor += 1 + header.word[2];
        auto wait_output = [&] {
            cb_wait_front(tt::CB::c_out0, 1);
            cb_wait_front(tt::CB::c_out1, 1);
        };
        if (header.word[0] == profile_batch) {
            DeviceZoneScopedN("Worker result wait"); wait_output();
        } else wait_output();
        auto write_partial = [&] {
            const uint32_t page = header.word[0] * total_workers + slot;
            noc_async_write_page(page, values, get_read_ptr(tt::CB::c_out0));
            noc_async_write_page(page, ids, get_read_ptr(tt::CB::c_out1));
            noc_async_write_barrier();
            // Completion is visible only after both DRAM payloads are complete.
            publish_word(local + scratch_offset + 64, agg_x, agg_y, agg_ctrl + done_offset + worker * 32, b + 1);
        };
        if (header.word[0] == profile_batch) {
            DeviceZoneScopedN("Worker partial write"); write_partial();
        } else write_partial();
        cb_pop_front(tt::CB::c_out0, 1);
        cb_pop_front(tt::CB::c_out1, 1);
    }
}
