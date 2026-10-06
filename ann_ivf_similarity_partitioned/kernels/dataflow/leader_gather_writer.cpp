#include "transport.hpp"
#include "tools/profiler/kernel_profiler.hpp"
using namespace ivf_transport;
void kernel_main() {
    const uint32_t scripts = get_arg_val<uint32_t>(0), first = get_arg_val<uint32_t>(1);
    const uint32_t batches = get_arg_val<uint32_t>(2), local = get_arg_val<uint32_t>(3);
    const uint32_t values_base = get_arg_val<uint32_t>(4), indices_base = get_arg_val<uint32_t>(5);
    const uint32_t total_workers = get_arg_val<uint32_t>(6), workers = get_arg_val<uint32_t>(7);
    const uint32_t final_values = get_arg_val<uint32_t>(8), final_indices = get_arg_val<uint32_t>(9);
    const uint32_t leader_x = get_arg_val<uint32_t>(10), leader_y = get_arg_val<uint32_t>(11);
    const uint32_t leader_ctrl = get_arg_val<uint32_t>(12);
    const uint32_t profile_batch = get_arg_val<uint32_t>(13);
    const InterleavedAddrGen<true> values{.bank_base_address = values_base, .page_size = 2048};
    const InterleavedAddrGen<true> indices{.bank_base_address = indices_base, .page_size = 4096};
    const InterleavedAddrGen<true> out_values{.bank_base_address = final_values, .page_size = 2048};
    const InterleavedAddrGen<true> out_indices{.bank_base_address = final_indices, .page_size = 4096};
    ScriptReader script(scripts, local + control_bytes);
    uint32_t cursor = first;
    for (uint32_t b = 0; b < batches; ++b) {
        auto header = script.read(cursor);
        cursor += 1 + header.word[2];
        const uint32_t global_batch = header.word[0];
        header.word[2] = workers;
        push_record(header);
        bool consumed[max_workers]{};
        uint32_t count = 0;
        auto gather_partials = [&] {
            while (count < workers) {
                invalidate_l1_cache();
                for (uint32_t w = 0; w < workers; ++w) if (!consumed[w] && read_word(local + done_offset + w * 32) >= b + 1) {
                    cb_reserve_back(tt::CB::c_in0, 1); cb_reserve_back(tt::CB::c_in1, 1);
                    const uint32_t slot = global_batch * total_workers + get_arg_val<uint32_t>(14 + w);
                    auto read_partial = [&] {
                        noc_async_read_page(slot, values, get_write_ptr(tt::CB::c_in0));
                        noc_async_read_page(slot, indices, get_write_ptr(tt::CB::c_in1));
                        noc_async_read_barrier();
                    };
                    if (global_batch == profile_batch) {
                        DeviceZoneScopedN("Partition gather partial"); read_partial();
                    } else read_partial();
                    cb_push_back(tt::CB::c_in0, 1); cb_push_back(tt::CB::c_in1, 1);
                    consumed[w] = true; ++count;
                }
            }
        };
        if (global_batch == profile_batch) {
            DeviceZoneScopedN("Partition gather"); gather_partials();
        } else gather_partials();
        auto wait_result = [&] { cb_wait_front(tt::CB::c_out0, 1); cb_wait_front(tt::CB::c_out1, 1); };
        if (global_batch == profile_batch) {
            DeviceZoneScopedN("Partition result wait"); wait_result();
        } else wait_result();
        auto write_result = [&] {
            noc_async_write_page(global_batch, out_values, get_read_ptr(tt::CB::c_out0));
            noc_async_write_page(global_batch, out_indices, get_read_ptr(tt::CB::c_out1));
            noc_async_write_barrier();
        };
        if (global_batch == profile_batch) {
            DeviceZoneScopedN("Partition final write"); write_result();
        } else write_result();
        cb_pop_front(tt::CB::c_out0, 1); cb_pop_front(tt::CB::c_out1, 1);
        publish_word(local + scratch_offset + 64, leader_x, leader_y, leader_ctrl + batch_done_offset, b + 1);
    }
}
