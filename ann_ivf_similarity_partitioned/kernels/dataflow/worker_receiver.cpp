#include "transport.hpp"
#include "tools/profiler/kernel_profiler.hpp"
using namespace ivf_transport;

// DMA directly into unpublished input-CB slots. cb_reserve_back checks free
// space but does not advance the write pointer or account for pending DMA;
// reservations must therefore include every outstanding, unpublished page.
template <uint32_t width, uint32_t input_pages>
inline void prefetch_direct_list(const Record& list, const InterleavedAddrGen<true>& data,
                                const InterleavedAddrGen<true>& ids, uint32_t data_ring_base,
                                uint32_t id_ring_base, uint32_t batch, uint32_t profile_batch,
                                uint32_t& ordinal, bool additive) {
    static_assert(width > 0 && input_pages > 0);
    constexpr uint32_t window = input_pages < 15 ? input_pages : 15; // ID 0 is for scripts/queries.
    constexpr uint32_t vector_bytes = width * 2048;
    constexpr uint32_t data_ring_bytes = input_pages * vector_bytes;
    constexpr uint32_t id_ring_bytes = input_pages * 4096;
    uint32_t issued = 0, published = 0;
    while (published < list.word[2]) {
        while (issued < list.word[2] && issued - published < window) {
            const uint32_t pending = issued - published;
            const uint32_t reserved_pages = pending + 1;
            // Publish the oldest read instead of blocking behind already
            // published pages when there are still outstanding reads.
            if (pending && (!cb_pages_reservable_at_back(tt::CB::c_in1, reserved_pages * width) ||
                            !cb_pages_reservable_at_back(tt::CB::c_in2, reserved_pages))) break;
            const bool sample_issue = batch == profile_batch && ordinal + pending < 16;
            auto reserve_pair = [&] {
                cb_reserve_back(tt::CB::c_in1, reserved_pages * width);
                cb_reserve_back(tt::CB::c_in2, reserved_pages);
            };
            if (sample_issue) {
                DeviceZoneScopedN("Worker free buffer wait"); reserve_pair();
            } else reserve_pair();
            uint32_t vector_offset = get_write_ptr(tt::CB::c_in1) - data_ring_base + pending * vector_bytes;
            uint32_t id_offset = get_write_ptr(tt::CB::c_in2) - id_ring_base + pending * 4096;
            if (vector_offset >= data_ring_bytes) vector_offset -= data_ring_bytes;
            if (id_offset >= id_ring_bytes) id_offset -= id_ring_bytes;
            ASSERT(vector_offset < data_ring_bytes && id_offset < id_ring_bytes);
            const uint32_t tag = issued % window + 1;
            auto issue = [&] {
                read_with_id(data.get_noc_addr(list.word[1] + issued), data_ring_base + vector_offset, vector_bytes, tag);
                read_with_id(ids.get_noc_addr(list.word[1] + issued), id_ring_base + id_offset, 4096, tag);
                noc_async_read_set_trid(0);
            };
            if (sample_issue) {
                DeviceZoneScopedN("Worker issue DRAM pair"); issue();
            } else issue();
            ++issued;
        }
        ASSERT(issued > published && issued - published <= window);
        // This tag covers BOTH the vectors and their full int32 ID tile.
        // Other tags remain in flight; completed slots are published FIFO.
        if (batch == profile_batch && ordinal < 16) {
            DeviceZoneScopedN("Worker read wait"); noc_async_read_barrier_with_trid(published % window + 1);
        } else noc_async_read_barrier_with_trid(published % window + 1);
        if (additive && needs_mask(published, list.word[2], list.word[4])) make_mask(list.word[4]);
        cb_push_back(tt::CB::c_in1, width);
        cb_push_back(tt::CB::c_in2, 1);
        ++published; ++ordinal;
    }
    // All tags are retired before the next list record or query is fetched.
    ASSERT(issued == published);
}

void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr bool relay = get_compile_time_arg_val(1) != 0;
    constexpr bool additive = get_compile_time_arg_val(2) == static_cast<uint32_t>(MaskMode::Additive);
    constexpr uint32_t input_pages = get_compile_time_arg_val(3);
    constexpr bool direct_prefetch = get_compile_time_arg_val(4) != 0;
    const uint32_t script_base = get_arg_val<uint32_t>(0), first = get_arg_val<uint32_t>(1);
    const uint32_t batches = get_arg_val<uint32_t>(2), local = get_arg_val<uint32_t>(3);
    const uint32_t leader_x = get_arg_val<uint32_t>(4), leader_y = get_arg_val<uint32_t>(5);
    const uint32_t leader_ctrl = get_arg_val<uint32_t>(6), worker_id = get_arg_val<uint32_t>(7);
    const uint32_t ready = get_semaphore(get_arg_val<uint32_t>(8));
    const uint32_t data_base = get_arg_val<uint32_t>(9), index_base = get_arg_val<uint32_t>(10);
    const uint32_t profile_batch = get_arg_val<uint32_t>(11);
    const InterleavedAddrGen<true> data{.bank_base_address = data_base, .page_size = width * 2048};
    const InterleavedAddrGen<true> ids{.bank_base_address = index_base, .page_size = 4096};
    ScriptReader script(script_base, local + cache_offset);
    uint32_t data_ring_base = 0, id_ring_base = 0;
    if constexpr (!relay && direct_prefetch) {
        // At kernel entry both CB write pointers are at their allocation base.
        // Reserve before querying them; reservation does not move the pointers.
        cb_reserve_back(tt::CB::c_in1, width);
        cb_reserve_back(tt::CB::c_in2, 1);
        data_ring_base = get_write_ptr(tt::CB::c_in1);
        id_ring_base = get_write_ptr(tt::CB::c_in2);
    }
    uint32_t cursor = first, ticket = 0;
    auto offer = [&](uint32_t kind, uint32_t batch, uint32_t value, uint32_t index) {
        auto* msg = words(local + scratch_offset);
        msg[0] = value; msg[1] = index; msg[2] = kind; msg[3] = batch;
        const uint32_t remote = leader_ctrl + offer_offset + worker_id * offer_stride;
        noc_async_write(local + scratch_offset, get_noc_addr(leader_x, leader_y, remote), 16);
        noc_async_write_barrier();
        publish_word(local + scratch_offset + 32, leader_x, leader_y, remote + 16, ++ticket);
        noc_semaphore_wait_min(words(ready), ticket);
    };
    for (uint32_t b = 0; b < batches; ++b) {
        const Record header = script.read(cursor++);
        uint32_t ordinal = 0;
        push_record(header);
        auto receive_query = [&] {
            cb_reserve_back(tt::CB::c_in0, width);
            offer(kind_query, header.word[0], get_write_ptr(tt::CB::c_in0), 0);
        };
        if (header.word[0] == profile_batch) {
            DeviceZoneScopedN("Worker query receive"); receive_query();
        } else receive_query();
        cb_push_back(tt::CB::c_in0, width);
        for (uint32_t task = 0; task < header.word[2]; ++task) {
            const Record list = script.read(cursor++);
            push_record(list);
            if constexpr (!relay && direct_prefetch) {
                prefetch_direct_list<width, input_pages>(list, data, ids, data_ring_base, id_ring_base,
                                                        header.word[0], profile_batch, ordinal, additive);
                continue;
            }
            for (uint32_t page = 0; page < list.word[2]; ++page, ++ordinal) {
                const bool sample = header.word[0] == profile_batch && ordinal < 16;
                auto reserve_pair = [&] {
                    cb_reserve_back(tt::CB::c_in1, width);
                    cb_reserve_back(tt::CB::c_in2, 1);
                };
                if (sample) {
                    DeviceZoneScopedN("Worker free buffer wait"); reserve_pair();
                } else reserve_pair();
                // Produce the optional mask before offering a candidate slot.
                // Compute waits for candidate publication before consuming it.
                if constexpr (additive) {
                    if (needs_mask(page, list.word[2], list.word[4])) make_mask(list.word[4]);
                }
                if constexpr (relay) {
                    auto receive_pair = [&] {
                        offer(kind_candidate, header.word[0], get_write_ptr(tt::CB::c_in1), get_write_ptr(tt::CB::c_in2));
                    };
                    if (sample) {
                        DeviceZoneScopedN("Worker relay receive wait"); receive_pair();
                    } else receive_pair();
                } else {
                    auto issue = [&] {
                        noc_async_read_page(list.word[1] + page, data, get_write_ptr(tt::CB::c_in1));
                        noc_async_read_page(list.word[1] + page, ids, get_write_ptr(tt::CB::c_in2));
                    };
                    if (sample) {
                        DeviceZoneScopedN("Worker issue DRAM pair"); issue();
                    } else issue();
                    if (sample) {
                        DeviceZoneScopedN("Worker read wait"); noc_async_read_barrier();
                    } else noc_async_read_barrier();
                }
                cb_push_back(tt::CB::c_in1, width);
                cb_push_back(tt::CB::c_in2, 1);
            }
        }
    }
    noc_async_atomic_barrier();
}
