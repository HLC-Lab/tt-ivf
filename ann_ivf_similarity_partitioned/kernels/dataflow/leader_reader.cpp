#include "transport.hpp"
#include "tools/profiler/kernel_profiler.hpp"
using namespace ivf_transport;
struct Pending { uint32_t worker, ordinal, active; };
void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr bool relay = get_compile_time_arg_val(1) != 0;
    constexpr bool stagger = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t depth = get_compile_time_arg_val(3);
    static_assert(width > 0 && depth > 0 && depth <= 15);
    const uint32_t scripts = get_arg_val<uint32_t>(0), first = get_arg_val<uint32_t>(1);
    const uint32_t batches = get_arg_val<uint32_t>(2), local = get_arg_val<uint32_t>(3);
    const uint32_t query_base = get_arg_val<uint32_t>(4), data_base = get_arg_val<uint32_t>(5);
    const uint32_t index_base = get_arg_val<uint32_t>(6), workers = get_arg_val<uint32_t>(7);
    const uint32_t ready = get_semaphore(get_arg_val<uint32_t>(8)), banks = get_arg_val<uint32_t>(9);
    const uint32_t controllers = get_arg_val<uint32_t>(10), partition = get_arg_val<uint32_t>(11);
    const uint32_t max_tasks = get_arg_val<uint32_t>(12), multicast = get_arg_val<uint32_t>(13);
    const uint32_t xmin = get_arg_val<uint32_t>(14), ymin = get_arg_val<uint32_t>(15);
    const uint32_t xmax = get_arg_val<uint32_t>(16), ymax = get_arg_val<uint32_t>(17);
    const uint32_t profile_batch = get_arg_val<uint32_t>(18);
    ASSERT(workers > 0 && workers <= max_workers);
    constexpr uint32_t coordinates = 19;
    const InterleavedAddrGen<true> queries{.bank_base_address = query_base, .page_size = width * 2048};
    const InterleavedAddrGen<true> data{.bank_base_address = data_base, .page_size = width * 2048};
    const InterleavedAddrGen<true> indices{.bank_base_address = index_base, .page_size = 4096};
    ScriptReader script(scripts, local + cache_offset);
    const uint32_t query_start = local + role_control_bytes;
    const uint32_t plans_start = query_start + 2 * width * 2048;
    const uint32_t stages_start = plans_start + (max_tasks + 1) * record_bytes;
    uint32_t ticket[max_workers]{};
    uint32_t cursor = first, issued = 0;
    bool prefetched_query = false;
    for (uint32_t b = 0; b < batches; ++b) {
        const Record header = script.read(cursor++);
        ASSERT(header.word[2] <= max_tasks);
        const uint32_t query_slot = query_start + (b % 2) * width * 2048;
        const bool sample_batch = header.word[0] == profile_batch;
        if (!prefetched_query) {
            auto read_query = [&] {
                noc_async_read_page(header.word[0], queries, query_slot);
                noc_async_read_barrier();
            };
            if (sample_batch) {
                DeviceZoneScopedN("Leader query read"); read_query();
            } else read_query();
        }
        prefetched_query = false;
        auto* plans = words(plans_start);
        uint32_t plan_pages = 0;
        for (uint32_t t = 0; t < header.word[2]; ++t) {
            const auto record = script.read(cursor++);
            if constexpr (relay) {
                ASSERT(record.word[0] < workers && record.word[3] > 0);
                if (t) ASSERT(plans[(t - 1) * record_words] <= record.word[0]);
                plan_pages += record.word[3];
            }
            for (uint32_t i = 0; i < record_words; ++i) plans[t * record_words + i] = record.word[i];
        }
        // Reject an inconsistent plan before entering the delivery loop. Its
        // exit count must match exactly the pages represented by these tasks.
        if constexpr (relay) ASSERT(plan_pages == header.word[3]);
        // Ready only concerns this partition's workers, including empty workers.
        auto wait_workers = [&] {
            for (uint32_t w = 0; w < workers; ++w) {
                const uint32_t offer = local + offer_offset + w * offer_stride;
                wait_word(offer + 16, ticket[w] + 1);
                ASSERT(words(offer)[2] == kind_query && words(offer)[3] == header.word[0]);
            }
        };
        if (sample_batch) {
            DeviceZoneScopedN("Leader query credit wait"); wait_workers();
        } else wait_workers();
        auto broadcast_query = [&] {
            bool same_pointer = true;
            const uint32_t target = words(local + offer_offset)[0];
            for (uint32_t w = 1; w < workers; ++w)
                same_pointer &= words(local + offer_offset + w * offer_stride)[0] == target;
            if (multicast && same_pointer) {
                // This reader runs on NoC 1: physical bounds must be reversed
                // before get_noc_multicast_addr mirrors the coordinates.
                noc_async_write_multicast(query_slot, get_noc_multicast_addr(xmax, ymax, xmin, ymin, target), width * 2048, workers);
            } else {
                for (uint32_t w = 0; w < workers; ++w) {
                    const uint32_t x = get_arg_val<uint32_t>(coordinates + w * 2);
                    const uint32_t y = get_arg_val<uint32_t>(coordinates + w * 2 + 1);
                    noc_async_write(query_slot, get_noc_addr(x, y, words(local + offer_offset + w * offer_stride)[0]), width * 2048);
                }
            }
            noc_async_write_barrier();
            for (uint32_t w = 0; w < workers; ++w) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(coordinates + w * 2),
                    get_arg_val<uint32_t>(coordinates + w * 2 + 1), ready), 1);
                ++ticket[w];
            }
            noc_async_atomic_barrier();
        };
        if (sample_batch) {
            DeviceZoneScopedN("Leader query broadcast"); broadcast_query();
        } else broadcast_query();
        if constexpr (relay) {
            uint32_t task[max_workers]{}, end[max_workers]{}, page[max_workers]{};
            uint32_t read_ordinal[max_workers]{}, sent[max_workers]{};
            for (uint32_t w = 0, t = 0; w < workers; ++w) {
                task[w] = t;
                while (t < header.word[2] && plans[t * record_words] == w) ++t;
                end[w] = t;
            }
            Pending pending[depth]{};
            uint32_t delivered = 0, round_robin = 0;
            while (delivered < header.word[3]) {
                invalidate_l1_cache();
                // Reads can be ahead of worker credits. Every stream is issued
                // in order, so a staged later page never lacks its staged head.
                for (uint32_t slot = 0; slot < depth; ++slot) if (!pending[slot].active) {
                    uint32_t chosen = workers, best_rank = 0xffffffffu;
                    for (uint32_t i = 0; i < workers; ++i) {
                        const uint32_t w = (round_robin + i) % workers;
                        if (task[w] == end[w]) continue;
                        const uint32_t candidate_page = plans[task[w] * record_words + 2] + page[w];
                        uint32_t rank = i;
                        if constexpr (stagger) {
                            const uint32_t bank = candidate_page % banks;
                            const uint32_t controller = get_arg_val<uint32_t>(coordinates + 2 * workers + bank);
                            const uint32_t preferred = (partition + issued) % controllers;
                            rank = ((controller + controllers - preferred) % controllers) * (banks * workers) +
                                   ((bank + banks - ((partition + issued / controllers) % banks)) % banks) * workers + i;
                        }
                        if (rank < best_rank) { best_rank = rank; chosen = w; }
                    }
                    if (chosen == workers) break;
                    const uint32_t source = plans[task[chosen] * record_words + 2] + page[chosen];
                    const uint32_t dest = stages_start + slot * (width * 2048 + 4096);
                    auto issue_pair = [&] {
                        read_with_id(data.get_noc_addr(source), dest, width * 2048, slot + 1);
                        read_with_id(indices.get_noc_addr(source), dest + width * 2048, 4096, slot + 1);
                    };
                    if (header.word[0] == profile_batch && chosen == 0 && read_ordinal[chosen] < 16) {
                        DeviceZoneScopedN("Leader issue DRAM pair"); issue_pair();
                    } else issue_pair();
                    pending[slot] = {chosen, read_ordinal[chosen]++, 1};
                    ++issued; round_robin = (chosen + 1) % workers;
                    if (++page[chosen] == plans[task[chosen] * record_words + 3]) { ++task[chosen]; page[chosen] = 0; }
                }
                for (uint32_t slot = 0; slot < depth; ++slot) {
                    auto& item = pending[slot];
                    if (!item.active || item.ordinal != sent[item.worker]) continue;
                    const uint32_t offer = local + offer_offset + item.worker * offer_stride;
                    if (read_word(offer + 16) < ticket[item.worker] + 1) continue;
                    ASSERT(words(offer)[2] == kind_candidate && words(offer)[3] == header.word[0]);
                    // Per-ID completion preserves other slots' outstanding reads.
                    if (header.word[0] == profile_batch && item.worker == 0 && item.ordinal < 16) {
                        DeviceZoneScopedN("Leader read wait"); noc_async_read_barrier_with_trid(slot + 1);
                    } else noc_async_read_barrier_with_trid(slot + 1);
                    const uint32_t src = stages_start + slot * (width * 2048 + 4096);
                    const uint32_t x = get_arg_val<uint32_t>(coordinates + item.worker * 2);
                    const uint32_t y = get_arg_val<uint32_t>(coordinates + item.worker * 2 + 1);
                    auto forward_pair = [&] {
                        noc_async_write(src, get_noc_addr(x, y, words(offer)[0]), width * 2048);
                        noc_async_write(src + width * 2048, get_noc_addr(x, y, words(offer)[1]), 4096);
                        // A slot cannot be reused, or its worker notified,
                        // until BOTH the vectors and full int32 IDs arrive.
                        noc_async_write_barrier();
                    };
                    if (header.word[0] == profile_batch && item.worker == 0 && item.ordinal < 16) {
                        DeviceZoneScopedN("Leader forward pair"); forward_pair();
                    } else forward_pair();
                    noc_semaphore_inc(get_noc_addr(x, y, ready), 1);
                    noc_async_atomic_barrier();
                    ++ticket[item.worker]; ++sent[item.worker]; ++delivered; item.active = 0;
                }
            }
            noc_async_read_barrier();
            noc_async_read_set_trid(0);
        }
        // Prefetch this leader's next query while the current batch is reduced.
        if (b + 1 < batches) {
            const auto next = script.read(cursor);
            auto prefetch_query = [&] {
                noc_async_read_page(next.word[0], queries, query_start + ((b + 1) % 2) * width * 2048);
                noc_async_read_barrier();
            };
            if (next.word[0] == profile_batch) {
                DeviceZoneScopedN("Leader query prefetch"); prefetch_query();
            } else prefetch_query();
            prefetched_query = true;
        }
        if (sample_batch) {
            DeviceZoneScopedN("Leader batch done wait"); wait_word(local + batch_done_offset, b + 1);
        } else wait_word(local + batch_done_offset, b + 1);
    }
}
