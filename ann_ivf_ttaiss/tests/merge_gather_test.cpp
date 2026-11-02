// Run the actual merge reader (per-query row gathering) against byte-level
// simulated DRAM and L1. Every partial tile row carries a unique pattern, so
// each assembled input row can be checked against the partial row it must
// come from (or the sentinel). Also checks the DRAM/L1 alignment of every
// NoC read and that the writer maps batches to output pages.
#include <sys/mman.h>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "../gather_schedule.hpp"
#include "../fine_tile_layout.hpp"

#define tt_l1_ptr
namespace tt::CB {
enum : uint32_t { c_in0 = 0, c_in1 = 1, c_out0 = 16, c_out1 = 17, c_intermed5 = 29, c_intermed6 = 30, c_intermed7 = 31 };
}
enum class DataFormat { Float16_b, Int32 };
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }

uint8_t* arena = nullptr;
constexpr uint32_t arena_bytes = 32u << 20;
uint32_t arena_next = 0;
std::vector<uint32_t> args;
template <class T> T get_arg_val(uint32_t index) { return static_cast<T>(args.at(index)); }

// DRAM: interleaved buffers are host vectors; a NoC address is a pointer.
// Pages start 64-byte aligned, like interleaved DRAM pages.
struct Dram {
    uint32_t page = 0;
    std::vector<uint8_t> storage;
    uint8_t* bytes = nullptr;
    size_t size = 0;
    Dram() = default;
    Dram(uint32_t page_size, size_t total) : page(page_size), storage(total + 64), size(total) {
        bytes = storage.data() + (64 - reinterpret_cast<uintptr_t>(storage.data()) % 64) % 64;
    }
};
std::map<uint32_t, Dram> dram;
template <bool Dram> struct InterleavedAddrGen {
    uint32_t bank_base_address, page_size;
    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) const {
        auto& buffer = dram.at(bank_base_address);
        require(page_size == buffer.page, "Address generator page size differs from the buffer");
        require(size_t(id + 1) * page_size <= buffer.size && offset < page_size, "DRAM read outside buffer");
        return reinterpret_cast<uintptr_t>(buffer.bytes + size_t(id) * page_size + offset);
    }
};
template <bool Dram> struct InterleavedAddrGenFast { uint32_t bank_base_address, page_size; DataFormat data_format; };
uint64_t get_noc_addr(uint32_t local_l1) { return local_l1; }  // arena is mapped below 4 GiB
uint64_t reads = 0;
void noc_async_read(uint64_t source, uint32_t destination, uint32_t bytes) {
    const bool local = source < (uint64_t(1) << 32);
    require(destination % 16 == 0 && source % (local ? 16 : 32) == 0 && source % 16 == destination % 16,
            "Misaligned NoC read");
    std::memcpy(reinterpret_cast<void*>(uintptr_t(destination)), reinterpret_cast<const void*>(uintptr_t(source)), bytes);
    ++reads;
}
template <class Gen> void noc_async_read_page(uint32_t id, const Gen& gen, uint32_t destination) {
    noc_async_read(gen.get_noc_addr(id), destination, gen.page_size);
}
void noc_async_read_barrier() {}
std::map<std::pair<uint32_t, uint32_t>, int> output_pages;
template <class Gen> void noc_async_write_tile(uint32_t id, const Gen& gen, uint32_t) {
    ++output_pages[{gen.bank_base_address, id}];
}
void noc_async_write_barrier() {}

// Circular buffers: sequential FIFOs of real L1 pages; pushed pages are
// snapshotted so later reuse of the same L1 cannot hide errors.
struct CB { uint32_t base, page, capacity, write = 0, read = 0, reserved = 0; std::deque<std::vector<uint8_t>> queue; };
std::map<uint32_t, CB> cbs;
void configure(uint32_t cb, uint32_t page, uint32_t capacity) {
    CB c;
    c.base = uint32_t(reinterpret_cast<uintptr_t>(arena)) + arena_next;
    c.page = page;
    c.capacity = capacity;
    cbs[cb] = std::move(c);
    arena_next += page * capacity;
    require(arena_next <= arena_bytes, "Arena exhausted");
}
void cb_reserve_back(uint32_t cb, uint32_t n) {
    auto& c = cbs.at(cb);
    require(n <= c.capacity && c.write + n <= c.capacity, "Reservation exceeds or wraps the CB");
    c.reserved = n;
}
uint32_t get_write_ptr(uint32_t cb) { auto& c = cbs.at(cb); return c.base + c.write * c.page; }
uint32_t get_read_ptr(uint32_t cb) { auto& c = cbs.at(cb); return c.base + c.read * c.page; }
void cb_push_back(uint32_t cb, uint32_t n) {
    auto& c = cbs.at(cb);
    require(c.reserved >= n, "Push without reservation");
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* page = reinterpret_cast<const uint8_t*>(uintptr_t(c.base + ((c.write + i) % c.capacity) * c.page));
        c.queue.emplace_back(page, page + c.page);
    }
    c.write = (c.write + n) % c.capacity;
    c.reserved = 0;
}
void cb_wait_front(uint32_t cb, uint32_t n) { require(cbs.at(cb).queue.size() >= n, "Wait for unproduced tiles"); }
void cb_pop_front(uint32_t cb, uint32_t n) {
    auto& c = cbs.at(cb);
    require(c.queue.size() >= n, "Pop of unpublished tiles");
    for (uint32_t i = 0; i < n; ++i) c.queue.pop_front();
    c.read = (c.read + n) % c.capacity;
}

namespace gather_reader {
#define kernel_main run
#include "../kernels/dataflow/reader_merge.cpp"
#undef kernel_main
}
namespace gather_writer {
#define kernel_main run
#include "../kernels/dataflow/writer_merge.cpp"
#undef kernel_main
}

using namespace ivf_cluster_gather;
constexpr uint32_t partial_val_base = 0x5000, partial_ind_base = 0x6000, script_base = 0x7000;
constexpr uint32_t out_val_base = 0x8000, out_ind_base = 0x9000;
uint16_t value_of(uint32_t tile, uint32_t row, uint32_t rank) { return uint16_t((tile * 37 + row * 5 + rank) & 0x7FFF); }
uint32_t id_of(uint32_t tile, uint32_t row, uint32_t rank) { return (tile << 10) | (row << 5) | rank; }

void check(uint32_t queries, uint32_t nprobe, uint32_t cores, uint32_t seed, uint32_t page_base) {
    std::mt19937 rng(seed);
    const uint32_t nlist = 48;
    std::vector<ListLayout> lists(nlist);
    for (uint32_t c = 0, start = 0; c < nlist; ++c) {
        const uint32_t count = rng() % 6 == 0 ? 0 : 1 + rng() % 90;
        lists[c] = {start, count, std::max(1u, (count + 31) / 32) * 32};
        start += lists[c].padded_count / 32;
    }
    std::vector<std::vector<uint32_t>> selections(queries);
    for (auto& row : selections) {
        std::set<uint32_t> chosen;
        while (chosen.size() < nprobe) chosen.insert(rng() % nlist);
        row.assign(chosen.begin(), chosen.end());
        std::shuffle(row.begin(), row.end(), rng);
    }
    const auto plan = plan_gather(selections, nprobe, lists, 32, cores, cores);
    const uint32_t tiles = std::max<uint32_t>(1, plan.tiles.size());

    // Partial tiles in row layout: element (row, rank) at fine_tile_offset.
    dram.clear();
    dram.emplace(partial_val_base, Dram(2048, size_t(tiles) * 2048));
    dram.emplace(partial_ind_base, Dram(4096, size_t(tiles) * 4096));
    for (uint32_t t = 0; t < tiles; ++t)
        for (uint32_t row = 0; row < 32; ++row)
            for (uint32_t rank = 0; rank < 32; ++rank) {
                const uint32_t offset = fine_tile_offset(row, rank);
                const uint16_t v = value_of(t, row, rank);
                const uint32_t id = id_of(t, row, rank);
                std::memcpy(dram.at(partial_val_base).bytes + size_t(t) * 2048 + 2 * offset, &v, 2);
                std::memcpy(dram.at(partial_ind_base).bytes + size_t(t) * 4096 + 4 * offset, &id, 4);
            }
    std::vector<std::vector<uint32_t>> scripts(cores);
    size_t max_words = 0;
    for (uint32_t m = 0; m < cores; ++m) {
        scripts[m] = encode_merge_script(plan, m);
        max_words = std::max(max_words, scripts[m].size());
    }
    const uint32_t page = uint32_t(ivf_cluster_major::script_page_bytes(max_words));
    dram.emplace(script_base, Dram(page, size_t(cores) * page));
    for (uint32_t m = 0; m < cores; ++m)
        std::memcpy(dram.at(script_base).bytes + size_t(m) * page, scripts[m].data(), scripts[m].size() * 4);

    output_pages.clear();
    for (uint32_t m = 0; m < cores; ++m) {
        cbs.clear();
        arena_next = 0;
        configure(tt::CB::c_in0, 2048, 2);
        configure(tt::CB::c_in1, 4096, 2);
        configure(tt::CB::c_intermed5, page, 1);
        configure(tt::CB::c_intermed6, 2048, 1);
        configure(tt::CB::c_intermed7, 4096, 1);
        configure(tt::CB::c_out0, 2048, 2);
        configure(tt::CB::c_out1, 4096, 2);
        args = {partial_val_base, partial_ind_base, out_val_base, out_ind_base, script_base, page, m, page_base};
        gather_reader::run();
        // Check every assembled input tile (batch order, j-major).
        const auto& batches = plan.merge_batches[m];
        require(cbs[tt::CB::c_in0].queue.size() == batches.size() * nprobe &&
                cbs[tt::CB::c_in1].queue.size() == batches.size() * nprobe, "Wrong number of input tiles");
        for (uint32_t b : batches)
            for (uint32_t j = 0; j < nprobe; ++j) {
                const auto values = cbs[tt::CB::c_in0].queue.front();
                const auto ids = cbs[tt::CB::c_in1].queue.front();
                cbs[tt::CB::c_in0].queue.pop_front();
                cbs[tt::CB::c_in1].queue.pop_front();
                for (uint32_t row = 0; row < 32; ++row) {
                    const uint64_t q = uint64_t(b) * 32 + row;
                    const uint32_t entry = q < queries ? plan.location[q * nprobe + j] : sentinel_entry;
                    for (uint32_t rank = 0; rank < 32; ++rank) {
                        const uint32_t offset = fine_tile_offset(row, rank);
                        uint16_t v; uint32_t id;
                        std::memcpy(&v, values.data() + 2 * offset, 2);
                        std::memcpy(&id, ids.data() + 4 * offset, 4);
                        if (entry == sentinel_entry) {
                            require(v == 0xC61C && id == 0xFFFFFFFFu, "Missing partial is not a sentinel row");
                        } else {
                            require(v == value_of(entry / 32, entry % 32, rank) && id == id_of(entry / 32, entry % 32, rank),
                                    "Assembled row does not match its partial row");
                        }
                    }
                }
            }
        // Writer: one output page per assigned batch, in script order.
        // (Sequential model: outputs are pushed first; capacity 2 is not a
        // limit here because the writer is the only consumer.)
        cbs[tt::CB::c_out0].capacity = cbs[tt::CB::c_out1].capacity = uint32_t(std::max<size_t>(2, batches.size()));
        for (size_t i = 0; i < batches.size(); ++i) {
            cb_reserve_back(tt::CB::c_out0, 1); cb_push_back(tt::CB::c_out0, 1);
            cb_reserve_back(tt::CB::c_out1, 1); cb_push_back(tt::CB::c_out1, 1);
        }
        gather_writer::run();
        require(cbs[tt::CB::c_out0].queue.empty() && cbs[tt::CB::c_out1].queue.empty(), "Writer left outputs");
    }
    for (uint32_t b = 0; b < plan.batches; ++b)
        require(output_pages[{out_val_base, page_base + b}] == 1 && output_pages[{out_ind_base, page_base + b}] == 1,
                "Output page not written exactly once");
    require(output_pages.size() == 2 * size_t(plan.batches), "Output pages written outside page_base + batch");
}

int main() {
    if (void* mapped = mmap(nullptr, arena_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS
#ifdef MAP_32BIT
                            | MAP_32BIT
#endif
                            , -1, 0); mapped != MAP_FAILED && uintptr_t(mapped) + arena_bytes <= UINT32_MAX) {
        arena = static_cast<uint8_t*>(mapped);
    } else {
        std::cout << "Skipped: kernels cast 32-bit L1 addresses and no low mapping is available\n";
        return 0;
    }
    try {
        uint32_t seed = 9;
        for (uint32_t queries : {1u, 32u, 45u, 400u})
            for (uint32_t nprobe : {1u, 3u, 16u})
                for (uint32_t cores : {1u, 5u})
                    for (uint32_t page_base : {0u, 77u})  // final merge / first-pass merge into partial slots
                        check(queries, nprobe, cores, seed++, page_base);
        std::cout << "Merge reader assembles every query row from its partial row or a sentinel (" << reads
                  << " NoC reads checked)\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
