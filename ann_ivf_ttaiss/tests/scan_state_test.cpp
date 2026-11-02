// Run the actual scan/merge kernel sources (reader, compute, writer) on the
// host against a tagged circular-buffer model. Tiles carry provenance tags
// (query batch, list page, champion state), so the test checks protocol and
// data routing: CB reserve/push/wait/pop balance, no multi-tile reservation
// across the CB end, champion ring order and occupancy, initialize_champion
// only on empty CBs, SrcA/pack formats, partial slot routing and the per-batch
// merge. The reader's device-side assembly of gathered query tiles is checked
// segment by segment: every row of every assembled tile must be the right
// query's 32-byte face segment of the right column tile (or zeros for an
// unused row), read with aligned 32-byte NoC reads. Threshold mode (second
// pass of the two-pass search) is checked too: threshold tiles hold each row's
// query code, every (tile, page) gets exactly one sub -> reduce -> flag check,
// skipped pages never reach the champion, the full ring rotates without data
// movement, and indexed tile reads never run past the CB end. CB data lives in
// slots: a push without a pack republishes whatever the slot already holds.
// It does not emulate Tensix arithmetic, NoC timing or concurrency.
#include <sys/mman.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "../query_grouping.hpp"
#include "../cluster_schedule.hpp"
#include "../gather_schedule.hpp"
#include "../fine_tile_layout.hpp"

#define tt_l1_ptr
#define UNPACK(...) do { __VA_ARGS__ } while (false)
#define PACK(...) do { __VA_ARGS__ } while (false)
namespace tt::CB {
enum : uint32_t { c_in0 = 0, c_in1 = 1, c_in2 = 2, c_in3 = 3, c_in4 = 4, c_in5 = 5, c_out0 = 16, c_out1 = 17, c_out2 = 18,
                  c_intermed0 = 24, c_intermed1 = 25, c_intermed2 = 26,
                  c_intermed3 = 27, c_intermed4 = 28, c_intermed5 = 29, c_intermed6 = 30, c_intermed7 = 31 };
}
enum class DataFormat { Float16_b, Int32 };

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

// ---- tiles, DRAM and L1 -----------------------------------------------------
enum class Kind { Garbage, Script, Query, Data, Index, Score, Champion, Sentinel, Partial, Threshold, Scaler, Diff, Flag };
struct Tag {
    Kind kind = Kind::Garbage;
    int64_t batch = -1, page = -1, sub = -1;  // Query: sub=k; Data: sub=k; Score: sub=k mask
    int64_t count = 0, first = -1, last = -1, sum = 0;  // champion accumulation
    bool ids = false, transposed = false;
};
bool same_champion(const Tag& a, const Tag& b) {
    return a.batch == b.batch && a.count == b.count && a.first == b.first && a.last == b.last && a.sum == b.sum;
}

uint8_t* arena = nullptr;
constexpr uint32_t arena_bytes = 64u << 20;
struct Buffer { uint32_t page; bool tagged; const std::vector<uint32_t>* words = nullptr; };
std::map<uint32_t, Buffer> dram;                         // base -> buffer kind
std::map<std::pair<uint32_t, uint32_t>, Tag> dram_tags;  // written partial/output pages
std::map<std::pair<uint32_t, uint32_t>, uint32_t> dram_writes;
constexpr uint32_t query_base = 0x1000, dataset_base = 0x2000, index_base = 0x3000, script_base = 0x4000;
constexpr uint32_t merge_script_base = 0x9000, tile_queries_base = 0xA000;
// Threshold mode: thresholds come from partial_val pages threshold_base + q / 32.
bool threshold_model = false;
uint32_t threshold_base = 0;
constexpr uint16_t unused_threshold_code = 0x7F7F;
uint16_t threshold_code(uint32_t q) { return uint16_t(0x3C00 | (q & 0x3FF)); }
// Merge decision of (gathered tile, page), as returned by the flag read.
bool decide(int64_t tile, int64_t page) { return ((uint64_t(tile) * 2654435761u) ^ (uint64_t(page) * 40503u)) % 3 == 0; }
std::map<std::pair<int64_t, int64_t>, bool> decisions;
const std::vector<uint32_t>* tile_query_table = nullptr;  // [tile][32] query IDs, UINT32_MAX = unused
// Assembled query rows: (L1 tile address, row, half) -> (query, column tile).
std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>> query_segments;
std::map<uint32_t, uint32_t> id_pages;  // L1 address of an ID page -> gathered tile
constexpr uint32_t partial_val_base = 0x5000, partial_ind_base = 0x6000, out_val_base = 0x7000, out_ind_base = 0x8000;

struct ModelCB {
    uint32_t base = 0, tile_bytes = 0, capacity = 0, write = 0, read = 0, reserved = 0;
    bool self_produced = false;  // compute is both producer and consumer
    uint32_t occupancy_limit = 0;
    std::deque<Tag> queue;
    std::vector<Tag> staged;
    std::vector<Tag> slots;  // what each L1 page currently holds
};
std::map<uint32_t, ModelCB> cbs;
std::map<uint32_t, std::pair<uint32_t, Tag>> reads;  // L1 address -> (bytes, tag at start)

void configure(uint32_t cb, uint32_t tile_bytes, uint32_t capacity, bool self_produced = false) {
    static uint32_t next = 0;
    if (cbs.empty()) next = 0;
    ModelCB model;
    model.base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(arena)) + next;
    model.tile_bytes = tile_bytes;
    model.capacity = capacity;
    model.self_produced = self_produced;
    model.occupancy_limit = capacity;
    model.slots.assign(capacity, Tag{});
    next += tile_bytes * capacity;
    require(next <= arena_bytes, "Model L1 arena exhausted");
    cbs[cb] = model;
}
ModelCB& cb_at(uint32_t cb) {
    auto found = cbs.find(cb);
    require(found != cbs.end(), "Kernel used an unconfigured CB " + std::to_string(cb));
    return found->second;
}
std::vector<uint32_t> args;
// The scan reader's compile-time threshold rank (compiled per variant below).
#ifndef SCAN_THRESHOLD_RANK
#define SCAN_THRESHOLD_RANK 31
#endif
constexpr uint32_t get_compile_time_arg_val(uint32_t) { return SCAN_THRESHOLD_RANK; }
template <class T> T get_arg_val(uint32_t index) {
    require(index < args.size(), "Runtime argument index out of range");
    return static_cast<T>(args[index]);
}

// ---- shared CB API ----------------------------------------------------------
void cb_reserve_back(uint32_t cb, uint32_t n) {
    auto& c = cb_at(cb);
    require(n && n <= c.capacity, "Reservation larger than the CB");
    require(c.write + n <= c.capacity, "Multi-tile reservation straddles the CB end on CB " + std::to_string(cb));
    if (c.self_produced) require(c.queue.size() + n <= c.occupancy_limit, "Self-produced CB over capacity (deadlock)");
    c.reserved = n;
    if (cb == tt::CB::c_in0) {  // poison the chunk so unwritten rows are detected
        id_pages.clear();  // the reader loads this chunk's ID pages after reserving it
        std::memset(reinterpret_cast<void*>(uintptr_t(c.base + c.write * c.tile_bytes)), 0xAB, n * c.tile_bytes);
        for (uint32_t t = 0; t < n; ++t) {
            const uint32_t tile = c.base + (c.write + t) * c.tile_bytes;
            for (auto it = query_segments.lower_bound({tile, 0, 0}); it != query_segments.end() && std::get<0>(it->first) == tile;)
                it = query_segments.erase(it);
        }
    }
}
uint32_t get_write_ptr(uint32_t cb) { auto& c = cb_at(cb); return c.base + c.write * c.tile_bytes; }
uint32_t get_read_ptr(uint32_t cb) { auto& c = cb_at(cb); return c.base + c.read * c.tile_bytes; }
uint32_t query_tiles = 4;
// Tag of the t-th tile of a c_in0 push: check every row of the assembled tile
// against the ID page of its chunk position and derive the gathered tile.
Tag assembled_query_tag(const ModelCB& c, uint32_t t) {
    const uint32_t tile = c.base + ((c.write + t) % c.capacity) * c.tile_bytes;
    const uint32_t ids_l1 = cb_at(tt::CB::c_in3).base + 128 * (t / query_tiles);
    const auto page = id_pages.find(ids_l1);
    if (page == id_pages.end()) {  // unused chunk slot (task with fewer tiles): never read by compute
        require(query_segments.lower_bound({tile, 0, 0}) == query_segments.end() ||
                std::get<0>(query_segments.lower_bound({tile, 0, 0})->first) != tile,
                "Query rows written into an unused chunk slot");
        return {};
    }
    const uint32_t k = page->second, column = t % query_tiles;
    for (uint32_t row = 0; row < 32; ++row) {
        const uint32_t q = (*tile_query_table)[size_t(k) * 32 + row];
        for (uint32_t half = 0; half < 2; ++half) {
            const auto found = query_segments.find({tile, row, half});
            if (q == UINT32_MAX) {
                require(found == query_segments.end(), "Unused query row was read from DRAM");
                const auto* bytes = reinterpret_cast<const uint8_t*>(uintptr_t(tile + 2 * fine_tile_offset(row, 16 * half)));
                for (uint32_t b = 0; b < 32; ++b) require(bytes[b] == 0, "Unused query row is not zero");
            } else {
                require(found != query_segments.end(), "Query row segment missing from an assembled tile");
                require(found->second == std::make_pair(q, column), "Assembled row holds the wrong query or column tile");
            }
        }
    }
    return {Kind::Query, k, -1, column};
}
// Tag of the t-th tile of a c_in4 push: every row must be the threshold code
// of the tile's query (or the unused code), repeated across the row.
Tag assembled_threshold_tag(const ModelCB& c, uint32_t t) {
    const uint32_t tile = c.base + ((c.write + t) % c.capacity) * c.tile_bytes;
    const auto page = id_pages.find(cb_at(tt::CB::c_in3).base + 128 * t);
    if (page == id_pages.end()) return {};  // unused chunk slot
    const uint32_t k = page->second;
    const auto* values = reinterpret_cast<const uint16_t*>(uintptr_t(tile));
    for (uint32_t row = 0; row < 32; ++row) {
        const uint32_t q = (*tile_query_table)[size_t(k) * 32 + row];
        const uint16_t expected = q == UINT32_MAX ? unused_threshold_code : threshold_code(q);
        for (uint32_t col = 0; col < 32; ++col)
            require(values[fine_tile_offset(row, col)] == expected, "Threshold row does not hold its query's threshold");
    }
    return {Kind::Threshold, k, -1, 0};
}
Tag tag_at(uint32_t address) {
    auto it = reads.upper_bound(address);
    if (it == reads.begin()) return {};
    --it;
    if (address >= it->first + it->second.first) return {};
    Tag tag = it->second.second;
    if (tag.kind == Kind::Query || tag.kind == Kind::Data) tag.sub += (address - it->first) / 2048;
    return tag;
}
bool reader_filled(uint32_t cb) {
    return cb == tt::CB::c_in1 || cb == tt::CB::c_in2 || cb == tt::CB::c_intermed5;
}
void cb_push_back(uint32_t cb, uint32_t n) {
    auto& c = cb_at(cb);
    require(c.reserved >= n, "Push without a matching reservation");
    for (uint32_t t = 0; t < n; ++t) {
        const uint32_t slot = (c.write + t) % c.capacity;
        Tag tag;
        if (!c.staged.empty()) {
            require(c.staged.size() == n, "Packed tile count differs from the push");
            tag = c.staged[t];
        } else if (cb == tt::CB::c_in0) {
            tag = assembled_query_tag(c, t);
        } else if (cb == tt::CB::c_in4) {
            tag = assembled_threshold_tag(c, t);
        } else if (reader_filled(cb)) {
            tag = tag_at(c.base + slot * c.tile_bytes);
        } else {
            tag = c.slots[slot];  // nothing packed: the page keeps its data
        }
        c.slots[slot] = tag;
        c.queue.push_back(tag);
    }
    c.staged.clear();
    c.write = (c.write + n) % c.capacity;
    c.reserved = 0;
}
void cb_wait_front(uint32_t cb, uint32_t n) {
    auto& c = cb_at(cb);
    require(n <= c.capacity, "Wait for more tiles than the CB holds (deadlock)");
    require(c.queue.size() >= n, "Consumer waits for tiles that were never produced on CB " + std::to_string(cb));
}
void cb_pop_front(uint32_t cb, uint32_t n) {
    auto& c = cb_at(cb);
    require(c.queue.size() >= n, "Pop of unpublished tiles");
    for (uint32_t t = 0; t < n; ++t) c.queue.pop_front();
    c.read = (c.read + n) % c.capacity;
}

// ---- dataflow API -----------------------------------------------------------
// NoC address of (buffer, page, byte offset), decoded by noc_async_read.
template <bool Dram> struct InterleavedAddrGen {
    uint32_t bank_base_address, page_size;
    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) const {
        require(offset < page_size && offset < 4096 && id < (1u << 28), "NoC offset outside the page");
        return (uint64_t(bank_base_address) << 40) | (uint64_t(id) << 12) | offset;
    }
};
template <bool Dram> struct InterleavedAddrGenFast { uint32_t bank_base_address, page_size; DataFormat data_format; };
template <class Gen> void read_page(uint32_t id, const Gen& gen, uint32_t dst) {
    auto& buffer = dram.at(gen.bank_base_address);
    require(gen.page_size == buffer.page, "Address generator page size differs from the buffer");
    Tag tag;
    if (buffer.words) {
        tag.kind = Kind::Script;
        std::memset(reinterpret_cast<void*>(uintptr_t(dst)), 0, gen.page_size);
        const auto& words = *buffer.words;
        const size_t per_page = gen.page_size / 4;
        require((id + 1) * per_page <= words.size(), "Script page outside the script buffer");
        std::memcpy(reinterpret_cast<void*>(uintptr_t(dst)), words.data() + id * per_page, gen.page_size);
    } else if (buffer.tagged) {
        auto found = dram_tags.find({gen.bank_base_address, id});
        require(found != dram_tags.end(), "Merge read an unwritten partial slot");
        tag = found->second;
    } else if (gen.bank_base_address == query_base) {
        tag = {Kind::Query, id / 4, -1, id % 4};
    } else if (gen.bank_base_address == dataset_base) {
        tag = {Kind::Data, -1, id, 0};
    } else {
        tag = {Kind::Index, -1, id, 0};
    }
    for (auto it = reads.lower_bound(dst); it != reads.end() && it->first < dst + gen.page_size;) it = reads.erase(it);
    reads[dst] = {gen.page_size, tag};
}
template <class Gen> void noc_async_read_page(uint32_t id, const Gen& gen, uint32_t dst) { read_page(id, gen, dst); }
template <class Gen> void noc_async_read_tile(uint32_t id, const Gen& gen, uint32_t dst) { read_page(id, gen, dst); }
void noc_async_read(uint64_t source, uint32_t destination, uint32_t bytes) {
    const uint32_t base = uint32_t(source >> 40), id = uint32_t(source >> 12) & ((1u << 28) - 1), offset = source & 4095;
    require(dram.count(base), "NoC read from an unknown buffer");
    if (base == tile_queries_base) {
        require(offset == 0 && bytes == 128 && destination % 16 == 0, "ID page read is not one aligned 128-byte page");
        require((size_t(id) + 1) * 32 <= tile_query_table->size(), "ID page outside the tile-query table");
        std::memcpy(reinterpret_cast<void*>(uintptr_t(destination)), tile_query_table->data() + size_t(id) * 32, 128);
        id_pages[destination] = id;
        return;
    }
    if (base == partial_val_base) {  // second-pass threshold: columns 16..31 of a first-pass merged row
        require(threshold_model && id >= threshold_base, "Threshold read outside the first-pass merged rows");
        require(bytes == 32 && offset % 32 == 0, "Threshold segment is not an aligned 32-byte read");
        const uint32_t src = offset / 2, src_face = src / 256;
        require(src % 16 == 0 && src_face % 2 == SCAN_THRESHOLD_RANK / 16,
                "Threshold segment is not the 16 columns holding the threshold rank");
        const uint32_t q = (id - threshold_base) * 32 + (src_face / 2) * 16 + (src % 256) / 16;
        auto& c = cb_at(tt::CB::c_in4);
        require(c.reserved && destination >= get_write_ptr(tt::CB::c_in4) &&
                destination + 32 <= get_write_ptr(tt::CB::c_in4) + c.reserved * c.tile_bytes,
                "Threshold segment outside the reserved threshold tiles");
        auto* values = reinterpret_cast<uint16_t*>(uintptr_t(destination));
        for (uint32_t e = 0; e < 16; ++e)  // only the threshold rank's column holds the code
            values[e] = e == SCAN_THRESHOLD_RANK % 16 ? threshold_code(q) : 0x0101;
        return;
    }
    require(base == query_base, "Unexpected partial-page NoC read in the scan reader");
    require(bytes == 32 && offset % 32 == 0, "Query segment is not an aligned 32-byte read");
    // Source: row (q % 32) and half of column tile t of query tile q / 32.
    const uint32_t src = offset / 2, src_face = src / 256;
    require((src % 16) == 0, "Query segment does not start a face row");
    const uint32_t src_row = (src_face / 2) * 16 + (src % 256) / 16, half = src_face % 2;
    const uint32_t q = (id / query_tiles) * 32 + src_row, column = id % query_tiles;
    auto& c = cb_at(tt::CB::c_in0);
    require(c.reserved && destination >= get_write_ptr(tt::CB::c_in0) &&
            destination + 32 <= get_write_ptr(tt::CB::c_in0) + c.reserved * c.tile_bytes,
            "Query segment outside the reserved chunk");
    const uint32_t tile = destination - (destination - c.base) % c.tile_bytes;
    const uint32_t dst = (destination - tile) / 2, dst_face = dst / 256;
    require(dst % 16 == 0 && (destination - tile) % 32 == 0, "Query segment destination is not an aligned face row");
    require(dst_face % 2 == half, "Query segment moved between column halves");
    const uint32_t row = (dst_face / 2) * 16 + (dst % 256) / 16;
    require(query_segments.emplace(std::make_tuple(tile, row, half), std::make_pair(q, column)).second,
            "Query row segment written twice");
}
void noc_async_read_barrier() {}
template <class Gen> void noc_async_write_tile(uint32_t id, const Gen& gen, uint32_t src) {
    for (auto cb : {uint32_t(tt::CB::c_out0), uint32_t(tt::CB::c_out1)}) {
        auto& c = cb_at(cb);
        if (src != get_read_ptr(cb)) continue;
        require(!c.queue.empty(), "Writer sent an unpublished tile");
        dram_tags[{gen.bank_base_address, id}] = c.queue.front();
        ++dram_writes[{gen.bank_base_address, id}];
        return;
    }
    throw std::runtime_error("Writer source is not the front of an output CB");
}
void noc_async_write_barrier() {}

// ---- compute API (formats as in the coarse model) ----------------------------
enum class Format { BF16, Int32 };
bool merge_mode = false;
bool merge_mode_flag() { return merge_mode; }
Format format(uint32_t cb) {
    return cb == tt::CB::c_in2 || cb == tt::CB::c_out1 || cb == tt::CB::c_intermed4 || cb == tt::CB::c_intermed5 ||
                   cb == tt::CB::c_intermed1 ||
                   cb == tt::CB::c_intermed7 || (merge_mode_flag() && cb == tt::CB::c_in1)
               ? Format::Int32 : Format::BF16;
}
Format srca = Format::Int32, pack_format = Format::Int32;
std::array<Tag, 4> dst;
std::array<bool, 4> loaded{};
std::array<Format, 4> dst_format{};
bool acquired = false;
uint64_t sorts = 0;
void acquire_dst() { require(!acquired, "DST acquired twice"); acquired = true; loaded.fill(false); dst = {}; }
void release_dst() { require(acquired, "DST released twice"); acquired = false; }
void reconfig_data_format_srca(uint32_t cb) { srca = format(cb); }
void reconfig_data_format_srca(uint32_t old_cb, uint32_t cb) { if (format(old_cb) != format(cb)) srca = format(cb); }
void pack_reconfig_data_format(uint32_t cb) { pack_format = format(cb); }
void copy_tile_to_dst_init_short(uint32_t cb) { require(srca == format(cb), "Copy init with the wrong SrcA format"); }
void copy_tile_to_dst_init_short_with_dt(uint32_t old_cb, uint32_t cb) {
    reconfig_data_format_srca(old_cb, cb);
    copy_tile_to_dst_init_short(cb);
}
const Tag& front(uint32_t cb, uint32_t index) {
    auto& c = cb_at(cb);
    require(index < c.queue.size(), "Tile index beyond the waited CB front");
    // Tile addresses are read pointer + index and do not wrap.
    require(c.read + index < c.capacity, "Indexed tile read past the CB end (window wraps) on CB " + std::to_string(cb));
    return c.queue[index];
}
void copy_tile(uint32_t cb, uint32_t index, uint32_t slot) {
    require(acquired && srca == format(cb), "copy_tile without DST or with the wrong SrcA format");
    dst[slot] = front(cb, index); loaded[slot] = true; dst_format[slot] = format(cb);
}
void transpose_wh_init_short(uint32_t cb) { require(srca == format(cb), "Transpose init with the wrong SrcA format"); }
void transpose_wh_init(uint32_t cb, uint32_t) { srca = format(cb); }
void transpose_wh_tile(uint32_t cb, uint32_t index, uint32_t slot) {
    require(acquired && srca == format(cb), "transpose without DST or with the wrong SrcA format");
    dst[slot] = front(cb, index); dst[slot].transposed = !dst[slot].transposed;
    loaded[slot] = true; dst_format[slot] = format(cb);
}
void mm_init(uint32_t a, uint32_t b, uint32_t out) {
    require(!acquired && format(a) == Format::BF16 && format(b) == Format::BF16, "Bad mm_init");
    srca = format(b); pack_format = format(out);
}
void mm_init_short(uint32_t a, uint32_t b) {
    require(!acquired && format(a) == Format::BF16 && srca == format(b), "Bad mm_init_short state");
}
void matmul_tiles(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t slot) {
    require(acquired && srca == Format::BF16, "matmul without DST or with the wrong SrcA format");
    const Tag q = front(in0, i0), d = front(in1, i1);
    require(q.kind == Kind::Query, "matmul in0 tile is not a query tile (unused chunk slot read?)");
    require(d.kind == Kind::Data && d.sub == int64_t(i1), "matmul in1 tile is not the expected page tile");
    require(q.sub == int64_t(i1) && i0 % query_tiles == i1, "matmul pairs mismatched inner-dimension tiles");
    Tag& out = dst[slot];
    if (!loaded[slot]) { out = {Kind::Score, q.batch, d.page, 0}; loaded[slot] = true; dst_format[slot] = Format::BF16; }
    require(out.kind == Kind::Score && out.batch == q.batch && out.page == d.page, "matmul accumulated mixed tiles");
    out.sub |= int64_t(1) << i1;
}
void pack_tile(uint32_t slot, uint32_t cb) {
    require(acquired && loaded[slot] && pack_format == format(cb) && dst_format[slot] == format(cb), "Bad pack format/state");
    if (dst[slot].kind == Kind::Score) require(dst[slot].sub == (int64_t(1) << query_tiles) - 1, "Score packed before all K tiles");
    auto& c = cb_at(cb);
    require(c.reserved > c.staged.size(), "pack_tile without a reservation");
    c.staged.push_back(dst[slot]);
}
uintptr_t get_tile_address(uint32_t cb, uint32_t) {
    auto& c = cb_at(cb);
    // PACK stores through this address fill an empty CB's next page.
    if (cb == tt::CB::c_intermed3 || cb == tt::CB::c_intermed4 || cb == tt::CB::c_intermed6 || cb == tt::CB::c_intermed7) {
        require(c.queue.empty(), "initialize_champion on a non-empty CB writes the wrong page");
        c.slots[c.read] = {Kind::Sentinel};
        c.slots[c.read].ids = cb == tt::CB::c_intermed4 || cb == tt::CB::c_intermed7;
    } else if (cb == tt::CB::c_in5) {
        require(c.queue.empty(), "Constant tile initialized on a non-empty CB");
        c.slots[c.read] = {Kind::Scaler};
    }
    return c.base + c.read * c.tile_bytes;
}
// Merge skipping: score - threshold, then a scalar max into a flag tile.
enum class PoolType { MAX };
enum class ReduceDim { REDUCE_SCALAR };
void sub_tiles_init(uint32_t a, uint32_t b) {
    require(srca == format(a) && format(b) == Format::BF16, "sub init with the wrong formats");
}
void sub_tiles(uint32_t a, uint32_t b, uint32_t ia, uint32_t ib, uint32_t slot) {
    require(acquired && srca == Format::BF16, "sub without DST or with the wrong SrcA format");
    const Tag s = front(a, ia), th = front(b, ib);
    require(s.kind == Kind::Score && s.sub == (int64_t(1) << query_tiles) - 1, "sub input is not a complete score tile");
    require(th.kind == Kind::Threshold && th.batch == s.batch, "Score tile checked against another tile's thresholds");
    dst[slot] = {Kind::Diff, s.batch, s.page, 0}; loaded[slot] = true; dst_format[slot] = Format::BF16;
}
bool reduce_active = false;
template <PoolType, ReduceDim> void reduce_init(uint32_t icb, uint32_t scaler, uint32_t) {
    require(!reduce_active, "reduce_init twice without reduce_uninit");
    require(format(icb) == Format::BF16 && format(scaler) == Format::BF16, "reduce init with the wrong formats");
    srca = format(icb); reduce_active = true;
}
void reduce_uninit() { require(reduce_active, "reduce_uninit without reduce_init"); reduce_active = false; }
template <PoolType, ReduceDim> void reduce_tile(uint32_t icb, uint32_t scaler, uint32_t i, uint32_t is, uint32_t slot) {
    require(acquired && reduce_active, "reduce without DST or reduce_init");
    const Tag d = front(icb, i), sc = front(scaler, is);
    require(d.kind == Kind::Diff && sc.kind == Kind::Scaler, "reduce input is not a diff tile with the scaler");
    dst[slot] = {Kind::Flag, d.batch, d.page, 0}; loaded[slot] = true; dst_format[slot] = Format::BF16;
}
uint32_t read_tile_value(uint32_t cb, uint32_t index, uint32_t offset) {
    const Tag f = front(cb, index);
    require(f.kind == Kind::Flag && offset == 0, "Flag read from a tile that is not a flag");
    const bool merge = decide(f.batch, f.page);
    require(decisions.emplace(std::make_pair(f.batch, f.page), merge).second, "A (tile, page) was checked twice");
    return merge ? 0x3F80u : 0xBF80u;  // +1.0 / -1.0 in the low half-word
}
void topk_tile_init() {}
void topk_local_sort(uint32_t, uint32_t, uint32_t) {
    require(acquired && srca == Format::BF16, "topk without DST or with the wrong SrcA format");
    for (auto present : loaded) require(present, "topk with an unloaded DST tile");
    require(dst_format[0] == Format::BF16 && dst_format[1] == Format::BF16 &&
            dst_format[2] == Format::Int32 && dst_format[3] == Format::Int32, "topk DST formats are wrong");
    Tag& value = dst[0]; Tag& index = dst[2];
    const Tag& in_value = dst[1]; const Tag& in_index = dst[3];
    require(value.kind == Kind::Sentinel || value.kind == Kind::Champion, "topk champion value tile is not a champion");
    require(index.kind == Kind::Sentinel || index.kind == Kind::Champion, "topk champion ID tile is not a champion");
    require(in_value.transposed && in_index.transposed, "topk candidates were not transposed");
    Tag next = value.kind == Kind::Sentinel ? Tag{Kind::Champion} : value;
    if (index.kind != Kind::Sentinel || value.kind != Kind::Sentinel) require(same_champion(value, index), "Champion value/ID tiles diverged");
    if (in_value.kind == Kind::Score) {  // scan: candidate page for one batch
        require(in_index.kind == Kind::Index && in_index.page == in_value.page, "Score and ID tiles are from different pages");
        if (next.batch < 0) { next.batch = in_value.batch; next.first = in_value.page; }
        require(next.batch == in_value.batch, "Champion ring front belongs to another batch");
        if (threshold_model) {
            const auto decided = decisions.find({in_value.batch, in_value.page});
            require(decided != decisions.end() && decided->second, "A page was merged without (or against) its flag");
            require(next.last < in_value.page, "List pages merged out of order or twice");
        } else {
            require(next.last < 0 || next.last + 1 == in_value.page, "List pages merged out of order or twice");
        }
        next.last = in_value.page; next.sum += in_value.page;
    } else {  // merge: partial of one (list, batch)
        require(in_value.kind == Kind::Partial && same_champion(in_value, in_index), "Merge input is not a partial pair");
        if (next.batch < 0) next.batch = in_value.batch;
        require(next.batch == in_value.batch, "Merge mixed batches");
        next.sum += in_value.first;  // identify partials by their first list page
    }
    ++next.count;
    next.transposed = in_value.transposed;  // champions live in the transposed candidate layout
    value = next; value.ids = false;
    index = next; index.ids = true;
    ++sorts;
}

#include "../kernels/compute/ivf_compute_l1.hpp"
namespace scan_reader {
#define kernel_main run
#include "../kernels/dataflow/reader_scan.cpp"
#undef kernel_main
}
namespace scan_compute {
#define kernel_main run
#include "../kernels/compute/compute_scan.cpp"
#undef kernel_main
}
namespace scan_writer {
#define kernel_main run
#include "../kernels/dataflow/writer_scan.cpp"
#undef kernel_main
}
using namespace ivf_cluster_major;

void reset_core() {
    cbs.clear(); reads.clear();
    srca = pack_format = Format::Int32; acquired = false;
}
void require_drained(const char* phase) {
    for (auto& [cb, c] : cbs)
        if (cb != tt::CB::c_intermed5 && cb != tt::CB::c_intermed6 && cb != tt::CB::c_intermed7 && cb != tt::CB::c_in5)
            require(c.queue.empty() && c.staged.empty(), std::string(phase) + ": CB " + std::to_string(cb) + " not drained");
}

void check(uint32_t queries, uint32_t nlist, uint32_t nprobe, uint32_t chunk, uint32_t cores,
           uint32_t prefetch, uint32_t seed, bool threshold) {
    // Synthetic lists: empty, single-vector, exact-page and tail lists.
    std::mt19937 rng(seed);
    std::vector<ListLayout> lists(nlist);
    uint32_t start = 0;
    for (uint32_t c = 0; c < nlist; ++c) {
        const uint32_t sizes[] = {0, 1, 31, 32, 33, 64, 70, 100};
        const uint32_t count = sizes[rng() % 8];
        const uint32_t blocks = std::max(1u, (count + 31) / 32);
        lists[c] = {start, count, blocks * 32};
        start += blocks;
    }
    std::vector<std::vector<uint32_t>> selections(queries);
    for (auto& row : selections) {
        std::set<uint32_t> chosen;
        while (chosen.size() < nprobe) chosen.insert(rng() % nlist);
        row.assign(chosen.begin(), chosen.end());
        std::shuffle(row.begin(), row.end(), rng);
    }
    // Gathered tiles play the role of batches: pair (tile, slot = tile).
    const auto plan = ivf_cluster_gather::plan_gather(selections, nprobe, lists, chunk, cores, cores);

    std::vector<std::vector<uint32_t>> scripts(cores);
    size_t max_words = 0;
    for (uint32_t w = 0; w < cores; ++w) {
        scripts[w] = encode_scan_script(plan.worker_tasks[w], chunk);
        max_words = std::max(max_words, scripts[w].size());
    }
    const uint32_t script_bytes = static_cast<uint32_t>(script_page_bytes(max_words));
    std::vector<uint32_t> host_scripts(size_t(cores) * script_bytes / 4, 0);
    for (uint32_t w = 0; w < cores; ++w)
        std::copy(scripts[w].begin(), scripts[w].end(), host_scripts.begin() + size_t(w) * script_bytes / 4);

    query_tiles = 4;  // dim 100 -> 128
    const uint32_t padded_dim = 128;
    dram.clear(); dram_tags.clear(); dram_writes.clear();
    dram[query_base] = {2048, false};
    dram[dataset_base] = {padded_dim * 64, false};
    dram[index_base] = {4096, false};
    dram[script_base] = {script_bytes, false, &host_scripts};
    dram[partial_val_base] = {2048, true};
    dram[partial_ind_base] = {4096, true};
    // Tile-query table as uploaded by ivf_tt.cpp.
    std::vector<uint32_t> tile_queries(std::max<size_t>(1, plan.tiles.size()) * 32, UINT32_MAX);
    for (size_t k = 0; k < plan.tiles.size(); ++k)
        std::copy_n(plan.requesters.begin() + plan.tiles[k].first, plan.tiles[k].rows, tile_queries.begin() + k * 32);
    dram[tile_queries_base] = {128, false};
    tile_query_table = &tile_queries;
    threshold_model = threshold;
    threshold_base = uint32_t(plan.tiles.size());  // first-pass merged rows follow the tiles
    decisions.clear();

    // Scan: same CB capacities as ivf_tt.cpp / scan_l1::worker_cb_bytes.
    merge_mode = false;
    for (uint32_t w = 0; w < cores; ++w) {
        reset_core();
        configure(tt::CB::c_in0, 2048, uint32_t(scan_l1::query_cb_tiles(100, chunk)));
        configure(tt::CB::c_in1, 2048, prefetch * query_tiles);
        configure(tt::CB::c_in2, 4096, prefetch);
        configure(tt::CB::c_intermed0, 2048, uint32_t(scan_l1::score_cb_tiles(chunk)), true);
        configure(tt::CB::c_intermed1, 4096, 1, true);
        configure(tt::CB::c_intermed3, 2048, chunk, true);
        configure(tt::CB::c_intermed4, 4096, chunk, true);
        configure(tt::CB::c_intermed6, 2048, 1, true);
        configure(tt::CB::c_intermed7, 4096, 1, true);
        configure(tt::CB::c_out0, 2048, 2);
        configure(tt::CB::c_out1, 4096, 2);
        configure(tt::CB::c_intermed5, script_bytes, 1);
        configure(tt::CB::c_in3, uint32_t(scan_l1::tile_query_page), chunk);
        query_segments.clear(); id_pages.clear();
        configure(tt::CB::c_in4, 2048, 2 * chunk);
        configure(tt::CB::c_in5, 2048, 1, true);
        configure(tt::CB::c_intermed2, 2048, chunk, true);
        configure(tt::CB::c_out2, 2048, chunk, true);
        args = {query_base, dataset_base, index_base, padded_dim, script_base, script_bytes, w, chunk, tile_queries_base,
                threshold ? 1u : 0u, partial_val_base, threshold_base};
        scan_reader::run();
        args = {query_tiles, chunk, threshold ? 1u : 0u};
        scan_compute::run();
        require(!reduce_active, "Scan compute left reduce configured");
        require(!acquired, "Scan compute left DST acquired");
        args = {partial_val_base, partial_ind_base};
        scan_writer::run();
        require_drained("scan");
    }
    // Every gathered tile lands once in its own slot, in row layout, after
    // merging every page of its own list against that tile's queries.
    require(dram_writes.size() == 2 * plan.tiles.size(), "Partial slot count differs from the tile count");
    for (auto& [key, writes] : dram_writes) require(writes == 1, "A partial slot was written more than once");
    uint32_t checked = 0;
    for (uint32_t w = 0; w < cores; ++w)
        for (const auto& task : plan.worker_tasks[w])
            for (const auto& pair : task.pairs) {
                const auto& value = dram_tags.at({partial_val_base, pair.slot});
                const auto& index = dram_tags.at({partial_ind_base, pair.slot});
                // Expected merges: every page, or the pages whose flag said merge.
                int64_t count = 0, first = -1, last = -1, sum = 0;
                for (uint32_t p = 0; p < task.pages; ++p) {
                    const int64_t page = task.start_block + p;
                    if (threshold) {
                        const auto decided = decisions.find({pair.batch, page});
                        require(decided != decisions.end(), "A (tile, page) pair was never checked");
                        if (!decided->second) continue;
                    }
                    if (first < 0) first = page;
                    last = page; sum += page; ++count;
                }
                if (count == 0) {
                    require(value.kind == Kind::Sentinel && index.kind == Kind::Sentinel && index.ids,
                            "A fully skipped tile did not publish the sentinel pair");
                } else {
                    require(value.kind == Kind::Champion && !value.transposed && same_champion(value, index),
                            "Partial slot does not hold a row-layout champion pair");
                    require(pair.slot == pair.batch && value.batch == pair.batch && value.first == first &&
                            value.count == count && value.last == last && value.sum == sum,
                            "Partial slot holds the wrong tile or the wrong merged pages");
                }
                ++checked;
            }
    require(checked == plan.tiles.size(), "Scan pairs differ from the tile count");
    if (threshold) require(decisions.size() == plan.pair_pages, "Not every (tile, page) pair was checked exactly once");
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
        uint32_t seed = 1;
        for (uint32_t queries : {1u, 33u, 333u, 1100u})
            for (uint32_t nprobe : {1u, 3u})
                for (uint32_t chunk : {1u, 3u, 32u})
                    for (uint32_t cores : {1u, 5u, 64u})
                        for (bool threshold : {false, true})
                            check(queries, 40, nprobe, chunk, cores, chunk == 3 ? 1 : 2, seed++, threshold);
        std::cout << "Gathered-tile scan protocol (single pass and threshold mode), full-ring rotation, merge skipping, "
                     "row-layout partials and slot routing passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
