#pragma once
#include "api/dataflow/dataflow_api.h"
#include "../../partition_protocol.hpp"
#include "../../fine_tile_layout.hpp"
namespace ivf_transport {
using namespace ivf_partitioned;
inline volatile tt_l1_ptr uint32_t* words(uint32_t addr) {
#ifdef IVF_TRANSPORT_SIM
    return reinterpret_cast<volatile uint32_t*>(ivf_sim_l1(addr));
#else
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
#endif
}
inline uint32_t read_word(uint32_t addr) {
#ifdef IVF_TRANSPORT_SIM
    // Model the acquire side of device publication without a C++ data race.
    return __atomic_load_n(const_cast<uint32_t*>(words(addr)), __ATOMIC_ACQUIRE);
#else
    return words(addr)[0];
#endif
}
inline void wait_word(uint32_t addr, uint32_t value) {
    while (read_word(addr) < value) { invalidate_l1_cache(); }
    invalidate_l1_cache();
}
inline void publish_word(uint32_t local_scratch, uint32_t x, uint32_t y, uint32_t addr, uint32_t value) {
    words(local_scratch)[0] = value;
    noc_async_write(local_scratch, get_noc_addr(x, y, addr), 4);
    noc_async_write_barrier();
}
struct ScriptReader {
    InterleavedAddrGen<true> generator;
    uint32_t cache, cached_page = 0xffffffffu;
    ScriptReader(uint32_t base, uint32_t cache_addr) :
        generator{.bank_base_address = base, .page_size = script_page_bytes}, cache(cache_addr) {}
    Record read(uint32_t record) {
        const uint32_t page = record / records_per_page;
        if (page != cached_page) {
            noc_async_read_page(page, generator, cache);
            noc_async_read_barrier();
            cached_page = page;
        }
        Record r;
        auto* src = words(cache + (record % records_per_page) * record_bytes);
        for (uint32_t i = 0; i < record_words; ++i) r.word[i] = src[i];
        return r;
    }
};
inline void push_record(const Record& r) {
    constexpr uint32_t cb = tt::CB::c_intermed5;
    cb_reserve_back(cb, 1);
    auto* dest = words(get_write_ptr(cb));
    for (uint32_t i = 0; i < record_words; ++i) dest[i] = r.word[i];
    cb_push_back(cb, 1);
}
inline void make_mask(uint32_t valid) {
    constexpr uint32_t cb = tt::CB::c_in3;
    cb_reserve_back(cb, 1);
    auto* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(words(get_write_ptr(cb)));
    for (uint32_t row = 0; row < 32; ++row)
        for (uint32_t lane = 0; lane < 32; ++lane)
            tile[fine_tile_offset(row, lane)] = lane < valid ? 0 : invalid_score_bf16;
    cb_push_back(cb, 1);
}
// The stateful API preserves the upper NoC address set above. src_base_addr
// and src_addr in with_trid are *lower-address addends*, not high/low halves.
inline void read_with_id(uint64_t src, uint32_t dst, uint32_t bytes, uint32_t id) {
    while (bytes) {
        const uint32_t chunk = bytes > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : bytes;
        noc_async_read_one_packet_set_state(src, chunk);
        noc_async_read_set_trid(id);
        noc_async_read_one_packet_with_state_with_trid(0, static_cast<uint32_t>(src), dst, id);
        src += chunk; dst += chunk; bytes -= chunk;
    }
}
} // namespace ivf_transport
