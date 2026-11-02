#pragma once
// Cluster-major fine-search plan. Each inverted list is read once per chunk of
// its active query batches instead of once per active batch:
//
//   scan:  task = (list, <= batch_chunk active batches). The reader loads the
//          chunk's query tiles, then streams every list page once; compute
//          multiplies each page with every query batch of the chunk and keeps
//          one top-32 champion per batch. The writer stores one partial top-32
//          tile pair per (list, batch) in a DRAM slot.
//   merge: every batch's partial slots are contiguous, so a merge core reduces
//          [slot_begin, slot_begin + slots) into that batch's final top-32.
//
// Work (page x batch matmul/top-k pairs) equals the packed batch-union page
// count, exactly as in the batch-major scheduler; DRAM candidate reads drop to
// sum(list pages * chunks of that list).
#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ivf_cluster_major {

struct ListLayout {
    uint32_t start_block = 0, count = 0, padded_count = 0;
};
struct ScanPair {
    uint32_t batch = 0, slot = 0;
};
struct ScanTask {
    uint32_t cid = 0, start_block = 0, count = 0, pages = 0;
    std::vector<ScanPair> pairs;
    uint64_t weight() const {
        // Matmul/top-k pairs dominate; the extra page term charges the read
        // and the per-task query load so that tiny chunks are not free.
        return uint64_t(pages) * (pairs.size() + 1) + pairs.size();
    }
};
struct MergeBatch {
    uint32_t batch = 0, slot_begin = 0, slots = 0;
};
struct ClusterMajorPlan {
    std::vector<std::vector<ScanTask>> worker_tasks;
    std::vector<std::vector<MergeBatch>> merge_cores;
    std::vector<uint32_t> batch_slot_begin, batch_slots;
    uint32_t partial_slots = 0;
    uint64_t tasks = 0, pair_pages = 0, scan_pages = 0;
    uint64_t min_worker_weight = 0, max_worker_weight = 0, total_worker_weight = 0;
};

// Script words per task: start_block, actual count, pages, pair count, then
// (packed batch, partial slot) per pair. Word 0 is the task count.
constexpr uint32_t scan_task_header_words = 4;

inline bool batch_active(const std::vector<uint32_t>& masks, uint32_t words, uint32_t cid, uint32_t batch) {
    return (masks[size_t(cid) * words + batch / 32] >> (batch % 32)) & 1u;
}

inline ClusterMajorPlan plan_cluster_major(
    const std::vector<uint32_t>& masks, uint32_t nlist, uint32_t num_batches,
    const std::vector<ListLayout>& lists, uint32_t batch_chunk,
    uint32_t scan_workers, uint32_t merge_cores) {
    if (!nlist || !num_batches || !batch_chunk || !scan_workers || !merge_cores)
        throw std::invalid_argument("Cluster-major plan needs lists, batches, a chunk size and cores");
    const uint32_t words = (num_batches + 31) / 32;
    if (masks.size() != size_t(nlist) * words || lists.size() != nlist)
        throw std::invalid_argument("Cluster-major plan received masks/lists of the wrong shape");
    for (uint32_t cid = 0; cid < nlist; ++cid) {
        if (num_batches % 32 && masks[size_t(cid) * words + words - 1] >> (num_batches % 32))
            throw std::invalid_argument("Batch mask activates a batch past the padded query count");
        const auto& list = lists[cid];
        if (list.padded_count % 32 || list.count > list.padded_count)
            throw std::invalid_argument("Inverted list layout is not padded to 32-vector pages");
    }

    ClusterMajorPlan plan;
    plan.batch_slots.assign(num_batches, 0);
    for (uint32_t cid = 0; cid < nlist; ++cid) {
        if (!lists[cid].count) continue;  // Empty lists are never read.
        for (uint32_t b = 0; b < num_batches; ++b) plan.batch_slots[b] += batch_active(masks, words, cid, b);
    }
    plan.batch_slot_begin.assign(num_batches, 0);
    uint64_t slots = 0;
    for (uint32_t b = 0; b < num_batches; ++b) {
        plan.batch_slot_begin[b] = uint32_t(slots);
        slots += plan.batch_slots[b];
    }
    if (slots > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Partial slot count exceeds uint32");
    plan.partial_slots = uint32_t(slots);

    // Slots of a batch are assigned in list-ID order; chunks split a list's
    // ascending active batches into near-equal pieces of at most batch_chunk.
    std::vector<uint32_t> filled(num_batches, 0);
    std::vector<ScanTask> tasks;
    for (uint32_t cid = 0; cid < nlist; ++cid) {
        const auto& list = lists[cid];
        if (!list.count) continue;
        std::vector<uint32_t> active;
        for (uint32_t b = 0; b < num_batches; ++b) if (batch_active(masks, words, cid, b)) active.push_back(b);
        if (active.empty()) continue;
        const uint32_t pages = list.padded_count / 32;
        const size_t chunks = (active.size() + batch_chunk - 1) / batch_chunk;
        for (size_t c = 0, begin = 0; c < chunks; ++c) {
            const size_t end = begin + (active.size() - begin + (chunks - c) - 1) / (chunks - c);
            ScanTask task{cid, list.start_block, list.count, pages, {}};
            for (size_t i = begin; i < end; ++i) {
                const uint32_t b = active[i];
                task.pairs.push_back({b, plan.batch_slot_begin[b] + filled[b]++});
            }
            plan.pair_pages += uint64_t(pages) * task.pairs.size();
            plan.scan_pages += pages;
            tasks.push_back(std::move(task));
            begin = end;
        }
    }
    for (uint32_t b = 0; b < num_batches; ++b)
        if (filled[b] != plan.batch_slots[b]) throw std::logic_error("Partial slot assignment is incomplete");
    plan.tasks = tasks.size();

    // Longest-processing-time-first placement on the scan cores.
    std::stable_sort(tasks.begin(), tasks.end(), [](const ScanTask& a, const ScanTask& b) {
        return std::tuple(b.weight(), b.pages, a.cid) < std::tuple(a.weight(), a.pages, b.cid);
    });
    using Load = std::pair<uint64_t, uint32_t>;
    std::priority_queue<Load, std::vector<Load>, std::greater<Load>> workers;
    for (uint32_t w = 0; w < scan_workers; ++w) workers.push({0, w});
    std::vector<uint64_t> worker_weight(scan_workers, 0);
    plan.worker_tasks.assign(scan_workers, {});
    for (auto& task : tasks) {
        const auto [load, w] = workers.top();
        workers.pop();
        worker_weight[w] = load + task.weight();
        workers.push({worker_weight[w], w});
        plan.worker_tasks[w].push_back(std::move(task));
    }
    for (auto weight : worker_weight) plan.total_worker_weight += weight;
    plan.min_worker_weight = *std::min_element(worker_weight.begin(), worker_weight.end());
    plan.max_worker_weight = *std::max_element(worker_weight.begin(), worker_weight.end());

    // Every batch is merged by exactly one core, including batches whose
    // selected lists are all empty (zero slots -> sentinel output).
    std::vector<uint32_t> order(num_batches);
    for (uint32_t b = 0; b < num_batches; ++b) order[b] = b;
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return plan.batch_slots[a] > plan.batch_slots[b]; });
    std::priority_queue<Load, std::vector<Load>, std::greater<Load>> mergers;
    for (uint32_t m = 0; m < merge_cores; ++m) mergers.push({0, m});
    plan.merge_cores.assign(merge_cores, {});
    for (auto b : order) {
        const auto [load, m] = mergers.top();
        mergers.pop();
        mergers.push({load + plan.batch_slots[b] + 1, m});
        plan.merge_cores[m].push_back({b, plan.batch_slot_begin[b], plan.batch_slots[b]});
    }
    for (auto& assigned : plan.merge_cores)
        std::sort(assigned.begin(), assigned.end(), [](auto& a, auto& b) { return a.batch < b.batch; });
    return plan;
}

inline std::vector<uint32_t> encode_scan_script(const std::vector<ScanTask>& tasks, uint32_t batch_chunk) {
    std::vector<uint32_t> script{uint32_t(tasks.size())};
    for (const auto& task : tasks) {
        if (task.pairs.empty() || task.pairs.size() > batch_chunk || !task.pages || !task.count)
            throw std::invalid_argument("Scan task has no pairs, too many pairs or no candidates");
        script.insert(script.end(), {task.start_block, task.count, task.pages, uint32_t(task.pairs.size())});
        for (const auto& pair : task.pairs) script.insert(script.end(), {pair.batch, pair.slot});
    }
    return script;
}

// Merge cores receive (batch, slot_begin, slots) triples as runtime arguments
// after `fixed_args` addresses; keep the total within the runtime-arg limit.
inline std::vector<uint32_t> encode_merge_args(
    std::vector<uint32_t> fixed_args, const std::vector<MergeBatch>& batches, uint32_t max_args) {
    fixed_args.push_back(uint32_t(batches.size()));
    for (const auto& batch : batches) fixed_args.insert(fixed_args.end(), {batch.batch, batch.slot_begin, batch.slots});
    if (fixed_args.size() > max_args)
        throw std::runtime_error("Merge assignment exceeds the runtime argument limit");
    return fixed_args;
}

// Math fidelity of the scan matmul and the coarse matmul, as the numeric
// value of tt::tt_metal::MathFidelity (LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4; there
// is no 1). HiFi4 is the full BF16 product used by every other implementation.
inline constexpr std::pair<const char*, uint32_t> fidelities[] = {{"LoFi", 0}, {"HiFi2", 2}, {"HiFi3", 3}, {"HiFi4", 4}};
inline uint32_t parse_fidelity(const std::string& name) {
    for (const auto& [label, value] : fidelities)
        if (name == label) return value;
    throw std::invalid_argument("Math fidelity must be LoFi, HiFi2, HiFi3 or HiFi4");
}
inline const char* fidelity_name(uint32_t value) {
    for (const auto& [label, fidelity] : fidelities)
        if (value == fidelity) return label;
    throw std::invalid_argument("Unknown math fidelity value");
}

// Worker L1 used by the scan circular buffers. The query CB holds two whole
// chunks: every task reserves/pushes a full chunk so multi-tile reservations
// never wrap; champion CBs hold one value/index tile per batch in a chunk.
namespace scan_l1 {
constexpr uint64_t bf16_tile = 2048, index_tile = 4096;
constexpr uint64_t tile_query_page = 128;  // 32 uint32 query IDs of one gathered tile
constexpr uint64_t query_cb_tiles(uint32_t dim, uint32_t batch_chunk) {
    return 2ull * batch_chunk * ((uint64_t(dim) + 31) / 32);
}
constexpr uint64_t score_cb_tiles(uint32_t batch_chunk) { return std::max<uint64_t>(2, batch_chunk); }
constexpr uint64_t worker_cb_bytes(uint32_t dim, uint32_t prefetch, uint32_t batch_chunk, uint64_t script_bytes) {
    const uint64_t qtw = (uint64_t(dim) + 31) / 32;
    return query_cb_tiles(dim, batch_chunk) * bf16_tile          // c_in0 query chunks
           + prefetch * (qtw * bf16_tile + index_tile)            // c_in1/c_in2 list pages
           + score_cb_tiles(batch_chunk) * bf16_tile              // c_intermed0 one score tile per batch
           + index_tile                                           // c_intermed1 transposed IDs
           + batch_chunk * (bf16_tile + index_tile)               // c_intermed3/4 champion ring
           + (bf16_tile + index_tile)                             // c_intermed6/7 sentinel tiles
           + 2 * (bf16_tile + index_tile)                         // c_out0/1 partial outputs
           + batch_chunk * tile_query_page                        // c_in3 reader tile-query IDs
           + 2 * batch_chunk * bf16_tile                          // c_in4 threshold tiles (double-buffered)
           + bf16_tile                                            // c_in5 reduce scaler
           + 2 * batch_chunk * bf16_tile                          // c_intermed2 diffs, c_out2 flags
           + script_bytes;                                        // c_intermed5 script
}
constexpr uint64_t merge_cb_bytes(uint64_t script_bytes) { return 4 * (bf16_tile + index_tile) + script_bytes; }
}  // namespace scan_l1

inline std::vector<uint32_t> encode_merge_script(const std::vector<MergeBatch>& batches) {
    std::vector<uint32_t> script{uint32_t(batches.size())};
    for (const auto& batch : batches) script.insert(script.end(), {batch.batch, batch.slot_begin, batch.slots});
    return script;
}

inline uint64_t script_page_bytes(size_t max_words) {
    const uint64_t bytes = std::max<uint64_t>(1, max_words) * sizeof(uint32_t);
    return (bytes + 63) / 64 * 64;
}

}  // namespace ivf_cluster_major
