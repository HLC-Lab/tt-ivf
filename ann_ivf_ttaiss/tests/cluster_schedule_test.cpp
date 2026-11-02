// Portable checks for the cluster-major plan built from grouped batch masks.
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include "../query_grouping.hpp"
#include "../cluster_schedule.hpp"

using namespace ivf_cluster_major;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void must_reject(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Malformed plan input was accepted");
}

void check_plan(uint32_t queries, uint32_t nlist, uint32_t nprobe, uint32_t chunk, uint32_t cores,
                const std::string& mode, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<ListLayout> lists(nlist);
    std::vector<uint32_t> pages(nlist);
    for (uint32_t c = 0, start = 0; c < nlist; ++c) {
        const uint32_t count = rng() % 5 == 0 ? 0 : rng() % 200 + 1;
        const uint32_t blocks = std::max(1u, (count + 31) / 32);
        lists[c] = {start, count, blocks * 32};
        pages[c] = count ? blocks : 0;
        start += blocks;
    }
    Selections selections(queries);
    for (auto& row : selections) {
        std::set<uint32_t> chosen;
        while (chosen.size() < nprobe) chosen.insert(rng() % nlist);
        row.assign(chosen.begin(), chosen.end());
    }
    const auto grouping = group_queries(selections, pages, mode, 0, 256);
    const uint32_t batches = (queries + 31) / 32, words = (batches + 31) / 32;
    const auto masks = make_batch_masks(selections, grouping, nlist);
    const auto plan = plan_cluster_major(masks, nlist, batches, lists, chunk, cores, cores);

    // Compute work equals the grouped batch-union pages; reads are one pass
    // per list chunk.
    require(plan.pair_pages == grouping.grouped_pages, "Page x batch pairs differ from the grouped union pages");
    uint64_t expected_reads = 0, expected_slots = 0;
    for (uint32_t c = 0; c < nlist; ++c) {
        if (!lists[c].count) continue;
        uint64_t active = 0;
        for (uint32_t b = 0; b < batches; ++b) active += (masks[c * words + b / 32] >> (b % 32)) & 1;
        expected_reads += uint64_t(pages[c]) * ((active + chunk - 1) / chunk);
        expected_slots += active;
    }
    require(plan.scan_pages == expected_reads, "List page reads differ from one pass per chunk");
    require(plan.partial_slots == expected_slots, "Partial slot count differs from active (list, batch) pairs");

    // Each active (non-empty list, batch) pair occurs once; slots are a
    // bijection onto [0, partial_slots) and contiguous per batch.
    std::set<std::pair<uint32_t, uint32_t>> pairs;
    std::vector<bool> slot_seen(plan.partial_slots, false);
    uint64_t total_weight = 0, max_task_weight = 0;
    for (const auto& tasks : plan.worker_tasks) {
        uint64_t worker_weight = 0;
        for (const auto& task : tasks) {
            require(!task.pairs.empty() && task.pairs.size() <= chunk, "Task chunk size out of range");
            require(lists[task.cid].count && task.start_block == lists[task.cid].start_block &&
                    task.count == lists[task.cid].count && task.pages == pages[task.cid], "Task list layout is wrong");
            for (size_t i = 0; i < task.pairs.size(); ++i) {
                const auto& pair = task.pairs[i];
                require(i == 0 || task.pairs[i - 1].batch < pair.batch, "Chunk batches are not ascending");
                require((masks[task.cid * words + pair.batch / 32] >> (pair.batch % 32)) & 1, "Inactive batch scheduled");
                require(pairs.insert({task.cid, pair.batch}).second, "A (list, batch) pair was scheduled twice");
                require(pair.slot < plan.partial_slots && !slot_seen[pair.slot], "Slot reused or out of range");
                slot_seen[pair.slot] = true;
                require(pair.slot >= plan.batch_slot_begin[pair.batch] &&
                        pair.slot < plan.batch_slot_begin[pair.batch] + plan.batch_slots[pair.batch],
                        "Slot outside its batch range");
            }
            worker_weight += task.weight();
            max_task_weight = std::max(max_task_weight, task.weight());
        }
        total_weight += worker_weight;
    }
    require(pairs.size() == plan.partial_slots, "Scheduled pairs differ from the slot count");
    require(total_weight == plan.total_worker_weight, "Worker weights do not add up");
    require(plan.max_worker_weight - plan.min_worker_weight <= max_task_weight, "LPT imbalance exceeds one task");

    // Every batch, including zero-slot and partial batches, is merged once.
    std::vector<uint32_t> merged(batches, 0);
    for (const auto& assigned : plan.merge_cores)
        for (const auto& batch : assigned) {
            require(batch.slot_begin == plan.batch_slot_begin[batch.batch] &&
                    batch.slots == plan.batch_slots[batch.batch], "Merge range differs from the batch slots");
            ++merged[batch.batch];
        }
    for (auto count : merged) require(count == 1, "A batch is merged zero or several times");

    // Script encoding is the exact layout the kernels parse.
    for (const auto& tasks : plan.worker_tasks) {
        const auto script = encode_scan_script(tasks, chunk);
        require(script[0] == tasks.size(), "Script task count is wrong");
        size_t ptr = 1;
        for (const auto& task : tasks) {
            require(script[ptr] == task.start_block && script[ptr + 1] == task.count &&
                    script[ptr + 2] == task.pages && script[ptr + 3] == task.pairs.size(), "Script task header is wrong");
            ptr += scan_task_header_words;
            for (const auto& pair : task.pairs) {
                require(script[ptr] == pair.batch && script[ptr + 1] == pair.slot, "Script pair is wrong");
                ptr += 2;
            }
        }
        require(ptr == script.size(), "Script has trailing words");
        require(script_page_bytes(script.size()) % 64 == 0 && script_page_bytes(script.size()) >= script.size() * 4,
                "Script page is too small or misaligned");
    }
}

int main() {
    try {
        uint32_t seed = 3;
        for (uint32_t queries : {1u, 31u, 32u, 33u, 1025u, 10000u})
            for (uint32_t chunk : {1u, 7u, 32u})
                for (uint32_t cores : {1u, 64u})
                    for (const std::string mode : {"none", "primary-list", "weighted-union"})
                        check_plan(queries, 128, 4, chunk, cores, mode, seed++);

        // Near-equal chunk sizes: 70 active batches with G=32 -> 24/23/23.
        std::vector<uint32_t> masks(3, 0xFFFFFFFFu);
        masks[2] = (1u << 6) - 1;
        const auto plan = plan_cluster_major(masks, 1, 70, {{0, 40, 64}}, 32, 4, 4);
        std::multiset<size_t> sizes;
        for (const auto& tasks : plan.worker_tasks) for (const auto& task : tasks) sizes.insert(task.pairs.size());
        require(sizes == std::multiset<size_t>{23, 23, 24} && plan.scan_pages == 6, "Chunks are not near-equal");

        // An all-empty selection still yields one zero-slot merge entry per batch.
        const auto empty = plan_cluster_major({1}, 1, 1, {{0, 0, 32}}, 4, 2, 2);
        require(empty.partial_slots == 0 && empty.tasks == 0, "Empty lists were scheduled");
        require(empty.merge_cores[0].size() + empty.merge_cores[1].size() == 1, "Zero-slot batch not merged");

        // L1 accounting for GloVe-100, depth 2, chunk 32: 2*32*4 query tiles,
        // 2 pages, 32 score tiles, transposed IDs, 32 champion pairs,
        // sentinels, outputs, 32 tile-query ID pages, 64 threshold tiles, the
        // scaler, 32 diff and 32 flag tiles, script.
        const uint64_t expected =
            256 * 2048ull + 2 * (4 * 2048 + 4096) + 32 * 2048 + 4096 + 32 * 6144ull + 6144 + 2 * 6144 + 32 * 128 +
            (64 + 1 + 64) * 2048ull + 64;
        require(scan_l1::worker_cb_bytes(100, 2, 32, 64) == expected, "Scan L1 accounting changed");

        must_reject([] { plan_cluster_major({1}, 1, 1, {{0, 1, 32}}, 0, 1, 1); });
        must_reject([] { plan_cluster_major({1}, 1, 1, {{0, 1, 32}}, 1, 0, 1); });
        must_reject([] { plan_cluster_major({2}, 1, 1, {{0, 1, 32}}, 1, 1, 1); });   // bit past last batch
        must_reject([] { plan_cluster_major({1, 1}, 1, 1, {{0, 1, 32}}, 1, 1, 1); }); // wrong mask shape
        must_reject([] { plan_cluster_major({1}, 1, 1, {{0, 40, 33}}, 1, 1, 1); });   // unpadded list
        must_reject([] { encode_scan_script({ScanTask{0, 0, 1, 1, {{0, 0}, {1, 1}}}}, 1); });
        must_reject([] { encode_merge_args({1, 2, 3, 4}, std::vector<MergeBatch>(200), 341); });
        must_reject([] { parse_fidelity("HiFi5"); });
        // tt::tt_metal::MathFidelity: LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4 (no 1).
        require(parse_fidelity("LoFi") == 0 && parse_fidelity("HiFi2") == 2 && parse_fidelity("HiFi3") == 3 &&
                    parse_fidelity("HiFi4") == 4 && std::string(fidelity_name(4)) == "HiFi4",
                "Fidelity names do not map to MathFidelity values");
        std::cout << "Cluster-major slots, chunks, LPT placement, merge coverage, scripts and L1 accounting passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
