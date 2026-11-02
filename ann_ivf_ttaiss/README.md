# ann_ivf_ttaiss: TTAISS IVF-Flat search

TTAISS (Tenstorrent AI Similarity Search) is the implementation described in
the thesis. Each query is compared only with the vectors of its own nprobe
lists, as in a standard IVF-Flat search, so its results are validated against
an exact per-query IVF search.

## Pipeline per search

1. **Coarse search on the device.** Batches of 32 consecutive queries are
   compared with all centroids; the device keeps an exact top-32 per query, or
   an exact top-64 (two champion tiles) when nprobe > 32. The host validates
   every returned list ID and score and keeps the first nprobe lists of each
   query in rank order.
2. **Host plan** (`gather_schedule.hpp`):
   - the requesters of every list (ascending query ID), cut into 32-query
     **query blocks** (tiles);
   - for every (query, j), the block and row holding its partial result;
   - scan tasks of one list and up to `--batch-chunk` of its blocks, placed
     longest-first on the 64 cores;
   - merge scripts.
3. **Scan program.** Each task reads every page of its list once and multiplies
   it with all of its blocks, keeping one top-32 per block. Partials are written
   in row layout: row r is the block's query r, columns are ranks.
4. **Merge program.** One core per group of 32 original-order queries. Row q'
   of input tile j is query q''s partial for its j-th list (32-byte aligned
   face-segment gathers); an empty selected list gives a sentinel row. The
   inputs are folded with the top-k sort, and outputs are in original query
   order.

Per-list blocks keep 96–98% of the computed rows useful at nprobe 16–32,
against about 12% when every 32-query batch scans the union of its lists.

Two further techniques make the scan fast:

- **(A)** The device builds the query blocks; the host only uploads a small ID table.
- **(B)** A two-pass search skips page × block top-k merges that cannot change a query's result.

The host also plans pass 2 while the device runs pass 1, and decodes the
coarse results and stages the queries in parallel (same bytes as the
sequential code).

## Background

Without (A) and (B), at nprobe 64 (10,000 queries):

| Part | Median |
|---|---:|
| Building the query blocks on the host | 126 ms |
| Their upload | 31 ms |
| Fine device search (scan + merge) | 172 ms |

Most of the fine device time is spent in the top-k merges. A diagnostic kernel that kept the matmuls but skipped every merge took 25 ms instead of 172 ms.

## (A) Query blocks assembled on the device

The host uploads a tile-query table: 32 query IDs per block (128 B), with `0xFFFFFFFF` marking an unused row. This is about 2.6 MB at nprobe 64.

For each task, the scan reader (`kernels/dataflow/reader_scan.cpp`):

1. reads the chunk's ID pages into a reader-private CB (`c_in3`);
2. copies each query row from the original tilized query buffer (already in DRAM for the coarse search) into its row of the chunk's tiles, as two 32-byte face segments per 32-column tile;
3. zero-fills unused rows with L1 stores.

Every read is 32-byte aligned in DRAM and L1.

This costs nothing measurable on the device: at nprobe 64 the fine device time is 171.6 ms with or without it, and the output is byte-identical to building the blocks on the host.

## (B) Skipping hopeless top-k merges (two passes)

A candidate whose score is not above a query's current 32nd-best score cannot enter its final top-32. With `--first-pass-lists m` the search runs four programs back to back on the command queue:

1. **Pass-1 scan** of each query's m best lists (coarse rank order).
2. **Pass-1 merge** into one row-layout top-32 per query, written to extra partial slots. Column 31 holds the 32nd-best score T_q, a lower bound on the final 32nd-best.
3. **Pass-2 scan** of the remaining nprobe − m lists:
   - The reader builds a threshold tile per gathered tile. Row r is T_q of the tile's query r: an aligned read of columns 16–31 of that query's pass-1 row, then column 31 repeated across the row. Unused rows get the largest finite BF16.
   - Per page, compute checks every pair's score tile first: `sub_tiles(score, threshold)`, then `reduce_tile<MAX, REDUCE_SCALAR>`, then a flag read with `read_tile_value`.
   - A pair whose maximum is not positive skips the copies, transposes, `topk_local_sort` and packs.
4. **Final merge** per 32 queries: input 0 is the pass-1 row, then the pass-2 partials.

Ties at T_q may be resolved differently, as BF16 ties already are. Otherwise the final top-32 is exactly the single-pass one.

Kernel protocol details:

- **Full champion ring.** The ring always holds `batch_chunk` pairs: the task's pairs plus sentinel dummies. The slot popped at the front is the slot the next push reserves. A skipped pair, and every dummy, rotates with pop and push only, with no data movement. A merged pair packs its update into the slot it just read, after the unpack → math → pack dependency.
- **Contiguous score windows.** The check reads score tiles by index, and tile addresses do not wrap at the CB end. Each page therefore pads its score window to `batch_chunk` slots (push and pop only), so every window starts at the CB base. Flag tiles are read only at the CB front.
  - Without this, indexed reads ran past the CB end. On hardware that cost 1.5 recall points: 16% of the flags disagreed with a software check. It is now covered by the host model.
- **L1.** The threshold CBs (`c_in4`, double-buffered), the scaler (`c_in5`), diffs (`c_intermed2`), flags (`c_out2`) and ID pages (`c_in3`) are in `scan_l1::worker_cb_bytes`: about 1,076 KiB at chunk 32 without the script, out of 1,362 KiB.
- **Batch chunk.** `--batch-chunk` must be ≤ 32 with two passes (one merge bit per tile).

Partial slots are laid out as `[pass-1 tiles][pass-1 merged rows, one per batch][pass-2 tiles]`. The pass-2 range is an upper bound (`max_tiles`), so pass 1 is enqueued before pass 2 is planned. The host then builds pass 2 while the device runs pass 1:

- plan, scripts and ID table;
- non-blocking uploads;
- programs.

`--first-pass-lists auto` (the default) picks the first-pass size from these measurements (10,000 queries):

| Setting | Fine device time |
|---|---|
| nprobe 64, m = 4 / 6 / 8 / 12 / 16 | 77.9 / 72.4 / 68.3 / 71.2 / 78.2 ms (m = 8 best) |
| nprobe 16, single pass vs m = 2 | 49.4 vs 48.2 ms, and more host preparation (no gain) |

So `auto` uses m = nprobe / 8 from nprobe 24 and a single pass below. `0` forces a single pass.

Offline estimate (`m = 8`, nprobe 64): 75% of the page × tile merges of pass 2 can be skipped. On the device, 11.0% of the pass-2 pairs were merged.

## Threshold rank (compile time)

The second pass's threshold is the first-pass score at rank `threshold_rank`, a
compile-time argument of the scan reader, so each variant is its own binary and
has no runtime branch on it:

- With `--rerank` it is rank 32 (column 31): the host re-scores the device
  top-32, so all 32 candidates must be exact.
- Without re-ranking it is rank k (column k − 1): only the top k are returned,
  and a candidate below the query's k-th best first-pass score cannot enter
  them. This is a much tighter bound. Offline at nprobe 64 with m = 8, the pass-2
  merges still needed fall from 11.9% to 3.3%. Device ranks k+1..32 are then not
  exact and are never returned.

The reader reads the aligned 16-column segment holding the rank (columns 0–15
for k ≤ 16) and spreads that value over the threshold row.

## Programs and buffers kept across searches

The scan, merge and coarse programs, and every DRAM buffer, persist across
searches instead of being rebuilt per search:

- **Programs.** Each search only uploads its scripts and ID tables, updates the
  runtime arguments and enqueues the existing programs.
- **When a program is rebuilt.**
  - A stage's script page (its script CB capacity) grows to the largest script
    any search has needed. Its program is rebuilt only when the page grows.
  - A scan stage is also rebuilt when its threshold rank changes.
  - The coarse program's CBs do not depend on the query count, so it is built
    once.
- **Buffers.** A buffer grows only when a search needs more. A search writes
  and reads only the whole-page region it uses (`write_region`/`read_region`,
  TT-Metal buffer regions).
- **No preset sizes.** Nothing is sized in advance from a maximum; sizes follow
  what searches have needed.

Rebuilding programs every search cost a build plus a kernel-cache lookup per
program in `TT_METAL_CACHE`, which is in the project directory on NFS. While NFS
was slow (5 ms per `open`), that made each search 10–20× slower. With persistent
programs, a traced steady-state search makes no file accesses except `/proc/stat`.

The benchmark warmup is a 32-query search, so the first measured search grows
the buffers and builds the full-size programs. The benchmark reports medians,
which absorb that one run. The energy workload has no warmup, so its first
search includes the build, as every search did before.

## Host speedups (identical output)

- **Coarse decode.** Elements are read in place from the tiled output, with no untilize. `decode_coarse_row_fast` uses no allocation and returns the same result as `decode_coarse_row`: a randomized equivalence test covers ordered, unordered, tied, duplicate, invalid and NaN rows. It runs on 8 threads; spawning 64 threads cost more than the decode.
- **Query staging.** Per-row normalization, BF16 rounding and tiling run in parallel. A one-off check compared the result byte for byte with `tilize_nfaces`.
- **Fewer threads for small jobs.** Small parallel jobs use 8 threads.

## Build, run, validate

```bash
source env.sh
bash build.sh ann_ivf_ttaiss
bash run.sh ann_ivf_ttaiss --dataset glove-100-angular --nlist 512 \
  --nprobe 64 --k 10 --runs 5 --max_num_queries 10000     # --first-pass-lists auto|0|m, --rerank
```

`--results-output`, `--selections-output` and `--assignment-output` feed `validate_gather.py`, which compares every query with an exact per-query IVF search on the host (same normalized, round-to-nearest BF16 inputs, FP32 products, list by list). It checks:

- list membership;
- every rank against the exact per-query search, within one BF16 ulp;
- rescored IDs;
- with `--centroids`, the coarse top-nprobe;
- with `--reranked`, results of `--rerank`.

`run_gather_comparison.sh` runs this executable with a single pass (`fast_single`) and with `auto` (`fast`) on the same queries, plus the measurement-only ablations `fast_rank32`, `fast_rebuild` and `fast_rank32_rebuild` on request. It validates every run and writes `summary.md` (`summarize_gather.py`). The overrides are `NPROBES`, `IMPLEMENTATIONS` and `RERANK=1`.

Timing stages are `h2d`, `cpu_coarse_config`, `tt_coarse_search`, `cpu_fine_prep`, `tt_fine_search` and `cpu_output`, where:

- `tt_fine_search` starts at the first enqueue and includes the overlapped pass-2 host work. The `pass2_host_prep_overlapped` detail row reports that work and is not added again.
- `cpu_fine_prep` covers the pass-1 plan and launch.

The energy workload (`--energy-events`) follows the `ann_ivf_energy` contract.
With `--rerank` it additionally reads the device top-32 back and re-ranks it in
FP32 inside each search interval, because re-ranking produces the results; there
is still no recall check, logging or dump.

`run_energy_comparison.sh` measures energy at matched recall levels against
batch-major `ann_ivf_energy` with the unchanged `ann_ivf_energy` sampler and
analyzer:

- sessions alternate between the implementations;
- each level is `batch_nlist:batch_nprobe/ttaiss_nlist:ttaiss_nprobe[:rerank]`;
- the defaults are the four TT + CPU reference configurations of `run_tt_energy.sh`, each paired with the cheapest TTAISS setting at or above the CPU recall at that level;
- one ordinary run per level, outside every measured window, records the TTAISS recall.

`summarize_energy.py` prints medians and quartiles per level.

## Portable checks

`ctest -R ttaiss`, built with `-DANN_HOST_TESTS_ONLY=ON`:

- **`scan_state_test`** runs the actual scan kernels on a tagged CB model whose pages keep their data: a push with no pack republishes the slot's contents.
  - **Device-built tiles:** every row segment of every assembled tile is the right query's face segment of the right column tile, or zeros for an unused row, read with aligned 32-byte NoC reads.
  - **Threshold mode:** every threshold row holds its query's 32nd-best value. Every (tile, page) is checked exactly once (sub → reduce → flag). Skipped pages never reach the champion. Fully skipped tiles publish sentinels.
  - **Ring and indexing:** the full ring rotates without data movement, and indexed reads never run past the CB end.
  - **Mutations caught:** no score padding, flag read by index, ring not full, merge mask ignored, wrong threshold column, wrong threshold batch page, wrong query row, unused row not zeroed, wrong column tile.
- **`scan_state_rank9_test`** is the same model built with threshold rank 9 (k = 10). A reader that ignores the rank and keeps column 31 fails it.
- **`gather_schedule_test`** covers the two-pass split (rank order), slot shifting with local table rows, the tile upper bound, and the final merge script (input 0 = pass-1 row).
- **`merge_gather_test`** runs the actual merge reader and writer byte by byte, including the merge writer's `page_base` (pass-1 rows into partial slots).
- **`cluster_schedule_test`** checks the L1 accounting with the new CBs.
- **`query_grouping_test`** checks that the fast coarse decode is equivalent to the reference.
- The inherited coarse, rerank, index-build and input tests.

Device arithmetic and timing need a hardware run (`run_gather_comparison.sh`).

## Results (node117; 10,000 queries, medians)

nlist 512, five runs (2026-10-08). "Host-built" is the same search without (A)
and (B); "single pass" has (A) only.

| nprobe | Recall@10 | host-built QPS | single pass QPS | two passes (`auto`) QPS |
|---:|---:|---:|---:|---:|
| 32 | 0.891 | 45,978 | 94,732 | 198,661 |
| 48 | 0.915 | 31,997 | 65,663 | 156,462 |
| 56 | 0.922 | 28,485 | 58,696 | 151,422 |
| 64 | 0.928 | 25,561 | 52,596 | 141,815 |
| 64 + `--rerank` | 0.9555 | 23,691 | 48,261 | 103,782 |

Every run validated against the exact per-query search. See `docs/RESULTS.md`
(the "Gather fast" sections, the name used at the time, and the thesis sweep)
for stages, the full nlist/nprobe grid, energy and limitations.
