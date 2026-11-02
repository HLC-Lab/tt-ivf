# Recorded results and limits

**Names in this log.** Sections keep the executable names used when they were
measured. `ann_ivf_cluster_gather_fast` ("gather fast") is now `ann_ivf_ttaiss`;
`ann_ivf_cluster_major`, `ann_ivf_cluster_major_opt`, `ann_ivf_cluster_gather`
and `ann_ivf_similarity_partitioned` have been removed from the repository
(the last one remains in the Git history). Result folders under `results/`
keep their original names.

Unless a section says it was locally rerun, the hardware numbers below were
supplied by the user from node117. They were not rerun on a TT device during
repository cleanup. Raw logs/captures remain
in the user's measurement directories and are excluded from this repository.
Preserve them separately for the thesis. Portable tests provide different
evidence from device measurements.

## Shared benchmark settings

GloVe angular dataset: 1,183,514 vectors, dimension 100. Unless stated otherwise:
nlist 512, k 10, 10,000 queries per measured run, five measured repetitions,
32 query lanes per device batch. Warmup is excluded. Timings are milliseconds.
Detail rows are included in their parent stage. Medians are per column and
need not sum exactly. Device intervals include enqueue/execution/Finish.

Each query scores its batch's entire candidate-list union. Query grouping
changes that union, including incidental lists from other queries; therefore
recall can change despite preserving each query's own selected lists.

## Host-side optimizations: index build and grouping — 2026-10-07

Locally measured on node117. `ann_ivf_cluster_major_opt` gained exact host
optimizations:

- **Index build:** parallel normalization; assignment vectorized across
  centroids with each score's dimension order unchanged; parallel per-list
  tiling.
- **Weighted-union grouping:** an incrementally merged sorted union; candidates
  costed in collection order with a first-zero stop and early abandonment of
  partial sums; duplicate checks without sorting.

**Identity checks.**

- `--index-digest` gives the same FNV-1a for `--index-build reference` and
  `fast` on GloVe-100 (`1499761db3d9dc89`). It covers 305,070,080 bytes of
  vector tiles, 152,535,040 bytes of ID tiles and the table of contents.
- The grouping keeps the reference permutation on the captured 10,000-query
  coarse selections (order hashes `9fdbbd62fd5aa78b` and `d7b4a3c5e07b190f`)
  and on 120 randomized portable cases.
- Against mem-log in the same session
  (`results/cluster_major_opt_fidelity/20261007_host_opt/`): 0 score
  mismatches at both nprobe values and all 200,000 IDs rescored.

| Step | Before | After |
|---|---:|---:|
| Index build (normalize/assign/tile) | 4.98–5.25 s | 0.90–1.33 s (with or without the zero-fill fix) |
| Energy-window initialization (device + index + upload) | ~5.8 s | 1.39–1.44 s (1.49–1.52 s median in the energy sessions) |
| Grouping, nprobe 16 / 32 (offline, same compiler/flags) | 107.2 / 182.7 ms | 62.8 / 97.6 ms |
| QPS, nprobe 16 / 32 (10,000 queries, 5 runs) | 19,857 / 11,808 | **21,895 / 13,102** |
| Stage sum, nprobe 16 / 32 | 503.2 / 846.4 ms | 456.2 / 762.7 ms |

The coarse device search was left unchanged: it takes 3.0–3.4 ms per 10,000
queries. On-device cluster assignment was not adopted: BF16 scores could
move boundary vectors to another list and change the index.

Energy, rerun with the same protocol as below (hwmon TT power, 10 sessions ×
10 searches × 10,000 queries, alternating order). Output:
`results/energy_raw/cluster_major_comparison/20261007_weighted_union_hwmon_hostopt/`.
Every session had at least 231 TT samples per search window, with a maximum
gap of 24 ms. Medians:

| NList/NProbe | Implementation | Session s | Init s | Searches s | Per search s | QPS (energy run) | Session J | Init J | Searches J | Per search J | Search mJ/query: TT / host / TT + host | Session mJ/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 512/16 | batch-major | 14.201 | 5.761 | 8.279 | 0.828 | 12,079 | 2,737.7 | 1,151.5 | 1,556.9 | 155.7 | 2.840 / 12.718 / 15.569 | 27.377 |
| 512/16 | cluster-major | **6.195** | **1.485** | **4.719** | **0.472** | **21,189** | **1,171.0** | **306.1** | **867.1** | **86.7** | **1.776 / 6.894 / 8.671 (0.557×)** | **11.710 (0.428×)** |
| 512/32 | batch-major | 19.785 | 5.902 | 13.884 | 1.388 | 7,203 | 3,830.3 | 1,173.7 | 2,637.7 | 263.8 | 4.839 / 21.542 / 26.377 | 38.303 |
| 512/32 | cluster-major | **9.286** | **1.522** | **7.752** | **0.775** | **12,901** | **1,765.7** | **309.6** | **1,453.4** | **145.3** | **2.994 / 11.529 / 14.534 (0.551×)** | **17.657 (0.461×)** |

Search energy per query fell from 0.66× to 0.56× of batch-major. That is more
than the QPS gain alone, because the shorter grouping also cuts host package
energy. Sessions now take 0.43–0.47× the time and energy, because
initialization is 3.9× shorter. Batch-major's host energy is about 4–7% above
the previous capture: different host background load at a different time.
Ratios compare sessions taken together within one capture.

## QPS–recall comparison with user-supplied FAISS and earlier TT data — 2026-10-07

**User-pasted measurements** (medians of 10 search runs; host and run
settings as supplied by the user, not rerun here):

- FAISS CPU at 112 threads (`cpu-batched-112`);
- FAISS CPU at 56 threads (`cpu-batched-56`);
- FAISS GPU (`gpu-batched`);
- the earlier Tenstorrent implementation (`tt-balanced-sort`);

each at nlist 512, 1024 and 2048. They are combined with the locally measured
node117 numbers above (medians of 5 runs, nlist 512 only). Combined data:
`results/qps_recall/20261007/qps_recall.csv`. Plot:
`qps_recall_nlist512.png`.

Best measured QPS at recall@10 >= 0.95, over all nlist values:

| Implementation | nlist / nprobe | Recall@10 | QPS | Source |
|---|---|---:|---:|---|
| FAISS GPU | 512 / 64 | 0.9553 | 43,271 | user-supplied |
| TT gather + FP32 re-ranking | 512 / 64 | 0.9555 | 23,818 | node117 |
| TT cluster-major opt + re-ranking | 512 / 24 | 0.9594 | 15,487 | node117 |
| FAISS CPU, 112 threads | 512 / 64 | 0.9554 | 5,822 | user-supplied |
| TT balanced-sort (earlier) | 512 / 32 | 0.9537 | 5,550 | user-supplied |
| FAISS CPU, 56 threads | 512 / 64 | 0.9554 | 3,190 | user-supplied |
| FAISS CPU, 64 threads | 512 / 64 | 0.9553 | 1,064 | node117 |

At >= 0.95, gather with re-ranking reaches:

- 4.3× the earlier TT implementation;
- 4.1× FAISS CPU at 112 threads;
- 0.55× FAISS GPU.

At >= 0.90, gather without re-ranking (nprobe 40, 36,663 QPS) reaches 5.1×
the earlier TT (512/16, 7,229 QPS).

**Limits.**

- The sources differ in host, repetition count (10 against 5) and timed
  interval. TT QPS covers the whole search; FAISS times `index.search` only.
- The new implementations were not measured at nlist 1024 or 2048.

## Gather fast follow-up: rank-k threshold and persistent programs — 2026-10-07/08

Two changes to `ann_ivf_ttaiss` (see its README):

- **Compile-time threshold rank.** Without `--rerank`, the second pass skips against each query's k-th best
  first-pass score instead of its 32nd best.
- **Persistent programs and buffers.** They grow only when a search needs more, and transfers cover only the
  region a search uses.

**Benchmark (locally rerun on node117, 2026-10-08).** `run_gather_comparison.sh` with every configuration
validated: GloVe-100, nlist 512, k 10, 10,000 queries, five measured runs, medians, warmup excluded,
unprofiled. NFS latency had recovered (about 0.23 ms per `open`); steady-state searches make no file accesses.
Raw data: `results/cluster_gather_fast_comparison/20261008_bf16/` and `.../20261008_rerank/`.

QPS, device ranking (BF16):

| nprobe | Recall@10 gather / fast | gather | fast_single | fast | fast before (2026-10-07) | Fine device ms (fast) |
|---:|---|---:|---:|---:|---:|---:|
| 32 | 0.89146 / 0.89095 | 45,978 | 94,732 | **198,661** | 123,557 | 37.3 |
| 48 | 0.91448 / 0.91471 | 31,997 | 65,663 | **156,462** | 111,533 | 45.3 |
| 56 | 0.92279 / 0.92222 | 28,485 | 58,696 | **151,422** | 111,556 | 50.6 |
| 64 | 0.92828 / 0.92888 | 25,561 | 52,596 | **141,815** | 106,538 | 55.8 |

QPS with FP32 re-ranking (recall identical for all three):

| nprobe | Recall@10 | gather | fast_single | fast | fast before (2026-10-07) | Fine device ms (fast) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.91384 | 42,895 | 85,623 | **120,921** | 100,435 | 56.8 |
| 48 | 0.93979 | 29,772 | 57,210 | **113,392** | 100,460 | 61.3 |
| 56 | 0.94880 | 25,437 | 54,005 | **107,191** | 96,580 | 64.0 |
| 64 | 0.95554 | 23,691 | 48,261 | **103,782** | 92,593 | 67.1 |

**Effects of the two changes (nprobe 64, medians):**

| Stage | Before (2026-10-07) | Now |
|---|---:|---:|
| CPU coarse config (coarse program kept) | 5.7 ms | 3.6 ms |
| TT coarse | 3.5 ms | 1.5 ms |
| CPU fine prep | 14.6 ms | 7.2 ms |
| Fine device, BF16 (rank k skips more) | 65.7 ms | 55.8 ms |
| Fine device, re-ranked (unchanged rank 32) | 65.4 ms | 67.1 ms |

- `fast_single` also gains (48,693 → 52,596 QPS at nprobe 64) because only its programs are now persistent.
- Speed-up of `fast` over `gather` at nprobe 64: 5.5× (BF16), 4.4× (re-ranked).

**Correctness:**

- All 24 runs validated: 0 membership violations and 0 ranks more than one BF16 ulp from the exact per-query
  search (worst 0.563); re-ranked runs had 0 misses.
- `--rerank` output at nprobe 64 is byte-identical to the 2026-10-07 run; `--first-pass-lists 0` is
  byte-identical to the original single-pass output.
- A temporary in-process check (10,000 → 3,000 → 10,000 queries on grown buffers) had 0 per-rank score
  mismatches in BF16, re-ranked and single-pass modes.
- Portable tests 80/80, including a rank-9 variant of the scan model.

**Limitations:**

- One session per configuration.
- The end-of-session load averages (about 50) come from the 64-thread validation between timed runs.
- The first measured search after the 32-query warmup grows the buffers and builds the full-size programs;
  the medians absorb it.
- Energy has not been re-measured with these changes; the energy tables below predate them.

## Repository cleanup check: `ann_ivf_ttaiss` — 2026-11-02

Locally rerun on node117 after renaming `ann_ivf_cluster_gather_fast` to `ann_ivf_ttaiss` and removing the
other experiments. Rebuilt against the pinned runtime; host tests: 22/22 passed. The C++ indexer of the
user's editor was running (about half a core); no other device or benchmark activity. Raw data:
`results/ttaiss_check_20261102/`.

- **`run_gather_comparison.sh`** (nlist 512, five runs): every run validated against the exact per-query
  search (0 membership violations, 0 ranks off by more than one BF16 ulp). Recall is identical to the
  earlier runs, and the page × block multiplication counts equal those of the thesis sweep.

  | Configuration | Recall@10 | QPS now | QPS before (2026-10-08) |
  |---|---:|---:|---:|
  | single pass, nprobe 32 | 0.89146 | 93,636 | 94,732 |
  | two passes, nprobe 32 | 0.89095 | 198,561 | 198,661 |
  | single pass, nprobe 64 | 0.92828 | 51,890 | 52,596 |
  | two passes, nprobe 64 | 0.92888 | 139,228 | 141,815 |

- **Thesis operating points** (ten runs, medians, validated; re-ranked run: 0 misses):

  | Configuration | Recall@10 now / thesis sweep | QPS now | QPS thesis sweep |
  |---|---:|---:|---:|
  | 2048/5 | 0.62726 / 0.62726 | 442,607 | 447,056 |
  | 2048/20 | 0.79668 / 0.79668 | 287,350 | 284,624 |
  | 1024/48 | 0.89139 / 0.89139 | 200,938 | 202,847 |
  | 512/64 + re-rank | 0.95554 / 0.95554 | 104,507 | 105,533 |

  Throughput is within ±1% and the fine device time within 0.1 ms. In the re-ranked run, repetition 8
  spent 75 ms instead of 13 ms in host result processing (host interference; device time unchanged),
  which raises the run-to-run RSD but not the median.

- **`run_energy_comparison.sh`** (level `512:32/512:64:rerank`, two sessions per implementation, a smoke
  test of the edited script, not a measurement): TTAISS 2.197 mJ/query for the searches and 5.10 mJ/query
  for the complete window (2026-10-08: 2.192 and 4.94); batch-based 47.1 mJ/query complete (user-supplied
  46.9).

## Thesis sweep: cluster-major implementations over N_list and N_probe — 2026-10-12

Locally rerun on node117 with `results/thesis_sweep_20261012/run_sweep.sh`. The sweep covers 58 configurations:

- `ann_ivf_ttaiss` (two passes `auto`) for N_list 512/1024/2048 and N_probe 1–64, plus FP32
  re-ranking at N_list 512 (the three N_probe = 1 points were added the same day, same settings);
- `ann_ivf_cluster_major_opt` (weighted-union) for N_list 512/1024/2048 and N_probe 1–32, plus re-ranked
  matched points.

Settings: 10,000 queries, ten measured repetitions per configuration, medians, a 32-query warmup excluded,
unprofiled.

- **Hygiene.** `TT_METAL_DPRINT_CORES`, which the user's `~/.bashrc` exports, is unset in the script. A first
  attempt inherited it and was discarded.
- **Other user.** A 30 s monitor of `tang`'s jobs saw no activity during the sweep.
- **Validation.** Gather fast's recall-matched points (2048/5, 2048/20, 1024/48, 512/64+re-rank) validated with
  0 membership violations and 0 rank errors (re-ranked: 0 misses).
- **First repetition.** The first measured repetition of gather fast is 16–41% slower: it builds the full-size
  programs and grows the buffers after the small warmup. The QPS RSD over repetitions 2–10 has a median of
  2.0% (gather fast) and 1.8% (cluster-major opt).

The thesis figures and tables are generated from this data in the separate thesis repository
(`scripts/plot_results.py` with the exported `data/tt_wormhole_*.csv`).

Baselines (Faiss CPU 56/112, A100, the earlier TT batch-major) are user-supplied and stored in
`results/thesis_inputs_user/`. Energy comes from the 2026-10-08 rerun below.

Key matched-recall medians:

| Target | Gather fast config | Recall | QPS | Cluster-major opt config | Recall | QPS |
|---|---|---:|---:|---|---:|---:|
| 60% | 2048/5 | 0.6273 | 447,056 | 2048/3 | 0.6371 | 106,596 |
| 78% | 2048/20 | 0.7967 | 284,624 | 2048/9 | 0.7947 | 60,084 |
| 88% | 1024/48 | 0.8914 | 202,847 | 1024/14 + re-rank | 0.8890 | 32,101 |
| 95% | 512/64 + re-rank | 0.9555 | 105,533 | 512/24 + re-rank | 0.9594 | 15,344 |

## Gather fast ablation of the two follow-up changes — 2026-10-08

Locally rerun on node117 with `ann_ivf_ttaiss/run_gather_comparison.sh` and the
measurement-only switches `--ablation-rank32` and `--ablation-rebuild`. All variants ran in one session:
GloVe-100, nlist 512, k 10, 10,000 queries, five measured runs, medians, warmup excluded, two passes (`auto`),
every run validated.

- The variant with both switches on reproduces the code before the two changes.
- A 30-second monitor of the other user's (`tang`) device jobs recorded no activity during the session.
- Raw data: `results/cluster_gather_fast_comparison/ablation_bf16_20261008T104833Z/` and
  `.../ablation_rerank_20261008T105814Z/`.

QPS, BF16:

| nprobe | Recall@10 | Before both | Programs kept only | Rank-k threshold only | Both |
|---:|---|---:|---:|---:|---:|
| 32 | 0.8908 / 0.8910 | 117,854 | 144,523 | 163,527 | **198,330** |
| 48 | 0.9146 / 0.9147 | 115,773 | 129,572 | 133,906 | **167,822** |
| 56 | 0.9220 / 0.9222 | 104,757 | 125,171 | 128,543 | **149,556** |
| 64 | 0.9288 / 0.9289 | 108,015 | 120,846 | 122,568 | **138,647** |

The first recall is for the rank-32 variants and the second for rank k; they differ only in BF16 tie order.

QPS, FP32 re-ranking (always rank 32; recall identical):

| nprobe | Recall@10 | Programs rebuilt | Programs kept |
|---:|---:|---:|---:|
| 32 | 0.91384 | 105,643 | **121,706** |
| 48 | 0.93979 | 98,940 | **113,120** |
| 56 | 0.94880 | 95,309 | **109,805** |
| 64 | 0.95554 | 93,653 | **105,356** |

**Results:**

- Both changes together give 1.28–1.68× in BF16 and 1.12–1.16× when re-ranking.
- The rank-k threshold cuts fine device time by about 11–19 ms (for example 65.5 → 53.9 ms at nprobe 64) and
  gives 1.13–1.39×.
- Keeping programs and buffers gives 1.12–1.23× and removes per-search program builds and kernel-cache lookups.
- Validation: 0 membership violations, 0 ranks off the exact search, and 0 re-ranking misses in all 24 runs.
- Limitation: one session per variant.

## Energy at matched recall, rerun with all three TT implementations — 2026-10-08

This section supersedes the 2026-10-07 energy section below. That run measured gather fast before the rank-k
threshold and persistent programs, and had only two clean cluster-major opt sessions because NFS slowed down.

**Local measurements.** Locally rerun on node117 with `ann_ivf_ttaiss/run_energy_comparison.sh`
(`IMPLEMENTATIONS="batch_major gather_fast cluster_major_opt"`), using the unchanged `ann_ivf_energy` sampler and
analyzer:

- TT board power from hwmon, one RAPL package counter per prefix;
- 10 independent sessions per implementation and level, rotating order, 10 s cool-down;
- each session runs ten 10,000-query searches.

Session quality:

- Maximum TT sampling gap: 0.021 s.
- NFS latency was about 0.25 ms per `open` at the start.
- Search durations were steady over all 10 sessions of every configuration (for example batch-major 512/32:
  19.37–19.45 s; gather fast 512/64: 1.08–1.16 s). The one outlier was a single cluster-major opt session at
  2048/9 (3.40 s against about 1.7 s); the median absorbs it.

Raw data: `results/energy_raw/gather_fast_comparison/20261008_all/`. Combined table and plot:
`results/energy_comparison/20261008/`.

**User-supplied measurements (not rerun here):** the CPU and GPU rows, and the TT batch-major row (the user's
TT + CPU). The local batch-major sessions reproduce that row within 2–5%: 29.74 / 33.83 / 33.22 / 45.75 mJ/query
against 30.50 / 35.43 / 34.38 / 46.93.

Complete window (index build + searches), system energy per query in mJ (recall in brackets):

| Nominal recall | CPU 56 (user) | CPU 112 (user) | GPU A100 + CPU (user) | TT batch-major + CPU (user) | TT cluster-major opt + CPU (local) | TT gather fast + CPU (local) |
|---|---:|---:|---:|---:|---:|---:|
| 60% | 69.96 (0.6046) | 69.15 (0.6046) | 9.33 (0.6047) | 30.50 (0.6224) | 7.21 (0.6371) | **6.10** (0.6273) |
| 78% | 79.82 (0.7833) | 75.27 (0.7833) | 9.41 (0.7833) | 35.43 (0.7916) | 8.59 (0.7947) | **6.33** (0.7967) |
| 88% | 92.12 (0.8849) | 72.92 (0.8849) | 10.64 (0.8849) | 34.38 (0.8803) | 9.66 (0.8890) | **4.93** (0.8914) |
| 95% | 218.72 (0.9554) | 149.29 (0.9554) | 20.11 (0.9553) | 46.93 (0.9537) | 14.79 (0.9594) | **4.94** (0.9555) |

Configurations (nlist/nprobe), each the cheapest measured setting at or above the CPU recall:

| Level | Cluster-major opt (weighted-union) | Gather fast |
|---|---|---|
| 60% | 2048/3 | 2048/5 |
| 78% | 2048/9 | 2048/20 |
| 88% | 1024/14 + re-ranking | 1024/48 |
| 95% | 512/24 + re-ranking | 512/64 + re-ranking |

Re-ranked levels include the readback and FP32 re-ranking inside the window.

Searches-only window (initialization excluded), local runs:

| Level | Batch-major mJ/q | Cluster-major opt mJ/q | Gather fast mJ/q | Batch-major s (100k) | Opt s | Gather fast s |
|---|---:|---:|---:|---:|---:|---:|
| 60% | 4.77 | 1.72 | **0.63** | 2.76 | 1.00 | 0.35 |
| 78% | 9.25 | 3.09 | **0.87** | 5.28 | 1.74 | 0.47 |
| 88% | 17.73 | 6.12 | **1.21** | 10.03 | 3.38 | 0.64 |
| 95% | 34.25 | 12.05 | **2.19** | 19.42 | 6.68 | 1.12 |

**Findings (complete window):**

- **Gather fast** is the lowest at every level.
  - Against the A100 + CPU: 0.65× (60%), 0.67× (78%), 0.46× (88%), 0.25× (95%).
  - Against the 112-core CPU at 95%: 0.033×.
  - Against local batch-major: 0.11–0.21×.
- **Cluster-major opt** is below the A100 + CPU at every level: 0.77×, 0.91×, 0.91×, 0.74×. It uses 0.24–0.32× of
  batch-major's energy.
- **Searches only:** gather fast uses 0.13× (60%) to 0.06× (95%) of batch-major's energy.
- The complete window of gather fast (2.4–2.8 s) is dominated by initialization (index build and device setup)
  and host idle power.

**Limitations:**

- System energy is TT board + CPU package energy, not wall-socket energy, and no idle baseline is subtracted.
  Host package power (about 145–200 W) dominates.
- The CPU and GPU rows are user-supplied; their window definition was not verified (assumed complete, like the
  user's TT row).
- Gather and opt recall come from one ordinary run per level outside the measured windows.
- Batch-major and cluster-major opt still build programs every search (only gather fast keeps them), so they
  are more exposed to filesystem latency. Their durations were steady in this run.

## Energy at matched recall: gather fast against CPU, GPU and batch-major TT — 2026-10-07 (superseded by 2026-10-08)

**Local measurements.** Locally rerun on node117 with
`ann_ivf_ttaiss/run_energy_comparison.sh`, using the unchanged
`ann_ivf_energy` sampler and analyzer:

- TT board power from hwmon, one RAPL package counter per prefix;
- 10 independent sessions per implementation and level, alternating order, 10 s cool-down;
- each session runs ten 10,000-query searches (100k queries);
- maximum TT sampling gap 0.021 s.

Raw data: `results/energy_raw/gather_fast_comparison/20261007_levels/`.
Combined table and plot: `results/energy_comparison/20261007/`.

The two local implementations:

- **Batch-major:** `ann_ivf_energy`, grouping `none`, at the four TT + CPU reference configurations.
- **Gather fast:** `ann_ivf_ttaiss`, the cheapest measured setting whose recall is at least the CPU recall of the level. At the 95% level its workload includes the readback and FP32 re-ranking inside every search interval.

**User-supplied measurements (not rerun here):** the CPU 56 cores, CPU 112
cores, GPU (A100) + CPU and TT + CPU rows below.

The user's TT + CPU numbers are the complete window (index build plus
searches). The local batch-major rerun reproduces them: 31.13 / 35.65 / 34.55 /
46.86 mJ/query against the user's 30.50 / 35.43 / 34.38 / 46.93. At 512/32 the
local run took 25.20 s against the user's 25.29 s. The table therefore compares
complete windows. The window definition of the user's CPU and GPU rows was not
supplied; it is assumed to be the same.

System energy per query (mJ; TT or GPU board plus CPU package; medians; recall in brackets):

| Nominal recall | CPU 56 (user) | CPU 112 (user) | GPU A100 + CPU (user) | TT batch-major + CPU (user) | TT cluster-major opt + CPU (local, 2 sessions) | TT gather fast + CPU (local) |
|---|---:|---:|---:|---:|---:|---:|
| 60% | 69.96 (0.6046) | 69.15 (0.6046) | 9.33 (0.6047) | 30.50 (0.6224) | 7.48 (0.6371) | **6.35** (0.6273) |
| 78% | 79.82 (0.7833) | 75.27 (0.7833) | 9.41 (0.7833) | 35.43 (0.7916) | 8.74 (0.7947) | **6.77** (0.7967) |
| 88% | 92.12 (0.8849) | 72.92 (0.8849) | 10.64 (0.8849) | 34.38 (0.8803) | 10.18 (0.8890) | **5.53** (0.8914) |
| 95% | 218.72 (0.9554) | 149.29 (0.9554) | 20.11 (0.9553) | 46.93 (0.9537) | 16.24 (0.9594) | **5.26** (0.9555) |

Configurations (nlist/nprobe), each the cheapest measured setting at or above the CPU recall:

| Level | Cluster-major opt (weighted-union) | Gather fast |
|---|---|---|
| 60% | 2048/3 | 2048/5 |
| 78% | 2048/9 | 2048/20 |
| 88% | 1024/14 + re-ranking | 1024/48 |
| 95% | 512/24 + re-ranking | 512/64 + re-ranking |

Cluster-major opt was measured in a second run with the same script (`IMPLEMENTATIONS="batch_major cluster_major_opt"`, raw data
`results/energy_raw/gather_fast_comparison/20261007_levels_opt/`). Its re-ranked levels read the results
back and re-rank them inside the window, like gather fast. During that run the NFS mount holding the project
(and the TT-Metal kernel cache, consulted when each search creates its programs) slowed down: from session 3
onward every session of both implementations took 3–6 s longer, while host power stayed normal. Afterwards a
`stat` took 0.5 ms and an `open` 8 ms on the project directory, against 5 µs and 12 µs on local disk. Only
sessions 1–2, before the slowdown, are used for cluster-major opt; the values above are their means, and the
two sessions agree within 5–8%. That run's batch-major sessions 1–2 reproduce the first run (complete window
31.16 / 35.98 / 34.37 / 48.19 mJ/query against 31.13 / 35.65 / 34.55 / 46.86). Searches-only, cluster-major
opt used 1.77 / 2.99 / 6.33 / 13.43 mJ/query. Rerun it with 10 clean sessions when the NFS latency is back to
normal.

Complete-window detail for gather fast:

| Level | 100k-query energy J (Q1–Q3) | Duration s | vs local batch-major |
|---|---|---:|---:|
| 60% | 635 (624–652) | 2.80 | 0.20× |
| 78% | 677 (666–680) | 2.99 | 0.19× |
| 88% | 553 (533–562) | 2.67 | 0.16× |
| 95% | 526 (516–529) | 2.60 | 0.11× |

Searches-only window (initialization excluded), local runs:

| Level | Gather fast mJ/query | Batch-major mJ/query | Ratio | Gather fast search s (100k) | Batch-major search s (100k) |
|---|---:|---:|---:|---:|---:|
| 60% | 0.78 | 4.82 | 0.16× | 0.45 | 2.76 |
| 78% | 1.06 | 9.50 | 0.11× | 0.57 | 5.28 |
| 88% | 1.67 | 18.31 | 0.09× | 0.89 | 10.04 |
| 95% | 2.44 | 35.10 | 0.07× | 1.22 | 19.42 |

**Findings:**

- At every level, gather fast uses less energy per query than every other platform in the complete window.
- Cluster-major opt (2 sessions) is below the A100 + CPU at every level (0.80× at 60%, 0.81× at 95%) and at
  0.25–0.35× of batch-major TT. Gather fast is a further 0.85× (60%) to 0.32× (95%) of cluster-major opt; the
  gap widens with recall because at 95% opt needs both batch-union scanning and re-ranking.
  - At 95% recall: 0.26× the A100 + CPU, 0.035× the 112-core CPU, and 0.11× batch-major TT.
  - At 60% recall: 0.68× the A100 + CPU.
- Gather fast's complete window is dominated by initialization (about 2.6 s of index build and device setup against 0.45–1.2 s of searching). The searches-only energy is 0.8–2.4 mJ/query.

**Limitations:**

- System energy is TT board + CPU package energy, not wall-socket energy, and no idle baseline is subtracted. Host package power (about 145–160 W, mostly idle power of the two-socket host) dominates the TT figures; the TT board draws 22–40 W.
- The CPU and GPU rows are user-supplied. Their measurement window and power sources were not verified locally.
- Gather fast's recall comes from one ordinary run per level outside the measured windows (ann_ivf_energy computes none). The batch-major recall shown is the user's.
- Gather-fast search windows are short (0.45–1.2 s; at least 22 TT samples per session at hwmon's whole-watt resolution). The quartile ranges stay within ±4% of the median.

## Gather fast: device-built query tiles and merge skipping — 2026-10-07

Locally rerun on node117 (Wormhole n150, TT-Metal v0.66.0 lock). Settings:

- `ann_ivf_ttaiss` (see its README) against `ann_ivf_cluster_gather`;
- GloVe-100, nlist 512, k 10, 10,000 queries;
- five measured runs, medians, warmup excluded, unprofiled;
- `--batch-chunk 32`, HiFi4.

Each configuration ran sequentially with
`ann_ivf_ttaiss/run_gather_comparison.sh`, and every run was
validated. Raw output: `results/cluster_gather_fast_comparison/20261007_bf16/`
and `.../20261007_rerank/`.

The three configurations:

- `gather`: `ann_ivf_cluster_gather`, which builds the gathered query tiles on the host.
- `single`: this executable with `--first-pass-lists 0`. The scan reader builds the tiles on the device from a 32-ID-per-tile table.
- `fast`: this executable with `--first-pass-lists auto` (m = nprobe / 8).
  - Two passes: the second pass skips page × tile top-k merges whose scores cannot beat the query's first-pass 32nd-best score.
  - The second pass's host preparation overlaps the first pass on the device.
  - Parallel coarse decode and query staging (same output).

Device ranking (BF16):

| nprobe | Recall@10 gather / single / fast | QPS gather | QPS single | QPS fast | Fine device ms (gather / single / fast) | Stage sum ms (gather / fast) |
|---:|---|---:|---:|---:|---|---|
| 32 | 0.89146 / 0.89146 / 0.89081 | 46,764 | 82,044 | **123,557** | 90.3 / 85.4 / 55.1 | 213.3 / 80.5 |
| 48 | 0.91448 / 0.91448 / 0.91463 | 32,209 | 61,129 | **111,533** | 131.4 / 126.8 / 59.8 | 309.8 / 89.2 |
| 56 | 0.92279 / 0.92279 / 0.92202 | 28,836 | 54,436 | **111,556** | 151.5 / 146.7 / 61.8 | 346.2 / 89.1 |
| 64 | 0.92828 / 0.92828 / 0.92878 | 24,962 | 48,693 | **106,538** | 171.9 / 167.0 / 65.7 | 399.9 / 93.3 |

With host FP32 re-ranking of the device top-32 (`--rerank`):

| nprobe | Recall@10 (all three) | QPS gather | QPS single | QPS fast | Speed-up fast / gather |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.91384 | 40,842 | 77,634 | **100,435** | 2.5× |
| 48 | 0.93979 | 29,127 | 57,210 | **100,460** | 3.4× |
| 56 | 0.94880 | 26,587 | 50,459 | **96,580** | 3.6× |
| 64 | 0.95554 | 23,977 | 45,011 | **92,593** | 3.9× |

**Where the time went (nprobe 64, BF16, medians in ms):**

| Configuration | CPU fine prep | H2D | CPU coarse config | Fine device |
|---|---:|---:|---:|---:|
| gather | 168.8 | 32.1 | 15.6 | 171.9 |
| single | 23.1 | 2.1 | 6.1 | 167.0 |
| fast | 14.6 | 1.0 | 5.7 | 65.7 |

`fast` shows parallel decode and staging. Its fine device time includes the overlapped pass-2 host work (about 17 ms).

**Diagnostics (single runs, nprobe 64; not benchmark medians):**

- Top-k merges dominate the fine device time: a kernel that skipped every merge took 25 ms instead of 172 ms.
- Offline estimate for m = 8: pass 2 needs 11.9% of its 1.32 M page × tile merges. A debug kernel measured 11.0% on the device, and its software check agreed with every device flag.
- m sweep (fine device ms): 4 → 77.9, 6 → 72.4, 8 → 68.3, 12 → 71.2, 16 → 78.2.
- At nprobe 16, two passes give no gain (fine device 49.4 → 48.2 ms, more host preparation), so `auto` uses a single pass below nprobe 24.

**Correctness:** every run (24) validated with:

- 0 list-membership violations;
- 0 ranks more than one BF16 ulp from the exact per-query IVF search (worst 0.563);
- all 100,000 IDs rescored within tolerance;
- coarse selections equal to the top-nprobe centroids within 1.5 BF16 ulp.

Re-ranked runs had 0 re-ranking misses. Beyond validation:

- `single` gives exactly the `gather` results (byte-identical neighbor dumps at nprobe 64).
- `fast` returns the same per-rank scores; its recall differs by at most 0.0008 because BF16 ties resolve to different IDs.
- With re-ranking, the recall of all three configurations is identical.

**Limitations:**

- One measurement session per configuration.
- Recorded load averages were high at the end (about 49). Afterwards, nothing used CPU apart from the VS Code C++ indexer (about 25% of one core), so the load comes from tasks waiting on I/O and the sequential numpy validation between runs, not from concurrent work during the timed runs.
- QPS covers the per-search stages only (the index is built once).
- A first two-pass version addressed score tiles past a wrapped CB window and lost 1.5 recall points. It was caught by validation and fixed, and is now covered by the host model.

## ~0.95 recall: FP32 re-ranking and FAISS reference — 2026-10-07

The device ranks with BF16 scores. `--rerank`, in `ann_ivf_cluster_gather` and
`ann_ivf_cluster_major_opt`, re-scores each query's device top-32 on the host
in FP32 with the normalized vectors and keeps the best 10. The time is
included in `cpu_output` (`rerank` detail row).

Locally run on node117. Settings: 10,000 queries, five measured runs,
nlist 512, k 10. Batch-based runs use weighted-union grouping. Output:
`results/cluster_gather_comparison/20261007_rerank/`.

| Configuration (FP32 re-ranking) | Recall@10 | QPS | Stage sum (ms) | Fine device (ms) |
|---|---:|---:|---:|---:|
| batch-based opt, nprobe 16 | 0.934060 | 21,052 | 474.6 | 357.4 |
| batch-based opt, nprobe 20 | 0.949230 | 17,391 | 574.5 | 431.9 |
| batch-based opt, nprobe 24 | 0.959390 | 15,487 | 645.2 | 502.5 |
| batch-based opt, nprobe 32 | 0.974830 | 12,679 | 788.1 | 621.2 |
| gather, nprobe 48 | 0.939790 | 27,957 | 356.0 | 131.5 |
| gather, nprobe 56 | 0.948800 | 26,262 | 380.2 | 151.1 |
| gather, nprobe 64 | 0.955540 | 23,818 | 419.2 | 171.6 |

FAISS reference (`ann_ivf_energy/run_faiss_benchmark.py`, faiss-cpu 1.15.1,
64 OpenMP threads, IVF512 with an FP16 scalar quantizer, inner product on
normalized vectors). Output: `results/faiss_reference/20261007_ivf512/`.

| FAISS nprobe | Recall@10 | QPS (median of 5; range) |
|---:|---:|---:|
| 16 | 0.8572 | 4,090 (3,833–4,121) |
| 32 | 0.9138 | 2,002 (1,932–2,040) |
| 48 | 0.9396 | 1,383 (1,317–1,434) |
| 64 | 0.9553 | 1,064 (1,032–1,085) |

**At about 0.95 recall:**

| Implementation | Recall@10 | QPS | Relative to FAISS nprobe 64 |
|---|---:|---:|---:|
| gather nprobe 64 + re-ranking | 0.9555 | 23,818 | 22.4× |
| gather nprobe 56 + re-ranking | 0.9488 | 26,262 | 24.7× |
| batch-based nprobe 20 + re-ranking | 0.9492 | 17,391 | 16.3× |
| FAISS CPU nprobe 64 | 0.9553 | 1,064 | 1× |

At equal recall, gather is 1.51× faster than batch-based (nprobe 56 against
20). Batch-based also gains from re-ranking: 0.934 at nprobe 16 and 0.975 at
nprobe 32.

**Validation.** Every re-ranked gather run equals the exact per-query FP32 IVF
search:

- 0 re-ranking misses, so no true top-10 neighbour fell outside the device
  BF16 top-32;
- 0 returned IDs outside the selected lists;
- all scores within 7e-7 of a host FP32 recomputation;
- the coarse check passed.

**Limits.**

- Re-ranking took 13–17 ms per search in most runs but 30–113 ms in some. It
  reads 32 scattered vectors per query from a 473 MB host array, so it is
  sensitive to host memory and scheduling.
- The TT QPS covers the whole search: coarse, host preparation, H2D, device
  fine search, readback and re-ranking. FAISS QPS times only `index.search`.
- FAISS trains its own k-means (default seed) instead of reading the project
  centroid file. Recall at nprobe 64 is nevertheless the same (0.9553 against
  0.9555).
- The host load average before the FAISS runs was raised by the preceding
  validation jobs, and FAISS uses every host thread. The spread across runs
  is given.

## Gather with coarse top-64 (nprobe up to 64) — 2026-10-07

`ann_ivf_cluster_gather` now asks the coarse kernel for an exact top-64 when
nprobe > 32 (nlist >= 64). Two champion tiles are kept per query batch: A for
ranks 1–32 and B for ranks 33–64. Each centroid tile C costs two runs of the
existing 64-element sort:

1. A with C. The top half is the new A; the 32 elements that drop out go
   through L1, because PACK reads DST only after the whole MATH section.
2. B with those dropped elements. The top half is the new B.

This is exact because every element of B is <= every element of A. The
coarse device stage stays about 2–3 ms.

Locally run on node117. Outputs: `results/cluster_gather_comparison/20261007_top64/`
and `20261007_batch_np20_24/`. Settings: 10,000 queries, five measured runs,
nlist 512, k 10. The batch-based reference is `ann_ivf_cluster_major_opt`
with weighted-union grouping. Medians; fine-search spread at most 1.1%.

| Configuration | Recall@10 | QPS | Stage sum (ms) | Fine device (ms) | CPU fine prep (ms) | Scan pairs |
|---|---:|---:|---:|---:|---:|---:|
| batch-based opt, nprobe 16 | 0.909980 | 21,810 | 458.0 | 357.3 | 80.3 | 3,232,391 |
| batch-based opt, nprobe 20 | 0.923310 | 18,455 | 541.4 | 432.3 | 90.5 | 3,923,281 |
| batch-based opt, nprobe 24 | 0.932310 | 15,932 | 626.9 | 502.6 | 102.3 | 4,581,196 |
| batch-based opt, nprobe 32 | 0.944660 | 13,090 | 763.5 | 621.5 | 116.7 | 5,661,369 |
| gather, nprobe 32 | 0.891460 | 47,025 | 212.2 | 90.4 | 83.9 | 786,690 |
| gather, nprobe 40 | 0.904270 | 36,663 | 271.9 | 111.3 | 121.2 | 970,118 |
| gather, nprobe 48 | 0.914480 | 32,158 | 310.5 | 131.3 | 138.7 | 1,155,442 |
| gather, nprobe 56 | 0.922790 | 28,120 | 355.1 | 151.1 | 153.9 | 1,339,694 |
| gather, nprobe 64 | 0.928280 | 25,994 | 384.1 | 171.5 | 167.9 | 1,525,034 |

Equal-recall comparisons:

| Recall | Gather | Batch-based | Gather speedup |
|---|---|---|---:|
| ≈0.89 | nprobe 32 | nprobe 12 (27,318 QPS, earlier session) | 1.72× |
| ≈0.91 | nprobe 48 (0.9145) | nprobe 16 (0.9100) | 1.47×, with higher recall |
| ≈0.923 | nprobe 56 (0.9228) | nprobe 20 (0.9233) | 1.52× |
| 0.928 / 0.932 | nprobe 64 (0.9283) | nprobe 24 (0.9323) | 1.63×, at 0.4 points lower recall |

Above 0.928 only the batch-based search is measured (0.945 at nprobe 32).

**Validation of every gather run** (`validate_gather.py --centroids`):

- **Coarse:** 0 queries whose selected lists were not their top-nprobe
  centroids beyond a 1.5 BF16 ulp tolerance. The worst margin, 1.009–1.054
  ulp, matches the earlier top-32 kernel's (1.018–1.053).
- **Fine:** 0 membership violations, 0 ranked scores more than one ulp from
  the exact per-query search (worst 0.563), and all 100,000 IDs rescored.
- **Recall gap:** the exact FP32-tie-broken recall is 0.951 at nprobe 64
  against 0.928 on the device. BF16 ties account for the difference.

At nprobe 48–64 the host work in CPU fine preparation equals the fine device
time (139–168 ms against 131–172 ms): plan, gathered tile copy and merge
scripts. It is the next bottleneck.

## Per-list query tiles (gather) — 2026-10-07

Locally run on node117 with `ann_ivf_cluster_gather/run_gather_comparison.sh`.
Outputs: `results/cluster_gather_comparison/20261007_first/` and
`20261007_batch_low_nprobe/`. Settings: nlist 512, k 10, 10,000 queries, five
measured runs, batch chunk 32. The batch-based reference is
`ann_ivf_cluster_major_opt` with weighted-union grouping. Gather uses no
grouping.

| Configuration | Recall@10 | QPS | Stage sum (ms) | Fine device (ms) | Scan pairs |
|---|---:|---:|---:|---:|---:|
| batch-based opt, nprobe 4 | 0.755800 | 67,056 | 148.7 | 85.2 | 741,170 |
| batch-based opt, nprobe 6 | 0.813860 | 47,465 | 210.0 | 135.6 | 1,192,351 |
| batch-based opt, nprobe 8 | 0.847610 | 37,691 | 264.8 | 182.1 | 1,623,285 |
| batch-based opt, nprobe 12 | 0.888850 | 27,318 | 365.6 | 275.0 | 2,472,481 |
| batch-based opt, nprobe 16 | 0.909980 | 21,769 | 458.9 | 357.2 | 3,232,391 |
| batch-based opt, nprobe 32 | 0.944660 | 13,211 | 756.4 | 620.9 | 5,661,369 |
| gather, nprobe 16 | 0.838290 | 74,324 / 74,664 | 134.1 / 133.4 | 49.4 / 49.0 | 412,702 |
| gather, nprobe 24 | 0.870710 | 56,917 / 55,965 | 175.1 / 178.1 | 69.5 / 69.8 | 599,674 |
| gather, nprobe 32 | 0.891460 | 45,933 / 45,432 | 217.2 / 219.5 | 90.9 / 90.5 | 786,690 |

Gather rows were measured twice, in two sessions. Every gather run validated
against an exact per-query IVF search on the host (`validate_gather.py`):

- 0 returned IDs outside the query's selected lists;
- 0 ranked scores more than one BF16 ulp from the exact search (worst 0.62);
- all 100,000 IDs rescored.

**At equal recall,** gather at nprobe 32 (0.8915, about 45.4–45.9k QPS) is 1.66×
faster than the batch-based search at nprobe 12 (0.8889, 27.3k QPS). At about
0.84 it is about 2× faster: nprobe 16 against batch nprobe 8.

**Limits.**

- Gather's recall stops at 0.8915. The coarse kernel returns at most 32 lists,
  and gather searches only each query's own lists. The batch-based search
  reaches 0.910 and 0.945 thanks to its batch-mates' extra lists.
- BF16 score ties cost about 2 points: the host exact search with FP32
  tie-breaking gives 0.910 at nprobe 32, against 0.891 on the device.
- Host-side gather costs about 50–90 ms per search (planning, tile copy, script
  build) plus 7–14 ms of uploads. This is now comparable to the fine device
  search.
- The load average during these runs included the processes' own 64-thread
  index build and the numpy validation between runs. Neither overlapped a
  timed search.

## Energy after the host optimizations — 2026-10-07

Locally measured on node117 with the same protocol as below:
`run_energy_comparison.sh`, hwmon TT power, 10 sessions per implementation and
configuration, each 10 searches × 10,000 queries, weighted-union, nlist 512.
Output: `results/energy_raw/cluster_major_comparison/20261007_weighted_union_hwmon_hostopt/`.
`ann_ivf_cluster_major_opt` now includes the exact parallel index build and
the faster grouping. All 40 sessions completed, each with at least 231 in-window
TT samples and a maximum gap of 20 ms.

**Host contamination.** From about 11:23 (session 4 onward) the Claude Code
session restarted and ran small analysis jobs on the host. Host package
power rose from a steady 138.5–139.1 W to 143–179 W for both implementations.
One batch-major 512/16 session's searches also took 13.5 s instead of 8.2 s.
The full medians (`summary.md`) therefore have high host-energy spread, up to
22% RSD. The table below uses only the uncontaminated sessions 1–3; TT
energy and durations of the other sessions agree with them.

Sessions 1–3, medians:

| NList/NProbe | Implementation | Session s | Init s | Searches s | QPS | TT mJ/query (searches) | TT + host mJ/query (searches) | TT + host mJ/query (session) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512/16 | batch-major | 13.750 | 5.523 | 8.229 | 12,153 | 2.826 | 14.248 | 25.056 |
| 512/16 | cluster-major opt | 6.089 | 1.417 | 4.681 | 21,365 | 1.766 | 8.258 | 11.072 |
| 512/32 | batch-major | 19.578 | 5.834 | 13.762 | 7,266 | 4.808 | 23.932 | 35.377 |
| 512/32 | cluster-major opt | 9.137 | 1.418 | 7.726 | 12,944 | 2.988 | 13.733 | 16.553 |

Cluster-major opt against batch-major:

| Metric | 512/16 | 512/32 |
|---|---:|---:|
| QPS | 1.76× | 1.78× |
| TT board energy per query (searches) | 0.62× | 0.62× |
| TT + host energy per query (searches) | 0.58× | 0.57× |
| Session time | 0.44× | 0.47× |
| Session energy | 0.44× | 0.47× |

Initialization drops from 5.5–5.8 s to 1.4 s. Search energy per query improves
on the earlier capture (9.97 → 8.26 and 16.09 → 13.73 mJ/query) because grouping
is faster.

Host package power is about the same for both implementations (≈139 W), so host
energy follows elapsed time. For final reporting, rerun the capture with nothing
else running on the host.

## Energy: optimized cluster-major versus batch-major — 2026-10-07

Locally measured on node117 with
`ann_ivf_cluster_major_opt/run_energy_comparison.sh`. Output:
`results/energy_raw/cluster_major_comparison/20261007_weighted_union_hwmon/`.

- Batch-major `ann_ivf_energy` was compared with
  `ann_ivf_cluster_major_opt --energy-events` at HiFi4.
- Settings: weighted-union, nlist 512, k 10.
- 10 independent processes per implementation and configuration, each running
  10 searches × 10,000 queries (100,000 queries per session). Implementation
  order alternated per session, with a 10 s cool-down and the kernel cache
  warmed outside the sessions.
- TT power: the kernel-driver hwmon `power1_input` (whole watts, sampled at
  about 50 Hz). Every session had at least 252 TT samples inside its window,
  with a maximum gap of 21 ms.
- Host: one RAPL package counter per prefix (`package-0-die-0`,
  `package-1-die-4`).
- Session-to-session RSD is at most 1.5% for TT energy and 2.7% for TT + host.
- Board + CPU package energy is not wall-socket energy, and no idle baseline
  is subtracted.

Search-only window (sum of the 10 search intervals), medians per query:

| NList/NProbe | Implementation | TT mJ/query | Host mJ/query | TT + host mJ/query | Search time (s / 100k queries) | QPS | TT W | Host W |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512/16 | batch-major | 2.831 | 12.188 | 15.018 | 8.246 | 12,127 | 34.37 | 147.95 |
| 512/16 | cluster-major | **1.860** | **8.108** | **9.966 (0.664×)** | 5.114 | 19,554 | 36.36 | 159.04 |
| 512/32 | batch-major | 4.814 | 20.094 | 24.911 | 13.799 | 7,247 | 34.89 | 145.54 |
| 512/32 | cluster-major | **3.167** | **12.921** | **16.085 (0.646×)** | 8.527 | 11,728 | 37.12 | 151.55 |

Energy-workload QPS is 100,000 queries divided by the median summed search time.
That workload has no final readback or recall.

Session times and energy (TT + host), medians over the 10 sessions. The times
come from each session's C++ markers, and each phase's energy from the
analyzer window of the same name (`full`, `initialization`, `searches`):

| NList/NProbe | Implementation | Session s | Initialization s | Searches total s | Per search s | Session J | Initialization J | Searches J | Per search J |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/16 | batch-major | 14.044 | 5.798 | 8.246 | 0.825 | 2,665.9 | 1,163.3 | 1,501.8 | 150.2 |
| 512/16 | cluster-major | 11.054 | 5.909 | 5.114 | 0.511 | 2,220.5 | 1,232.4 | 996.6 | 99.7 |
| 512/32 | batch-major | 19.618 | 5.819 | 13.799 | 1.380 | 3,671.9 | 1,185.4 | 2,491.1 | 249.1 |
| 512/32 | cluster-major | 14.370 | 5.841 | 8.527 | 0.853 | 2,815.4 | 1,212.8 | 1,608.5 | 160.9 |

- A session is the full window: device and index creation, then 10 searches of
  10,000 queries. There is no measurable gap between searches.
- Initialization is device creation, normalization, IVF assignment, tiling
  and uploads, and takes about 5.8–5.9 s in both. Its energy is about 6%
  higher for cluster-major (1.23 vs 1.16–1.19 kJ) at the same duration.
  That is host package power during the CPU index build, not a difference in
  the search.

The full window (index creation and searches), TT + host per query:

| NList/NProbe | Batch-major (mJ/query) | Cluster-major (mJ/query) | Ratio |
|---|---:|---:|---:|
| 512/16 | 26.659 | 22.205 | 0.833× |
| 512/32 | 36.719 | 28.154 | 0.767× |

Initialization is the same in both, so it dilutes the search savings.

**Interpretation.**

- Cluster-major draws about 6% more board power, because the Tensix cores
  are busier. It finishes searches in 0.62× the time, so TT energy per query
  falls by 34%.
- Host CPU packages account for about 80% of the measured energy. They idle
  at roughly 145–160 W on this two-socket host, so shorter searches save host
  energy roughly in proportion.
- Recall is that of the timing runs: 0.909980 / 0.944660 against
  0.910120 / 0.945070. The energy workload checks no recall.

**Invalid earlier capture.** `20261007_weighted_union`, taken with the
`tt-smi` power source, is invalid for TT energy. While a TT-Metal process held
the device, every `tt-smi` call failed with `Failed to map bar0_uc`. Most
sessions contained only the idle 11 W readings taken before and after the
workload, and the analyzer integrated those. Its RAPL host energy is
unaffected.

The sampler now prefers hwmon and stops a session when TT samples inside the
window are more than 2 s apart. Any older capture made with the `tt-smi`
source should be checked for in-window TT samples (`TTSamples`, `TTMaxGap_s`)
before use.

## Math fidelity of the optimized cluster-major search — 2026-10-07

Locally rerun on node117 with `ann_ivf_cluster_major_opt/run_fidelity_sweep.sh`
on an idle host. Output: `results/cluster_major_opt_fidelity/20261007_weighted_union/`.
The executable is the pruned optimized build: batched matmul, IDs once,
narrow mask, top-k partials, no program caching. Settings: weighted-union,
nlist 512, k 10, 10,000 queries, five measured runs, batch chunk 32. Each
variant's final run is compared with mem-log (HiFi4) in the same session.
Medians in ms; within-configuration spread is at most 0.4%.

| Fine / coarse fidelity | Fine, nprobe 16 | QPS, nprobe 16 | Recall@10, nprobe 16 | Fine, nprobe 32 | QPS, nprobe 32 | Recall@10, nprobe 32 | Score mismatches / 100,000 (nprobe 16) | Worst rescore (ulp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem-log HiFi4 | 651.684 | 12,301.47 | 0.910120 | 1128.898 | 7,312.24 | 0.945070 | reference | — |
| HiFi4 / HiFi4 | 357.227 | 19,857.45 | 0.909980 | 620.937 | 11,808.27 | 0.944660 | 0 | 0.56 |
| HiFi3 / HiFi4 | 357.404 | 19,884.91 | 0.909990 | 621.788 | 11,805.51 | 0.944700 | 886 | 0.55 |
| HiFi2 / HiFi4 | 356.783 | 19,899.11 | 0.909160 | 619.872 | 11,829.56 | 0.943980 | 47,259 | 1.72 |
| LoFi / HiFi4 | 357.794 | 19,741.42 | 0.896640 | 621.732 | 11,843.73 | 0.929780 | 100,000 | 8.62 |
| HiFi4 / HiFi3 | 360.590 | 19,779.47 | 0.910440 | 622.354 | 11,758.87 | 0.944310 | 11,232 | 0.56 |
| HiFi4 / HiFi2 | 360.441 | 19,725.61 | 0.909220 | 623.167 | 11,893.55 | 0.945390 | 11,991 | 0.56 |
| HiFi4 / LoFi | 356.029 | 19,899.57 | 0.908820 | 615.167 | 11,886.75 | 0.944780 | 11,857 | 0.56 |

QPS is the median over the five runs of the end-to-end search interval:
coarse search, grouping, scripts, fine search and output.

The fine-search time does not depend on fidelity: all fine fidelities are
within 0.3% at nprobe 16 and 0.4% at nprobe 32. The scan matmul is a minor
part of the per-pair cost; the BF16 top-k merge and its copies and transposes
dominate. Accuracy does depend on fidelity:

- HiFi3 is almost exact.
- HiFi2 changes 47% of the ranked scores by one BF16 step.
- LoFi changes every score and costs 1.3–1.5 recall points.

The coarse device stage is about 3 ms, so coarse fidelity cannot pay off. It
changes the selected lists, and recall moves by up to ±0.0013. **HiFi4/HiFi4
stays the default.** Its HiFi4 run validated exactly against mem-log: 0 score
mismatches, the same 24,656/25,455 tie IDs, and all IDs rescored. Its timing
matches the ablation's "all but cache-programs" (357.1/621.1 ms).

## Cluster-major optimization ablation — 2026-10-07

Locally rerun on node117 (not user-pasted) with the then-current
`ann_ivf_cluster_major_opt/run_ablation.sh`, output
`results/cluster_major_opt_ablation/20261007_weighted_union/`. That version
had per-optimization switches (`--optimizations`). Afterwards the folder was
pruned to the kept optimizations, without program caching or switches. The exact
ablation-time sources are archived beside the results in
`ablation_sources.tar.gz`. Settings:
weighted-union, nlist 512, k 10, 10,000 queries, five measured runs after the
warmup, batch chunk 32, reader depth 2. All runs were sequential, with no
other work on the host. In the same session, mem-log and the unoptimized
cluster-major reproduced the earlier quiet medians within 0.1%.

Median fine device search (ms; min–max over five runs within 0.2% unless
stated):

| Variant | nprobe 16 | vs cluster-major | nprobe 32 | vs cluster-major |
|---|---:|---:|---:|---:|
| mem-log | 651.815 | 0.607× | 1129.881 | 0.610× |
| cluster-major | 395.483 | 1.000× | 689.135 | 1.000× |
| opt `none` | 395.080 | 1.001× | 688.633 | 1.001× |
| `narrow-mask` only | 388.589 | 1.018× | 676.140 | 1.019× |
| `ids-once` only | 382.757 | 1.033× | 667.030 | 1.033× |
| `batched-matmul` only | 378.398 | 1.045× | 659.824 | 1.044× |
| `topk-partials` only | 395.250 | 1.001× | 689.171 | 1.000× |
| `cache-programs` only | 392.855 | 1.007× | 686.568 | 1.004× |
| **all** | **355.099** | **1.114×** | **618.799** | **1.114×** |
| all but `batched-matmul` | 375.379 | 1.054× | 654.346 | 1.053× |
| all but `ids-once` | 363.799 | 1.087× | 634.167 | 1.087× |
| all but `narrow-mask` | 361.820 | 1.093× | 631.810 | 1.091× |
| all but `cache-programs` | 357.120 (357.0–390.5) | 1.107× | 621.138 | 1.109× |
| all but `topk-partials` | 356.816 | 1.108× | 621.798 | 1.108× |

With `all`, against mem-log:

| Metric | nprobe 16 | nprobe 32 |
|---|---:|---:|
| Fine device search | 1.836× faster | 1.826× faster |
| Median stage sum (ms) | 491.888 vs 812.230 | 833.276 vs 1372.047 |
| Median QPS | 20,315 vs 12,305 | 11,993 vs 7,285 |
| Recall@10 | 0.909980 vs 0.910120 | 0.944660 vs 0.945070 |

All 26 cluster-major variants passed validation against mem-log:

- 0 score mismatches in 100,000 ranked entries per configuration;
- all 200,000 returned IDs rescored within one BF16 ulp;
- the same equal-score ID-difference counts as the unoptimized cluster-major
  (24,656 and 25,455).

**Contributions.** `batched-matmul`, `ids-once` and `narrow-mask` each give a
measurable gain. Each costs 1.9–5.7% when removed from `all`.
`cache-programs` and `topk-partials` are within about 0.6% each: real given
the spread, but marginal. They mainly remove per-search host work and
per-slot transposes, and the merge is small.

**DRAM is not the bottleneck.** This batch-chunk sweep was locally rerun with
`--optimizations all` (the ablation-time build), at nprobe 16 with weighted-union, 10,000 queries and five
runs. Output: `results/cluster_major_opt_chunk_sweep/20261007_nprobe16/`.

| Batch chunk | List page reads | Fine device median (ms) | Recall@10 |
|---:|---:|---:|---:|
| 4 | 821,878 | 361.8 | 0.909980 |
| 8 | 419,825 | 357.3 | 0.909980 |
| 16 | 218,424 | 354.5 | 0.909980 |
| 32 | 118,611 | 355.1 | 0.909980 |

The work (page × batch pairs) is constant. 6.9× more page reads (about 9.9 GB
instead of 1.4 GB) cost 1.9% more fine time. DRAM-throughput techniques
therefore have at most a few percent of headroom here. That includes
DRAM-sharded or bank-local reads, multicasting shared tiles and deeper
outstanding reads. Partitioned relay/multicast readers also gave no gain in the
batch-major setting (see the 2026-10-03 relay comparison).

**Limits.** Work per page × batch pair is unchanged. At about 7.0 µs per
pair per core, the per-pair BF16 top-k sort is still the bottleneck. Larger
gains need fewer pairs or cheaper top-k, not leaner bookkeeping.

## Cluster-major fine search versus mem-log — 2026-10-07

Locally rerun on node117 (not user-pasted) with
`ann_ivf_cluster_major/run_comparison.sh`, output
`results/cluster_major_comparison/20261007_weighted_union_quiet/`. Both
executables ran sequentially on an idle host (load average about 1.4 at the
start). Settings: weighted-union grouping, nlist 512, k 10, 10,000 queries,
five measured runs after the warmup, batch chunk 32, reader depth 2. The fine
device stage is enqueue plus `Finish` of the scan and merge programs for
cluster-major, and of the fine program for mem_log. In both, it includes
per-search program finalisation inside `EnqueueMeshWorkload`.

| nprobe | Metric | mem-log | Cluster-major | Change |
|---|---|---:|---:|---:|
| 16 | Median fine device search | 652.342 (651.0–652.5) | 395.057 (394.9–395.6) | 1.651× faster |
| 16 | Median CPU fine preparation | 143.014 | 123.314 | — |
| 16 | Median stage sum | 815.046 | 536.411 | 1.519× faster |
| 16 | Median QPS | 12,259.69 | 18,626.25 | 1.519× |
| 16 | Recall@10 | 0.910120 | 0.909980 | −0.000140 |
| 16 | Page × batch pairs (work) | 3,232,391 | 3,232,391 | same |
| 16 | List page reads | 3,232,391 | 118,611 | 27.3× fewer |
| 16 | Partial slots | — | 46,688 | — |
| 32 | Median fine device search | 1128.986 (1128.0–1129.6) | 688.722 (688.5–688.8) | 1.639× faster |
| 32 | Median CPU fine preparation | 217.245 | 200.336 | — |
| 32 | Median stage sum | 1368.022 | 907.169 | 1.508× faster |
| 32 | Median QPS | 7,307.14 | 11,017.99 | 1.508× |
| 32 | Recall@10 | 0.945070 | 0.944660 | −0.000410 |
| 32 | Page × batch pairs (work) | 5,661,369 | 5,661,369 | same |
| 32 | List page reads | 5,661,369 | 195,709 | 28.9× fewer |
| 32 | Partial slots | — | 80,454 | — |

The mem-log rows reproduce the 2026-10-06 recorded medians (651.954 and
1129.377 ms). Validation of the final run's 10,000 queries at both nprobe
values found:

- 0 score mismatches in 100,000 ranked entries;
- about 24.7k–25.5k ID differences, all at equal BF16 scores;
- all 200,000 returned IDs (both dumps) within one BF16 ulp of a host rescore
  from the same RNE-BF16 normalized inputs.

The recall differences therefore come only from tie order: ground truth uses
FP32 scores. Runs 1–4 were checked by recall only.

Reads fall about 28×, but work is unchanged by design, so the fine stage
gains about 1.65×. Cluster-major is now bound by per-pair matmul/top-k
compute, about 7.8 µs per page × batch pair per core. Unverified review
proposals for reducing that per-pair cost:

- transpose each page's ID tile once instead of once per batch;
- batch the G matmuls per page;
- narrow the tail-page mask;
- keep partials in top-k layout.

An earlier full run (`20261006_full`, all three groupings) was measured while
builds, tests and review agents loaded the host. Its timings are discarded:
mem-log weighted-union fine search was 977–1079 ms. Its validation results
stand: zero score mismatches and zero rescore failures for none, primary-list
and weighted-union at nprobe 16 and 32. Hardware runs at 320 queries and the
portable checks (52 CTest cases, `tools/test_project_setup.py`) also passed.

## Query grouping in mem-log — 2026-10-06

Evidence: pasted `memcopy_cost_nlist512.csv` and console logs under
`memcopy_results/grouping_20261006T073715Z/{none,weighted-union}/nlist_512/`.
Five completed measured runs per configuration; warmup excluded.

| nprobe | Metric | None | Weighted union | Reduction |
|---|---|---:|---:|---:|
| 16 | Median fine device search | 1477.976 | 651.954 | 55.89% |
| 16 | Median stage sum | 1533.056 | 812.221 | 47.02% |
| 16 | Median CPU fine preparation | 34.779 | 140.730 | — |
| 16 | Union pages per repetition | 7,257,318 | 3,232,391 | 55.46% |
| 16 | Recall@10 | 0.923700 | 0.910120 | — |
| 32 | Median fine device search | 1865.192 | 1129.377 | 39.45% |
| 32 | Median stage sum | 1924.999 | 1363.289 | 29.18% |
| 32 | Median CPU fine preparation | 41.816 | 216.044 | — |
| 32 | Union pages per repetition | 9,520,345 | 5,661,369 | 40.53% |
| 32 | Recall@10 | 0.953300 | 0.945070 | — |

Console medians for grouping itself: 107.965 ms (nprobe 16) and 182.822 ms
(nprobe 32). Reordering/tiling: 2.925 and 2.203 ms. These are included in CPU
fine preparation. Fine search saves 826.022 and 735.814 ms per 10,000 queries;
stage sums save 720.835 and 561.710 ms. Recall falls by 1.358 and 0.823 percentage
points respectively. Report that tradeoff with the speedup.

The `stack_total` CSV is the stage sum. Console QPS uses the end-to-end timer,
which includes additional bookkeeping; do not require QPS to be its exact inverse.

## Partition topology — 2026-10-03

User-pasted topology report and timing report, weighted-union grouping,
row layout, direct reader, FIFO ordering:

| nprobe | Partitions | Median QPS | Median fine search | p95 worker imbalance |
|---:|---:|---:|---:|---:|
| 16 | 1 | 11,454.97 | 700.257 | 2.240 |
| 16 | 4 | 13,115.41 | 591.766 | 1.048 |
| 16 | 8 | 13,406.02 | 582.694 | 1.022 |
| 32 | 1 | 7,012.10 | 1162.008 | 1.101 |
| 32 | 4 | 7,663.87 | 1043.101 | 1.017 |
| 32 | 8 | 7,817.72 | 1024.246 | 1.007 |

Candidate page counts are identical across partition counts within each
nprobe: 3,232,542 and 5,666,850. This isolates topology more closely than a
comparison that also changes grouping. Four to eight partitions adds about
2% QPS in these runs. No hardware result for more than eight is recorded here.
Imbalance includes idle workers within each partition. Repeated deterministic
batches are repeated observations, not independent query samples.

Eight-partition weighted-union timing reports show query-weighted dataset
coverage of 27.721% per query at nprobe 16 and 48.598% at nprobe 32. Each query's
own selected lists cover 3.397% and 6.602%; these are smaller than the full
batch union actually scored. Dataset percentage counts valid vectors, excluding
list padding. Coverage is not an energy measurement or a DRAM bandwidth figure.

## Worker buffering and leader relay — 2026-10-03

Matched prefetch comparison directory:
`similarity_prefetch/20261003T123958Z_pid179117/`. Eight partitions, weighted
union, FIFO; each configuration has five measured runs:

| Reader / worker input depth | Fine search, nprobe 16 | Fine search, nprobe 32 |
|---|---:|---:|
| Direct serial / 2 | 582.722 | 1024.863 |
| Direct / 2 | 582.907 | 1024.768 |
| Direct serial / 4 | 588.891 | 1033.922 |
| Direct / 4 | 594.807 | 1045.962 |
| Direct serial / 8 | 603.647 | 1061.558 |
| Direct / 8 | 630.311 | 1110.949 |

Depth is a CB page-pair capacity, not a partition count. Depth two is the
useful operating point in these measurements; no prefetch speedup is established.
Buffer placement, read-ahead/backpressure, control overhead and NoC effects
are possible explanations. Aggregate timing does not identify their cause.

Unprofiled relay comparison:
`similarity_relay/20261003T151354Z_pid223166/`, worker depth two,
eight partitions, weighted union, FIFO:

| Reader / leader staging depth | QPS, nprobe 16 | Fine search, nprobe 16 | QPS, nprobe 32 | Fine search, nprobe 32 |
|---|---:|---:|---:|---:|
| Direct serial | 13,528.03 | 582.697 | 7,855.38 | 1024.628 |
| Relay / 4 | 12,303.33 | 656.568 | 7,113.76 | 1156.095 |
| Relay / 8 | 12,265.47 | 655.536 | 7,151.75 | 1148.048 |

Relay leaders read candidate/query payloads and forward them over NoC, reducing
payload reader count. Descriptors/results still use DRAM; leaders still share
controllers. The extra forwarding path did not improve throughput here.
Recall was close but not identical (approximately 0.909–0.910 and 0.945–0.946).
Timing/recall alone does not establish score/ID equality.

Capture labels reported by the user: `tracy1.tracy` = serial2/nprobe16;
`tracy2.tracy` = direct8/nprobe16; `tracy3.tracy`/`trace3` denotes the subsequent
relay capture. The third capture's exact settings must be read from its run
metadata before using it. Captures are not bundled. Plots retain leaders,
aggregators and workers; scope overlap must not be added as serial elapsed time.

## Numerical validation and earlier incomplete run

The user supplied validation results for 16, 32, 64 and 320 queries. Scores
and valid-result counts agreed; several paths reported equal-score ID
differences. The validator did not independently rescore the alternative IDs.
Retain this limitation in correctness claims.

The earlier timing sweep `20261003T090344Z_pid179117` completed only two of five
512/32 runs and failed with insufficient valid distinct coarse lists. Its
summary was explicitly allowed incomplete. The later
`20261003T092945Z_pid179117` completed five runs for both configurations.
Use complete runs for the comparison; do not treat an incomplete report as a
full successful validation.

## External runtime build and smoke run — 2026-10-06

User-confirmed external runtime: TT-Metal v0.66.0 at
`e7c251da3bf626fa9a74b790f71612ff1ed573bf`, Clang 20.1.8, recorded tracked
runtime changes. Preparation attached `/home/tenstorrent/tt-metal/build_Release`
and compiled the three ANN targets in `tt-ann-ivf`.

The subsequent mem-log smoke run used weighted union, 512/16, k10, 256 queries,
one run: fine search 25.839 ms, end-to-end 43.784 ms, QPS 5846.838,
Recall@10 0.911719. It establishes build/runtime/kernel-path integration on
that host; it is not the full-dataset throughput result.

## Energy status

See "Energy: optimized cluster-major versus batch-major — 2026-10-07" above for
the first valid joule measurements, taken locally with hwmon board power. No
user-supplied energy captures have been recorded. Captures made with the
`tt-smi` power source must show in-window TT samples before use.

## Repository cleanup

Standalone-repository cleanup did not change any algorithm or device kernel.
