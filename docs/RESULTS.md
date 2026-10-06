# Recorded results and limits

The hardware numbers below were supplied by the user from node117. They were
not rerun on a TT device during repository cleanup. Raw logs/captures remain
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

The energy runner supports independent sequential sessions and complete/search
interval analyses, TT board telemetry and one CPU RAPL reading per package
prefix X. No final joule measurements were supplied here. Energy savings must
be computed from those captures; throughput improvement alone cannot establish
them. Initialization, background CPU work and sampler overhead affect the
reported board + CPU package energy; this is not wall-socket energy.

## Repository cleanup

Standalone-repository cleanup did not change any algorithm or device kernel.
