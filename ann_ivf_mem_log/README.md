# ANN IVF chrono-stage logging

This example is an instrumented copy of `ann_ivf_dram_args`. It uses ordinary host-side `std::chrono` timers; the
default workflow does not enable Tracy or the TT-Metal device profiler.

Every search writes six disjoint stages to the `--mem_log` CSV:

1. `h2d`: blocking query and fine-worker-script H2D copies.
2. `cpu_coarse_config`: query preparation, buffer/program setup, and coarse runtime configuration, excluding H2D.
3. `tt_coarse_search`: coarse workload enqueue plus `distributed::Finish()`.
4. `cpu_fine_prep`: coarse-result handling, optional query grouping/repacking, and fine workload preparation, excluding H2D.
5. `tt_fine_search`: fine workload enqueue plus `distributed::Finish()`.
6. `cpu_output`: final D2H reads, untilization, and result construction.

The stage sum covers the complete timed search path. The TT stages contain the complete device workloads, including
their compute and device memory traffic. They are wall-clock execution stages, not modeled DRAM bandwidth.

## Build

From the `tt-ann-ivf` project root after `source env.sh /path/to/tt-metal`:

```bash
bash build.sh ann_ivf_mem_log
```

## Run the complete nprobe sweep

Activate the Python environment containing NumPy and Matplotlib, then run:

```bash
bash ann_ivf_mem_log/run_memcopy_sweep.sh
```

The defaults are:

- dataset: `glove-100-angular`;
- `nlist=512`;
- `nprobe={1,8,16,32}`;
- 32 queries per measured run;
- 5 measured runs plus the unreported warmup.
- interleaved DRAM fine-result staging.
- two buffered 32-vector blocks per fine-search reader by default.

The outputs are placed directly in `memcopy_results/nlist_512/`:

```text
nprobe_1_mem_log.csv
nprobe_8_mem_log.csv
nprobe_16_mem_log.csv
nprobe_32_mem_log.csv
nprobe_1_memcopy_summary.csv
nprobe_8_memcopy_summary.csv
nprobe_16_memcopy_summary.csv
nprobe_32_memcopy_summary.csv
nprobe_1_console.log
nprobe_8_console.log
nprobe_16_console.log
nprobe_32_console.log
memcopy_cost_nlist512.csv
memcopy_cost_nlist512.png
memcopy_cost_nlist512_without_tt_fine.png
memcopy_cost_nlist512.html
```

The four `nprobe_*_mem_log.csv` files are written directly by the C++ executable through `--mem_log`.
`memcopy_cost_nlist512.csv` contains the median of each stage across the measured runs. The first PNG displays the
complete stacked latency. The `_without_tt_fine.png` detail view omits TT fine search and rescales the remaining
stages; its labels report visible-stage subtotals rather than end-to-end totals. The standalone HTML displays the same
data interactively: click any entry in the legend inside the figure to hide or restore that stage, click a bar segment
to hide it, or use **Show all** and **Hide all** inside the legend. It has no external JavaScript or network
dependency.

Override sweep settings with environment variables:

```bash
RUNS=10 MAX_QUERIES=32 \
bash ann_ivf_mem_log/run_memcopy_sweep.sh
```

Other supported overrides are `DATASET`, `NLIST`, `NPROBES`,
`RESULT_STAGING`, `PREFETCH_READER`, `QUERY_GROUPING`, `GROUPING_WINDOW`,
`GROUPING_LOOKAHEAD`, `PYTHON_BIN`, `BINARY`,
`OUTPUT_ROOT`, and `SWEEP_ID`.
The named outputs are replaced when the sweep is run again.

## Run one workload manually

```bash
./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 8 \
  --max_num_queries 32 \
  --runs 5 \
  --result-staging dram \
  --prefetch-reader 2 \
  --mem_log memcopy_results/nlist_512/nprobe_8_mem_log.csv
```

`--prefetch-reader X` sets the number of 32-vector blocks that each fine-search
worker can hold in its value and ID input circular buffers. It defaults to 2;
any positive integer is accepted if the resulting worker buffers fit the
device's usable L1. The program prints the buffer footprint and maximum depth
for the dataset dimension, and rejects an oversized value before indexing.
For GloVe-100, each additional block costs 12 KiB of L1. Try `4` or `8` with
separate output CSV paths when comparing throughput. The reader still waits for
each block's DRAM reads to finish before issuing the next block, so increasing
this depth raises the amount of completed data it can stage, not the number of
simultaneous DRAM page pairs in flight.

## Group similar queries into batches of 32

Enable the same host grouping heuristic used in
`ann_ivf_similarity_partitioned` with `--query-grouping weighted-union`.
The default is `none` for baseline comparisons; `primary-list` selects a
simpler lexicographic sort of each query's ranked list IDs.

Grouping runs **after device coarse search**, using its BF16 top-32 scores
and actual selected `N_probe` list IDs per valid query. There is no host FP32
coarse recomputation. The heuristic groups queries that add few new list
pages to a batch union, then copies the already normalized BF16 query rows,
tilizes them, and uploads the reordered query buffer for fine search. Worker
script masks are rebuilt in packed batch order. Results are restored to
original query order before recall checks or result files are produced.

The existing whole-list scheduling, 63-worker topology, and direct reader
remain in use. Empty lists and padded query rows do not add scanned pages.
If a proposed ordering increases total union pages, it falls back to the
original order. Coarse output must have 32 valid, distinct IDs and finite
scores per valid query; invalid output stops the run rather than silently
executing a different union. The coarse kernel also includes the initialization
and BF16/Int32 transition corrections used by the partitioned version.

Options:

- `--query-grouping none|primary-list|weighted-union` (default `none`).
- `--grouping-window X`: group within consecutive windows of X queries;
  X must be a multiple of 32, or 0 for the entire repetition (default 0).
- `--grouping-lookahead X`: candidate-search budget in the grouping heuristic
  (default 256; must be positive).

Use more than one batch to measure the reduction in repeated list scans:

```bash
QUERY_GROUPING=weighted-union NPROBES="16 32" MAX_QUERIES=10000 RUNS=5 \
OUTPUT_ROOT=memcopy_results/grouped \
bash ann_ivf_mem_log/run_memcopy_sweep.sh
```

For a sequential baseline/grouped comparison with separate output files:

```bash
for grouping in none weighted-union; do
  QUERY_GROUPING="$grouping" NPROBES="16 32" MAX_QUERIES=10000 RUNS=5 \
    OUTPUT_ROOT="memcopy_results/$grouping" \
    bash ann_ivf_mem_log/run_memcopy_sweep.sh || break
done
```

`[Grouping]` reports union pages before/after the ordering. The memory log
also stores those counts in the `union_pages` record's `notes` field. New
`chrono_detail` rows show `query_grouping`, `query_reorder`, and
`query_batch_masks` inside `cpu_fine_prep`, plus `query_restore` inside
`cpu_output`. A nonidentity ordering adds a blocking `fine_query_tiles` H2D
copy, included in `h2d` and excluded from `cpu_fine_prep` to avoid double
counting. The six-stage plots remain compatible; detail rows are not added
again. `[Run X]` prints QPS and recall per repetition. QPS uses the same
interval as the stages, including grouping, repacking, uploads and result
restoration; timing/diagnostic printout and recall checks occur afterward.

Grouping changes each query's batch candidate union. Consequently recall
can change even though each query keeps its original coarse list selection;
report recall alongside QPS when comparing modes.

Portable checks (no Tenstorrent device required):

```bash
cmake -S ann_ivf_mem_log \
  -B /tmp/ivf-mem-log-host -DIVF_MEM_LOG_HOST_TESTS_ONLY=ON
cmake --build /tmp/ivf-mem-log-host
ctest --test-dir /tmp/ivf-mem-log-host --output-on-failure
```

These check deterministic grouping, page-count conservation, identity fallback,
partial batches, multiword batch masks, exact query-row copying, original-order
scores/IDs and invalid coarse results. The coarse lifecycle model compiles the
actual kernel source and checks its CB/format transitions; it does not emulate
Tensix arithmetic. Device compilation and recall/QPS still require a hardware run.

Create a copy-only H2D/D2H report from that CSV:

```bash
python3 ann_ivf_mem_log/summarize_memcopy_log.py \
  --mem-log memcopy_results/nlist_512/nprobe_8_mem_log.csv \
  --output memcopy_results/nlist_512/nprobe_8_memcopy_summary.csv
```

Create the stacked plot after all four workload CSVs exist:

```bash
python3 ann_ivf_mem_log/plot_memcopy_costs.py \
  --nlist 512 \
  --mem-log 1=memcopy_results/nlist_512/nprobe_1_mem_log.csv \
  --mem-log 8=memcopy_results/nlist_512/nprobe_8_mem_log.csv \
  --mem-log 16=memcopy_results/nlist_512/nprobe_16_mem_log.csv \
  --mem-log 32=memcopy_results/nlist_512/nprobe_32_mem_log.csv \
  --output memcopy_results/nlist_512/memcopy_cost_nlist512.png \
  --without-tt-fine-output \
    memcopy_results/nlist_512/memcopy_cost_nlist512_without_tt_fine.png \
  --html-output memcopy_results/nlist_512/memcopy_cost_nlist512.html
```

The plotter excludes the warmup context and reports medians across `run_1`, `run_2`, and so on.
Both PNGs use milliseconds on the stage axis. Their legends use the timing
symbols from the thesis and list the stages from the top of a bar downward.

## Compare result-staging strategies

Interleaved DRAM is the default, so omitting `--result-staging` is equivalent
to:

```bash
--result-staging dram
```

The centralized NoC/L1 experiment remains reproducible with:

```bash
--result-staging core0-l1
```

Use different output CSVs and otherwise identical arguments when comparing
them. Disable the device profiler for throughput measurements:

```bash
unset TT_METAL_DEVICE_PROFILER

./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular --nlist 512 --nprobe 32 \
  --max_num_queries 10000 --runs 10 \
  --result-staging dram \
  --mem_log memcopy_results/nlist_512/dram_mem_log.csv

./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular --nlist 512 --nprobe 32 \
  --max_num_queries 10000 --runs 10 \
  --result-staging core0-l1 \
  --mem_log memcopy_results/nlist_512/core0_l1_mem_log.csv
```

Each selected inverted list is one indivisible fine-search task. The host
assigns whole lists to workers; no list is split across workers.

## Profile the DRAM-args kernels by 32-query batch

The folder starts from the `ann_ivf_dram_args` data representation but applies
additional fine-search optimizations. Each selected cluster is one task.
Tasks use longest-processing-time-first (LPT) ordering with weight
`cluster_blocks * active_query_batches`. Placement minimizes the worst
per-query-batch worker load first and aggregate worker load second.

Within a worker, the query is read from DRAM only once per logical batch and
reused across all active cluster tasks. Dataset and vector-index pages are
queued together and completed with one NoC read barrier.

Worker top-32 results use interleaved DRAM staging by default. This restored
the faster measured path: pages are distributed across DRAM banks instead of
forming a 63-worker incast into core `(0,0)`. Select it explicitly with
`--result-staging dram`.

The experimental direct path remains available with
`--result-staging core0-l1`. It allocates
`63 * (2 KiB + 4 KiB) = 378 KiB` on core `(0,0)` and reuses those slots for
every logical batch. The ready and done semaphores make the path correct, but
the centralized destination measured slower on this workload. It is retained
only to make controlled A/B comparisons reproducible.

The fine reader uses NoC 1 and the writer uses NoC 0. The reader queues the
dataset-vector and vector-ID page reads together before one read barrier.

The fine-search kernels currently emit these `DeviceZoneScopedN` intervals:

- `Query`: the NCRISC query transfer and the corresponding TRISC input wait;
- `Batch`: one dataset/index DRAM read through its barrier and one matching
  compute/local-sort block;
- `Send Res`: one worker's value/index result write to DRAM;
- `Partial Res`: one worker result read by the aggregator NCRISC;
- `Compute Res`: the aggregator TRISCs merging all worker partial results.

The earlier timestamped-data markers remain removed. A commented custom-zone
template remains at the start of `kernels/dataflow/reader_fine.cpp`.

Each RISC profiler buffer can retain at most 125 complete optional scope pairs
per kernel invocation. `Batch` is emitted once per candidate block and can
exceed that limit even for one 32-query batch. Missing tail scopes therefore
mean the device profiler dropped markers, rather than that the kernel stopped
processing candidates.

Use a short, dedicated profiling run:

```bash
TT_METAL_DEVICE_PROFILER=1 \
./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 32 \
  --max_num_queries 32 \
  --runs 1 \
  --result-staging dram \
  --mem_log memcopy_results/nlist_512/nprobe_32_mem_log.csv
```

For the first remote validation, compare the optimized executable with a
pre-change result at exactly the same dataset, `nlist`, `nprobe`, query count,
and clock settings. Check all of the following:

1. Recall and returned indices remain unchanged within the expected BF16
   behavior.
2. The host reports `Fine-result staging: interleaved DRAM (default)`.
3. `tt_fine_search` decreases; this is the primary success metric.
4. The maximum worker duration and the spread between worker durations
   decrease; this tests batch-aware LPT assignment.

Older profiler captures may contain these memory-zone names for both
result-staging implementations:

```text
MemLog_Fine_DatasetAndIndices_DRAM_to_L1
MemLog_Fine_Worker_L1_to_DRAM
MemLog_FinalSort_DRAM_to_L1
MemLog_Fine_Worker_L1_to_Core0_L1
MemLog_FinalSort_Core0_L1_to_CB
```

The final two appear only for `core0-l1` and are on-chip NoC/L1 movements.

Keep device-profiler runs short: each RISC has finite marker capacity. This
limit does not affect ordinary chrono-only benchmark runs with
`TT_METAL_DEVICE_PROFILER` unset.

The device-profiler CSV is written to
`generated/profiler/.logs/profile_log_device.csv`. The existing
`summarize_device_mem_log.py`, `plot_device_kernel_timeline.py`,
`plot_fine_reader_deep_timeline.py`, and
`plot_fine_low_level_timeline.py` can still read older captures.

`plot_fine_pipeline_timeline.py` reads the current short scoped-zone names:
`Query`, `Cluster`, `Send Res`, `Partial Res`, `Compute Res`, and `Res`.
It uses no `DeviceTimestampedData` markers. Batch membership is inferred from
the final aggregator `Res` scopes, each worker's `Send Res` scopes, the fixed
number of compute `Cluster` scopes per worker batch, and reader `Query`
delimiters.

Capture 64 queries to expose two measured batches:

```bash
ivf_pipeline_dir=/tmp/ivf-pipeline-64
mkdir -p "$ivf_pipeline_dir"

TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_DIR="$ivf_pipeline_dir" \
./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 8 \
  --max_num_queries 64 \
  --runs 1 \
  --result-staging dram \
  --mem_log "$ivf_pipeline_dir/host.csv"
```

Plot the aggregator `(1,1)` and the next three profiler cores. The final two
batch completions exclude the preceding one-batch warmup:

```bash
python3 ann_ivf_mem_log/plot_fine_pipeline_timeline.py \
  --profile-log "$ivf_pipeline_dir/.logs/profile_log_device.csv" \
  --aggregator-core 1,1 \
  --worker-count 3 \
  --last-batches 2 \
  --cores-per-page 4 \
  --unit us \
  --view trace \
  --label-mode compact \
  --output "$ivf_pipeline_dir/fine_pipeline.png"
```

For a clearer steady-state view, capture 320 queries and change
`--last-batches` to `10`. The light timeline and companion CSV contain only
the original scoped phases. BRISC writer scopes are retained in the CSV but
omitted from the full-width figure because their durations are sub-pixel at
this scale. Inactive compute slots are also retained in the CSV and omitted
from the figure; displayed reader and compute clusters therefore share the
same active-list index. The vertical `B<n> done` lines preserve the
aggregator's result-write completions.
Reader work is purple, query reads are lighter purple, and compute work is
yellow in both the batch timeline and the one-cluster view. Cluster labels
shorter than 200 µs are hidden in the batch timeline to avoid collisions.
Use repeated `--core X,Y` arguments to select particular workers. When more
cores are selected than fit on a page, the aggregator is repeated on every
page and the output receives `_page_01`, `_page_02`, and similar suffixes.
The selected intervals and inferred batch assignments are also written to a
CSV beside the PNG.

For per-batch profiling, prefer a deliberately short workload such as
`--max_num_queries 320` (ten batches). A RISC can retain only 125 device
scopes, so a 10,000-query run may fill its profiler buffer before the true
last batches execute.

### One-cluster reader/compute bubble view

`plot_fine_cluster_bubbles.py` zooms into one active cluster task on one
worker. The kernels use direct `DeviceZoneScopedN` scopes only:
`FineBubble.Reader.Query`, `FineBubble.Reader.Cluster`,
`FineBubble.Reader.Fetch`, `FineBubble.Compute.QueryWait`,
`FineBubble.Compute.Cluster`, and `FineBubble.Compute.Work`. Reader CB reserve
and compute input wait remain outside the colored per-block scopes, so their
cost appears as a white pipeline gap.

Use one query batch and `nprobe=1` for this diagnostic. Plot the first 16
blocks of one whole cluster. Later blocks may be missing from the profiler log
if its per-RISC scope capacity is exhausted. The plotter selects the last
invocation and one zero-based active cluster occurrence on the chosen worker.

```bash
ivf_bubble_dir=$(mktemp -d /tmp/ivf-bubble.XXXXXX)

TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_DIR="$ivf_bubble_dir" \
./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 1 \
  --k 10 \
  --max_num_queries 32 \
  --runs 1 \
  --result-staging dram \
  --mem_log "$ivf_bubble_dir/host.csv"
```

Plot the first worker for which the selected batch has an active cluster:

```bash
python3 ann_ivf_mem_log/plot_fine_cluster_bubbles.py \
  --profile-log "$ivf_bubble_dir/.logs/profile_log_device.csv" \
  --invocation last \
  --cluster-index 0 \
  --block-start 0 \
  --block-count 16 \
  --unit us \
  --output "$ivf_bubble_dir/fine_cluster_bubbles.png"
```

Pass `--core 1,2` to select a particular profiler worker coordinate. The light
purple `Q` bar is the query read. Each dark purple `P<n>` bar is one complete
reader page fetch, including both vector and ID DRAM reads and their shared
barrier; the adjacent `Q` and `P0` bars are separate scopes. Every page fetch
is labeled above its bar, including short fetches. Yellow bars are compute
work after the input CB is ready. White gaps on NCRISC include CB reserve
backpressure; white gaps on the TRISCs include input waiting and setup. The
script also writes exact intervals and durations to a CSV beside the PNG.

### Zoom into the fine-reader loop

`plot_fine_reader_deep_timeline.py` creates a two-level view of one 32-query
batch. Its upper panel shows the complete reader/compute/writer batch zones on
the aggregation core and the selected workers. The kernel names use logical
core `(0,0)`, while the profiler CSV normally reports that core as `(1,1)` on
this Wormhole mapping. Its lower panel extracts the workers' existing
`MemLog_Fine_Cluster_slice` scopes and shows the
enclosing task scopes plus the one-time query read. Each slice is associated
with its reader core, measured invocation, worker-script task ordinal,
batch-wide slice ordinal, and slice ordinal within the task.

Worker numbering begins at 1 and excludes the detected aggregation coordinate.
Without explicit `--core` arguments, the script chooses workers 1, 2, and 3
using the same Y-then-X core order as the host scheduler. It detects the
aggregator from `Core_0_0_Final_Sort_*` zones and falls back to profiler core
`(1,1)`. Use `--aggregator-core 1,1` to make that mapping explicit.

Use a short profiler run and then plot the first 32 recorded slices from batch
zero of the last captured invocation:

```bash
TT_METAL_DEVICE_PROFILER=1 \
./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 32 \
  --max_num_queries 32 \
  --runs 1 \
  --result-staging dram \
  --mem_log memcopy_results/nlist_512/deep_profile_host.csv

python3 ann_ivf_mem_log/plot_fine_reader_deep_timeline.py \
  --batch-id 0 \
  --invocation last \
  --aggregator-core 1,1 \
  --worker-count 3 \
  --slice-start 0 \
  --slice-count 32 \
  --cores-per-page 3 \
  --unit us \
  --output memcopy_results/nlist_512/fine_reader_deep.png
```

To isolate exactly one recorded cluster-page transfer on profiler worker core
`(1,2)`, while retaining profiler core `(1,1)` in the upper aggregation panel,
use:

```bash
python3 ann_ivf_mem_log/plot_fine_reader_deep_timeline.py \
  --batch-id 0 \
  --aggregator-core 1,1 \
  --core 1,2 \
  --slice-start 7 \
  --slice-count 1 \
  --unit us \
  --output memcopy_results/nlist_512/fine_reader_slice_7.png
```

The adjacent CSV distinguishes `full_batch` and `deep_zoom` rows. It contains
the exact aggregation and worker duration in microseconds, absolute profiler
cycles, cycles relative to the first selected slice, and all task/slice
ordinals. A task ordinal is its occurrence in that worker's packed script; the
reader currently does not emit the actual IVF cluster ID. The
`MemLog_Fine_Cluster_slice` scope starts after both circular-buffer reserves
and includes pointer lookup, the dataset and index NoC read issue, the shared
read barrier, and both circular-buffer pushes.

One query batch can contain more slice scopes than the profiler buffer can
hold. In that case the script plots the complete zone pairs that were captured
and reports unmatched markers. Use `--slice-count` to keep the image readable;
it filters the CSV after capture and does not increase the device buffer.

### Legacy low-level timeline utility

`plot_fine_low_level_timeline.py` remains available for profiler CSV files
captured before the low-level markers were removed. Current kernels do not
emit the marker intervals that this script consumes. Add a custom profiling
zone at the commented template when collecting a new trace.

### Keep the process alive until Tracy connects

Tracy already provides the `TRACY_NO_EXIT=1` runtime variable. It keeps a
short-lived program alive at shutdown until a Tracy client connects and
receives the buffered capture:

```bash
TRACY_NO_EXIT=1 \
TRACY_PORT=8086 \
TT_METAL_DEVICE_PROFILER=1 \
./build_Release/bin/ann_ivf_mem_log \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 32 \
  --max_num_queries 32 \
  --runs 1 \
  --mem_log memcopy_results/nlist_512/nprobe_32_mem_log.csv
```

Open the Tracy GUI and connect to the Tenstorrent node on the same port.
`TRACY_NO_EXIT` does not pause the benchmark before its workload starts; it
prevents process shutdown while Tracy is still waiting to connect. This is
normally the desired behavior for capturing a short executable without
adding application-specific synchronization code.

## Project records

See [memory_layout.md](memory_layout.md) for memory organization and
[recorded results](../docs/RESULTS.md) for the complete grouping comparison.

## File locations

Dataset inputs are in `data/datasets/` and centroids in `data/centroids/`.
`ANN_DATA_DIR` overrides the data root. Defaults write under `results/`;
`ANN_RESULTS_DIR` overrides that root and explicit output arguments keep their
exact paths. See [input layout](../data/README.md) and
[measurement layout](../results/README.md).

Dataset converters are shared in `tools/export_hdf5.py` and
`tools/convert_centroids.py` at the project root.
