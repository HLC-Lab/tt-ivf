# Similarity grouping and independent IVF partitions

This is a separate experimental example, `ann_ivf_similarity_partitioned`.
It implements the proposed query grouping and reader leaders. Its fine search
uses BF16 scores, full int32 IDs, `topk_local_sort`, and DRAM result staging.
Whole lists remain indivisible tasks; there is no cluster chunking.

## Workflow

1. Normalize queries once and run the device coarse search in original order.
2. Decode the device's actual BF16 top-32 results, keeping the requested
   distinct `N_probe` list IDs for every valid query. No CPU centroid search
   substitutes for these selections.
3. Group queries into batches of at most 32 by overlap of selected lists,
   weighted by list pages. Pack those queries into a new fine-query DRAM buffer.
   A bounded inverted-list lookup makes grouping deterministic. Fall back to
   original order if the heuristic increases the total union page count.
4. Assign each batch to exactly one partition, using its predicted whole-list
   LPT makespan and the partition's accumulated load. Within that partition,
   assign entire lists to its least-loaded workers, in descending page order.
5. The leader fetches its query batch and distributes it to its own workers.
   Multicast is used only when the physical workers form an exact rectangle,
   the leader is outside it, and their reserved query CB addresses agree.
   Otherwise distribution uses unicast. NoC 1 bounds are reversed correctly.
6. In `relay` mode, the leader issues bounded asynchronous vector/ID reads
   into its own L1 slots, then forwards each pair into CB space reserved by
   that worker. Slot transaction IDs and generation credits preserve ordering.
   In `direct` mode, each worker reads its assigned candidates from DRAM,
   with multiple outstanding vector/ID page pairs. `direct-serial` preserves
   the previous per-page full read barrier as a comparison baseline.
7. Workers compute similarities and update their local top-32 with
   `topk_local_sort`. They write paired partial values and IDs to dedicated
   DRAM slots and publish completion only after both writes finish.
8. The partition aggregator gathers partials as each worker finishes and
   runs the same local sort reduction. It writes final results and acknowledges
   the batch. Results are scattered back to the original query order on host.

Each leader owns its own batch queue. There are no interleader barriers or
shared bank-turn counters. Each partition admits one active batch and prefetches
its next query while the current batch is being reduced; it does not prefetch
the next batch's candidates. Host synchronization finishes the complete
workload before readback.

## Build and first run on Wormhole

Run from the repository root on your remote machine, using its existing
configured Release build:

```bash
cmake --build build_Release --target ann_ivf_similarity_partitioned ann_ivf_similarity_partitioned_host_test -j8
./build_Release/bin/ann_ivf_similarity_partitioned \
  --dataset glove-100-angular --nlist 512 --nprobe 8 --k 10 \
  --max_num_queries 32 --runs 2 \
  --partitions 8 --partition-layout rows \
  --query-grouping weighted-union \
  --candidate-reader relay --bank-schedule staggered \
  --leader-prefetch-pages 8 --worker-input-pages 2 \
  --aggregation leader --candidate-mask auto \
  --mem_log /tmp/ivf_similarity_host.csv \
  --results-csv /tmp/ivf_similarity_results.csv \
  --diagnostics /tmp/ivf_similarity_diagnostics
```

The executable expects the same files as `ann_ivf_mem_log` in the working
directory: `data/datasets/glove-100-angular_train.bin`, `..._queries.bin`,
`..._neighbors.bin`, and `data/centroids/centroids-glove-100-angular-512.bin`.
The copied conversion scripts are available for preparing these files.
Use `build` in place of `build_Release` if that is your configured build.

Before a full benchmark, run the sequential validation script:

```bash
bash ann_ivf_similarity_partitioned/run_validation.sh
```

It runs 16, 32, 33, 64 and 320 queries twice per path, checks identical coarse selections
and grouping across repetitions and transport modes, and compares the serial
direct reader against direct depths 2/4/8, relay, additive mask, depth-one,
and separate-aggregator result dumps. Score and valid-result count
differences fail. Equal-score ID differences are reported because BF16 ties
can depend on reduction order; inspect those differences and recall. These
comparisons do not independently rescore alternative IDs. You can require
identical IDs by running `compare_results.py` without `--allow-ties`.

Use `IVF_EXECUTABLE=/absolute/path/to/binary` to select another build. The
validation script requires GNU `timeout`, writes each process's logs separately,
and waits for it to exit before starting another. It requires a fresh output
directory and checks the exact query/rank count, sorted scores and unique IDs.
A timeout sends TERM, then kills the process group after 30 seconds if needed;
the script stops on failure. It does not reset the device after a timeout.

For the full six-path ablation:

```bash
bash ann_ivf_similarity_partitioned/run_ablation.sh
```

Defaults are GloVe, `(512,8)`, 10,000 queries and three measured repetitions,
with warmup excluded. Override with `NLIST`, `NPROBE`, `QUERIES` and `RUNS`.
Add `--matrix` for `{512,1024,2048} × {1,2,4,8,16,32}` plus `(2048,12)`.
Add `--sweep-topology` for 4/6/8/12 partitions in rows and columns. All benchmark
processes execute sequentially. The scripts unset profiler, DPRINT, watcher and
Tracy client wait variables for measurements.

To compare only grouped queries with direct readers and global, four-partition
and eight-partition layouts at 512/16 and 512/32:

```bash
ivf_topology_dir="$PWD/similarity_topology/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
for probe in 16 32; do
  NLIST=512 NPROBE="$probe" QUERIES=10000 RUNS=5 \
    bash ann_ivf_similarity_partitioned/run_ablation.sh \
      --grouped-topology "$ivf_topology_dir/nprobe_$probe" || break
done
```

This runs six benchmark processes sequentially, each with five measured
searches and a separate warmup. Grouping, direct readers, FIFO bank scheduling,
leader aggregation, tail masking and input depth are held constant. Each
process saves `command.txt`, `results.csv`, `timings.csv`, `host.csv` and scan
diagnostics; each probe directory receives an `analysis/report.md` for QPS,
recall and worker loads and `analysis/timing_report.md` for stage timings and
dataset coverage. The focused option cannot be combined with
`--sweep-topology`, which uses relay readers.

Analysis produces a CSV, Markdown report and plots:

```bash
python3 ann_ivf_similarity_partitioned/analyze_grouping.py \
  /path/to/similarity_ablation/run_directory
```

Use `--no-plots` if Matplotlib is unavailable. Inspect median QPS and recall
alongside union pages and p95/max worker loads. Grouping changes batch candidate
unions, so it can change recall even though every query retains its own selected
lists. Fewer pages alone do not establish a speedup.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--query-grouping` | `weighted-union` | `none`, `primary-list`, or page-weighted union heuristic |
| `--grouping-window` | `0` | Whole input; otherwise consecutive windows divisible by 32 |
| `--grouping-lookahead` | `256` | Bound on posting/candidate visits per grouping decision |
| `--organization` | `partitioned` | `global` uses one leader and all remaining workers |
| `--partitions` | `8` | Independent groups; 4/6/8/12 are supported on a 64-core grid |
| `--partition-layout` | `rows` | `rows`, `columns`, or alternating-row `compact` traversal |
| `--candidate-reader` | `relay` | Leader forwards, worker `direct` prefetches, or `direct-serial` uses the previous full barrier |
| `--bank-schedule` | `staggered` | Independent rotating controller/bank preference; `fifo` is round-robin among worker streams |
| `--leader-prefetch-pages` | `8` | Relay staging entries, 1–15; each entry holds a vector/ID page pair |
| `--worker-input-pages` | `2` | Worker candidate CB capacity; direct outstanding read window is `min(depth,15)`; alias `--prefetch-reader` |
| `--query-broadcast` | `auto` | Verified rectangle multicast with unicast fallback, or always `unicast` |
| `--aggregation` | `leader` | Leader's compute/BRISC, or one `separate` aggregator per partition |
| `--candidate-mask` | `auto` | Metadata tail replacement, `additive` tail tile, or checked `off` |
| `--profile-batch` | disabled | Original packed global batch ID for sampled fine-work scopes |
| `--diagnostics` | disabled | Untimed union, worker load, permutation and list-size CSVs |
| `--results-output` | disabled | Last measured run's `query,rank,index,score` dump in original order |

Direct reads require FIFO; selecting direct automatically changes the default
bank policy to FIFO, and an explicitly requested direct/staggered combination
is rejected. Layouts are core traversals divided into balanced groups; 6 or 12
partitions need not correspond to complete rows or columns. On 64 cores, eight
partitions give 56 workers with leader aggregation, or 48 workers with separate
aggregation. `global` is a control using this new protocol, not an exact copy
of the old `ann_ivf_mem_log` scheduler.

Wormhole exposes twelve logical DRAM views over six physical controller groups.
The program reads and prints the actual allocator bank-to-view-to-controller
mapping at runtime. Staggering ranks the next pages available in each worker
stream; it cannot reserve a bank, ensure collision-free access, or enforce a
fixed phase difference between leaders. Candidate and ID buffers remain
interleaved.

## Tail safety and L1 accounting

The uploaded last page of a list contains padded vectors and `0xFFFFFFFF` IDs.
Valid vector ID zero and zero-valued vectors are not treated as padding.

- `auto`: only the last partial page receives replacement scores of -1000 and
  invalid IDs in its padded columns. The valid count is read through the
  compute API's mailbox synchronization. Only UNPACK modifies that tail;
  only PACK initializes champion tiles.
- `additive`: the receiver produces a BF16 tile containing zero on valid
  columns and -1000 on padded columns, only for the partial last page. Compute
  adds that tile to the score tile before local sort. No mask is fetched or
  consumed for full pages. IDs remain the uploaded full int32 tile.
- `off`: the host rejects an executed list with a partial last page. Empty
  lists are skipped. This mode is safe only for aligned selected lists.

Padding in the last query batch never contributes coarse selections or union
tasks, and padded query output rows are discarded. The -1000 sentinel is below
valid normalized cosine scores. Scores and IDs stay paired through every sort.

Let `Q = ceil(D/32) * 2048`, `S = Q + 4096`, `W` be worker input depth,
`R` relay depth, and `M` the largest number of scheduled lists in any batch.

| Role | Per-core accounted bytes |
|---|---|
| Worker, auto/off | `16384 + 2*Q + W*S + 32768 + 128` |
| Worker, additive | Previous row plus `W*2048` |
| Leader raw staging/control | `16384 + 2*Q + (M+1)*32 + R*S` |
| Aggregator CBs | `36992` |
| Separate aggregator | `16384 + 36992` |
| Coarse CBs | `4*Q + 36864` |

Leader aggregation adds the aggregator CBs to leader raw storage. Direct mode
uses `R=0`. For GloVe-100, worker depth two occupies **88.125 KiB** for auto
or **92.125 KiB** for additive masking. At relay depth eight and at most 512
lists, a leader that also aggregates uses at most **180.15625 KiB**.

All role working sets are checked against a conservative 512 KiB cap and the
device's allocator L1 base/size before fine launch. Buffer allocation remains
the final capacity check. Control/cache/mailbox regions do not overlap; reader
and writer script caches are separate. The reported sizes cover payloads and
explicit raw buffers, not firmware or every program metadata allocation.
Relay depth is capped at fifteen because transaction ID zero is reserved for
ordinary query/script reads. Prefetching stages real pending reads; a depth
parameter alone does not guarantee useful overlap or higher throughput.

## Direct-reader prefetch and comparison

The `direct` reader issues up to `min(worker-input-pages,15)` page pairs before
waiting for the oldest pair. Both the vector page and its full int32 ID tile
use the same transaction ID. A per-ID completion barrier retires that pair
while other IDs can remain in flight. Both CBs are published in page order;
the tag is reused only after retirement. Large vector pages are split into
NoC packets with the same tag.

Reads target free slots in the existing worker CB rings. Reservations include
every unpublished pending page, and vector/ID free space is checked separately
because compute releases vectors after matmul but retains IDs through top-k.
Future slots wrap within each allocation; every push still advances exactly
one page pair. Pending reads drain at each list boundary before another script
record can trigger a normal read. Transaction ID zero is restored after each
issue. Tail handling is unchanged: only a partial last page needs a mask.

This adds no staging allocation or forwarding copy. At D=100, auto masking
uses 88.125 KiB/core at depth 2, 112.125 KiB at depth 4 and 160.125 KiB at depth
8, including the existing worker control and other CBs. The host retains its
allocator and 512 KiB capacity checks. Depths above 15 provide more CB storage
but do not increase the outstanding DMA window. `--leader-prefetch-pages`
controls relay staging and does not change this direct-reader window.

Use `direct-serial` with the **same input depth** to isolate the effect of the
new DMA scheduling. It keeps the old full barrier after every vector/ID pair;
using depth one instead would also change compute/read overlap and capacity.
The existing CSV `reader` and `worker_input` columns distinguish these runs.
The executable prints the configured CB capacity and effective direct window.

After rebuilding and running `run_validation.sh`, compare serial depth 2 with
direct depths 2/4/8 at 512/16 and 512/32. This loop runs every process
sequentially, with five measured 10,000-query searches per configuration:

```bash
ivf_prefetch_dir="$PWD/results/similarity_prefetch/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
for mode in serial2 direct2 direct4 direct8; do
  reader=direct
  depth=${mode#direct}
  if [[ "$mode" == serial2 ]]; then reader=direct-serial; depth=2; fi
  CANDIDATE_READER="$reader" WORKER_INPUT_PAGES="$depth" RUNS=5 QUERIES=10000 \
    bash ann_ivf_similarity_partitioned/run_timing_sweep.sh \
      "$ivf_prefetch_dir/$mode" || break
done
```

Compare `analysis/timing_report.md` in each mode directory, including TT fine
time, end-to-end QPS and recall. Grouping, eight row partitions, FIFO policy,
query broadcast and aggregation are held constant. Warmup is separate and
profiling is disabled. The supplied remote sweep shows depth two matching
serial depth two, depth four adding about 2% to fine-search time and depth eight
adding about 8%. Retain depth two for this workload; medians are in
[docs/RESULTS.md](../docs/RESULTS.md). These timings do not establish the hardware cause of the regression.
The follow-up serial depths four/eight also slow fine search by about 1%/3.6%
relative to serial depth two. At the same capacity, direct depths four/eight
are another 1–1.2%/4.4–4.7% slower than serial. Larger queues change read-ahead
as well as memory allocation; these results do not isolate L1 placement from
NoC/DRAM scheduling effects.

## Timings, profiling and verification status

Measured pipeline time includes normalization, coarse search/readback,
grouping, repacking, scheduling, transfers, program preparation, fine execution,
readback and inverse permutation. Index creation/upload precede the search
benchmark. Warmup is separate unless `--skip-warmup` is used. Recall checking,
result dumps and diagnostic file writes occur after each measured pipeline.
The six existing `chrono_stage` records remain the aggregate stages. New
`chrono_detail` records divide their parent stages; their `included_in` notes
identify that parent. Each run prints these details in milliseconds:

- Query normalization, coarse buffer/tiling/program setup and coarse runtime arguments.
- TT coarse execution, then result readback and decoding of the device's selected lists.
- Query similarity grouping, followed by query reordering/copying/tiling.
- Partition/worker scheduling, descriptor packing and remaining fine buffer/program setup.
- Original/reordered query uploads, descriptor upload and L1 control initialization.
- TT fine execution, final readback and output restoration to original query order.

TT coarse/fine times span enqueue through the blocking `Finish()`; a cache miss
can therefore include compilation. Fine setup/misc is the remaining host
preparation, including allocations and existing bookkeeping. L1 initialization
also includes creation of the zero-filled host payload. Detail values are
included in the parent stage and must not be added a second time.

`--timings-csv PATH` writes one row per measured repetition, with durations in
microseconds. Its default is `<results-csv stem>_timings.csv`. Warmup appears in
the console and memory log with context `warmup`, but is excluded from this CSV.
Timing CSV writes, detailed printing, scan statistics and diagnostic file writes
occur after the pipeline timer stops. A few additional host clock reads occur
inside the timer and are included in measured wall time.

`--diagnostics DIRECTORY` also writes `batches_run_N.csv`, one row per internal
batch, including `union_vectors`, `database_vectors`, `scanned_fraction`,
`scanned_percent` and `own_scanned_percent`. Coverage is computed as
`100 * union_vectors / database_vectors`, using the actual device-selected lists
and the scheduled union. It counts real vectors, excluding list and query
padding. Every valid query in that batch scores that union. `own_scanned_percent`
uses the mean vector count of each query's own selected lists; this is a scan
comparison, not an independent Faiss run. The console prints mean, median, p95
and maximum batch coverage for each invocation.

For the grouped direct-reader path with 8 row partitions, measure 512/16 and
512/32 sequentially, with 10,000 queries and five measured runs each:

```bash
cmake --build build_Release --target ann_ivf_similarity_partitioned -j8
RUNS=5 QUERIES=10000 \
  bash ann_ivf_similarity_partitioned/run_timing_sweep.sh
```

This retains one separate 32-query warmup per process. Set `IVF_EXECUTABLE` to
your `build_Release/bin/ann_ivf_similarity_partitioned` path if needed.
The runner creates a fresh `similarity_timings/<timestamp>_pid<PID>/` directory
and stops on failure before starting the next configuration. Per-configuration
files are `benchmark.log`, `timings.csv`, `results.csv`, `host.csv` and the batch
diagnostics. Recall is still checked after each timed search.

### Leader fetch and distribution comparison

`relay` is the path where each partition's leader reads the query, cluster
vector pages and full int32 ID tiles, then sends them to its workers over NoC.
Workers reserve their input slots and advertise the destination addresses;
the leader acknowledges only after both payload writes finish. Leader staging
slots have separate read transaction IDs and cannot be reused before forwarding
finishes. Workers consume each list's pages in order. Partitions have independent
queues and completion signals; leaders do not synchronize with other leaders.

This reduces the number of query/vector/ID DRAM readers to the leaders. It does
not eliminate all DRAM traffic or contention: workers still read descriptors and
write partial results, aggregators read those results, and leaders share DRAM
controllers. Forwarding also adds NoC traffic and can bottleneck at a leader.
Compare complete pipeline time and recall with the direct-reader baseline.

The timing and Tracy runners accept `CANDIDATE_READER=relay`,
`LEADER_PREFETCH_PAGES=1..15`, `WORKER_INPUT_PAGES` and
`BANK_SCHEDULE=fifo|staggered`. Leader depth counts staged vector/ID **page
pairs across the partition**, not pages per worker. For GloVe-100, each pair
occupies 12 KiB, so four/eight leader slots use 48/96 KiB for payload staging,
in addition to queries, descriptors, control and aggregation buffers. The host
checks the full L1 allocation. `staggered` is an independent local preference,
not synchronized bank ownership; first compare FIFO with FIFO.

Run the direct-serial/depth-two baseline, relay with four leader slots, and relay
with eight leader slots at both `(512,16)` and `(512,32)`:

```bash
cmake --build build_Release --target ann_ivf_similarity_partitioned -j8
RUNS=5 QUERIES=10000 \
  bash ann_ivf_similarity_partitioned/run_relay_comparison.sh
```

All six processes run sequentially, with grouping, topology, aggregation,
masking and two worker input slots fixed. Results go to a fresh
`similarity_relay/<timestamp>_pid<PID>/` directory; the combined
`analysis/timing_report.md` distinguishes reader mode and both depths.
For a standalone relay sweep:

```bash
CANDIDATE_READER=relay LEADER_PREFETCH_PAGES=8 WORKER_INPUT_PAGES=2 \
  BANK_SCHEDULE=fifo RUNS=5 QUERIES=10000 \
  bash ann_ivf_similarity_partitioned/run_timing_sweep.sh
```

The runner automatically writes `analysis/timing_report.md`,
`analysis/timing_summary.csv`, `analysis/batch_scans.csv` and
`analysis/scan_summary.csv`. The report shows timing medians/minima/maxima and
scan means/medians/p95/maxima. The combined batch CSV retains every measured
batch and run; 10,000 queries yield 312 full batches and one 16-query batch.
Coverage means are reported both per batch and weighted by valid queries.
Individual stage totals sum to each run's end-to-end time; independently
computed column medians need not sum exactly.

To regenerate the summary without rerunning the hardware:

```bash
python3 ann_ivf_similarity_partitioned/summarize_timings.py \
  /path/to/similarity_timings/run_directory --expected-runs 5
```

The summarizer uses only the Python standard library. Existing ablation runs
captured with the new executable also get `results_timings.csv` automatically.
Older logs cannot recover the new component timings or coverage columns.

If a later repetition fails, the runner preserves the raw files and produces
a report of completed repetitions with an explicit incomplete label. To
summarize the existing partial run manually after updating the script:

```bash
python3 ann_ivf_similarity_partitioned/summarize_timings.py \
  /path/to/partial_run --expected-runs 5 --allow-incomplete
```

The coarse kernel uses one full matmul initialization per kernel invocation,
short matmul initialization between scoring and sorting, and explicit BF16 /
Int32 format switches. The host validates all 32 output list IDs for every
valid query, even when `N_probe` is smaller. On failure it reports the query,
batch, logical core, duplicate/invalid counts and raw IDs. With `--diagnostics`,
the failing row is saved as `coarse_failure_run_N.csv`; this failed repetition
does not contribute QPS or timing rows. The subsequent remote sweep completed
five 10,000-query measured runs each for 512/16 and 512/32 without a coarse
validation failure. This verifies that sweep; it does not establish the
original failure's cause or exclude recurrence.

## Device scopes in Tracy

### Analyse saved captures locally

`analyze_device_traces.py` reads saved `.tracy` captures and creates
`scope_summary.csv`, `pages.csv`, `partition_mapping.csv`,
`partition_completion.csv`, `partition_scopes.csv`, `report.md`, and white-background plots:

- `page_waits.png`: reader residual wait, unpacker input wait, and packer top-k distributions;
- `partition_completion.png`: last fine-kernel elapsed duration per partition;
- `worker_pages.png`: the same worker's first 16 page pairs, showing reader and all three TRISCs.
- Relay captures also produce `leader_pages.csv` and `leader_relay_pages.png`:
  leader residual DRAM waits, vector/ID forwarding time and issue-to-forward
  completion for the first 16 page pairs **sent to worker 0 only**.
- `LABEL_partition_N_overview.png`: the leader/aggregator and every worker in the
  selected partition, including all five RISC rows, over the selected batch;
- `LABEL_partition_N_pages.png`: the same rows zoomed into query distribution
  and the first 16 page pairs. Both partition views also have SVG files for zooming.

Tracy's standard `tracy-csvexport` exports host zones only. The adjacent
`export_tt_device_zones.cpp` uses Tracy's file reader to extract device contexts.
It needs matching **Tenstorrent Tracy sources**, a C++17 compiler, and the
capstone/zstd development libraries. It builds once into the output directory.
For the Homebrew `v0.10-tt.0` GUI used for `tracy1.tracy` and `tracy2.tracy`:

```bash
curl -L --fail https://codeload.github.com/tenstorrent/tracy/tar.gz/refs/tags/v0.10-tt.0 \
  -o /tmp/ivf-tracy-source.tar.gz
tar -xzf /tmp/ivf-tracy-source.tar.gz -C /tmp
python3 ann_ivf_similarity_partitioned/analyze_device_traces.py \
  --trace serial2="$HOME/Downloads/tracy1.tracy" \
  --trace direct8="$HOME/Downloads/tracy2.tracy" \
  --tracy-source /tmp/tracy-0.10-tt.0 \
  --partitions 8 --layout rows \
  --partition-view 1 \
  --output similarity_trace_analysis
```

Use a Python environment with Matplotlib, or add `--no-plots` for CSV/report
output without plotting dependencies. A subsequent run with the same output
directory reuses the compiled helper and can omit `--tracy-source`. An explicit
`--exporter PATH` also selects a prebuilt helper.

You can instead pass Metalium's `.logs/profile_log_device.csv` or an exported
`raw/*_device.csv` as each `--trace` input. CSV input needs no Tracy sources or
compiler and does not require repeating the workload. A host-only Tracy CSV
is rejected.

Partition IDs are inferred from physical NoC coordinate order and the specified
rows/columns layout; check `partition_mapping.csv` against `benchmark.log`.
The default capture instruments packed batch 1, which belongs to only one
partition. Other partitions have kernel envelopes, **not activity breakdowns**.
Partition completion includes waiting and must not be called utilization.
Page ordinals match reader issue/barrier/compute order, can cross lists, and
are not cluster IDs. Closely spaced read issues are labelled as ordinal ranges.
Durations are local to each RISC; the worker plot uses a local time origin and
does not require synchronized timestamps between different cores.
Partition timelines preserve the recorded timestamps with **one origin shared
by the leader and every worker**. Clock offsets between cores are not corrected
by this script. Uncalibrated inputs are labelled accordingly. Gaps after the
first 16 pages mean missing instrumentation, not an idle reader or compute core.
Wait scopes are grey, and aggregation scopes are labelled as including synchronization.
`pages.csv` includes every sampled worker in direct and relay captures. Missing
direct-read metrics in a relay row are blank, not zero. Relay receive waits
include the offer, leader scheduling, read completion, forwarding and
acknowledgement, so they are not pure DRAM latency or directly equivalent to
direct residual barrier waits. Leader samples are kept separately and are
never attributed to the other workers.
`--partition-view N` chooses a partition (repeat to choose several); without it,
every partition with detailed scopes gets a timeline. `--partition-riscs math`
keeps reader/math/writer rows for workers to make the figure more compact, while
retaining every leader/aggregator RISC. `partition_scopes.csv` records the role,
core, RISC, original scope name and shared-origin timestamps for each event.

All fine-search custom zones now use `--profile-batch N`, the packed global
batch ID after grouping. Other batches still execute normally, so all partitions
compete for DRAM as in the benchmark. Default firmware/kernel envelopes remain.
Per-page zones cover the first **16 page pairs per worker** of the selected
batch; they can span several assigned lists. The reader and compute process
those pairs in the same order. No `DeviceTimestampedData` is used.

| RISC / role | Device scope | Included work |
|---|---|---|
| NCRISC worker | `Worker query receive` | Query CB reservation, leader offer and completion semaphore |
| NCRISC worker | `Worker free buffer wait` | Vector/ID CB reservations; prefetch includes unpublished slots |
| NCRISC direct / serial | `Worker issue DRAM pair` | Issue vector-page and full int32 ID-tile reads |
| NCRISC direct / serial | `Worker read wait` | Completion barrier for that pair, before CB publication |
| NCRISC relay | `Worker relay receive wait` | Candidate offer and leader completion semaphore |
| TRISC worker | `Worker query wait` | Wait for the query tile batch |
| TRISC worker | `Worker input wait` | Wait for this page's vector tiles |
| TRISC worker | `Worker matmul` | Format/short-matmul setup, destination acquire and matmul calls |
| TRISC worker | `Worker score pack` | Score CB reservation, packing/publication, destination release and vector pop |
| TRISC worker | `Worker score and ID wait` | Wait for packed scores and the paired ID tile |
| TRISC worker | `Worker tail mask` | Existing auto/additive masking, only on a partial list's last page |
| TRISC worker | `Worker local topk` | Champion waits/copies, score/ID transpose, local sort, repack and input release |
| TRISC worker | `Worker result pack` | Transpose and publish final local values and IDs |
| BRISC worker | `Worker result wait` / `Worker partial write` | Wait for local results; write both DRAM partials and notify aggregator |
| NCRISC leader | `Leader query read` / `Leader query prefetch` | Current query read, or its earlier prefetch in the previous owning batch |
| NCRISC leader | `Leader query credit wait` / `Leader query broadcast` | Wait for this partition's worker offers; distribute query and signal completion |
| NCRISC leader | `Leader batch done wait` | Wait for this partition's final result acknowledgement |
| NCRISC relay leader | `Leader issue DRAM pair` / `Leader read wait` / `Leader forward pair` | First 16 pairs for worker zero in the selected partition |
| BRISC aggregator | `Partition gather` / `Partition gather partial` | Worker-ready polling and all partial reads; nested scope for each DRAM pair |
| TRISC aggregator | `Partition partial wait` / `Partition aggregate` | Wait for the first 16 partials; reduce every worker's partial |
| BRISC aggregator | `Partition result wait` / `Partition final write` | Wait for reduced result and write final DRAM values and IDs |

`Worker read wait` is **residual waiting at the barrier**, not the complete
issue-to-completion DRAM latency. The prefetch reader can have other tags in
flight. On TRISC_1, inspect `Worker matmul` versus `Worker local topk`; scopes
also include the setup/synchronization listed above. TRISC_0/1/2 execute the
same kernel concurrently: do not add their durations. `Partition gather`
contains its nested read scopes: do not count those twice. Script-cache reads,
metadata transfer and other unscoped work can occupy gaps; a gap alone does not
prove a hardware stall.

The compute worker uses at most **98 custom scopes per RISC** (six per page
including a tail mask, plus query/output), below the 125-scope optional buffer.
Reader/writer/leader zones are bounded to the same selected batch; aggregator
waits are capped at 16 even with 63 workers. A full 10,000-query execution is
therefore suitable for this selected-batch diagnostic. Keep profiler features
disabled when measuring QPS; profiling itself changes the measured execution.

### Capture commands

Copy this entire example to the remote, then rebuild the host executable as
well as letting Metalium JIT the updated kernels. The writer now receives the
profile batch through an extra host runtime argument.

On your Mac:

```bash
scp -r ./ann_ivf_similarity_partitioned \
  your-tt-host:tt-ann-ivf/
```

On the remote:

```bash
cd tt-ann-ivf
source env.sh
# setup_tt_metal.sh builds with Tracy by default. If it was built with
# TT_METAL_BUILD_ARGS=--disable-profiler, enable Tracy and reattach:
cmake -S third_party/tt-metal -B third_party/tt-metal/build_Release -DENABLE_TRACY=ON
cmake --build third_party/tt-metal/build_Release --target tt_metal -j8
bash prepare_runtime.sh
bash build.sh ann_ivf_similarity_partitioned
```

Use `build` instead if that is your configured build; then set
`IVF_EXECUTABLE=./build_Release/bin/ann_ivf_similarity_partitioned`.
Use a matching Tenstorrent Tracy GUI. In a separate Mac terminal keep this SSH
tunnel running:

```bash
ssh -N -L 8086:127.0.0.1:8086 your-tt-host
```

Open `tracy` in another Mac terminal and connect the GUI to `127.0.0.1:8086`
before starting the remote workload. In the remote terminal:

```bash
ivf_scope_dir=$(mktemp -d /tmp/ivf-device-scopes.XXXXXX)
NPROBE=16 CANDIDATE_READER=direct-serial WORKER_INPUT_PAGES=2 \
  bash ann_ivf_similarity_partitioned/run_device_profile.sh \
  "$ivf_scope_dir/serial2_nprobe16"
```

This runs 10,000 queries once after the ordinary warmup and samples batch 1;
the 32-query warmup has only batch 0, so its custom zones are excluded. To
compare the regression, save the GUI capture, reconnect for the next process,
and repeat with `CANDIDATE_READER=direct` / depth 2, then serial/direct at depth
8, using a fresh output subdirectory each time. Set `NPROBE=32` to repeat the
comparison at 512/32. Run processes sequentially. `QUERIES=320` gives a shorter
smoke capture but changes grouped batches and reduces the sustained workload.

The runner sets `TT_METAL_DEVICE_PROFILER=1`, `TT_METAL_PROFILER_DIR`,
`TRACY_PORT=8086` and `TRACY_NO_EXIT=1`. A fresh `TT_METAL_CACHE` below the
capture directory ensures compilation emits scope source-location metadata
for this capture. The runner clears conflicting DPRINT/watcher
and alternate profiler modes. `TRACY_NO_EXIT` keeps shutdown alive until Tracy
connects; it does not pause before the search. If the terminal remains at exit,
connect Tracy and save the capture. Device CSVs are at
`$ivf_scope_dir/serial2_nprobe16/.logs/profile_log_device.csv`; raw commands,
results and grouping/timing diagnostics are saved alongside them. In Tracy,
zoom into the custom worker zones in the last fine-kernel invocation. The CSV
`batches_run_1.csv` identifies the owning partition for batch 1.

To capture the leader-fetch path with the same partition view:

```bash
NPROBE=16 CANDIDATE_READER=relay LEADER_PREFETCH_PAGES=8 \
  WORKER_INPUT_PAGES=2 BANK_SCHEDULE=fifo \
  bash ann_ivf_similarity_partitioned/run_device_profile.sh \
  "$ivf_scope_dir/relay8_nprobe16"
```

Save it as `relay8.tracy`, then compare it with the existing serial2 capture
locally (reuse the existing exporter, or specify matching Tracy sources):

```bash
python3 ann_ivf_similarity_partitioned/analyze_device_traces.py \
  --trace serial2="$HOME/Downloads/tracy1.tracy" \
  --trace relay8="$HOME/Downloads/relay8.tracy" \
  --exporter similarity_trace_analysis/export_tt_device_zones \
  --partition-view 1 --output similarity_relay_trace_analysis
```

The partition plots show the leader's DRAM issue/wait/forward scopes alongside
all seven workers' relay receive waits and compute scopes. They sample one
partition while every other partition runs normally.

The portable test suite uses `-Wall -Wextra -Werror` for C++ targets.
Coverage includes grouping, permutation, scheduling, tails/L1, compute format
transitions and sampled/unsampled worker lifecycle checks in all three mask
modes, twenty-four transport configurations and Python artifact
validation. The transport simulations compile the actual dataflow sources,
defer read and write completion until barriers, and complete reads in reverse
order. They exercise separate aggregators, depths one and fifteen, vector reads
larger than one NoC packet, unequal queues, reordered global batches, empty
workers/partitions, multicast/unicast, paired generations and backpressure.
Direct cases additionally test depths 1/2/3/4/8/15/16, short lists, partial tails,
script-cache boundaries, separate vector/ID consumption, and early completion
of a later page. Pending DMA cannot overlap occupied CB space or be published
before completion; all CBs must be empty when the simulated pipeline ends.
DRAM byte accounting checks the issuing core: query payloads come only from
leaders, while vector/ID payloads come only from leaders in relay mode and only
from workers in direct mode. It also checks exact byte counts for every owning
batch, so duplicated or missing reads fail. Runner tests check sequential
execution, stopping on failure, environment cleanup and both relay depths.
They emulate compute consumers; they do not execute Tensix math or reproduce
physical NoC timing.
Additional timing tests check stage conservation, shared CSV column names,
duplicate/missing runs, warmup exclusion, complete final batches and dataset
percentage calculations.

Run these portable checks locally:

```bash
cmake -S ann_ivf_similarity_partitioned \
  -B /tmp/ivf-partition-host-build -DIVF_PARTITION_HOST_TESTS_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ivf-partition-host-build -j4
ctest --test-dir /tmp/ivf-partition-host-build --output-on-failure
```

**Full Metalium compilation, device JIT compilation, numerical validation and
performance have not been run here:** this workspace has no configured TT
build or Wormhole device. Run remote validation before using benchmark values.
The allocator/controller mapping uses internal repository APIs, so build this
example against the same checkout.

## Project records

The root [AGENTS.md](../AGENTS.md) records kernel invariants and verification commands.
[Project results](../docs/RESULTS.md) consolidate the user-reported
complete timing comparisons.

## File locations

Dataset inputs are in `data/datasets/` and centroids in `data/centroids/`.
`ANN_DATA_DIR` overrides the data root. Defaults write under `results/`;
`ANN_RESULTS_DIR` overrides that root and explicit output arguments keep their
exact paths. See [input layout](../data/README.md) and
[measurement layout](../results/README.md).

Dataset converters are shared in `tools/export_hdf5.py` and
`tools/convert_centroids.py` at the project root.
