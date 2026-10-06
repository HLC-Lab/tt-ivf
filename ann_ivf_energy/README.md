# ANN IVF energy workload

This executable uses the data layout, LPT cluster scheduling, 63-worker fine
search, local sort, aggregator, and DRAM result staging from `ann_ivf_mem_log`.
It is intended for external board-power sampling.

The measured interval begins immediately before IVF index construction. It
includes device creation, centroid-file reading, centroid and dataset
normalization, IVF cluster assignment, tile-layout preparation, device-buffer
allocation, dataset and centroid upload, and every search. Training and query
dataset files, plus any centroid format conversion, are handled before the
interval. There is no warmup search, ground-truth loading, recall calculation,
result validation, final-result device-to-host readback,
per-query text output, or memory-transfer CSV logging. The final sorter still
writes its result tiles to DRAM so the device workload remains complete.

Build and run:

```bash
bash build.sh ann_ivf_energy

./build_Release/bin/ann_ivf_energy \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 32 \
  --k 10 \
  --max_num_queries 10000 \
  --runs 10 \
  --result-staging dram \
  --query-grouping none
```

`ENERGY_MEASUREMENT_START` and `ENERGY_MEASUREMENT_END` delimit the complete
index-initialization and search interval that an external sampler integrates.

Run the Tenstorrent board-power sampler from the repository root:

```bash
python3 ann_ivf_energy/plot_energy_recall.py \
  --target tt \
  --dataset glove-100-angular \
  --nlist 512 \
  --nprobe 32 \
  --k 10 \
  --max-num-queries 10000 \
  --runs 10 \
  --result-staging dram \
  --query-grouping none
```

The script writes raw power samples, a power timeline, and an energy summary
containing average power, joules, and joules per query.

There is no warmup inside the measurement. The sweep prepares the kernel cache with separate 32-query invocations by default (`WARM_CACHE=1`); use `WARM_CACHE=0` to include cold-cache/JIT costs. With a cold TT-Metal kernel cache,
the first measured run can include JIT compilation. For steady-state energy,
populate the kernel cache with a separate one-run invocation before launching
the sampler.

Run the predefined pairwise sweep with:

```bash
bash ann_ivf_energy/run_tt_energy.sh
```

The script uses zipped pairs rather than a Cartesian product:
`(2048,4)`, `(2048,12)`, `(1024,16)`, and `(512,32)`.

It executes ten independent fresh processes for every pair. Each process runs
ten searches of 10,000 queries, giving 100,000 measured queries per process,
40 processes, and 4,000,000 measured queries across the default sweep. All
processes are strictly sequential. The sampler thread is joined before the
next process starts, a host lock prevents two copies of the sweep from
overlapping, and the default pause between processes is ten seconds.

The output hierarchy is:

```text
energy_raw/tt_wormhole/<timestamp>_pid<pid>/
  sessions.csv
  none/nlist2048_nprobe4/session_1/
  none/nlist2048_nprobe4/session_2/
  ...
```

Each session directory contains its exact command, environment information,
benchmark output, measurement boundary timestamps, raw TT and host power
samples, plots, and the energy summary. To change the cooling interval while
keeping the same serial execution:

```bash
COOL_DOWN_SECONDS=30 bash ann_ivf_energy/run_tt_energy.sh
```

Analyze a completed sweep with:

```bash
python3 ann_ivf_energy/analyze_tt_energy.py \
  energy_raw/tt_wormhole/<timestamp>_pid<pid>
```

The analyzer integrates every raw power log over its recorded measurement
markers, validates the manifest and per-session summary, and writes aggregate
CSV files and energy-distribution plots below the run's `analysis` directory.

## Compare query grouping energy

The same `ann_ivf_energy` executable now accepts `--query-grouping none`,
`primary-list`, or `weighted-union`. Grouping uses actual device coarse top-32
BF16 scores and the selected `N_probe` IDs, checks for corrupt coarse rows,
reorders normalized BF16 query rows, uploads them for fine search, and rebuilds
batch masks. Padding does not select lists. The fine schedule assigns whole
lists; cluster chunking has been removed. The old zero-valued chunk option is
accepted for compatibility; nonzero values are rejected. The coarse kernel
uses the same initialization and format-transition corrections validated in
the timing implementation. The fine kernels and result staging are retained.

Run the controlled 512/16 and 512/32 comparison on the Tenstorrent machine:

```bash
cmake --build build_Release --target ann_ivf_energy -j2
ivf_energy_compare_dir="$PWD/results/energy_raw/tt_wormhole/grouping_$(date -u +%Y%m%dT%H%M%SZ)"
CONFIGURATIONS="512:16 512:32" GROUPINGS="none weighted-union" \
SESSIONS_PER_CONFIG=10 SEARCH_REPETITIONS=10 MAX_NUM_QUERIES=10000 \
BINARY="$PWD/build_Release/bin/ann_ivf_energy" \
ENERGY_ROOT="$ivf_energy_compare_dir" \
bash ann_ivf_energy/run_tt_energy.sh
```

This executes 40 independent processes, each with ten 10,000-query searches.
They run sequentially under a shared host lock, with a ten-second cooling
interval. Mode order alternates between sessions. `CONFIGURATIONS` contains
`NList:NProbe` pairs; `GROUPINGS` lists the modes to compare. Override `DATASET`,
`GROUPING_WINDOW=0`, or `GROUPING_LOOKAHEAD=256` when needed. Use a fresh
`ENERGY_ROOT` for each experiment. `BINARY` may point to your existing `build`
instead of `build_Release` executable.

Sampling begins before device/index creation. Exact C++ timestamps are retained
in `events.csv`; `sampler_events.csv` records process/stdout arrival times for
auditing. TT board power is integrated with interpolated boundaries; host
energy is the difference of unwrapped cumulative RAPL counters, with the same
boundary interpolation. One counter is selected per package prefix X, accepting
`package-X`, `package-X-die-Y`, and `X-die-Y`. If no aggregate counter is
available, the lowest numbered die is selected: for example `0-die-0` and
`1-die-0`, even when four readings exist per prefix. This also applies when
reanalyzing raw captures that recorded all dies. Core/uncore and psys zones are
excluded. Host and TT sampling use separate threads. New
captures must bracket both boundaries. The sweep requires readable package
counters so it cannot present device-only energy as system energy.

The default full window includes initialization (normalization, IVF cluster
assignment, tilization, device allocation and uploads) and all searches,
including query grouping, repacking and its extra transfer. Dataset loading and
format conversion precede the window. There is no final-result readback,
recall calculation or memory-transfer CSV logging in this executable. Use the
separately validated timing experiment for recall. Host package energy includes
background work and the sampler; system means board + host CPU packages, not
wall-socket energy. No idle baseline is subtracted.

The script automatically produces both analyses:

- `analysis/`: full initialization + searches;
- `analysis/searches/`: sum of the individual search intervals, excluding
  index creation and gaps between searches.

Each contains `tt_energy_sessions.csv`,
`tt_energy_configuration_summary.csv`, `tt_energy_grouping_comparison.csv`,
`tt_energy_report.md`, energy distribution and RSD plots. Quartiles, IQR,
standard deviation and RSD are computed separately for device energy and the
per-session device + host sum. Energy savings compare the medians of equal
query-count sessions. The source archive and binary SHA-256 identify the code
used for the measurement, including untracked example files.

Reanalyze without rerunning the workload:

```bash
python3 ann_ivf_energy/analyze_tt_energy.py "$ivf_energy_compare_dir"
python3 ann_ivf_energy/analyze_tt_energy.py "$ivf_energy_compare_dir" --window searches
python3 ann_ivf_energy/analyze_tt_energy.py "$ivf_energy_compare_dir" --window initialization
```

For initialization-only analysis, mJ/query means initialization energy amortized
across all measured queries. Legacy captures remain readable with the default
full window; separate phase analysis requires the new C++ event records.

Portable checks:

```bash
cmake -S ann_ivf_energy \
  -B /tmp/ivf-energy-host -DIVF_ENERGY_HOST_TESTS_ONLY=ON
cmake --build /tmp/ivf-energy-host
ctest --test-dir /tmp/ivf-energy-host --output-on-failure
python3 ann_ivf_energy/tests/test_energy_analysis.py
```

These verify grouping, batch masks, query packing, partial batches, page counts,
coarse kernel CB/format lifecycle, counter wraparound, boundary interpolation,
package selection and system RSD. Device arithmetic and energy measurements
require a hardware run.

## Project records

See [project results](../docs/RESULTS.md) for measured timing and the current
energy status. Build/run from the standalone project root after sourcing `env.sh`.

## File locations

Dataset inputs are in `data/datasets/` and centroids in `data/centroids/`.
`ANN_DATA_DIR` overrides the data root. Defaults write under `results/`;
`ANN_RESULTS_DIR` overrides that root and explicit output arguments keep their
exact paths. See [input layout](../data/README.md) and
[measurement layout](../results/README.md).

Dataset converters are shared in `tools/export_hdf5.py` and
`tools/convert_centroids.py` at the project root.
