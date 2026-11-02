# tt-ann-ivf

Approximate nearest-neighbor (IVF-Flat) search on a Tenstorrent Wormhole n150,
with reproducible timing, validation and energy measurement.

| Executable | Purpose |
|---|---|
| `ann_ivf_ttaiss` | **TTAISS**, the final implementation: queries grouped by the lists they select, each list read once per search, query tiles built on the device, and a two-pass search that skips hopeless top-k updates. Exact per-query IVF results, optional FP32 re-ranking |
| `ann_ivf_mem_log` | Earlier batch-based search (each 32-query batch scans the union of its lists), with stage and transfer logging; used as a baseline |
| `ann_ivf_energy` | Energy workload of the batch-based search, with initialization/search markers for TT power and host RAPL sampling; also hosts the shared energy sampler and analyzer |

The project builds and runs against **its own pinned TT-Metal** in
`third_party/tt-metal` (git-ignored). `tt_metal.lock.json` records the exact
TT-Metal and submodule revisions, and `dependency_state/*.patch` holds the
local runtime edits of the tested node117 build. Setup recreates that runtime
from the lock, so no other TT-Metal folder on the host is used.
The optional FAISS baseline uses IVF scalar quantization with fp16 vectors.

## Layout

```text
tt-ann-ivf/
├── ann_ivf_ttaiss/                  # final implementation: sources, kernels, tests, comparison/energy scripts
├── ann_ivf_mem_log/                 # batch-based baseline: sources, kernels, tests, plotting scripts
├── ann_ivf_energy/                  # batch-based energy workload, sampler/analysis, FAISS baseline
├── docs/figures/                    # TikZ overview figures of TTAISS
├── common/
├── data/                            # datasets/ and centroids/ (assets ignored)
├── results/                         # generated measurements (ignored)
├── tools/                           # data conversion, runtime attach/pinning
├── cmake/
├── dependencies/                    # CMake import of the runtime
├── dependency_state/                # recorded runtime patches (with the lock)
├── third_party/tt-metal/            # pinned runtime from setup_tt_metal.sh (ignored)
├── docs/RESULTS.md                  # recorded hardware results
├── CMakeLists.txt
└── build_Release/bin/               # generated executables, ignored by Git
```

## Setup and build

The host needs Wormhole drivers/firmware and TT-Metal's system build
dependencies (its `install_dependencies.sh`). Then:

```bash
git clone <this repository> tt-ann-ivf && cd tt-ann-ivf
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash setup_tt_metal.sh     # once: fetch + build pinned TT-Metal (slow)
source env.sh
bash build.sh
```

`setup_tt_metal.sh` clones the locked revision into `third_party/tt-metal`,
applies the recorded patches, verifies their hashes and runs TT-Metal's
`build_metal.sh --build-type Release` (Tracy enabled). Extra build options go
in `TT_METAL_BUILD_ARGS`. An existing `third_party/tt-metal` is only verified.

`source env.sh` selects `third_party/tt-metal`; a `TT_METAL_HOME` exported by
your login shell is ignored. To deliberately use another checkout at the
locked revision, pass it explicitly: `source env.sh /path/to/tt-metal`.
`TT_METAL_RUNTIME_ROOT` points to the runtime's descriptors/firmware/JIT
headers; `TT_METAL_KERNEL_PATH` points here, including energy runs with a
different CWD.

All three executables are built by default. To rebuild one, use
`bash build.sh ann_ivf_ttaiss`. `JOBS=8` changes build parallelism. The first
build records the runtime's resolved headers, compile options and link
libraries under `.runtime/` (`prepare_runtime.sh`); rerun it after rebuilding
TT-Metal with different options. The compiler is taken from the runtime build.

To move to a different TT-Metal version, pin a working checkout (its revision
and tracked edits are recorded; the checkout itself is not modified), then
delete `third_party/tt-metal` and rerun `bash setup_tt_metal.sh`:

```bash
python3 tools/use_tt_metal.py /path/to/working/tt-metal --record-local-changes
```

Commit the updated lock with its referenced `dependency_state/*.patch` files.
Untracked files in that checkout are not recorded.

## Dataset

Data and trained centroids are distributed separately. Dataset files go in
`data/datasets/`; centroid files go in `data/centroids/`. Import existing exports:

```bash
python3 tools/link_data.py /path/to/dataset/files
```

GloVe runs need `data/datasets/glove-100-angular_{train,queries,neighbors,distances}.bin`
and `data/centroids/centroids-glove-100-angular-512.bin`. `--dataset` remains
`glove-100-angular`. Set `ANN_DATA_DIR` to another root containing these two
folders. HDF5 conversion writes beside its input; FAISS writes trained centroids
to the centroid folder. The link helper refuses conflicting destinations.

The conversion scripts are shared by all executables:

```bash
python3 tools/export_hdf5.py data/datasets/glove-100-angular.hdf5
python3 tools/convert_centroids.py data/centroids/centroids-glove-100-angular-512.npy
```

See [data/README.md](data/README.md) for the full layout.

## Run

Use 10,000 queries and five measured runs for throughput measurements:

```bash
bash run.sh ann_ivf_ttaiss --dataset glove-100-angular --nlist 512 \
  --nprobe 64 --k 10 --runs 5 --max_num_queries 10000          # add --rerank for FP32 re-ranking

NPROBES="32 48 56 64" RUNS=5 QUERIES=10000 \
  bash ann_ivf_ttaiss/run_gather_comparison.sh                 # validated against an exact per-query search

bash run.sh ann_ivf_mem_log --dataset glove-100-angular --nlist 512 \
  --nprobe 16 --k 10 --runs 5 --max_num_queries 10000 --query-grouping weighted-union
```

Default outputs are under `results/`, grouped by experiment. `ANN_RESULTS_DIR`
selects another root; explicit output arguments retain their exact paths.
Run from this root after sourcing `env.sh`. The executable READMEs document
validation, grouping modes and profiling.
Profiling changes timing; use unprofiled runs for throughput claims.

For energy, the sampler reads TT board power from the kernel driver's hwmon
interface and needs readable CPU package RAPL counters. TTAISS is compared
with the batch-based search at matched recall by
`ann_ivf_ttaiss/run_energy_comparison.sh`; a batch-based sweep alone runs as:

```bash
ivf_energy_compare_dir="$PWD/results/energy_raw/grouping_$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
CONFIGURATIONS='512:16 512:32' GROUPINGS='none weighted-union' \
SESSIONS_PER_CONFIG=10 SEARCH_REPETITIONS=10 MAX_NUM_QUERIES=10000 \
BINARY="$PWD/build_Release/bin/ann_ivf_energy" \
ENERGY_ROOT="$ivf_energy_compare_dir" bash ann_ivf_energy/run_tt_energy.sh
```

Use a fresh output directory. The sampler selects one counter per package
prefix X (`0-die-*`, `1-die-*`); it does not sum duplicate die readings.
Reports distinguish complete initialization/search energy from search-only
energy. Recorded energy results are in `docs/RESULTS.md`.

## Verification and recorded results

```bash
cmake -S . -B build_host -DANN_HOST_TESTS_ONLY=ON
cmake --build build_host --parallel 2
ctest --test-dir build_host --output-on-failure
python3 -B tools/test_project_setup.py
```

Host models check grouping, padding, coarse/worker lifecycle, transport and
reporting. They cannot establish actual Tensix arithmetic or NoC timing;
run device validation before accepting a kernel change. See
[docs/RESULTS.md](docs/RESULTS.md) for user-reported hardware results and
[AGENTS.md](AGENTS.md) for development rules. GitHub CI runs the portable checks.

## License

Parts of this project are derived from TT-Metal programming examples.
`LICENSE` (Apache-2.0) and `NOTICE` are retained from TT-Metal; the runtime's
own license applies to its external checkout.
