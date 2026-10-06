# tt-ann-ivf

Three ANN IVF experiments for Tenstorrent Wormhole, with reproducible timing,
query grouping, device profiling and energy analysis.

| Executable | Purpose |
|---|---|
| `ann_ivf_mem_log` | Whole-list search, grouping comparison and stage/transfer logging |
| `ann_ivf_energy` | Initialization/search markers for external TT power and host RAPL sampling |
| `ann_ivf_similarity_partitioned` | Independent partitions, direct/relay readers and Tracy device zones |

The project builds and runs against **its own pinned TT-Metal** in
`third_party/tt-metal` (git-ignored). `tt_metal.lock.json` records the exact
TT-Metal and submodule revisions, and `dependency_state/*.patch` holds the
local runtime edits of the tested node117 build. Setup recreates that runtime
from the lock, so no other TT-Metal folder on the host is used.
The optional FAISS baseline uses IVF scalar quantization with fp16 vectors.

## Layout

```text
tt-ann-ivf/
├── ann_ivf_mem_log/                 # sources, kernels, tests, plotting scripts
├── ann_ivf_energy/                  # plus run_tt_energy.sh, sampler/analysis, FAISS baseline
├── ann_ivf_similarity_partitioned/  # sources, kernels, tests, sweep/profiling scripts
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

All three experiments are built by default. To rebuild one, use
`bash build.sh ann_ivf_mem_log`. `JOBS=8` changes build parallelism. The first
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

The conversion scripts are shared by all three implementations:

```bash
python3 tools/export_hdf5.py data/datasets/glove-100-angular.hdf5
python3 tools/convert_centroids.py data/centroids/centroids-glove-100-angular-512.npy
```

See [data/README.md](data/README.md) for the full layout.

## Run

Use 10,000 queries and five measured runs for the recorded throughput comparison:

```bash
bash run.sh ann_ivf_mem_log --dataset glove-100-angular --nlist 512 \
  --nprobe 16 --k 10 --runs 5 --max_num_queries 10000 \
  --query-grouping weighted-union

RUNS=5 QUERIES=10000 \
  bash ann_ivf_similarity_partitioned/run_timing_sweep.sh
```

Default outputs are under `results/`, grouped by experiment. `ANN_RESULTS_DIR`
selects another root; explicit output arguments retain their exact paths.
Run from this root after sourcing `env.sh`. Experiment READMEs document
validation, grouping modes, relay comparisons and Tracy capture/plots.
Profiling changes timing; use unprofiled runs for throughput claims.

For energy, install the compatible `tt-smi` telemetry tool and ensure CPU
package RAPL counters are readable:

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
energy. No final energy results are bundled yet.

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
