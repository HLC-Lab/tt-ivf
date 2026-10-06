# Project guide

## Scope and build

Keep exactly `ann_ivf_mem_log`, `ann_ivf_energy` and
`ann_ivf_similarity_partitioned`. Keep runtime sources and libraries in
`third_party/tt-metal` (git-ignored); do not vendor a second dependency graph.

Read `README.md`, `tt_metal.lock.json` and `docs/RESULTS.md` before changing
runtime integration or interpreting performance. The node117 working build
reported TT-Metal v0.66.0 at
`e7c251da3bf626fa9a74b790f71612ff1ed573bf` with recorded runtime edits.
Preserve the lock and patches of the runtime actually tested. Never silently
replace them.

The runtime is this project's own `third_party/tt-metal`, created from the
lock and patches by `bash setup_tt_metal.sh`. Never make the build or runs
depend on another TT-Metal folder (e.g. `/home/tenstorrent/tt-metal`) or on an
inherited `TT_METAL_HOME`. Build from this root with `source env.sh`, then
`bash build.sh`. Preserve executable names, the root implementation folders and
the separation of runtime and ANN kernel paths. Executables live in
`build_Release/bin/`; JIT paths start with the `ann_ivf_*` folder, relative to
`TT_METAL_KERNEL_PATH`. Runtime preparation (`tools/attach_runtime.py`) changes
generated CMake configuration only; it must restore external project hooks
and leave runtime sources/libraries intact. In `cmake/`, do not add upstream
export sets: they caused duplicate fmt/xtl exports.

Keep tool argument handling safe for spaces: subprocess argument lists,
quoted shell paths, no flattening of paired compiler/link flags or reordering
of libraries. `export_hdf5.py` and `convert_centroids.py` binary formats are
read by all three executables; failed conversions must abort the run.

## Correctness and code changes

- Make bounded, readable changes. Use checked dimensions and explicit ownership;
  retain assertions around protocol boundaries and explain barrier ordering.
- Group only valid queries using the device-selected coarse list IDs. Validate
  all coarse top-32 IDs/scores before selecting nprobe. Preserve exact BF16 rows,
  padded shapes, batch masks and a bijective inverse output permutation.
- Grouping changes each batch's candidate union and can change recall. Report
  recall alongside speed and compare the same query set and settings.
- Match host runtime arguments, L1 offsets, CB shapes and compile arguments
  with the kernels; CB capacity changes must agree with host L1 accounting.
- Preserve paired vector/ID ordering, CB reserve/publish/consume ordering,
  transaction completion before buffer reuse, and payload completion before
  semaphore notification.
- Partitioned experiment: include idle workers in control flow; handle
  zero-work partitions, short lists, partial last batches and out-of-order read
  completion. Multicast only to an exact physical receiver rectangle with the
  leader outside it. Keep profiled and unprofiled paths operationally identical.
- Preserve int32 IDs, tail masks and sentinel filtering. BF16 score ties permit
  alternate IDs; score/count agreement does not independently validate those IDs.
- Deprecation warnings in this pinned runtime are not evidence of a failed build.

## Energy measurement

`ann_ivf_energy` omits recall checks, readback and transfer logging; do not
import the mem-log executable or add timed result dumping. The complete
interval includes device/index creation and searches; per-search intervals
exclude them. Do not add in-window warmup or silently move interval boundaries.
Select one RAPL package counter per prefix (`0-die-*`, `1-die-*`), never sum
duplicates; interpolate boundaries and unwrap rollover. TT board + host package
energy is not wall-socket energy. Use equal query counts, fresh output
directories and independent sessions; keep full and search-only reports apart.

## Verification and evidence

Run the portable CMake/CTest checks (`-DANN_HOST_TESTS_ONLY=ON`) and
`tools/test_project_setup.py` after changes to shared build or protocol code.
Host shims test lifecycle/protocol rules, not Tensix numerics, real race
timing or physical bandwidth. Hardware changes require a matching build,
device JIT, validation (`ann_ivf_similarity_partitioned/run_validation.sh`)
and unprofiled timing on Wormhole.

Benchmark reports exclude warmup. Use 10,000 queries, five measured repetitions
and medians for timing comparisons. Detail timings are included in their
parent stages; do not add them twice. Profiler QPS is diagnostic only.
Update `docs/RESULTS.md` with settings, dates, evidence and limitations; label
user-pasted measurements and distinguish them from locally rerun tests.

## Repository hygiene

Keep data, results, captures, `.runtime/`, `.venv/`, CPM caches and builds out
of Git (see `.gitignore`). Inputs belong in `data/datasets/` and
`data/centroids/`; measurements in `results/`. Preserve explicit output paths
and the original `LICENSE` and `NOTICE`. Never use `git clean -fdx` on a
measurement workspace. Do not commit or push unless the task asks for it.
