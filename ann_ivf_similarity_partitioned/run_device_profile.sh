#!/usr/bin/env bash
# Capture one measured search with DeviceZoneScopedN and a connected Tracy GUI.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
if (( $# > 1 )); then
    echo "Usage: bash run_device_profile.sh [fresh-output-directory]" >&2
    exit 1
fi
output=${1:-"${ANN_RESULTS_DIR:-$repo_dir/results}/similarity_device_profiles/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"}
executable=${IVF_EXECUTABLE:-"$repo_dir/build_Release/bin/ann_ivf_similarity_partitioned"}
nprobe=${NPROBE:-16}
queries=${QUERIES:-10000}
profile_batch=${PROFILE_BATCH:-1}
input_pages=${WORKER_INPUT_PAGES:-2}
reader=${CANDIDATE_READER:-direct-serial}
leader_pages=${LEADER_PREFETCH_PAGES:-8}
bank_schedule=${BANK_SCHEDULE:-fifo}
for number in "$nprobe" "$queries" "$input_pages" "$leader_pages"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo "NPROBE, QUERIES and buffer depths must be positive integers" >&2; exit 1; }
done
(( leader_pages <= 15 )) || { echo "LEADER_PREFETCH_PAGES must be in 1..15" >&2; exit 1; }
[[ "$profile_batch" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "PROFILE_BATCH must be a nonnegative integer" >&2; exit 1; }
(( nprobe <= 32 )) || { echo "NPROBE must be <=32" >&2; exit 1; }
(( profile_batch < (queries + 31) / 32 )) || { echo "PROFILE_BATCH is outside the measured query batches" >&2; exit 1; }
[[ "$reader" == direct || "$reader" == direct-serial || "$reader" == relay ]] || { echo "CANDIDATE_READER must be direct, direct-serial or relay" >&2; exit 1; }
[[ "$bank_schedule" == fifo || "$bank_schedule" == staggered ]] || { echo "BANK_SCHEDULE must be fifo or staggered" >&2; exit 1; }
[[ "$reader" == relay || "$bank_schedule" == fifo ]] || { echo "BANK_SCHEDULE=staggered requires CANDIDATE_READER=relay" >&2; exit 1; }
[[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "Choose a fresh output directory: $output" >&2; exit 1; }
mkdir -p "$output"
output=$(cd -- "$output" && pwd)
executable=$(cd -- "$(dirname -- "$executable")" && printf '%s/%s' "$PWD" "$(basename -- "$executable")")
# Watcher and DPRINT compete with the profiler's SRAM. Use regular scoped
# profiling, preserving separate RISC rows and both CSV/Tracy output.
unset TT_METAL_DPRINT_CORES TT_METAL_WATCHER TRACY_WAIT_FOR_CLIENT
unset TT_METAL_DEVICE_PROFILER_DISPATCH TT_METAL_DEVICE_PROFILER_NOC_EVENTS TT_METAL_PROFILE_PERF_COUNTERS
unset TT_METAL_TRACE_PROFILER TT_METAL_PROFILER_SUM TT_METAL_PROFILER_SYNC
unset TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_DIR="$output"
# Recompile into this capture's cache so profiler source-location metadata is
# generated here even if another capture has already cached these kernels.
export TT_METAL_CACHE="$output/kernel_cache"
export TRACY_NO_EXIT=1
export TRACY_PORT="${TRACY_PORT:-8086}"
cd "$repo_dir"
command=("$executable"
    --dataset "${DATASET:-glove-100-angular}" --nlist 512 --nprobe "$nprobe" --k 10
    --max_num_queries "$queries" --runs 1 --result-staging dram
    --organization partitioned --query-grouping weighted-union
    --partitions 8 --partition-layout rows --aggregation leader
    --candidate-reader "$reader" --bank-schedule "$bank_schedule" --candidate-mask auto
    --worker-input-pages "$input_pages" --leader-prefetch-pages "$leader_pages" --query-broadcast auto
    --profile-batch "$profile_batch" --diagnostics "$output"
    --mem_log "$output/host.csv" --results-csv "$output/results.csv"
    --timings-csv "$output/timings.csv" --results-output "$output/results_dump.csv")
printf '%q ' "${command[@]}" > "$output/command.txt"
printf '\n' >> "$output/command.txt"
printf 'TT_METAL_DEVICE_PROFILER=1\nTT_METAL_PROFILER_DIR=%s\nTT_METAL_CACHE=%s\nTRACY_NO_EXIT=1\nTRACY_PORT=%s\n' "$output" "$TT_METAL_CACHE" "$TRACY_PORT" > "$output/profiler_environment.txt"
echo "Connect the matching Tenstorrent Tracy GUI to port $TRACY_PORT before this run."
echo "Profiling 512/$nprobe, $reader, worker depth $input_pages, leader depth $leader_pages, bank schedule $bank_schedule, $queries queries; sampled global batch $profile_batch."
echo "TRACY_NO_EXIT keeps shutdown waiting for a Tracy connection; it does not pause before search."
if (( profile_batch == 0 )); then
    echo "Batch 0 also occurs in warmup: inspect the last fine-kernel invocation in Tracy."
fi
"${command[@]}" 2>&1 | tee "$output/benchmark.log"
echo "Capture directory: $output"
echo "Device scope CSV: $output/.logs/profile_log_device.csv"
echo "Save the .tracy capture from the GUI. Profiled QPS is diagnostic only."
