#!/usr/bin/env bash
# Sequential full-dataset runs with direct readers or leader-to-worker relay.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
if (( $# > 1 )); then
    echo "Usage: bash run_timing_sweep.sh [fresh-output-directory]" >&2
    exit 1
fi
output=${1:-"${ANN_RESULTS_DIR:-$repo_dir/results}/similarity_timings/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"}
executable=${IVF_EXECUTABLE:-"$repo_dir/build_Release/bin/ann_ivf_similarity_partitioned"}
runs=${RUNS:-5}
queries=${QUERIES:-10000}
input_pages=${WORKER_INPUT_PAGES:-2}
reader=${CANDIDATE_READER:-direct}
leader_pages=${LEADER_PREFETCH_PAGES:-8}
bank_schedule=${BANK_SCHEDULE:-fifo}
for number in "$runs" "$queries" "$input_pages" "$leader_pages"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo "RUNS, QUERIES and buffer depths must be positive integers" >&2; exit 1; }
done
(( leader_pages <= 15 )) || { echo "LEADER_PREFETCH_PAGES must be in 1..15" >&2; exit 1; }
[[ "$reader" == direct || "$reader" == direct-serial || "$reader" == relay ]] || { echo "CANDIDATE_READER must be direct, direct-serial or relay" >&2; exit 1; }
[[ "$bank_schedule" == fifo || "$bank_schedule" == staggered ]] || { echo "BANK_SCHEDULE must be fifo or staggered" >&2; exit 1; }
[[ "$reader" == relay || "$bank_schedule" == fifo ]] || { echo "BANK_SCHEDULE=staggered requires CANDIDATE_READER=relay" >&2; exit 1; }
[[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 1; }
executable=$(cd -- "$(dirname -- "$executable")" && printf '%s/%s' "$PWD" "$(basename -- "$executable")")
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "Choose a fresh output directory: $output" >&2; exit 1; }
mkdir -p "$output"
output=$(cd -- "$output" && pwd)
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_DIR TT_METAL_DPRINT_CORES TT_METAL_WATCHER TRACY_WAIT_FOR_CLIENT
cd "$repo_dir"
for nprobe in 16 32; do
    directory="$output/nlist512_nprobe${nprobe}"
    mkdir -p "$directory"
    command=("$executable"
        --dataset "${DATASET:-glove-100-angular}" --nlist 512 --nprobe "$nprobe" --k 10
        --max_num_queries "$queries" --runs "$runs" --result-staging dram
        --organization partitioned --query-grouping weighted-union
        --partitions 8 --partition-layout rows --aggregation leader
        --candidate-reader "$reader" --bank-schedule "$bank_schedule" --candidate-mask auto
        --worker-input-pages "$input_pages" --leader-prefetch-pages "$leader_pages" --query-broadcast auto
        --mem_log "$directory/host.csv" --results-csv "$directory/results.csv"
        --timings-csv "$directory/timings.csv" --diagnostics "$directory")
    printf '%q ' "${command[@]}" > "$directory/command.txt"
    printf '\n' >> "$directory/command.txt"
    echo "Measuring 512/$nprobe: $reader, worker depth $input_pages, leader depth $leader_pages, bank schedule $bank_schedule, $queries queries, $runs runs; log: $directory/benchmark.log"
    if timeout --kill-after=30s "${IVF_TIMEOUT_SECONDS:-1800}" "${command[@]}" > "$directory/benchmark.log" 2>&1; then
        grep -E '^\[Run |^Average QPS=' "$directory/benchmark.log"
    else
        status=$?
        tail -n 30 "$directory/benchmark.log" >&2
        echo "Benchmark failed (status $status); stopped before the next configuration" >&2
        if ! python3 "$script_dir/summarize_timings.py" "$output" --expected-runs "$runs" --allow-incomplete; then
            echo "Could not summarize the completed measurements; raw logs remain in $output" >&2
        fi
        exit "$status"
    fi
done
python3 "$script_dir/summarize_timings.py" "$output" --expected-runs "$runs"
echo "Sequential timing sweep complete: $output"
