#!/usr/bin/env bash
# Six changes measured separately; optional focused grouped topology comparison.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
matrix=false
sweep=false
grouped_topology=false
output=""
for arg in "$@"; do
    case "$arg" in
        --matrix) matrix=true ;;
        --sweep-topology) sweep=true ;;
        --grouped-topology) grouped_topology=true ;;
        --*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *) [[ -z "$output" ]] || { echo "Pass only one output directory" >&2; exit 1; }; output=$arg ;;
    esac
done
if "$grouped_topology" && "$sweep"; then
    echo "Choose --grouped-topology or --sweep-topology; they measure different reader paths" >&2
    exit 1
fi
output=${output:-"${ANN_RESULTS_DIR:-$repo_dir/results}/similarity_ablation/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"}
executable=${IVF_EXECUTABLE:-"$repo_dir/build_Release/bin/ann_ivf_similarity_partitioned"}
[[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 1; }
executable=$(cd -- "$(dirname -- "$executable")" && printf '%s/%s' "$PWD" "$(basename -- "$executable")")
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 1; }
if [[ -e "$output" ]]; then
    echo "Choose a new output directory; existing files could mix runs: $output" >&2
    exit 1
fi
mkdir -p "$output"
output=$(cd -- "$output" && pwd)
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_DIR TT_METAL_DPRINT_CORES TT_METAL_WATCHER TRACY_WAIT_FOR_CLIENT
configs=("${NLIST:-512}:${NPROBE:-8}")
if "$matrix"; then
    configs=()
    for nlist in 512 1024 2048; do
        for nprobe in 1 2 4 8 16 32; do configs+=("$nlist:$nprobe"); done
    done
    configs+=(2048:12)
fi
cd "$repo_dir"
measure() {
    local label=$1; shift
    local directory="$output/nlist${nlist}_nprobe${nprobe}/$label"
    mkdir -p "$directory"
    echo "Measuring $nlist/$nprobe: $label"
    local -a benchmark_command=("$executable"
        --dataset "${DATASET:-glove-100-angular}" --nlist "$nlist" --nprobe "$nprobe" --k 10
        --max_num_queries "${QUERIES:-10000}" --runs "${RUNS:-3}" --result-staging dram
        --partitions 8 --partition-layout rows --aggregation leader --candidate-mask auto
        --worker-input-pages 2 --leader-prefetch-pages 8 --query-broadcast auto
        --diagnostics "$directory" --mem_log "$directory/host.csv" --results-csv "$directory/results.csv"
        --timings-csv "$directory/timings.csv" "$@")
    printf '%q ' "${benchmark_command[@]}" > "$directory/command.txt"
    printf '\n' >> "$directory/command.txt"
    timeout --kill-after=30s "${IVF_TIMEOUT_SECONDS:-1800}" "${benchmark_command[@]}" > "$directory/benchmark.log" 2>&1
    grep -E '^\[Run |^Average QPS=' "$directory/benchmark.log" || true
}
for config in "${configs[@]}"; do
    nlist=${config%:*}; nprobe=${config#*:}
    if "$grouped_topology"; then
        measure global_grouped --organization global --query-grouping weighted-union --candidate-reader direct --bank-schedule fifo
        measure partition_grouped_p4 --organization partitioned --partitions 4 --query-grouping weighted-union --candidate-reader direct --bank-schedule fifo
        measure partition_grouped_p8 --organization partitioned --partitions 8 --query-grouping weighted-union --candidate-reader direct --bank-schedule fifo
        continue
    fi
    measure global_identity --organization global --query-grouping none --candidate-reader direct --bank-schedule fifo
    measure global_grouped --organization global --query-grouping weighted-union --candidate-reader direct --bank-schedule fifo
    measure partition_identity --query-grouping none --candidate-reader direct --bank-schedule fifo
    measure partition_grouped --query-grouping weighted-union --candidate-reader direct --bank-schedule fifo
    measure relay_fifo --query-grouping weighted-union --candidate-reader relay --bank-schedule fifo
    measure relay_staggered --query-grouping weighted-union --candidate-reader relay --bank-schedule staggered
    if "$sweep"; then
        for partitions in 4 6 8 12; do
            for layout in rows columns; do
                measure "relay_p${partitions}_${layout}" --query-grouping weighted-union \
                    --candidate-reader relay --bank-schedule staggered --partitions "$partitions" --partition-layout "$layout"
            done
        done
    fi
done
python3 "$script_dir/analyze_grouping.py" "$output"
if "$grouped_topology"; then
    python3 "$script_dir/summarize_timings.py" "$output" --expected-runs "${RUNS:-3}"
fi
echo "Sequential ablation complete: $output"
