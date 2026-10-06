#!/usr/bin/env bash
# Run on Wormhole, from any directory. No benchmark processes overlap.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
executable=${IVF_EXECUTABLE:-"$repo_dir/build_Release/bin/ann_ivf_similarity_partitioned"}
output=${1:-"${ANN_RESULTS_DIR:-$repo_dir/results}/similarity_validation/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"}
[[ -x "$executable" ]] || { echo "Build ann_ivf_similarity_partitioned first: $executable" >&2; exit 1; }
executable=$(cd -- "$(dirname -- "$executable")" && printf '%s/%s' "$PWD" "$(basename -- "$executable")")
command -v timeout >/dev/null || { echo "GNU timeout is required for hang detection" >&2; exit 1; }
if [[ -e "$output" ]]; then
    echo "Choose a new output directory; existing files could mix runs: $output" >&2
    exit 1
fi
mkdir -p "$output"
output=$(cd -- "$output" && pwd)
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_DIR TT_METAL_DPRINT_CORES TT_METAL_WATCHER TRACY_WAIT_FOR_CLIENT
cd "$repo_dir"
for queries in 16 32 33 64 320; do
    reference=""
    for mode in direct_serial direct direct_depth4 direct_depth8 relay additive direct_additive shallow separate; do
        directory="$output/queries${queries}/$mode"
        mkdir -p "$directory"
        options=(--candidate-reader relay --bank-schedule staggered --candidate-mask auto
                 --aggregation leader --partition-layout rows --query-broadcast auto
                 --leader-prefetch-pages 8 --worker-input-pages 2)
        case "$mode" in
            direct_serial) options=(--candidate-reader direct-serial --bank-schedule fifo --candidate-mask auto --query-broadcast unicast --worker-input-pages 2) ;;
            direct) options=(--candidate-reader direct --bank-schedule fifo --candidate-mask auto --query-broadcast unicast --worker-input-pages 2) ;;
            direct_depth4) options=(--candidate-reader direct --bank-schedule fifo --candidate-mask auto --query-broadcast unicast --worker-input-pages 4) ;;
            direct_depth8) options=(--candidate-reader direct --bank-schedule fifo --candidate-mask auto --query-broadcast auto --worker-input-pages 8) ;;
            direct_additive) options=(--candidate-reader direct --bank-schedule fifo --candidate-mask additive --query-broadcast unicast --worker-input-pages 4) ;;
            additive) options+=(--candidate-mask additive) ;;
            shallow) options+=(--leader-prefetch-pages 1 --worker-input-pages 1) ;;
            separate) options+=(--aggregation separate --partition-layout columns) ;;
        esac
        echo "Validating $mode with $queries queries"
        command=("$executable"
            --dataset "${DATASET:-glove-100-angular}" --nlist "${NLIST:-512}" --nprobe "${NPROBE:-8}"
            --k 10 --max_num_queries "$queries" --runs 2 --partitions 8 --result-staging dram
            --query-grouping weighted-union --diagnostics "$directory"
            --mem_log "$directory/host.csv" --results-csv "$directory/results.csv"
            --results-output "$directory/neighbors.csv" "${options[@]}")
        printf '%q ' "${command[@]}" > "$directory/command.txt"
        printf '\n' >> "$directory/command.txt"
        if ! timeout --kill-after=30s "${IVF_TIMEOUT_SECONDS:-600}" "${command[@]}" > "$directory/benchmark.log" 2>&1; then
            tail -n 30 "$directory/benchmark.log" >&2
            echo "Validation failed; stopped before the next mode. See $directory" >&2
            exit 1
        fi
        cmp "$directory/queries_run_1.csv" "$directory/queries_run_2.csv"
        if [[ -z "$reference" ]]; then
            reference="$directory/neighbors.csv"
            python3 "$script_dir/compare_results.py" "$reference" "$reference" --queries "$queries" --k 10
        else
            cmp "$(dirname -- "$reference")/queries_run_1.csv" "$directory/queries_run_1.csv"
            python3 "$script_dir/compare_results.py" "$reference" "$directory/neighbors.csv" --allow-ties \
                --queries "$queries" --k 10 \
                > "$directory/comparison.log"
            cat "$directory/comparison.log"
        fi
    done
done
echo "Validation complete. Inspect recall and equal-score ID differences in: $output"
