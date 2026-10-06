#!/usr/bin/env bash
# Same grouping/topology/compute; compare direct reads with leader relay.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
if (( $# > 1 )); then
    echo "Usage: bash run_relay_comparison.sh [fresh-output-directory]" >&2
    exit 1
fi
output=${1:-"${ANN_RESULTS_DIR:-$repo_dir/results}/similarity_relay/$(date -u +%Y%m%dT%H%M%SZ)_pid$$"}
[[ ! -e "$output" ]] || { echo "Choose a fresh output directory: $output" >&2; exit 1; }
mkdir -p "$output"
output=$(cd -- "$output" && pwd)
# Each child completes both N_probe values and exits before the next starts.
# Keep FIFO ordering and two worker input slots fixed; vary only the relay
# staging depth. The direct-serial baseline has no leader candidate staging.
CANDIDATE_READER=direct-serial WORKER_INPUT_PAGES=2 LEADER_PREFETCH_PAGES=8 BANK_SCHEDULE=fifo \
    bash "$script_dir/run_timing_sweep.sh" "$output/serial2"
for depth in 4 8; do
    CANDIDATE_READER=relay WORKER_INPUT_PAGES=2 LEADER_PREFETCH_PAGES="$depth" BANK_SCHEDULE=fifo \
        bash "$script_dir/run_timing_sweep.sh" "$output/relay$depth"
done
python3 "$script_dir/summarize_timings.py" "$output" --expected-runs "${RUNS:-5}"
echo "Sequential reader comparison complete: $output"
