#!/usr/bin/env bash
# Sequential, unprofiled comparison on the same queries of
#   fast_single  this executable, --first-pass-lists 0 (device-built query tiles)
#   fast         this executable, --first-pass-lists auto (two passes with merge
#                skipping from nprobe 24, host pass-2 work overlapped)
# Every run is validated against the exact per-query IVF search
# (validate_gather.py, with the coarse top-nprobe check). Run after
# `source env.sh` with nothing else loading the host.
#
# Environment overrides:
#   DATASET=glove-100-angular NLIST=512 K=10 QUERIES=10000 RUNS=5 BATCH_CHUNK=32
#   NPROBES="32 48 56 64"  IMPLEMENTATIONS="fast_single fast"
#     ablations: fast_rank32 (32nd-best threshold), fast_rebuild (programs/buffers
#     rebuilt per search), fast_rank32_rebuild (both, i.e. before the two changes)
#   RERANK=1   host FP32 re-ranking of the device top-32 (every implementation)
#   OUTPUT=results/ttaiss_comparison/<UTC>_pid<PID>  (must not exist)
set -euo pipefail
unset TT_METAL_DEVICE_PROFILER TT_METAL_DEVICE_PROFILER_NOC_EVENTS TT_METAL_PROFILER_DIR \
      TT_METAL_DPRINT_CORES TT_METAL_WATCHER TRACY_WAIT_FOR_CLIENT
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
DATASET="${DATASET:-glove-100-angular}"
NLIST="${NLIST:-512}"
K="${K:-10}"
QUERIES="${QUERIES:-10000}"
RUNS="${RUNS:-5}"
BATCH_CHUNK="${BATCH_CHUNK:-32}"
RERANK="${RERANK:-0}"
rerank_args=()
validate_args=()
if [[ "$RERANK" == 1 ]]; then rerank_args=(--rerank); validate_args=(--reranked); fi
read -r -a nprobes <<< "${NPROBES-32 48 56 64}"
read -r -a implementations <<< "${IMPLEMENTATIONS-fast_single fast}"
OUTPUT="${OUTPUT:-${ANN_RESULTS_DIR:-$repo_dir/results}/ttaiss_comparison/$(date -u +%Y%m%dT%H%M%SZ)_pid$$}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN=python3
    [[ -x "$repo_dir/.venv/bin/python" ]] && PYTHON_BIN="$repo_dir/.venv/bin/python"
fi
command -v timeout >/dev/null || { echo "GNU timeout is required for hang detection" >&2; exit 1; }
[[ -x "$repo_dir/build_Release/bin/ann_ivf_ttaiss" ]] \
    || { echo "Build ann_ivf_ttaiss first: bash build.sh ann_ivf_ttaiss" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "Choose a new output directory: $OUTPUT" >&2; exit 1; }
mkdir -p "$OUTPUT"
OUTPUT="$(cd -- "$OUTPUT" && pwd)"
cd "$repo_dir"
git rev-parse HEAD > "$OUTPUT/git_head.txt" 2>/dev/null || true
git status --short > "$OUTPUT/git_status.txt" 2>/dev/null || true
tar --exclude=__pycache__ -czf "$OUTPUT/measurement_sources.tar.gz" ann_ivf_ttaiss common
sha256sum build_Release/bin/ann_ivf_ttaiss > "$OUTPUT/binaries.sha256"
cp tt_metal.lock.json "$OUTPUT/"
uptime > "$OUTPUT/uptime_start.txt"

run_one() {  # directory target [extra args...]
    local directory="$1" target="$2"
    shift 2
    mkdir -p "$directory"
    local command=("$repo_dir/build_Release/bin/$target" --dataset "$DATASET" --nlist "$NLIST" --k "$K"
        --max_num_queries "$QUERIES" --runs "$RUNS" --batch-chunk "$BATCH_CHUNK"
        --mem_log "$directory/mem_log.csv" --results-output "$directory/neighbors.csv"
        --selections-output "$directory/selections.bin" --assignment-output "$directory/assignment.bin"
        "${rerank_args[@]}" "$@")
    printf '%q ' "${command[@]}" > "$directory/command.txt"
    printf '\n' >> "$directory/command.txt"
    echo "$(basename -- "$directory")"
    if ! ANN_RESULTS_DIR="$directory/results" timeout --kill-after=30s "${IVF_TIMEOUT_SECONDS:-1800}" \
            "${command[@]}" > "$directory/console.log" 2>&1; then
        tail -n 30 "$directory/console.log" >&2
        echo "Run failed; stopped. See $directory" >&2
        exit 1
    fi
    "$PYTHON_BIN" "$script_dir/validate_gather.py" --dataset "$DATASET" "${validate_args[@]}" \
        --centroids "${ANN_DATA_DIR:-$repo_dir/data}/centroids/centroids-${DATASET}-${NLIST}.bin" "$directory/neighbors.csv" \
        "$directory/selections.bin" "$directory/assignment.bin" | tee "$directory/validation.log"
}
for nprobe in "${nprobes[@]}"; do
    for implementation in "${implementations[@]}"; do
        directory="$OUTPUT/${implementation}_nprobe_${nprobe}"
        case "$implementation" in
            fast_single) run_one "$directory" ann_ivf_ttaiss --nprobe "$nprobe" --first-pass-lists 0 ;;
            fast) run_one "$directory" ann_ivf_ttaiss --nprobe "$nprobe" --first-pass-lists auto ;;
            # Ablations of the rank-k threshold and the persistent programs/buffers.
            fast_rank32) run_one "$directory" ann_ivf_ttaiss --nprobe "$nprobe" --first-pass-lists auto \
                             --ablation-rank32 ;;
            fast_rebuild) run_one "$directory" ann_ivf_ttaiss --nprobe "$nprobe" --first-pass-lists auto \
                              --ablation-rebuild ;;
            fast_rank32_rebuild) run_one "$directory" ann_ivf_ttaiss --nprobe "$nprobe" \
                                     --first-pass-lists auto --ablation-rank32 --ablation-rebuild ;;
            *) echo "Unknown implementation: $implementation" >&2; exit 1 ;;
        esac
    done
done
uptime > "$OUTPUT/uptime_end.txt"
"$PYTHON_BIN" "$script_dir/summarize_gather.py" "$OUTPUT" | tee "$OUTPUT/summary.md"
