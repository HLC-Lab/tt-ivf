#!/usr/bin/env bash
# Energy comparison at matched recall levels: batch-major ann_ivf_energy (the
# TT + CPU reference configurations of ann_ivf_energy/run_tt_energy.sh) and
# ann_ivf_ttaiss --energy-events, measured by the unchanged
# ann_ivf_energy sampler and analyzer.
#
# Every session is an independent fresh process: TT board power (hwmon) and one
# RAPL package counter per prefix are sampled; the C++ markers bound the
# complete window (device/index creation + all searches) and each search.
# Implementations alternate their order per session to cancel thermal and
# order drift, with the same query counts and cool-down. Each implementation
# gets its own run directory/manifest so the analyzer stays unchanged;
# summarize_energy.py then compares them per level. Board + CPU package energy
# is not wall-socket energy; no idle baseline is subtracted.
#
# A level is batch/gather, each part nlist:nprobe, the gather part optionally
# with :rerank: the workload then also reads its results back and re-ranks them
# in FP32 (part of producing the results). Before the sessions, one ordinary
# run per level of gather (outside every measured window) records its
# Recall@10; ann_ivf_energy computes no recall.
#
# Environment overrides:
#   LEVELS="2048:4/2048:5 2048:12/2048:20 1024:16/1024:48 512:32/512:64:rerank"
#   IMPLEMENTATIONS="batch_major ttaiss"  (either or both)
#   BATCH_GROUPING=none DATASET=glove-100-angular
#   SESSIONS_PER_CONFIG=10 SEARCH_REPETITIONS=10 MAX_NUM_QUERIES=10000
#   BATCH_CHUNK=32 COOL_DOWN_SECONDS=10
#   ENERGY_ROOT=results/energy_raw/ttaiss_comparison/<UTC>_pid<PID> (fresh)
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_dir"
ENERGY_DIR="ann_ivf_energy"
read -r -a levels <<< "${LEVELS:-2048:4/2048:5 2048:12/2048:20 1024:16/1024:48 512:32/512:64:rerank}"
read -r -a implementations <<< "${IMPLEMENTATIONS:-batch_major ttaiss}"
BATCH_GROUPING="${BATCH_GROUPING:-none}"
DATASET="${DATASET:-glove-100-angular}"
SESSIONS_PER_CONFIG="${SESSIONS_PER_CONFIG:-10}"
SEARCH_REPETITIONS="${SEARCH_REPETITIONS:-10}"
MAX_NUM_QUERIES="${MAX_NUM_QUERIES:-10000}"
BATCH_CHUNK="${BATCH_CHUNK:-32}"
COOL_DOWN_SECONDS="${COOL_DOWN_SECONDS:-10}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN=python3
    [[ -x "$repo_dir/.venv/bin/python" ]] && PYTHON_BIN="$repo_dir/.venv/bin/python"
fi
BATCH_BINARY="$(realpath "${BATCH_BINARY:-$repo_dir/build_Release/bin/ann_ivf_energy}")"
GATHER_BINARY="$(realpath "${GATHER_BINARY:-$repo_dir/build_Release/bin/ann_ivf_ttaiss}")"

exec 9>"${TT_ENERGY_LOCK_FILE:-/tmp/ann_ivf_energy.lock}"
flock -n 9 || { echo "Another ANN IVF energy sweep is running on this host" >&2; exit 1; }
for binary in "$BATCH_BINARY" "$GATHER_BINARY"; do
    [[ -x "$binary" ]] || { echo "Missing executable: $binary" >&2; exit 1; }
done
for value in "$SESSIONS_PER_CONFIG" "$SEARCH_REPETITIONS" "$MAX_NUM_QUERIES" "$BATCH_CHUNK"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Counts must be positive integers" >&2; exit 1; }
done
(( MAX_NUM_QUERIES <= 10000 )) || { echo "At most 10000 queries per repetition are supported" >&2; exit 1; }
[[ "$BATCH_GROUPING" == none || "$BATCH_GROUPING" == weighted-union || "$BATCH_GROUPING" == primary-list ]] \
    || { echo "Unknown grouping" >&2; exit 1; }
level_pattern='^([0-9]+):([0-9]+)/([0-9]+):([0-9]+)(:rerank)?$'
for level in "${levels[@]}"; do
    [[ "$level" =~ $level_pattern ]] || { echo "Use LEVELS='2048:4/2048:5 512:32/512:64:rerank'" >&2; exit 1; }
done
(( ${#implementations[@]} >= 1 )) || { echo "IMPLEMENTATIONS is empty" >&2; exit 1; }
for name in "${implementations[@]}"; do
    [[ "$name" == batch_major || "$name" == ttaiss ]] \
        || { echo "Unknown implementation: $name" >&2; exit 1; }
done
unset TT_METAL_DEVICE_PROFILER TT_METAL_DEVICE_PROFILER_NOC_EVENTS TT_METAL_PROFILER_DIR TRACY_NO_EXIT \
      TT_METAL_DPRINT_CORES TT_METAL_WATCHER
ENERGY_ROOT="${ENERGY_ROOT:-${ANN_RESULTS_DIR:-$repo_dir/results}/energy_raw/ttaiss_comparison/$(date -u +%Y%m%dT%H%M%SZ)_pid$$}"
[[ ! -e "$ENERGY_ROOT" ]] || { echo "Choose a fresh ENERGY_ROOT: $ENERGY_ROOT" >&2; exit 1; }
mkdir -p "$ENERGY_ROOT"
ENERGY_ROOT="$(realpath "$ENERGY_ROOT")"
# Telemetry and host package counters must work before any device workload.
PYTHONPATH="$ENERGY_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c 'from energy_sampling import discover_rapl; from plot_energy_recall import select_tt_power_reader; assert discover_rapl(), "Readable host CPU package RAPL counters are required"; select_tt_power_reader("auto")[0]()'

sha256sum "$BATCH_BINARY" "$GATHER_BINARY" > "$ENERGY_ROOT/binaries.sha256"
{ cat tt_metal.lock.json; git rev-parse HEAD 2>/dev/null || true; git status --short 2>/dev/null || true; } \
    > "$ENERGY_ROOT/source_revision.txt"
# Includes untracked experiment folders, which git diff alone would omit.
measurement_sources=(ann_ivf_energy ann_ivf_ttaiss common tools/data_layout.py tools/export_hdf5.py
                     tools/convert_centroids.py tt_metal.lock.json)
[[ -d dependency_state ]] && measurement_sources+=(dependency_state)
tar --exclude=__pycache__ -czf "$ENERGY_ROOT/measurement_source.tar.gz" "${measurement_sources[@]}"
printf '%s\n' "${levels[@]}" > "$ENERGY_ROOT/levels.txt"
for name in "${implementations[@]}"; do
    mkdir -p "$ENERGY_ROOT/$name"
    printf 'ordinal,grouping,nlist,nprobe,session,searches,queries_per_search,start_utc,end_utc,status,output_dir\n' \
        > "$ENERGY_ROOT/$name/sessions.csv"
done

# Level -> sampler arguments in the array `extra`, nlist/nprobe, grouping and
# the rerank flag (space-safe).
level_args() {  # level implementation
    [[ "$1" =~ $level_pattern ]]
    rerank_flag=""
    case "$2" in
        batch_major)
            nlist="${BASH_REMATCH[1]}" nprobe="${BASH_REMATCH[2]}" grouping="$BATCH_GROUPING"
            extra=(--executable "$BATCH_BINARY" --fine-schedule batch-major --result-staging dram) ;;
        ttaiss)
            nlist="${BASH_REMATCH[3]}" nprobe="${BASH_REMATCH[4]}" grouping=none rerank_flag="${BASH_REMATCH[5]}"
            extra=(--executable "$GATHER_BINARY" --fine-schedule cluster-major --batch-chunk "$BATCH_CHUNK"
                   --fine-math-fidelity HiFi4) ;;
    esac
    extra+=(--query-grouping "$grouping")
    [[ -n "$rerank_flag" ]] && extra+=(--executable-arg=--rerank)
    return 0
}

# Kernel caches and the gather recall, outside every measured window.
echo "Populating kernel caches and recording recall outside measured sessions"
for level in "${levels[@]}"; do
    for name in "${implementations[@]}"; do
        level_args "$level" "$name"
        if [[ "$name" == batch_major ]]; then
            "$BATCH_BINARY" --dataset "$DATASET" --k 10 --nlist "$nlist" --nprobe "$nprobe" --runs 1 --max_num_queries 32 \
                --query-grouping "$grouping" --result-staging dram \
                > "$ENERGY_ROOT/cache_batch_major_${nlist}_${nprobe}.log" 2>&1
            continue
        fi
        binary="$GATHER_BINARY"
        rerank=()
        [[ -n "$rerank_flag" ]] && rerank=(--rerank)
        "$binary" --dataset "$DATASET" --k 10 --nlist "$nlist" --nprobe "$nprobe" --runs 1 \
            --max_num_queries "$MAX_NUM_QUERIES" --batch-chunk "$BATCH_CHUNK" --query-grouping "$grouping" "${rerank[@]}" \
            --mem_log "$ENERGY_ROOT/recall_${name}_${nlist}_${nprobe}.csv" \
            > "$ENERGY_ROOT/recall_${name}_${nlist}_${nprobe}.log" 2>&1
    done
done
sleep "$COOL_DOWN_SECONDS"

count=${#implementations[@]}
total=$((SESSIONS_PER_CONFIG * ${#levels[@]} * count))
ordinal=0
declare -A per_impl_ordinal
for ((session = 1; session <= SESSIONS_PER_CONFIG; session++)); do
    for level in "${levels[@]}"; do
        for ((i = 0; i < count; i++)); do
            # Rotate the order per session so every implementation runs in
            # every position (two implementations: plain alternation).
            index=$(((i + session - 1) % count))
            name="${implementations[$index]}"
            level_args "$level" "$name"
            root="$ENERGY_ROOT/$name"
            ordinal=$((ordinal + 1))
            per_impl_ordinal[$name]=$(( ${per_impl_ordinal[$name]:-0} + 1 ))
            session_dir="$root/$grouping/nlist${nlist}_nprobe${nprobe}/session_$session"
            mkdir -p "$session_dir"
            echo "Energy session $ordinal/$total: $name $nlist/$nprobe ($level); $SEARCH_REPETITIONS x $MAX_NUM_QUERIES queries"
            { hostname; lscpu; echo "Implementation=$name Level=$level"; } > "$session_dir/environment.txt"
            start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            set +e
            MPLBACKEND=Agg "$PYTHON_BIN" "$ENERGY_DIR/plot_energy_recall.py" --target tt "${extra[@]}" \
                --dataset "$DATASET" --k 10 --grouping-window 0 --grouping-lookahead 256 \
                --nlist "$nlist" --nprobe "$nprobe" --runs "$SEARCH_REPETITIONS" --max-num-queries "$MAX_NUM_QUERIES" \
                --require-host --no-plots --output-dir "$session_dir" > "$session_dir/sampler.log" 2>&1
            status=$?
            set -e
            end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf '%d,%s,%d,%d,%d,%d,%d,%s,%s,%d,%s\n' "${per_impl_ordinal[$name]}" "$grouping" "$nlist" "$nprobe" \
                "$session" "$SEARCH_REPETITIONS" "$MAX_NUM_QUERIES" "$start_utc" "$end_utc" "$status" "$session_dir" \
                >> "$root/sessions.csv"
            (( status == 0 )) || { tail -n 20 "$session_dir/sampler.log" >&2; echo "Session failed: $session_dir" >&2; exit "$status"; }
            (( ordinal < total )) && sleep "$COOL_DOWN_SECONDS"
        done
    done
done
for name in "${implementations[@]}"; do
    "$PYTHON_BIN" "$ENERGY_DIR/analyze_tt_energy.py" "$ENERGY_ROOT/$name" --expected-sessions "$SESSIONS_PER_CONFIG" --no-plots
    "$PYTHON_BIN" "$ENERGY_DIR/analyze_tt_energy.py" "$ENERGY_ROOT/$name" --expected-sessions "$SESSIONS_PER_CONFIG" \
        --no-plots --window searches
done
"$PYTHON_BIN" "$script_dir/summarize_energy.py" "$ENERGY_ROOT" | tee "$ENERGY_ROOT/summary.md"
echo "Results: $ENERGY_ROOT"
