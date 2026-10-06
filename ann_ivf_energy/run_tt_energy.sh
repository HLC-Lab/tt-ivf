#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
# Run from the project root so data/results defaults and archived paths match.
cd "$PROJECT_ROOT"
ENERGY_DIR="ann_ivf_energy"
# These are pairs, not a Cartesian product. Override for the grouping study.
read -r -a CONFIG_VALUES <<< "${CONFIGURATIONS:-2048:4 2048:12 1024:16 512:32}"
read -r -a GROUPING_VALUES <<< "${GROUPINGS:-none}"
SESSIONS_PER_CONFIG="${SESSIONS_PER_CONFIG:-10}"
SEARCH_REPETITIONS="${SEARCH_REPETITIONS:-10}"
MAX_NUM_QUERIES="${MAX_NUM_QUERIES:-10000}"
GROUPING_WINDOW="${GROUPING_WINDOW:-0}"
GROUPING_LOOKAHEAD="${GROUPING_LOOKAHEAD:-256}"
COOL_DOWN_SECONDS="${COOL_DOWN_SECONDS:-10}"
WARM_CACHE="${WARM_CACHE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BINARY="${BINARY:-$PROJECT_ROOT/build_Release/bin/ann_ivf_energy}"
BINARY="$(realpath "$BINARY")"
DATASET="${DATASET:-glove-100-angular}"

exec 9>"${TT_ENERGY_LOCK_FILE:-/tmp/ann_ivf_energy.lock}"
if ! flock -n 9; then
    echo "Another ANN IVF energy sweep is running on this host" >&2
    exit 1
fi
if [[ ! -x "$BINARY" ]]; then
    echo "Missing executable: $BINARY; build target ann_ivf_energy first" >&2
    exit 1
fi
for value in "$SESSIONS_PER_CONFIG" "$SEARCH_REPETITIONS" "$MAX_NUM_QUERIES" "$GROUPING_LOOKAHEAD"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Session/run/query/lookahead counts must be positive integers" >&2; exit 1; }
done
[[ "$GROUPING_WINDOW" =~ ^[0-9]+$ ]] && (( GROUPING_WINDOW % 32 == 0 )) || { echo "GROUPING_WINDOW must be 0 or a multiple of 32" >&2; exit 1; }
(( MAX_NUM_QUERIES <= 10000 )) || { echo "At most 10000 queries per repetition are supported" >&2; exit 1; }
[[ "$WARM_CACHE" == 0 || "$WARM_CACHE" == 1 ]] || { echo "WARM_CACHE must be 0 or 1" >&2; exit 1; }
for pair in "${CONFIG_VALUES[@]}"; do
    [[ "$pair" =~ ^([0-9]+):([0-9]+)$ ]] || { echo "Use CONFIGURATIONS='512:16 512:32'" >&2; exit 1; }
    (( BASH_REMATCH[1] >= 32 && BASH_REMATCH[1] % 32 == 0 && BASH_REMATCH[2] >= 1 && BASH_REMATCH[2] <= 32 )) || exit 1
done
for grouping in "${GROUPING_VALUES[@]}"; do
    [[ "$grouping" == none || "$grouping" == weighted-union || "$grouping" == primary-list ]] || { echo "Unknown grouping: $grouping" >&2; exit 1; }
done
# Disable profiler instrumentation for energy measurements.
unset TT_METAL_DEVICE_PROFILER TT_METAL_DEVICE_PROFILER_NOC_EVENTS TT_METAL_PROFILER_DIR TRACY_NO_EXIT
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
ENERGY_ROOT="${ENERGY_ROOT:-${ANN_RESULTS_DIR:-$PROJECT_ROOT/results}/energy_raw/tt_wormhole/$RUN_STAMP}"
mkdir -p "$ENERGY_ROOT"
ENERGY_ROOT="$(realpath "$ENERGY_ROOT")"
MANIFEST="$ENERGY_ROOT/sessions.csv"
[[ ! -e "$MANIFEST" ]] || { echo "Refusing to overwrite existing manifest: $MANIFEST" >&2; exit 1; }
# Check telemetry and host package counters before any device workload.
PYTHONPATH="$ENERGY_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c 'from energy_sampling import discover_rapl; from plot_energy_recall import read_tt_power_once; assert discover_rapl(), "Readable host CPU package RAPL counters are required"; read_tt_power_once()'
printf 'ordinal,grouping,nlist,nprobe,session,searches,queries_per_search,start_utc,end_utc,status,output_dir\n' > "$MANIFEST"
sha256sum "$BINARY" > "$ENERGY_ROOT/binary.sha256"
{
    cat "$PROJECT_ROOT/tt_metal.lock.json"
    git rev-parse HEAD 2>/dev/null || echo "ANN project: source archive (no Git HEAD)"
    git diff -- ann_ivf_energy 2>/dev/null || true
} > "$ENERGY_ROOT/source_revision.txt"
# Includes untracked example files, which git diff alone would omit.
measurement_sources=(ann_ivf_energy common tools/data_layout.py tools/export_hdf5.py tools/convert_centroids.py tt_metal.lock.json)
if [[ -d "$PROJECT_ROOT/dependency_state" ]]; then measurement_sources+=(dependency_state); fi
tar --exclude=__pycache__ -czf "$ENERGY_ROOT/measurement_source.tar.gz" "${measurement_sources[@]}"
common=(--dataset "$DATASET" --k 10 --result-staging dram --grouping-window "$GROUPING_WINDOW" --grouping-lookahead "$GROUPING_LOOKAHEAD")
if [[ "$WARM_CACHE" == 1 ]]; then
    echo "Populating kernel cache outside measured sessions"
    for pair in "${CONFIG_VALUES[@]}"; do
        for grouping in "${GROUPING_VALUES[@]}"; do
            "$BINARY" "${common[@]}" --nlist "${pair%:*}" --nprobe "${pair#*:}" --query-grouping "$grouping" --runs 1 --max_num_queries 32 > "$ENERGY_ROOT/cache_${pair/:/_}_${grouping}.log" 2>&1
        done
    done
    sleep "$COOL_DOWN_SECONDS"
fi
TOTAL_SESSIONS=$((SESSIONS_PER_CONFIG * ${#CONFIG_VALUES[@]} * ${#GROUPING_VALUES[@]}))
ordinal=0
for ((session=1; session<=SESSIONS_PER_CONFIG; session++)); do
    for pair in "${CONFIG_VALUES[@]}"; do
        nlist="${pair%:*}"
        nprobe="${pair#*:}"
        for ((g=0; g<${#GROUPING_VALUES[@]}; g++)); do
            # Alternate mode order to reduce systematic thermal/order effects.
            gi=$g
            if (( session % 2 == 0 )); then gi=$((${#GROUPING_VALUES[@]} - 1 - g)); fi
            grouping="${GROUPING_VALUES[$gi]}"
            ordinal=$((ordinal + 1))
            session_dir="$ENERGY_ROOT/$grouping/nlist${nlist}_nprobe${nprobe}/session_$session"
            mkdir -p "$session_dir"
            echo "Energy session $ordinal/$TOTAL_SESSIONS: $nlist/$nprobe $grouping; $SEARCH_REPETITIONS x $MAX_NUM_QUERIES queries"
            {
                hostname
                lscpu
                echo "Grouping=$grouping Window=$GROUPING_WINDOW Lookahead=$GROUPING_LOOKAHEAD WarmCache=$WARM_CACHE"
                echo "Binary=$BINARY"
            } > "$session_dir/environment.txt"
            start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            set +e
            MPLBACKEND=Agg "$PYTHON_BIN" "$ENERGY_DIR/plot_energy_recall.py" --target tt --executable "$BINARY" "${common[@]}" --nlist "$nlist" --nprobe "$nprobe" --query-grouping "$grouping" --runs "$SEARCH_REPETITIONS" --max-num-queries "$MAX_NUM_QUERIES" --require-host --no-plots --output-dir "$session_dir"
            status=$?
            set -e
            end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            actual_queries="$MAX_NUM_QUERIES"
            if (( status == 0 )); then
                actual_queries="$("$PYTHON_BIN" -c 'import csv,sys; r=next(csv.DictReader(open(sys.argv[1]))); print(int(r["queries"]) // int(r["runs"]))' "$session_dir/energy_summary_${nlist}_${nprobe}_tt.csv")"
            fi
            printf '%d,%s,%d,%d,%d,%d,%d,%s,%s,%d,%s\n' "$ordinal" "$grouping" "$nlist" "$nprobe" "$session" "$SEARCH_REPETITIONS" "$actual_queries" "$start_utc" "$end_utc" "$status" "$session_dir" >> "$MANIFEST"
            (( status == 0 )) || { echo "Session failed; stopped: $session_dir" >&2; exit "$status"; }
            if (( ordinal < TOTAL_SESSIONS )); then sleep "$COOL_DOWN_SECONDS"; fi
        done
    done
done
"$PYTHON_BIN" "$ENERGY_DIR/analyze_tt_energy.py" "$ENERGY_ROOT" --expected-sessions "$SESSIONS_PER_CONFIG"
"$PYTHON_BIN" "$ENERGY_DIR/analyze_tt_energy.py" "$ENERGY_ROOT" --expected-sessions "$SESSIONS_PER_CONFIG" --window searches

echo "Results: $ENERGY_ROOT"
echo "Manifest: $MANIFEST"
