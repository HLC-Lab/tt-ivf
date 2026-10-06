#!/usr/bin/env bash

set -euo pipefail

# This sweep intentionally uses only host std::chrono measurements.
unset TT_METAL_DEVICE_PROFILER
unset TT_METAL_DEVICE_PROFILER_NOC_EVENTS
unset TT_METAL_PROFILER_DIR

# Environment-variable overrides:
#   DATASET=glove-100-angular
#   NLIST=512
#   NPROBES="1 8 16 32"
#   MAX_QUERIES=32
#   RUNS=5
#   RESULT_STAGING=dram
#   PREFETCH_READER=2
#   QUERY_GROUPING=none  # or weighted-union / primary-list
#   GROUPING_WINDOW=0
#   GROUPING_LOOKAHEAD=256
#   PYTHON_BIN=python3
#   BINARY=./build_Release/bin/ann_ivf_mem_log
#   OUTPUT_ROOT=results/memcopy_results
#   SWEEP_ID=my_run_label

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-glove-100-angular}"
NLIST="${NLIST:-512}"
NPROBES_TEXT="${NPROBES:-1 8 16 32}"
MAX_QUERIES="${MAX_QUERIES:-32}"
RUNS="${RUNS:-5}"
RESULT_STAGING="${RESULT_STAGING:-dram}"
PREFETCH_READER="${PREFETCH_READER:-2}"
QUERY_GROUPING="${QUERY_GROUPING:-none}"
GROUPING_WINDOW="${GROUPING_WINDOW:-0}"
GROUPING_LOOKAHEAD="${GROUPING_LOOKAHEAD:-256}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BINARY="${BINARY:-${REPO_ROOT}/build_Release/bin/ann_ivf_mem_log}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ANN_RESULTS_DIR:-${REPO_ROOT}/results}/memcopy_results}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

read -r -a NPROBE_VALUES <<< "${NPROBES_TEXT}"

if [[ ! -x "${BINARY}" ]]; then
    echo "Error: executable not found: ${BINARY}" >&2
    echo "Build it first with: bash build.sh ann_ivf_mem_log" >&2
    exit 1
fi

if [[ "${NLIST}" -lt 32 || $((NLIST % 32)) -ne 0 ]]; then
    echo "Error: NLIST must be at least 32 and divisible by 32." >&2
    exit 1
fi

if [[ "${RUNS}" -lt 1 || "${MAX_QUERIES}" -lt 1 ]]; then
    echo "Error: RUNS and MAX_QUERIES must be positive." >&2
    exit 1
fi

if [[ "${RESULT_STAGING}" != "dram" && "${RESULT_STAGING}" != "core0-l1" ]]; then
    echo "Error: RESULT_STAGING must be 'dram' or 'core0-l1'." >&2
    exit 1
fi

if [[ ! "${PREFETCH_READER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: PREFETCH_READER must be a positive integer." >&2
    exit 1
fi

for nprobe in "${NPROBE_VALUES[@]}"; do
    if [[ "${nprobe}" -lt 1 || "${nprobe}" -gt 32 ]]; then
        echo "Error: every nprobe must be between 1 and 32; received ${nprobe}." >&2
        exit 1
    fi
done

if ! "${PYTHON_BIN}" -c "import matplotlib.pyplot, numpy" >/dev/null 2>&1; then
    echo "Error: ${PYTHON_BIN} needs NumPy and Matplotlib to create the PNG." >&2
    echo "Activate your python_env and verify: python3 -c 'import matplotlib.pyplot, numpy'" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/nlist_${NLIST}"
mkdir -p "${OUTPUT_DIR}"

echo "=== ANN IVF memcopy sweep ==="
echo "Dataset:       ${DATASET}"
echo "nlist:         ${NLIST}"
echo "nprobe values: ${NPROBE_VALUES[*]}"
echo "Measured runs: ${RUNS}"
echo "Queries/run:   ${MAX_QUERIES}"
echo "Result stage:  ${RESULT_STAGING}"
echo "Reader depth:  ${PREFETCH_READER} blocks"
echo "Grouping:      ${QUERY_GROUPING} (window=${GROUPING_WINDOW}, lookahead=${GROUPING_LOOKAHEAD})"
echo "Output:        ${OUTPUT_DIR}"

cd "${REPO_ROOT}"

plot_arguments=(
    --nlist "${NLIST}"
    --output "${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}.png"
    --without-tt-fine-output "${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}_without_tt_fine.png"
    --data-output "${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}.csv"
    --html-output "${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}.html"
)

for nprobe in "${NPROBE_VALUES[@]}"; do
    echo
    echo "=== Running nlist=${NLIST}, nprobe=${nprobe} ==="

    mem_log="${OUTPUT_DIR}/nprobe_${nprobe}_mem_log.csv"
    memcopy_summary="${OUTPUT_DIR}/nprobe_${nprobe}_memcopy_summary.csv"
    console_log="${OUTPUT_DIR}/nprobe_${nprobe}_console.log"

    "${BINARY}" \
        --run_id "${SWEEP_ID}_nlist${NLIST}_nprobe${nprobe}" \
        --dataset "${DATASET}" \
        --nlist "${NLIST}" \
        --nprobe "${nprobe}" \
        --max_num_queries "${MAX_QUERIES}" \
        --runs "${RUNS}" \
        --result-staging "${RESULT_STAGING}" \
        --prefetch-reader "${PREFETCH_READER}" \
        --query-grouping "${QUERY_GROUPING}" \
        --grouping-window "${GROUPING_WINDOW}" \
        --grouping-lookahead "${GROUPING_LOOKAHEAD}" \
        --mem_log "${mem_log}" \
        2>&1 | tee "${console_log}"

    if [[ ! -s "${mem_log}" ]]; then
        echo "Error: main.cpp did not produce ${mem_log}" >&2
        exit 1
    fi

    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_memcopy_log.py" \
        --mem-log "${mem_log}" \
        --output "${memcopy_summary}"

    plot_arguments+=(--mem-log "${nprobe}=${mem_log}")
done

echo
echo "=== Generating comparison CSV and PNG ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_memcopy_costs.py" "${plot_arguments[@]}"

echo
echo "Sweep complete."
echo "Per-workload CSVs from main.cpp: ${OUTPUT_DIR}/nprobe_*_mem_log.csv"
echo "Combined plotted CSV:            ${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}.csv"
echo "PNG:                             ${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}.png"
echo "PNG without TT fine search:      ${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}_without_tt_fine.png"
echo "Interactive HTML:                ${OUTPUT_DIR}/memcopy_cost_nlist${NLIST}.html"
