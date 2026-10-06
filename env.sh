#!/usr/bin/env bash
# Source this before the experiment/energy/profiler commands: source env.sh [tt-metal]
ann_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Runtime selection: an explicit argument, then the runtime chosen by an earlier
# `source env.sh <path>` in this shell, then this project's own copy. A
# TT_METAL_HOME inherited from the login shell is deliberately not used.
if [[ -n "${1:-}" ]]; then
    ann_metal_path="$1"
elif [[ -n "${ANN_TT_METAL_HOME:-}" ]]; then
    ann_metal_path="$ANN_TT_METAL_HOME"
elif [[ -d "$ann_project_root/third_party/tt-metal" ]]; then
    ann_metal_path="$ann_project_root/third_party/tt-metal"
else
    echo 'Build the pinned runtime first: bash setup_tt_metal.sh' >&2
    echo '(or use another checkout explicitly: source env.sh /path/to/tt-metal)' >&2
    return 1 2>/dev/null || exit 1
fi
if ! python3 "$ann_project_root/tools/check_tt_metal.py" "$ann_metal_path"; then
    return 1 2>/dev/null || exit 1
fi
export TT_METAL_HOME="$(cd -- "$ann_metal_path" && pwd)"
export ANN_TT_METAL_HOME="$TT_METAL_HOME"
# v0.66+ uses this variable for hardware descriptors, firmware and JIT headers.
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
# Energy sessions change CWD. Resolve their kernels in this project first.
export TT_METAL_KERNEL_PATH="$ann_project_root"
export ANN_DATA_DIR="${ANN_DATA_DIR:-$ann_project_root/data}"
export ANN_RESULTS_DIR="${ANN_RESULTS_DIR:-$ann_project_root/results}"
export TT_METAL_CACHE="$ann_project_root/.cache"
export TT_METAL_LOGS_PATH="$ann_project_root/generated"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
unset ann_metal_path ann_project_root
