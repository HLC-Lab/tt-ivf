#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/env.sh" ""
python3 -B "$project_root/tools/attach_runtime.py" --runtime "$TT_METAL_HOME" \
    --build "${TT_METAL_BUILD_DIR:-$TT_METAL_HOME/build_Release}" \
    --output "$project_root/.runtime"
echo 'External runtime connected. Run: bash build.sh ann_ivf_mem_log ann_ivf_energy ann_ivf_similarity_partitioned'
