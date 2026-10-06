#!/usr/bin/env bash
# Fetch and build this project's own pinned TT-Metal in third_party/tt-metal.
# No other TT-Metal folder on the host is read or modified.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="$project_root/third_party/tt-metal"
python3 -B "$project_root/tools/fetch_tt_metal.py" --directory "$runtime"
# Release with Tracy (the build_metal.sh default), as used by the experiments.
# Extra build_metal.sh options, e.g. TT_METAL_BUILD_ARGS='--enable-ccache'.
read -r -a extra_args <<< "${TT_METAL_BUILD_ARGS:-}"
(
    cd "$runtime"
    # build_metal.sh may consult these; keep a caller's other checkout out of it.
    export TT_METAL_HOME="$runtime" TT_METAL_RUNTIME_ROOT="$runtime"
    ./build_metal.sh --build-type Release ${extra_args[@]+"${extra_args[@]}"}
)
echo "TT-Metal built in $runtime"
echo 'Next: source env.sh && bash build.sh'
