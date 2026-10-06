#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# == 0 )); then
    echo "Usage: bash run.sh ann_ivf_mem_log [benchmark arguments...]" >&2
    exit 1
fi
example="$1"
shift
case "$example" in
    ann_ivf_mem_log|ann_ivf_energy|ann_ivf_similarity_partitioned) ;;
    *) echo "Unknown ANN target: $example" >&2; exit 1 ;;
esac
source "$project_root/env.sh" ""
ann_run_build="${ANN_BUILD_DIR:-$project_root/build_Release}"
if [[ -f "$ann_run_build/CMakeCache.txt" ]]; then
    while IFS= read -r ann_run_cache_line; do
        case "$ann_run_cache_line" in
            TT_METAL_HOME:*=*)
                if [[ "${ann_run_cache_line#*=}" != "$TT_METAL_HOME" ]]; then
                    echo "Rebuild against the selected runtime first: bash build.sh $example" >&2
                    exit 1
                fi
                ;;
        esac
    done < "$ann_run_build/CMakeCache.txt"
fi
python3 -B "$project_root/tools/attach_runtime.py" --runtime "$TT_METAL_HOME" --check
binary="$ann_run_build/bin/$example"
[[ -x "$binary" ]] || { echo "Build it first: bash build.sh $example" >&2; exit 1; }
cd "$project_root"
exec "$binary" "$@"
