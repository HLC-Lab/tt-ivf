#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/env.sh" ""
build_dir="${ANN_BUILD_DIR:-$project_root/build_Release}"
build_type="${ANN_BUILD_TYPE:-Release}"
if ! python3 -B "$project_root/tools/attach_runtime.py" --runtime "$TT_METAL_HOME" \
    --build "${TT_METAL_BUILD_DIR:-$TT_METAL_HOME/build_Release}" \
    --output "$project_root/.runtime" --check >/dev/null 2>&1; then
    bash "$project_root/prepare_runtime.sh"
fi
args=(-S "$project_root" -B "$build_dir" -G "${ANN_CMAKE_GENERATOR:-Ninja}"
    "-DCMAKE_BUILD_TYPE=$build_type" "-DTT_METAL_HOME=$TT_METAL_HOME"
    -UANN_METALIUM_MODE -UANN_EXAMPLES -UCPM_SOURCE_CACHE)
# Clear the old target list when upgrading from the export with 13 targets.
# Direct CMake users can still select a subset with -DANN_EXAMPLES=...
# A previous configuration may still contain dependency paths outside this
# project. Reset that CMake cache when changing the runtime source location.
ann_cached_metal=""
ann_cached_mode=""
ann_cached_layout=""
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    while IFS= read -r ann_cache_line; do
        case "$ann_cache_line" in
            TT_METAL_HOME:*=*) ann_cached_metal="${ann_cache_line#*=}" ;;
            ANN_METALIUM_MODE:*=*) ann_cached_mode="${ann_cache_line#*=}" ;;
            ANN_SOURCE_LAYOUT:*=*) ann_cached_layout="${ann_cache_line#*=}" ;;
        esac
    done < "$build_dir/CMakeCache.txt"
fi
if [[ ( -n "$ann_cached_metal" && "$ann_cached_metal" != "$TT_METAL_HOME" ) ||
      ( -n "$ann_cached_mode" && "$ann_cached_mode" != external ) ||
      ( -f "$build_dir/CMakeCache.txt" && "$ann_cached_layout" != flat ) ]]; then
    echo "Runtime location, build mode or source layout changed; refreshing the CMake cache"
    args=(--fresh "${args[@]}")
fi
# Respect an explicitly supplied toolchain or the one from this TT revision.
if [[ -n "${ANN_TOOLCHAIN_FILE:-}" ]]; then
    args+=("-DCMAKE_TOOLCHAIN_FILE=$ANN_TOOLCHAIN_FILE")
else
    if [[ -z "${CXX:-}" ]]; then
        ann_runtime_cxx="$(python3 -B "$project_root/tools/attach_runtime.py" --runtime "$TT_METAL_HOME" --compiler CXX)"
        args+=("-DCMAKE_CXX_COMPILER=$ann_runtime_cxx")
    fi
    if [[ -z "${CC:-}" ]]; then
        ann_runtime_cc="$(python3 -B "$project_root/tools/attach_runtime.py" --runtime "$TT_METAL_HOME" --compiler C)"
        args+=("-DCMAKE_C_COMPILER=$ann_runtime_cc")
    fi
fi
if [[ -n "${ANN_CMAKE_ARGS:-}" ]]; then
    echo "Use ANN_TOOLCHAIN_FILE, CC/CXX, or cmake directly for extra CMake flags" >&2
    exit 1
fi
cmake "${args[@]}"
targets=("$@")
if (( ${#targets[@]} == 0 )); then targets=(ann_examples); fi
cmake --build "$build_dir" --parallel "${JOBS:-2}" --target "${targets[@]}"
# Old experiment runners use build/ or build_Release/ by default.
if [[ "$build_dir" == "$project_root/build_Release" && ! -e "$project_root/build" && ! -L "$project_root/build" ]]; then
    ln -s build_Release "$project_root/build"
fi
