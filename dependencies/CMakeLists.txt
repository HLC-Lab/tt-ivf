# Link a previously built external TT-Metal runtime.
# The tiny generated file carries the actual build's public headers and link
# libraries, including its dependency versions and Tracy options. This avoids
# copying a runtime checkout or maintaining a second set of CPM versions.
set(_runtime_build "${CMAKE_CURRENT_SOURCE_DIR}/.runtime")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/attach_runtime.py"
        --runtime "${TT_METAL_HOME}" --output "${_runtime_build}" --check
    RESULT_VARIABLE _external_status OUTPUT_VARIABLE _external_output ERROR_VARIABLE _external_error)
if(NOT _external_status EQUAL 0)
    message(FATAL_ERROR "${_external_output}${_external_error}\nRun: bash prepare_runtime.sh")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ann_import_runtime.cmake")
