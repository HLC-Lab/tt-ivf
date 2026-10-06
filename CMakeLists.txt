cmake_minimum_required(VERSION 3.24...4.2)
project(ANN_IVF LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/ann_examples.cmake)
set(ANN_EXAMPLES "${ANN_DEFAULT_EXAMPLES}" CACHE STRING "ANN example folders to build (semicolon separated)")
option(ANN_HOST_TESTS_ONLY "Build the portable ANN tests without TT-Metal or hardware" OFF)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

foreach(example IN LISTS ANN_EXAMPLES)
    if(NOT example IN_LIST ANN_DEFAULT_EXAMPLES)
        message(FATAL_ERROR "Unknown or archived ANN target: ${example}. Supported: ${ANN_DEFAULT_EXAMPLES}")
    endif()
endforeach()

if(ANN_HOST_TESTS_ONLY)
    add_executable(ann_paths_test common/tests/paths_test.cpp)
    target_compile_features(ann_paths_test PRIVATE cxx_std_20)
    target_compile_options(ann_paths_test PRIVATE -Wall -Wextra -Werror)
    add_test(NAME ann_paths COMMAND ann_paths_test "${CMAKE_CURRENT_BINARY_DIR}/path fixture")
    set(IVF_ENERGY_HOST_TESTS_ONLY ON)
    set(IVF_MEM_LOG_HOST_TESTS_ONLY ON)
    set(IVF_PARTITION_HOST_TESTS_ONLY ON)
    set(_host_examples ann_ivf_energy ann_ivf_mem_log ann_ivf_similarity_partitioned)
    foreach(example IN LISTS ANN_EXAMPLES)
        if(example IN_LIST _host_examples)
            add_subdirectory(${example})
        endif()
    endforeach()
    if(ann_ivf_energy IN_LIST ANN_EXAMPLES)
        add_test(NAME ann_energy_analysis COMMAND ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/ann_ivf_energy/tests/test_energy_analysis.py)
    endif()
    if(ann_ivf_mem_log IN_LIST ANN_EXAMPLES)
        foreach(plot_test IN ITEMS fine_cluster_bubbles fine_pipeline_timeline fine_low_level_timeline fine_reader_deep_timeline)
            add_test(NAME ann_mem_log_plot_${plot_test} COMMAND ${Python3_EXECUTABLE} -B
                ${CMAKE_CURRENT_SOURCE_DIR}/ann_ivf_mem_log/tests/test_plot_${plot_test}.py)
        endforeach()
    endif()
    return()
endif()

if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME} AND NOT "$ENV{TT_METAL_HOME}" STREQUAL "")
        set(TT_METAL_HOME "$ENV{TT_METAL_HOME}" CACHE PATH "Pinned TT-Metal source checkout")
    else()
        message(FATAL_ERROR "Set the runtime first: bash setup_tt_metal.sh && source env.sh")
    endif()
endif()
get_filename_component(TT_METAL_HOME "${TT_METAL_HOME}" ABSOLUTE)
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_tt_metal.py "${TT_METAL_HOME}"
    RESULT_VARIABLE _pin_status
    OUTPUT_VARIABLE _pin_output ERROR_VARIABLE _pin_error)
if(NOT _pin_status EQUAL 0)
    message(FATAL_ERROR "${_pin_output}${_pin_error}")
endif()
message(STATUS "${_pin_output}")

if(DEFINED ANN_METALIUM_MODE AND NOT ANN_METALIUM_MODE STREQUAL "external")
    message(FATAL_ERROR "Only the external runtime is supported. Reconfigure using bash build.sh.")
endif()
include(dependencies/CMakeLists.txt)

set(ANN_SOURCE_LAYOUT flat CACHE INTERNAL "Standalone ANN source layout")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
add_compile_definitions(TT_ENABLE_LIGHT_METAL_TRACE=$<BOOL:${TT_ENABLE_LIGHT_METAL_TRACE}>)
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>" "$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")
if(NOT ENABLE_LIBCXX)
    add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fsized-deallocation>")
endif()
# Match the original example directory's STL link and the internal includes
# used by the partitioned implementation. Paths all refer to the pinned tree.
link_libraries(TT::STL)
include_directories(
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/tt_metal/common"
    "${TT_METAL_HOME}/tt_metal/api"
    "${TT_METAL_HOME}/tt_metal/api/tt-metalium"
    "${TT_METAL_HOME}/tt_metal/impl/dispatch"
    "${TT_METAL_HOME}/tt_metal/llrt"
    "${TT_METAL_HOME}/tt_metal/hw/inc")
set(_ann_targets)
foreach(example IN LISTS ANN_EXAMPLES)
    add_subdirectory(${example})
    list(APPEND _ann_targets ${example})
endforeach()
add_custom_target(ann_examples DEPENDS ${_ann_targets})
