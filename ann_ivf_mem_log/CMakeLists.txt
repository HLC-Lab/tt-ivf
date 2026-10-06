# ann_ivf_mem_log/CMakeLists.txt

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(ann_ivf_mem_log LANGUAGES CXX)
endif()
option(IVF_MEM_LOG_HOST_TESTS_ONLY "Build portable grouping/coarse lifecycle checks without Metalium" OFF)
if(IVF_MEM_LOG_HOST_TESTS_ONLY)
    enable_testing()
    add_executable(ann_ivf_mem_log_input_test tests/benchmark_input_test.cpp)
    target_compile_features(ann_ivf_mem_log_input_test PRIVATE cxx_std_20)
    target_compile_options(ann_ivf_mem_log_input_test PRIVATE -Wall -Wextra -Werror)
    add_test(NAME ivf_mem_log_input COMMAND ann_ivf_mem_log_input_test)
    add_executable(ann_ivf_mem_log_grouping_test tests/query_grouping_test.cpp)
    target_compile_features(ann_ivf_mem_log_grouping_test PRIVATE cxx_std_20)
    target_compile_options(ann_ivf_mem_log_grouping_test PRIVATE -Wall -Wextra -Werror)
    add_test(NAME ivf_mem_log_grouping COMMAND ann_ivf_mem_log_grouping_test)
    set(COARSE_MODEL_INCLUDES "${CMAKE_CURRENT_BINARY_DIR}/coarse_model_includes")
    file(MAKE_DIRECTORY "${COARSE_MODEL_INCLUDES}/compute_kernel_api")
    file(WRITE "${COARSE_MODEL_INCLUDES}/compute_kernel_api.h" "#pragma once\n")
    foreach(api matmul tile_move_copy transpose_wh reconfig_data_format pack)
        file(WRITE "${COARSE_MODEL_INCLUDES}/compute_kernel_api/${api}.h" "#pragma once\n")
    endforeach()
    add_executable(ann_ivf_mem_log_coarse_state_test tests/coarse_state_test.cpp)
    target_compile_features(ann_ivf_mem_log_coarse_state_test PRIVATE cxx_std_20)
    target_compile_options(ann_ivf_mem_log_coarse_state_test PRIVATE -Wall -Wextra -Werror)
    target_include_directories(ann_ivf_mem_log_coarse_state_test PRIVATE "${COARSE_MODEL_INCLUDES}")
    add_test(NAME ivf_mem_log_coarse_state COMMAND ann_ivf_mem_log_coarse_state_test)
    return()
endif()

add_executable(ann_ivf_mem_log)
target_compile_features(ann_ivf_mem_log PRIVATE cxx_std_20)
target_sources(
    ann_ivf_mem_log
    PRIVATE
        main.cpp
        ivf_tt.cpp
)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(ann_ivf_mem_log PRIVATE TT::Metalium)
