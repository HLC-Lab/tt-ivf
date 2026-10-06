#include "../pipeline_timings.hpp"
#include <iostream>
#include <limits>

using ivf_partitioned::SearchTimings;

namespace {
SearchTimings example() {
    SearchTimings times;
    times.h2d_us = 40;
    times.coarse_query_h2d_us = 10;
    times.fine_query_h2d_us = 12;
    times.descriptor_h2d_us = 8;
    times.control_h2d_us = 10;
    times.cpu_coarse_config_us = 50;
    times.query_normalize_us = 20;
    times.coarse_runtime_setup_us = 10;
    times.tt_coarse_search_us = 60;
    times.cpu_fine_prep_us = 100;
    times.coarse_readback_us = 10;
    times.coarse_decode_us = 10;
    times.list_metadata_us = 5;
    times.query_grouping_us = 20;
    times.query_reorder_us = 10;
    times.partition_schedule_us = 10;
    times.descriptor_pack_us = 5;
    times.tt_fine_search_us = 200;
    times.cpu_output_us = 30;
    times.final_readback_us = 10;
    times.pipeline_us = 480;
    return times;
}

void require(bool ok) {
    if (!ok) throw std::runtime_error("Timing conservation check failed");
}

template<class Function> void rejects(Function change) {
    auto times = example();
    change(times);
    try { times.finish(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("Invalid timing was accepted");
}
}  // namespace

int main() {
    auto times = example();
    times.finish();
    require(times.stage_sum_us == times.pipeline_us);
    require(times.coarse_host_setup_us == 20 && times.fine_other_prep_us == 30 && times.output_restore_us == 20);
    rejects([](auto& t) { t.pipeline_us += 1; });
    rejects([](auto& t) { t.descriptor_h2d_us += 1; });
    rejects([](auto& t) { t.query_grouping_us = 500; });
    rejects([](auto& t) { t.final_readback_us = 50; });
    rejects([](auto& t) { t.query_normalize_us = -1; });
    rejects([](auto& t) { t.query_grouping_us = std::numeric_limits<double>::quiet_NaN(); });
    rejects([](auto& t) { t.pipeline_us = std::numeric_limits<double>::infinity(); });
    std::cout << "Timing stages and included details conserve pipeline time\n";
}
