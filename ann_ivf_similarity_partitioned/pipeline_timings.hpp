#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ivf_partitioned {

// Host wall-clock durations in microseconds for one search_all invocation.
// Detail fields partition their parent stage; they must not be added again.
struct SearchTimings {
    double h2d_us = 0, cpu_coarse_config_us = 0, tt_coarse_search_us = 0;
    double cpu_fine_prep_us = 0, tt_fine_search_us = 0, cpu_output_us = 0;
    double stage_sum_us = 0, pipeline_us = 0;
    double query_normalize_us = 0, coarse_host_setup_us = 0, coarse_runtime_setup_us = 0;
    double coarse_query_h2d_us = 0, coarse_readback_us = 0, coarse_decode_us = 0;
    double list_metadata_us = 0, query_grouping_us = 0, query_reorder_us = 0;
    double fine_query_h2d_us = 0, partition_schedule_us = 0, descriptor_pack_us = 0;
    double descriptor_h2d_us = 0, control_h2d_us = 0, fine_other_prep_us = 0;
    double final_readback_us = 0, output_restore_us = 0;

    // Called after the pipeline timer stops. Residuals include allocations and
    // existing in-memory bookkeeping between explicitly measured operations.
    void finish();
};

struct TimingField {
    const char* name;
    const char* label;
    const char* parent;
    double SearchTimings::* value;
};

inline constexpr std::array timing_stages{
    TimingField{"h2d_us", "H2D", "", &SearchTimings::h2d_us},
    TimingField{"cpu_coarse_config_us", "CPU coarse search config", "", &SearchTimings::cpu_coarse_config_us},
    TimingField{"tt_coarse_search_us", "TT coarse search", "", &SearchTimings::tt_coarse_search_us},
    TimingField{"cpu_fine_prep_us", "CPU fine search prep", "", &SearchTimings::cpu_fine_prep_us},
    TimingField{"tt_fine_search_us", "TT fine search", "", &SearchTimings::tt_fine_search_us},
    TimingField{"cpu_output_us", "CPU output", "", &SearchTimings::cpu_output_us},
};

inline constexpr std::array timing_details{
    TimingField{"coarse_query_h2d_us", "Original query upload", "h2d_us", &SearchTimings::coarse_query_h2d_us},
    TimingField{"fine_query_h2d_us", "Reordered query upload", "h2d_us", &SearchTimings::fine_query_h2d_us},
    TimingField{"descriptor_h2d_us", "Descriptor upload", "h2d_us", &SearchTimings::descriptor_h2d_us},
    TimingField{"control_h2d_us", "L1 control initialization", "h2d_us", &SearchTimings::control_h2d_us},
    TimingField{"query_normalize_us", "Query normalization", "cpu_coarse_config_us", &SearchTimings::query_normalize_us},
    TimingField{"coarse_host_setup_us", "Coarse buffers / tiling / program setup", "cpu_coarse_config_us", &SearchTimings::coarse_host_setup_us},
    TimingField{"coarse_runtime_setup_us", "Coarse runtime arguments / workload", "cpu_coarse_config_us", &SearchTimings::coarse_runtime_setup_us},
    TimingField{"coarse_readback_us", "Coarse results readback", "cpu_fine_prep_us", &SearchTimings::coarse_readback_us},
    TimingField{"coarse_decode_us", "Coarse untilize / list selection", "cpu_fine_prep_us", &SearchTimings::coarse_decode_us},
    TimingField{"list_metadata_us", "List metadata", "cpu_fine_prep_us", &SearchTimings::list_metadata_us},
    TimingField{"query_grouping_us", "Query similarity grouping", "cpu_fine_prep_us", &SearchTimings::query_grouping_us},
    TimingField{"query_reorder_us", "Query reordering / tiling", "cpu_fine_prep_us", &SearchTimings::query_reorder_us},
    TimingField{"partition_schedule_us", "Partition / worker scheduling", "cpu_fine_prep_us", &SearchTimings::partition_schedule_us},
    TimingField{"descriptor_pack_us", "Descriptor packing", "cpu_fine_prep_us", &SearchTimings::descriptor_pack_us},
    TimingField{"fine_other_prep_us", "Fine buffers / program setup / misc", "cpu_fine_prep_us", &SearchTimings::fine_other_prep_us},
    TimingField{"final_readback_us", "Final results readback", "cpu_output_us", &SearchTimings::final_readback_us},
    TimingField{"output_restore_us", "Output untilize / original order", "cpu_output_us", &SearchTimings::output_restore_us},
};

inline constexpr std::array timing_totals{
    TimingField{"stage_sum_us", "Stage sum", "", &SearchTimings::stage_sum_us},
    TimingField{"pipeline_us", "End-to-end", "", &SearchTimings::pipeline_us},
};

inline void SearchTimings::finish() {
    coarse_host_setup_us = cpu_coarse_config_us - query_normalize_us - coarse_runtime_setup_us;
    fine_other_prep_us = cpu_fine_prep_us - coarse_readback_us - coarse_decode_us - list_metadata_us
        - query_grouping_us - query_reorder_us - partition_schedule_us - descriptor_pack_us;
    output_restore_us = cpu_output_us - final_readback_us;
    stage_sum_us = 0;
    auto check = [](const auto& fields, const SearchTimings& times) {
        for (const auto& field : fields) {
            const double value = times.*(field.value);
            if (!std::isfinite(value) || value < 0)
                throw std::runtime_error(std::string("Invalid search timing: ") + field.name);
        }
    };
    check(timing_stages, *this);
    check(timing_details, *this);
    for (const auto& field : timing_stages) stage_sum_us += this->*(field.value);
    check(timing_totals, *this);
    const double tolerance = 1e-6 * std::max(1.0, pipeline_us);
    if (std::abs(stage_sum_us - pipeline_us) > tolerance)
        throw std::runtime_error("Search stages do not sum to pipeline time");
    if (std::abs(h2d_us - coarse_query_h2d_us - fine_query_h2d_us - descriptor_h2d_us - control_h2d_us) > tolerance)
        throw std::runtime_error("Search transfer details do not sum to H2D time");
}

}  // namespace ivf_partitioned
