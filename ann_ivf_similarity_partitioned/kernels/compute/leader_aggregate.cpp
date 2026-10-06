#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "fine_common.hpp"
void kernel_main() {
    const uint32_t batches = get_arg_val<uint32_t>(0);
    const uint32_t profile_batch = get_arg_val<uint32_t>(1);
    transpose_wh_init(tt::CB::c_in0, tt::CB::c_out0);
    for (uint32_t b = 0; b < batches; ++b) {
        cb_wait_front(tt::CB::c_intermed5, 1);
        const uint32_t batch = read_tile_value(tt::CB::c_intermed5, 0, 0);
        const uint32_t workers = read_tile_value(tt::CB::c_intermed5, 0, 2);
        cb_pop_front(tt::CB::c_intermed5, 1);
        initialize_champion(tt::CB::c_intermed3, tt::CB::c_intermed4);
        for (uint32_t w = 0; w < workers; ++w) {
            auto wait_partial = [&] { cb_wait_front(tt::CB::c_in0, 1); cb_wait_front(tt::CB::c_in1, 1); };
            // Up to 63 workers can share an aggregator. Bound extra wait
            // scopes to 16, leaving room for all existing reduction scopes.
            if (batch == profile_batch && w < 16) {
                DeviceZoneScopedN("Partition partial wait"); wait_partial();
            } else wait_partial();
            if (batch == profile_batch) {
                DeviceZoneScopedN("Partition aggregate"); combine_page(tt::CB::c_in0, tt::CB::c_in1);
            } else combine_page(tt::CB::c_in0, tt::CB::c_in1);
        }
        transpose_output(tt::CB::c_intermed3, tt::CB::c_out0);
        transpose_output(tt::CB::c_intermed4, tt::CB::c_out1);
    }
}
