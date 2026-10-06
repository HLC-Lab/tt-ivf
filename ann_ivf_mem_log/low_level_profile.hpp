#pragma once

#include <stdint.h>

// Execute the operation body exactly once. Low-level profiler instrumentation
// is currently disabled; the arguments remain for old call-site compatibility.
#define IVF_LOW_LEVEL_SCOPE(name, condition, ...) \
    do {                                           \
        __VA_ARGS__;                               \
    } while (false)

namespace ivf_low_level {
constexpr uint32_t batch = 0;
constexpr uint32_t worker_slice_count = 8;
constexpr uint32_t aggregator_partial_count = 3;

inline constexpr bool worker_slice(uint32_t batch_id, uint32_t slice) {
    return batch_id == batch && slice < worker_slice_count;
}

inline constexpr bool aggregator_partial(uint32_t batch_id, uint32_t partial) {
    return batch_id == batch && partial < aggregator_partial_count;
}
}  // namespace ivf_low_level
