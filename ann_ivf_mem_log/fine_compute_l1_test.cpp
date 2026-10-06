#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace {

enum class ComputeThread { Unpack, Math, Pack };

ComputeThread current_thread;
std::array<uint16_t, 1024> score_tile;
std::array<uint32_t, 1024> index_tile;
uint32_t publications;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void cb_reserve_back(uint32_t, uint32_t) {}

uintptr_t get_tile_address(uint32_t cb, uint32_t tile) {
    require(tile == 0 && cb < 2, "invalid test CB address");
    return cb == 0 ? reinterpret_cast<uintptr_t>(score_tile.data())
                   : reinterpret_cast<uintptr_t>(index_tile.data());
}

void cb_push_back(uint32_t cb, uint32_t count) {
    if (current_thread != ComputeThread::Pack) {
        return;
    }
    require(count == 1, "invalid test publication");
    if (cb == 0) {
        require(
            std::all_of(score_tile.begin(), score_tile.end(), [](uint16_t value) { return value == 0xC61C; }),
            "score tile published before initialization completed");
    } else {
        require(
            std::all_of(
                index_tile.begin(), index_tile.end(), [](uint32_t value) { return value == 0xFFFFFFFFu; }),
            "index tile published before initialization completed");
    }
    ++publications;
}

}  // namespace

#define PACK(...) \
    do { \
        if (current_thread == ComputeThread::Pack) { \
            __VA_ARGS__ \
        } \
    } while (false)
#define UNPACK(...) \
    do { \
        if (current_thread == ComputeThread::Unpack) { \
            __VA_ARGS__ \
        } \
    } while (false)
#include "kernels/compute/ivf_compute_l1.hpp"
#undef PACK
#undef UNPACK

int main() {
    for (const auto thread : {ComputeThread::Unpack, ComputeThread::Math, ComputeThread::Pack}) {
        current_thread = thread;
        score_tile.fill(0xBF00);
        index_tile.fill(16777217u);
        publications = 0;
        initialize_champion(0, 1);

        for (uint32_t i = 0; i < 1024; ++i) {
            require(
                score_tile[i] == (thread == ComputeThread::Pack ? 0xC61C : 0xBF00),
                "winner scores initialized by the wrong compute thread");
            require(
                index_tile[i] == (thread == ComputeThread::Pack ? 0xFFFFFFFFu : 16777217u),
                "winner IDs initialized by the wrong compute thread");
        }
        require(
            publications == (thread == ComputeThread::Pack ? 2u : 0u),
            "winner publication has the wrong owner");

        for (uint32_t valid_lanes : {0u, 1u, 17u, 31u, 32u}) {
            score_tile.fill(0xBF00);
            index_tile.fill(16777217u);
            mask_fine_candidates(0, 1, 16777216u, (1u << 1) | (1u << 31), 16777216u + valid_lanes);
            for (uint32_t row = 0; row < 32; ++row) {
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const bool valid = (row == 1 || row == 31) && lane < valid_lanes;
                    const bool masked = thread == ComputeThread::Unpack && !valid;
                    const uint32_t offset = fine_tile_offset(row, lane);
                    require(
                        score_tile[offset] == (masked ? 0xC61C : 0xBF00),
                        "score mask has the wrong owner or lane");
                    require(
                        index_tile[offset] == (masked ? 0xFFFFFFFFu : 16777217u),
                        "score and Int32 ID masks disagree");
                }
            }
        }
    }
}
