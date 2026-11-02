#pragma once
#include <chrono>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ivf_energy {
// Buffer events until the measurement ends: no CSV I/O during initialization
// or search. Timestamps are Unix seconds for integration with host telemetry.
class EnergyEvents {
public:
    explicit EnergyEvents(const std::string& path) {
        if (path.empty()) return;
        output_.open(path);
        if (!output_) throw std::runtime_error("Cannot open energy event file: " + path);
        events_.reserve(32);
    }
    ~EnergyEvents() { flush(); }
    void reserve(size_t count) { if (output_.is_open()) events_.reserve(count); }
    void record(const std::string& name) {
        if (!output_.is_open()) return;
        const double timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        events_.emplace_back(name, timestamp);
    }
    void flush() {
        if (!output_.is_open()) return;
        output_ << "Event,Timestamp\n";
        for (const auto& [name, timestamp] : events_)
            output_ << name << ',' << std::fixed << std::setprecision(9) << timestamp << '\n';
        output_.close();
    }
private:
    std::ofstream output_;
    std::vector<std::pair<std::string, double>> events_;
};
}  // namespace ivf_energy
