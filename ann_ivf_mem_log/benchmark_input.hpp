#pragma once

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ivf_benchmark_input {

inline std::string option_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc)
        throw std::invalid_argument(std::string(argv[index]) + " requires a value");
    return argv[++index];
}

inline uint32_t unsigned_option(int& index, int argc, char** argv) {
    const std::string option = argv[index];
    const std::string value = option_value(index, argc, argv);
    uint32_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument(option + " requires a nonnegative integer");
    return parsed;
}

inline std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char character : value)
        quoted += character == '\'' ? "'\\''" : std::string(1, character);
    return quoted + "'";
}

template <typename T>
std::vector<T> load_binary(const std::string& filename, uint32_t& rows, uint32_t& columns) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open " + filename);
    uint32_t n = 0, d = 0;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    file.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!file || !n || !d) throw std::runtime_error("Invalid or incomplete header in " + filename);
    const uint64_t count = uint64_t(n) * d;
    // Check the real payload before allocating and before narrowing read sizes.
    constexpr uint64_t header_bytes = 2 * sizeof(uint32_t);
    const uint64_t bytes = std::filesystem::file_size(filename);
    if (bytes < header_bytes || count > (bytes - header_bytes) / sizeof(T) ||
        count > std::vector<T>().max_size() ||
        count > uint64_t(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
        throw std::runtime_error("Invalid or incomplete payload in " + filename);
    std::vector<T> data(static_cast<size_t>(count));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!file) throw std::runtime_error("Could not read payload from " + filename);
    rows = n;
    columns = d;
    return data;
}

inline void validate_search(uint32_t nlist, uint32_t nprobe, uint32_t k,
                            uint32_t runs, uint32_t queries, uint32_t batch_size) {
    if (nlist < 32 || !nprobe || nprobe > 32 || nprobe > nlist || !k || k > 32)
        throw std::invalid_argument("Require nlist >= 32 and nprobe/k in 1..32");
    if (!runs || !queries) throw std::invalid_argument("--runs and --max_num_queries must be positive");
    if (batch_size != 32) throw std::invalid_argument("--batch_size must be 32");
}

inline void validate_dataset(uint32_t train_dim, uint32_t queries, uint32_t query_dim,
                             uint32_t neighbors, uint32_t neighbor_dim,
                             uint32_t distances, uint32_t distance_dim, uint32_t k) {
    if (!queries || query_dim != train_dim || neighbors < queries || distances < queries ||
        neighbor_dim < k || distance_dim < k)
        throw std::invalid_argument("Dataset query dimensions or ground-truth shape do not match the search");
}

}  // namespace ivf_benchmark_input
