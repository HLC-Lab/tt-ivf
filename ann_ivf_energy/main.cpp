#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ivf_tt.hpp"
#include "../common/benchmark_paths.hpp"
#include "energy_events.hpp"

namespace {

struct Config {
    std::string dataset = "glove-100-angular";
    uint32_t nlist = 1024;
    uint32_t nprobe = 32;
    uint32_t k = 10;
    uint32_t max_num_queries = 10000;
    uint32_t runs = 10;
    std::string result_staging = "dram";
    uint32_t cluster_chunk_blocks = 0;
    ivf_energy::GroupingConfig query_grouping;
    std::string energy_events;
};

class BinaryLoader {
public:
    template <typename T>
    static std::vector<T> load(const std::string& filename, uint32_t& rows, uint32_t& columns) {
        if (!std::filesystem::exists(filename)) {
            throw std::runtime_error("File " + filename + " not found");
        }

        std::ifstream file(filename, std::ios::binary);
        file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&columns), sizeof(uint32_t));
        if (!file) {
            throw std::runtime_error("Could not read header from " + filename);
        }

        const size_t element_count = static_cast<size_t>(rows) * columns;
        std::vector<T> data(element_count);
        file.read(reinterpret_cast<char*>(data.data()), element_count * sizeof(T));
        if (!file) {
            throw std::runtime_error("Could not read payload from " + filename);
        }

        std::cout << "[Loader] " << filename << " (N=" << rows << ", D=" << columns << ")\n";
        return data;
    }

    static void convert_hdf5_if_needed(const std::string& dataset) {
        const std::string hdf5_file = ann_paths::dataset(dataset, ".hdf5");
        const std::string train_file = ann_paths::dataset(dataset, "_train.bin");
        if (!std::filesystem::exists(train_file) && std::filesystem::exists(hdf5_file)) {
            const std::string command =
                ann_paths::converter("export_hdf5.py", hdf5_file);
            if (std::system(command.c_str()) != 0) {
                throw std::runtime_error("Dataset conversion failed");
            }
        }
    }
};

std::string next_value(int& index, int argc, char** argv, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(option + " requires a value");
    }
    return argv[++index];
}

uint32_t next_uint(int& index, int argc, char** argv, const std::string& option) {
    const auto text = next_value(index, argc, argv, option);
    uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument(option + " requires a nonnegative 32-bit integer");
    return value;
}

Config parse_config(int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--dataset") {
            config.dataset = next_value(index, argc, argv, argument);
        } else if (argument == "--nlist") {
            config.nlist = next_uint(index, argc, argv, argument);
        } else if (argument == "--nprobe") {
            config.nprobe = next_uint(index, argc, argv, argument);
        } else if (argument == "--k") {
            config.k = next_uint(index, argc, argv, argument);
        } else if (argument == "--max_num_queries" || argument == "--max-num-queries") {
            config.max_num_queries = next_uint(index, argc, argv, argument);
        } else if (argument == "--runs") {
            config.runs = next_uint(index, argc, argv, argument);
        } else if (argument == "--result-staging" || argument == "--result_staging") {
            config.result_staging = next_value(index, argc, argv, argument);
        } else if (argument == "--cluster-chunk-blocks" || argument == "--cluster_chunk_blocks") {
            config.cluster_chunk_blocks = next_uint(index, argc, argv, argument);
        } else if (argument == "--query-grouping") {
            config.query_grouping.mode = next_value(index, argc, argv, argument);
        } else if (argument == "--grouping-window") {
            config.query_grouping.window = next_uint(index, argc, argv, argument);
        } else if (argument == "--grouping-lookahead") {
            config.query_grouping.lookahead = next_uint(index, argc, argv, argument);
        } else if (argument == "--energy-events") {
            config.energy_events = next_value(index, argc, argv, argument);
        } else if (argument == "--batch_size" || argument == "--batch-size") {
            const uint32_t requested = next_uint(index, argc, argv, argument);
            if (requested != 32) {
                throw std::invalid_argument("The hardware batch size is fixed at 32");
            }
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: ann_ivf_energy [--dataset NAME] [--nlist N] [--nprobe N] [--k N] "
                << "[--max_num_queries N] [--runs N] [--result-staging dram|core0-l1] "
                << "[--query-grouping none|primary-list|weighted-union] "
                << "[--grouping-window N] [--grouping-lookahead N] [--energy-events PATH]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + argument);
        }
    }

    if (config.runs == 0) {
        throw std::invalid_argument("--runs must be greater than zero");
    }
    if (config.max_num_queries == 0) {
        throw std::invalid_argument("--max_num_queries must be greater than zero");
    }
    ivf_energy::validate_grouping_config(config.query_grouping);
    if (config.cluster_chunk_blocks != 0)
        throw std::invalid_argument("Cluster chunking was removed; omit --cluster-chunk-blocks or use 0");
    return config;
}

std::string find_centroids(const Config& config) {
    const std::string npy_path =
        ann_paths::centroids(config.dataset, config.nlist, ".npy");
    const std::string binary_path = std::filesystem::path(npy_path).replace_extension(".bin").string();

    bool conversion_needed = !std::filesystem::exists(binary_path);
    if (!conversion_needed && std::filesystem::exists(npy_path)) {
        conversion_needed =
            std::filesystem::last_write_time(binary_path) < std::filesystem::last_write_time(npy_path);
    }
    if (conversion_needed) {
        if (!std::filesystem::exists(npy_path)) {
            throw std::runtime_error("Centroid file not found: " + npy_path);
        }
        const std::string command =
            ann_paths::converter("convert_centroids.py", npy_path);
        if (std::system(command.c_str()) != 0) {
            throw std::runtime_error("Centroid conversion failed");
        }
    }
    return binary_path;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_config(argc, argv);

        FineResultStaging result_staging;
        if (config.result_staging == "dram") {
            result_staging = FineResultStaging::Dram;
        } else if (config.result_staging == "core0-l1") {
            result_staging = FineResultStaging::Core0L1;
        } else {
            throw std::invalid_argument("--result-staging must be 'dram' or 'core0-l1'");
        }

        std::cout << "\n=== Tenstorrent ANN IVF Energy Workload ===\n"
                  << "[Config] Dataset: " << config.dataset << '\n'
                  << "[Config] NList/NProbe/K: " << config.nlist << '/' << config.nprobe << '/'
                  << config.k << '\n'
                  << "[Config] Result staging: " << config.result_staging << '\n'
                  << "[Config] Query grouping: " << config.query_grouping.mode
                  << "; window=" << config.query_grouping.window
                  << "; lookahead=" << config.query_grouping.lookahead << '\n'
                  << "[Config] Runs: " << config.runs << '\n';

        BinaryLoader::convert_hdf5_if_needed(config.dataset);
        uint32_t train_count = 0;
        uint32_t train_dim = 0;
        uint32_t query_count = 0;
        uint32_t query_dim = 0;
        auto training =
            BinaryLoader::load<float>(ann_paths::dataset(config.dataset, "_train.bin"), train_count, train_dim);
        auto queries =
            BinaryLoader::load<float>(ann_paths::dataset(config.dataset, "_queries.bin"), query_count, query_dim);
        if (train_dim != query_dim) {
            throw std::runtime_error("Training and query dimensions differ");
        }

        const std::string centroids = find_centroids(config);
        const uint32_t queries_per_run = std::min(query_count, config.max_num_queries);
        if (!queries_per_run) throw std::runtime_error("Query dataset is empty");
        ivf_energy::EnergyEvents events(config.energy_events);
        events.reserve(2 * static_cast<size_t>(config.runs) + 4);
        const auto measurement_start = std::chrono::steady_clock::now();
        events.record("measurement_start");
        events.record("initialization_start");
        std::cout << "[Energy] ENERGY_MEASUREMENT_START runs=" << config.runs
                  << " queries_per_run=" << queries_per_run << std::endl;

        IVF_TT index(
            0,
            train_dim,
            config.nlist,
            config.nprobe,
            result_staging,
            config.cluster_chunk_blocks,
            config.query_grouping);
        index.create_index(training, centroids);
        index.load_to_device();
        events.record("initialization_end");

        double search_seconds = 0.0;
        for (uint32_t run = 0; run < config.runs; ++run) {
            std::cout << "--- Run " << run + 1 << " / " << config.runs << " ---" << std::endl;
            events.record("run_" + std::to_string(run + 1) + "_start");
            const auto start = std::chrono::steady_clock::now();
            index.search_all(queries.data(), queries_per_run, config.k);
            const auto end = std::chrono::steady_clock::now();
            events.record("run_" + std::to_string(run + 1) + "_end");
            const double run_seconds = std::chrono::duration<double>(end - start).count();
            search_seconds += run_seconds;
            std::cout << "[Energy] Run " << run + 1 << " completed in " << std::fixed
                      << std::setprecision(6) << run_seconds << " s" << std::endl;
        }

        const auto measurement_end = std::chrono::steady_clock::now();
        events.record("measurement_end");
        events.flush();
        std::cout << "[Energy] ENERGY_MEASUREMENT_END" << std::endl;
        const double measured_seconds =
            std::chrono::duration<double>(measurement_end - measurement_start).count();
        const uint64_t total_queries = static_cast<uint64_t>(queries_per_run) * config.runs;
        std::cout << "[Energy] Initialization + search time: " << std::fixed
                  << std::setprecision(6) << measured_seconds << " s\n"
                  << "[Energy] Search-loop time: " << search_seconds << " s\n"
                  << "[Energy] Total measured queries: " << total_queries << '\n'
                  << "[Energy] Throughput: " << std::setprecision(2)
                  << static_cast<double>(total_queries) / search_seconds << " queries/s\n";
    } catch (const std::exception& error) {
        std::cerr << "[Error] " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
