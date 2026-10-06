#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include "ivf_tt.hpp"
#include "../common/benchmark_paths.hpp"
namespace {
constexpr const char* configuration_columns =
    "run,nlist,nprobe,queries,k,grouping,partitions,reader,bank_schedule,aggregation,mask,layout,query_broadcast,leader_prefetch,worker_input,grouping_window,grouping_lookahead";

template<class Function> void for_each_timing_field(Function function) {
    for (const auto& field : ivf_partitioned::timing_stages) function(field);
    for (const auto& field : ivf_partitioned::timing_details) function(field);
    for (const auto& field : ivf_partitioned::timing_totals) function(field);
}

std::ofstream open_timing_csv(const std::filesystem::path& path) {
    std::ostringstream header;
    header << configuration_columns << ",qps,recall";
    for_each_timing_field([&](const auto& field) { header << ',' << field.name; });
    const bool exists = std::filesystem::exists(path) && std::filesystem::file_size(path);
    if (exists) {
        std::ifstream previous(path);
        std::string line;
        if (!std::getline(previous, line)) throw std::runtime_error("Could not read timing CSV header: " + path.string());
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line != header.str()) throw std::runtime_error("Timing CSV header differs; choose a fresh path: " + path.string());
    }
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("Could not open timing CSV: " + path.string());
    if (!exists) output << header.str() << '\n';
    output << std::setprecision(12);
    return output;
}

uint32_t number(const std::string& text) {
    uint32_t value;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("Expected unsigned integer: " + text);
    return value;
}
template<class T> std::vector<T> load(const std::string& path, uint32_t& rows, uint32_t& columns) {
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(&rows), 4) || !file.read(reinterpret_cast<char*>(&columns), 4) || !rows || !columns)
        throw std::runtime_error("Missing or invalid binary dataset: " + path);
    const uint64_t elements = uint64_t(rows) * columns;
    const uint64_t file_bytes = std::filesystem::file_size(path);
    if (file_bytes < 8 || (file_bytes - 8) % sizeof(T) || elements != (file_bytes - 8) / sizeof(T))
        throw std::runtime_error("Truncated or unexpected binary size: " + path);
    if (elements > std::numeric_limits<size_t>::max() / sizeof(T) ||
        elements > uint64_t(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
        throw std::runtime_error("Binary dataset exceeds the host size limit: " + path);
    const uint64_t bytes = elements * sizeof(T);
    std::vector<T> data(elements);
    if (!file.read(reinterpret_cast<char*>(data.data()), bytes)) throw std::runtime_error("Could not read " + path);
    return data;
}
void usage() {
    std::cout << "ann_ivf_similarity_partitioned --dataset glove-100-angular --nlist 512 --nprobe 8 --k 10\n"
        "  --max_num_queries 10000 --runs 3 --mem_log host.csv\n"
        "  --query-grouping none|primary-list|weighted-union (default weighted-union)\n"
        "  --grouping-window 0 --grouping-lookahead 256\n"
        "  --partitions 8 --partition-layout rows|columns|compact\n"
        "  --candidate-reader direct|direct-serial|relay --bank-schedule fifo|staggered\n"
        "  --leader-prefetch-pages 8 (1..15) --worker-input-pages 2\n"
        "  --query-broadcast auto|unicast --aggregation leader|separate\n"
        "  --candidate-mask auto|additive|off --profile-batch N\n"
        "  --organization global|partitioned --diagnostics DIRECTORY\n"
        "  --results-csv PATH --timings-csv PATH --results-output PATH --skip-warmup\n"
        "Timings default to <results-csv stem>_timings.csv; measured runs only, durations in us.\n"
        "Direct readers require FIFO. Direct uses up to min(worker-input-pages,15) outstanding page pairs;\n"
        "direct-serial retains the per-page full read barrier for comparisons.\n"
        "Global organization uses one leader and all remaining workers.\n";
}
}
int main(int argc, char** argv) {
    try {
        PartitionConfig config;
        std::string dataset = "glove-100-angular", log;
        std::string results_csv, results_output, timings_csv;
        std::string organization = "partitioned";
        uint32_t nlist = 512, nprobe = 8, k = 10, maximum_queries = 10000, runs = 1;
        bool warmup = true, explicit_bank = false, explicit_log = false, explicit_results = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--skip-warmup") { warmup = false; continue; }
            if (i + 1 == argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--dataset") dataset = value;
            else if (arg == "--nlist") nlist = number(value);
            else if (arg == "--nprobe") nprobe = number(value);
            else if (arg == "--k") k = number(value);
            else if (arg == "--max_num_queries" || arg == "--max-num-queries") maximum_queries = number(value);
            else if (arg == "--runs") runs = number(value);
            else if (arg == "--mem_log" || arg == "--mem-log") { log = value; explicit_log = true; }
            else if (arg == "--query-grouping") config.grouping = value;
            else if (arg == "--grouping-window") config.grouping_window = number(value);
            else if (arg == "--grouping-lookahead") config.grouping_lookahead = number(value);
            else if (arg == "--partitions") config.partitions = number(value);
            else if (arg == "--partition-layout") config.layout = value;
            else if (arg == "--candidate-reader") config.reader = value;
            else if (arg == "--bank-schedule") { config.bank_schedule = value; explicit_bank = true; }
            else if (arg == "--leader-prefetch-pages") config.leader_prefetch = number(value);
            else if (arg == "--worker-input-pages" || arg == "--prefetch-reader") config.worker_input = number(value);
            else if (arg == "--aggregation") config.aggregation = value;
            else if (arg == "--query-broadcast") config.query_broadcast = value;
            else if (arg == "--profile-batch") config.profile_batch = number(value);
            else if (arg == "--organization") organization = value;
            else if (arg == "--diagnostics") config.diagnostics = value;
            else if (arg == "--results-csv") { results_csv = value; explicit_results = true; }
            else if (arg == "--timings-csv") timings_csv = value;
            else if (arg == "--results-output") results_output = value;
            else if (arg == "--result-staging") {
                if (value != "dram") throw std::invalid_argument("This example uses DRAM result staging");
            } else if (arg == "--candidate-mask") {
                if (value == "auto") config.mask = ivf_partitioned::MaskMode::Auto;
                else if (value == "additive") config.mask = ivf_partitioned::MaskMode::Additive;
                else if (value == "off") config.mask = ivf_partitioned::MaskMode::Off;
                else throw std::invalid_argument("Candidate mask must be auto, additive, or off");
            } else throw std::invalid_argument("Unknown argument: " + arg);
        }
        if (!runs || !maximum_queries || !k || k > 32) throw std::invalid_argument("Runs/queries must be positive; k must be 1..32");
        if (organization == "global") config.partitions = 1;
        else if (organization != "partitioned") throw std::invalid_argument("Organization must be global or partitioned");
        if (config.reader != "relay" && !explicit_bank) config.bank_schedule = "fifo";
        if (config.grouping != "none" && config.grouping != "primary-list" && config.grouping != "weighted-union")
            throw std::invalid_argument("Unknown grouping mode");
        if (!config.grouping_lookahead || (config.grouping_window && config.grouping_window % 32))
            throw std::invalid_argument("Grouping lookahead must be positive; window must be 0 or divisible by 32");
        if (!explicit_log) log = ann_paths::result("similarity_partitioned", "host.csv");
        if (!explicit_results) results_csv = ann_paths::result("similarity_partitioned", "results.csv");
        if (timings_csv.empty()) {
            const std::filesystem::path result_path(results_csv);
            timings_csv = (result_path.parent_path() / (result_path.stem().string() + "_timings.csv")).string();
        }
        const auto timing_path = std::filesystem::absolute(timings_csv).lexically_normal();
        for (const auto& other : {results_csv, log, results_output})
            if (!other.empty() && timing_path == std::filesystem::absolute(other).lexically_normal())
                throw std::invalid_argument("Timing CSV must use a different path from result and memory logs");
        uint32_t train_n, train_d, query_n, query_d, gt_n, gt_d;
        const auto train = load<float>(ann_paths::dataset(dataset, "_train.bin"), train_n, train_d);
        const auto query = load<float>(ann_paths::dataset(dataset, "_queries.bin"), query_n, query_d);
        const auto gt = load<uint32_t>(ann_paths::dataset(dataset, "_neighbors.bin"), gt_n, gt_d);
        const uint32_t query_count = std::min({query_n, maximum_queries, 10000u});
        if (train_d != query_d || gt_n < query_count || gt_d < k) throw std::invalid_argument("Dataset/query/ground-truth dimensions disagree");
        const std::string centroid_path = ann_paths::centroids(dataset, nlist, ".bin");
        std::cout << "[Config] grouping=" << config.grouping << ", partitions=" << config.partitions
                  << ", reader=" << config.reader << ", bank schedule=" << config.bank_schedule
                  << ", mask=" << static_cast<uint32_t>(config.mask) << ", fidelity=hifi4, IDs=int32, result staging=dram\n";
        if (config.reader == "direct" || config.reader == "direct-serial")
            std::cout << "[Config] Worker input capacity: " << config.worker_input
                      << " page pairs; direct DMA window: "
                      << (config.reader == "direct" ? std::min(config.worker_input, 15u) : 1u) << "\n";
        else if (config.reader == "relay") {
            std::cout << "[Config] Relay: leaders read queries, vectors and IDs from DRAM; workers receive them over NoC\n"
                      << "[Config] Leader staging: " << config.leader_prefetch
                      << " page pairs; worker input capacity: " << config.worker_input << " page pairs\n"
                      << "[Config] Partitions advance independently; descriptor reads and result staging still use DRAM\n";
        }
        IVF_TT index(train_d, nlist, nprobe, log, config);
        index.set_memory_log_context("index_build"); index.create_index(train, centroid_path);
        index.set_memory_log_context("index_load"); index.load_to_device();
        if (warmup) {
            std::cout << "Warming up with " << std::min(query_count, 32u) << " queries...\n";
            index.set_memory_log_context("warmup"); index.search_all(query.data(), std::min(query_count, 32u), k);
        }
        const bool have_results = std::filesystem::exists(results_csv) && std::filesystem::file_size(results_csv);
        std::ofstream summary(results_csv, std::ios::app);
        if (!summary) throw std::runtime_error("Could not open results CSV");
        if (!have_results) summary << configuration_columns << ",latency_us,qps,recall\n";
        auto timings = open_timing_csv(timings_csv);
        auto write_configuration = [&](std::ostream& output, uint64_t run) {
            output << run << ',' << nlist << ',' << nprobe << ',' << query_count << ',' << k << ',' << config.grouping << ','
                   << config.partitions << ',' << config.reader << ',' << config.bank_schedule << ',' << config.aggregation << ','
                   << static_cast<uint32_t>(config.mask) << ',' << config.layout << ',' << config.query_broadcast << ','
                   << config.leader_prefetch << ',' << config.worker_input << ',' << config.grouping_window << ',' << config.grouping_lookahead;
        };
        double latency_sum = 0, recall_sum = 0;
        for (uint64_t run = 1; run <= runs; ++run) {
            index.set_memory_log_context("run_" + std::to_string(run));
            const auto result = index.search_all(query.data(), query_count, k);
            const double duration = index.last_elapsed_us();
            uint64_t hits = 0;
            for (uint32_t q = 0; q < query_count; ++q) {
                std::set<int64_t> expected, seen;
                for (uint32_t lane = 0; lane < k; ++lane) expected.insert(gt[q * gt_d + lane]);
                for (uint32_t lane = 0; lane < k; ++lane)
                    if (seen.insert(result.second[q * k + lane]).second && expected.count(result.second[q * k + lane])) ++hits;
            }
            const double recall = double(hits) / (query_count * k), qps = query_count * 1e6 / duration;
            latency_sum += duration; recall_sum += recall;
            write_configuration(summary, run);
            summary << ',' << std::setprecision(12) << duration << ',' << qps << ',' << recall << '\n';
            summary.flush();
            if (!summary) throw std::runtime_error("Could not finish writing results CSV");
            write_configuration(timings, run);
            timings << ',' << qps << ',' << recall;
            const auto& measured = index.last_timings();
            for_each_timing_field([&](const auto& field) { timings << ',' << measured.*(field.value); });
            timings << '\n';
            timings.flush();
            if (!timings) throw std::runtime_error("Could not finish writing timing CSV");
            std::cout << "[Run " << run << "] QPS=" << qps << ", Recall@" << k << '=' << recall << '\n';
            if (!results_output.empty()) {
                std::ofstream out(results_output);
                if (!out) throw std::runtime_error("Could not open result dump");
                out << "query,rank,index,score\n";
                out << std::setprecision(9);
                for (uint32_t q = 0; q < query_count; ++q) for (uint32_t lane = 0; lane < k; ++lane)
                    out << q << ',' << lane << ',' << result.second[q * k + lane] << ',' << result.first[q * k + lane] << '\n';
                out.close();
                if (!out) throw std::runtime_error("Could not finish writing result dump");
            }
        }
        index.flush_memory_log();
        timings.close();
        if (!timings) throw std::runtime_error("Could not close timing CSV");
        std::cout << "[Timing] Per-run timing CSV (us): " << timings_csv << '\n';
        std::cout << "Average QPS=" << double(query_count) * runs * 1e6 / latency_sum
                  << ", average Recall@" << k << '=' << recall_sum / runs << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "[Error] " << error.what() << '\n'; return 1; }
}
