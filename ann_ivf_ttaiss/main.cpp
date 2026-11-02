// ann_ivf_ttaiss/main.cpp

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <charconv>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include "ivf_tt.hpp"
#include "../common/benchmark_paths.hpp"
#include "benchmark_input.hpp"
#include "energy_events.hpp"

struct Config {
    std::string run_id = "0";
    std::string dataset = "glove-100-angular";
    uint32_t nlist = 1024;
    uint32_t nprobe = 32;
    uint32_t k = 10;
    uint32_t batch_size = 32;
    uint32_t dim = 0;
    uint32_t num_vectors = 0;
    uint32_t max_num_queries = 10000;
    uint32_t runs = 1;
    std::string mem_log;
    bool explicit_mem_log = false;
    std::string results_output;
    uint32_t prefetch_reader = 2;
    uint32_t batch_chunk = 32;
    std::string fine_fidelity = "HiFi4";
    std::string coarse_fidelity = "HiFi4";
    std::string energy_events;
    std::string index_build = "fast";
    std::string selections_output, assignment_output;
    bool index_digest = false;
    bool rerank = false;
    bool ablation_rank32 = false, ablation_rebuild = false;
    std::string first_pass_lists = "auto";  // auto | 0 (single pass) | lists in the first pass
    ivf_cluster_major::GroupingConfig query_grouping;
};

using ivf_benchmark_input::option_value;
using ivf_benchmark_input::unsigned_option;

class HDF5BinaryLoader {
public:
    template<typename T>
    static std::vector<T> load_generic(const std::string& filename, uint32_t& n, uint32_t& d) {
        auto data = ivf_benchmark_input::load_binary<T>(filename, n, d);
        std::cout << "[Loader] " << filename << " (N=" << n << ", D=" << d << ")" << std::endl;
        return data;
    }

    static void try_convert_hdf5(const std::string& dataset_name) {
        std::string hdf5_file = ann_paths::dataset(dataset_name, ".hdf5");
        std::string train_bin = ann_paths::dataset(dataset_name, "_train.bin");
        if (!std::filesystem::exists(train_bin) && std::filesystem::exists(hdf5_file)) {
            std::cout << "[Loader] Converting " << hdf5_file << " to binary..." << std::endl;
            std::string cmd = ann_paths::converter("export_hdf5.py", hdf5_file);
            if (std::system(cmd.c_str()) != 0) throw std::runtime_error("Dataset conversion failed");
        }
    }
};

/**
 * calculate_recall_at_k
 * Intersection of top-k retrieved and top-k ground truth, divided by k.
 * Averaged over all queries in the batch.
 */
double calculate_recall_at_k(const std::vector<int64_t>& retrieved, const std::vector<uint32_t>& gt, uint32_t num_queries, uint32_t k, uint32_t gt_dim) {
    if (retrieved.empty() || gt.empty()) return 0.0;

    double batch_recall = 0;
    uint32_t res_per_query = retrieved.size() / num_queries;

    for (uint32_t i = 0; i < num_queries; ++i) {
        // 1. Build set of top-k true neighbors for this query
        std::set<uint32_t> gt_top_k;
        for (uint32_t j = 0; j < k && j < gt_dim; ++j) {
            gt_top_k.insert(gt[i * gt_dim + j]);
        }

        // 2. Count retrieved indices that are in the true top-k
        uint32_t hits = 0;
        for (uint32_t j = 0; j < k && j < res_per_query; ++j) {
            if (gt_top_k.count((uint32_t)retrieved[i * res_per_query + j])) {
                hits++;
            }
        }
        batch_recall += (double)hits / k;
    }
    return batch_recall / num_queries;
}

void log_to_csv(const Config& cfg, double avg_latency, double qps, double recall) {
    std::string filename = ann_paths::result("ttaiss", "results_" + cfg.dataset + ".csv");
    bool exists = std::filesystem::exists(filename);
    std::ofstream f(filename, std::ios::app);
    if (!exists) f << "RunID,Type,Dataset,N,D,NList,NProbe,K,BatchSize,AvgLatency_us,QPS,Recall@K\n";
    f << cfg.run_id << ",tt-cluster-gather-fast-fine" << cfg.fine_fidelity << "-coarse" << cfg.coarse_fidelity
      << (cfg.rerank ? "-rerank" : "") << (cfg.ablation_rank32 ? "-ablrank32" : "")
      << (cfg.ablation_rebuild ? "-ablrebuild" : "") << "-first" << cfg.first_pass_lists << "-chunk" << cfg.batch_chunk
      << "-prefetch" << cfg.prefetch_reader
      << "-grouping-" << cfg.query_grouping.mode
      << "-window" << cfg.query_grouping.window
      << "-lookahead" << cfg.query_grouping.lookahead << ","
      << cfg.dataset << "," << cfg.num_vectors << "," << cfg.dim << ","
      << cfg.nlist << "," << cfg.nprobe << "," << cfg.k << "," << cfg.batch_size << ","
      << std::fixed << std::setprecision(2) << avg_latency << ","
      << std::setprecision(2) << qps << ","
      << std::fixed << std::setprecision(6) << recall << "\n";
}

// Energy workload with the ann_ivf_energy contract: the window starts before
// device/index creation and covers every search. Dataset files and centroid
// conversion precede it. There is no warmup, ground truth, recall, result
// dump or transfer log; markers match ann_ivf_energy. Result readback is
// omitted, except with --rerank, whose host re-ranking needs the device top-32
// and is part of producing the results.
int run_energy(const Config& cfg, const RunOptions& options, const std::vector<float>& train_data,
               const std::vector<float>& query_data, uint32_t query_count, const std::string& centroids_bin) {
    const uint32_t queries_per_run = std::min(query_count, cfg.max_num_queries);
    if (!queries_per_run) throw std::runtime_error("Query dataset is empty");
    ivf_energy::EnergyEvents events(cfg.energy_events);
    events.reserve(2 * static_cast<size_t>(cfg.runs) + 4);
    const auto measurement_start = std::chrono::steady_clock::now();
    events.record("measurement_start");
    events.record("initialization_start");
    std::cout << "[Energy] ENERGY_MEASUREMENT_START runs=" << cfg.runs
              << " queries_per_run=" << queries_per_run << std::endl;

    IVF_TT index(0, cfg.dim, cfg.nlist, cfg.nprobe, /*memory_log_path=*/"", cfg.prefetch_reader,
                 cfg.batch_chunk, cfg.query_grouping, options);
    index.create_index(train_data, centroids_bin);
    index.load_to_device();
    events.record("initialization_end");

    double search_seconds = 0.0;
    for (uint32_t run = 0; run < cfg.runs; ++run) {
        std::cout << "--- Run " << run + 1 << " / " << cfg.runs << " ---" << std::endl;
        events.record("run_" + std::to_string(run + 1) + "_start");
        const auto start = std::chrono::steady_clock::now();
        index.search_all(query_data.data(), queries_per_run, cfg.k);
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
    const double measured_seconds = std::chrono::duration<double>(measurement_end - measurement_start).count();
    const uint64_t total_queries = static_cast<uint64_t>(queries_per_run) * cfg.runs;
    std::cout << "[Energy] Initialization + search time: " << std::fixed << std::setprecision(6)
              << measured_seconds << " s\n"
              << "[Energy] Search-loop time: " << search_seconds << " s\n"
              << "[Energy] Total measured queries: " << total_queries << '\n'
              << "[Energy] Throughput: " << std::setprecision(2)
              << static_cast<double>(total_queries) / search_seconds << " queries/s\n";
    return 0;
}

int main(int argc, char** argv) {
    Config cfg;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--run_id") cfg.run_id = option_value(i, argc, argv);
            else if (arg == "--dataset") cfg.dataset = option_value(i, argc, argv);
            else if (arg == "--nlist") cfg.nlist = unsigned_option(i, argc, argv);
            else if (arg == "--nprobe") cfg.nprobe = unsigned_option(i, argc, argv);
            else if (arg == "--batch_size") cfg.batch_size = unsigned_option(i, argc, argv);
            else if (arg == "--k") cfg.k = unsigned_option(i, argc, argv);
            else if (arg == "--max_num_queries") cfg.max_num_queries = unsigned_option(i, argc, argv);
            else if (arg == "--runs") cfg.runs = unsigned_option(i, argc, argv);
            else if (arg == "--mem_log") {
                cfg.mem_log = option_value(i, argc, argv);
                cfg.explicit_mem_log = true;
            }
            else if (arg == "--batch-chunk") cfg.batch_chunk = unsigned_option(i, argc, argv);
            else if (arg == "--fine-math-fidelity") cfg.fine_fidelity = option_value(i, argc, argv);
            else if (arg == "--coarse-math-fidelity") cfg.coarse_fidelity = option_value(i, argc, argv);
            else if (arg == "--energy-events") cfg.energy_events = option_value(i, argc, argv);
            else if (arg == "--index-build") cfg.index_build = option_value(i, argc, argv);
            else if (arg == "--index-digest") cfg.index_digest = true;
            else if (arg == "--rerank") cfg.rerank = true;
            else if (arg == "--ablation-rank32") cfg.ablation_rank32 = true;    // measurement only
            else if (arg == "--ablation-rebuild") cfg.ablation_rebuild = true;  // measurement only
            else if (arg == "--first-pass-lists") cfg.first_pass_lists = option_value(i, argc, argv);
            else if (arg == "--selections-output") cfg.selections_output = option_value(i, argc, argv);
            else if (arg == "--assignment-output") cfg.assignment_output = option_value(i, argc, argv);
            else if (arg == "--results-output") cfg.results_output = option_value(i, argc, argv);
            else if (arg == "--query-grouping") cfg.query_grouping.mode = option_value(i, argc, argv);
            else if (arg == "--grouping-window") cfg.query_grouping.window = unsigned_option(i, argc, argv);
            else if (arg == "--grouping-lookahead") cfg.query_grouping.lookahead = unsigned_option(i, argc, argv);
            else if (arg == "--prefetch-reader") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("--prefetch-reader requires a positive integer value");
                }
                const std::string value = argv[++i];
                const auto [end, error] = std::from_chars(
                    value.data(), value.data() + value.size(), cfg.prefetch_reader);
                if (error != std::errc{} || end != value.data() + value.size() || cfg.prefetch_reader == 0) {
                    throw std::invalid_argument("--prefetch-reader must be a positive integer");
                }
            } else throw std::invalid_argument("Unknown option: " + arg);
        }
        ivf_cluster_major::validate_grouping_config(cfg.query_grouping);
        ivf_benchmark_input::validate_search(cfg.nlist, cfg.nprobe, cfg.k,
                                             cfg.runs, cfg.max_num_queries, cfg.batch_size);

        if (!cfg.batch_chunk) throw std::invalid_argument("--batch-chunk must be a positive integer");
        // auto: two passes from nprobe 24 (measured: no gain at nprobe 16), with
        // nprobe / 8 lists in the first pass (best of 4..16 at nprobe 64).
        uint32_t first_pass_lists = 0;
        if (cfg.first_pass_lists == "auto") {
            first_pass_lists = cfg.nprobe >= 24 && cfg.batch_chunk <= 32 ? cfg.nprobe / 8 : 0;
        } else {
            const auto [end, error] = std::from_chars(cfg.first_pass_lists.data(),
                                                      cfg.first_pass_lists.data() + cfg.first_pass_lists.size(),
                                                      first_pass_lists);
            if (error != std::errc{} || end != cfg.first_pass_lists.data() + cfg.first_pass_lists.size())
                throw std::invalid_argument("--first-pass-lists must be auto or a non-negative integer");
        }
        if (first_pass_lists >= cfg.nprobe)
            throw std::invalid_argument("--first-pass-lists must be smaller than --nprobe (0 = single pass)");
        if (first_pass_lists && cfg.batch_chunk > 32)
            throw std::invalid_argument("--first-pass-lists needs --batch-chunk <= 32 (one merge bit per tile)");
        cfg.first_pass_lists = std::to_string(first_pass_lists);
        RunOptions options;
        options.fine_fidelity = ivf_cluster_major::parse_fidelity(cfg.fine_fidelity);
        options.coarse_fidelity = ivf_cluster_major::parse_fidelity(cfg.coarse_fidelity);
        options.energy = !cfg.energy_events.empty();
        if (cfg.index_build != "fast" && cfg.index_build != "reference")
            throw std::invalid_argument("--index-build must be fast or reference");
        options.reference_index_build = cfg.index_build == "reference";
        options.index_digest = cfg.index_digest;
        options.rerank = cfg.rerank;
        options.first_pass_lists = first_pass_lists;
        options.ablation_rank32 = cfg.ablation_rank32;
        options.ablation_rebuild = cfg.ablation_rebuild;
        if (cfg.ablation_rank32 || cfg.ablation_rebuild)
            std::cout << "[Config] Ablation:" << (cfg.ablation_rank32 ? " threshold rank 32" : "")
                      << (cfg.ablation_rebuild ? " programs/buffers rebuilt per search" : "") << std::endl;
        // --rerank with --energy-events: the energy workload then also reads the
        // results back and re-ranks them, because re-ranking produces the
        // final results; without --rerank it omits the readback as before.

        std::cout << "\n=== Tenstorrent ANN IVF Benchmark Tool ===" << std::endl;
        std::cout << "[Config] Fine schedule: per-list 32-query tiles (cluster-major scan) + per-query merge" << std::endl;
        std::cout << "[Config] Math fidelity: fine " << cfg.fine_fidelity << ", coarse " << cfg.coarse_fidelity
                  << (options.energy ? "; energy workload" : "") << std::endl;
        std::cout << "[Config] FP32 re-ranking of the device top-32: " << (cfg.rerank ? "on" : "off") << std::endl;
        std::cout << "[Config] Merge skipping: "
                  << (first_pass_lists ? "two passes, first pass over the best " + std::to_string(first_pass_lists) +
                                             " lists of each query"
                                       : std::string("off (single pass)")) << std::endl;
        std::cout << "[Config] Batch chunk: " << cfg.batch_chunk << " query batches per list scan" << std::endl;
        std::cout << "[Config] Fine reader L1 staging depth: " << cfg.prefetch_reader << " pages" << std::endl;
        std::cout << "[Config] Query grouping: not used by per-list tiles (accepted for interface compatibility): "
                  << cfg.query_grouping.mode
                  << "; window=" << cfg.query_grouping.window
                  << "; lookahead=" << cfg.query_grouping.lookahead << std::endl;
        if (!cfg.explicit_mem_log) cfg.mem_log = ann_paths::result("ttaiss", "ann_ivf_ttaiss.csv");
        HDF5BinaryLoader::try_convert_hdf5(cfg.dataset);

        uint32_t nt, dt, nq, dq, ng, dg, nd, dd;
        auto train_data = HDF5BinaryLoader::load_generic<float>(ann_paths::dataset(cfg.dataset, "_train.bin"), nt, dt);
        auto query_data = HDF5BinaryLoader::load_generic<float>(ann_paths::dataset(cfg.dataset, "_queries.bin"), nq, dq);
        cfg.dim = dt; cfg.num_vectors = nt;

        std::string centroids_path = ann_paths::centroids(cfg.dataset, cfg.nlist, ".npy");
        std::string centroids_bin = std::filesystem::path(centroids_path).replace_extension(".bin").string();

        bool need_convert = false;
        if (!std::filesystem::exists(centroids_bin)) {
            need_convert = true;
        } else if (std::filesystem::exists(centroids_path)) {
            if (std::filesystem::last_write_time(centroids_bin) < std::filesystem::last_write_time(centroids_path)) {
                need_convert = true;
            }
        }

        if (need_convert) {
            if (!std::filesystem::exists(centroids_path)) {
                std::cout << "[Loader] Error: File: " << centroids_path << " not found" << std::endl;
                return -1;
            } else {
                std::cout << "[Loader] Converting centroids .npy to .bin..." << std::endl;
                std::string cmd = ann_paths::converter("convert_centroids.py", centroids_path);
                if (std::system(cmd.c_str()) != 0) throw std::runtime_error("Centroid conversion failed");
            }
        }

        if (options.energy) {
            if (dt != dq) throw std::runtime_error("Training and query dimensions differ");
            return run_energy(cfg, options, train_data, query_data, nq, centroids_bin);
        }
        auto ground_truth = HDF5BinaryLoader::load_generic<uint32_t>(ann_paths::dataset(cfg.dataset, "_neighbors.bin"), ng, dg);
        auto ground_truth_distances = HDF5BinaryLoader::load_generic<float>(ann_paths::dataset(cfg.dataset, "_distances.bin"), nd, dd);
        ivf_benchmark_input::validate_dataset(dt, nq, dq, ng, dg, nd, dd, cfg.k);

        IVF_TT index(
            0,
            cfg.dim,
            cfg.nlist,
            cfg.nprobe,
            cfg.mem_log,
            cfg.prefetch_reader,
            cfg.batch_chunk,
            cfg.query_grouping,
            options);
        index.set_memory_log_context("index_build");
        index.create_index(train_data, centroids_bin);
        index.set_memory_log_context("index_load");
        index.load_to_device();

        std::cout << "\nWarming up device with a 32-query search (triggers kernel compilation if not cached)..." << std::endl;
        index.set_memory_log_context("warmup");
        index.search_all(query_data.data(), std::min(nq, 32u), cfg.k);

        std::cout << "\nProcessing entire dataset with cluster-major fine search..." << std::endl;
        std::string txt_filename = ann_paths::result("ttaiss", "results_" + cfg.dataset + "_" + std::to_string(cfg.nlist) + "_" + std::to_string(cfg.nprobe) + ".txt");
        std::ofstream txt_f(txt_filename);

        uint32_t max_queries_to_run = std::min(nq, cfg.max_num_queries);
        std::pair<std::vector<float>, std::vector<int64_t>> results;
        double total_latency_us = 0;

        for (uint32_t r = 0; r < cfg.runs; r++) {
            if (cfg.runs > 1) {
                std::cout << "\n--- Run " << (r + 1) << " / " << cfg.runs << " ---" << std::endl;
            }
            index.set_memory_log_context("run_" + std::to_string(r + 1));
            results = index.search_all(query_data.data(), max_queries_to_run, cfg.k);
            // Same interval as the six stages; console/diagnostic output is
            // emitted after it stops. Grouping, repacking and upload are timed.
            double run_latency_us = index.last_search_us();
            total_latency_us += run_latency_us;

            double run_avg_lat = run_latency_us / (max_queries_to_run / 32.0);
            double run_qps = (double)max_queries_to_run / (run_latency_us / 1e6);

            std::vector<int64_t> actual_retrieved(results.second.begin(), results.second.end());
            std::vector<uint32_t> actual_gt(ground_truth.begin(), ground_truth.begin() + max_queries_to_run * dg);
            double run_recall = calculate_recall_at_k(actual_retrieved, actual_gt, max_queries_to_run, cfg.k, dg);
            std::cout << "[Run " << (r + 1) << "] QPS=" << std::fixed << std::setprecision(3)
                      << run_qps << ", Recall@" << cfg.k << '=' << std::setprecision(6)
                      << run_recall << std::endl;

            log_to_csv(cfg, run_avg_lat, run_qps, run_recall);
        }

        auto& res_scores = results.first;
        auto& res_indices = results.second;
        // [N][D] uint32 header, then data (the tools/export_hdf5.py layout).
        auto write_matrix = [](const std::string& path, uint32_t rows, uint32_t cols, const auto& fill) {
            std::ofstream out(path, std::ios::binary);
            if (!out) throw std::runtime_error("Could not open " + path);
            const uint32_t header[2] = {rows, cols};
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            fill(out);
            out.close();
            if (!out) throw std::runtime_error("Could not finish writing " + path);
        };
        if (!cfg.selections_output.empty()) {
            const auto& selections = index.last_selections();
            write_matrix(cfg.selections_output, uint32_t(selections.size()), cfg.nprobe, [&](std::ofstream& out) {
                for (const auto& row : selections) out.write(reinterpret_cast<const char*>(row.data()), row.size() * 4);
            });
            std::cout << "[Results] Final-run coarse selections saved to " << cfg.selections_output << std::endl;
        }
        if (!cfg.assignment_output.empty()) {
            const auto& lists = index.vector_lists();
            write_matrix(cfg.assignment_output, uint32_t(lists.size()), 1, [&](std::ofstream& out) {
                out.write(reinterpret_cast<const char*>(lists.data()), lists.size() * 4);
            });
            std::cout << "[Results] Vector-to-list assignment saved to " << cfg.assignment_output << std::endl;
        }
        if (!cfg.results_output.empty()) {
            // Final run, original query order; written outside the timed interval.
            std::ofstream out(cfg.results_output);
            if (!out) throw std::runtime_error("Could not open result dump: " + cfg.results_output);
            const uint32_t per_query = res_indices.size() / max_queries_to_run;
            out << "query,rank,index,score\n" << std::setprecision(9);
            for (uint32_t q = 0; q < max_queries_to_run; ++q)
                for (uint32_t rank = 0; rank < per_query; ++rank)
                    out << q << ',' << rank << ',' << res_indices[q * per_query + rank] << ','
                        << res_scores[q * per_query + rank] << '\n';
            out.close();
            if (!out) throw std::runtime_error("Could not finish writing result dump: " + cfg.results_output);
            std::cout << "[Results] Final-run neighbors saved to " << cfg.results_output << std::endl;
        }

        double avg_total_latency_us = total_latency_us / cfg.runs;
        double avg_lat = avg_total_latency_us / (max_queries_to_run / 32.0); // equivalent avg per 32-batch
        double qps = (double)max_queries_to_run / (avg_total_latency_us / 1e6);

        uint32_t res_per_query = res_indices.size() / max_queries_to_run;
        for (uint32_t q = 0; q < std::min(max_queries_to_run, 32u); ++q) {
            txt_f << "Query " << q + 1 << " expected: ";
            for (uint32_t j = 0; j < cfg.k && j < dg; ++j) txt_f << ground_truth[q * dg + j] << (j == cfg.k - 1 ? "" : ", ");
            txt_f << "\nQuery " << q + 1 << " expected scores: ";
            for (uint32_t j = 0; j < cfg.k && j < dd; ++j) txt_f << std::fixed << std::setprecision(4) << ground_truth_distances[q * dd + j] << (j == cfg.k - 1 ? "" : ", ");
            txt_f << "\nQuery " << q + 1 << " retrieved: ";
            for (uint32_t j = 0; j < cfg.k && j < res_per_query; ++j) txt_f << res_indices[q * res_per_query + j] << (j == cfg.k - 1 ? "" : ", ");
            txt_f << "\nQuery " << q + 1 << " scores:    ";
            for (uint32_t j = 0; j < cfg.k && j < res_per_query; ++j) txt_f << std::fixed << std::setprecision(4) << res_scores[q * res_per_query + j] << (j == cfg.k - 1 ? "" : ", ");
            txt_f << "\n\n";
        }
        txt_f.close();

        std::vector<int64_t> actual_retrieved(res_indices.begin(), res_indices.end());
        std::vector<uint32_t> actual_gt(ground_truth.begin(), ground_truth.begin() + max_queries_to_run * dg);
        double avg_recall = calculate_recall_at_k(actual_retrieved, actual_gt, max_queries_to_run, cfg.k, dg);

        std::cout << "\n========================================================" << std::endl;
        std::cout << "  ANN IVF BENCHMARK SUMMARY (Batch Size: 32)" << std::endl;
        std::cout << "========================================================" << std::endl;
        std::cout << "  Average Latency/Batch: " << std::setw(10) << std::fixed << std::setprecision(2) << avg_lat << " us" << std::endl;
        std::cout << "  Throughput (QPS):      " << std::setw(10) << std::fixed << std::setprecision(2) << qps << " queries/sec" << std::endl;
        std::cout << "  Final Recall@" << std::left << std::setw(3) << cfg.k << ":      " << std::fixed << std::setprecision(6) << avg_recall << std::endl;
        std::cout << "========================================================\n" << std::endl;

        std::cout << "[Benchmark] Summary recap saved to " << ann_paths::result("ttaiss", "results_" + cfg.dataset + ".csv") << std::endl;
        index.flush_memory_log();
        std::cout << "[MemLog] Memory-transfer measurements saved to " << cfg.mem_log << std::endl;

    } catch (const std::exception& e) { std::cerr << "[Error] " << e.what() << std::endl; return 1; }
    return 0;
}
