#include "../benchmark_paths.hpp"

#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const std::filesystem::path project(argv[1]);
    const auto input_root = project / "inputs with ' quote";
    setenv("TT_METAL_KERNEL_PATH", project.c_str(), 1);
    setenv("ANN_DATA_DIR", input_root.c_str(), 1);
    unsetenv("ANN_RESULTS_DIR");
    const auto data = ann_paths::dataset("fixture", "_train.bin");
    const auto centroid = ann_paths::centroids("fixture", 512, ".npy");
    if (std::filesystem::path(data) != input_root / "datasets/fixture_train.bin" ||
        std::filesystem::path(centroid) != input_root / "centroids/centroids-fixture-512.npy") return 2;
    std::filesystem::create_directories(input_root / "datasets");
    std::ofstream(data) << "unchanged input bytes";
    std::filesystem::create_directories(project / "tools");
    std::ofstream(project / "tools/script with ' quote.py") <<
        "from pathlib import Path\nimport sys\np = Path(sys.argv[1])\n"
        "assert p.read_text() == 'unchanged input bytes'\np.with_suffix('.ok').write_text('converted')\n";
    const auto command = ann_paths::converter("script with ' quote.py", data);
    if (std::system(command.c_str()) != 0 ||
        !std::filesystem::exists(std::filesystem::path(data).replace_extension(".ok"))) return 3;
    const auto output = ann_paths::result("mem_log", "summary.csv");
    if (std::filesystem::path(output) != project / "results/mem_log/summary.csv" ||
        !std::filesystem::is_directory(project / "results/mem_log")) return 4;
    try {
        ann_paths::dataset("../fixture", "_train.bin");
        return 5;
    } catch (const std::invalid_argument&) {}
    std::cout << "Organized input/output paths and quoted converter execution agree\n";
}
