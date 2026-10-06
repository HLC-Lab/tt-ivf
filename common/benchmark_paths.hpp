#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace ann_paths {

inline std::filesystem::path project_root() {
    const char* value = std::getenv("TT_METAL_KERNEL_PATH");
    return value && *value ? std::filesystem::path(value) : std::filesystem::current_path();
}

inline std::filesystem::path root(const char* variable, const char* folder) {
    const char* value = std::getenv(variable);
    return value && *value ? std::filesystem::path(value) : project_root() / folder;
}

inline void validate_dataset(const std::string& name) {
    if (name.empty() || name == "." || name == ".." ||
        name.find_first_of("/\\") != std::string::npos)
        throw std::invalid_argument("--dataset must be a dataset name, e.g. glove-100-angular");
}

inline std::string dataset(const std::string& name, const std::string& suffix) {
    validate_dataset(name);
    return (root("ANN_DATA_DIR", "data") / "datasets" / (name + suffix)).string();
}

inline std::string centroids(const std::string& name, unsigned nlist, const std::string& extension) {
    validate_dataset(name);
    return (root("ANN_DATA_DIR", "data") / "centroids" /
            ("centroids-" + name + "-" + std::to_string(nlist) + extension)).string();
}

inline std::string result(const std::string& experiment, const std::string& filename) {
    const auto folder = root("ANN_RESULTS_DIR", "results") / experiment;
    std::filesystem::create_directories(folder);
    return (folder / filename).string();
}

inline std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char c : value) quoted += c == '\'' ? "'\\''" : std::string(1, c);
    return quoted + "'";
}

inline std::string converter(const std::string& script, const std::string& input) {
    return "python3 " + shell_quote((project_root() / "tools" / script).string()) +
           " " + shell_quote(input);
}

}  // namespace ann_paths
