#include "../benchmark_input.hpp"

#include <chrono>
#include <iostream>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("Input check failed");
}
template <class F> void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
}  // namespace

int main() {
    using namespace ivf_benchmark_input;
    try {
        char option[] = "--runs", valid[] = "5", negative[] = "-1", bad[] = "5x", overflow[] = "4294967296";
        char* arguments[] = {option, valid};
        int index = 0;
        require(unsigned_option(index, 2, arguments) == 5 && index == 1);
        require(shell_quote("data file.bin") == "'data file.bin'");
        require(shell_quote("it's;literal") == "'it'\\''s;literal'");
        rejects([&] { index = 0; option_value(index, 1, arguments); });
        for (char* value : {negative, bad, overflow}) {
            arguments[1] = value;
            rejects([&] { index = 0; unsigned_option(index, 2, arguments); });
        }
        validate_search(512, 16, 10, 5, 10000, 32);
        rejects([] { validate_search(512, 33, 10, 5, 10000, 32); });
        rejects([] { validate_search(512, 16, 0, 5, 10000, 32); });
        rejects([] { validate_search(512, 16, 10, 0, 10000, 32); });
        rejects([] { validate_search(512, 16, 10, 5, 10000, 16); });
        validate_dataset(100, 256, 100, 10000, 100, 10000, 100, 10);
        rejects([] { validate_dataset(100, 256, 99, 10000, 100, 10000, 100, 10); });
        rejects([] { validate_dataset(100, 256, 100, 255, 100, 10000, 100, 10); });
        rejects([] { validate_dataset(100, 256, 100, 10000, 9, 10000, 100, 10); });
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto path = std::filesystem::temp_directory_path() / ("ann-input-" + std::to_string(stamp) + ".bin");
        struct Remove { std::filesystem::path path; ~Remove() { std::error_code ec; std::filesystem::remove(path, ec); } } remove{path};
        uint32_t rows = 99, columns = 99;
        rejects([&] { load_binary<float>(path.string(), rows, columns); });
        auto write = [&](uint32_t n, uint32_t d, size_t values) {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<char*>(&n), sizeof(n));
            file.write(reinterpret_cast<char*>(&d), sizeof(d));
            for (size_t i = 0; i < values; ++i) {
                const float value = static_cast<float>(i);
                file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        };
        write(2, 3, 6);
        const auto loaded = load_binary<float>(path.string(), rows, columns);
        require(rows == 2 && columns == 3 && loaded.size() == 6 && loaded.back() == 5);
        write(2, 3, 5);
        rejects([&] { load_binary<float>(path.string(), rows, columns); });
        write(UINT32_MAX, UINT32_MAX, 0);
        rejects([&] { load_binary<float>(path.string(), rows, columns); });
        write(0, 3, 0);
        rejects([&] { load_binary<float>(path.string(), rows, columns); });
        std::filesystem::resize_file(path, 3);
        rejects([&] { load_binary<float>(path.string(), rows, columns); });
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
