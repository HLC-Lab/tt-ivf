// Read Tenstorrent device contexts with Tracy's own version-matched file reader.
// Compile against the source of the Tracy version used to save the capture.
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "server/TracyFileRead.hpp"
#include "server/TracyWorker.hpp"
#include "public/common/TracyTTDeviceData.hpp"

static void quoted(const char* value) {
    std::cout << '"';
    for (; *value; ++value) {
        if (*value == '"') std::cout << '"';
        std::cout << *value;
    }
    std::cout << '"';
}

static void export_level(const tracy::Worker& worker,
                         const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& events,
                         const tracy::GpuCtxData& context, uint64_t thread, int depth) {
    const auto emit = [&](const tracy::GpuEvent& event) {
        const tracy::TTDeviceEvent tt(thread);
        const auto& source = worker.GetSourceLocation(event.SrcLoc());
        quoted(worker.GetString(context.name));
        std::cout << ',' << tt.chip_id << ',' << tt.core_x << ',' << tt.core_y << ',';
        quoted(tt.risc < 6 ? tracy::riscName[tt.risc].c_str() : "UNKNOWN");
        std::cout << ',';
        quoted(worker.GetString(source.name.active ? source.name : source.function));
        std::cout << ',';
        quoted(worker.GetString(source.file));
        std::cout << ',' << source.line << ',' << depth << ',' << event.GpuStart()
                  << ',' << event.GpuEnd() << ',' << context.hasCalibration << '\n';
        if (event.Child() >= 0)
            export_level(worker, worker.GetGpuChildren(event.Child()), context, thread, depth + 1);
    };
    if (events.is_magic()) {
        // Tracy loads timelines as a contiguous GpuEvent vector with this flag.
        const auto& direct = reinterpret_cast<const tracy::Vector<tracy::GpuEvent>&>(events);
        for (const auto& event : direct) emit(event);
    } else {
        for (const auto& event : events) emit(*event);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: export_tt_device_zones CAPTURE.tracy\n";
        return 2;
    }
    try {
        auto file = std::unique_ptr<tracy::FileRead>(tracy::FileRead::Open(argv[1]));
        if (!file) throw std::runtime_error("Cannot open capture");
        tracy::Worker worker(*file, tracy::EventType::All, false);
        std::cout << "context,device,core_x,core_y,risc,zone,source_file,source_line,depth,start_ns,end_ns,calibrated\n";
        size_t contexts = 0;
        for (const auto* context : worker.GetGpuData()) {
            if (context->type != tracy::GpuContextType::tt_device) continue;
            ++contexts;
            for (const auto& thread : context->threadData)
                export_level(worker, thread.second.timeline, *context, thread.first, 0);
        }
        if (!contexts) throw std::runtime_error("No Tenstorrent device contexts in capture");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Device export failed: " << error.what() << '\n';
        return 1;
    }
}
