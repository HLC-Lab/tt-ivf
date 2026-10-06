#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class MemoryTransferLogger {
public:
    explicit MemoryTransferLogger(std::string path) : path_(std::move(path)) {}

    ~MemoryTransferLogger() {
        try {
            flush();
        } catch (...) {
            // Destructors must not throw. Explicit flush() is called by main for error reporting.
        }
    }

    void set_context(std::string context) { context_ = std::move(context); }

    void record(
        const std::string& layer,
        const std::string& direction,
        const std::string& operation,
        uint64_t bytes,
        double duration_us,
        const std::string& completion,
        const std::string& notes = {}) {
        records_.push_back(
            Record{sequence_++, context_, layer, direction, operation, bytes, duration_us, completion, notes});
    }

    void flush() {
        if (flushed_ || path_.empty()) {
            return;
        }

        std::ofstream output(path_, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Could not open memory-transfer log: " + path_);
        }

        output
            << "sequence,context,layer,direction,operation,bytes,duration_us,bandwidth_GB_s,completion,notes\n";
        for (const auto& record : records_) {
            output << record.sequence << ',' << csv_escape(record.context) << ',' << csv_escape(record.layer) << ','
                   << csv_escape(record.direction) << ',' << csv_escape(record.operation) << ',' << record.bytes << ',';

            if (record.duration_us > 0.0) {
                output << std::fixed << std::setprecision(3) << record.duration_us << ','
                       << std::setprecision(6)
                       << (static_cast<double>(record.bytes) / record.duration_us / 1000.0);
            } else {
                output << ',';
            }

            output << ',' << csv_escape(record.completion) << ',' << csv_escape(record.notes) << '\n';
        }
        flushed_ = true;
    }

private:
    struct Record {
        uint64_t sequence;
        std::string context;
        std::string layer;
        std::string direction;
        std::string operation;
        uint64_t bytes;
        double duration_us;
        std::string completion;
        std::string notes;
    };

    static std::string csv_escape(const std::string& value) {
        if (value.find_first_of(",\"\n") == std::string::npos) {
            return value;
        }

        std::string escaped = "\"";
        for (char ch : value) {
            if (ch == '"') {
                escaped += "\"\"";
            } else {
                escaped += ch;
            }
        }
        escaped += '"';
        return escaped;
    }

    std::string path_;
    std::string context_ = "startup";
    std::vector<Record> records_;
    uint64_t sequence_ = 0;
    bool flushed_ = false;
};
