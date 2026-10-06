import subprocess
import time
import json
import threading
import argparse
import math
import re
import shlex
import csv
from energy_sampling import discover_rapl, save_rapl, rapl_energy, derived_host_power
from pathlib import Path

import sys

# Shared data/result helpers live in the project root's tools/ package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.data_layout import results_root

# Configuration

TT_EXECUTABLE_PATH = "./build_Release/bin/ann_ivf_energy"
FAISS_SCRIPT_PATH = str(Path(__file__).resolve().with_name("run_faiss_benchmark.py"))
DATASET = "glove-100-angular"
K = 10


def _numeric_power(value, key_hint=""):
    if isinstance(value, dict):
        for nested_key in ("value", "current", "reading"):
            if nested_key in value:
                return _numeric_power(value[nested_key], key_hint)
        return None
    if isinstance(value, (int, float)):
        power = float(value)
        unit_text = key_hint.lower()
    elif isinstance(value, str):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        if match is None:
            return None
        power = float(match.group(0))
        unit_text = f"{key_hint} {value}".lower()
    else:
        return None

    if not math.isfinite(power) or power < 0:
        return None
    if re.search(r"(?:^|[^a-z])uw(?:$|[^a-z])|microwatt", unit_text):
        power /= 1_000_000.0
    elif re.search(r"(?:^|[^a-z])mw(?:$|[^a-z])|milliwatt", unit_text):
        power /= 1_000.0
    return power


def extract_tt_power_watts(output):
    """Extract one board-power reading from tt-smi JSON output."""
    text = output.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace < 0 or last_brace <= first_brace:
            raise ValueError("tt-smi output does not contain JSON")
        payload = json.loads(text[first_brace : last_brace + 1])

    candidates = []

    def visit(value, path):
        if isinstance(value, dict):
            for raw_key, nested in value.items():
                key = str(raw_key)
                normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
                nested_path = (*path, normalized)
                if "power" in normalized and not any(
                    excluded in normalized
                    for excluded in ("limit", "maximum", "max", "cap")
                ):
                    power = _numeric_power(nested, normalized)
                    if power is not None:
                        score = 0
                        if normalized in {
                            "power",
                            "power_w",
                            "board_power",
                            "board_power_w",
                            "power_draw",
                        }:
                            score += 100
                        if "telemetry" in path:
                            score += 50
                        if "board" in normalized:
                            score += 10
                        candidates.append((score, nested_path, power))
                visit(nested, nested_path)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, (*path, str(index)))

    visit(payload, ())
    if not candidates:
        raise ValueError("no board-power field was found in tt-smi JSON")
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return candidates[0][2]


def read_tt_power_once():
    before = time.time()
    result = subprocess.run(
        ["tt-smi", "-s", "--snapshot_no_tty"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    after = time.time()
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"tt-smi exited with {result.returncode}: {detail}")
    return (before + after) * 0.5, extract_tt_power_watts(result.stdout), result.stdout

def read_rapl_energy():
    zones = discover_rapl()
    return sum(float(zone.path.read_text()) for zone in zones) / 1e6 if zones else None

class PowerProfiler:
    def __init__(self, target, require_host=False):
        self.target = target
        self.stop_event = threading.Event()
        self.tt_ready = threading.Event()
        self.host_ready = threading.Event()
        self.tt_samples = []
        self.cpu_samples = []
        self.rapl_samples = []
        self.last_tt_error = None
        self.last_tt_output = ""
        self.host_error = None
        self.zones = discover_rapl()
        if require_host and not self.zones:
            raise RuntimeError("No readable host CPU package RAPL counters; system energy cannot be measured")
        self.threads = []

    def start(self):
        if self.zones:
            self.threads.append(threading.Thread(target=self._poll_host))
        if self.target == "tt":
            self.threads.append(threading.Thread(target=self._poll_tt))
        for thread in self.threads:
            thread.start()
        # Capture a boundary sample before launching the benchmark process.
        if self.target == "tt" and not self.tt_ready.wait(10):
            self.stop()
            raise RuntimeError(self.last_tt_error or "No TT sample before launch")
        if self.zones and not self.host_ready.wait(10):
            self.stop()
            raise RuntimeError(self.host_error or "No host sample before launch")

    def trigger_monitoring(self):
        pass  # Sampling already started, before the benchmark process.

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        self.cpu_samples = derived_host_power(self.zones, self.rapl_samples)
        return self.tt_samples, self.cpu_samples

    def _poll_tt(self):
        while not self.stop_event.is_set():
            try:
                timestamp, power, output = read_tt_power_once()
                self.last_tt_output = output
                self.tt_samples.append((timestamp, power))
                self.tt_ready.set()
            except Exception as error:
                self.last_tt_error = str(error)
            self.stop_event.wait(0.1)

    def _poll_host(self):
        while not self.stop_event.is_set():
            try:
                before = time.time()
                readings = [float(zone.path.read_text()) for zone in self.zones]
                after = time.time()
                if any(not math.isfinite(value) for value in readings):
                    raise ValueError("Nonfinite host counter")
                self.rapl_samples.append(((before + after) / 2, *readings))
                self.host_ready.set()
            except Exception as error:
                self.host_error = str(error)
                return
            self.stop_event.wait(0.1)

    def bracket_end(self, timestamp):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            tt_ok = self.target != "tt" or (self.tt_samples and self.tt_samples[-1][0] >= timestamp)
            host_ok = not self.zones or (self.rapl_samples and self.rapl_samples[-1][0] >= timestamp)
            if tt_ok and host_ok:
                return
            if self.host_error:
                raise RuntimeError(self.host_error)
            time.sleep(0.05)
        raise RuntimeError("Power samples do not bracket the end of measurement")

def calc_energy(samples, start_time, end_time):
    ordered = sorted(samples)
    valid_samples = [s for s in ordered if start_time <= s[0] <= end_time]
    if not ordered or end_time <= start_time:
        return 0.0, 0.0, len(valid_samples)

    def power_at(timestamp):
        if timestamp <= ordered[0][0]:
            return ordered[0][1]
        if timestamp >= ordered[-1][0]:
            return ordered[-1][1]
        for (left_t, left_p), (right_t, right_p) in zip(ordered, ordered[1:]):
            if left_t <= timestamp <= right_t:
                weight = (timestamp - left_t) / (right_t - left_t)
                return left_p + weight * (right_p - left_p)
        return ordered[-1][1]

    integration_points = [(start_time, power_at(start_time))]
    integration_points.extend(s for s in ordered if start_time < s[0] < end_time)
    integration_points.append((end_time, power_at(end_time)))

    total_energy = 0.0
    for (t1, p1), (t2, p2) in zip(integration_points, integration_points[1:]):
        total_energy += (p1 + p2) * 0.5 * (t2 - t1)
    avg_power = total_energy / (end_time - start_time)
    return avg_power, total_energy, len(valid_samples)


def save_energy_summary(path, args, duration, samples, avg_power, energy, measured_queries):
    with open(path, "w") as output:
        output.write(
            "target,dataset,nlist,nprobe,k,runs,queries,duration_s,samples,"
            "average_power_w,energy_j,energy_per_query_j,query_grouping,grouping_window,grouping_lookahead\n"
        )
        energy_per_query = energy / measured_queries if measured_queries else 0.0
        output.write(
            f"{args.target},{args.dataset},{args.nlist},{args.nprobe},{args.k},"
            f"{args.runs},{measured_queries},{duration:.9f},{samples},"
            f"{avg_power:.9f},{energy:.9f},{energy_per_query:.12f},"
            f"{args.query_grouping},{args.grouping_window},{args.grouping_lookahead}\n"
        )
    print(f"-> Saved energy summary to: {path}")

def plot_and_save_data(samples, target_name, nlist, nprobe, search_start_time, avg_power, output_dir):
    import matplotlib.pyplot as plt
    if not samples:
        return

    csv_file = output_dir / f"power_log_{nlist}_{nprobe}_{target_name}.csv"
    with open(csv_file, "w") as f:
        f.write("Timestamp,Power_Watts\n")
        for t, p in samples:
            f.write(f"{t},{p}\n")

    print(f"-> Saved {target_name} power data to: {csv_file}")

    if len(samples) < 2:
        return

    t0 = samples[0][0]
    rel_times = [t - t0 for t, p in samples]
    powers = [p for t, p in samples]

    plt.figure(figsize=(10, 5))
    plt.plot(rel_times, powers, label=f"Instantaneous {target_name.upper()} Power", color='#ff7f0e', alpha=0.6)

    window = 5
    if len(powers) >= window:
        import numpy as np
        smoothed = np.convolve(powers, np.ones(window)/window, mode='valid')
        plt.plot(rel_times[window-1:], smoothed, label="Smoothed (0.5s Avg)", color='#d62728', linewidth=2)

    # Mark the start of the measured loop.
    if search_start_time is not None:
        rel_search_start = search_start_time - t0
        plt.axvline(x=rel_search_start, color='gray', linestyle='--', label='Measurement starts', linewidth=1.5)

        # Add the horizontal average line during the search phase
        if avg_power > 0:
            plt.axhline(y=avg_power, color='green', linestyle='-', label=f'Average measured power: {avg_power:.1f} W', linewidth=2)

    plt.title(f"{target_name.upper()} Power Over Time (NList={nlist}, NProbe={nprobe})", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Power (Watts)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    png_file = output_dir / f"power_log_{nlist}_{nprobe}_{target_name}_timeline.png"
    plt.savefig(png_file, dpi=300)
    plt.close()
    print(f"-> Saved {target_name} timeline graph to: {png_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=["tt", "cpu"], default="tt", help="Target architecture to profile")
    parser.add_argument("--nlist", type=int, default=2048)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--max-num-queries", type=int, default=10000)
    parser.add_argument("--result-staging", choices=["dram", "core0-l1"], default="dram")
    parser.add_argument("--cluster-chunk-blocks", type=int, default=0)
    parser.add_argument("--query-grouping", choices=["none", "primary-list", "weighted-union"], default="none")
    parser.add_argument("--grouping-window", type=int, default=0)
    parser.add_argument("--grouping-lookahead", type=int, default=256)
    parser.add_argument("--require-host", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--executable", default=TT_EXECUTABLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=results_root() / "energy_profiles")
    args = parser.parse_args()
    if args.runs <= 0 or not 1 <= args.max_num_queries <= 10000 or not 1 <= args.k <= 32:
        parser.error("runs must be positive; queries must be 1..10000 and k 1..32")
    if args.cluster_chunk_blocks != 0:
        parser.error("cluster chunking was removed; omit --cluster-chunk-blocks or use 0")
    if args.grouping_window < 0 or args.grouping_window % 32 or args.grouping_lookahead <= 0:
        parser.error("grouping window must be 0 or a multiple of 32; lookahead must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target == "tt":
        try:
            _, preflight_power, preflight_output = read_tt_power_once()
        except Exception as error:
            (output_dir / "tt_smi_preflight_error.txt").write_text(
                f"{error}\n", encoding="utf-8"
            )
            raise SystemExit(f"TT power preflight failed: {error}")
        (output_dir / "tt_smi_preflight.json").write_text(
            preflight_output, encoding="utf-8"
        )
        print(f"[Power] tt-smi preflight: {preflight_power:.3f} W")

    print(f"\n==========================================================")
    print(f"Running Power Profiler for [{args.target.upper()}]")
    print(f"Dataset: {args.dataset} | NList: {args.nlist} | NProbe: {args.nprobe} | Runs: {args.runs}")
    print(f"==========================================================\n")

    profiler = PowerProfiler(args.target, args.require_host)
    if not profiler.zones:
        print("[Power] Host package counters unavailable; this capture measures device energy only")
    else:
        print("[Power] Selected one host counter per package prefix: " +
              ", ".join(f"{zone.column}={zone.name}" for zone in profiler.zones))

    if args.target == "tt":
        cmd = [
            args.executable,
            "--dataset", args.dataset,
            "--nlist", str(args.nlist),
            "--nprobe", str(args.nprobe),
            "--k", str(args.k),
            "--max_num_queries", str(args.max_num_queries),
            "--runs", str(args.runs),
            "--result-staging", args.result_staging,
            "--cluster-chunk-blocks", str(args.cluster_chunk_blocks),
            "--query-grouping", args.query_grouping,
            "--grouping-window", str(args.grouping_window),
            "--grouping-lookahead", str(args.grouping_lookahead),
            "--energy-events", str(output_dir / "events.csv"),
        ]
        start_trigger = "ENERGY_MEASUREMENT_START"
        end_trigger = "ENERGY_MEASUREMENT_END"
    else:
        cmd = [
            "python3", "-u", FAISS_SCRIPT_PATH,
            "--dataset", args.dataset + ".hdf5",
            "--nlist", str(args.nlist),
            "--nprobe", str(args.nprobe),
            "--k", str(args.k),
            "--runs", str(args.runs)
        ]
        start_trigger = "--- Run 1 /"
        end_trigger = "Saving per-query recall"

    (output_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    events_path = output_dir / ("sampler_events.csv" if args.target == "tt" else "events.csv")
    events_path.write_text("Event,Timestamp\n")

    def record_event(name, timestamp=None):
        event_time = time.time() if timestamp is None else timestamp
        with events_path.open("a") as event_file:
            event_file.write(f"{name},{event_time:.9f}\n")

    profiler.start()

    search_start_time = None
    search_end_time = None
    measured_queries = 0

    benchmark_status = 1
    process = None
    benchmark_log_path = output_dir / "benchmark.log"
    try:
        record_event("process_start")
        with benchmark_log_path.open("w") as benchmark_log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                benchmark_log.write(line)
                benchmark_log.flush()

                if start_trigger in line and search_start_time is None:
                    search_start_time = time.time()
                    record_event("measurement_start", search_start_time)
                    profiler.trigger_monitoring()
                    marker = re.search(r"runs=(\d+)\s+queries_per_run=(\d+)", line)
                    if marker:
                        measured_queries = int(marker.group(1)) * int(marker.group(2))
                elif end_trigger in line and search_end_time is None:
                    search_end_time = time.time()
                    record_event("measurement_end", search_end_time)

            process.wait()
            benchmark_status = process.returncode
        record_event("process_end")
        if benchmark_status != 0:
            print(f"\n[Error] Benchmark exited with code {benchmark_status}")
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        try:
            profiler.bracket_end(time.time())
        finally:
            tt_samples, cpu_samples = profiler.stop()
            if profiler.zones:
                save_rapl(output_dir, profiler.zones, profiler.rapl_samples)
            for label, samples in (("tt", tt_samples), ("cpu", cpu_samples)):
                if samples:
                    with (output_dir / f"power_log_{args.nlist}_{args.nprobe}_{label}.csv").open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["Timestamp", "Power_Watts"])
                        writer.writerows(samples)
    if profiler.host_error:
        raise SystemExit(f"Host energy sampling failed: {profiler.host_error}")
    if args.target == "tt":
        # Use exact C++ timestamps, not arrival times of stdout lines.
        path = output_dir / "events.csv"
        if benchmark_status != 0:
            raise SystemExit(benchmark_status)
        if not path.is_file():
            raise SystemExit("Missing C++ energy events; rebuild ann_ivf_energy")
        with path.open(newline="") as handle:
            events = {row["Event"]: float(row["Timestamp"]) for row in csv.DictReader(handle)}
        search_start_time = events.get("measurement_start")
        search_end_time = events.get("measurement_end")
        if search_start_time is None or search_end_time is None:
            raise SystemExit("Missing C++ initialization/search boundaries")
        if not measured_queries:
            raise SystemExit("Missing actual measured query count in benchmark output")
        tt_times = [t for t, _ in tt_samples]
        if len(tt_times) < 2 or tt_times[0] > search_start_time or tt_times[-1] < search_end_time:
            raise SystemExit("TT samples do not bracket the C++ measurement window")

    if args.target == "tt" and not tt_samples:
        diagnostic = profiler.last_tt_error or "tt-smi returned no usable samples"
        (output_dir / "tt_smi_sampling_error.txt").write_text(
            diagnostic + "\n" + profiler.last_tt_output,
            encoding="utf-8",
        )
        print(f"\n[Error] No TT power samples were collected: {diagnostic}")
        raise SystemExit(2)

    if search_start_time is None or search_end_time is None:
        print("\n-> Warning: Could not detect exact search window in output. Defaulting to full execution time.")
        search_start_time = cpu_samples[0][0] if cpu_samples else (tt_samples[0][0] if tt_samples else 0)
        search_end_time = cpu_samples[-1][0] if cpu_samples else (tt_samples[-1][0] if tt_samples else 0)

    # Calculate energies
    if profiler.zones:
        cpu_energy, cpu_count, _ = rapl_energy(output_dir, search_start_time, search_end_time)
        cpu_avg_power = cpu_energy / (search_end_time - search_start_time)
    else:
        cpu_avg_power, cpu_energy, cpu_count = 0.0, 0.0, 0
    tt_avg_power, tt_energy, tt_count = calc_energy(tt_samples, search_start_time, search_end_time)

    print(f"\n==========================================================")
    print(f"                    ENERGY SUMMARY [{args.target.upper()}]")
    print(f"==========================================================")
    print(f"-> Measurement Time (initialization + searches): {search_end_time - search_start_time:.3f} seconds")
    if measured_queries:
        print(f"-> Queries Processed:          {measured_queries}")

    total_system_energy = 0
    total_system_power = 0

    if args.target == "tt":
        print(f"\n[Tenstorrent Chip]")
        print(f"-> Samples Collected:          {tt_count}")
        print(f"-> Average Measured Power:       {tt_avg_power:.2f} Watts")
        print(f"-> Total TT Energy:            {tt_energy:.2f} Joules")
        if measured_queries:
            print(f"-> TT Energy / Query:          {tt_energy * 1000.0 / measured_queries:.4f} mJ/query")
        total_system_power += tt_avg_power
        total_system_energy += tt_energy

    if cpu_count > 0:
        print(f"\n[Host CPU]")
        print(f"-> Samples Collected:          {cpu_count}")
        print(f"-> Average Measured Power:       {cpu_avg_power:.2f} Watts")
        print(f"-> Total CPU Energy:           {cpu_energy:.2f} Joules")
        if measured_queries:
            print(f"-> CPU Energy / Query:         {cpu_energy * 1000.0 / measured_queries:.4f} mJ/query")
        total_system_power += cpu_avg_power
        total_system_energy += cpu_energy

    if args.target == "tt" and cpu_count > 0:
        print(f"\n[System Total (TT + CPU)]")
        print(f"-> Combined Measured Power:      {total_system_power:.2f} Watts")
        print(f"-> Combined Total Energy:      {total_system_energy:.2f} Joules")
        if measured_queries:
            print(
                f"-> Combined Energy / Query:    "
                f"{total_system_energy * 1000.0 / measured_queries:.4f} mJ/query"
            )

    print(f"==========================================================\n")

    # Raw CSVs were retained in the cleanup path, including failed captures.
    for label, samples in (("tt", tt_samples), ("cpu", cpu_samples)):
        if samples and not args.no_plots:
            plot_and_save_data(samples, label, args.nlist, args.nprobe, search_start_time,
                               tt_avg_power if label == "tt" else cpu_avg_power, output_dir)
    selected_count = tt_count if args.target == "tt" else cpu_count
    selected_power = tt_avg_power if args.target == "tt" else cpu_avg_power
    selected_energy = tt_energy if args.target == "tt" else cpu_energy
    save_energy_summary(
        output_dir / f"energy_summary_{args.nlist}_{args.nprobe}_{args.target}.csv",
        args,
        search_end_time - search_start_time,
        selected_count,
        selected_power,
        selected_energy,
        measured_queries,
    )
    if benchmark_status != 0:
        raise SystemExit(benchmark_status)

if __name__ == "__main__":
    main()
