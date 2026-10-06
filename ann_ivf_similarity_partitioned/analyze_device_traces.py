#!/usr/bin/env python3
"""Analyse scoped IVF device traces, without CPU zones or timestamp markers.

Example:
  python3 analyze_device_traces.py --trace serial2=/path/tracy1.tracy \
      --trace direct8=/path/tracy2.tracy --tracy-source /path/tracy-0.10-tt.0 \
      --output similarity_trace_analysis

.tracy inputs use the adjacent C++ helper, compiled once against matching Tracy
sources. --exporter selects an existing helper. CSV inputs can be helper exports
or Metalium's .logs/profile_log_device.csv, requiring no C++ compiler.
Partition numbering assumes the specified rows/columns layout; coordinates are
physical NoC coordinates. Durations use each device scope's start/end timestamps.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUSTOM_PREFIXES = ("Worker ", "Leader ", "Partition ")
COLORS = {
    "Worker issue DRAM pair": "#7559E8", "Worker read wait": "#9B83EE",
    "Worker relay receive wait": "#9B83EE", "Worker free buffer wait": "#B5BAC3",
    "Worker input wait": "#B5BAC3", "Worker matmul": "#F2C500",
    "Worker score pack": "#F6DC68", "Worker score and ID wait": "#D2D6DD",
    "Worker local topk": "#EE9824", "Worker tail mask": "#D87737",
}
PARTITION_COLORS = dict(COLORS, **{
    "Worker query receive": "#B29BF2", "Worker query wait": "#B5BAC3",
    "Worker result pack": "#F6DC68", "Worker result wait": "#B5BAC3",
    "Worker partial write": "#F45151",
    "Leader query read": "#B29BF2", "Leader query prefetch": "#B29BF2",
    "Leader query credit wait": "#B5BAC3", "Leader query broadcast": "#7559E8",
    "Leader batch done wait": "#B5BAC3", "Leader issue DRAM pair": "#7559E8",
    "Leader read wait": "#9B83EE", "Leader forward pair": "#7559E8",
    "Partition gather": "#D2D6DD", "Partition gather partial": "#7559E8",
    "Partition partial wait": "#B5BAC3", "Partition aggregate": "#F2C500",
    "Partition result wait": "#B5BAC3", "Partition final write": "#F45151",
})
PARTITION_LABELS = {
    "Worker free buffer wait": "buffer wait", "Worker input wait": "input wait",
    "Worker issue DRAM pair": "issue reads", "Worker read wait": "read wait",
    "Worker relay receive wait": "relay wait", "Worker query receive": "receive query",
    "Worker query wait": "query wait", "Worker matmul": "matmul",
    "Worker score pack": "pack scores", "Worker score and ID wait": "score/ID wait",
    "Worker local topk": "top-k", "Worker tail mask": "tail mask",
    "Worker result pack": "pack results", "Worker result wait": "wait for results",
    "Worker partial write": "write partial", "Leader query read": "read query",
    "Leader query prefetch": "prefetch query", "Leader query credit wait": "wait for credits",
    "Leader query broadcast": "broadcast query", "Leader batch done wait": "wait for batch completion",
    "Leader issue DRAM pair": "issue reads", "Leader read wait": "read wait",
    "Leader forward pair": "forward data", "Partition gather": "gather (includes waiting)",
    "Partition gather partial": "read partial", "Partition partial wait": "wait for partial",
    "Partition aggregate": "aggregate (includes synchronization)",
    "Partition result wait": "wait for result", "Partition final write": "write final",
}
RISC_ROLES = {"NCRISC": "reader", "TRISC_0": "unpacker", "TRISC_1": "math",
              "TRISC_2": "packer", "BRISC": "writer"}


@dataclass(frozen=True)
class Event:
    device: int
    x: int
    y: int
    risc: str
    zone: str
    start: int
    end: int
    calibrated: bool
    source: str = ""

    @property
    def core(self) -> tuple[int, int, int]:
        return self.device, self.x, self.y

    @property
    def stream(self) -> tuple[int, int, int, str]:
        return *self.core, self.risc

    @property
    def us(self) -> float:
        return (self.end - self.start) / 1000


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def stats(values: list[float]) -> dict:
    return dict(samples=len(values), mean_us=statistics.mean(values),
                median_us=statistics.median(values), p95_us=percentile(values, 95),
                max_us=max(values), over_10_us=sum(v > 10 for v in values))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_exporter(source: Path, target: Path) -> Path:
    """Use the upstream file reader, avoiding a fragile binary format parser."""
    source = source.resolve()
    if not (source / "public/common/TracyTTDeviceData.hpp").is_file():
        raise ValueError(f"{source}: matching Tenstorrent Tracy sources are required")
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise ValueError("A C++17 compiler is required for .tracy input; CSV input needs none")
    pkg = shutil.which("pkg-config")
    if pkg:
        result = subprocess.run([pkg, "--cflags", "--libs", "capstone", "libzstd"],
                                text=True, capture_output=True)
        if result.returncode:
            raise ValueError("Install capstone and zstd development packages: " + result.stderr)
        flags = shlex.split(result.stdout)
    else:
        # Homebrew's Tracy dependencies may exist even without pkg-config.
        prefix = next((p for p in (Path("/usr/local/opt"), Path("/opt/homebrew/opt"))
                       if (p / "capstone/include/capstone/capstone.h").exists()
                       and (p / "zstd/lib").is_dir()), None)
        if prefix is None:
            raise ValueError("Install pkg-config, capstone and zstd, or provide --exporter")
        flags = [f"-I{prefix}/capstone/include/capstone", f"-L{prefix}/capstone/lib",
                 f"-L{prefix}/zstd/lib", "-lcapstone", "-lzstd"]
    common = ("TracySocket", "TracyStackFrames", "TracySystem", "tracy_lz4", "tracy_lz4hc")
    server = ("TracyMemory", "TracyMmap", "TracyPrint", "TracyTaskDispatch",
              "TracyTextureCompression", "TracyThreadCompress", "TracyWorker")
    files = [source / f"public/common/{name}.cpp" for name in common]
    files += [source / f"server/{name}.cpp" for name in server]
    temporary = target.with_suffix(".building")
    command = [compiler, "-O2", "-std=c++17", "-DNDEBUG", "-DTRACY_NO_STATISTICS",
               f"-I{source}", str(HERE / "export_tt_device_zones.cpp"),
               *map(str, files), *flags, "-lpthread", "-o", str(temporary)]
    print("Building the Tracy device exporter (once)...", flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise ValueError("Device exporter build failed:\n" + result.stderr[-6000:])
    temporary.replace(target)
    return target


def read_export(path: Path) -> tuple[list[Event], int]:
    events, rejected = [], 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"device", "core_x", "core_y", "risc", "zone", "start_ns", "end_ns", "calibrated"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected device scopes, not Tracy's host-only CSV export")
        for row in reader:
            start, end = int(row["start_ns"]), int(row["end_ns"])
            if start < 0 or end < start:
                rejected += 1
                continue
            events.append(Event(int(row["device"]), int(row["core_x"]), int(row["core_y"]),
                                row["risc"], row["zone"], start, end, row["calibrated"] == "1",
                                row.get("source_file", "")))
    return events, rejected


def read_metal_csv(path: Path) -> tuple[list[Event], int]:
    events, unmatched = [], 0
    with path.open(newline="", encoding="utf-8") as handle:
        metadata = handle.readline()
        match = re.search(r"CHIP_FREQ\[MHz\]:\s*([\d.]+)", metadata)
        if not match or float(match[1]) <= 0:
            raise ValueError(f"{path}: missing positive CHIP_FREQ[MHz]")
        ns_per_cycle = 1000 / float(match[1])
        reader = csv.DictReader(handle, skipinitialspace=True)
        def field(*names: str) -> str:
            for name in names:
                if name in (reader.fieldnames or []):
                    return name
            raise ValueError(f"{path}: missing column {names}")
        run = field("run host ID", "run id", "run ID")
        time = field("time[cycles since reset]")
        risc = field("RISC processor type")
        device = field("PCIe slot")
        for name in ("core_x", "core_y", "zone name", "type"):
            field(name)
        streams = defaultdict(list)
        for row in reader:
            phase = row["type"].strip().lower()
            if phase not in {"begin", "end", "start", "stop", "zone_start", "zone_end"}:
                continue
            key = int(row[device]), int(row["core_x"]), int(row["core_y"]), row[risc].strip(), row[run]
            streams[key].append((int(row[time]), phase, row))
        for key, markers in streams.items():
            stack = []
            for timestamp, phase, row in sorted(markers, key=lambda item: item[0]):
                name = row["zone name"].strip()
                if phase in {"begin", "start", "zone_start"}:
                    stack.append((name, timestamp, row.get("source file", "")))
                elif stack and stack[-1][0] == name:
                    _, start, source = stack.pop()
                    events.append(Event(*key[:4], name, round(start * ns_per_cycle),
                                        round(timestamp * ns_per_cycle), False, source))
                else:
                    unmatched += 1
            unmatched += len(stack)
    return events, unmatched


def read_capture(path: Path, exporter: Path | None, raw_output: Path) -> tuple[list[Event], int]:
    if path.suffix.lower() == ".tracy":
        if exporter is None:
            raise ValueError("Use --tracy-source SOURCE_DIR to build the device exporter, or --exporter BINARY")
        temporary = raw_output.with_suffix(".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            result = subprocess.run([str(exporter.resolve()), str(path.resolve())],
                                    stdout=handle, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            raise ValueError(f"{path}: {result.stderr.strip()}; use Tracy sources matching the capture version")
        temporary.replace(raw_output)
        return read_export(raw_output)
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
    return read_metal_csv(path) if "CHIP_FREQ[MHz]" in first else read_export(path)


def partition_map(events: list[Event], count: int, layout: str) -> dict:
    devices = {e.device for e in events}
    if len(devices) != 1:
        raise ValueError("Analyse one device per invocation; multiple device contexts were found")
    cores = sorted({e.core for e in events}, key=(lambda c: (c[2], c[1]))
                   if layout == "rows" else (lambda c: (c[1], c[2])))
    if len(cores) < 2 * count:
        raise ValueError("Each partition must contain at least a leader and a worker")
    mapping, offset = {}, 0
    for partition in range(count):
        size = len(cores) // count + (partition < len(cores) % count)
        for core in cores[offset:offset + size]:
            mapping[core] = partition
        offset += size
    return mapping


def select_fine(events: list[Event]) -> tuple[list[Event], list[Event]]:
    """Use last kernel on each RISC, excluding coarse kernels and warmup."""
    kernels = {}
    for event in events:
        if event.zone.endswith("-KERNEL"):
            if event.stream not in kernels or event.start > kernels[event.stream].start:
                kernels[event.stream] = event
    custom = []
    for event in events:
        if not event.zone.startswith(CUSTOM_PREFIXES):
            continue
        kernel = kernels.get(event.stream)
        if kernel and kernel.start <= event.start <= event.end <= kernel.end:
            custom.append(event)
    if not custom:
        raise ValueError("No IVF Worker/Leader/Partition scopes in the last kernel invocation")
    return sorted(custom, key=lambda e: (e.stream, e.start)), list(kernels.values())


def scope_summary(label: str, custom: list[Event], mapping: dict) -> list[dict]:
    groups = defaultdict(list)
    for e in custom:
        groups[(mapping[e.core], e.risc, e.zone)].append(e.us)
    return [dict(trace=label, partition=p, risc=risc, zone=zone, **stats(values))
            for (p, risc, zone), values in sorted(groups.items())]


def page_summary(label: str, custom: list[Event], mapping: dict) -> list[dict]:
    groups = defaultdict(list)
    for e in custom:
        groups[e.stream, e.zone].append(e)
    for values in groups.values():
        values.sort(key=lambda e: e.start)
    cores = sorted({e.core for e in custom if e.zone in
                    ("Worker issue DRAM pair", "Worker relay receive wait")})
    rows = []
    for core in cores:
        issue = groups[((*core, "NCRISC"), "Worker issue DRAM pair")]
        wait = groups[((*core, "NCRISC"), "Worker read wait")]
        receive = groups[((*core, "NCRISC"), "Worker relay receive wait")]
        relay = bool(receive)
        if relay and (issue or wait):
            raise ValueError(f"{label} core {core}: mixed direct and relay page scopes")
        sequences = {}
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            for name in ("Worker input wait", "Worker matmul", "Worker score pack",
                         "Worker score and ID wait", "Worker local topk"):
                sequences[risc, name] = groups[((*core, risc), name)]
        counts = {*(len(v) for v in sequences.values()),
                  *([len(receive)] if relay else [len(issue), len(wait)])}
        if len(counts) != 1:
            raise ValueError(f"{label} core {core}: incomplete per-page scopes, counts={sorted(counts)}")
        for page in range(len(receive) if relay else len(issue)):
            issued, retired = (None, receive[page]) if relay else (issue[page], wait[page])
            if issued and retired.start < issued.end:
                raise ValueError(f"{label} core {core}: read wait precedes paired issue")
            row = dict(trace=label, partition=mapping[core], device=core[0], core_x=core[1],
                       core_y=core[2], page_ordinal=page, transport="relay" if relay else "direct",
                       reader_wait_us=retired.us,
                       reader_issue_us=issued.us if issued else None,
                       reader_residual_wait_us=retired.us if not relay else None,
                       worker_relay_receive_wait_us=retired.us if relay else None,
                       issue_to_retirement_us=(retired.end - issued.start) / 1000 if issued else None)
            for (risc, name), values in sequences.items():
                row[risc.lower() + "_" + name.removeprefix("Worker ").replace(" ", "_") + "_us"] = values[page].us
            rows.append(row)
    return rows


def leader_page_summary(label: str, custom: list[Event], mapping: dict) -> list[dict]:
    """Relay leader scopes sample only worker zero, in its delivery order."""
    groups = defaultdict(list)
    for event in custom:
        if event.zone in ("Leader issue DRAM pair", "Leader read wait", "Leader forward pair"):
            groups[event.core, event.zone].append(event)
    rows = []
    for core in sorted({key[0] for key in groups}):
        sequences = [sorted(groups[core, name], key=lambda e: e.start) for name in
                     ("Leader issue DRAM pair", "Leader read wait", "Leader forward pair")]
        if len({len(values) for values in sequences}) != 1:
            raise ValueError(f"{label} leader {core}: incomplete relay per-page scopes")
        for page, (issue, wait, forward) in enumerate(zip(*sequences)):
            if wait.start < issue.end or forward.start < wait.end:
                raise ValueError(f"{label} leader {core}: relay wait/forward precedes paired read")
            rows.append(dict(trace=label, partition=mapping[core], device=core[0],
                             core_x=core[1], core_y=core[2], target_worker=0, page_ordinal=page,
                             leader_issue_us=issue.us, leader_residual_wait_us=wait.us,
                             leader_forward_us=forward.us,
                             issue_to_forward_completion_us=(forward.end - issue.start) / 1000))
    return rows


def completion_summary(label: str, kernels: list[Event], mapping: dict) -> list[dict]:
    # Local elapsed intervals remain meaningful even when cores are not synced.
    groups = defaultdict(list)
    for e in kernels:
        groups[mapping[e.core], e.risc].append(e.us / 1000)
    return [dict(trace=label, partition=p, risc=risc, cores=len(v),
                 kernel_median_ms=statistics.median(v), kernel_max_ms=max(v))
            for (p, risc), v in sorted(groups.items())]


def partition_details(label: str, data: dict, partition: int) -> tuple[list[dict], list[dict]]:
    """Preserve one shared origin; never align individual cores independently."""
    owned = [core for core, p in data["mapping"].items() if p == partition]
    selected = [e for e in data["custom"] if e.core in owned]
    if not selected:
        raise ValueError(f"{label}: partition {partition} has no detailed scopes")
    leaders = {e.core for e in selected if e.zone.startswith("Leader ")}
    aggregators = {e.core for e in selected if e.zone.startswith("Partition ")}
    if len(leaders) > 1 or len(aggregators) > 1:
        raise ValueError(f"{label}: partition {partition} has ambiguous leader/aggregator cores")
    roles, worker = {}, 0
    for core in owned:
        if core in leaders:
            roles[core] = "Leader / aggregator" if core in aggregators else "Leader"
        elif core in aggregators:
            roles[core] = "Aggregator"
        else:
            roles[core] = f"Worker {worker} (slave)"
            worker += 1
    origin = min(e.start for e in selected)
    rows = [dict(trace=label, partition=partition, device=e.device, core_x=e.x, core_y=e.y,
                 role=roles[e.core], risc=e.risc, zone=e.zone,
                 start_us=(e.start - origin) / 1000, end_us=(e.end - origin) / 1000,
                 duration_us=e.us, calibrated=e.calibrated)
            for e in sorted(selected, key=lambda event: (owned.index(event.core), event.risc, event.start))]
    cores = [dict(core=core, role=roles[core]) for core in owned]
    return rows, cores


def plot_partition(label: str, partition: int, details: list[dict], cores: list[dict],
                   output: Path, compute_riscs: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    riscs = ["NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "BRISC"]
    positions, tick_positions, tick_labels, separators = {}, [], [], []
    y = 0
    for index, entry in enumerate(cores):
        core, role = entry["core"], entry["role"]
        visible = riscs if compute_riscs == "all" or not role.startswith("Worker") else ["NCRISC", "TRISC_1", "BRISC"]
        for number, risc in enumerate(visible):
            positions[core, risc] = y
            prefix = f"{role} ({core[1]},{core[2]})" if number == 0 else ""
            tick_positions.append(y)
            tick_labels.append(f"{prefix}\n{risc} {RISC_ROLES[risc]}" if prefix else f"{risc} {RISC_ROLES[risc]}")
            y += 1
        if index < len(cores) - 1:
            separators.append(y - .2)
            y += .65
    full_end = max(row["end_us"] for row in details) * 1.02
    page_end = max((row["end_us"] for row in details if row["zone"] in COLORS), default=full_end) * 1.04
    calibrated = all(row["calibrated"] for row in details)
    for view, end, divisor, unit in (("overview", full_end, 1000, "ms"),
                                     ("pages", min(page_end, full_end), 1, "µs")):
        figure, axis = plt.subplots(figsize=(16, max(7, y * .28 + 1.8)))
        # Reserve stable space for two-line core labels and the outer legend.
        # Constrained layout can clip labels with dozens of closely spaced rows.
        figure.subplots_adjust(left=.19, right=.99, bottom=.065, top=.94)
        axis.set_xlim(0, end / divisor)
        axis.set_ylim(y - .25, -1)
        axis.set_yticks(tick_positions, tick_labels, fontsize=9)
        for separator in separators:
            axis.axhline(separator, color="#D5D8DD", linewidth=.6)
        # Draw outer gather first; its nested read scopes are real intervals.
        # CSV/statistics keep every scope separately instead of adding them.
        for row in sorted(details, key=lambda r: r["zone"] != "Partition gather"):
            key = (row["device"], row["core_x"], row["core_y"]), row["risc"]
            if key not in positions or row["start_us"] >= end:
                continue
            start, finish = row["start_us"] / divisor, min(row["end_us"], end) / divisor
            lane = positions[key]
            color = PARTITION_COLORS.get(row["zone"], "#D87737")
            height = .34 if row["zone"] == "Partition gather partial" else .58
            axis.broken_barh([(start, finish - start)], (lane - height / 2, height),
                             facecolors=color, edgecolors="none")
            width_fraction = (finish - start) / (end / divisor)
            if width_fraction < .0015:
                # A visible tick marks an event; it does not widen its duration.
                axis.vlines(start, lane - height / 2, lane + height / 2, color=color, linewidth=.8)
            name = PARTITION_LABELS.get(row["zone"], row["zone"])
            if width_fraction > max(.05, len(name) * .005):
                axis.text((start + finish) / 2, lane, name, ha="center", va="center", fontsize=8)
        axis.set_xlabel(f"{label} · partition {partition} · time from first selected partition scope ({unit})")
        axis.grid(axis="x", alpha=.2)
        axis.legend(handles=[Patch(facecolor=color, label=name) for color, name in (
            ("#7559E8", "Read / broadcast"), ("#B29BF2", "Query receive / fetch"),
            ("#B5BAC3", "Waiting"), ("#F2C500", "Compute / aggregation incl. sync"),
            ("#EE9824", "Local top-k"), ("#F45151", "Result writes"))],
            ncol=3, frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.01))
        note = "First 16 page pairs per worker are scoped; gaps after them are uninstrumented. Coordinates are physical NoC."
        if not calibrated:
            note += " Cross-core clock synchronization is not established."
        figure.text(.5, .012, note, ha="center", fontsize=9)
        target = output / f"{label}_partition_{partition}_{view}"
        figure.savefig(target.with_suffix(".png"), dpi=180)
        figure.savefig(target.with_suffix(".svg"))
        plt.close(figure)


def plots(captures: dict, pages: list[dict], leader_pages: list[dict], completions: list[dict], output: Path,
          partition_views: list[tuple], compute_riscs: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    for label, partition, details, cores in partition_views:
        plot_partition(label, partition, details, cores, output, compute_riscs)
    labels = list(captures)
    comparison_colors = ["#4E79A7", "#E38238", "#59A14F", "#B07AA1"]
    metrics = [("reader_wait_us", "Worker read / relay receive wait"),
               ("trisc_0_input_wait_us", "Unpacker input wait"),
               ("trisc_2_local_topk_us", "Packer local top-k")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for ax, (column, name) in zip(axes, metrics):
        values = [[row[column] for row in pages if row["trace"] == label] for label in labels]
        if not all(values):
            ax.set_visible(False)
            continue
        boxes = ax.boxplot(values, patch_artist=True, showfliers=True, widths=.5,
                           medianprops={"color": "black", "linewidth": 1.4}, showmeans=True,
                           meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
        for box, color in zip(boxes["boxes"], comparison_colors * len(labels)):
            box.set_facecolor(color)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_ylabel(name + " (µs)")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "page_waits.png", dpi=200)
    plt.close(fig)

    if leader_pages:
        relay_labels = [label for label in labels if any(r["trace"] == label for r in leader_pages)]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
        for ax, (column, title) in zip(axes, (
            ("leader_residual_wait_us", "Leader residual DRAM wait"),
            ("leader_forward_us", "Leader forward vectors + IDs"),
            ("issue_to_forward_completion_us", "Leader issue to forward completion"))):
            values = [[r[column] for r in leader_pages if r["trace"] == label] for label in relay_labels]
            boxes = ax.boxplot(values, patch_artist=True, showfliers=True, widths=.5,
                               medianprops={"color": "black", "linewidth": 1.4})
            for box, label in zip(boxes["boxes"], relay_labels):
                box.set_facecolor(comparison_colors[labels.index(label) % len(comparison_colors)])
            ax.set_xticks(range(1, len(relay_labels) + 1), relay_labels)
            ax.set_ylabel(title + " (µs)")
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Leader samples for worker 0 only")
        fig.savefig(output / "leader_relay_pages.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    partitions = sorted({row["partition"] for row in completions})
    width = .8 / len(labels)
    for index, label in enumerate(labels):
        selected = {row["partition"]: row["kernel_max_ms"] for row in completions
                    if row["trace"] == label and row["risc"] == "TRISC_1"}
        positions = [p + (index - (len(labels) - 1) / 2) * width for p in partitions]
        ax.bar(positions, [selected.get(p, math.nan) for p in partitions], width=width,
               color=comparison_colors[index % len(comparison_colors)], label=label)
    ax.set_xticks(partitions, [f"P{p}" for p in partitions])
    ax.set_xlabel("Partition (inferred from physical core order)")
    ax.set_ylabel("Maximum TRISC_1 kernel elapsed time (ms)")
    ax.legend(frameon=False, ncol=min(len(labels), 4), loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "partition_completion.png", dpi=200)
    plt.close(fig)

    # Same worker in each trace; local origin preserves timing without claiming
    # cross-core synchronization. Show all three TRISCs, which run concurrently.
    shared = set.intersection(*[{e.core for e in data["custom"] if e.zone in
                                ("Worker issue DRAM pair", "Worker relay receive wait")}
                                for data in captures.values()])
    if not shared:
        return
    core = min(shared)
    fig, axes = plt.subplots(len(labels), 1, figsize=(13, 5.2 * len(labels)),
                             squeeze=False, layout="constrained")
    for ax, (label, data) in zip(axes[:, 0], captures.items()):
        selected = [e for e in data["custom"] if e.core == core and e.zone in COLORS]
        origin = min(e.start for e in selected)
        riscs = ["NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
        for row, risc in enumerate(riscs):
            stream = [e for e in selected if e.risc == risc]
            for e in stream:
                ax.broken_barh([((e.start - origin) / 1000, e.us)], (row - .26, .52),
                               facecolors=COLORS[e.zone], edgecolors="none")
            numbered = [e for e in stream if e.zone in
                        (("Worker issue DRAM pair", "Worker relay receive wait")
                         if risc == "NCRISC" else ("Worker local topk",))]
            # Prefetch issues several pages within ~1 µs. Use ordinal ranges
            # for these bursts; individual overlapping digits obscure the trace.
            groups = []
            min_spacing = max((e.end - origin) / 1000 for e in selected) * .013
            for number, e in enumerate(numbered):
                x = (e.start - origin) / 1000
                if risc == "NCRISC" and groups and x - groups[-1][-1][1] < min_spacing:
                    groups[-1].append((number, x))
                else:
                    groups.append([(number, x)])
            for group in groups:
                first, last = group[0][0], group[-1][0]
                text = str(first) if first == last else f"{first}–{last}"
                ax.text(group[0][1], row - .32, text,
                        ha="left" if len(group) > 1 else "center", va="bottom", fontsize=8)
        ax.set_yticks(range(4), ["NCRISC reader", "TRISC_0 unpacker", "TRISC_1 math", "TRISC_2 packer"])
        ax.set_ylim(3.7, -.8)
        ax.set_xlim(0, 1.02 * max((e.end - origin) / 1000 for e in selected))
        ax.set_xlabel(f"{label} · physical core ({core[1]},{core[2]}) · time from first sampled page scope (µs)")
        ax.grid(axis="x", alpha=.2)
    present = {e.zone for data in captures.values() for e in data["custom"]}
    handles = [Patch(facecolor=COLORS[name], label=title) for name, title in (
        ("Worker issue DRAM pair", "Issue reads"), ("Worker read wait", "Read wait"),
        ("Worker relay receive wait", "Relay offer / acknowledgement"),
        ("Worker input wait", "Buffer/input wait"), ("Worker score and ID wait", "Score/ID wait"),
        ("Worker matmul", "Matmul"), ("Worker score pack", "Score pack"),
        ("Worker local topk", "Local top-k")) if name in present]
    axes[0, 0].legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02),
                      frameon=False, fontsize=10)
    fig.savefig(output / "worker_pages.png", dpi=200)
    plt.close(fig)


def report(captures: dict, summaries: list[dict], pages: list[dict], completions: list[dict], output: Path) -> None:
    lines = ["# IVF device scope comparison", "",
             "Partition IDs are inferred from the requested layout and physical core order; "
             "confirm against benchmark.log. The last kernel invocation on each RISC is used; "
             "coarse search and warmup are excluded. Input labels/configurations are supplied by the user.", "",
             "## Comparison", "",
             "| Capture | Worker wait scope | Wait mean (µs) | Wait p95 (µs) | Waits >10 µs | "
             "Unpacker input wait max (µs) | Packer top-k median (µs) | Full kernel max (ms) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, data in captures.items():
        def metric(risc: str, name: str) -> dict | None:
            values = [e.us for e in data["custom"] if e.risc == risc and e.zone == name]
            return stats(values) if values else None
        reader = metric("NCRISC", "Worker read wait")
        wait_kind = "DRAM residual"
        if reader is None:
            reader = metric("NCRISC", "Worker relay receive wait")
            wait_kind = "Relay offer / acknowledgement"
        unpacker = metric("TRISC_0", "Worker input wait")
        topk = metric("TRISC_2", "Worker local topk")
        full = max(r["kernel_max_ms"] for r in completions
                   if r["trace"] == label and r["risc"] == "TRISC_1")
        if reader and unpacker and topk:
            lines.append(f"| {label} | {wait_kind} | {reader['mean_us']:.3f} | {reader['p95_us']:.3f} | "
                         f"{reader['over_10_us']}/{reader['samples']} | {unpacker['max_us']:.3f} | "
                         f"{topk['median_us']:.3f} | {full:.3f} |")
    lines += ["", "The comparison samples one batch's initial pages. Reader residual waits and "
              "unpacker input waits expose different parts of the pipeline; their distributions "
              "must not be added as elapsed search time. Relay receive waits include the worker's "
              "offer, leader scheduling, read completion, forwarding and acknowledgement; they "
              "are not the same measurement as a direct reader's residual DRAM barrier.", ""]
    for label, data in captures.items():
        parts = sorted({data["mapping"][e.core] for e in data["custom"]})
        rows = [r for r in pages if r["trace"] == label]
        lines += [f"## {label}", "", f"Source: `{data['path']}`", "",
                  f"{len(data['events'])} valid device scopes; {len(data['custom'])} fine custom scopes. "
                  f"Detailed partitions: {parts}. Sampled reader pages: {len(rows)}. "
                  f"Rejected/unmatched markers: {data['rejected']}.", ""]
        if not all(e.calibrated for e in data["events"]):
            lines += ["Cross-core alignment is not established. Durations and the worker's local timeline "
                      "remain usable; partition plots show the raw clock alignment without correction.", ""]
        lines += ["| RISC | Scope | N | Mean (µs) | Median (µs) | p95 (µs) | Max (µs) | >10 µs |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in summaries:
            if r["trace"] == label and r["zone"] in PARTITION_COLORS:
                lines.append(f"| {r['risc']} | {r['zone']} | {r['samples']} | {r['mean_us']:.3f} | "
                             f"{r['median_us']:.3f} | {r['p95_us']:.3f} | {r['max_us']:.3f} | {r['over_10_us']} |")
        lines += [""]
    lines += ["## Interpretation limits", "",
              "- `Worker read wait` measures residual barrier waiting, not full DRAM latency. "
              "A lower median can coexist with a larger tail. `issue_to_retirement_us` in pages.csv "
              "also includes delays before the reader reaches the barrier.",
              "- TRISC_0/1/2 execute concurrently. Do not add their durations. Matmul scopes "
              "can issue asynchronous work; top-k scopes include synchronization and cannot be "
              "interpreted as pure arithmetic cost.",
              "- Per-page detail samples the first 16 page pairs per worker of the selected packed "
              "batch. Ordinals can cross list boundaries and are not cluster IDs. A number above "
              "a reader bar labels its read issue; closely spaced issue bursts use ordinal ranges. "
              "The matching barrier is paired by order in pages.csv.",
              "- In relay captures, workers receive vectors/IDs from their leader. `leader_pages.csv` "
              "pairs the leader's reads, residual barriers and forward completions for worker 0 only. "
              "Issue-to-forward duration includes scheduling/credit delays and is not pure DRAM latency. "
              "Other workers' receive waits are recorded independently in pages.csv; leader samples "
              "must not be assigned to those workers.",
              "- Partition completion plots show full kernel elapsed intervals, including waits. "
              "Only the detailed partitions have activity breakdowns; completion is not utilization.",
              "- The partition overview and page zoom include the leader, aggregator and all workers, "
              "with one shared time origin from the recorded timestamps. Worker reader/compute detail "
              "ends after 16 page pairs; subsequent gaps do not imply idle cores. Leader aggregation "
              "scopes include synchronization, and outer gather scopes contain their partial-read scopes.",
              "- These are individual profiled executions, potentially with different reader algorithms and "
              "buffer depths. They cannot isolate the depth effect or prove DRAM bandwidth saturation. "
              "Use the unprofiled sweeps for QPS.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, metavar="LABEL=FILE")
    parser.add_argument("--output", type=Path, default=Path("similarity_trace_analysis"))
    parser.add_argument("--tracy-source", type=Path)
    parser.add_argument("--exporter", type=Path)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--layout", choices=("rows", "columns"), default="rows")
    parser.add_argument("--partition-view", type=int, action="append", metavar="N",
                        help="partition timeline to plot (default: every partition with detailed scopes)")
    parser.add_argument("--partition-riscs", choices=("all", "math"), default="all",
                        help="all five RISC rows, or reader/math/writer for workers (default: all)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.partitions < 1:
        parser.error("--partitions must be positive")
    if args.partition_view and any(p < 0 or p >= args.partitions for p in args.partition_view):
        parser.error("--partition-view must be within 0..partitions-1")
    specifications = {}
    for item in args.trace:
        label, separator, path = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in specifications:
            parser.error("Each --trace needs a unique simple LABEL=FILE")
        specifications[label] = Path(path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    exporter = args.exporter or output / "export_tt_device_zones"
    try:
        native = any(path.suffix.lower() == ".tracy" for path in specifications.values())
        rebuild = (not args.exporter and args.tracy_source and exporter.is_file()
                   and (HERE / "export_tt_device_zones.cpp").stat().st_mtime > exporter.stat().st_mtime)
        if native and (not exporter.is_file() or rebuild):
            if not args.tracy_source:
                raise ValueError("Native captures need --tracy-source SOURCE_DIR or --exporter BINARY; see README")
            build_exporter(args.tracy_source, exporter)
        captures, summaries, pages, leader_pages, completions, mappings, partition_views = {}, [], [], [], [], [], []
        for label, path in specifications.items():
            print(f"Reading {label}: {path}", flush=True)
            events, rejected = read_capture(path, exporter, output / "raw" / f"{label}_device.csv")
            custom, kernels = select_fine(events)
            mapping = partition_map(kernels, args.partitions, args.layout)
            captures[label] = dict(events=events, custom=custom, mapping=mapping,
                                   path=path.resolve(), rejected=rejected)
            selected_partitions = sorted(set(args.partition_view) if args.partition_view is not None
                                         else {mapping[e.core] for e in custom})
            for partition in selected_partitions:
                details, cores = partition_details(label, captures[label], partition)
                partition_views.append((label, partition, details, cores))
            summaries += scope_summary(label, custom, mapping)
            pages += page_summary(label, custom, mapping)
            leader_pages += leader_page_summary(label, custom, mapping)
            completions += completion_summary(label, kernels, mapping)
            mappings += [dict(trace=label, device=d, core_x=x, core_y=y, partition=p)
                         for (d, x, y), p in sorted(mapping.items())]
        write_csv(output / "scope_summary.csv", summaries)
        write_csv(output / "pages.csv", pages)
        write_csv(output / "leader_pages.csv", leader_pages)
        write_csv(output / "partition_completion.csv", completions)
        write_csv(output / "partition_mapping.csv", mappings)
        write_csv(output / "partition_scopes.csv", [row for _, _, rows, _ in partition_views for row in rows])
        report(captures, summaries, pages, completions, output)
        if not args.no_plots:
            plots(captures, pages, leader_pages, completions, output, partition_views, args.partition_riscs)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(f"Report, CSVs and plots: {output}")


if __name__ == "__main__":
    main()
