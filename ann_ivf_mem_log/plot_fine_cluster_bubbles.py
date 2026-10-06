#!/usr/bin/env python3

"""Plot reader/compute overlap for one profiled IVF fine-search cluster."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from plot_device_kernel_timeline import (
    PLOTTING_IMPORT_ERROR,
    Patch,
    discover_profile_log,
    find_column,
    find_exact_column,
    normalize_risc,
    parse_core,
    plt,
)
from ivf_plot_style import (
    COMPUTE_COLOR,
    QUERY_READ_COLOR,
    READER_COLOR,
    RISC_STAGE,
    WAIT_COLOR,
    add_kernel_legend,
)


ZONE_STYLES = {
    "FineBubble.Reader.Query": ("query read", QUERY_READ_COLOR),
    "FineBubble.Reader.Cluster": ("reader cluster", READER_COLOR),
    "FineBubble.Reader.Fetch": ("reader page fetch (vector + ID)", READER_COLOR),
    "FineBubble.Compute.QueryWait": ("query wait", WAIT_COLOR),
    "FineBubble.Compute.Cluster": ("compute cluster", COMPUTE_COLOR),
    "FineBubble.Compute.Work": ("compute work", COMPUTE_COLOR),
}
QUERY_ZONES = {
    "FineBubble.Reader.Query",
    "FineBubble.Compute.QueryWait",
}
CLUSTER_ZONES = {
    "FineBubble.Reader.Cluster",
    "FineBubble.Compute.Cluster",
}
BLOCK_ZONES = {
    "FineBubble.Reader.Fetch",
    "FineBubble.Compute.Work",
}
START_PHASES = {"begin", "start", "zone_start"}
END_PHASES = {"end", "stop", "zone_end"}
RISC_ORDER = {"NCRISC": 0, "TRISC_0": 1, "TRISC_1": 2, "TRISC_2": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot direct FineBubble.* DeviceZoneScopedN intervals for one "
            "worker invocation and cluster."
        )
    )
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument(
        "--core",
        metavar="X,Y",
        help="Profiler worker coordinate; defaults to the first core with FineBubble scopes",
    )
    parser.add_argument("--invocation", choices=("first", "last"), default="last")
    parser.add_argument(
        "--cluster-index",
        type=int,
        default=0,
        help="Zero-based active cluster occurrence within the selected invocation",
    )
    parser.add_argument(
        "--block-start",
        type=int,
        default=0,
        help="First vector block to display within the selected cluster",
    )
    parser.add_argument(
        "--block-count",
        type=int,
        default=16,
        help="Number of vector blocks to display; zero plots the cluster remainder",
    )
    parser.add_argument("--unit", choices=("cycles", "us"), default="us")
    parser.add_argument("--output", type=Path, default=Path("fine_cluster_bubbles.png"))
    parser.add_argument("--events-output", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_bubble_events(profile_log: Path) -> tuple[float, list[dict[str, object]], int]:
    with profile_log.open(newline="") as source:
        rows = list(csv.reader(source))

    frequency_mhz: float | None = None
    header_index: int | None = None
    for row_index, row in enumerate(rows):
        match = re.search(r"CHIP_FREQ\[MHz\]\s*:\s*([0-9.]+)", ",".join(row))
        if match:
            frequency_mhz = float(match.group(1))
        if any("zone name" in field.strip().lower() for field in row):
            header_index = row_index
            break
    if frequency_mhz is None:
        raise ValueError(f"{profile_log}: CHIP_FREQ[MHz] was not found")
    if header_index is None:
        raise ValueError(f"{profile_log}: profiler CSV header was not found")

    header = rows[header_index]
    slot_col = find_column(header, "pcie", "slot")
    core_x_col = find_column(header, "core", "x")
    core_y_col = find_column(header, "core", "y")
    risc_col = find_column(header, "risc", "processor")
    time_col = find_column(header, "time", "cycle")
    run_col = find_column(header, "run", "id")
    zone_col = find_column(header, "zone", "name")
    try:
        type_col = find_exact_column(header, "type", "zone phase")
    except ValueError:
        type_col = find_column(header, "zone", "phase")
    needed = max(slot_col, core_x_col, core_y_col, risc_col, time_col, run_col, zone_col, type_col)

    markers_by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    marker_type_counts: dict[str, int] = defaultdict(int)
    zone_counts: dict[str, int] = defaultdict(int)
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if len(row) <= needed:
            continue
        zone = row[zone_col].strip()
        if zone not in ZONE_STYLES:
            continue
        marker_type = row[type_col].strip().lower()
        marker_type_counts[marker_type] += 1
        zone_counts[zone] += 1
        try:
            cycle = int(row[time_col].strip())
            core_x = int(row[core_x_col].strip())
            core_y = int(row[core_y_col].strip())
        except ValueError:
            continue
        marker = {
            "pcie_slot": row[slot_col].strip(),
            "core_x": core_x,
            "core_y": core_y,
            "risc": normalize_risc(row[risc_col]),
            "run_id": row[run_col].strip(),
            "zone": zone,
            "marker_type": marker_type,
            "cycle": cycle,
            "row_number": row_number,
        }
        stream = (
            marker["pcie_slot"],
            core_x,
            core_y,
            marker["risc"],
            marker["run_id"],
        )
        markers_by_stream[stream].append(marker)

    events: list[dict[str, object]] = []
    unmatched = 0
    for markers in markers_by_stream.values():
        markers.sort(key=lambda marker: (int(marker["cycle"]), int(marker["row_number"])))
        open_zones: dict[str, list[int]] = defaultdict(list)
        for marker in markers:
            zone = str(marker["zone"])
            marker_type = str(marker["marker_type"])
            if marker_type in START_PHASES:
                open_zones[zone].append(int(marker["cycle"]))
            elif marker_type in END_PHASES:
                if not open_zones[zone]:
                    unmatched += 1
                    continue
                start = open_zones[zone].pop()
                end = int(marker["cycle"])
                if end < start:
                    continue
                label, color = ZONE_STYLES[zone]
                event = dict(marker)
                event.update(
                    {
                        "start_cycle": start,
                        "end_cycle": end,
                        "duration_cycles": end - start,
                        "phase": label,
                        "color": color,
                        "block": None,
                    }
                )
                events.append(event)
    if not events:
        if not zone_counts:
            detail = "the CSV contains zero FineBubble.* rows"
        else:
            types = ", ".join(f"{name}={count}" for name, count in sorted(marker_type_counts.items()))
            zones = ", ".join(f"{name}={count}" for name, count in sorted(zone_counts.items()))
            detail = f"marker types: {types}; zones: {zones}"
        raise RuntimeError(
            f"No paired FineBubble.* scopes were found ({detail}). Rebuild the executable and run "
            "a fresh short workload with TT_METAL_DEVICE_PROFILER=1."
        )
    return frequency_mhz, events, unmatched


def select_core_cluster(
    events: list[dict[str, object]],
    core: tuple[int, int],
    invocation: str,
    cluster_index: int,
) -> list[dict[str, object]]:
    core_events = [
        event
        for event in events
        if (int(event["core_x"]), int(event["core_y"])) == core
    ]
    if not core_events:
        available = sorted({(int(event["core_x"]), int(event["core_y"])) for event in events})
        available_text = ", ".join(f"{x},{y}" for x, y in available)
        raise RuntimeError(f"Core({core[0]},{core[1]}) has no FineBubble scopes; available: {available_text}")

    reader_anchors = sorted(
        [event for event in core_events if str(event["zone"]) == "FineBubble.Reader.Query"],
        key=lambda event: int(event["start_cycle"]),
    )
    if not reader_anchors:
        raise RuntimeError(f"Core({core[0]},{core[1]}) has no reader query invocation anchors")
    if len(reader_anchors) <= 1:
        invocation_events = core_events
    else:
        anchor_index = 0 if invocation == "first" else len(reader_anchors) - 1
        anchor = reader_anchors[anchor_index]
        query_candidates = [
            event
            for event in core_events
            if str(event["zone"]) in QUERY_ZONES
            and (
                (anchor_index == 0 and int(event["start_cycle"]) < int(reader_anchors[1]["start_cycle"]))
                or (
                    anchor_index > 0
                    and int(event["start_cycle"]) >= int(reader_anchors[anchor_index - 1]["end_cycle"])
                )
            )
        ]
        start = min(
            [int(anchor["start_cycle"]), *[int(event["start_cycle"]) for event in query_candidates]]
        )
        end = (
            int(reader_anchors[anchor_index + 1]["start_cycle"])
            if anchor_index + 1 < len(reader_anchors)
            else 2**63 - 1
        )
        invocation_events = [
            event for event in core_events if start <= int(event["start_cycle"]) < end
        ]

    reader_clusters = sorted(
        [
            event
            for event in invocation_events
            if str(event["zone"]) == "FineBubble.Reader.Cluster"
        ],
        key=lambda event: int(event["start_cycle"]),
    )
    if cluster_index >= len(reader_clusters):
        raise RuntimeError(
            f"Core({core[0]},{core[1]}) invocation has {len(reader_clusters)} active "
            f"reader cluster(s); --cluster-index {cluster_index} is out of range"
        )
    reader_cluster = reader_clusters[cluster_index]

    selected = [event for event in invocation_events if str(event["zone"]) in QUERY_ZONES]
    selected.extend(
        event
        for event in invocation_events
        if str(event["zone"]) == "FineBubble.Reader.Fetch"
        and int(reader_cluster["start_cycle"]) <= int(event["start_cycle"])
        and int(event["end_cycle"]) <= int(reader_cluster["end_cycle"])
    )

    for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
        compute_clusters = sorted(
            [
                event
                for event in invocation_events
                if str(event["risc"]) == risc
                and str(event["zone"]) == "FineBubble.Compute.Cluster"
            ],
            key=lambda event: int(event["start_cycle"]),
        )
        if cluster_index >= len(compute_clusters):
            continue
        compute_cluster = compute_clusters[cluster_index]
        selected.extend(
            event
            for event in invocation_events
            if str(event["risc"]) == risc
            and str(event["zone"]) == "FineBubble.Compute.Work"
            and int(compute_cluster["start_cycle"]) <= int(event["start_cycle"])
            and int(event["end_cycle"]) <= int(compute_cluster["end_cycle"])
        )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for event in selected:
        if str(event["zone"]) in BLOCK_ZONES:
            grouped[(str(event["risc"]), str(event["zone"]))].append(event)
    for phase_events in grouped.values():
        phase_events.sort(key=lambda event: int(event["start_cycle"]))
        for block, event in enumerate(phase_events):
            event["block"] = block
    return selected


def filter_blocks(
    events: list[dict[str, object]], block_start: int, block_count: int
) -> list[dict[str, object]]:
    block_end = 2**31 if block_count == 0 else block_start + block_count
    return [
        event
        for event in events
        if (
            (event["block"] is None and block_start == 0)
            or (
                event["block"] is not None
                and block_start <= int(event["block"]) < block_end
            )
        )
    ]


def plot_events(
    events: list[dict[str, object]],
    frequency_mhz: float,
    core: tuple[int, int],
    unit: str,
    output: Path,
    dpi: int,
) -> None:
    if PLOTTING_IMPORT_ERROR is not None:
        raise RuntimeError(f"Matplotlib is required: {PLOTTING_IMPORT_ERROR}")
    assert plt is not None
    assert Patch is not None

    riscs = sorted({str(event["risc"]) for event in events}, key=lambda risc: RISC_ORDER.get(risc, 99))
    y_by_risc = {risc: float(index) for index, risc in enumerate(riscs)}
    origin = min(int(event["start_cycle"]) for event in events)
    end = max(int(event["end_cycle"]) for event in events)
    scale = 1.0 if unit == "cycles" else 1.0 / frequency_mhz
    width = max((end - origin) * scale, 1.0)

    fig, ax = plt.subplots(figsize=(13.5, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for index, risc in enumerate(riscs):
        ax.axhspan(index - 0.42, index + 0.42, color="#F3F5F7" if index % 2 == 0 else "white", zorder=0)

    for event in sorted(events, key=lambda item: int(item["start_cycle"])):
        risc = str(event["risc"])
        start = (int(event["start_cycle"]) - origin) * scale
        duration = max(int(event["duration_cycles"]) * scale, 0.001)
        ax.barh(
            y_by_risc[risc],
            duration,
            left=start,
            height=0.66,
            color=str(event["color"]),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        zone = str(event["zone"])
        block = event["block"]
        if zone in {"FineBubble.Reader.Query", "FineBubble.Reader.Fetch"}:
            # Every reader page needs an identifier, even when the fetch bar
            # is too short to fit text. Q distinguishes the query read from P0.
            label = "Q" if block is None else f"P{int(block)}"
            ax.annotate(
                label,
                (start + duration / 2.0, y_by_risc[risc] + 0.32),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.5,
                color="#202020",
                annotation_clip=False,
                zorder=5,
            )
        elif duration / width >= 0.018:
            label = str(event["phase"]) if block is None else f"P{int(block)}"
            ax.text(
                start + duration / 2.0,
                y_by_risc[risc],
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="#202020" if str(event["color"]) in {"#F2C500", "#F39C34", "#AAB2BD"} else "white",
                clip_on=True,
                zorder=4,
            )

    labels = [f"Core({core[0]},{core[1]})-{risc} {RISC_STAGE.get(risc, '')}".rstrip() for risc in riscs]
    ax.set_yticks([y_by_risc[risc] for risc in riscs], labels)
    ax.set_xlim(0, width * 1.02)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel("Cycles since first selected scope" if unit == "cycles" else "Time since first selected scope (µs)")
    ax.set_ylabel("Tensix core + RISC")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, axis="x", color="#AAB0B6", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    present_zones = {str(event["zone"]) for event in events}
    legend = [
        Patch(facecolor=color, label=label)
        for zone, label, color in (
            ("FineBubble.Reader.Query", "Query read (Q)", QUERY_READ_COLOR),
            ("FineBubble.Reader.Fetch", "Reader page (vector + ID)", READER_COLOR),
            ("FineBubble.Compute.Work", "Compute work", COMPUTE_COLOR),
            ("FineBubble.Compute.QueryWait", "Query wait", WAIT_COLOR),
        )
        if zone in present_zones
    ]
    add_kernel_legend(ax, legend)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_events(
    path: Path,
    events: list[dict[str, object]],
    frequency_mhz: float,
) -> None:
    origin = min(int(event["start_cycle"]) for event in events)
    fields = [
        "pcie_slot", "core_x", "core_y", "risc", "run_id", "zone", "phase",
        "block", "start_cycle", "end_cycle", "duration_cycles", "start_us", "end_us", "duration_us",
        "profile_start_cycle", "profile_end_cycle",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for event in sorted(events, key=lambda item: (int(item["start_cycle"]), str(item["risc"]))):
            start = int(event["start_cycle"])
            end = int(event["end_cycle"])
            block = event["block"]
            writer.writerow(
                {
                    "pcie_slot": event["pcie_slot"],
                    "core_x": event["core_x"],
                    "core_y": event["core_y"],
                    "risc": event["risc"],
                    "run_id": event["run_id"],
                    "zone": event["zone"],
                    "phase": event["phase"],
                    "block": "" if block is None else int(block),
                    "start_cycle": start - origin,
                    "end_cycle": end - origin,
                    "duration_cycles": end - start,
                    "start_us": f"{(start - origin) / frequency_mhz:.6f}",
                    "end_us": f"{(end - origin) / frequency_mhz:.6f}",
                    "duration_us": f"{(end - start) / frequency_mhz:.6f}",
                    "profile_start_cycle": start,
                    "profile_end_cycle": end,
                }
            )


def print_summary(events: list[dict[str, object]], frequency_mhz: float) -> None:
    by_phase: dict[str, list[int]] = defaultdict(list)
    for event in events:
        by_phase[str(event["phase"])].append(int(event["duration_cycles"]))
    print("Selected phase durations:")
    for phase, durations in sorted(by_phase.items()):
        total = sum(durations) / frequency_mhz
        average = total / len(durations)
        print(f"  {phase:16s} count={len(durations):2d} total={total:9.3f} us avg={average:8.3f} us")


def main() -> None:
    args = parse_args()
    if args.cluster_index < 0:
        raise ValueError("--cluster-index cannot be negative")
    if args.block_start < 0:
        raise ValueError("--block-start cannot be negative")
    if args.block_count < 0:
        raise ValueError("--block-count cannot be negative")
    profile_log = discover_profile_log(args.profile_log)
    print(f"Reading fine-cluster bubble scopes: {profile_log}")
    frequency_mhz, events, unmatched = read_bubble_events(profile_log)
    core = (
        parse_core(args.core)
        if args.core
        else min({(int(event["core_x"]), int(event["core_y"])) for event in events})
    )
    print(f"Selected profiler worker Core({core[0]},{core[1]})")
    selected = select_core_cluster(events, core, args.invocation, args.cluster_index)
    selected = filter_blocks(selected, args.block_start, args.block_count)
    if not selected:
        raise RuntimeError("No FineBubble scopes remain after block filtering")

    events_output = args.events_output or args.output.with_suffix(".csv")
    write_events(events_output, selected, frequency_mhz)
    plot_events(selected, frequency_mhz, core, args.unit, args.output, args.dpi)
    print_summary(selected, frequency_mhz)
    if unmatched:
        print(f"Warning: ignored {unmatched} unmatched FineBubble zone ends")
    print(f"Wrote event data to {events_output}")
    print(f"Wrote timeline to {args.output}")


if __name__ == "__main__":
    main()
