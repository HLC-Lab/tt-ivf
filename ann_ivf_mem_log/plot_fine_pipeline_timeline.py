#!/usr/bin/env python3

"""Plot the current ANN IVF scoped zones as a multi-batch RISC pipeline."""

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
    output_page_path,
    parse_core,
    plt,
)
from ivf_plot_style import (
    COMPUTE_COLOR,
    QUERY_READ_COLOR,
    READER_COLOR,
    RESULT_COLOR,
    RISC_STAGE,
    WAIT_COLOR,
    add_kernel_legend,
)


WORKER_ZONES = {"Query", "Cluster", "Send Res"}
AGGREGATOR_ZONES = {"Partial Res", "Compute Res", "Res"}
PIPELINE_ZONES = WORKER_ZONES | AGGREGATOR_ZONES
START_PHASES = {"begin", "start", "zone_start"}
END_PHASES = {"end", "stop", "zone_end"}

PHASE_COLORS = {
    "Query read": QUERY_READ_COLOR,
    "Query wait": WAIT_COLOR,
    "Cluster read": READER_COLOR,
    "Cluster compute": COMPUTE_COLOR,
    "Send result": RESULT_COLOR,
    "Gather partials": READER_COLOR,
    "Reduce partials": COMPUTE_COLOR,
    "Write result": RESULT_COLOR,
}
RISC_ORDER = {"BRISC": 0, "NCRISC": 1, "TRISC_0": 2, "TRISC_1": 3, "TRISC_2": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Query, Cluster, Send Res, Partial Res, Compute Res, and Res "
            "DeviceZoneScopedN intervals as aligned 32-query batches."
        )
    )
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ann_ivf_fine_pipeline_timeline.png"))
    parser.add_argument("--events-output", type=Path)
    parser.add_argument(
        "--aggregator-core",
        metavar="X,Y",
        help="Override automatic detection; the normal Wormhole profiler coordinate is 1,1",
    )
    parser.add_argument(
        "--core",
        action="append",
        default=[],
        metavar="X,Y",
        help="Worker profiler coordinate; repeat as needed",
    )
    parser.add_argument("--worker-count", type=int, default=3)
    parser.add_argument(
        "--last-batches",
        type=int,
        default=10,
        help="Number of final 32-query batches to select; use 2 for 64 queries and 10 for 320",
    )
    parser.add_argument("--run-id", help="Optional profiler run host ID")
    parser.add_argument(
        "--view",
        choices=("trace", "phases"),
        default="trace",
        help="Both choices show only the real scoped intervals",
    )
    parser.add_argument(
        "--label-mode",
        choices=("compact", "phase", "none"),
        default="compact",
    )
    parser.add_argument(
        "--cores-per-page",
        type=int,
        default=4,
        help="Aggregator plus three workers per image by default",
    )
    parser.add_argument("--unit", choices=("cycles", "us"), default="us")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _optional_exact_column(header: list[str], *names: str) -> int | None:
    try:
        return find_exact_column(header, *names)
    except ValueError:
        return None


def core_key(item: dict[str, object]) -> tuple[str, int, int]:
    return str(item["pcie_slot"]), int(item["core_x"]), int(item["core_y"])


def stream_key(item: dict[str, object], include_zone: bool = False) -> tuple[object, ...]:
    key: tuple[object, ...] = (
        item["pcie_slot"],
        item["core_x"],
        item["core_y"],
        item["risc"],
        item["run_id"],
    )
    return key + (item["zone"],) if include_zone else key


def role_for_risc(risc: str) -> str:
    if risc == "NCRISC":
        return "reader"
    if risc == "BRISC":
        return "writer"
    return "compute"


def phase_for_event(event: dict[str, object]) -> str:
    zone = str(event["zone"])
    risc = str(event["risc"])
    if zone == "Query":
        return "Query read" if risc == "NCRISC" else "Query wait"
    if zone == "Cluster":
        return "Cluster read" if risc == "NCRISC" else "Cluster compute"
    return {
        "Send Res": "Send result",
        "Partial Res": "Gather partials",
        "Compute Res": "Reduce partials",
        "Res": "Write result",
    }.get(zone, zone)


def read_scoped_events(profile_log: Path) -> tuple[float, list[dict[str, object]], int]:
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
    trace_col = _optional_exact_column(header, "trace id")
    trace_counter_col = _optional_exact_column(header, "trace id counter")
    needed = max(slot_col, core_x_col, core_y_col, risc_col, time_col, run_col, zone_col, type_col)

    raw_by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if len(row) <= needed:
            continue
        zone = row[zone_col].strip()
        if zone not in PIPELINE_ZONES:
            continue
        try:
            cycle = int(row[time_col].strip())
            core_x = int(row[core_x_col].strip())
            core_y = int(row[core_y_col].strip())
        except ValueError:
            continue
        item = {
            "pcie_slot": row[slot_col].strip(),
            "core_x": core_x,
            "core_y": core_y,
            "risc": normalize_risc(row[risc_col]),
            "run_id": row[run_col].strip(),
            "trace_id": row[trace_col].strip() if trace_col is not None and len(row) > trace_col else "",
            "trace_counter": (
                row[trace_counter_col].strip()
                if trace_counter_col is not None and len(row) > trace_counter_col
                else ""
            ),
            "zone": zone,
            "marker_type": row[type_col].strip().lower(),
            "cycle": cycle,
            "row_number": row_number,
        }
        raw_key = stream_key(item) + (item["trace_id"], item["trace_counter"])
        raw_by_stream[raw_key].append(item)

    events: list[dict[str, object]] = []
    unmatched = 0
    for rows_for_stream in raw_by_stream.values():
        rows_for_stream.sort(key=lambda item: (int(item["cycle"]), int(item["row_number"])))
        open_zones: dict[str, list[int]] = defaultdict(list)
        for item in rows_for_stream:
            zone = str(item["zone"])
            marker_type = str(item["marker_type"])
            if marker_type in START_PHASES:
                open_zones[zone].append(int(item["cycle"]))
            elif marker_type in END_PHASES:
                if not open_zones[zone]:
                    unmatched += 1
                    continue
                start = open_zones[zone].pop()
                end = int(item["cycle"])
                if end >= start:
                    event = dict(item)
                    event.update(
                        {
                            "start_cycle": start,
                            "end_cycle": end,
                            "duration_cycles": end - start,
                            "phase": phase_for_event(item),
                            "role": role_for_risc(str(item["risc"])),
                            "kind": "phase",
                            "batch_id": None,
                        }
                    )
                    events.append(event)
    if not events:
        raise RuntimeError(
            "No current fine-search scoped zones were paired. Capture a fresh log containing "
            "Query, Cluster, Send Res, Partial Res, Compute Res, and Res."
        )
    return frequency_mhz, events, unmatched


def detect_aggregator(
    events: list[dict[str, object]], requested: tuple[int, int] | None
) -> tuple[str, int, int]:
    if requested is not None:
        matches = sorted({core_key(event) for event in events if core_key(event)[1:] == requested})
        if not matches:
            raise RuntimeError(f"Aggregator Core({requested[0]},{requested[1]}) was not found")
        return matches[0]
    counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for event in events:
        if str(event["zone"]) in AGGREGATOR_ZONES:
            counts[core_key(event)] += 1
    if not counts:
        raise RuntimeError("Could not detect the aggregator from Partial Res/Compute Res/Res")
    return max(counts, key=counts.get)


def choose_cores(
    events: list[dict[str, object]],
    aggregator: tuple[str, int, int],
    requested_workers: set[tuple[int, int]],
    worker_count: int,
) -> list[tuple[str, int, int]]:
    workers = sorted(
        {core_key(event) for event in events if core_key(event) != aggregator and str(event["zone"]) in WORKER_ZONES},
        key=lambda core: (
            0 if core[1] == aggregator[1] and core[2] > aggregator[2] else 1,
            core[1], core[2], core[0],
        ),
    )
    if requested_workers:
        selected = [core for core in workers if core[1:] in requested_workers]
        missing = requested_workers - {core[1:] for core in selected}
        if missing:
            missing_text = ", ".join(f"{x},{y}" for x, y in sorted(missing))
            raise RuntimeError(f"Requested worker core(s) were not found: {missing_text}")
    else:
        selected = workers[:worker_count]
    return [aggregator, *selected]


def _assign_tail(events: list[dict[str, object]], batches: list[int]) -> None:
    tail = sorted(events, key=lambda event: int(event["start_cycle"]))[-len(batches) :]
    offset = len(batches) - len(tail)
    for index, event in enumerate(tail):
        event["batch_id"] = batches[offset + index]


def assign_batches_from_scopes(
    events: list[dict[str, object]],
    selected_cores: list[tuple[str, int, int]],
    aggregator: tuple[str, int, int],
    last_batches: int,
) -> tuple[list[dict[str, object]], list[int], list[str]]:
    core_set = set(selected_cores)
    events = [event for event in events if core_key(event) in core_set]
    aggregator_completions = sorted(
        [event for event in events if core_key(event) == aggregator and str(event["zone"]) == "Res"],
        key=lambda event: int(event["start_cycle"]),
    )
    if not aggregator_completions:
        raise RuntimeError("No aggregator Res zones were found")
    selected_count = min(last_batches, len(aggregator_completions))
    batches = list(range(selected_count))
    warnings: list[str] = []

    # These zones occur exactly once per batch on their respective streams.
    one_per_batch: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if str(event["zone"]) != "Cluster":
            one_per_batch[stream_key(event, include_zone=True)].append(event)
    for stream_events in one_per_batch.values():
        _assign_tail(stream_events, batches)

    # Compute executes one Cluster scope for every assigned task, including an
    # inactive task. The number of task scopes per batch is therefore constant.
    for core in selected_cores[1:]:
        sends = sorted(
            [event for event in events if core_key(event) == core and str(event["zone"]) == "Send Res"],
            key=lambda event: int(event["start_cycle"]),
        )
        selected_sends = sends[-selected_count:]
        # The warmup search can assign a different number of tasks than the
        # measured search. The Send Res immediately preceding the requested
        # tail is therefore the invocation boundary for scoped-only grouping.
        previous_send_end = (
            int(sends[-selected_count - 1]["end_cycle"])
            if len(sends) > selected_count
            else -1
        )
        selected_end = (
            int(selected_sends[-1]["end_cycle"])
            if selected_sends
            else 2**63 - 1
        )
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            clusters = sorted(
                [
                    event for event in events
                    if core_key(event) == core and str(event["risc"]) == risc and str(event["zone"]) == "Cluster"
                    and previous_send_end < int(event["start_cycle"]) <= selected_end
                ],
                key=lambda event: int(event["start_cycle"]),
            )
            if not clusters or selected_count == 0:
                continue
            scopes_per_batch, remainder = divmod(len(clusters), selected_count)
            if scopes_per_batch == 0 or remainder:
                warnings.append(
                    f"Core({core[1]},{core[2]}) {risc}: {len(clusters)} Cluster scopes "
                    f"cannot be divided across {selected_count} selected Send Res anchors"
                )
                continue
            groups = [
                clusters[index * scopes_per_batch : (index + 1) * scopes_per_batch]
                for index in range(selected_count)
            ]
            group_offset = 0
            for group_index, group in enumerate(groups):
                for event in group:
                    event["batch_id"] = batches[group_offset + group_index]

        # Reader Cluster scopes exist only for active tasks. Query is the
        # delimiter emitted immediately before the first active task.
        reader_queries = sorted(
            [
                event for event in events
                if core_key(event) == core and str(event["risc"]) == "NCRISC" and str(event["zone"]) == "Query"
                and previous_send_end < int(event["start_cycle"]) <= selected_end
            ],
            key=lambda event: int(event["start_cycle"]),
        )
        reader_clusters = sorted(
            [
                event for event in events
                if core_key(event) == core and str(event["risc"]) == "NCRISC" and str(event["zone"]) == "Cluster"
                and previous_send_end < int(event["start_cycle"]) <= selected_end
            ],
            key=lambda event: int(event["start_cycle"]),
        )
        selected_queries = reader_queries[-selected_count:]
        query_offset = selected_count - len(selected_queries)
        for query_index, query in enumerate(selected_queries):
            batch_id = batches[query_offset + query_index]
            query["batch_id"] = batch_id
            next_start = (
                int(selected_queries[query_index + 1]["start_cycle"])
                if query_index + 1 < len(selected_queries)
                else 2**63 - 1
            )
            for cluster in reader_clusters:
                if int(query["start_cycle"]) <= int(cluster["start_cycle"]) < next_start:
                    cluster["batch_id"] = batch_id

    selected = [event for event in events if event["batch_id"] is not None]
    return selected, batches, warnings


def page_cores(
    selected_cores: list[tuple[str, int, int]], cores_per_page: int
) -> list[list[tuple[str, int, int]]]:
    aggregator = selected_cores[0]
    workers = selected_cores[1:]
    if not workers:
        return [[aggregator]]
    per_page = max(1, cores_per_page - 1)
    return [[aggregator, *workers[index : index + per_page]] for index in range(0, len(workers), per_page)]


def add_phase_ordinals(events: list[dict[str, object]]) -> None:
    # Reader Cluster scopes exist only for lists that contain data. Compute,
    # however, emits a Cluster scope for every scheduled slot, including its
    # very short inactive slots. Numbering every RISC independently therefore
    # gives the same list different labels (for example reader C3 versus
    # compute C4). Use the reader's active-list count and the longest compute
    # scopes to identify the matching active slots, then apply one ordinal map
    # to all three TRISCs.
    for event in events:
        event.pop("phase_ordinal", None)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if str(event["zone"]) != "Cluster":
            continue
        grouped[
            (
                event["pcie_slot"],
                event["core_x"],
                event["core_y"],
                event["run_id"],
                int(event["batch_id"]),
            )
        ].append(event)

    for group in grouped.values():
        reader = sorted(
            [event for event in group if str(event["risc"]) == "NCRISC"],
            key=lambda event: int(event["start_cycle"]),
        )
        for ordinal, event in enumerate(reader, start=1):
            event["phase_ordinal"] = ordinal

        compute_by_risc = {
            risc: sorted(
                [event for event in group if str(event["risc"]) == risc],
                key=lambda event: int(event["start_cycle"]),
            )
            for risc in ("TRISC_0", "TRISC_1", "TRISC_2")
        }
        compute_by_risc = {risc: scopes for risc, scopes in compute_by_risc.items() if scopes}
        if not compute_by_risc:
            continue

        canonical = compute_by_risc.get("TRISC_1", next(iter(compute_by_risc.values())))
        active_count = min(len(reader), len(canonical))
        if active_count:
            active_positions = sorted(
                sorted(
                    range(len(canonical)),
                    key=lambda index: int(canonical[index]["duration_cycles"]),
                    reverse=True,
                )[:active_count]
            )
            ordinal_by_position = {
                position: ordinal for ordinal, position in enumerate(active_positions, start=1)
            }
            for scopes in compute_by_risc.values():
                if len(scopes) == len(canonical):
                    for position, ordinal in ordinal_by_position.items():
                        scopes[position]["phase_ordinal"] = ordinal
                    continue

                # Profiler truncation can leave one TRISC with fewer scopes.
                # Preserve consistent compact numbering on the scopes that
                # remain instead of reusing their raw scheduled-slot number.
                fallback_positions = sorted(
                    sorted(
                        range(len(scopes)),
                        key=lambda index: int(scopes[index]["duration_cycles"]),
                        reverse=True,
                    )[: min(active_count, len(scopes))]
                )
                for ordinal, position in enumerate(fallback_positions, start=1):
                    scopes[position]["phase_ordinal"] = ordinal
        elif not reader:
            # A truncated capture may have lost the reader stream. Retain
            # useful sequential compute labels in that case.
            for scopes in compute_by_risc.values():
                for ordinal, event in enumerate(scopes, start=1):
                    event["phase_ordinal"] = ordinal


def compact_event_label(event: dict[str, object], batch_name: str) -> str:
    zone = str(event["zone"])
    if zone == "Cluster":
        ordinal = event.get("phase_ordinal")
        return f"{batch_name} C{ordinal}" if ordinal is not None else ""
    return {
        "Query": f"{batch_name} query",
        "Send Res": f"{batch_name} send",
        "Partial Res": f"{batch_name} gather",
        "Compute Res": f"{batch_name} reduce",
        "Res": f"{batch_name} out",
    }.get(zone, batch_name)


def plot_pipeline(
    events: list[dict[str, object]],
    selected_cores: list[tuple[str, int, int]],
    aggregator: tuple[str, int, int],
    batches: list[int],
    frequency_mhz: float,
    output: Path,
    cores_per_page: int,
    unit: str,
    view: str,
    label_mode: str,
    dpi: int,
) -> list[Path]:
    if PLOTTING_IMPORT_ERROR is not None:
        raise RuntimeError(f"Matplotlib is required: {PLOTTING_IMPORT_ERROR}")
    assert plt is not None
    assert Patch is not None
    add_phase_ordinals(events)
    # Send Res and Res are real but sub-pixel at this full-pipeline scale.
    # Their BRISC rows looked empty, so omit those rows and retain result
    # completion as the vertical B<n> done marker.
    visible_events = [
        event
        for event in events
        if str(event["role"]) != "writer"
        and not (
            str(event["zone"]) == "Cluster"
            and str(event["role"]) == "compute"
            and event.get("phase_ordinal") is None
        )
    ]
    if not visible_events:
        raise RuntimeError("No reader or compute scopes remain to plot")
    origin = min(int(item["start_cycle"]) for item in visible_events)
    global_end = max(int(item["end_cycle"]) for item in events)
    scale = 1.0 if unit == "cycles" else 1.0 / frequency_mhz
    batch_names = {batch: f"B{index + 1}" for index, batch in enumerate(batches)}
    pages = page_cores(selected_cores, cores_per_page)
    written: list[Path] = []

    for page_index, cores in enumerate(pages):
        lanes: list[tuple[tuple[str, int, int], str]] = []
        for core in cores:
            riscs = sorted(
                {str(item["risc"]) for item in visible_events if core_key(item) == core},
                key=lambda risc: (RISC_ORDER.get(risc, 99), risc),
            )
            lanes.extend((core, risc) for risc in riscs)
        lane_y: dict[tuple[tuple[str, int, int], str], float] = {}
        labels: list[str] = []
        ticks: list[float] = []
        y = 0.0
        for core_index, core in enumerate(cores):
            if core_index:
                y += 0.55
            for lane in [candidate for candidate in lanes if candidate[0] == core]:
                lane_y[lane] = y
                stage = RISC_STAGE.get(lane[1], "")
                labels.append(f"Core({core[1]},{core[2]})-{lane[1]} {stage}".rstrip())
                ticks.append(y)
                y += 0.86

        fig, ax = plt.subplots(
            figsize=(max(18.0, 1.2 * len(batches) + 8.0), max(6.0, 0.38 * len(lanes) + 2.5)),
            constrained_layout=True,
        )
        figure_color = "white"
        axes_color = "white"
        text_color = "#202020"
        muted_color = "#666666"
        grid_color = "#AAB0B6"
        bar_edge = "white"
        fig.patch.set_facecolor(figure_color)
        ax.set_facecolor(axes_color)

        # Tracy-style alternating RISC tracks make it easier to follow a lane
        # across a long multi-batch capture.
        for lane_index, lane in enumerate(lanes):
            if lane not in lane_y:
                continue
            shade = "#F3F5F7" if lane_index % 2 == 0 else "#FFFFFF"
            ax.axhspan(lane_y[lane] - 0.39, lane_y[lane] + 0.39, color=shade, zorder=0)

        if view in {"trace", "phases"}:
            timeline_width = max((global_end - origin) * scale, 1.0)
            for event in sorted(visible_events, key=lambda item: int(item["start_cycle"])):
                lane = (core_key(event), str(event["risc"]))
                if lane not in lane_y:
                    continue
                start = (int(event["start_cycle"]) - origin) * scale
                duration = max(int(event["duration_cycles"]) * scale, 0.001)
                phase = str(event["phase"])
                ax.barh(
                    lane_y[lane], duration, left=start,
                    height=0.62,
                    color=PHASE_COLORS.get(phase, "#667788"), edgecolor=bar_edge, linewidth=0.45,
                    zorder=3,
                )
                compact_label = compact_event_label(event, batch_names[int(event["batch_id"])])
                duration_us = int(event["duration_cycles"]) / frequency_mhz
                label_allowed = (
                    duration_us >= 200.0
                    and duration / timeline_width >= (0.018 if label_mode == "compact" else 0.012)
                )
                if label_allowed and label_mode in {"compact", "phase"}:
                    batch_name = batch_names[int(event["batch_id"])]
                    label = (
                        compact_label
                        if label_mode == "compact"
                        else f"{batch_name} {phase}"
                    )
                    if not label:
                        continue
                    narrow = duration / timeline_width < 0.018
                    ax.text(
                        start + duration / 2.0,
                        lane_y[lane],
                        label,
                        ha="center", va="center", fontsize=5.8 if narrow else 7.0,
                        color="#202020" if narrow or str(event["role"]) == "compute" else "white",
                        clip_on=True, zorder=4,
                    )

        completions = sorted(
            [event for event in events if core_key(event) == aggregator and str(event["zone"]) == "Res"],
            key=lambda event: int(event["end_cycle"]),
        )
        for event in completions:
            x = (int(event["end_cycle"]) - origin) * scale
            ax.axvline(x, color=muted_color, linestyle=":", linewidth=0.8, alpha=0.75, zorder=2)
            ax.text(
                x, 1.002, f"{batch_names[int(event['batch_id'])]} done",
                transform=ax.get_xaxis_transform(), rotation=90,
                ha="left", va="bottom", fontsize=6.5, color=muted_color,
            )
        ax.set_yticks(ticks, labels)
        ax.set_xlim(0, max((global_end - origin) * scale, 1.0) * 1.02)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_ylabel("Tensix core + RISC", color=text_color)
        ax.set_xlabel(
            "Cycles since first selected scope" if unit == "cycles" else "Time since first selected scope (µs)",
            color=text_color,
        )
        ax.tick_params(axis="both", colors=text_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(muted_color)
        ax.grid(True, axis="x", color=grid_color, alpha=0.28, linewidth=0.6)
        ax.set_axisbelow(True)
        legend = [
            Patch(facecolor=QUERY_READ_COLOR, label="Query read"),
            Patch(facecolor=READER_COLOR, label="Reader work"),
            Patch(facecolor=COMPUTE_COLOR, label="Compute work"),
            Patch(facecolor=WAIT_COLOR, label="Query wait"),
        ]
        add_kernel_legend(ax, legend)

        page_output = output_page_path(output, page_index, len(pages))
        page_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(page_output)
    return written


def write_events(
    path: Path,
    events: list[dict[str, object]],
    aggregator: tuple[str, int, int],
    frequency_mhz: float,
) -> None:
    fields = [
        "kind", "pcie_slot", "core_x", "core_y", "core_role", "risc", "role",
        "zone", "phase", "batch_id", "start_cycle", "end_cycle", "duration_cycles",
        "start_us", "end_us", "duration_us", "profile_start_cycle", "profile_end_cycle",
    ]
    items = events
    origin = min(int(item["start_cycle"]) for item in items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for item in sorted(items, key=lambda value: (int(value["start_cycle"]), str(value["risc"]))):
            start = int(item["start_cycle"])
            end = int(item["end_cycle"])
            writer.writerow(
                {
                    "kind": item["kind"],
                    "pcie_slot": item["pcie_slot"],
                    "core_x": item["core_x"],
                    "core_y": item["core_y"],
                    "core_role": "aggregator" if core_key(item) == aggregator else "worker",
                    "risc": item["risc"],
                    "role": item["role"],
                    "zone": item["zone"],
                    "phase": item["phase"],
                    "batch_id": item["batch_id"],
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


def main() -> None:
    args = parse_args()
    if args.worker_count < 0:
        raise ValueError("--worker-count cannot be negative")
    if args.last_batches <= 0:
        raise ValueError("--last-batches must be greater than zero")
    if args.cores_per_page < 2:
        raise ValueError("--cores-per-page must be at least two")

    profile_log = discover_profile_log(args.profile_log)
    print(f"Reading current scoped pipeline zones: {profile_log}")
    frequency_mhz, all_events, unmatched = read_scoped_events(profile_log)
    if args.run_id is not None:
        all_events = [event for event in all_events if str(event["run_id"]) == args.run_id]
        if not all_events:
            raise RuntimeError(f"No scoped events have run ID {args.run_id}")
    aggregator = detect_aggregator(
        all_events, parse_core(args.aggregator_core) if args.aggregator_core else None
    )
    selected_cores = choose_cores(
        all_events,
        aggregator,
        {parse_core(value) for value in args.core},
        args.worker_count,
    )
    events, batches, grouping_warnings = assign_batches_from_scopes(
        all_events, selected_cores, aggregator, args.last_batches
    )
    if not events:
        raise RuntimeError("No scoped events remained after batch inference")
    events_output = args.events_output or args.output.with_suffix(".csv")
    write_events(events_output, events, aggregator, frequency_mhz)
    outputs = plot_pipeline(
        events,
        selected_cores,
        aggregator,
        batches,
        frequency_mhz,
        args.output,
        args.cores_per_page,
        args.unit,
        args.view,
        args.label_mode,
        args.dpi,
    )
    print(
        f"Selected the final {len(batches)} batch(es), {len(events)} scoped phases, "
        f"and {len(selected_cores)} cores."
    )
    if len(batches) < args.last_batches:
        print(f"Warning: requested {args.last_batches} batches but only {len(batches)} aggregator Res zones exist.")
    for warning in grouping_warnings:
        print(f"Warning: {warning}")
    if unmatched:
        print(f"Warning: ignored {unmatched} unmatched zone ends, usually due to profiler-buffer truncation.")
    print(f"Wrote event data to {events_output}")
    for output in outputs:
        print(f"Wrote pipeline timeline to {output}")


if __name__ == "__main__":
    main()
