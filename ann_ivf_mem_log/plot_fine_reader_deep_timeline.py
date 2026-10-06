#!/usr/bin/env python3

"""Zoom into fine-reader cluster-slice zones for one 32-query batch."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from plot_device_kernel_timeline import (
    PLOTTING_IMPORT_ERROR,
    ROLE_COLORS,
    discover_profile_log,
    find_column,
    find_exact_column,
    normalize_risc,
    output_page_path,
    parse_core,
    plt,
    read_profile_events,
    risc_sort_key,
)


BATCH_ID_ZONE = "Fine_Search_Reader_QueryBatch_ID"
BATCH_ZONE = "Fine_Search_Reader_QueryBatch"
TASK_ZONE = "MemLog_Fine_DatasetAndIndices_DRAM_to_L1"
QUERY_ZONE = "MemLog_Fine_Query_DRAM_to_L1"
SLICE_ZONE = "MemLog_Fine_Cluster_slice"

DEEP_ZONES = {BATCH_ZONE, TASK_ZONE, QUERY_ZONE, SLICE_ZONE}
START_PHASES = {"begin", "start", "zone_start"}
END_PHASES = {"end", "stop", "zone_end"}
ZONE_LANES = {
    BATCH_ZONE: ("batch", "#B8C1CC"),
    TASK_ZONE: ("cluster task", "#4C9BE8"),
    QUERY_ZONE: ("query read", "#F39C3D"),
    SLICE_ZONE: ("cluster slice", "#7559E8"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MemLog_Fine_Cluster_slice occurrences inside one fine-reader "
            "32-query batch. Parent batch/task zones are shown as context."
        )
    )
    parser.add_argument(
        "--profile-log",
        type=Path,
        help=(
            "TT-Metal profile_log_device.csv; if omitted, use the same "
            "auto-discovery as plot_device_kernel_timeline.py"
        ),
    )
    parser.add_argument("--batch-id", type=int, default=0, help="Zero-based 32-query batch ID")
    parser.add_argument(
        "--invocation",
        default="last",
        help="Search invocation number, or 'last' (default)",
    )
    parser.add_argument(
        "--run-id",
        help="Optional profiler run host ID; useful when one CSV contains several processes",
    )
    parser.add_argument(
        "--core",
        action="append",
        default=[],
        metavar="X,Y",
        help=(
            "Select a worker profiler coordinate; repeat as needed. Core "
            "used by the aggregator is added automatically."
        ),
    )
    parser.add_argument(
        "--aggregator-core",
        metavar="X,Y",
        help=(
            "Override the aggregator profiler coordinate. By default it is "
            "detected from Core_0_0_Final_Sort_* zones, with 1,1 as fallback."
        ),
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=3,
        help="When --core is omitted, select workers 1 through this value",
    )
    parser.add_argument(
        "--task-ordinal",
        type=int,
        help=(
            "Restrict to one worker-script task occurrence within the batch. "
            "This is an ordinal, not the IVF cluster ID."
        ),
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=0,
        help="First batch-wide cluster-slice ordinal to retain on each core",
    )
    parser.add_argument(
        "--slice-count",
        type=int,
        default=32,
        help="Slices to retain per core; zero keeps every recorded slice",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ann_ivf_fine_reader_deep_timeline.png"),
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        help="Selected deep-event CSV; defaults to the output path with .csv",
    )
    parser.add_argument("--cores-per-page", type=int, default=3)
    parser.add_argument("--unit", choices=("cycles", "us"), default="cycles")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _optional_exact_column(header: list[str], *names: str) -> int | None:
    try:
        return find_exact_column(header, *names)
    except ValueError:
        return None


def _context_copy(context: dict[str, object] | None) -> dict[str, object]:
    if context is None:
        return {
            "batch_id": None,
            "batch_id_from_marker": False,
            "invocation": 0,
            "task_ordinal": None,
            "slice_ordinal": None,
            "task_slice_ordinal": None,
        }
    return dict(context)


def read_deep_events(profile_log: Path) -> tuple[float, list[dict[str, object]], dict[str, int]]:
    with profile_log.open(newline="") as profile_file:
        rows = list(csv.reader(profile_file))

    frequency_mhz: float | None = None
    header_index: int | None = None
    import re

    for row_index, row in enumerate(rows):
        joined = ",".join(row)
        match = re.search(r"CHIP_FREQ\[MHz\]\s*:\s*([0-9.]+)", joined)
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
    try:
        data_col = find_exact_column(header, "data", "stat value")
    except ValueError:
        data_col = find_column(header, "stat", "value")
    trace_col = _optional_exact_column(header, "trace id")
    trace_counter_col = _optional_exact_column(header, "trace id counter")

    needed = max(slot_col, core_x_col, core_y_col, risc_col, time_col, run_col, zone_col, type_col, data_col)
    markers_by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if len(row) <= needed:
            continue
        name = row[zone_col].strip()
        if name not in DEEP_ZONES and name != BATCH_ID_ZONE:
            continue
        try:
            cycle = int(row[time_col].strip())
            core_x = int(row[core_x_col].strip())
            core_y = int(row[core_y_col].strip())
        except ValueError:
            continue
        trace_id = row[trace_col].strip() if trace_col is not None and len(row) > trace_col else ""
        trace_counter = (
            row[trace_counter_col].strip()
            if trace_counter_col is not None and len(row) > trace_counter_col
            else ""
        )
        stream = (
            row[slot_col].strip(),
            core_x,
            core_y,
            normalize_risc(row[risc_col]),
            row[run_col].strip(),
            trace_id,
            trace_counter,
        )
        data_text = row[data_col].strip()
        try:
            data = int(data_text, 0) if data_text else 0
        except ValueError:
            data = 0
        markers_by_stream[stream].append(
            {
                "cycle": cycle,
                "name": name,
                "type": row[type_col].strip().lower(),
                "data": data,
                "row_number": row_number,
            }
        )

    events: list[dict[str, object]] = []
    unmatched: dict[str, int] = defaultdict(int)
    for stream, markers in markers_by_stream.items():
        markers.sort(key=lambda marker: (int(marker["cycle"]), int(marker["row_number"])))
        pending_batch_ids: list[int] = []
        batch_stack: list[dict[str, object]] = []
        task_stack: list[dict[str, object]] = []
        open_zones: dict[str, list[dict[str, object]]] = defaultdict(list)
        previous_batch: int | None = None
        invocation = 0
        fallback_batch = 0
        task_count: dict[tuple[int, int], int] = defaultdict(int)
        slice_count: dict[tuple[int, int], int] = defaultdict(int)

        for marker in markers:
            name = str(marker["name"])
            marker_type = str(marker["type"])
            if name == BATCH_ID_ZONE:
                if marker_type in {"ts_data", "data", "timestamped_data"}:
                    pending_batch_ids.append(int(marker["data"]))
                continue

            if marker_type in START_PHASES:
                if name == BATCH_ZONE:
                    from_marker = bool(pending_batch_ids)
                    batch_id = pending_batch_ids.pop() if from_marker else fallback_batch
                    fallback_batch += 1
                    if previous_batch is not None and batch_id <= previous_batch:
                        invocation += 1
                    previous_batch = batch_id
                    context = {
                        "batch_id": batch_id,
                        "batch_id_from_marker": from_marker,
                        "invocation": invocation,
                        "task_ordinal": None,
                        "slice_ordinal": None,
                        "task_slice_ordinal": None,
                    }
                    batch_stack.append(context)
                elif name == TASK_ZONE:
                    context = _context_copy(batch_stack[-1] if batch_stack else None)
                    key = (int(context["invocation"]), int(context["batch_id"] or 0))
                    context["task_ordinal"] = task_count[key]
                    context["next_task_slice"] = 0
                    task_count[key] += 1
                    task_stack.append(context)
                elif name == SLICE_ZONE:
                    context = _context_copy(task_stack[-1] if task_stack else (batch_stack[-1] if batch_stack else None))
                    key = (int(context["invocation"]), int(context["batch_id"] or 0))
                    context["slice_ordinal"] = slice_count[key]
                    slice_count[key] += 1
                    if task_stack:
                        context["task_slice_ordinal"] = int(task_stack[-1]["next_task_slice"])
                        task_stack[-1]["next_task_slice"] = int(task_stack[-1]["next_task_slice"]) + 1
                else:
                    context = _context_copy(batch_stack[-1] if batch_stack else None)

                open_zones[name].append(
                    {
                        "start_cycle": int(marker["cycle"]),
                        "row_number": int(marker["row_number"]),
                        "context": context,
                    }
                )
                continue

            if marker_type not in END_PHASES:
                continue
            if not open_zones[name]:
                unmatched[f"{name}:end"] += 1
                continue

            opened = open_zones[name].pop()
            start_cycle = int(opened["start_cycle"])
            end_cycle = int(marker["cycle"])
            context = dict(opened["context"])
            if end_cycle >= start_cycle:
                events.append(
                    {
                        "pcie_slot": stream[0],
                        "core_x": stream[1],
                        "core_y": stream[2],
                        "risc": stream[3],
                        "run_id": stream[4],
                        "trace_id": stream[5],
                        "trace_counter": stream[6],
                        "zone": name,
                        "start_cycle": start_cycle,
                        "end_cycle": end_cycle,
                        "duration_cycles": end_cycle - start_cycle,
                        **context,
                    }
                )
            if name == TASK_ZONE and task_stack:
                task_stack.pop()
            elif name == BATCH_ZONE and batch_stack:
                batch_stack.pop()

        for name, scopes in open_zones.items():
            if scopes:
                unmatched[f"{name}:start"] += len(scopes)

    return frequency_mhz, events, dict(unmatched)


def stream_key(event: dict[str, object]) -> tuple[object, ...]:
    return event["pcie_slot"], event["core_x"], event["core_y"], event["risc"]


def select_deep_events(
    events: list[dict[str, object]],
    batch_id: int,
    invocation_text: str,
    run_id: str | None,
    requested_cores: set[tuple[int, int]],
    task_ordinal: int | None,
    slice_start: int,
    slice_count_limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], tuple[str, int]]:
    targets = [
        event
        for event in events
        if event["zone"] == SLICE_ZONE
        and event["batch_id"] == batch_id
        and (run_id is None or str(event["run_id"]) == run_id)
        and (
            not requested_cores
            or (int(event["core_x"]), int(event["core_y"])) in requested_cores
        )
        and (task_ordinal is None or event["task_ordinal"] == task_ordinal)
    ]
    if not targets:
        raise RuntimeError(
            f"No {SLICE_ZONE} pairs were found for batch {batch_id}. "
            "The profiler buffer may have filled before this batch."
        )

    invocation_groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for event in targets:
        invocation_groups[(str(event["run_id"]), int(event["invocation"]))].append(event)

    if invocation_text == "last":
        selected_invocation = max(
            invocation_groups,
            key=lambda key: max(int(event["start_cycle"]) for event in invocation_groups[key]),
        )
    else:
        try:
            requested_invocation = int(invocation_text)
        except ValueError as error:
            raise ValueError("--invocation must be 'last' or a non-negative integer") from error
        if requested_invocation < 0:
            raise ValueError("--invocation must be 'last' or a non-negative integer")
        matching = [key for key in invocation_groups if key[1] == requested_invocation]
        if not matching:
            available = sorted(invocation_groups)
            raise RuntimeError(f"Invocation {requested_invocation} is unavailable; found {available}")
        selected_invocation = max(
            matching,
            key=lambda key: max(int(event["start_cycle"]) for event in invocation_groups[key]),
        )

    by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for event in invocation_groups[selected_invocation]:
        by_stream[stream_key(event)].append(event)

    selected_targets: list[dict[str, object]] = []
    selected_windows: dict[tuple[object, ...], tuple[int, int]] = {}
    for key, stream_events in by_stream.items():
        stream_events.sort(key=lambda event: int(event["slice_ordinal"]))
        chosen = [event for event in stream_events if int(event["slice_ordinal"]) >= slice_start]
        if slice_count_limit:
            chosen = chosen[:slice_count_limit]
        if not chosen:
            continue
        for event in chosen:
            event["target_selected"] = True
        selected_targets.extend(chosen)
        selected_windows[key] = (
            min(int(event["start_cycle"]) for event in chosen),
            max(int(event["end_cycle"]) for event in chosen),
        )

    if not selected_targets:
        raise RuntimeError(
            f"No slices remain at --slice-start {slice_start}; reduce the start ordinal"
        )

    context: list[dict[str, object]] = []
    run, invocation = selected_invocation
    for event in events:
        key = stream_key(event)
        if key not in selected_windows:
            continue
        if str(event["run_id"]) != run or int(event["invocation"]) != invocation:
            continue
        if event["batch_id"] != batch_id or event["zone"] == SLICE_ZONE:
            continue
        window_start, window_end = selected_windows[key]
        overlaps_window = (
            int(event["start_cycle"]) <= window_end
            and int(event["end_cycle"]) >= window_start
        )
        # The query is read once immediately before slice zero. Retain that
        # one-time setup when zooming from the beginning of a batch even
        # though it does not overlap the first slice itself.
        precedes_first_slice = (
            slice_start == 0
            and event["zone"] == QUERY_ZONE
            and int(event["end_cycle"]) <= window_start
        )
        if overlaps_window or precedes_first_slice:
            event["target_selected"] = False
            context.append(event)

    return selected_targets, context, selected_invocation


def _core_key(event: dict[str, object]) -> tuple[object, ...]:
    return event["pcie_slot"], int(event["core_x"]), int(event["core_y"])


def _worker_core_sort_key(core: tuple[object, ...]) -> tuple[str, int, int]:
    # Match the host's worker enumeration: Y is the outer loop and X is the
    # inner loop. The caller removes the profiler coordinate used by the
    # logical aggregation core before assigning worker numbers.
    return str(core[0]), int(core[2]), int(core[1])


def select_batch_context(
    events: list[dict[str, object]],
    batch_id: int,
    selected_invocation: tuple[str, int],
    worker_cores: set[tuple[object, ...]],
    aggregator_core: tuple[int, int],
) -> list[dict[str, object]]:
    run_id, invocation = selected_invocation
    allowed_coordinates = {
        (int(core[1]), int(core[2])) for core in worker_cores
    }
    allowed_coordinates.add(aggregator_core)
    selected = [
        event
        for event in events
        if event["granularity"] == "batch"
        and event["batch_id"] == batch_id
        and str(event["run_id"]) == run_id
        and int(event["invocation"]) == invocation
        and (int(event["core_x"]), int(event["core_y"])) in allowed_coordinates
    ]
    for event in selected:
        event["target_selected"] = False
    return selected


def plot_pages(
    targets: list[dict[str, object]],
    context: list[dict[str, object]],
    batch_events: list[dict[str, object]],
    aggregator_core: tuple[int, int],
    frequency_mhz: float,
    output: Path,
    cores_per_page: int,
    unit: str,
    dpi: int,
    subtitle: str,
) -> list[Path]:
    if PLOTTING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "plot_fine_reader_deep_timeline.py requires Matplotlib in the active "
            f"Python environment: {PLOTTING_IMPORT_ERROR}"
        )
    assert plt is not None

    cores = sorted({_core_key(event) for event in targets}, key=_worker_core_sort_key)
    pages = [cores[index : index + cores_per_page] for index in range(0, len(cores), cores_per_page)]
    all_deep_events = context + targets
    deep_origin = min(int(event["start_cycle"]) for event in targets)
    selected_end = max(int(event["end_cycle"]) for event in targets)
    query_context_starts = [
        int(event["start_cycle"]) for event in context if event["zone"] == QUERY_ZONE
    ]
    detail_start = min([deep_origin, *query_context_starts])
    span = max(1, selected_end - detail_start)
    padding = max(1, span // 50)
    view_start = detail_start - padding
    view_end = selected_end + padding
    scale = 1.0 if unit == "cycles" else 1.0 / frequency_mhz
    written: list[Path] = []

    for page_index, page_cores in enumerate(pages):
        page_core_set = set(page_cores)
        page_deep_events = [event for event in all_deep_events if _core_key(event) in page_core_set]
        page_coordinates = {(int(core[1]), int(core[2])) for core in page_cores}
        page_coordinates.add(aggregator_core)
        page_batch_events = [
            event
            for event in batch_events
            if (int(event["core_x"]), int(event["core_y"])) in page_coordinates
        ]

        worker_number = {core: cores.index(core) + 1 for core in page_cores}
        batch_core_order: list[tuple[object, ...]] = []
        aggregator_cores = sorted(
            {
                _core_key(event)
                for event in page_batch_events
                if (int(event["core_x"]), int(event["core_y"])) == aggregator_core
            }
        )
        batch_core_order.extend(aggregator_cores)
        batch_core_order.extend(page_cores)

        batch_lanes = sorted(
            {(_core_key(event), str(event["risc"])) for event in page_batch_events},
            key=lambda lane: (
                batch_core_order.index(lane[0]) if lane[0] in batch_core_order else 999,
                risc_sort_key(lane[1]),
            ),
        )
        batch_lane_positions: dict[tuple[tuple[object, ...], str], float] = {}
        batch_lane_labels: list[str] = []
        batch_y_ticks: list[float] = []
        y = 0.0
        previous_core: tuple[object, ...] | None = None
        for core, risc in batch_lanes:
            if previous_core is not None and core != previous_core:
                y += 0.6
            batch_lane_positions[(core, risc)] = y
            _, core_x, core_y = core
            if (core_x, core_y) == aggregator_core:
                owner = "Aggregator"
            else:
                owner = f"Worker {worker_number.get(core, '?')}"
            batch_lane_labels.append(f"{owner} Core({core_x},{core_y})-{risc}")
            batch_y_ticks.append(y)
            y += 1.0
            previous_core = core

        deep_lane_positions: dict[tuple[tuple[object, ...], str], float] = {}
        deep_lane_labels: list[str] = []
        deep_y_ticks: list[float] = []
        y = 0.0
        for core in page_cores:
            core_events = [event for event in page_deep_events if _core_key(event) == core]
            risc = str(core_events[0]["risc"])
            _, core_x, core_y = core
            for zone in (BATCH_ZONE, TASK_ZONE, QUERY_ZONE, SLICE_ZONE):
                deep_lane_positions[(core, zone)] = y
                deep_lane_labels.append(
                    f"Worker {worker_number[core]} Core({core_x},{core_y})-{risc} "
                    f"{ZONE_LANES[zone][0]}"
                )
                deep_y_ticks.append(y)
                y += 1.0
            y += 0.7

        top_height = max(3.8, 0.32 * max(1, len(batch_y_ticks)) + 1.4)
        bottom_height = max(4.5, 0.34 * len(deep_y_ticks) + 1.5)
        fig, (batch_ax, deep_ax) = plt.subplots(
            2,
            1,
            figsize=(16.0, top_height + bottom_height),
            gridspec_kw={"height_ratios": [top_height, bottom_height]},
            constrained_layout=True,
        )

        if page_batch_events:
            batch_origin = min(int(event["start_cycle"]) for event in page_batch_events)
            batch_end = max(int(event["end_cycle"]) for event in page_batch_events)
            for event in sorted(page_batch_events, key=lambda item: int(item["start_cycle"])):
                lane = (_core_key(event), str(event["risc"]))
                left = (int(event["start_cycle"]) - batch_origin) * scale
                raw_duration = int(event["duration_cycles"])
                duration = max(raw_duration * scale, 0.001)
                role = str(event["role"])
                batch_ax.barh(
                    batch_lane_positions[lane],
                    duration,
                    left=left,
                    height=0.70,
                    color=ROLE_COLORS[role],
                    edgecolor="white",
                    linewidth=0.4,
                )
                duration_us = raw_duration / frequency_mhz
                batch_ax.text(
                    left + duration / 2.0,
                    batch_lane_positions[lane],
                    f"{duration_us:.2f} µs",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="#202020" if role == "compute" else "white",
                    clip_on=True,
                )
            batch_ax.set_xlim(0, max(1.0, (batch_end - batch_origin) * scale))
            batch_ax.set_yticks(batch_y_ticks, batch_lane_labels)
        else:
            batch_ax.text(
                0.5,
                0.5,
                "No complete aggregator/worker QueryBatch zone pairs were captured",
                ha="center",
                va="center",
                transform=batch_ax.transAxes,
            )
            batch_ax.set_yticks([])
        batch_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        batch_ax.set_xlabel(
            "Cycles since first full-batch event"
            if unit == "cycles"
            else "Time since first full-batch event (µs)"
        )
        batch_ax.set_title(
            f"Full batch: aggregator profiler Core({aggregator_core[0]},{aggregator_core[1]}) "
            "and worker 1+ costs"
        )
        batch_ax.grid(True, axis="x", linestyle="-", alpha=0.25)
        batch_ax.set_axisbelow(True)

        for event in sorted(page_deep_events, key=lambda item: int(item["start_cycle"])):
            zone = str(event["zone"])
            clipped_start = max(int(event["start_cycle"]), view_start)
            clipped_end = min(int(event["end_cycle"]), view_end)
            if clipped_end < clipped_start:
                continue
            left = (clipped_start - deep_origin) * scale
            duration = max((clipped_end - clipped_start) * scale, 0.001)
            _, color = ZONE_LANES[zone]
            deep_ax.barh(
                deep_lane_positions[(_core_key(event), zone)],
                duration,
                left=left,
                height=0.70,
                color=color,
                alpha=1.0 if bool(event.get("target_selected")) else 0.55,
                edgecolor="white",
                linewidth=0.35,
            )
            if zone == SLICE_ZONE:
                label = f"S{event['slice_ordinal']}"
                if event["task_ordinal"] is not None:
                    label += f"/T{event['task_ordinal']}"
                deep_ax.text(
                    left + duration / 2.0,
                    deep_lane_positions[(_core_key(event), zone)],
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white",
                    clip_on=True,
                )

        deep_ax.set_yticks(deep_y_ticks, deep_lane_labels)
        deep_ax.set_xlim((view_start - deep_origin) * scale, (view_end - deep_origin) * scale)
        deep_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        deep_ax.set_xlabel(
            "Cycles relative to first selected slice"
            if unit == "cycles"
            else "Time relative to first selected slice (µs)"
        )
        deep_ax.set_title(f"Worker 1+ deep reader zoom\n{subtitle}")
        deep_ax.grid(True, axis="x", linestyle="-", alpha=0.25)
        deep_ax.set_axisbelow(True)

        page_output = output_page_path(output, page_index, len(pages))
        page_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(page_output)

    return written


def write_events(
    path: Path,
    targets: list[dict[str, object]],
    context: list[dict[str, object]],
    batch_events: list[dict[str, object]],
    frequency_mhz: float,
) -> None:
    fields = [
        "panel",
        "target_selected",
        "pcie_slot",
        "core_x",
        "core_y",
        "risc",
        "run_id",
        "trace_id",
        "trace_counter",
        "invocation",
        "batch_id",
        "batch_id_from_marker",
        "task_ordinal",
        "slice_ordinal",
        "task_slice_ordinal",
        "role",
        "zone",
        "timeline_origin_cycle",
        "start_cycle",
        "end_cycle",
        "duration_cycles",
        "start_us",
        "end_us",
        "duration_us",
        "profile_start_cycle",
        "profile_end_cycle",
    ]
    origin = min(int(event["start_cycle"]) for event in targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        rows = [
            ("full_batch", event) for event in batch_events
        ] + [
            ("deep_zoom", event) for event in context + targets
        ]
        for panel, event in sorted(
            rows,
            key=lambda item: (
                str(item[1]["pcie_slot"]),
                int(item[1]["core_x"]),
                int(item[1]["core_y"]),
                int(item[1]["start_cycle"]),
                int(item[1]["end_cycle"]),
            ),
        ):
            profile_start = int(event["start_cycle"])
            profile_end = int(event["end_cycle"])
            start = profile_start - origin
            end = profile_end - origin
            writer.writerow(
                {
                    **{field: event.get(field, "") for field in fields},
                    "panel": panel,
                    "timeline_origin_cycle": origin,
                    "start_cycle": start,
                    "end_cycle": end,
                    "duration_cycles": profile_end - profile_start,
                    "start_us": f"{start / frequency_mhz:.6f}",
                    "end_us": f"{end / frequency_mhz:.6f}",
                    "duration_us": f"{(profile_end - profile_start) / frequency_mhz:.6f}",
                    "profile_start_cycle": profile_start,
                    "profile_end_cycle": profile_end,
                }
            )


def main() -> None:
    args = parse_args()
    if args.batch_id < 0:
        raise ValueError("--batch-id must be non-negative")
    if args.task_ordinal is not None and args.task_ordinal < 0:
        raise ValueError("--task-ordinal must be non-negative")
    if args.slice_start < 0:
        raise ValueError("--slice-start must be non-negative")
    if args.slice_count < 0:
        raise ValueError("--slice-count must be non-negative")
    if args.worker_count <= 0:
        raise ValueError("--worker-count must be greater than zero")
    if args.cores_per_page <= 0:
        raise ValueError("--cores-per-page must be greater than zero")

    profile_log = discover_profile_log(args.profile_log)
    print(f"Reading device-profiler log: {profile_log}")
    frequency_mhz, events, unmatched = read_deep_events(profile_log)
    overview_frequency_mhz, overview_events, overview_unmatched = read_profile_events(profile_log)
    if overview_frequency_mhz != frequency_mhz:
        raise RuntimeError("Profiler clock frequency changed while reading the same log")

    if args.aggregator_core:
        aggregator_core = parse_core(args.aggregator_core)
        aggregator_source = "command line"
    else:
        detected_aggregators = sorted(
            {
                (int(event["core_x"]), int(event["core_y"]))
                for event in overview_events
                if str(event["zone"]).startswith("Core_0_0_Final_Sort_")
            },
            key=lambda core: (core[1], core[0]),
        )
        if detected_aggregators:
            aggregator_core = detected_aggregators[0]
            aggregator_source = "Core_0_0_Final_Sort_* zones"
        else:
            aggregator_core = (1, 1)
            aggregator_source = "one-based fallback"
    print(
        f"Aggregator profiler core=Core({aggregator_core[0]},{aggregator_core[1]}) "
        f"from {aggregator_source}."
    )

    requested_cores = {parse_core(value) for value in args.core}
    if aggregator_core in requested_cores:
        requested_cores.remove(aggregator_core)
        print(
            f"Core({aggregator_core[0]},{aggregator_core[1]}) is the aggregator and "
            "is included automatically, not as a worker."
        )
    if not requested_cores:
        available_worker_cores = sorted(
            {
                (int(event["core_x"]), int(event["core_y"]))
                for event in events
                if event["zone"] == SLICE_ZONE
                and event["batch_id"] == args.batch_id
                and (int(event["core_x"]), int(event["core_y"])) != aggregator_core
            },
            key=lambda core: (core[1], core[0]),
        )
        requested_cores = set(available_worker_cores[: args.worker_count])
        if not requested_cores:
            raise RuntimeError(f"No worker cores have recorded slices for batch {args.batch_id}")
        selected_text = ", ".join(
            f"Worker {index}=Core({core[0]},{core[1]})"
            for index, core in enumerate(available_worker_cores[: args.worker_count], start=1)
        )
        print(
            f"Automatically selected {selected_text}; "
            f"Core({aggregator_core[0]},{aggregator_core[1]})=Aggregator"
        )

    targets, context, selected_invocation = select_deep_events(
        events,
        args.batch_id,
        args.invocation,
        args.run_id,
        requested_cores,
        args.task_ordinal,
        args.slice_start,
        args.slice_count,
    )
    worker_cores = {_core_key(event) for event in targets}
    batch_events = select_batch_context(
        overview_events,
        args.batch_id,
        selected_invocation,
        worker_cores,
        aggregator_core,
    )

    run_id, invocation = selected_invocation
    slice_ordinals = [int(event["slice_ordinal"]) for event in targets]
    subtitle = (
        f"run {run_id}, invocation {invocation}, batch {args.batch_id}; "
        f"recorded slices {min(slice_ordinals)}–{max(slice_ordinals)}"
    )
    events_output = args.events_output or args.output.with_suffix(".csv")
    write_events(events_output, targets, context, batch_events, frequency_mhz)
    page_outputs = plot_pages(
        targets,
        context,
        batch_events,
        aggregator_core,
        frequency_mhz,
        args.output,
        args.cores_per_page,
        args.unit,
        args.dpi,
        subtitle,
    )

    core_count = len({_core_key(event) for event in targets})
    print(
        f"Paired {len(events)} deep reader zones at {frequency_mhz:g} MHz; "
        f"selected {len(targets)} cluster slices on {core_count} worker cores and "
        f"{len(batch_events)} full-batch zones including aggregator "
        f"Core({aggregator_core[0]},{aggregator_core[1]})."
    )
    if unmatched:
        details = ", ".join(f"{name}={count}" for name, count in sorted(unmatched.items()))
        print(
            "Warning: unmatched profiler markers were ignored "
            f"({details}). This commonly means the per-RISC profiler buffer filled."
        )
    if overview_unmatched:
        print(
            f"Warning: ignored {overview_unmatched} unmatched full-batch zone ends."
        )
    if not any(
        (int(event["core_x"]), int(event["core_y"])) == aggregator_core
        for event in batch_events
    ):
        print(
            f"Warning: no complete Core({aggregator_core[0]},{aggregator_core[1]}) "
            "aggregation batch zones were captured; "
            "the upper panel will contain only available worker batch zones."
        )
    print(f"Wrote deep event data to {events_output}")
    for page_output in page_outputs:
        print(f"Wrote deep timeline to {page_output}")


if __name__ == "__main__":
    main()
