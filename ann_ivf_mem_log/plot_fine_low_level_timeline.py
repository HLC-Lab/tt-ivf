#!/usr/bin/env python3

"""Plot only bounded IVF fine-search low-level RISC zones."""

from __future__ import annotations

import argparse
import csv
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


BATCH_ID = "IVF.Low.BatchID"
SLICE_ID = "IVF.Low.SliceID"
PARTIAL_ID = "IVF.Low.PartialID"
DATA_MARKERS = {BATCH_ID, SLICE_ID, PARTIAL_ID}
START_PHASES = {"begin", "start", "zone_start"}
END_PHASES = {"end", "stop", "zone_end"}
DATA_PHASES = {"ts_data", "data", "timestamped_data"}
TIMESTAMP_BEGIN_SUFFIX = ".Begin"
TIMESTAMP_END_SUFFIX = ".End"

RISC_ORDER = {
    "NCRISC": 0,
    "TRISC_0": 1,
    "TRISC_1": 2,
    "TRISC_2": 3,
    "BRISC": 4,
}
RISC_ROLE = {
    "NCRISC": "reader",
    "TRISC_0": "unpacker",
    "TRISC_1": "compute/math",
    "TRISC_2": "packer",
    "BRISC": "writer",
}

PHASE_COLORS = {
    "wait": "#8D99A8",
    "read": "#3C8DDE",
    "compute": "#E4B600",
    "pack": "#3BA272",
    "write": "#E05A47",
    "publish": "#7559E8",
}

COMPACT_PHASE_NAMES = {
    "QueryRead": "QRead",
    "BufferWait": "CBWait",
    "ReadIssue": "Issue",
    "ReadBarrier": "Barrier",
    "Publish": "Pub",
    "InputWait": "InWait",
    "MatmulAndScorePack": "MM",
    "SortInputWait": "SortWait",
    "LocalSortAndWinnerPack": "TopK",
    "OutputPack": "OutPack",
    "ReadyWait": "Ready",
    "ResultWait": "ResultWait",
    "OutputWrite": "Write",
    "WorkersDoneWait": "WorkersWait",
    "PartialRead": "Read",
    "ReturnCredit": "Credit",
    "PartialSort": "TopK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot only IVF.Low.* zones on NCRISC reader, TRISC_0 unpacker, "
            "TRISC_1 math, TRISC_2 packer, and BRISC writer lanes."
        )
    )
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--invocation", default="last")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--aggregator-core",
        metavar="X,Y",
        help="Override automatic aggregator detection; one-based Wormhole fallback is 1,1",
    )
    parser.add_argument(
        "--core",
        action="append",
        default=[],
        metavar="X,Y",
        help="Select a worker profiler coordinate; repeat as needed",
    )
    parser.add_argument("--worker-count", type=int, default=3)
    parser.add_argument("--slice-start", type=int, default=0)
    parser.add_argument(
        "--slice-count",
        type=int,
        default=8,
        help="Worker candidate slices to retain; instrumentation records at most eight",
    )
    parser.add_argument("--cores-per-page", type=int, default=3)
    parser.add_argument("--unit", choices=("cycles", "us"), default="us")
    parser.add_argument(
        "--view",
        choices=("split", "full"),
        default="split",
        help=(
            "split gives sampled-slice and completion/aggregation windows; "
            "full retains one continuous batch axis"
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=("compact", "full", "none"),
        default="compact",
        help="Control labels inside timing bars",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ann_ivf_fine_low_level_timeline.png"),
    )
    parser.add_argument("--events-output", type=Path)
    return parser.parse_args()


def _optional_exact_column(header: list[str], name: str) -> int | None:
    try:
        return find_exact_column(header, name)
    except ValueError:
        return None


def _zone_item_metadata(
    zone: str,
    current_kind: str | None,
    current_id: int | None,
) -> tuple[str | None, int | None]:
    """Keep slice/partial IDs only on operations that actually consume them."""
    if ".Worker.Reader." in zone:
        uses_item = not zone.endswith(".QueryRead")
    elif ".Worker.Compute." in zone:
        uses_item = not zone.endswith(".OutputPack")
    elif ".Aggregator.Reader." in zone:
        uses_item = not zone.endswith(".WorkersDoneWait")
    elif ".Aggregator.Compute." in zone:
        uses_item = zone.endswith(".PartialSort")
    else:
        uses_item = False
    return (current_kind, current_id) if uses_item else (None, None)


def read_low_level_events(
    profile_log: Path,
) -> tuple[float, list[dict[str, object]], dict[str, int]]:
    import re

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
    try:
        data_col = find_exact_column(header, "data", "stat value")
    except ValueError:
        data_col = find_column(header, "stat", "value")
    trace_col = _optional_exact_column(header, "trace id")
    trace_counter_col = _optional_exact_column(header, "trace id counter")
    needed = max(
        slot_col,
        core_x_col,
        core_y_col,
        risc_col,
        time_col,
        run_col,
        zone_col,
        type_col,
        data_col,
    )

    markers_by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    raw_low_level_rows = 0
    raw_marker_types: dict[str, int] = defaultdict(int)
    raw_zone_names: set[str] = set()
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if len(row) <= needed:
            continue
        name = row[zone_col].strip()
        if not name.startswith("IVF.Low."):
            continue
        raw_low_level_rows += 1
        raw_zone_names.add(name)
        raw_marker_types[row[type_col].strip().lower()] += 1
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
        # Batch/slice timestamp-data markers can carry different trace fields
        # from their neighboring zones. Associate metadata at the stable
        # core/RISC/run level; retain trace fields on each marker for output.
        stream = (
            row[slot_col].strip(),
            core_x,
            core_y,
            normalize_risc(row[risc_col]),
            row[run_col].strip(),
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
                "trace_id": trace_id,
                "trace_counter": trace_counter,
            }
        )

    events: list[dict[str, object]] = []
    unmatched: dict[str, int] = defaultdict(int)
    for stream, markers in markers_by_stream.items():
        markers.sort(key=lambda marker: (int(marker["cycle"]), int(marker["row_number"])))
        open_zones: dict[str, list[dict[str, object]]] = defaultdict(list)
        # Every IVF.Low scope currently has a compile-time batch-0 guard. This
        # also makes truncated logs useful when the BatchID marker itself was
        # dropped but complete low-level zone pairs remain.
        current_batch: int | None = 0
        current_item_kind: str | None = None
        current_item_id: int | None = None
        previous_batch: int | None = None
        invocation = -1

        for marker in markers:
            name = str(marker["name"])
            marker_type = str(marker["type"])
            if name in DATA_MARKERS:
                if marker_type not in DATA_PHASES:
                    continue
                value = int(marker["data"])
                if name == BATCH_ID:
                    if previous_batch is None or value <= previous_batch:
                        invocation += 1
                    previous_batch = value
                    current_batch = value
                    current_item_kind = None
                    current_item_id = None
                elif name == SLICE_ID:
                    current_item_kind = "slice"
                    current_item_id = value
                else:
                    current_item_kind = "partial"
                    current_item_id = value
                continue

            # Wormhole currently retains timestamped-data records emitted by
            # the conditional instrumentation wrapper but drops its scoped
            # zone records. Treat explicit .Begin/.End timestamp names as a
            # regular interval while continuing to accept native zone pairs.
            if marker_type in DATA_PHASES:
                if name.endswith(TIMESTAMP_BEGIN_SUFFIX):
                    name = name[: -len(TIMESTAMP_BEGIN_SUFFIX)]
                    marker_type = "zone_start"
                elif name.endswith(TIMESTAMP_END_SUFFIX):
                    name = name[: -len(TIMESTAMP_END_SUFFIX)]
                    marker_type = "zone_end"
                else:
                    continue

            if marker_type in START_PHASES:
                item_kind, item_id = _zone_item_metadata(
                    name,
                    current_item_kind,
                    current_item_id,
                )
                open_zones[name].append(
                    {
                        "start_cycle": int(marker["cycle"]),
                        "batch_id": current_batch,
                        "invocation": max(invocation, 0),
                        "item_kind": item_kind,
                        "item_id": item_id,
                        "trace_id": marker["trace_id"],
                        "trace_counter": marker["trace_counter"],
                    }
                )
                continue
            if marker_type not in END_PHASES:
                continue
            if not open_zones[name]:
                unmatched[f"{name}:end"] += 1
                continue
            opened = open_zones[name].pop()
            start = int(opened["start_cycle"])
            end = int(marker["cycle"])
            if end < start:
                unmatched[f"{name}:backwards"] += 1
                continue
            events.append(
                {
                    "pcie_slot": stream[0],
                    "core_x": stream[1],
                    "core_y": stream[2],
                    "risc": stream[3],
                    "run_id": stream[4],
                    "trace_id": opened["trace_id"],
                    "trace_counter": opened["trace_counter"],
                    "zone": name,
                    "start_cycle": start,
                    "end_cycle": end,
                    "duration_cycles": end - start,
                    **{
                        key: value
                        for key, value in opened.items()
                        if key not in {"start_cycle", "trace_id", "trace_counter"}
                    },
                }
            )
        for name, scopes in open_zones.items():
            if scopes:
                unmatched[f"{name}:start"] += len(scopes)

    if not events:
        if raw_low_level_rows == 0:
            raise RuntimeError(
                f"{profile_log} contains no IVF.Low.* rows. The saved capture "
                "cannot produce the low-level plot. Re-capture with fresh "
                "TT_METAL_PROFILER_DIR and TT_METAL_CACHE directories."
            )
        marker_summary = ", ".join(
            f"{marker_type}={count}"
            for marker_type, count in sorted(raw_marker_types.items())
        )
        name_preview = ", ".join(sorted(raw_zone_names)[:8])
        raise RuntimeError(
            f"{profile_log} contains {raw_low_level_rows} IVF.Low.* rows but "
            f"no complete zone pairs; marker types: {marker_summary}; "
            f"zone names: {name_preview}."
        )

    return frequency_mhz, events, dict(unmatched)


def _core(event: dict[str, object]) -> tuple[object, int, int]:
    return event["pcie_slot"], int(event["core_x"]), int(event["core_y"])


def _core_order(core: tuple[object, int, int]) -> tuple[str, int, int]:
    return str(core[0]), core[2], core[1]


def select_events(
    events: list[dict[str, object]],
    batch_id: int,
    invocation_text: str,
    run_id: str | None,
    aggregator_override: tuple[int, int] | None,
    requested_workers: set[tuple[int, int]],
    worker_count: int,
    slice_start: int,
    slice_count: int,
) -> tuple[list[dict[str, object]], tuple[int, int], list[tuple[object, int, int]], tuple[str, int]]:
    candidates = [
        event
        for event in events
        if event["batch_id"] == batch_id and (run_id is None or str(event["run_id"]) == run_id)
    ]
    if not candidates:
        recorded_batches = sorted(
            {event["batch_id"] for event in events if event["batch_id"] is not None}
        )
        raise RuntimeError(
            f"No paired IVF.Low.* zones were found for batch {batch_id}; "
            f"recorded batches={recorded_batches}, paired low-level zones={len(events)}. "
            "Use a freshly rebuilt executable and a fresh profiler directory."
        )

    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for event in candidates:
        groups[(str(event["run_id"]), int(event["invocation"]))].append(event)
    if invocation_text == "last":
        selected_invocation = max(
            groups,
            key=lambda key: max(int(event["start_cycle"]) for event in groups[key]),
        )
    else:
        requested_invocation = int(invocation_text)
        matching = [key for key in groups if key[1] == requested_invocation]
        if not matching:
            raise RuntimeError(f"Invocation {requested_invocation} is unavailable; found {sorted(groups)}")
        selected_invocation = max(
            matching,
            key=lambda key: max(int(event["start_cycle"]) for event in groups[key]),
        )
    candidates = groups[selected_invocation]

    aggregator_candidates = sorted(
        {
            (int(event["core_x"]), int(event["core_y"]))
            for event in candidates
            if ".Aggregator." in str(event["zone"])
        },
        key=lambda core: (core[1], core[0]),
    )
    aggregator = aggregator_override or (aggregator_candidates[0] if aggregator_candidates else (1, 1))

    available_workers = sorted(
        {
            _core(event)
            for event in candidates
            if ".Worker." in str(event["zone"])
            and (int(event["core_x"]), int(event["core_y"])) != aggregator
        },
        key=_core_order,
    )
    if requested_workers:
        workers = [
            core for core in available_workers if (core[1], core[2]) in requested_workers
        ]
    else:
        workers = available_workers[:worker_count]
    if not workers:
        raise RuntimeError("No requested worker cores contain IVF.Low.* zones")

    allowed = {(core[1], core[2]) for core in workers}
    allowed.add(aggregator)
    slice_end = slice_start + slice_count
    selected: list[dict[str, object]] = []
    for event in candidates:
        coordinate = (int(event["core_x"]), int(event["core_y"]))
        if coordinate not in allowed:
            continue
        if event["item_kind"] == "slice":
            item_id = int(event["item_id"])
            if item_id < slice_start or (slice_count and item_id >= slice_end):
                continue
        selected.append(event)
    return selected, aggregator, workers, selected_invocation


def _phase(zone: str) -> tuple[str, str]:
    leaf = zone.rsplit(".", 1)[-1]
    lower = leaf.lower()
    if "wait" in lower:
        category = "wait"
    elif "read" in lower or "issue" in lower:
        category = "read"
    elif "write" in lower:
        category = "write"
    elif "pack" in lower:
        category = "pack"
    elif "publish" in lower or "credit" in lower:
        category = "publish"
    else:
        category = "compute"
    return leaf, PHASE_COLORS[category]


def _event_label(event: dict[str, object], phase: str, label_mode: str) -> str:
    if label_mode == "none":
        return ""
    display_phase = COMPACT_PHASE_NAMES.get(phase, phase) if label_mode == "compact" else phase
    if event["item_id"] is None:
        return display_phase
    prefix = "S" if event["item_kind"] == "slice" else "P"
    return f"{display_phase} {prefix}{event['item_id']}"


def _timeline_windows(
    events: list[dict[str, object]],
    global_origin: int,
    global_end: int,
    scale: float,
    view: str,
) -> list[tuple[float, float, str]]:
    full_end = max(1.0, (global_end - global_origin) * scale)
    if view == "full":
        return [(0.0, full_end, "Complete batch")]

    def relative_bounds(
        selected: list[dict[str, object]],
        padding_fraction: float,
    ) -> tuple[float, float] | None:
        if not selected:
            return None
        start = (min(int(event["start_cycle"]) for event in selected) - global_origin) * scale
        end = (max(int(event["end_cycle"]) for event in selected) - global_origin) * scale
        span = max(end - start, full_end * 0.002, 0.001)
        padding = max(span * padding_fraction, full_end * 0.002)
        return max(0.0, start - padding), min(full_end, end + padding)

    sampled_worker_events = []
    completion_events = []
    for event in events:
        zone = str(event["zone"])
        worker_sample = (
            ".Worker.Reader." in zone
            or (".Worker.Compute." in zone and not zone.endswith(".OutputPack"))
        )
        if worker_sample:
            sampled_worker_events.append(event)

        aggregator_active = ".Aggregator." in zone and not zone.endswith(
            (".WorkersDoneWait", ".ValuesWait", ".IndicesWait")
        )
        worker_completion = zone.endswith(
            (
                ".Worker.Compute.OutputPack",
                ".Worker.Writer.OutputWrite",
                ".Worker.Writer.Publish",
            )
        )
        if aggregator_active or worker_completion:
            completion_events.append(event)

    sampled_window = relative_bounds(sampled_worker_events, 0.06)
    completion_window = relative_bounds(completion_events, 0.04)
    if sampled_window is None or completion_window is None:
        return [(0.0, full_end, "Complete batch")]

    minimum_break = max(full_end * 0.02, 0.001)
    if sampled_window[1] + minimum_break >= completion_window[0]:
        return [(0.0, full_end, "Complete batch")]
    return [
        (*sampled_window, "Sampled worker slices"),
        (*completion_window, "Completion and aggregation"),
    ]


def plot_pages(
    events: list[dict[str, object]],
    aggregator: tuple[int, int],
    workers: list[tuple[object, int, int]],
    frequency_mhz: float,
    output: Path,
    cores_per_page: int,
    unit: str,
    view: str,
    label_mode: str,
    dpi: int,
    subtitle: str,
) -> list[Path]:
    if PLOTTING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "plot_fine_low_level_timeline.py requires Matplotlib in the active "
            f"Python environment: {PLOTTING_IMPORT_ERROR}"
        )
    assert plt is not None
    assert Patch is not None

    pages = [workers[index : index + cores_per_page] for index in range(0, len(workers), cores_per_page)]
    scale = 1.0 if unit == "cycles" else 1.0 / frequency_mhz
    global_origin = min(int(event["start_cycle"]) for event in events)
    global_end = max(int(event["end_cycle"]) for event in events)
    windows = _timeline_windows(events, global_origin, global_end, scale, view)
    written: list[Path] = []

    if len(windows) == 2:
        omitted_start = windows[0][1]
        omitted_end = windows[1][0]
        suffix = " cycles" if unit == "cycles" else " µs"
        print(
            "Timeline uses a broken axis; omitted middle interval "
            f"{omitted_start:.3f}–{omitted_end:.3f}{suffix} contains "
            "the unsampled candidate slices."
        )

    aggregator_cores = sorted(
        {_core(event) for event in events if (int(event["core_x"]), int(event["core_y"])) == aggregator},
        key=_core_order,
    )
    for page_index, page_workers in enumerate(pages):
        page_cores = aggregator_cores + page_workers
        page_core_set = set(page_cores)
        page_events = [event for event in events if _core(event) in page_core_set]
        core_position = {core: index for index, core in enumerate(page_cores)}
        lanes = sorted(
            {(_core(event), str(event["risc"])) for event in page_events},
            key=lambda lane: (core_position[lane[0]], RISC_ORDER.get(lane[1], 99), lane[1]),
        )
        lane_y: dict[tuple[tuple[object, int, int], str], float] = {}
        labels: list[str] = []
        ticks: list[float] = []
        worker_number = {core: workers.index(core) + 1 for core in page_workers}
        y = 0.0
        previous_core: tuple[object, int, int] | None = None
        for core, risc in lanes:
            if previous_core is not None and core != previous_core:
                y += 0.7
            lane_y[(core, risc)] = y
            _, core_x, core_y = core
            owner = "Aggregator" if (core_x, core_y) == aggregator else f"Worker {worker_number[core]}"
            labels.append(f"{owner} Core({core_x},{core_y}) {risc} {RISC_ROLE.get(risc, 'other')}")
            ticks.append(y)
            y += 1.0
            previous_core = core

        figure_height = max(6.0, 0.47 * len(lanes) + 2.1)
        if len(windows) == 1:
            fig, single_axis = plt.subplots(
                figsize=(17.0, figure_height),
                constrained_layout=True,
            )
            axes = [single_axis]
        else:
            fig, axis_array = plt.subplots(
                1,
                2,
                figsize=(18.5, figure_height),
                sharey=True,
                gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.05},
                constrained_layout=True,
            )
            axes = list(axis_array)

        sorted_events = sorted(page_events, key=lambda item: int(item["start_cycle"]))
        for axis_index, (ax, (window_start, window_end, window_title)) in enumerate(
            zip(axes, windows)
        ):
            window_span = max(window_end - window_start, 0.001)
            for event in sorted_events:
                lane = (_core(event), str(event["risc"]))
                left = (int(event["start_cycle"]) - global_origin) * scale
                duration = max(int(event["duration_cycles"]) * scale, 0.001)
                right = left + duration
                visible_start = max(left, window_start)
                visible_end = min(right, window_end)
                if visible_end <= visible_start:
                    continue
                phase, color = _phase(str(event["zone"]))
                ax.barh(
                    lane_y[lane],
                    duration,
                    left=left,
                    height=0.72,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                )
                label = _event_label(event, phase, label_mode)
                minimum_label_fraction = max(0.035, min(0.18, len(label) * 0.006))
                if label and visible_end - visible_start >= window_span * minimum_label_fraction:
                    ax.text(
                        (visible_start + visible_end) / 2.0,
                        lane_y[lane],
                        label,
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color=(
                            "white"
                            if color not in {PHASE_COLORS["compute"], PHASE_COLORS["pack"]}
                            else "#202020"
                        ),
                        clip_on=True,
                    )

            ax.set_xlim(window_start, window_end)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.set_title(window_title, fontsize=9)
            ax.grid(True, axis="x", linestyle="-", alpha=0.25)
            ax.set_axisbelow(True)
            if axis_index == 0:
                ax.set_yticks(ticks, labels)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)

        if len(axes) == 2:
            axes[0].spines["right"].set_visible(False)
            axes[1].spines["left"].set_visible(False)
            diagonal = 0.009
            break_style = {"color": "#303030", "clip_on": False, "linewidth": 1.0}
            axes[0].plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), transform=axes[0].transAxes, **break_style)
            axes[0].plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), transform=axes[0].transAxes, **break_style)
            axes[1].plot((-diagonal, +diagonal), (-diagonal, +diagonal), transform=axes[1].transAxes, **break_style)
            axes[1].plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), transform=axes[1].transAxes, **break_style)

        fig.suptitle(f"ANN IVF fine search: low-level Tensix RISC timeline only\n{subtitle}")
        fig.supxlabel(
            "Cycles since first selected low-level event"
            if unit == "cycles"
            else "Time since first selected low-level event (µs)"
        )
        axes[-1].legend(
            handles=[
                Patch(facecolor=color, label=category)
                for category, color in PHASE_COLORS.items()
            ],
            title="Low-level phase",
            loc="upper right",
            ncols=3,
            fontsize=7,
        )

        page_output = output_page_path(output, page_index, len(pages))
        page_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(page_output)
    return written


def write_events(path: Path, events: list[dict[str, object]], frequency_mhz: float) -> None:
    fields = [
        "pcie_slot",
        "core_x",
        "core_y",
        "risc",
        "risc_role",
        "run_id",
        "invocation",
        "batch_id",
        "item_kind",
        "item_id",
        "zone",
        "phase",
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
    origin = min(int(event["start_cycle"]) for event in events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for event in sorted(events, key=lambda item: (_core_order(_core(item)), RISC_ORDER.get(str(item["risc"]), 99), int(item["start_cycle"]))):
            profile_start = int(event["start_cycle"])
            profile_end = int(event["end_cycle"])
            start = profile_start - origin
            end = profile_end - origin
            phase, _ = _phase(str(event["zone"]))
            writer.writerow(
                {
                    "pcie_slot": event["pcie_slot"],
                    "core_x": event["core_x"],
                    "core_y": event["core_y"],
                    "risc": event["risc"],
                    "risc_role": RISC_ROLE.get(str(event["risc"]), "other"),
                    "run_id": event["run_id"],
                    "invocation": event["invocation"],
                    "batch_id": event["batch_id"],
                    "item_kind": event["item_kind"] or "",
                    "item_id": "" if event["item_id"] is None else event["item_id"],
                    "zone": event["zone"],
                    "phase": phase,
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
    if args.batch_id < 0 or args.slice_start < 0 or args.slice_count < 0:
        raise ValueError("batch and slice arguments must be non-negative")
    if args.worker_count <= 0 or args.cores_per_page <= 0:
        raise ValueError("worker and page counts must be greater than zero")

    profile_log = discover_profile_log(args.profile_log)
    print(f"Reading low-level device-profiler zones: {profile_log}")
    frequency_mhz, all_events, unmatched = read_low_level_events(profile_log)
    aggregator_override = parse_core(args.aggregator_core) if args.aggregator_core else None
    requested_workers = {parse_core(value) for value in args.core}
    events, aggregator, workers, invocation = select_events(
        all_events,
        args.batch_id,
        args.invocation,
        args.run_id,
        aggregator_override,
        requested_workers,
        args.worker_count,
        args.slice_start,
        args.slice_count,
    )
    mapping = ", ".join(
        f"Worker {index}=Core({core[1]},{core[2]})"
        for index, core in enumerate(workers, start=1)
    )
    print(f"Aggregator=Core({aggregator[0]},{aggregator[1]}); {mapping}")

    run_id, invocation_id = invocation
    subtitle = (
        f"run {run_id}, invocation {invocation_id}, batch {args.batch_id}; "
        f"worker slices {args.slice_start}–{args.slice_start + max(args.slice_count - 1, 0)}"
    )
    events_output = args.events_output or args.output.with_suffix(".csv")
    write_events(events_output, events, frequency_mhz)
    pages = plot_pages(
        events,
        aggregator,
        workers,
        frequency_mhz,
        args.output,
        args.cores_per_page,
        args.unit,
        args.view,
        args.label_mode,
        args.dpi,
        subtitle,
    )
    print(f"Selected {len(events)} low-level zones; wrote {events_output}")
    if unmatched:
        detail = ", ".join(f"{name}={count}" for name, count in sorted(unmatched.items()))
        print(f"Warning: unmatched low-level profiler markers were ignored ({detail})")
    for page in pages:
        print(f"Wrote low-level timeline to {page}")


if __name__ == "__main__":
    main()
