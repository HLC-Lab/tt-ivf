#!/usr/bin/env python3

"""Plot ANN IVF device-profiler zones as per-Tensix RISC timelines."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ModuleNotFoundError as error:
    plt = None
    Patch = None
    PLOTTING_IMPORT_ERROR = error
else:
    PLOTTING_IMPORT_ERROR = None


ROLE_COLORS = {
    "writer": "#F45151",
    "compute": "#F2C500",
    "reader": "#7559E8",
}

# Current names plus the previous generic core-(0,0) names, so the plotter can
# also consume profiler logs captured immediately before the explicit rename.
ZONE_SPECS = {
    # Fine-search workers.
    "Fine_Search_Reader": ("reader", "kernel"),
    "Fine_Search_Compute": ("compute", "kernel"),
    "Fine_Search_Writer": ("writer", "kernel"),
    "Fine_Search_Reader_QueryBatch": ("reader", "batch"),
    "Fine_Search_Compute_QueryBatch": ("compute", "batch"),
    "Fine_Search_Writer_QueryBatch": ("writer", "batch"),
    # Core (0,0) final-sort aggregator: current names.
    "Core_0_0_Final_Sort_Reader_Main": ("reader", "kernel"),
    "Core_0_0_Final_Sort_Compute_Main": ("compute", "kernel"),
    "Core_0_0_Final_Sort_Writer_Main": ("writer", "kernel"),
    "Core_0_0_Final_Sort_Reader_QueryBatch": ("reader", "batch"),
    "Core_0_0_Final_Sort_Compute_QueryBatch": ("compute", "batch"),
    "Core_0_0_Final_Sort_Writer_QueryBatch": ("writer", "batch"),
    # Core (0,0) final-sort aggregator: legacy names.
    "Final_Sort_Reader_Main": ("reader", "kernel"),
    "Final_Sort_Compute_Main": ("compute", "kernel"),
    "Final_Sort_Writer_Main": ("writer", "kernel"),
    "Final_Sort_Batch": ("reader", "batch"),
    "Final_Sort_Compute_Batch": ("compute", "batch"),
    "Final_Sort_Writer_Batch": ("writer", "batch"),
}

ID_TO_BATCH_ZONE = {
    "Fine_Search_Reader_QueryBatch_ID": "Fine_Search_Reader_QueryBatch",
    "Fine_Search_Compute_QueryBatch_ID": "Fine_Search_Compute_QueryBatch",
    "Fine_Search_Writer_QueryBatch_ID": "Fine_Search_Writer_QueryBatch",
    "Core_0_0_Final_Sort_Reader_QueryBatch_ID": "Core_0_0_Final_Sort_Reader_QueryBatch",
    "Core_0_0_Final_Sort_Compute_QueryBatch_ID": "Core_0_0_Final_Sort_Compute_QueryBatch",
    "Core_0_0_Final_Sort_Writer_QueryBatch_ID": "Core_0_0_Final_Sort_Writer_QueryBatch",
    "Final_Sort_Reader_QueryBatch_ID": "Final_Sort_Batch",
    "Final_Sort_Compute_QueryBatch_ID": "Final_Sort_Compute_Batch",
    "Final_Sort_Writer_QueryBatch_ID": "Final_Sort_Writer_Batch",
}

START_PHASES = {"begin", "start", "zone_start"}
END_PHASES = {"end", "stop", "zone_end"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create per-core reader/compute/writer timelines from the " "TT-Metal device-profiler CSV.")
    )
    parser.add_argument(
        "--profile-log",
        type=Path,
        help=(
            "TT-Metal profile_log_device.csv; if omitted, discover it from "
            "TT_METAL_PROFILER_DIR, TT_METAL_HOME, or generated/profiler"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ann_ivf_device_kernel_timeline.png"),
        help="PNG path; multiple pages receive a _page_NN suffix",
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        help="Selected event CSV; defaults to the output path with .csv",
    )
    parser.add_argument(
        "--granularity",
        choices=("batch", "kernel"),
        default="batch",
        help=(
            "batch plots per-32-query DeviceZoneScopedN zones; kernel plots " "one complete kernel invocation per lane"
        ),
    )
    parser.add_argument(
        "--last-batches",
        type=int,
        default=10,
        help="For batch granularity, retain this many batches from the last invocation",
    )
    parser.add_argument(
        "--last-invocations",
        type=int,
        default=1,
        help=(
            "For kernel granularity, retain this many measured invocations "
            "after dropping the first warmup invocation"
        ),
    )
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="For kernel granularity, do not remove the first invocation",
    )
    parser.add_argument(
        "--core",
        action="append",
        default=[],
        metavar="X,Y",
        help="Restrict to a profiler core coordinate; repeat as needed",
    )
    parser.add_argument(
        "--cores-per-page",
        type=int,
        default=4,
        help="Maximum Tensix cores in each PNG",
    )
    parser.add_argument("--unit", choices=("cycles", "us"), default="cycles")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def discover_profile_log(requested: Path | None) -> Path:
    filename = "profile_log_device.csv"

    if requested is not None and requested.is_file():
        return requested

    # "$TT_METAL_PROFILER_DIR/.logs/..." becomes "/.logs/..." when the
    # variable is unset. Treat only this recognizable expansion mistake as
    # auto-discovery; other explicit missing paths remain hard errors.
    empty_env_expansion = requested == Path("/.logs") / filename
    if requested is not None and not empty_env_expansion:
        raise FileNotFoundError(
            f"Device-profiler log does not exist: {requested}\n"
            "Run with TT_METAL_DEVICE_PROFILER=1, or pass the correct "
            "--profile-log path."
        )

    candidates: list[Path] = []
    profiler_dir = os.environ.get("TT_METAL_PROFILER_DIR")
    if profiler_dir:
        candidates.append(Path(profiler_dir) / ".logs" / filename)

    metal_home = os.environ.get("TT_METAL_HOME")
    if metal_home:
        candidates.append(Path(metal_home) / "generated" / "profiler" / ".logs" / filename)

    candidates.append(Path.cwd() / "generated" / "profiler" / ".logs" / filename)
    repository_root = Path(__file__).resolve().parents[1]
    candidates.append(repository_root / "generated" / "profiler" / ".logs" / filename)

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
        if candidate.is_file():
            if empty_env_expansion:
                print(
                    "Warning: TT_METAL_PROFILER_DIR was empty and expanded to "
                    f"{requested}; using {candidate} instead."
                )
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in unique_candidates)
    environment_note = "TT_METAL_PROFILER_DIR is unset. " if not profiler_dir else ""
    raise FileNotFoundError(
        f"Could not find {filename}. {environment_note}"
        f"Searched:\n  {searched}\n"
        "Run the benchmark with TT_METAL_DEVICE_PROFILER=1 first, or pass "
        "--profile-log /absolute/path/to/profile_log_device.csv."
    )


def find_column(header: list[str], *needles: str) -> int:
    normalized = [name.strip().lower().replace("_", " ") for name in header]
    for index, name in enumerate(normalized):
        if all(needle in name for needle in needles):
            return index
    raise ValueError(f"Could not find column containing {needles!r}")


def find_exact_column(header: list[str], *names: str) -> int:
    normalized = [name.strip().lower().replace("_", " ") for name in header]
    accepted = {name.strip().lower().replace("_", " ") for name in names}
    for index, name in enumerate(normalized):
        if name in accepted:
            return index
    raise ValueError(f"Could not find any column named {sorted(accepted)!r}")


def parse_core(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", value)
    if match is None:
        raise ValueError(f"Invalid --core {value!r}; expected X,Y")
    return int(match.group(1)), int(match.group(2))


def normalize_risc(value: str) -> str:
    text = value.strip().upper().replace("TENSIX_", "")
    text = re.sub(r"^TRISC(\d)$", r"TRISC_\1", text)
    return text


def read_profile_events(profile_log: Path) -> tuple[float, list[dict[str, object]], int]:
    with profile_log.open(newline="") as profile_file:
        rows = list(csv.reader(profile_file))

    frequency_mhz: float | None = None
    header_index: int | None = None
    for row_index, row in enumerate(rows):
        joined = ",".join(row)
        frequency_match = re.search(r"CHIP_FREQ\[MHz\]\s*:\s*([0-9.]+)", joined)
        if frequency_match:
            frequency_mhz = float(frequency_match.group(1))
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

    needed_column = max(
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
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if len(row) <= needed_column:
            continue

        name = row[zone_col].strip()
        if name not in ZONE_SPECS and name not in ID_TO_BATCH_ZONE:
            continue

        try:
            cycle = int(row[time_col].strip())
            core_x = int(row[core_x_col].strip())
            core_y = int(row[core_y_col].strip())
        except ValueError:
            continue

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
            }
        )

    events: list[dict[str, object]] = []
    unmatched_zone_ends = 0
    for stream, markers in markers_by_stream.items():
        markers.sort(key=lambda marker: (int(marker["cycle"]), int(marker["row_number"])))
        pending_ids: dict[str, list[int]] = defaultdict(list)
        open_zones: dict[str, list[tuple[int, int | None]]] = defaultdict(list)

        for marker in markers:
            name = str(marker["name"])
            marker_type = str(marker["type"])

            if name in ID_TO_BATCH_ZONE:
                if marker_type in {"ts_data", "data", "timestamped_data"}:
                    pending_ids[ID_TO_BATCH_ZONE[name]].append(int(marker["data"]))
                continue

            if marker_type in START_PHASES:
                batch_id = pending_ids[name].pop() if pending_ids[name] else None
                open_zones[name].append((int(marker["cycle"]), batch_id))
                continue

            if marker_type not in END_PHASES:
                continue
            if not open_zones[name]:
                unmatched_zone_ends += 1
                continue

            start_cycle, batch_id = open_zones[name].pop()
            end_cycle = int(marker["cycle"])
            if end_cycle < start_cycle:
                continue

            role, granularity = ZONE_SPECS[name]
            events.append(
                {
                    "pcie_slot": stream[0],
                    "core_x": stream[1],
                    "core_y": stream[2],
                    "risc": stream[3],
                    "run_id": stream[4],
                    "zone": name,
                    "role": role,
                    "granularity": granularity,
                    "batch_id": batch_id,
                    "batch_id_from_marker": batch_id is not None,
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "duration_cycles": end_cycle - start_cycle,
                }
            )

    assign_invocations(events)
    return frequency_mhz, events, unmatched_zone_ends


def event_stream_key(event: dict[str, object]) -> tuple[object, ...]:
    return (
        event["pcie_slot"],
        event["core_x"],
        event["core_y"],
        event["risc"],
        event["zone"],
    )


def assign_invocations(events: list[dict[str, object]]) -> None:
    by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        by_stream[event_stream_key(event)].append(event)

    for stream_events in by_stream.values():
        stream_events.sort(key=lambda event: int(event["start_cycle"]))
        if str(stream_events[0]["granularity"]) == "kernel":
            for invocation, event in enumerate(stream_events):
                event["invocation"] = invocation
            continue

        invocation = 0
        previous_batch: int | None = None
        for occurrence, event in enumerate(stream_events):
            batch_id = event["batch_id"]
            if isinstance(batch_id, int):
                if previous_batch is not None and batch_id <= previous_batch:
                    invocation += 1
                previous_batch = batch_id
                event["invocation"] = invocation
            else:
                # Logs captured before QueryBatch_ID markers were added cannot
                # separate search invocations. Preserve chronological order.
                event["invocation"] = 0
                event["batch_id"] = occurrence


def select_batch_events(
    events: list[dict[str, object]], last_batches: int
) -> tuple[list[dict[str, object]], list[int], bool]:
    batch_events = [event for event in events if event["granularity"] == "batch"]
    if not batch_events:
        raise RuntimeError("No ANN IVF QueryBatch zone pairs were found")

    had_real_batch_ids = all(bool(event["batch_id_from_marker"]) for event in batch_events)

    by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for event in batch_events:
        by_stream[event_stream_key(event)].append(event)

    last_invocation_events: list[dict[str, object]] = []
    for stream_events in by_stream.values():
        last_invocation = max(int(event["invocation"]) for event in stream_events)
        last_invocation_events.extend(event for event in stream_events if int(event["invocation"]) == last_invocation)

    batch_ids = sorted(
        {int(event["batch_id"]) for event in last_invocation_events if isinstance(event["batch_id"], int)}
    )
    selected_batch_ids = batch_ids[-last_batches:]
    selected_batch_id_set = set(selected_batch_ids)
    selected = [event for event in last_invocation_events if int(event["batch_id"]) in selected_batch_id_set]

    # Give the selected window compact, presentation-friendly names while
    # preserving the original profiler batch_id on every event. Thus the last
    # ten recorded batches are always shown as Query_B1 ... Query_B10 even
    # when their source IDs are, for example, 302 ... 311.
    display_name_by_batch_id = {
        batch_id: f"Query_B{display_index}" for display_index, batch_id in enumerate(selected_batch_ids, start=1)
    }
    for event in selected:
        event["query_batch"] = display_name_by_batch_id[int(event["batch_id"])]

    return selected, selected_batch_ids, had_real_batch_ids


def select_kernel_events(
    events: list[dict[str, object]],
    last_invocations: int,
    include_warmup: bool,
) -> list[dict[str, object]]:
    kernel_events = [event for event in events if event["granularity"] == "kernel"]
    if not kernel_events:
        raise RuntimeError("No ANN IVF complete-kernel zone pairs were found")

    by_stream: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for event in kernel_events:
        by_stream[event_stream_key(event)].append(event)

    selected: list[dict[str, object]] = []
    for stream_events in by_stream.values():
        stream_events.sort(key=lambda event: int(event["start_cycle"]))
        measured_events = stream_events if include_warmup else stream_events[1:]
        if not measured_events:
            measured_events = stream_events
        selected.extend(measured_events[-last_invocations:])
    return selected


def risc_sort_key(risc: str) -> tuple[int, str]:
    order = {
        "BRISC": 0,
        "NCRISC": 1,
        "TRISC_0": 2,
        "TRISC_1": 3,
        "TRISC_2": 4,
    }
    return order.get(risc, 10), risc


def core_key(event: dict[str, object]) -> tuple[object, ...]:
    return event["pcie_slot"], int(event["core_x"]), int(event["core_y"])


def output_page_path(output: Path, page_index: int, page_count: int) -> Path:
    if page_count == 1:
        return output
    return output.with_name(f"{output.stem}_page_{page_index + 1:02d}{output.suffix}")


def plot_pages(
    events: list[dict[str, object]],
    frequency_mhz: float,
    output: Path,
    cores_per_page: int,
    unit: str,
    dpi: int,
    subtitle: str,
) -> list[Path]:
    if PLOTTING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "plot_device_kernel_timeline.py requires Matplotlib in the active "
            f"Python environment: {PLOTTING_IMPORT_ERROR}"
        )
    assert plt is not None
    assert Patch is not None

    cores = sorted({core_key(event) for event in events})
    if not cores:
        raise RuntimeError("No cores remain after filtering")

    pages = [cores[index : index + cores_per_page] for index in range(0, len(cores), cores_per_page)]
    # Profiler timestamps are absolute device cycles. Keep durations in raw
    # cycles, but translate the selected timeline so its first event is t=0.
    # One shared origin keeps every output page directly comparable.
    global_origin = min(int(event["start_cycle"]) for event in events)
    global_end = max(int(event["end_cycle"]) for event in events)
    scale = 1.0 if unit == "cycles" else 1.0 / frequency_mhz
    x_start = 0.0
    x_end = (global_end - global_origin) * scale
    if x_end <= x_start:
        x_end = x_start + 1.0
    written: list[Path] = []

    for page_index, page_cores in enumerate(pages):
        page_core_set = set(page_cores)
        page_events = [event for event in events if core_key(event) in page_core_set]
        lanes = sorted(
            {(core_key(event), str(event["risc"])) for event in page_events},
            key=lambda lane: (lane[0], risc_sort_key(lane[1])),
        )

        lane_positions: dict[tuple[tuple[object, ...], str], float] = {}
        lane_labels: list[str] = []
        y_ticks: list[float] = []
        y = 0.0
        previous_core: tuple[object, ...] | None = None
        for lane_core, risc in lanes:
            if previous_core is not None and lane_core != previous_core:
                y += 0.8
            lane_positions[(lane_core, risc)] = y
            slot, core_x, core_y = lane_core
            slot_prefix = f"Chip{slot}-" if len({core[0] for core in cores}) > 1 else ""
            lane_labels.append(f"{slot_prefix}Core({core_x},{core_y})-{risc}")
            y_ticks.append(y)
            y += 1.0
            previous_core = lane_core

        figure_height = max(4.8, 0.42 * len(lanes) + 1.8)
        displayed_batches = {str(event["query_batch"]) for event in page_events if event.get("query_batch")}
        figure_width = max(12.5, 4.0 + 1.05 * len(displayed_batches))
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)

        for event in sorted(page_events, key=lambda item: int(item["start_cycle"])):
            lane = (core_key(event), str(event["risc"]))
            start = (int(event["start_cycle"]) - global_origin) * scale
            duration = max(int(event["duration_cycles"]) * scale, 0.001)
            query_batch = str(event.get("query_batch", ""))
            role = str(event["role"])
            ax.barh(
                lane_positions[lane],
                duration,
                left=start,
                height=0.72,
                color=ROLE_COLORS[role],
                edgecolor="white" if query_batch else "none",
                linewidth=0.45 if query_batch else 0.0,
            )
            if query_batch:
                ax.text(
                    start + duration / 2.0,
                    lane_positions[lane],
                    query_batch,
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#202020" if role == "compute" else "white",
                    clip_on=True,
                )

        ax.set_yticks(y_ticks, lane_labels)
        ax.set_xlim(x_start, x_end)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlabel(
            "Cycles since first selected event" if unit == "cycles" else "Time since first selected event (µs)"
        )
        ax.set_ylabel("Tensix core + RISC")
        ax.set_title(f"ANN IVF Device Kernel Timeline per Core\n{subtitle}")
        ax.grid(True, axis="x", linestyle="-", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(
            handles=[
                Patch(facecolor=ROLE_COLORS["writer"], label="Writer kernels"),
                Patch(facecolor=ROLE_COLORS["compute"], label="Compute kernels"),
                Patch(facecolor=ROLE_COLORS["reader"], label="Reader kernels"),
            ],
            title="Kernel type",
            loc="upper right",
        )

        page_output = output_page_path(output, page_index, len(pages))
        page_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(page_output)

    return written


def write_selected_events(
    path: Path,
    events: list[dict[str, object]],
    frequency_mhz: float,
) -> None:
    fields = [
        "pcie_slot",
        "core_x",
        "core_y",
        "risc",
        "role",
        "zone",
        "run_id",
        "invocation",
        "query_batch",
        "batch_id",
        "batch_id_from_marker",
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
    timeline_origin = min(int(event["start_cycle"]) for event in events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for event in sorted(
            events,
            key=lambda item: (
                str(item["pcie_slot"]),
                int(item["core_x"]),
                int(item["core_y"]),
                risc_sort_key(str(item["risc"])),
                int(item["start_cycle"]),
            ),
        ):
            profile_start = int(event["start_cycle"])
            profile_end = int(event["end_cycle"])
            start_cycle = profile_start - timeline_origin
            end_cycle = profile_end - timeline_origin
            writer.writerow(
                {
                    "pcie_slot": event["pcie_slot"],
                    "core_x": event["core_x"],
                    "core_y": event["core_y"],
                    "risc": event["risc"],
                    "role": event["role"],
                    "zone": event["zone"],
                    "run_id": event["run_id"],
                    "invocation": event["invocation"],
                    "query_batch": event.get("query_batch", ""),
                    "batch_id": "" if event["batch_id"] is None else event["batch_id"],
                    "batch_id_from_marker": event["batch_id_from_marker"],
                    "timeline_origin_cycle": timeline_origin,
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "duration_cycles": event["duration_cycles"],
                    "start_us": f"{start_cycle / frequency_mhz:.6f}",
                    "end_us": f"{end_cycle / frequency_mhz:.6f}",
                    "duration_us": f"{int(event['duration_cycles']) / frequency_mhz:.6f}",
                    "profile_start_cycle": profile_start,
                    "profile_end_cycle": profile_end,
                }
            )


def main() -> None:
    args = parse_args()
    if args.last_batches <= 0:
        raise ValueError("--last-batches must be greater than zero")
    if args.last_invocations <= 0:
        raise ValueError("--last-invocations must be greater than zero")
    if args.cores_per_page <= 0:
        raise ValueError("--cores-per-page must be greater than zero")

    requested_cores = {parse_core(value) for value in args.core}
    profile_log = discover_profile_log(args.profile_log)
    print(f"Reading device-profiler log: {profile_log}")
    frequency_mhz, events, unmatched_zone_ends = read_profile_events(profile_log)

    if args.granularity == "batch":
        selected, batch_ids, had_real_batch_ids = select_batch_events(events, args.last_batches)
        if not selected:
            raise RuntimeError("No batch events remain after selecting the last invocation")
        batch_text = (
            f"last measured invocation; Query_B1–Query_B{len(batch_ids)} "
            f"(source IDs {batch_ids[0]}–{batch_ids[-1]})"
            if batch_ids
            else "last measured invocation"
        )
        subtitle = f"{batch_text}; warmup invocation excluded"
    else:
        selected = select_kernel_events(
            events,
            args.last_invocations,
            args.include_warmup,
        )
        had_real_batch_ids = True
        warmup_text = "warmup included" if args.include_warmup else "first warmup invocation excluded"
        subtitle = f"last {args.last_invocations} measured kernel invocation(s); {warmup_text}"

    if requested_cores:
        selected = [event for event in selected if (int(event["core_x"]), int(event["core_y"])) in requested_cores]
        if not selected:
            raise RuntimeError("None of the requested --core coordinates have selected ANN IVF events")

    events_output = args.events_output or args.output.with_suffix(".csv")
    write_selected_events(events_output, selected, frequency_mhz)
    page_outputs = plot_pages(
        selected,
        frequency_mhz,
        args.output,
        args.cores_per_page,
        args.unit,
        args.dpi,
        subtitle,
    )

    core_count = len({core_key(event) for event in selected})
    print(
        f"Paired {len(events)} ANN IVF zones at {frequency_mhz:g} MHz; "
        f"selected {len(selected)} events on {core_count} cores."
    )
    if args.granularity == "batch" and not had_real_batch_ids:
        print(
            "Warning: some QueryBatch_ID markers were unavailable; chronological "
            "occurrence numbers were used and warmup separation may be approximate."
        )
    if unmatched_zone_ends:
        print(
            f"Warning: ignored {unmatched_zone_ends} unmatched zone ends, "
            "usually caused by the 125-scope profiler-buffer limit."
        )
    print(f"Wrote event data to {events_output}")
    for page_output in page_outputs:
        print(f"Wrote timeline to {page_output}")


if __name__ == "__main__":
    main()
