#!/usr/bin/env python3

"""Summarize MemLog_* zones from a TT-Metal device profiler CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-log",
        type=Path,
        default=Path("generated/profiler/.logs/profile_log_device.csv"),
        help="TT-Metal device-profiler CSV",
    )
    parser.add_argument("--dim", type=int, required=True, help="Unpadded vector dimension, for example 100")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("ann_ivf_device_mem_summary.csv"),
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=Path("ann_ivf_device_mem_events.csv"),
    )
    return parser.parse_args()


def find_column(header: list[str], *needles: str) -> int:
    normalized = [name.strip().lower().replace("_", " ") for name in header]
    for index, name in enumerate(normalized):
        if all(needle in name for needle in needles):
            return index
    raise ValueError(f"Could not find column containing {needles!r} in profiler header")


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def zone_bytes(padded_dim: int) -> dict[str, int]:
    vector_block_bytes = padded_dim * 64
    return {
        "MemLog_Fine_Script_DRAM_to_L1": 8192,
        "MemLog_Fine_Query_DRAM_to_L1": vector_block_bytes,
        "MemLog_Fine_Dataset_DRAM_to_L1": vector_block_bytes,
        "MemLog_Fine_Indices_DRAM_to_L1": 4096,
        "MemLog_Fine_DatasetAndIndices_DRAM_to_L1": vector_block_bytes + 4096,
        "MemLog_Fine_Worker_L1_to_DRAM": 2048 + 4096,
        "MemLog_Fine_Worker_L1_to_Core0_L1": 2048 + 4096,
        "MemLog_FinalSort_DRAM_to_L1": 2048 + 4096,
        "MemLog_FinalSort_Core0_L1_to_CB": 2048 + 4096,
        "MemLog_FinalVals_L1_to_DRAM": 2048,
        "MemLog_FinalInds_L1_to_DRAM": 4096,
        "MemLog_Coarse_Query_DRAM_to_L1": vector_block_bytes,
        "MemLog_Coarse_CentroidsAndIndices_DRAM_to_L1": vector_block_bytes + 4096,
        "MemLog_Coarse_OutputValues_L1_to_DRAM": 2048,
        "MemLog_Coarse_OutputIndices_L1_to_DRAM": 4096,
    }


def read_events(profile_log: Path, dim: int) -> tuple[float, list[dict[str, object]]]:
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
        raise ValueError("Could not find CHIP_FREQ[MHz] in profiler log")
    if header_index is None:
        raise ValueError("Could not find profiler CSV header")

    header = rows[header_index]
    slot_col = find_column(header, "pcie", "slot")
    core_x_col = find_column(header, "core", "x")
    core_y_col = find_column(header, "core", "y")
    risc_col = find_column(header, "risc", "processor")
    time_col = find_column(header, "time", "cycle")
    run_col = find_column(header, "run", "id")
    zone_col = find_column(header, "zone", "name")
    try:
        phase_col = find_column(header, "zone", "phase")
    except ValueError:
        # Newer TT-Metal revisions call this column "type" and use ZONE_START/ZONE_END.
        phase_col = next(index for index, name in enumerate(header) if name.strip().lower() == "type")

    padded_dim = (dim + 31) // 32 * 32
    bytes_by_zone = zone_bytes(padded_dim)
    open_zones: dict[tuple[str, ...], list[int]] = defaultdict(list)
    events: list[dict[str, object]] = []

    for row in rows[header_index + 1 :]:
        if len(row) <= max(slot_col, core_x_col, core_y_col, risc_col, time_col, run_col, zone_col, phase_col):
            continue

        zone = row[zone_col].strip()
        if zone not in bytes_by_zone:
            continue

        phase = row[phase_col].strip().lower()
        key = (
            row[slot_col].strip(),
            row[core_x_col].strip(),
            row[core_y_col].strip(),
            row[risc_col].strip(),
            row[run_col].strip(),
            zone,
        )
        cycle = int(row[time_col].strip())

        if phase in {"begin", "start", "zone_start"}:
            open_zones[key].append(cycle)
            continue
        if phase not in {"end", "stop", "zone_end"} or not open_zones[key]:
            continue

        start_cycle = open_zones[key].pop()
        duration_cycles = cycle - start_cycle
        duration_us = duration_cycles / frequency_mhz
        byte_count = bytes_by_zone[zone]
        events.append(
            {
                "run_id": key[4],
                "pcie_slot": key[0],
                "core_x": key[1],
                "core_y": key[2],
                "risc": key[3],
                "zone": zone,
                "bytes": byte_count,
                "start_cycle": start_cycle,
                "end_cycle": cycle,
                "duration_cycles": duration_cycles,
                "duration_us": duration_us,
                "bandwidth_GB_s": byte_count / duration_us / 1000.0,
            }
        )

    # TT-Metal's run_id can remain unchanged across repeated executions of the
    # same program.  Each MemLog zone is deliberately emitted once per core per
    # search, so its per-core occurrence number separates warmup from measured
    # invocations even when run_id does not.
    by_stream: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        stream_key = (
            str(event["run_id"]),
            str(event["pcie_slot"]),
            str(event["core_x"]),
            str(event["core_y"]),
            str(event["risc"]),
            str(event["zone"]),
        )
        by_stream[stream_key].append(event)
    for stream_events in by_stream.values():
        stream_events.sort(key=lambda event: int(event["start_cycle"]))
        for invocation, event in enumerate(stream_events):
            event["invocation"] = invocation

    return frequency_mhz, events


def write_events(path: Path, events: list[dict[str, object]]) -> None:
    fields = [
        "run_id",
        "invocation",
        "pcie_slot",
        "core_x",
        "core_y",
        "risc",
        "zone",
        "bytes",
        "start_cycle",
        "end_cycle",
        "duration_cycles",
        "duration_us",
        "bandwidth_GB_s",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["duration_us"] = f"{float(row['duration_us']):.6f}"
            row["bandwidth_GB_s"] = f"{float(row['bandwidth_GB_s']):.6f}"
            writer.writerow(row)


def write_summary(path: Path, events: list[dict[str, object]]) -> None:
    by_invocation_and_zone: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        key = (str(event["run_id"]), int(event["invocation"]), str(event["zone"]))
        by_invocation_and_zone[key].append(event)

    fields = [
        "run_id",
        "invocation",
        "zone",
        "count",
        "bytes_per_event",
        "min_us",
        "median_us",
        "p95_us",
        "max_us",
        "median_GB_s",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run_id, invocation, zone in sorted(by_invocation_and_zone):
            zone_events = by_invocation_and_zone[(run_id, invocation, zone)]
            durations = [float(event["duration_us"]) for event in zone_events]
            bandwidths = [float(event["bandwidth_GB_s"]) for event in zone_events]
            writer.writerow(
                {
                    "run_id": run_id,
                    "invocation": invocation,
                    "zone": zone,
                    "count": len(zone_events),
                    "bytes_per_event": zone_events[0]["bytes"],
                    "min_us": f"{min(durations):.6f}",
                    "median_us": f"{statistics.median(durations):.6f}",
                    "p95_us": f"{percentile_nearest_rank(durations, 0.95):.6f}",
                    "max_us": f"{max(durations):.6f}",
                    "median_GB_s": f"{statistics.median(bandwidths):.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    frequency_mhz, events = read_events(args.profile_log, args.dim)
    if not events:
        raise RuntimeError("No MemLog_* begin/end zone pairs were found")

    write_events(args.events_output, events)
    write_summary(args.summary_output, events)
    print(
        f"Parsed {len(events)} events at {frequency_mhz:g} MHz; "
        f"wrote {args.events_output} and {args.summary_output}"
    )


if __name__ == "__main__":
    main()
