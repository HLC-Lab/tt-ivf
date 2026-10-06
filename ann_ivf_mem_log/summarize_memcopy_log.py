#!/usr/bin/env python3

"""Create a memcopy-only report from ann_ivf_mem_log.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


FRIENDLY_OPERATION_NAMES = {
    "index_centroids": "Centroids",
    "index_coarse_indices": "Coarse cluster IDs",
    "index_dataset": "Dataset",
    "index_vector_ids": "Dataset vector IDs",
    "query_tiles": "Query",
    "fine_query_tiles": "Reordered fine queries",
    "fine_worker_scripts": "Fine worker scripts",
    "coarse_output_values": "Coarse output values",
    "coarse_output_indices": "Coarse output indices",
    "fine_output_values": "Final output values",
    "fine_output_indices": "Final output indices",
    "query_row_staging_memcpy": "CPU query staging",
}

DIRECTION_NAMES = {
    "host_to_device": "H2D",
    "device_to_host": "D2H",
    "host_to_host": "CPU",
}

SCOPE_ORDER = {
    "one_time_index_load": 0,
    "measured_search": 1,
    "warmup_search": 2,
}

DIRECTION_ORDER = {
    "host_to_device": 0,
    "device_to_host": 1,
    "host_to_host": 2,
}

OPERATION_ORDER = {
    "index_dataset": 0,
    "index_centroids": 1,
    "index_coarse_indices": 2,
    "index_vector_ids": 3,
    "query_tiles": 4,
    "fine_query_tiles": 4.5,
    "fine_worker_scripts": 5,
    "coarse_output_values": 6,
    "coarse_output_indices": 7,
    "fine_output_values": 8,
    "fine_output_indices": 9,
    "query_row_staging_memcpy": 10,
}

OUTPUT_FIELDS = [
    "record_type",
    "scope",
    "context",
    "direction",
    "operation",
    "count",
    "bytes",
    "duration_us",
    "bandwidth_GB_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report only completed CPU/H2D/D2H memory copies. Compute time, "
            "whole-pipeline time, and modeled device traffic are excluded."
        )
    )
    parser.add_argument(
        "--mem-log",
        type=Path,
        default=Path("ann_ivf_mem_log.csv"),
        help="CSV written by the ann_ivf_mem_log executable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ann_ivf_memcopy_summary.csv"),
        help="Machine-readable detail and totals CSV",
    )
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include the warmup search; it is excluded by default",
    )
    return parser.parse_args()


def read_completed_copies(path: Path, include_warmup: bool) -> list[dict[str, object]]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "context",
            "layer",
            "direction",
            "operation",
            "bytes",
            "duration_us",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        copies: list[dict[str, object]] = []
        for row in reader:
            context = row["context"].strip()
            layer = row["layer"].strip()
            direction = row["direction"].strip()
            duration_text = row["duration_us"].strip()

            # These are the only layers containing directly timed copy calls.
            # In particular, device_pipeline_effective includes compute and is
            # deliberately not a memcopy measurement.
            if layer not in {"pcie", "cpu"} or not duration_text:
                continue
            if context == "warmup" and not include_warmup:
                continue
            if direction not in DIRECTION_NAMES:
                continue

            copies.append(
                {
                    "context": context,
                    "layer": layer,
                    "direction": direction,
                    "operation": row["operation"].strip(),
                    "bytes": int(row["bytes"]),
                    "duration_us": float(duration_text),
                }
            )
    return copies


def context_scope(context: str) -> str:
    if context == "index_load":
        return "one_time_index_load"
    if context == "warmup":
        return "warmup_search"
    if context.startswith("run_"):
        return "measured_search"
    return context


def context_order(context: str) -> tuple[int, int | str]:
    if context == "index_load":
        return (0, 0)
    if context.startswith("run_") and context[4:].isdigit():
        return (1, int(context[4:]))
    if context == "warmup":
        return (2, 0)
    return (3, context)


def aggregate_rows(copies: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for copy in copies:
        context = str(copy["context"])
        key = (
            context_scope(context),
            context,
            str(copy["direction"]),
            str(copy["operation"]),
        )
        grouped[key].append(copy)

    rows: list[dict[str, object]] = []
    by_context_and_direction: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    by_context_pcie: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (
            SCOPE_ORDER.get(item[0][0], 99),
            context_order(item[0][1]),
            DIRECTION_ORDER.get(item[0][2], 99),
            OPERATION_ORDER.get(item[0][3], 99),
            item[0][3],
        ),
    )
    for (scope, context, direction, operation), records in sorted_groups:
        byte_count = sum(int(record["bytes"]) for record in records)
        duration_us = sum(float(record["duration_us"]) for record in records)
        rows.append(
            make_output_row(
                "detail",
                scope,
                context,
                direction,
                operation,
                len(records),
                byte_count,
                duration_us,
            )
        )
        by_context_and_direction[(scope, context, direction)].extend(records)
        if direction in {"host_to_device", "device_to_host"}:
            by_context_pcie[(scope, context)].extend(records)

    sorted_direction_totals = sorted(
        by_context_and_direction.items(),
        key=lambda item: (
            SCOPE_ORDER.get(item[0][0], 99),
            context_order(item[0][1]),
            DIRECTION_ORDER.get(item[0][2], 99),
        ),
    )
    for (scope, context, direction), records in sorted_direction_totals:
        rows.append(
            make_output_row(
                "direction_total",
                scope,
                context,
                direction,
                f"total_{direction}",
                len(records),
                sum(int(record["bytes"]) for record in records),
                sum(float(record["duration_us"]) for record in records),
            )
        )

    sorted_pcie_totals = sorted(
        by_context_pcie.items(),
        key=lambda item: (
            SCOPE_ORDER.get(item[0][0], 99),
            context_order(item[0][1]),
        ),
    )
    for (scope, context), records in sorted_pcie_totals:
        rows.append(
            make_output_row(
                "pcie_total",
                scope,
                context,
                "host_device_round_trip",
                "total_H2D_plus_D2H",
                len(records),
                sum(int(record["bytes"]) for record in records),
                sum(float(record["duration_us"]) for record in records),
            )
        )

    return rows


def make_output_row(
    record_type: str,
    scope: str,
    context: str,
    direction: str,
    operation: str,
    count: int,
    byte_count: int,
    duration_us: float,
) -> dict[str, object]:
    bandwidth = byte_count / duration_us / 1000.0 if duration_us > 0.0 else 0.0
    return {
        "record_type": record_type,
        "scope": scope,
        "context": context,
        "direction": direction,
        "operation": operation,
        "count": count,
        "bytes": byte_count,
        "duration_us": duration_us,
        "bandwidth_GB_s": bandwidth,
    }


def format_size(byte_count: int) -> str:
    if byte_count >= 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.2f} MiB"
    if byte_count >= 1024:
        return f"{byte_count / 1024:.2f} KiB"
    return f"{byte_count} B"


def print_report(rows: list[dict[str, object]]) -> None:
    detail_rows = [row for row in rows if row["record_type"] == "detail"]
    contexts: list[tuple[str, str]] = []
    for row in detail_rows:
        key = (str(row["scope"]), str(row["context"]))
        if key not in contexts:
            contexts.append(key)

    print("\n=== ANN IVF MEMCOPY-ONLY REPORT ===")
    print("Compute and whole-pipeline timings are excluded from all totals.")

    for scope, context in contexts:
        if scope == "one_time_index_load":
            title = "ONE-TIME INDEX LOAD"
        elif scope == "measured_search":
            title = "MEASURED SEARCH"
        elif scope == "warmup_search":
            title = "WARMUP SEARCH"
        else:
            title = scope.replace("_", " ").upper()

        print(f"\n[{title}: {context}]")
        print(f"{'Direction':<10} {'Operation':<25} {'Size':>12} {'Time (us)':>12} {'GB/s':>10}")
        print("-" * 73)
        for row in detail_rows:
            if row["scope"] != scope or row["context"] != context:
                continue
            operation = FRIENDLY_OPERATION_NAMES.get(str(row["operation"]), str(row["operation"]))
            direction = DIRECTION_NAMES.get(str(row["direction"]), str(row["direction"]))
            print(
                f"{direction:<10} {operation:<25} "
                f"{format_size(int(row['bytes'])):>12} "
                f"{float(row['duration_us']):>12.3f} "
                f"{float(row['bandwidth_GB_s']):>10.3f}"
            )

        totals = [
            row
            for row in rows
            if row["scope"] == scope
            and row["context"] == context
            and row["record_type"] in {"direction_total", "pcie_total"}
        ]
        totals.sort(
            key=lambda row: (
                3 if row["record_type"] == "pcie_total" else DIRECTION_ORDER.get(str(row["direction"]), 99)
            )
        )
        print("-" * 73)
        for row in totals:
            direction = str(row["direction"])
            if row["record_type"] == "pcie_total":
                label = "TOTAL PCIe (H2D + D2H)"
            else:
                label = f"TOTAL {DIRECTION_NAMES.get(direction, direction)}"
            print(
                f"{label:<36} {format_size(int(row['bytes'])):>12} "
                f"{float(row['duration_us']):>12.3f} "
                f"{float(row['bandwidth_GB_s']):>10.3f}"
            )

    print(
        "\nOne-time index H2D is intentionally separate from per-search PCIe cost. "
        "CPU copy time is also not added to the PCIe total."
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["duration_us"] = f"{float(row['duration_us']):.6f}"
            formatted["bandwidth_GB_s"] = f"{float(row['bandwidth_GB_s']):.6f}"
            writer.writerow(formatted)


def main() -> None:
    args = parse_args()
    copies = read_completed_copies(args.mem_log, args.include_warmup)
    if not copies:
        raise RuntimeError("No completed CPU/H2D/D2H copy records were found. " "Check --mem-log and --include-warmup.")

    rows = aggregate_rows(copies)
    print_report(rows)
    write_csv(args.output, rows)
    print(f"\nMachine-readable memcopy report written to {args.output}")


if __name__ == "__main__":
    main()
