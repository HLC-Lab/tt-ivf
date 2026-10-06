#!/usr/bin/env python3
"""Summarize measured search timings; no third-party packages are needed.

Accept a timing CSV or a run directory. Warmup is excluded by the CSV writer.
CSV durations are in microseconds; this report presents medians in milliseconds.
Detail rows are components of their parent stages, not additional costs. Column
medians need not sum exactly even though each individual run's stages do.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


CONFIG_COLUMNS = (
    "nlist", "nprobe", "queries", "k", "grouping", "partitions", "reader",
    "bank_schedule", "aggregation", "mask", "layout", "query_broadcast",
    "leader_prefetch", "worker_input", "grouping_window", "grouping_lookahead",
)

# (CSV column, display label, parent). Parentless entries are stage totals.
TIMINGS = (
    ("h2d_us", "H2D", ""),
    ("coarse_query_h2d_us", "Original query upload", "h2d_us"),
    ("fine_query_h2d_us", "Reordered query upload", "h2d_us"),
    ("descriptor_h2d_us", "Descriptor upload", "h2d_us"),
    ("control_h2d_us", "L1 control initialization", "h2d_us"),
    ("cpu_coarse_config_us", "CPU coarse setup", ""),
    ("query_normalize_us", "Query normalization", "cpu_coarse_config_us"),
    ("coarse_host_setup_us", "Coarse buffers / tiling / program", "cpu_coarse_config_us"),
    ("coarse_runtime_setup_us", "Coarse runtime arguments / workload", "cpu_coarse_config_us"),
    ("tt_coarse_search_us", "TT coarse search", ""),
    ("cpu_fine_prep_us", "CPU fine prep", ""),
    ("coarse_readback_us", "Coarse results readback", "cpu_fine_prep_us"),
    ("coarse_decode_us", "Coarse untilize / list selection", "cpu_fine_prep_us"),
    ("list_metadata_us", "List metadata", "cpu_fine_prep_us"),
    ("query_grouping_us", "Query similarity grouping", "cpu_fine_prep_us"),
    ("query_reorder_us", "Query reordering / tiling", "cpu_fine_prep_us"),
    ("partition_schedule_us", "Partition / worker scheduling", "cpu_fine_prep_us"),
    ("descriptor_pack_us", "Descriptor packing", "cpu_fine_prep_us"),
    ("fine_other_prep_us", "Fine buffers / program setup / misc", "cpu_fine_prep_us"),
    ("tt_fine_search_us", "TT fine search", ""),
    ("cpu_output_us", "CPU output", ""),
    ("final_readback_us", "Final results readback", "cpu_output_us"),
    ("output_restore_us", "Output untilize / original order", "cpu_output_us"),
    ("stage_sum_us", "Stage sum", ""),
    ("pipeline_us", "End-to-end", ""),
)
STAGE_COLUMNS = (
    "h2d_us", "cpu_coarse_config_us", "tt_coarse_search_us",
    "cpu_fine_prep_us", "tt_fine_search_us", "cpu_output_us",
)


def read_runs(paths: list[Path], allow_empty: bool = False) -> dict[tuple[str, ...], list[dict[str, object]]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[tuple[str, ...], int]] = set()
    required = {"run", "qps", "recall", *CONFIG_COLUMNS, *(name for name, _, _ in TIMINGS)}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing timing columns {sorted(missing)}; rebuild and capture new runs")
            count = 0
            for line, row in enumerate(reader, start=2):
                context = f"{path}:{line}"
                if None in row or any(row.get(name) is None or not row[name].strip() for name in required):
                    raise ValueError(f"{context}: incomplete or unexpected CSV row")
                config = tuple(row[name].strip() for name in CONFIG_COLUMNS)
                run = int(row["run"])
                if run <= 0 or (config, run) in seen:
                    raise ValueError(f"{context}: non-positive or duplicate measured run {run}")
                seen.add((config, run))
                numeric = {name: float(row[name]) for name in {"qps", "recall", *(field[0] for field in TIMINGS)}}
                if any(not math.isfinite(value) or value < 0 for value in numeric.values()):
                    raise ValueError(f"{context}: negative or non-finite timing/result")
                if numeric["qps"] <= 0 or numeric["recall"] > 1 or numeric["pipeline_us"] <= 0:
                    raise ValueError(f"{context}: invalid QPS, recall or pipeline time")
                tolerance = max(1e-5, numeric["pipeline_us"] * 1e-8)
                for total, parts in (
                    ("pipeline_us", STAGE_COLUMNS),
                    ("stage_sum_us", STAGE_COLUMNS),
                    *((parent, tuple(name for name, _, root in TIMINGS if root == parent))
                      for parent in ("h2d_us", "cpu_coarse_config_us", "cpu_fine_prep_us", "cpu_output_us")),
                ):
                    if abs(numeric[total] - sum(numeric[name] for name in parts)) > tolerance:
                        raise ValueError(f"{context}: {total} does not equal its components")
                queries = int(row["queries"])
                if queries <= 0 or not math.isclose(numeric["qps"], queries * 1e6 / numeric["pipeline_us"], rel_tol=1e-8):
                    raise ValueError(f"{context}: QPS disagrees with queries/pipeline time")
                groups[config].append({"run": run, "source": path, **numeric})
                count += 1
            if not count and not allow_empty:
                raise ValueError(f"{path}: no measured runs")
    return dict(sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]), item[0])))


SCAN_COLUMNS = (
    "batch", "valid_queries", "partition", "union_lists", "union_pages", "union_vectors",
    "database_vectors", "scanned_fraction", "scanned_percent", "own_vectors_mean",
    "own_scanned_percent", "extra_scan_factor",
)


def read_scans(groups: dict[tuple[str, ...], list[dict[str, object]]]) -> dict[tuple[str, ...], list[dict[str, object]]]:
    scans: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for config, runs in groups.items():
        for run in runs:
            path = Path(run["source"]).parent / f"batches_run_{run['run']}.csv"
            if not path.is_file():
                continue  # --diagnostics is optional for standalone timing runs.
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                required = {*SCAN_COLUMNS, *CONFIG_COLUMNS}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{path}: missing scan columns {sorted(missing)}; capture new diagnostics")
                batches = set()
                query_count = int(config[2])
                batch_count = (query_count + 31) // 32
                for line, row in enumerate(reader, start=2):
                    context = f"{path}:{line}"
                    if None in row or any(row.get(name) is None or not row[name].strip() for name in required):
                        raise ValueError(f"{context}: incomplete or unexpected scan row")
                    if tuple(row[name].strip() for name in CONFIG_COLUMNS) != config:
                        raise ValueError(f"{context}: scan configuration differs from timing CSV")
                    values = {name: float(row[name]) for name in SCAN_COLUMNS}
                    if any(not math.isfinite(value) or value < 0 for value in values.values()):
                        raise ValueError(f"{context}: negative or non-finite scan data")
                    for name in SCAN_COLUMNS[:7]:
                        if not values[name].is_integer():
                            raise ValueError(f"{context}: {name} must be an integer")
                        values[name] = int(values[name])
                    batch = values["batch"]
                    if batch in batches or batch >= batch_count or values["valid_queries"] != min(32, query_count - batch * 32):
                        raise ValueError(f"{context}: invalid batch index/count")
                    batches.add(batch)
                    vectors, database = values["union_vectors"], values["database_vectors"]
                    own = values["own_vectors_mean"]
                    if database <= 0 or vectors > database or own > vectors:
                        raise ValueError(f"{context}: invalid scanned vector count")
                    for name, expected in (
                        ("scanned_fraction", vectors / database),
                        ("scanned_percent", vectors / database * 100),
                        ("own_scanned_percent", own / database * 100),
                        ("extra_scan_factor", vectors / own if own else 0),
                    ):
                        if not math.isclose(values[name], expected, rel_tol=1e-8, abs_tol=1e-8):
                            raise ValueError(f"{context}: {name} disagrees with vector counts")
                    scans[config].append({"run": run["run"], **values})
                if batches != set(range(batch_count)):
                    raise ValueError(f"{path}: expected {batch_count} complete batches")
    return dict(scans)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def summarize(groups: dict[tuple[str, ...], list[dict[str, object]]], output: Path,
              expected_runs: int | None = None) -> str:
    scans = read_scans(groups)
    output.mkdir(parents=True, exist_ok=True)
    report = [
        "# Search timing medians", "",
        "Durations are milliseconds. Detail rows are included in their parent stage. "
        "Medians are computed per column and need not add exactly.", "",
        "TT coarse/fine intervals include enqueue, execution and the blocking Finish call. "
        "Query grouping uses device coarse results. Reordering includes host copying and tiling. "
        "Fine setup/misc is the remaining host preparation, including allocations, kernel/CB setup "
        "and existing bookkeeping. Index construction/upload and warmup are excluded. "
        "Recall checks, diagnostic files, timing reports and CSV writes occur after the timer stops.", "",
    ]
    if scans:
        with (output / "batch_scans.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([*CONFIG_COLUMNS, "run", *SCAN_COLUMNS])
            for config, batches in scans.items():
                for batch in sorted(batches, key=lambda row: (int(row["run"]), int(row["batch"]))):
                    writer.writerow([*config, batch["run"], *(batch[name] for name in SCAN_COLUMNS)])
        with (output / "scan_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([*CONFIG_COLUMNS, "recorded_runs", "batch_samples", "metric", "mean", "median", "p95", "max"])
            for config, batches in scans.items():
                recorded_runs = len({batch["run"] for batch in batches})
                for name in ("union_vectors", "union_pages", "scanned_percent", "own_scanned_percent", "extra_scan_factor"):
                    values = [float(batch[name]) for batch in batches]
                    writer.writerow([*config, recorded_runs, len(batches), name, statistics.mean(values),
                                     statistics.median(values), percentile(values, 0.95), max(values)])
    with (output / "timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*CONFIG_COLUMNS, "runs", "metric", "parent", "median_ms", "min_ms", "max_ms"])
        for config, runs in groups.items():
            qps = [float(run["qps"]) for run in runs]
            recalls = [float(run["recall"]) for run in runs]
            report.extend([
                f"## N_list={config[0]}, N_probe={config[1]}", "",
                f"{config[2]} queries/run; {len(runs)} measured runs; grouping={config[4]}, "
                f"partitions={config[5]}, reader={config[6]}, bank schedule={config[7]}, "
                f"worker input={config[13]} page pairs, leader prefetch={config[12]} page pairs.", "",
                f"Median QPS: **{statistics.median(qps):.2f}** "
                f"({min(qps):.2f}–{max(qps):.2f}); median Recall@{config[3]}: "
                f"**{statistics.median(recalls):.6f}** ({min(recalls):.6f}–{max(recalls):.6f}).", "",
                "| Stage / included detail | Median (ms) | Min (ms) | Max (ms) |",
                "|---|---:|---:|---:|",
            ])
            if expected_runs is not None and len(runs) != expected_runs:
                report.insert(len(report) - 2,
                    f"**Incomplete configuration: {len(runs)}/{expected_runs} completed runs. "
                    "Only completed measurements below are included.**\n")
            for name, label, parent in TIMINGS:
                values = [float(run[name]) / 1000 for run in runs]
                median, minimum, maximum = statistics.median(values), min(values), max(values)
                writer.writerow([*config, len(runs), name, parent, f"{median:.9f}", f"{minimum:.9f}", f"{maximum:.9f}"])
                label = f"↳ {label}" if parent else f"**{label}**"
                report.append(f"| {label} | {median:.3f} | {minimum:.3f} | {maximum:.3f} |")
            report.append("")
            batches = scans.get(config, [])
            if batches:
                recorded_runs = len({batch["run"] for batch in batches})
                report.extend([
                    f"Dataset coverage across {len(batches)} batches from {recorded_runs}/{len(runs)} measured runs. "
                    "Each valid query scores the entire batch union. Padding is excluded from vector counts; "
                    "the own-list value uses each query's device-selected lists, not a Faiss rerun.", "",
                    "| Scan metric | Mean / batch | Median | p95 | Max |",
                    "|---|---:|---:|---:|---:|",
                ])
                for name, label in (
                    ("union_vectors", "Union vectors"), ("union_pages", "Union pages"),
                    ("scanned_percent", "Dataset scanned (%)"),
                    ("own_scanned_percent", "Own-list dataset scanned (%)"),
                    ("extra_scan_factor", "Union / own-list scan factor"),
                ):
                    values = [float(batch[name]) for batch in batches]
                    report.append(f"| {label} | {statistics.mean(values):.3f} | {statistics.median(values):.3f} | "
                                  f"{percentile(values, 0.95):.3f} | {max(values):.3f} |")
                queries = sum(int(batch["valid_queries"]) for batch in batches)
                tt_mean = sum(float(batch["scanned_percent"]) * int(batch["valid_queries"]) for batch in batches) / queries
                own_mean = sum(float(batch["own_scanned_percent"]) * int(batch["valid_queries"]) for batch in batches) / queries
                report.extend(["", f"Mean dataset coverage weighted by valid queries: TT union **{tt_mean:.3f}%**; "
                               f"own selected lists **{own_mean:.3f}%**. Exact batch counts are in `batch_scans.csv`.", ""])
    text = "\n".join(report)
    (output / "timing_report.md").write_text(text + "\n", encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="timing CSV or run directory")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="summarize a completed prefix of runs after a benchmark failure; requires --expected-runs")
    args = parser.parse_args()
    if args.expected_runs is not None and args.expected_runs <= 0:
        parser.error("--expected-runs must be positive")
    if args.allow_incomplete and args.expected_runs is None:
        parser.error("--allow-incomplete requires --expected-runs")
    if args.input.is_file():
        paths = [args.input]
        default_output = args.input.parent / "analysis"
    elif args.input.is_dir():
        paths = sorted(set(args.input.rglob("timings.csv")) | set(args.input.rglob("*_timings.csv")))
        default_output = args.input / "analysis"
    else:
        parser.error(f"input does not exist: {args.input}")
    if not paths:
        parser.error("no timings.csv or *_timings.csv found; pass a CSV explicitly or capture new runs")
    try:
        groups = read_runs(paths, allow_empty=args.allow_incomplete)
        if not groups:
            raise ValueError("no completed measured runs are available")
        if args.expected_runs is not None:
            for config, runs in groups.items():
                recorded = sorted(int(run["run"]) for run in runs)
                if recorded != list(range(1, args.expected_runs + 1)):
                    if not args.allow_incomplete or recorded != list(range(1, len(runs) + 1)) or len(runs) > args.expected_runs:
                        raise ValueError(f"{config[0]}/{config[1]}: expected measured runs 1..{args.expected_runs}")
                    print(f"WARNING: {config[0]}/{config[1]}: only {len(runs)}/{args.expected_runs} completed runs", file=sys.stderr)
        output = args.output_directory or default_output
        report = summarize(groups, output, args.expected_runs)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(report)
    print(f"Timing summary: {output / 'timing_summary.csv'}")
    print(f"Timing report: {output / 'timing_report.md'}")
    if (output / "batch_scans.csv").is_file():
        print(f"Per-batch vector counts and dataset percentages: {output / 'batch_scans.csv'}")
        print(f"Scan summary: {output / 'scan_summary.csv'}")


if __name__ == "__main__":
    main()
