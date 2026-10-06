#!/usr/bin/env python3
"""Summarize untimed batch diagnostics and measured QPS/recall from the ablation."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

CONFIG_FIELDS = ("nlist", "nprobe", "queries", "k", "grouping", "partitions", "reader", "bank_schedule", "aggregation", "mask",
                 "layout", "query_broadcast", "leader_prefetch", "worker_input", "grouping_window", "grouping_lookahead")
METRICS = ("union_lists", "union_pages", "union_vectors", "extra_scan_factor", "active_workers",
           "worker_pages_max", "worker_pages_mean", "imbalance")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def numeric(row: dict[str, str], column: str, path: Path) -> float:
    value = float(row[column])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{path}: invalid {column}={row[column]}")
    return value


def read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data")
    return rows


def summarize(root: Path) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    repetitions: dict[tuple[str, ...], list[float]] = defaultdict(list)
    timings: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    runs: dict[tuple[Path, int], tuple[str, ...]] = {}
    paths = sorted(root.rglob("batches_run_*.csv"))
    if not paths:
        raise ValueError(f"No batches_run_*.csv below {root}; run with --diagnostics DIRECTORY")
    for path in paths:
        rows = read_rows(path, CONFIG_FIELDS + METRICS + ("batch", "valid_queries"))
        keys = {tuple(row[field] for field in CONFIG_FIELDS) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"{path}: mixed configurations")
        key = next(iter(keys))
        queries = int(rows[0]["queries"])
        if not 1 <= queries <= 10000 or not 1 <= int(rows[0]["k"]) <= 32:
            raise ValueError(f"{path}: invalid query count or k")
        batch_ids = [int(row["batch"]) for row in rows]
        if sorted(batch_ids) != list(range((queries + 31) // 32)):
            raise ValueError(f"{path}: missing/duplicate batch IDs")
        for row in rows:
            if int(row["valid_queries"]) != min(32, queries - int(row["batch"]) * 32):
                raise ValueError(f"{path}: invalid valid_queries")
            for metric in METRICS:
                numeric(row, metric, path)
        groups[key].extend(rows)
        repetitions[key].append(sum(float(row["union_pages"]) for row in rows))
        run = int(path.stem.removeprefix("batches_run_"))
        if run <= 0:
            raise ValueError(f"{path}: invalid repetition number")
        runs[(path.parent.resolve(), run)] = key
    for path in sorted(root.rglob("results.csv")):
        seen: set[int] = set()
        for row in read_rows(path, CONFIG_FIELDS + ("run", "qps", "recall", "latency_us")):
            key = tuple(row[field] for field in CONFIG_FIELDS)
            run = int(row["run"])
            if run in seen or runs.get((path.parent.resolve(), run)) != key:
                raise ValueError(f"{path}: duplicate repetition or timing without matching batch diagnostics (run {run})")
            seen.add(run)
            for field in ("qps", "recall", "latency_us"):
                numeric(row, field, path)
            if not 0 <= float(row["recall"]) <= 1 or float(row["latency_us"]) <= 0:
                raise ValueError(f"{path}: recall or latency outside its valid range")
            expected_qps = int(row["queries"]) * 1e6 / float(row["latency_us"])
            if not math.isclose(float(row["qps"]), expected_qps, rel_tol=1e-6):
                raise ValueError(f"{path}: QPS does not match query count and latency")
            timings[key].append(row)
        expected = {run for directory, run in runs if directory == path.parent.resolve()}
        if seen != expected:
            raise ValueError(f"{path}: timing rows do not cover every diagnosed repetition")
    summaries = []
    for key, rows in sorted(groups.items()):
        summary: dict[str, object] = dict(zip(CONFIG_FIELDS, key))
        summary.update(repetitions=len(repetitions[key]), batch_samples=len(rows),
                       total_pages_per_repetition_median=statistics.median(repetitions[key]))
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary.update({f"{metric}_mean": statistics.mean(values), f"{metric}_median": statistics.median(values),
                            f"{metric}_p95": percentile(values, 0.95), f"{metric}_max": max(values)})
        for metric in ("qps", "recall", "latency_us"):
            values = [float(row[metric]) for row in timings.get(key, [])]
            summary[f"{metric}_median"] = statistics.median(values) if values else ""
        summaries.append(summary)
    return summaries


def plots(summaries: list[dict[str, object]], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError("Install matplotlib or use --no-plots") from error
    configs = sorted({tuple(int(s[field]) for field in ("nlist", "nprobe", "queries", "k")) for s in summaries})
    for nlist, nprobe, queries, k in configs:
        rows = [s for s in summaries if tuple(int(s[field]) for field in ("nlist", "nprobe", "queries", "k")) == (nlist, nprobe, queries, k)]
        labels = [f"{s['grouping']}\nP{s['partitions']} {s['layout']}\n{s['reader']}/{s['bank_schedule']}" for s in rows]
        figure, axes = plt.subplots(1, 2, figsize=(max(12, len(rows) * 1.7), 5), layout="constrained")
        axes[0].bar(range(len(rows)), [float(s["total_pages_per_repetition_median"]) for s in rows], color="#7559E8")
        axes[0].set_ylabel("Candidate pages per repetition (median)")
        axes[1].bar(range(len(rows)), [float(s["qps_median"]) if s["qps_median"] != "" else math.nan for s in rows], color="#59A14F")
        axes[1].set_ylabel("End-to-end QPS (median)")
        for axis in axes:
            axis.set_xticks(range(len(rows)), labels, rotation=40, ha="right")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        figure.savefig(output / f"nlist{nlist}_nprobe{nprobe}_queries{queries}_k{k}_ablation.png", dpi=200)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_directory or args.root / "analysis"
    try:
        summaries = summarize(args.root)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "grouping_summary.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        report = ["# Similarity grouping and partition ablation", "",
                  "Warmup files are excluded. Diagnostics are written after the measured pipeline.", "",
                  "| N_list/N_probe | Queries / k | Grouping | Partitions / layout | Reader / bank schedule | Pages/repetition | QPS | Recall | p95 imbalance |",
                  "|---|---|---|---|---|---:|---:|---:|---:|"]
        for summary in summaries:
            qps = f"{float(summary['qps_median']):.2f}" if summary["qps_median"] != "" else "unavailable"
            recall = f"{float(summary['recall_median']):.6f}" if summary["recall_median"] != "" else "unavailable"
            report.append(f"| {summary['nlist']}/{summary['nprobe']} | {summary['queries']} / {summary['k']} | {summary['grouping']} | "
                          f"{summary['partitions']} / {summary['layout']} | {summary['reader']} / {summary['bank_schedule']} | "
                          f"{float(summary['total_pages_per_repetition_median']):.0f} | {qps} | {recall} | {float(summary['imbalance_p95']):.3f} |")
        report.extend(["", "Grouping changes batch candidate unions, so compare recall as well as QPS.",
                       "Worker imbalance is computed over all workers inside each owning partition, including idle workers.",
                       "Repeated deterministic batches are repeated observations, not independent query samples."])
        (output / "report.md").write_text("\n".join(report) + "\n")
        if not args.no_plots:
            plots(summaries, output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(f"Summarized {len(summaries)} configurations: {output}")


if __name__ == "__main__":
    main()
