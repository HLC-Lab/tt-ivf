#!/usr/bin/env python3
"""Summarize run_gather_comparison.sh output: per configuration the medians of
the measured runs (warmup excluded) of every stage, QPS and recall, plus the
scan work (page x tile pairs over all passes) and the validation. Rows are
ordered by nprobe, then gather, fast_single, fast."""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

STAGES = ["h2d", "cpu_coarse_config", "tt_coarse_search", "cpu_fine_prep", "tt_fine_search", "cpu_output"]
RUN = re.compile(r"^\[Run (\d+)\] QPS=([0-9.]+), Recall@\d+=([0-9.]+)")
PAIRS = re.compile(r"(?:page x batch pairs|page x tile pairs)=(\d+)")


def measure(directory: Path) -> dict[str, object]:
    per_run: dict[str, dict[str, float]] = {}
    with (directory / "mem_log.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["layer"] == "chrono_stage" and row["context"].startswith("run_"):
                per_run.setdefault(row["context"], {})[row["operation"]] = float(row["duration_us"]) / 1000.0
    for stages in per_run.values():
        stages["stage_sum"] = sum(stages[s] for s in STAGES)
    text = (directory / "console.log").read_text(errors="replace")
    runs = [m for m in map(RUN.match, text.splitlines()) if m]
    if len(runs) != len(per_run) or not runs:
        raise ValueError(f"{directory}: measured runs missing or inconsistent")
    pairs = PAIRS.findall(text)
    validation = directory / "validation.log"
    return {
        **{c: statistics.median(r[c] for r in per_run.values()) for c in STAGES + ["stage_sum"]},
        "qps": statistics.median(float(m.group(2)) for m in runs),
        "recall": statistics.median(float(m.group(3)) for m in runs),
        "pairs": int(pairs[-1]) if pairs else None,
        "validation": validation.read_text().strip().splitlines()[0] if validation.exists() else "—",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    order = {"gather": 0, "fast_single": 1, "fast_rank32_rebuild": 2, "fast_rank32": 3, "fast_rebuild": 4, "fast": 5}

    def key(name: str) -> tuple[int, int, str]:
        implementation, _, nprobe = name.rpartition("_nprobe_")
        return (int(nprobe) if nprobe.isdigit() else 0, order.get(implementation, 9), name)

    rows = sorted(((p.name, measure(p)) for p in args.directory.iterdir() if (p / "mem_log.csv").exists()),
                  key=lambda row: key(row[0]))
    print("| Configuration | Recall@10 | QPS | Stage sum | Fine device | CPU fine prep | CPU coarse config "
          "| TT coarse | H2D | CPU output | Scan pairs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in rows:
        print(f"| {name} | {r['recall']:.6f} | {r['qps']:,.0f} | {r['stage_sum']:.3f} | {r['tt_fine_search']:.3f} "
              f"| {r['cpu_fine_prep']:.3f} | {r['cpu_coarse_config']:.3f} | {r['tt_coarse_search']:.3f} "
              f"| {r['h2d']:.3f} | {r['cpu_output']:.3f} | {r['pairs']:,} |")
    print()
    for name, r in rows:
        if r["validation"] != "—":
            print(f"- {name}: {r['validation']}")


if __name__ == "__main__":
    main()
