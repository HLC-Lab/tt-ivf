#!/usr/bin/env python3
"""Check timing conservation, measured-run selection and dataset percentages."""
from __future__ import annotations

import csv
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import summarize_timings as analysis


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "timings.csv"
        self.config = dict(zip(analysis.CONFIG_COLUMNS,
            (512, 16, 33, 10, "weighted-union", 8, "direct", "fifo", "leader", 0,
             "rows", "auto", 8, 2, 0, 256)))
        detail_values = (10, 12, 8, 10, 20, 20, 10, 10, 10, 5, 20, 10, 10, 5, 30, 10, 20)
        detail_names = [name for name, _, parent in analysis.TIMINGS if parent]
        self.times = dict(zip(detail_names, detail_values))
        self.times.update(h2d_us=40, cpu_coarse_config_us=50, tt_coarse_search_us=60,
                          cpu_fine_prep_us=100, tt_fine_search_us=200, cpu_output_us=30,
                          stage_sum_us=480, pipeline_us=480)
        self.row = dict(self.config, run=1, qps=33 * 1e6 / 480, recall=.9, **self.times)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def groups(self) -> dict:
        write(self.path, [self.row, dict(self.row, run=2)])
        return analysis.read_runs([self.path])

    def batches(self, run: int) -> list[dict[str, object]]:
        # Different final-batch coverage checks both per-batch and per-query means.
        rows = []
        for batch, queries, vectors in ((0, 32, 100), (1, 1, 200)):
            rows.append(dict(self.config, batch=batch, valid_queries=queries, partition=batch,
                             union_lists=16, union_pages=10, union_vectors=vectors,
                             database_vectors=1000, scanned_fraction=vectors / 1000,
                             scanned_percent=vectors / 10, own_vectors_mean=50,
                             own_scanned_percent=5, extra_scan_factor=vectors / 50))
        write(self.root / f"batches_run_{run}.csv", rows)
        return rows

    def test_cpp_and_python_column_names_agree(self) -> None:
        names = re.findall(r'TimingField\{"([a-z0-9_]+_us)"', (ROOT / "pipeline_timings.hpp").read_text())
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {name for name, _, _ in analysis.TIMINGS})

    def test_summary_and_partial_batch_coverage(self) -> None:
        groups = self.groups()
        self.batches(1)
        self.batches(2)
        # Warmup diagnostics must never contribute to the measured summaries.
        self.batches(0)
        (self.root / "batches_run_0.csv").rename(self.root / "batches_warmup.csv")
        report = analysis.summarize(groups, self.root / "analysis")
        self.assertIn("Dataset scanned (%) | 15.000 | 15.000 | 20.000 | 20.000", report)
        self.assertIn("TT union **10.303%**", report)
        self.assertIn("own selected lists **5.000%**", report)
        with (self.root / "analysis" / "batch_scans.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["valid_queries"] for row in rows}, {"32", "1"})
        self.assertEqual({row["run"] for row in rows}, {"1", "2"})
        result = subprocess.run([sys.executable, "-B", str(ROOT / "summarize_timings.py"),
                                 str(self.root), "--expected-runs", "2"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Per-batch vector counts", result.stdout)

    def test_wrong_stage_total_or_qps_rejected(self) -> None:
        for change in ({"pipeline_us": 900}, {"query_grouping_us": 500}, {"qps": 1}, {"recall": float("nan")}):
            with self.subTest(change=change):
                write(self.path, [dict(self.row, **change)])
                with self.assertRaises(ValueError):
                    analysis.read_runs([self.path])

    def test_duplicates_or_missing_runs_rejected(self) -> None:
        write(self.path, [self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.read_runs([self.path])
        write(self.path, [self.row])
        result = subprocess.run([sys.executable, "-B", str(ROOT / "summarize_timings.py"),
                                 str(self.path), "--expected-runs", "2"], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected measured runs", result.stderr)

    def test_explicit_partial_report_after_failure(self) -> None:
        write(self.path, [self.row])
        # A later process may fail after creating only its CSV header.
        empty = self.root / "later_timings.csv"
        with empty.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(self.row)).writeheader()
        result = subprocess.run([sys.executable, "-B", str(ROOT / "summarize_timings.py"),
                                 str(self.root), "--expected-runs", "5", "--allow-incomplete"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("only 1/5 completed runs", result.stderr)
        self.assertIn("Incomplete configuration: 1/5", result.stdout)
        # Partial mode must still reject missing measurements in the middle.
        write(self.path, [self.row, dict(self.row, run=3)])
        result = subprocess.run([sys.executable, "-B", str(ROOT / "summarize_timings.py"),
                                 str(self.root), "--expected-runs", "5", "--allow-incomplete"], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected measured runs", result.stderr)

    def test_incomplete_or_wrong_scan_percentages_rejected(self) -> None:
        groups = self.groups()
        rows = self.batches(1)
        for broken in (rows[:1], [dict(rows[0], scanned_percent=99), rows[1]],
                       [rows[0], dict(rows[1], valid_queries=32)]):
            with self.subTest(broken=broken):
                write(self.root / "batches_run_1.csv", broken)
                with self.assertRaises(ValueError):
                    analysis.read_scans(groups)


if __name__ == "__main__":
    unittest.main()
