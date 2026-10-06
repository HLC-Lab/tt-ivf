#!/usr/bin/env python3
"""Validate that incomplete or mixed hardware results cannot pass analysis."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_grouping as analysis
import compare_results as comparison


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dump(self, name: str, values: list[tuple[int, int, int, float]]) -> Path:
        path = self.root / name
        write(path, [dict(query=q, rank=r, index=i, score=s) for q, r, i, s in values])
        return path

    def test_complete_results_and_explicit_ties(self) -> None:
        left = self.dump("left.csv", [(0, 0, 2, .5), (0, 1, 3, .5), (1, 0, 7, .25), (1, 1, -1, -1000)])
        right = self.dump("right.csv", [(0, 0, 3, .5), (0, 1, 2, .5), (1, 0, 7, .25), (1, 1, -1, -1000)])
        self.assertEqual(comparison.compare(left, right, 0, True, 2, 2), (4, 2))
        with self.assertRaises(ValueError):
            comparison.compare(left, right, 0, False, 2, 2)

    def test_truncated_dumps_cannot_agree(self) -> None:
        path = self.dump("truncated.csv", [(0, 0, 2, .5), (0, 1, 3, .4)])
        with self.assertRaisesRegex(ValueError, "query count"):
            comparison.compare(path, path, 0, True, 2, 2)
        with self.assertRaisesRegex(ValueError, "ranks"):
            comparison.compare(path, path, 0, True, 1, 3)

    def test_invalid_result_layout(self) -> None:
        for values in (
            [(0, 0, 2, .5), (0, 2, 3, .4)],
            [(0, 0, 2, .4), (0, 1, 3, .5)],
            [(0, 0, -1, -1000), (0, 1, 3, .5)],
            [(0, 0, 3, .5), (0, 1, 3, .4)],
        ):
            with self.subTest(values=values):
                path = self.dump("bad.csv", values)
                with self.assertRaises(ValueError):
                    comparison.read(path)

    def diagnostics(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        config = dict(zip(analysis.CONFIG_FIELDS,
            (512, 8, 33, 2, "weighted-union", 8, "relay", "staggered", "leader", 0,
             "rows", "auto", 8, 2, 0, 256)))
        batches = [dict(config, batch=b, valid_queries=q, union_lists=2, union_pages=4,
                        union_vectors=100, extra_scan_factor=2, active_workers=2,
                        worker_pages_max=3, worker_pages_mean=4/7, imbalance=5.25)
                   for b, q in ((0, 32), (1, 1))]
        timing = dict(config, run=1, qps=330000, recall=.9, latency_us=100)
        return batches, timing

    def test_complete_summary(self) -> None:
        batches, timing = self.diagnostics()
        write(self.root / "batches_run_1.csv", batches)
        write(self.root / "results.csv", [timing])
        summary = analysis.summarize(self.root)[0]
        self.assertEqual(summary["batch_samples"], 2)
        self.assertEqual(summary["total_pages_per_repetition_median"], 8)
        self.assertEqual(summary["qps_median"], 330000)

    def test_incomplete_or_padded_batches_rejected(self) -> None:
        batches, _ = self.diagnostics()
        write(self.root / "batches_run_1.csv", batches[:1])
        with self.assertRaisesRegex(ValueError, "batch IDs"):
            analysis.summarize(self.root)
        batches[1]["valid_queries"] = 32
        write(self.root / "batches_run_1.csv", batches)
        with self.assertRaisesRegex(ValueError, "valid_queries"):
            analysis.summarize(self.root)

    def test_reused_or_mismatched_timings_rejected(self) -> None:
        batches, timing = self.diagnostics()
        write(self.root / "batches_run_1.csv", batches)
        for timings in ([timing, timing], [dict(timing, run=2)],
                        [dict(timing, grouping="none")], [dict(timing, qps=1)]):
            with self.subTest(timings=timings):
                write(self.root / "results.csv", timings)
                with self.assertRaises(ValueError):
                    analysis.summarize(self.root)


if __name__ == "__main__":
    unittest.main()
