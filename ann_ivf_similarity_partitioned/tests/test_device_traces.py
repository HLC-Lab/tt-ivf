"""Check trace selection, page pairing and numeric units without hardware."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_device_traces as analysis


class DeviceTraceTests(unittest.TestCase):
    def event(self, zone, start, end, risc="NCRISC", x=2, y=2):
        return analysis.Event(0, x, y, risc, zone, start, end, True)

    def test_warmup_is_excluded_and_nested_scopes_are_not_added(self):
        events = [self.event("NCRISC-KERNEL", 0, 100),
                  self.event("Worker read wait", 10, 50),
                  self.event("NCRISC-KERNEL", 1000, 2000),
                  self.event("Worker issue DRAM pair", 1100, 1200),
                  self.event("Worker read wait", 1800, 1900)]
        custom, kernels = analysis.select_fine(events)
        self.assertEqual([e.start for e in custom], [1100, 1800])
        self.assertEqual(kernels[0].start, 1000)

    def test_page_pairing_distinguishes_residual_wait_and_read_lifetime(self):
        events = [self.event("Worker issue DRAM pair", 0, 100),
                  self.event("Worker read wait", 9000, 10000)]
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            for name in ("Worker input wait", "Worker matmul", "Worker score pack",
                         "Worker score and ID wait", "Worker local topk"):
                events.append(self.event(name, 11000, 12000, risc))
        rows = analysis.page_summary("test", events, {(0, 2, 2): 1})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reader_residual_wait_us"], 1)
        self.assertEqual(rows[0]["issue_to_retirement_us"], 10)
        with self.assertRaisesRegex(ValueError, "incomplete per-page"):
            analysis.page_summary("test", events[:-1], {(0, 2, 2): 1})

    def test_physical_rows_include_harvested_coordinate_gaps(self):
        events = [self.event("TRISC-KERNEL", 0, 100, "TRISC_1", x, y)
                  for y in (1, 2, 4) for x in (1, 3)]
        mapping = analysis.partition_map(events, 3, "rows")
        self.assertEqual(mapping[(0, 3, 4)], 2)
        self.assertEqual(mapping[(0, 1, 2)], 1)

    def relay_events(self):
        events = []
        for x in (2, 3):
            for page in range(2):
                start = page * 20000
                events.append(self.event("Worker relay receive wait", start + 1000, start + 6000, x=x))
                for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
                    for name in ("Worker input wait", "Worker matmul", "Worker score pack",
                                 "Worker score and ID wait", "Worker local topk"):
                        events.append(self.event(name, start + 7000, start + 8000, risc, x=x))
        # The leader samples worker 0 only, not every worker's stream.
        for page in range(2):
            start = page * 20000
            events += [self.event("Leader issue DRAM pair", start, start + 100, x=1),
                       self.event("Leader read wait", start + 3000, start + 4000, x=1),
                       self.event("Leader forward pair", start + 4000, start + 5000, x=1)]
        return events

    def test_relay_pages_keep_receive_waits_for_every_worker(self):
        events = self.relay_events()
        mapping = {(0, x, 2): 1 for x in (1, 2, 3)}
        rows = analysis.page_summary("relay", events, mapping)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["core_x"] for row in rows}, {2, 3})
        self.assertTrue(all(row["reader_wait_us"] == 5 for row in rows))
        self.assertTrue(all(row["reader_residual_wait_us"] is None for row in rows))
        self.assertTrue(all(row["reader_issue_us"] is None for row in rows))
        self.assertTrue(all(row["transport"] == "relay" for row in rows))
        leaders = analysis.leader_page_summary("relay", events, mapping)
        self.assertEqual(len(leaders), 2)
        self.assertTrue(all(row["target_worker"] == 0 and row["core_x"] == 1 for row in leaders))
        self.assertEqual(leaders[0]["leader_residual_wait_us"], 1)
        self.assertEqual(leaders[0]["leader_forward_us"], 1)
        self.assertEqual(leaders[0]["issue_to_forward_completion_us"], 5)

    def test_mixed_capture_csv_has_the_same_columns_without_invented_reads(self):
        relay = self.relay_events()
        direct = [event for event in relay if event.x == 2 and event.risc != "NCRISC"]
        for page in range(2):
            start = page * 20000
            direct += [self.event("Worker issue DRAM pair", start, start + 100),
                       self.event("Worker read wait", start + 3000, start + 4000)]
        rows = analysis.page_summary("direct", direct, {(0, 2, 2): 1})
        rows += analysis.page_summary("relay", relay, {(0, x, 2): 1 for x in (1, 2, 3)})
        self.assertEqual(set(rows[0]), set(rows[-1]))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pages.csv"
            analysis.write_csv(path, rows)
            with path.open() as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(len(written), 6)
            self.assertEqual(written[-1]["reader_issue_us"], "")

    def test_relay_rejects_missing_or_reversed_leader_pairs(self):
        events = self.relay_events()
        mapping = {(0, x, 2): 1 for x in (1, 2, 3)}
        with self.assertRaisesRegex(ValueError, "incomplete relay"):
            analysis.leader_page_summary("relay", events[:-1], mapping)
        events[-1] = self.event("Leader forward pair", 23000, 23500, x=1)
        with self.assertRaisesRegex(ValueError, "precedes paired read"):
            analysis.leader_page_summary("relay", events, mapping)
        with self.assertRaisesRegex(ValueError, "mixed direct and relay"):
            analysis.page_summary("relay", self.relay_events() +
                                  [self.event("Worker issue DRAM pair", 0, 100)], mapping)

    def test_metal_scope_csv_frequency_pairing_and_unmatched_end(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "profile_log_device.csv"
            with path.open("w", newline="") as handle:
                handle.write("ARCH: test, CHIP_FREQ[MHz]: 500\n")
                writer = csv.writer(handle)
                writer.writerow(["PCIe slot", "core_x", "core_y", "RISC processor type",
                                 "run host ID", "time[cycles since reset]", "zone name", "type"])
                writer.writerows([(0, 1, 1, "NCRISC", 0, 1000, "NCRISC-KERNEL", "begin"),
                                  (0, 1, 1, "NCRISC", 0, 1200, "Worker read wait", "begin"),
                                  (0, 1, 1, "NCRISC", 0, 2700, "Worker read wait", "end"),
                                  (0, 1, 1, "NCRISC", 0, 3000, "NCRISC-KERNEL", "end"),
                                  (0, 1, 1, "NCRISC", 1, 4000, "missing", "end")])
            events, unmatched = analysis.read_metal_csv(path)
            self.assertEqual(unmatched, 1)
            self.assertEqual(events[0].us, 3)
            self.assertFalse(events[0].calibrated)

    def test_host_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "cpu.csv"
            path.write_text("name,ns_since_start,exec_time_ns\nfoo,0,1000\n")
            with self.assertRaisesRegex(ValueError, "host-only"):
                analysis.read_export(path)

    def test_tail_statistics_use_linear_percentile(self):
        summary = analysis.stats([1, 2, 3, 100])
        self.assertAlmostEqual(summary["p95_us"], 85.45)
        self.assertEqual(summary["over_10_us"], 1)

    def test_partition_view_keeps_leader_workers_and_shared_time_origin(self):
        events = [self.event("Leader query read", 100, 200, x=1),
                  self.event("Partition gather", 110, 900, "BRISC", x=1),
                  self.event("Partition gather partial", 700, 750, "BRISC", x=1),
                  self.event("Worker result wait", 90, 800, "BRISC", x=2),
                  self.event("Worker matmul", 500, 600, "TRISC_1", x=2)]
        data = dict(custom=events, mapping={(0, 1, 2): 1, (0, 2, 2): 1, (0, 3, 2): 1})
        rows, cores = analysis.partition_details("trace", data, 1)
        self.assertEqual([c["role"] for c in cores],
                         ["Leader / aggregator", "Worker 0 (slave)", "Worker 1 (slave)"])
        leader = next(r for r in rows if r["zone"] == "Leader query read")
        compute = next(r for r in rows if r["zone"] == "Worker matmul")
        self.assertAlmostEqual(leader["start_us"], .01)
        self.assertAlmostEqual(compute["start_us"], .41)
        self.assertEqual(len(rows), 5)  # Keep nested scopes independently.

    def test_partition_view_rejects_uninstrumented_partition(self):
        data = dict(custom=[self.event("Worker read wait", 100, 200)],
                    mapping={(0, 2, 2): 1, (0, 1, 1): 0})
        with self.assertRaisesRegex(ValueError, "no detailed scopes"):
            analysis.partition_details("trace", data, 0)


if __name__ == "__main__":
    unittest.main()
