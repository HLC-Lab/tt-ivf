#!/usr/bin/env python3

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_fine_cluster_bubbles import (
    filter_blocks,
    read_bubble_events,
    select_core_cluster,
)


class FineClusterBubblePlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.profile_log = Path(self.temporary_directory.name) / "profile_log_device.csv"

    @staticmethod
    def marker(risc, cycle, zone, marker_type):
        return [0, 1, 2, risc, 0, cycle, 0, 0, "", "", zone, marker_type, 1, "kernel.cpp", ""]

    def write_profile(self):
        rows = []
        for invocation_base in (100, 1000):
            rows.extend(
                [
                    self.marker("NCRISC", invocation_base, "FineBubble.Reader.Query", "ZONE_START"),
                    self.marker("NCRISC", invocation_base + 10, "FineBubble.Reader.Query", "ZONE_END"),
                    self.marker("TRISC_0", invocation_base - 2, "FineBubble.Compute.QueryWait", "ZONE_START"),
                    self.marker("TRISC_0", invocation_base + 11, "FineBubble.Compute.QueryWait", "ZONE_END"),
                    self.marker("NCRISC", invocation_base + 15, "FineBubble.Reader.Cluster", "ZONE_START"),
                    self.marker("NCRISC", invocation_base + 210, "FineBubble.Reader.Cluster", "ZONE_END"),
                ]
            )
            for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
                rows.extend(
                    [
                        self.marker(risc, invocation_base + 18, "FineBubble.Compute.Cluster", "ZONE_START"),
                        self.marker(risc, invocation_base + 215, "FineBubble.Compute.Cluster", "ZONE_END"),
                    ]
                )
            for block in range(2):
                base = invocation_base + 20 + block * 100
                rows.extend(
                    [
                        self.marker("NCRISC", base + 3, "FineBubble.Reader.Fetch", "ZONE_START"),
                        self.marker("NCRISC", base + 25, "FineBubble.Reader.Fetch", "ZONE_END"),
                    ]
                )
                for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
                    rows.extend(
                        [
                            self.marker(risc, base + 28, "FineBubble.Compute.Work", "ZONE_START"),
                            self.marker(risc, base + 82, "FineBubble.Compute.Work", "ZONE_END"),
                        ]
                    )

        with self.profile_log.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["ARCH: wormhole_b0", " CHIP_FREQ[MHz]: 1000"])
            writer.writerow(
                [
                    "PCIe slot", " core_x", " core_y", " RISC processor type", " timer_id",
                    " time[cycles since reset]", " data", " run host ID", " trace id",
                    " trace id counter", " zone name", " type", " source line", " source file",
                    " meta data",
                ]
            )
            writer.writerows(rows)

    def test_selects_last_invocation_and_aligns_block_ordinals(self):
        self.write_profile()
        frequency_mhz, events, unmatched = read_bubble_events(self.profile_log)
        selected = select_core_cluster(events, (1, 2), "last", cluster_index=0)
        self.assertEqual(frequency_mhz, 1000)
        self.assertEqual(unmatched, 0)
        self.assertGreaterEqual(min(int(event["start_cycle"]) for event in selected), 998)
        for risc, zone in (
            ("NCRISC", "FineBubble.Reader.Fetch"),
            ("TRISC_0", "FineBubble.Compute.Work"),
            ("TRISC_1", "FineBubble.Compute.Work"),
            ("TRISC_2", "FineBubble.Compute.Work"),
        ):
            blocks = [
                event["block"]
                for event in selected
                if event["risc"] == risc and event["zone"] == zone
            ]
            self.assertEqual(blocks, [0, 1])
        one_block = filter_blocks(selected, 0, 1)
        self.assertFalse(any(event["block"] == 1 for event in one_block))


if __name__ == "__main__":
    unittest.main()
