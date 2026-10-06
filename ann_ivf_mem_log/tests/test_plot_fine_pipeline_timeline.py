#!/usr/bin/env python3

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_fine_pipeline_timeline import (
    PLOTTING_IMPORT_ERROR,
    add_phase_ordinals,
    assign_batches_from_scopes,
    choose_cores,
    detect_aggregator,
    plot_pipeline,
    read_scoped_events,
)


class FinePipelineTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.profile_log = Path(self.temporary_directory.name) / "profile_log_device.csv"

    @staticmethod
    def marker(core_x, core_y, risc, cycle, zone, marker_type):
        return [0, core_x, core_y, risc, 0, cycle, 0, 0, "", "", zone, marker_type, 1, "kernel.cpp", ""]

    def write_profile(self):
        rows = []
        # One warmup batch followed by a measured two-batch invocation. The
        # second measured worker batch starts before the first result is sent,
        # which is the overlap this plotter must preserve.
        for batch, base in enumerate((10, 100, 140)):
            rows.extend(
                [
                    self.marker(1, 2, "NCRISC", base, "Query", "ZONE_START"),
                    self.marker(1, 2, "NCRISC", base + 2, "Query", "ZONE_END"),
                    self.marker(1, 2, "NCRISC", base + 3, "Cluster", "ZONE_START"),
                    self.marker(1, 2, "NCRISC", base + 18, "Cluster", "ZONE_END"),
                ]
            )
            for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
                rows.extend(
                    [
                        self.marker(1, 2, risc, base + 1, "Query", "ZONE_START"),
                        self.marker(1, 2, risc, base + 2, "Query", "ZONE_END"),
                        self.marker(1, 2, risc, base + 4, "Cluster", "ZONE_START"),
                        self.marker(1, 2, risc, base + 9, "Cluster", "ZONE_END"),
                    ]
                )
                # The warmup intentionally has one task while the measured
                # invocation has two, matching real batch-union scheduling.
                if batch > 0:
                    rows.extend(
                        [
                            self.marker(1, 2, risc, base + 10, "Cluster", "ZONE_START"),
                            self.marker(1, 2, risc, base + 17, "Cluster", "ZONE_END"),
                        ]
                    )
            send_base = (35, 160, 205)[batch]
            rows.extend(
                [
                    self.marker(1, 2, "BRISC", send_base, "Send Res", "ZONE_START"),
                    self.marker(1, 2, "BRISC", send_base + 3, "Send Res", "ZONE_END"),
                ]
            )

        for base in (50, 180, 225):
            rows.extend(
                [
                    self.marker(1, 1, "NCRISC", base, "Partial Res", "ZONE_START"),
                    self.marker(1, 1, "NCRISC", base + 8, "Partial Res", "ZONE_END"),
                    self.marker(1, 1, "TRISC_0", base + 3, "Compute Res", "ZONE_START"),
                    self.marker(1, 1, "TRISC_0", base + 12, "Compute Res", "ZONE_END"),
                    self.marker(1, 1, "TRISC_1", base + 3, "Compute Res", "ZONE_START"),
                    self.marker(1, 1, "TRISC_1", base + 12, "Compute Res", "ZONE_END"),
                    self.marker(1, 1, "TRISC_2", base + 3, "Compute Res", "ZONE_START"),
                    self.marker(1, 1, "TRISC_2", base + 12, "Compute Res", "ZONE_END"),
                    self.marker(1, 1, "BRISC", base + 13, "Res", "ZONE_START"),
                    self.marker(1, 1, "BRISC", base + 16, "Res", "ZONE_END"),
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

    def test_scoped_only_batch_inference_excludes_warmup(self):
        self.write_profile()
        frequency_mhz, events, unmatched = read_scoped_events(self.profile_log)
        aggregator = detect_aggregator(events, None)
        cores = choose_cores(events, aggregator, set(), worker_count=1)
        selected, batches, warnings = assign_batches_from_scopes(events, cores, aggregator, last_batches=2)
        self.assertEqual(frequency_mhz, 1000)
        self.assertEqual(unmatched, 0)
        self.assertEqual(aggregator, ("0", 1, 1))
        self.assertEqual(cores, [("0", 1, 1), ("0", 1, 2)])
        self.assertEqual(batches, [0, 1])
        self.assertEqual(warnings, [])
        self.assertGreaterEqual(min(int(event["start_cycle"]) for event in selected), 100)
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            clusters = [
                event
                for event in selected
                if event["core_y"] == 2 and event["risc"] == risc and event["zone"] == "Cluster"
            ]
            self.assertEqual([event["batch_id"] for event in clusters], [0, 0, 1, 1])

        if PLOTTING_IMPORT_ERROR is None:
            image_path = Path(self.temporary_directory.name) / "pipeline.png"
            outputs = plot_pipeline(
                selected,
                cores,
                aggregator,
                batches,
                frequency_mhz,
                image_path,
                cores_per_page=4,
                unit="us",
                view="trace",
                label_mode="compact",
                dpi=72,
            )
            self.assertEqual(outputs, [image_path])
            self.assertTrue(image_path.is_file())

    def test_reader_and_compute_use_the_same_active_list_ordinals(self):
        def cluster(risc, start, duration):
            return {
                "pcie_slot": "0",
                "core_x": 1,
                "core_y": 2,
                "risc": risc,
                "run_id": "0",
                "zone": "Cluster",
                "batch_id": 0,
                "start_cycle": start,
                "end_cycle": start + duration,
                "duration_cycles": duration,
            }

        events = [cluster("NCRISC", 10, 40), cluster("NCRISC", 60, 30)]
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            events.extend(
                [
                    cluster(risc, 20, 25),
                    cluster(risc, 46, 1),  # inactive scheduled slot
                    cluster(risc, 70, 20),
                    cluster(risc, 91, 1),  # inactive scheduled slot
                ]
            )

        add_phase_ordinals(events)
        reader_ordinals = [event.get("phase_ordinal") for event in events if event["risc"] == "NCRISC"]
        self.assertEqual(reader_ordinals, [1, 2])
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            compute_ordinals = [event.get("phase_ordinal") for event in events if event["risc"] == risc]
            self.assertEqual(compute_ordinals, [1, None, 2, None])


if __name__ == "__main__":
    unittest.main()
