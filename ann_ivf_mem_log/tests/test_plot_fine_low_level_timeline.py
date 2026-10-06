import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_fine_low_level_timeline import (
    _event_label,
    _timeline_windows,
    read_low_level_events,
    select_events,
)


class FineLowLevelTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.profile_log = Path(self.temporary_directory.name) / "profile_log_device.csv"

    @staticmethod
    def row(core_x, core_y, risc, cycle, zone, marker_type, data=0, trace_id="zone-trace"):
        return [
            0,
            core_x,
            core_y,
            risc,
            100,
            cycle,
            data,
            7,
            trace_id,
            "3" if trace_id == "zone-trace" else "9",
            zone,
            marker_type,
            1,
            "kernel.cpp",
            "",
        ]

    def write_profile(self):
        rows = []
        for risc, zone in (
            ("NCRISC", "IVF.Low.Worker.Reader.ReadBarrier"),
            ("TRISC_0", "IVF.Low.Worker.Compute.LocalSortAndWinnerPack"),
            ("TRISC_1", "IVF.Low.Worker.Compute.LocalSortAndWinnerPack"),
            ("TRISC_2", "IVF.Low.Worker.Compute.LocalSortAndWinnerPack"),
            ("BRISC", "IVF.Low.Worker.Writer.OutputWrite"),
        ):
            rows.extend(
                [
                    self.row(2, 1, risc, 100, "IVF.Low.BatchID", "TS_DATA", 0, "data-trace"),
                    self.row(2, 1, risc, 105, "IVF.Low.SliceID", "TS_DATA", 0, "data-trace"),
                    self.row(2, 1, risc, 110, f"{zone}.Begin", "TS_DATA"),
                    self.row(2, 1, risc, 140, f"{zone}.End", "TS_DATA"),
                ]
            )
        for risc, zone in (
            ("NCRISC", "IVF.Low.Aggregator.Reader.PartialRead"),
            ("TRISC_0", "IVF.Low.Aggregator.Compute.PartialSort"),
            ("TRISC_1", "IVF.Low.Aggregator.Compute.PartialSort"),
            ("TRISC_2", "IVF.Low.Aggregator.Compute.PartialSort"),
            ("BRISC", "IVF.Low.Aggregator.Writer.ValuesWrite"),
        ):
            rows.extend(
                [
                    self.row(1, 1, risc, 100, "IVF.Low.BatchID", "TS_DATA", 0, "data-trace"),
                    self.row(1, 1, risc, 145, "IVF.Low.PartialID", "TS_DATA", 0, "data-trace"),
                    self.row(1, 1, risc, 150, f"{zone}.Begin", "TS_DATA"),
                    self.row(1, 1, risc, 190, f"{zone}.End", "TS_DATA"),
                ]
            )

        with self.profile_log.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["ARCH: wormhole_b0", " CHIP_FREQ[MHz]: 1000"])
            writer.writerow(
                [
                    "PCIe slot",
                    " core_x",
                    " core_y",
                    " RISC processor type",
                    " timer_id",
                    " time[cycles since reset]",
                    " data",
                    " run host ID",
                    " trace id",
                    " trace id counter",
                    " zone name",
                    " type",
                    " source line",
                    " source file",
                    " meta data",
                ]
            )
            writer.writerows(rows)

    def test_all_risc_lanes_and_one_based_aggregator(self):
        self.write_profile()
        frequency_mhz, events, unmatched = read_low_level_events(self.profile_log)
        selected, aggregator, workers, invocation = select_events(
            events,
            batch_id=0,
            invocation_text="last",
            run_id=None,
            aggregator_override=None,
            requested_workers=set(),
            worker_count=3,
            slice_start=0,
            slice_count=8,
        )

        self.assertEqual(frequency_mhz, 1000)
        self.assertEqual(unmatched, {})
        self.assertEqual(aggregator, (1, 1))
        self.assertEqual(workers, [("0", 2, 1)])
        self.assertEqual(invocation, ("7", 0))
        self.assertEqual(
            {event["risc"] for event in selected},
            {"NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "BRISC"},
        )
        self.assertEqual(
            {event["item_kind"] for event in selected if event["item_kind"] is not None},
            {"slice", "partial"},
        )

    def test_split_view_separates_sampled_slices_from_completion(self):
        events = [
            {
                "zone": "IVF.Low.Worker.Reader.ReadBarrier",
                "start_cycle": 100,
                "end_cycle": 220,
            },
            {
                "zone": "IVF.Low.Worker.Compute.OutputPack",
                "start_cycle": 3000,
                "end_cycle": 3040,
            },
            {
                "zone": "IVF.Low.Aggregator.Compute.PartialSort",
                "start_cycle": 3300,
                "end_cycle": 3450,
            },
        ]
        windows = _timeline_windows(events, 100, 3500, 1.0, "split")

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][2], "Sampled worker slices")
        self.assertEqual(windows[1][2], "Completion and aggregation")
        self.assertLess(windows[0][1], windows[1][0])

    def test_compact_labels_keep_phase_and_item_identity(self):
        event = {"item_kind": "slice", "item_id": 7}
        self.assertEqual(
            _event_label(event, "LocalSortAndWinnerPack", "compact"),
            "TopK S7",
        )
        self.assertEqual(_event_label(event, "LocalSortAndWinnerPack", "none"), "")

    def test_active_scoped_zones_include_profiler_header(self):
        example_root = Path(__file__).resolve().parents[1]
        kernel_paths = sorted((example_root / "kernels").rglob("*.cpp"))
        sources = {path: path.read_text() for path in kernel_paths}
        combined_source = "\n".join(sources.values())

        self.assertNotIn("IVF.Low.", combined_source)
        self.assertNotIn("IVF_LOW_LEVEL", combined_source)
        self.assertNotIn("DeviceTimestampedData(", combined_source)

        active_zone_count = 0
        for path, source in sources.items():
            active_zone_lines = [
                line.strip()
                for line in source.splitlines()
                if "DeviceZoneScoped" in line and not line.lstrip().startswith("//")
            ]
            if not active_zone_lines:
                continue
            active_zone_count += len(active_zone_lines)
            self.assertIn(
                '#include "tools/profiler/kernel_profiler.hpp"',
                source,
                f"{path}: active device zones require kernel_profiler.hpp",
            )

        self.assertGreater(active_zone_count, 0)
        self.assertIn('// DeviceZoneScopedN("IVF_Custom_Profile_Zone");', combined_source)


if __name__ == "__main__":
    unittest.main()
