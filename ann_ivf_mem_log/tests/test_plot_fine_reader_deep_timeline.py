import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_fine_reader_deep_timeline import (
    PLOTTING_IMPORT_ERROR,
    plot_pages,
    read_deep_events,
    select_batch_context,
    select_deep_events,
    write_events,
)


class FineReaderDeepTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.profile_log = Path(self.temporary_directory.name) / "profile_log_device.csv"

    @staticmethod
    def marker(cycle, zone, marker_type, data=0):
        return [0, 1, 2, "NCRISC", 100, cycle, data, 7, "", "", zone, marker_type, 1, "reader_fine.cpp", ""]

    def write_profile(self):
        markers = [
            self.marker(100, "Fine_Search_Reader_QueryBatch_ID", "TS_DATA", 0),
            self.marker(101, "Fine_Search_Reader_QueryBatch", "ZONE_START"),
            self.marker(110, "MemLog_Fine_DatasetAndIndices_DRAM_to_L1", "ZONE_START"),
            self.marker(112, "MemLog_Fine_Query_DRAM_to_L1", "ZONE_START"),
            self.marker(120, "MemLog_Fine_Query_DRAM_to_L1", "ZONE_END"),
            self.marker(121, "MemLog_Fine_Cluster_slice", "ZONE_START"),
            self.marker(140, "MemLog_Fine_Cluster_slice", "ZONE_END"),
            self.marker(141, "MemLog_Fine_Cluster_slice", "ZONE_START"),
            self.marker(170, "MemLog_Fine_Cluster_slice", "ZONE_END"),
            self.marker(180, "MemLog_Fine_DatasetAndIndices_DRAM_to_L1", "ZONE_END"),
            self.marker(190, "Fine_Search_Reader_QueryBatch", "ZONE_END"),
            # A repeated batch ID marks the next search invocation.
            self.marker(300, "Fine_Search_Reader_QueryBatch_ID", "TS_DATA", 0),
            self.marker(301, "Fine_Search_Reader_QueryBatch", "ZONE_START"),
            self.marker(310, "MemLog_Fine_DatasetAndIndices_DRAM_to_L1", "ZONE_START"),
            self.marker(312, "MemLog_Fine_Query_DRAM_to_L1", "ZONE_START"),
            self.marker(320, "MemLog_Fine_Query_DRAM_to_L1", "ZONE_END"),
            self.marker(321, "MemLog_Fine_Cluster_slice", "ZONE_START"),
            self.marker(350, "MemLog_Fine_Cluster_slice", "ZONE_END"),
            self.marker(360, "MemLog_Fine_DatasetAndIndices_DRAM_to_L1", "ZONE_END"),
            self.marker(370, "Fine_Search_Reader_QueryBatch", "ZONE_END"),
        ]
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
            writer.writerows(markers)

    def test_last_invocation_and_slice_metadata(self):
        self.write_profile()
        frequency_mhz, events, unmatched = read_deep_events(self.profile_log)
        targets, context, selected_invocation = select_deep_events(
            events,
            batch_id=0,
            invocation_text="last",
            run_id=None,
            requested_cores=set(),
            task_ordinal=None,
            slice_start=0,
            slice_count_limit=1,
        )

        self.assertEqual(frequency_mhz, 1000)
        self.assertEqual(unmatched, {})
        self.assertEqual(selected_invocation, ("7", 1))
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["slice_ordinal"], 0)
        self.assertEqual(targets[0]["task_ordinal"], 0)
        self.assertEqual(targets[0]["task_slice_ordinal"], 0)
        self.assertEqual(targets[0]["duration_cycles"], 29)
        self.assertEqual(
            {event["zone"] for event in context},
            {
                "Fine_Search_Reader_QueryBatch",
                "MemLog_Fine_DatasetAndIndices_DRAM_to_L1",
                "MemLog_Fine_Query_DRAM_to_L1",
            },
        )

        all_batch_events = [
                {
                    "pcie_slot": "0",
                    "core_x": 1,
                    "core_y": 1,
                    "risc": "NCRISC",
                    "run_id": "7",
                    "invocation": 1,
                    "batch_id": 0,
                    "granularity": "batch",
                    "role": "reader",
                    "zone": "Core_0_0_Final_Sort_Reader_QueryBatch",
                    "start_cycle": 380,
                    "end_cycle": 410,
                    "duration_cycles": 30,
                },
                {
                    "pcie_slot": "0",
                    "core_x": 1,
                    "core_y": 2,
                    "risc": "NCRISC",
                    "run_id": "7",
                    "invocation": 1,
                    "batch_id": 0,
                    "granularity": "batch",
                    "role": "reader",
                    "zone": "Fine_Search_Reader_QueryBatch",
                    "start_cycle": 301,
                    "end_cycle": 370,
                    "duration_cycles": 69,
                },
                {
                    "pcie_slot": "0",
                    "core_x": 9,
                    "core_y": 9,
                    "risc": "NCRISC",
                    "run_id": "7",
                    "invocation": 1,
                    "batch_id": 0,
                    "granularity": "batch",
                    "role": "reader",
                    "zone": "Fine_Search_Reader_QueryBatch",
                    "start_cycle": 301,
                    "end_cycle": 370,
                    "duration_cycles": 69,
                },
            ]
        batch_events = select_batch_context(
            all_batch_events,
            batch_id=0,
            selected_invocation=selected_invocation,
            worker_cores={("0", 1, 2)},
            aggregator_core=(1, 1),
        )
        self.assertEqual(
            {(event["core_x"], event["core_y"]) for event in batch_events},
            {(1, 1), (1, 2)},
        )

        if PLOTTING_IMPORT_ERROR is None:
            image_path = Path(self.temporary_directory.name) / "combined.png"
            pages = plot_pages(
                targets,
                context,
                batch_events,
                (1, 1),
                frequency_mhz,
                image_path,
                cores_per_page=3,
                unit="us",
                dpi=72,
                subtitle="synthetic",
            )
            self.assertEqual(pages, [image_path])
            self.assertTrue(image_path.is_file())

            csv_path = Path(self.temporary_directory.name) / "combined.csv"
            write_events(csv_path, targets, context, batch_events, frequency_mhz)
            with csv_path.open(newline="") as source:
                panels = {row["panel"] for row in csv.DictReader(source)}
            self.assertEqual(panels, {"full_batch", "deep_zoom"})


if __name__ == "__main__":
    unittest.main()
