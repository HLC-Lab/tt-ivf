"""Exercise the real shell runners with a fake, mutually exclusive benchmark."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RelayRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.executable = self.directory / "benchmark"
        self.calls = self.directory / "calls.jsonl"
        # The fake timeout forwards the exact argv without needing GNU
        # coreutils on macOS. Production runners still use GNU timeout.
        timeout = self.directory / "timeout"
        timeout.write_text('#!/usr/bin/env bash\nshift 2\nexec "$@"\n')
        timeout.chmod(0o755)
        self.executable.write_text(f"#!{sys.executable}\n" + r'''
import csv, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["MOCK_SOURCE"])
import summarize_timings as analysis
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
calls = Path(os.environ["MOCK_CALLS"])
lock = calls.with_suffix(".lock")
fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
os.close(fd)
try:
    with calls.open("a") as handle:
        handle.write(json.dumps({"args": args, "profiler": os.getenv("TT_METAL_DEVICE_PROFILER"),
                                 "watcher": os.getenv("TT_METAL_WATCHER"),
                                 "wait": os.getenv("TRACY_WAIT_FOR_CLIENT")}) + "\n")
    time.sleep(.03)
    if args["--candidate-reader"] == os.getenv("MOCK_FAIL_READER"):
        sys.exit(7)
    config = dict(zip(analysis.CONFIG_COLUMNS,
        (args["--nlist"], args["--nprobe"], args["--max_num_queries"], args["--k"],
         args["--query-grouping"], args["--partitions"], args["--candidate-reader"],
         args["--bank-schedule"], args["--aggregation"], 0, args["--partition-layout"],
         args["--query-broadcast"], args["--leader-prefetch-pages"],
         args["--worker-input-pages"], 0, 256)))
    timings = {name: 1 for name, _, _ in analysis.TIMINGS}
    for stage in ("h2d_us", "cpu_coarse_config_us", "cpu_fine_prep_us", "cpu_output_us"):
        timings[stage] = sum(timings[name] for name, _, parent in analysis.TIMINGS if parent == stage)
    timings["pipeline_us"] = timings["stage_sum_us"] = sum(timings[name] for name in analysis.STAGE_COLUMNS)
    rows = [dict(config, run=run, qps=int(config["queries"]) * 1e6 / timings["pipeline_us"],
                 recall=.9, **timings) for run in range(1, int(args["--runs"]) + 1)]
    with Path(args["--timings-csv"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("[Run 1] QPS=1000, Recall@10=0.9")
finally:
    lock.unlink()
''')
        self.executable.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.directory) + os.pathsep + os.environ["PATH"],
                        IVF_EXECUTABLE=str(self.executable), RUNS="2", QUERIES="64",
                        MOCK_SOURCE=str(ROOT), MOCK_CALLS=str(self.calls),
                        TT_METAL_DEVICE_PROFILER="1", TT_METAL_WATCHER="1", TRACY_WAIT_FOR_CLIENT="1")

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, script, **environment):
        output = self.directory / "output"
        return subprocess.run(["bash", str(ROOT / script), str(output)],
                              env=dict(self.env, **environment), text=True, capture_output=True, timeout=20)

    def read_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_flat_layout_uses_default_binary_and_project_root_from_another_cwd(self):
        project = self.directory / 'flat project'
        example = project / ROOT.name
        shutil.copytree(ROOT, example, ignore=shutil.ignore_patterns('__pycache__'))
        binary = project / 'build_Release/bin/ann_ivf_similarity_partitioned'
        binary.parent.mkdir(parents=True)
        shutil.copy2(self.executable, binary)
        environment = dict(self.env)
        environment.pop('IVF_EXECUTABLE')
        result = subprocess.run(['bash', str(example / 'run_timing_sweep.sh'),
                                 str(self.directory / 'flat output')],
                                cwd=self.directory, env=environment,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.read_calls()), 2)

    def test_comparison_is_sequential_and_retains_both_depths_in_summary(self):
        result = self.run_script("run_relay_comparison.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.read_calls()
        self.assertEqual([(call["args"]["--candidate-reader"], call["args"]["--nprobe"],
                           call["args"]["--leader-prefetch-pages"]) for call in calls],
                         [("direct-serial", "16", "8"), ("direct-serial", "32", "8"),
                          ("relay", "16", "4"), ("relay", "32", "4"),
                          ("relay", "16", "8"), ("relay", "32", "8")])
        self.assertTrue(all(call["profiler"] is None and call["watcher"] is None and
                            call["wait"] is None for call in calls))
        self.assertTrue(all(call["args"]["--worker-input-pages"] == "2" and
                            call["args"]["--bank-schedule"] == "fifo" for call in calls))
        report = (self.directory / "output/analysis/timing_report.md").read_text()
        self.assertEqual(report.count("## N_list=512"), 6)
        self.assertIn("leader prefetch=4 page pairs", report)
        self.assertIn("leader prefetch=8 page pairs", report)

    def test_failed_relay_stops_before_another_configuration(self):
        result = self.run_script("run_relay_comparison.sh", MOCK_FAIL_READER="relay")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(self.read_calls()), 3)
        self.assertFalse((self.directory / "output/relay8").exists())

    def test_relay_capture_uses_requested_staging_and_bank_schedule(self):
        result = self.run_script("run_device_profile.sh", CANDIDATE_READER="relay",
                                 LEADER_PREFETCH_PAGES="4", BANK_SCHEDULE="staggered", NPROBE="32")
        self.assertEqual(result.returncode, 0, result.stderr)
        call, = self.read_calls()
        self.assertEqual(call["args"]["--leader-prefetch-pages"], "4")
        self.assertEqual(call["args"]["--bank-schedule"], "staggered")
        self.assertEqual(call["profiler"], "1")
        self.assertIsNone(call["watcher"])
        self.assertIsNone(call["wait"])

    def test_invalid_staging_or_direct_staggering_is_rejected_before_launch(self):
        for environment in ({"LEADER_PREFETCH_PAGES": "16"},
                            {"CANDIDATE_READER": "direct", "BANK_SCHEDULE": "staggered"}):
            for script in ("run_timing_sweep.sh", "run_device_profile.sh"):
                with self.subTest(script=script, environment=environment):
                    result = self.run_script(script, **environment)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.calls.exists())
                    self.assertFalse((self.directory / "output").exists())


if __name__ == "__main__":
    unittest.main()
