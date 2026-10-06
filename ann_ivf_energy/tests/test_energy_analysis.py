"""Boundary integration and grouping comparison checks, without hardware."""
import csv
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_tt_energy as analysis
from energy_sampling import discover_rapl, rapl_energy, unwrap, unique_package_indices, derived_host_power
from energy_sampling import RaplZone


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


class EnergyAnalysisTests(unittest.TestCase):
    def test_wrap_and_interpolated_boundaries(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_csv(root / 'rapl_zones.csv', ['Column', 'ZoneName', 'EnergyPath', 'MaxEnergyRange_uJ'],
                      [('Package0_uJ', 'package-0', '/sys/package/energy_uj', 1000)])
            write_csv(root / 'host_rapl.csv', ['Timestamp', 'Package0_uJ'],
                      [(0, 800), (1, 900), (2, 0), (3, 100)])
            energy, count, gap = rapl_energy(root, .5, 2.5)
            self.assertAlmostEqual(energy, .0002)
            self.assertEqual(count, 2)
            self.assertEqual(gap, 1)
            with self.assertRaises(ValueError):
                rapl_energy(root, -.1, 2.5)
        with self.assertRaises(ValueError):
            unwrap([1, 1001], 1000, 'counter')

    def test_package_domains_only(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for directory, zone in [('intel-rapl:0', 'package-0'), ('intel-rapl:0:0', 'core'),
                                    ('intel-rapl:1', 'package-1'), ('intel-rapl:2', 'psys')]:
                path = root / directory
                path.mkdir()
                (path / 'name').write_text(zone)
                (path / 'energy_uj').write_text('10')
                (path / 'max_energy_range_uj').write_text('1000')
            zones = discover_rapl(root)
            self.assertEqual([z.name for z in zones], ['package-0', 'package-1'])

    def test_one_die_reading_per_package_prefix(self):
        for prefix in ('', 'package-'):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                # Four distinct sysfs paths per socket, repeating its reading.
                for package in (0, 1):
                    for die in (0, 1, 2, 3):
                        path = root / f'intel-rapl:{package * 4 + die}'
                        path.mkdir()
                        (path / 'name').write_text(f'{prefix}{package}-die-{die}')
                        (path / 'energy_uj').write_text('10')
                        (path / 'max_energy_range_uj').write_text('1000')
                zones = discover_rapl(root)
                self.assertEqual([z.name for z in zones], [f'{prefix}0-die-0', f'{prefix}1-die-0'])
                self.assertEqual([z.column for z in zones], ['Package0_uJ', 'Package1_uJ'])
        # Prefer an aggregate when it exists; compare numeric prefixes and die
        # IDs rather than string startswith (prefix 1 must not collide with 10).
        names = ['package-0-die-0', 'package-0', '1-die-10', '1-die-2', '10-die-0', 'psys', 'core']
        self.assertEqual(unique_package_indices(names), [1, 3, 4])

    def test_reanalyze_all_dies_without_quadrupling_host_energy(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            zones = [RaplZone(f'Zone{i}_uJ', f'package-{i // 4}-die-{i % 4}',
                              Path(f'/sys/intel-rapl:{i}/energy_uj'), 1e9) for i in range(8)]
            write_csv(root / 'rapl_zones.csv', ['Column', 'ZoneName', 'EnergyPath', 'MaxEnergyRange_uJ'],
                      [(z.column, z.name, z.path, z.maximum_uj) for z in zones])
            samples = [(t, *[(i // 4 + 1) * t * 1e6 for i in range(8)]) for t in (0, 1, 2)]
            write_csv(root / 'host_rapl.csv', ['Timestamp', *(z.column for z in zones)], samples)
            energy, _, _ = rapl_energy(root, .25, 1.75)
            self.assertAlmostEqual(energy, 4.5)  # (1 W + 2 W) * 1.5 s
            power = derived_host_power(zones, samples)
            self.assertEqual([value for _, value in power], [3, 3])

    def test_modes_are_separate_and_system_rsd_uses_sum(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = []
            for ordinal, (mode, session, tt_power, host_power) in enumerate([
                ('none', 1, 2, 4), ('none', 2, 4, 2),
                ('weighted-union', 1, 1, 2), ('weighted-union', 2, 1, 2),
            ], 1):
                directory = root / mode / 'nlist512_nprobe16' / f'session_{session}'
                directory.mkdir(parents=True)
                # Deliberately wrong remote path: discovery must be relocatable.
                manifest.append((ordinal, mode, 512, 16, session, 2, 100, 0, '/remote/path'))
                write_csv(directory / 'events.csv', ['Event', 'Timestamp'], [
                    ('measurement_start', .5), ('initialization_start', .5),
                    ('initialization_end', .9), ('run_1_start', 1), ('run_1_end', 1.5),
                    ('run_2_start', 2), ('run_2_end', 3), ('measurement_end', 3.5)])
                (directory / 'sampler_events.csv').write_text('Event,Timestamp\n')
                write_csv(directory / 'tt_power.csv', ['Timestamp', 'Power_W'],
                          [(i, tt_power) for i in range(5)])
                write_csv(directory / 'rapl_zones.csv', ['Column', 'ZoneName', 'EnergyPath', 'MaxEnergyRange_uJ'],
                          [('Package0_uJ', 'package-0', '/sys/package/energy_uj', 1e9)])
                write_csv(directory / 'host_rapl.csv', ['Timestamp', 'Package0_uJ'],
                          [(i, host_power * i * 1e6) for i in range(5)])
            write_csv(root / 'sessions.csv', ['ordinal', 'grouping', 'nlist', 'nprobe', 'session',
                      'searches', 'queries_per_search', 'status', 'output_dir'], manifest)
            specs = analysis.read_manifest(root)
            self.assertEqual(len(specs), 4)
            results = [analysis.analyse_session(s) for s in specs]
            summaries = analysis.summarize(results)
            self.assertEqual(len(summaries), 2)
            baseline = next(s for s in summaries if s.grouping == 'none')
            self.assertGreater(baseline.tt_energy.rsd_percent, 0)
            self.assertAlmostEqual(baseline.total_energy.rsd_percent, 0)
            self.assertAlmostEqual(baseline.total_energy.median, 18)
            searches = analysis.analyse_session(specs[0], 'searches')
            self.assertAlmostEqual(searches.duration_s, 1.5)
            self.assertAlmostEqual(searches.total_energy_j, 9)
            init = analysis.analyse_session(specs[0], 'initialization')
            self.assertAlmostEqual(init.total_energy_j, 2.4)
            output = root / 'analysis'
            output.mkdir()
            analysis.write_session_csv(results, output / 'sessions.csv')
            analysis.write_summary_csv(summaries, output / 'summary.csv')
            analysis.write_grouping_comparison(summaries, results, output, 'full')
            if os.environ.get('IVF_ENERGY_TEST_PLOTS') == '1':
                analysis.save_distribution_plot(results, output / 'distribution.png')
                analysis.save_rsd_plot(summaries, output / 'rsd.png')
                self.assertGreater((output / 'distribution.png').stat().st_size, 1000)
                self.assertGreater((output / 'rsd.png').stat().st_size, 1000)
            with (output / 'tt_energy_grouping_comparison.csv').open() as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row['SystemEnergySaving_percent']), 50)
            # Missing phase markers must not quietly fall back to full energy.
            write_csv(specs[0].directory / 'events.csv', ['Event', 'Timestamp'],
                      [('measurement_start', .5), ('measurement_end', 3.5)])
            with self.assertRaises(ValueError):
                analysis.analyse_session(specs[0], 'searches')

    def test_short_power_window_uses_bracketed_samples(self):
        result = analysis.integrate_power([(0, 2), (1, 4)], .25, .75, 'power')
        self.assertAlmostEqual(result.energy_j, 1.5)
        self.assertEqual(result.samples_in_window, 0)

    def test_sampler_runs_old_energy_interface_and_brackets_events(self):
        import time
        import plot_energy_recall as sampler
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            executable = root / 'fake_energy'
            executable.write_text('''#!/usr/bin/env python3
import sys,time,csv
events=[]
def record(name): events.append((name,time.time()))
record('measurement_start')
record('initialization_start')
print('[Energy] ENERGY_MEASUREMENT_START runs=2 queries_per_run=100',flush=True)
time.sleep(.12)
record('initialization_end')
for run in (1,2):
    record(f'run_{run}_start')
    time.sleep(.18)
    record(f'run_{run}_end')
record('measurement_end')
assert '--mem_log' not in sys.argv
assert '--query-grouping' in sys.argv
path=sys.argv[sys.argv.index('--energy-events')+1]
with open(path,'w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['Event','Timestamp'])
    writer.writerows(events)
print('[Energy] ENERGY_MEASUREMENT_END',flush=True)
''')
            executable.chmod(0o755)
            origin = time.time()
            class Counter:
                def read_text(self):
                    return str((time.time() - origin) * 2e6)
                def __str__(self):
                    return '/sys/package/energy_uj'
            zones = [RaplZone('Package0_uJ', 'package-0', Counter(), 1e12)]
            argv = ['plot_energy_recall.py', '--executable', str(executable), '--nlist', '512',
                    '--nprobe', '16', '--runs', '2', '--max-num-queries', '100',
                    '--query-grouping', 'weighted-union', '--require-host', '--no-plots',
                    '--output-dir', str(root / 'output')]
            with patch.object(sys, 'argv', argv), patch.object(sampler, 'discover_rapl', return_value=zones), \
                 patch.object(sampler, 'read_tt_power_once', side_effect=lambda: (time.time(), 7., '{}')), \
                 contextlib.redirect_stdout(io.StringIO()):
                sampler.main()
            directory = root / 'output'
            spec = analysis.SessionSpec(1, 512, 16, 1, 2, 100, directory, 'weighted-union')
            result = analysis.analyse_session(spec)
            self.assertAlmostEqual(result.tt.average_power_w, 7)
            self.assertAlmostEqual(result.host.average_power_w, 2, delta=.05)
            self.assertEqual(result.spec.query_count, 200)
            self.assertGreater(result.tt.samples_in_window, 2)
            self.assertTrue((directory / 'sampler_events.csv').exists())


if __name__ == '__main__':
    unittest.main()
