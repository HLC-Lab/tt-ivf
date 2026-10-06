#!/usr/bin/env python3
"""Analyze sequential ANN IVF Tenstorrent energy sessions.

The input is one directory produced by ``run_tt_energy.sh``. If no run
directory is supplied, the newest directory below ``energy_raw/tt_wormhole``
is selected. Every session is integrated over the exact
``measurement_start`` to ``measurement_end`` interval in ``events.csv``.

Outputs are written below ``<run>/analysis`` unless overridden:

* ``tt_energy_sessions.csv`` contains one row per process/session;
* ``tt_energy_configuration_summary.csv`` contains aggregate statistics;
* ``tt_energy_per_query_distribution.png`` compares TT and system energy;
* ``tt_energy_rsd.png`` shows run to run variability.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import re
import shlex
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from energy_sampling import rapl_energy


CONFIGURATION_ORDER = (
    (2048, 4),
    (2048, 12),
    (1024, 16),
    (512, 32),
)

CONFIG_PATTERN = re.compile(r"^nlist(?P<nlist>\d+)_nprobe(?P<nprobe>\d+)$")
SESSION_PATTERN = re.compile(r"^session_(?P<session>\d+)$")


@dataclass(frozen=True)
class SessionSpec:
    ordinal: int
    nlist: int
    nprobe: int
    session_number: int
    searches: int
    queries_per_search: int
    directory: Path
    grouping: str = "none"

    @property
    def query_count(self) -> int:
        return self.searches * self.queries_per_search


@dataclass(frozen=True)
class PowerResult:
    energy_j: float
    average_power_w: float
    samples_in_window: int
    maximum_gap_s: float


@dataclass(frozen=True)
class SessionResult:
    spec: SessionSpec
    start_time_s: float
    end_time_s: float
    duration_s: float
    tt: PowerResult
    host: PowerResult | None
    total_energy_j: float
    total_average_power_w: float
    stored_tt_energy_j: float | None
    stored_tt_delta_percent: float | None


@dataclass(frozen=True)
class MetricStats:
    mean: float
    standard_deviation: float
    rsd_percent: float
    minimum: float
    first_quartile: float
    median: float
    third_quartile: float
    maximum: float


@dataclass(frozen=True)
class ConfigurationSummary:
    grouping: str
    nlist: int
    nprobe: int
    session_count: int
    tt_energy: MetricStats
    total_energy: MetricStats
    median_tt_energy_per_query_mj: float
    median_total_energy_per_query_mj: float
    median_duration_s: float
    median_tt_power_w: float
    median_host_power_w: float | None
    median_total_power_w: float


def finite_float(text: str, context: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{context}: expected a number, got {text!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{context}: value is not finite")
    return value


def positive_int(text: str, context: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{context}: expected an integer, got {text!r}") from error
    if value <= 0:
        raise ValueError(f"{context}: value must be positive")
    return value


def newest_run_directory(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise ValueError(f"no run directories were found below {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def read_manifest(run_directory: Path) -> list[SessionSpec]:
    path = run_directory / "sessions.csv"
    required = {
        "ordinal",
        "nlist",
        "nprobe",
        "session",
        "searches",
        "queries_per_search",
        "status",
        "output_dir",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        specs: list[SessionSpec] = []
        seen: set[tuple[str, int, int, int]] = set()
        for line_number, row in enumerate(reader, start=2):
            context = f"{path}: line {line_number}"
            ordinal = positive_int((row.get("ordinal") or "").strip(), context)
            nlist = positive_int((row.get("nlist") or "").strip(), context)
            nprobe = positive_int((row.get("nprobe") or "").strip(), context)
            session = positive_int((row.get("session") or "").strip(), context)
            searches = positive_int((row.get("searches") or "").strip(), context)
            queries = positive_int(
                (row.get("queries_per_search") or "").strip(), context
            )
            try:
                status = int((row.get("status") or "").strip())
            except ValueError as error:
                raise ValueError(f"{context}: invalid process status") from error
            if status != 0:
                raise ValueError(f"{context}: benchmark status is {status}")

            grouping = (row.get("grouping") or "none").strip()
            if grouping not in {"none", "primary-list", "weighted-union"}:
                raise ValueError(f"{context}: invalid grouping {grouping!r}")
            key = (grouping, nlist, nprobe, session)
            if key in seen:
                raise ValueError(f"{context}: duplicate session {key}")
            seen.add(key)

            # Reconstruct the path so analysis still works after the run
            # directory has been copied from the remote machine.
            directory = (
                (run_directory / grouping if "grouping" in (reader.fieldnames or []) else run_directory)
                / f"nlist{nlist}_nprobe{nprobe}"
                / f"session_{session}"
            )
            specs.append(
                SessionSpec(
                    ordinal=ordinal,
                    nlist=nlist,
                    nprobe=nprobe,
                    session_number=session,
                    searches=searches,
                    queries_per_search=queries,
                    directory=directory,
                    grouping=grouping,
                )
            )

    if not specs:
        raise ValueError(f"{path}: manifest contains no completed sessions")
    specs.sort(key=lambda spec: spec.ordinal)
    return specs


def discover_without_manifest(run_directory: Path) -> list[SessionSpec]:
    specs: list[SessionSpec] = []
    ordinal = 0
    for config_directory in sorted(run_directory.iterdir()):
        if not config_directory.is_dir():
            continue
        config_match = CONFIG_PATTERN.match(config_directory.name)
        if config_match is None:
            continue
        nlist = int(config_match.group("nlist"))
        nprobe = int(config_match.group("nprobe"))
        for session_directory in sorted(config_directory.iterdir()):
            if not session_directory.is_dir():
                continue
            session_match = SESSION_PATTERN.match(session_directory.name)
            if session_match is None:
                continue
            ordinal += 1
            runs, queries_per_search = read_workload_from_command(session_directory)
            specs.append(
                SessionSpec(
                    ordinal=ordinal,
                    nlist=nlist,
                    nprobe=nprobe,
                    session_number=int(session_match.group("session")),
                    searches=runs,
                    queries_per_search=queries_per_search,
                    directory=session_directory,
                )
            )
    if not specs:
        raise ValueError(f"no session directories were found below {run_directory}")
    return specs


def read_events(path: Path) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"Event", "Timestamp"}:
            raise ValueError(f"{path}: expected Event,Timestamp")
        events: dict[str, float] = {}
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("Event") or "").strip()
            if not name:
                raise ValueError(f"{path}: line {line_number} has no event")
            if name in events:
                raise ValueError(f"{path}: duplicate event {name!r}")
            events[name] = finite_float(
                (row.get("Timestamp") or "").strip(),
                f"{path}: line {line_number}",
            )

    missing = {"measurement_start", "measurement_end"}.difference(events)
    if missing:
        raise ValueError(f"{path}: missing events {sorted(missing)}")
    start = events["measurement_start"]
    end = events["measurement_end"]
    if end <= start:
        raise ValueError(f"{path}: measurement interval is not positive")
    return start, end


def read_power_samples(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = {
            normalize_column(field): field for field in (reader.fieldnames or [])
        }
        timestamp_column = columns.get("timestamp", columns.get("time"))
        power_column = None
        for name in ("power_watts", "power_w", "power", "watts"):
            if name in columns:
                power_column = columns[name]
                break
        if timestamp_column is None or power_column is None:
            raise ValueError(
                f"{path}: expected a Timestamp column and one of "
                "Power_Watts/Power_W/Power"
            )
        samples: list[tuple[float, float]] = []
        for line_number, row in enumerate(reader, start=2):
            timestamp = finite_float(
                (row.get(timestamp_column) or "").strip(),
                f"{path}: line {line_number} timestamp",
            )
            power = finite_float(
                (row.get(power_column) or "").strip(),
                f"{path}: line {line_number} power",
            )
            if power < 0:
                raise ValueError(f"{path}: line {line_number} has negative power")
            samples.append((timestamp, power))

    if not samples:
        raise ValueError(f"{path}: no power samples")
    if any(second[0] <= first[0] for first, second in zip(samples, samples[1:])):
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    return samples


def boundary_power(samples: list[tuple[float, float]], timestamp: float) -> float:
    times = [sample[0] for sample in samples]
    position = bisect.bisect_left(times, timestamp)
    if position == 0:
        return samples[0][1]
    if position == len(samples):
        return samples[-1][1]
    right_time, right_power = samples[position]
    if right_time == timestamp:
        return right_power
    left_time, left_power = samples[position - 1]
    weight = (timestamp - left_time) / (right_time - left_time)
    return left_power + weight * (right_power - left_power)


def integrate_power(
    samples: list[tuple[float, float]], start: float, end: float, context: str
) -> PowerResult:
    samples_in_window = [sample for sample in samples if start <= sample[0] <= end]
    if len(samples) < 2:
        raise ValueError(f"{context}: fewer than two power samples")

    points = [(start, boundary_power(samples, start))]
    points.extend(sample for sample in samples if start < sample[0] < end)
    points.append((end, boundary_power(samples, end)))

    energy = 0.0
    maximum_gap = 0.0
    for (time_a, power_a), (time_b, power_b) in zip(points, points[1:]):
        gap = time_b - time_a
        energy += 0.5 * (power_a + power_b) * gap
        maximum_gap = max(maximum_gap, gap)
    duration = end - start
    return PowerResult(
        energy_j=energy,
        average_power_w=energy / duration,
        samples_in_window=len(samples_in_window),
        maximum_gap_s=maximum_gap,
    )


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_workload_from_command(directory: Path) -> tuple[int, int]:
    path = directory / "command.txt"
    if not path.is_file():
        raise ValueError(
            f"{directory}: sessions.csv is absent and command.txt is unavailable"
        )
    tokens = shlex.split(path.read_text(encoding="utf-8"))

    def option_value(*names: str) -> str:
        for name in names:
            if name in tokens:
                position = tokens.index(name)
                if position + 1 >= len(tokens):
                    break
                return tokens[position + 1]
        raise ValueError(f"{path}: missing option {'/'.join(names)}")

    runs = positive_int(option_value("--runs"), str(path))
    queries = positive_int(
        option_value("--max_num_queries", "--max-num-queries"), str(path)
    )
    return runs, queries


def read_optional_stored_tt_energy(
    directory: Path, nlist: int, nprobe: int
) -> float | None:
    path = directory / f"energy_summary_{nlist}_{nprobe}_tt.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None

    # Current summaries have one wide row and an ``energy_j`` column. Older
    # samplers used different capitalization or a two-column Metric/Value
    # layout. This file is only an audit reference; raw power is authoritative.
    normalized_rows = [
        {normalize_column(key): (value or "").strip() for key, value in row.items()}
        for row in rows
    ]
    for row in normalized_rows:
        for key in (
            "energy_j",
            "tt_energy_j",
            "total_energy_j",
            "total_tt_energy",
            "total_tt_energy_j",
        ):
            text = row.get(key, "")
            if text:
                return finite_float(text, f"{path}: {key}")

    for row in normalized_rows:
        metric = row.get("metric", row.get("measurement", row.get("name", "")))
        value = row.get("value", "")
        normalized_metric = normalize_column(metric)
        if value and normalized_metric in {
            "energy_j",
            "tt_energy_j",
            "total_energy_j",
            "total_tt_energy",
            "total_tt_energy_j",
        }:
            return finite_float(value, f"{path}: {metric}")
    return None


def analyse_session(spec: SessionSpec, window: str = "full") -> SessionResult:
    if not spec.directory.is_dir():
        raise ValueError(f"session directory does not exist: {spec.directory}")
    start, end = read_events(spec.directory / "events.csv")
    intervals = [(start, end)]
    if window != "full":
        with (spec.directory / "events.csv").open(newline="") as handle:
            events = {row["Event"]: finite_float(row["Timestamp"], str(spec.directory)) for row in csv.DictReader(handle)}
        names = [("initialization_start", "initialization_end")] if window == "initialization" else [
            (f"run_{run}_start", f"run_{run}_end") for run in range(1, spec.searches + 1)]
        intervals = []
        previous_end = start
        for a, b in names:
            if a not in events or b not in events:
                raise ValueError(f"{spec.directory}: missing boundaries {a}/{b}; needs a new capture")
            if not previous_end <= events[a] < events[b] <= end:
                raise ValueError(f"{spec.directory}: invalid or overlapping {a}/{b} interval")
            intervals.append((events[a], events[b]))
            previous_end = events[b]
        start, end = intervals[0][0], intervals[-1][1]
    duration = sum(b - a for a, b in intervals)
    stored_tt_energy = read_optional_stored_tt_energy(
        spec.directory, spec.nlist, spec.nprobe
    )

    tt_candidates = (
        spec.directory / f"power_log_{spec.nlist}_{spec.nprobe}_tt.csv",
        spec.directory / "tt_power.csv",
    )
    tt_path = next((path for path in tt_candidates if path.is_file()), None)
    if tt_path is not None:
        samples = read_power_samples(tt_path)
        # New capture files must bracket exact C++ timestamps; old captures
        # retain their historical boundary handling for compatibility.
        is_new = (spec.directory / "host_rapl.csv").is_file() and (spec.directory / "sampler_events.csv").is_file()
        if is_new and (samples[0][0] > start or samples[-1][0] < end):
            raise ValueError(f"{tt_path}: samples do not bracket the C++ interval")
        parts = [integrate_power(samples, a, b, str(tt_path)) for a, b in intervals]
        energy = sum(part.energy_j for part in parts)
        tt = PowerResult(energy, energy / duration, sum(part.samples_in_window for part in parts),
                         max(part.maximum_gap_s for part in parts))
    elif window == "full" and stored_tt_energy is not None and stored_tt_energy > 0:
        # Legacy samplers sometimes wrote only their already-integrated
        # summary. Preserve those results, while marking the raw sample fields
        # as unavailable. New runs always retain the raw log.
        tt = PowerResult(
            energy_j=stored_tt_energy,
            average_power_w=stored_tt_energy / duration,
            samples_in_window=0,
            maximum_gap_s=math.nan,
        )
    else:
        raise ValueError(
            f"neither {tt_candidates[0].name} nor tt_power.csv exists, and "
            "the legacy summary contains no "
            "positive TT energy; this session has no recoverable TT power data"
        )

    host_candidates = (
        spec.directory / f"power_log_{spec.nlist}_{spec.nprobe}_cpu.csv",
        spec.directory / "host_power_derived.csv",
    )
    host_path = next((path for path in host_candidates if path.is_file()), None)
    host = None
    if (spec.directory / "host_rapl.csv").is_file():
        parts = [rapl_energy(spec.directory, a, b) for a, b in intervals]
        energy = sum(part[0] for part in parts)
        host = PowerResult(energy, energy / duration, sum(part[1] for part in parts), max(part[2] for part in parts))
    elif host_path is not None:
        samples = read_power_samples(host_path)
        parts = [integrate_power(samples, a, b, str(host_path)) for a, b in intervals]
        energy = sum(part.energy_j for part in parts)
        host = PowerResult(energy, energy / duration, sum(part.samples_in_window for part in parts),
                           max(part.maximum_gap_s for part in parts))

    stored_delta = (
        (tt.energy_j - stored_tt_energy) / stored_tt_energy * 100.0
        if window == "full" and stored_tt_energy is not None and stored_tt_energy > 0
        else None
    )
    total_energy = tt.energy_j + (host.energy_j if host is not None else 0.0)
    return SessionResult(
        spec=spec,
        start_time_s=start,
        end_time_s=end,
        duration_s=duration,
        tt=tt,
        host=host,
        total_energy_j=total_energy,
        total_average_power_w=total_energy / duration,
        stored_tt_energy_j=stored_tt_energy,
        stored_tt_delta_percent=stored_delta,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_stats(values: list[float]) -> MetricStats:
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) >= 2 else math.nan
    return MetricStats(
        mean=mean,
        standard_deviation=deviation,
        rsd_percent=deviation / mean * 100.0 if mean > 0 else math.nan,
        minimum=min(values),
        first_quartile=percentile(values, 0.25),
        median=statistics.median(values),
        third_quartile=percentile(values, 0.75),
        maximum=max(values),
    )


def group_results(
    results: Iterable[SessionResult],
) -> dict[tuple[str, int, int], list[SessionResult]]:
    grouped: dict[tuple[str, int, int], list[SessionResult]] = {}
    for result in results:
        key = (result.spec.grouping, result.spec.nlist, result.spec.nprobe)
        grouped.setdefault(key, []).append(result)
    return grouped


def summarize(results: list[SessionResult]) -> list[ConfigurationSummary]:
    summaries: list[ConfigurationSummary] = []
    for (grouping, nlist, nprobe), group in group_results(results).items():
        tt_energies = [result.tt.energy_j for result in group]
        total_energies = [result.total_energy_j for result in group]
        query_counts = {result.spec.query_count for result in group}
        if len(query_counts) != 1:
            raise ValueError(f"configuration {nlist}/{nprobe} has mixed query counts")
        host_presence = {result.host is not None for result in group}
        if len(host_presence) != 1:
            raise ValueError(
                f"configuration {nlist}/{nprobe} has host power for only some sessions"
            )
        query_count = next(iter(query_counts))
        host_powers = [
            result.host.average_power_w
            for result in group
            if result.host is not None
        ]
        summaries.append(
            ConfigurationSummary(
                grouping=grouping,
                nlist=nlist,
                nprobe=nprobe,
                session_count=len(group),
                tt_energy=metric_stats(tt_energies),
                total_energy=metric_stats(total_energies),
                median_tt_energy_per_query_mj=(
                    statistics.median(tt_energies) / query_count * 1000.0
                ),
                median_total_energy_per_query_mj=(
                    statistics.median(total_energies) / query_count * 1000.0
                ),
                median_duration_s=statistics.median(
                    result.duration_s for result in group
                ),
                median_tt_power_w=statistics.median(
                    result.tt.average_power_w for result in group
                ),
                median_host_power_w=(
                    statistics.median(host_powers) if host_powers else None
                ),
                median_total_power_w=statistics.median(
                    result.total_average_power_w for result in group
                ),
            )
        )

    summaries.sort(
        key=lambda summary: (summary.nlist, summary.nprobe, summary.grouping)
    )
    return summaries


def write_session_csv(results: list[SessionResult], path: Path, window: str = "full") -> None:
    fields = [
        "Window",
        "Grouping",
        "Ordinal",
        "Session",
        "NList",
        "NProbe",
        "Searches",
        "QueriesPerSearch",
        "TotalQueries",
        "StartTimestamp",
        "EndTimestamp",
        "Duration_s",
        "TTSamples",
        "TTMaxGap_s",
        "TTAveragePower_W",
        "TTEnergy_J",
        "TTEnergyPerQuery_mJ",
        "HostSamples",
        "HostMaxGap_s",
        "HostAveragePower_W",
        "HostEnergy_J",
        "TotalAveragePower_W",
        "TotalEnergy_J",
        "TotalEnergyPerQuery_mJ",
        "StoredTTEnergy_J",
        "StoredTTDelta_percent",
        "SessionDirectory",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            spec = result.spec
            host = result.host
            writer.writerow(
                {
                    "Window": window,
                    "Grouping": spec.grouping,
                    "Ordinal": spec.ordinal,
                    "Session": spec.session_number,
                    "NList": spec.nlist,
                    "NProbe": spec.nprobe,
                    "Searches": spec.searches,
                    "QueriesPerSearch": spec.queries_per_search,
                    "TotalQueries": spec.query_count,
                    "StartTimestamp": f"{result.start_time_s:.9f}",
                    "EndTimestamp": f"{result.end_time_s:.9f}",
                    "Duration_s": f"{result.duration_s:.9f}",
                    "TTSamples": result.tt.samples_in_window,
                    "TTMaxGap_s": f"{result.tt.maximum_gap_s:.9f}",
                    "TTAveragePower_W": f"{result.tt.average_power_w:.9f}",
                    "TTEnergy_J": f"{result.tt.energy_j:.9f}",
                    "TTEnergyPerQuery_mJ": (
                        f"{result.tt.energy_j / spec.query_count * 1000.0:.9f}"
                    ),
                    "HostSamples": host.samples_in_window if host else "",
                    "HostMaxGap_s": f"{host.maximum_gap_s:.9f}" if host else "",
                    "HostAveragePower_W": (
                        f"{host.average_power_w:.9f}" if host else ""
                    ),
                    "HostEnergy_J": f"{host.energy_j:.9f}" if host else "",
                    "TotalAveragePower_W": f"{result.total_average_power_w:.9f}",
                    "TotalEnergy_J": f"{result.total_energy_j:.9f}",
                    "TotalEnergyPerQuery_mJ": (
                        f"{result.total_energy_j / spec.query_count * 1000.0:.9f}"
                    ),
                    "StoredTTEnergy_J": (
                        f"{result.stored_tt_energy_j:.9f}"
                        if result.stored_tt_energy_j is not None
                        else ""
                    ),
                    "StoredTTDelta_percent": (
                        f"{result.stored_tt_delta_percent:.9f}"
                        if result.stored_tt_delta_percent is not None
                        else ""
                    ),
                    "SessionDirectory": str(spec.directory),
                }
            )


def write_summary_csv(
    summaries: list[ConfigurationSummary], path: Path, window: str = "full"
) -> None:
    fields = [
        "Window",
        "Grouping",
        "NList",
        "NProbe",
        "SessionCount",
        "MeanTTEnergy_J",
        "StdTTEnergy_J",
        "TTEnergyRSD_percent",
        "MinTTEnergy_J",
        "Q1TTEnergy_J",
        "MedianTTEnergy_J",
        "Q3TTEnergy_J",
        "IQR_TTEnergy_J",
        "MaxTTEnergy_J",
        "MedianTTEnergyPerQuery_mJ",
        "MeanTotalEnergy_J",
        "StdTotalEnergy_J",
        "TotalEnergyRSD_percent",
        "MinTotalEnergy_J",
        "Q1TotalEnergy_J",
        "MedianTotalEnergy_J",
        "Q3TotalEnergy_J",
        "IQR_TotalEnergy_J",
        "MaxTotalEnergy_J",
        "MedianTotalEnergyPerQuery_mJ",
        "MedianDuration_s",
        "MedianTTAveragePower_W",
        "MedianHostAveragePower_W",
        "MedianTotalAveragePower_W",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            tt = summary.tt_energy
            total = summary.total_energy
            writer.writerow(
                {
                    "Window": window,
                    "Grouping": summary.grouping,
                    "NList": summary.nlist,
                    "NProbe": summary.nprobe,
                    "SessionCount": summary.session_count,
                    "MeanTTEnergy_J": f"{tt.mean:.9f}",
                    "StdTTEnergy_J": f"{tt.standard_deviation:.9f}",
                    "TTEnergyRSD_percent": f"{tt.rsd_percent:.9f}",
                    "MinTTEnergy_J": f"{tt.minimum:.9f}",
                    "Q1TTEnergy_J": f"{tt.first_quartile:.9f}",
                    "MedianTTEnergy_J": f"{tt.median:.9f}",
                    "Q3TTEnergy_J": f"{tt.third_quartile:.9f}",
                    "IQR_TTEnergy_J": f"{tt.third_quartile - tt.first_quartile:.9f}",
                    "MaxTTEnergy_J": f"{tt.maximum:.9f}",
                    "MedianTTEnergyPerQuery_mJ": (
                        f"{summary.median_tt_energy_per_query_mj:.9f}"
                    ),
                    "MeanTotalEnergy_J": f"{total.mean:.9f}",
                    "StdTotalEnergy_J": f"{total.standard_deviation:.9f}",
                    "TotalEnergyRSD_percent": f"{total.rsd_percent:.9f}",
                    "MinTotalEnergy_J": f"{total.minimum:.9f}",
                    "Q1TotalEnergy_J": f"{total.first_quartile:.9f}",
                    "MedianTotalEnergy_J": f"{total.median:.9f}",
                    "Q3TotalEnergy_J": f"{total.third_quartile:.9f}",
                    "IQR_TotalEnergy_J": f"{total.third_quartile - total.first_quartile:.9f}",
                    "MaxTotalEnergy_J": f"{total.maximum:.9f}",
                    "MedianTotalEnergyPerQuery_mJ": (
                        f"{summary.median_total_energy_per_query_mj:.9f}"
                    ),
                    "MedianDuration_s": f"{summary.median_duration_s:.9f}",
                    "MedianTTAveragePower_W": f"{summary.median_tt_power_w:.9f}",
                    "MedianHostAveragePower_W": (
                        f"{summary.median_host_power_w:.9f}"
                        if summary.median_host_power_w is not None
                        else ""
                    ),
                    "MedianTotalAveragePower_W": (
                        f"{summary.median_total_power_w:.9f}"
                    ),
                }
            )


def print_summary(summaries: list[ConfigurationSummary]) -> None:
    heading = (
        f"{'Grouping':<15} {'Config':<10} {'N':>3} {'TT Q1/Median/Q3 (J)':>25} "
        f"{'TT IQR (J)':>11} {'TT Min/Max (J)':>19} {'TT RSD':>8} "
        f"{'System Q1/Median/Q3 (J)':>29} {'System IQR (J)':>15} "
        f"{'System Min/Max (J)':>23} {'System RSD':>11} "
        f"{'TT mJ/q':>9} {'System mJ/q':>12} {'Time s':>9}"
    )
    print(heading)
    print("-" * len(heading))
    for summary in summaries:
        tt_rsd = (
            f"{summary.tt_energy.rsd_percent:.2f}%"
            if math.isfinite(summary.tt_energy.rsd_percent)
            else "N/A"
        )
        system_rsd = (
            f"{summary.total_energy.rsd_percent:.2f}%"
            if math.isfinite(summary.total_energy.rsd_percent)
            else "N/A"
        )
        tt_quartiles = (
            f"{summary.tt_energy.first_quartile:.2f}/"
            f"{summary.tt_energy.median:.2f}/"
            f"{summary.tt_energy.third_quartile:.2f}"
        )
        tt_extrema = (
            f"{summary.tt_energy.minimum:.2f}/"
            f"{summary.tt_energy.maximum:.2f}"
        )
        total_quartiles = (
            f"{summary.total_energy.first_quartile:.2f}/"
            f"{summary.total_energy.median:.2f}/"
            f"{summary.total_energy.third_quartile:.2f}"
        )
        total_extrema = (
            f"{summary.total_energy.minimum:.2f}/"
            f"{summary.total_energy.maximum:.2f}"
        )
        tt_iqr = summary.tt_energy.third_quartile - summary.tt_energy.first_quartile
        total_iqr = summary.total_energy.third_quartile - summary.total_energy.first_quartile
        print(
            f"{summary.grouping:<15} {summary.nlist}/{summary.nprobe:<5} {summary.session_count:>3d} "
            f"{tt_quartiles:>25} {tt_iqr:>11.2f} {tt_extrema:>19} {tt_rsd:>8} "
            f"{total_quartiles:>29} {total_iqr:>15.2f} {total_extrema:>23} "
            f"{system_rsd:>11} "
            f"{summary.median_tt_energy_per_query_mj:>9.4f} "
            f"{summary.median_total_energy_per_query_mj:>12.4f} "
            f"{summary.median_duration_s:>9.3f}"
        )


def save_distribution_plot(results: list[SessionResult], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError("matplotlib is required unless --no-plots is used") from error

    grouped = group_results(results)
    configurations = sorted(grouped, key=lambda key: (key[1], key[2], key[0]))
    positions = list(range(len(configurations)))
    figure, axis = plt.subplots(figsize=(12.5, 5.5))

    have_host = all(result.host is not None for result in results)
    for index, configuration in enumerate(configurations):
        group = grouped[configuration]
        tt_values = [
            result.tt.energy_j / result.spec.query_count * 1000.0
            for result in group
        ]
        box = axis.boxplot(
            [tt_values],
            positions=[index - (0.18 if have_host else 0.0)],
            widths=0.28,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 5,
            },
            medianprops={"color": "black"},
        )
        box["boxes"][0].set_facecolor("#4E79A7")
        axis.scatter(
            [index - (0.18 if have_host else 0.0)] * len(tt_values),
            tt_values,
            color="black",
            s=13,
            alpha=0.45,
            zorder=3,
        )

        if have_host:
            total_values = [
                result.total_energy_j / result.spec.query_count * 1000.0
                for result in group
            ]
            box = axis.boxplot(
                [total_values],
                positions=[index + 0.18],
                widths=0.28,
                patch_artist=True,
                showmeans=True,
                manage_ticks=False,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 5,
                },
                medianprops={"color": "black"},
            )
            box["boxes"][0].set_facecolor("#59A14F")
            axis.scatter(
                [index + 0.18] * len(total_values),
                total_values,
                color="black",
                s=13,
                alpha=0.45,
                zorder=3,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels([f"{nlist}/{nprobe}\n{grouping}" for grouping, nlist, nprobe in configurations])
    axis.set_xlabel("NList / NProbe")
    axis.set_ylabel("Energy per query (mJ)")
    axis.grid(True, axis="y", linestyle=":", alpha=0.5)
    handles = [Patch(facecolor="#4E79A7", label="TT board")]
    if have_host:
        handles.append(Patch(facecolor="#59A14F", label="TT board + host CPU"))
    axis.legend(handles=handles, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_rsd_plot(summaries: list[ConfigurationSummary], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required unless --no-plots is used") from error

    positions = list(range(len(summaries)))
    figure, axis = plt.subplots(figsize=(11.5, 5.0))
    width = 0.34
    have_host = all(summary.median_host_power_w is not None for summary in summaries)
    axis.bar(
        [position - width / 2 if have_host else position for position in positions],
        [summary.tt_energy.rsd_percent for summary in summaries],
        width=width,
        color="#4E79A7",
        label="TT board",
    )
    if have_host:
        axis.bar(
            [position + width / 2 for position in positions],
            [summary.total_energy.rsd_percent for summary in summaries],
            width=width,
            color="#59A14F",
            label="TT board + host CPU",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{summary.nlist}/{summary.nprobe}\n{summary.grouping}" for summary in summaries]
    )
    axis.set_xlabel("NList / NProbe")
    axis.set_ylabel("Energy RSD (%)")
    axis.grid(True, axis="y", linestyle=":", alpha=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_grouping_comparison(
    summaries: list[ConfigurationSummary], results: list[SessionResult],
    directory: Path, window: str,
) -> None:
    lookup = {(s.grouping, s.nlist, s.nprobe): s for s in summaries}
    grouped = group_results(results)
    rows = []
    for candidate in summaries:
        if candidate.grouping == "none":
            continue
        baseline = lookup.get(("none", candidate.nlist, candidate.nprobe))
        if baseline is None:
            continue
        a = grouped[("none", candidate.nlist, candidate.nprobe)]
        b = grouped[(candidate.grouping, candidate.nlist, candidate.nprobe)]
        if len({r.spec.query_count for r in a + b}) != 1:
            raise ValueError("Cannot compare modes with different total query counts")
        if any(r.host is None for r in a + b):
            raise ValueError("Grouping comparison requires host energy in both modes")
        before = baseline.median_total_energy_per_query_mj
        after = candidate.median_total_energy_per_query_mj
        tt_before = baseline.median_tt_energy_per_query_mj
        tt_after = candidate.median_tt_energy_per_query_mj
        rows.append({
            "Window": window, "NList": candidate.nlist, "NProbe": candidate.nprobe,
            "Baseline": "none", "Candidate": candidate.grouping,
            "BaselineSessions": baseline.session_count, "CandidateSessions": candidate.session_count,
            "BaselineTT_mJQuery": tt_before, "CandidateTT_mJQuery": tt_after,
            "TTEnergySaving_percent": (1 - tt_after / tt_before) * 100 if tt_before > 0 else math.nan,
            "BaselineSystem_mJQuery": before, "CandidateSystem_mJQuery": after,
            "SystemEnergySaving_percent": (1 - after / before) * 100 if before > 0 else math.nan,
            "BaselineSystemRSD_percent": baseline.total_energy.rsd_percent,
            "CandidateSystemRSD_percent": candidate.total_energy.rsd_percent,
            "BaselineSystemIQR_J": baseline.total_energy.third_quartile - baseline.total_energy.first_quartile,
            "CandidateSystemIQR_J": candidate.total_energy.third_quartile - candidate.total_energy.first_quartile,
        })
    report = ["# Tenstorrent energy comparison", "", f"Integration window: **{window}**.", "",
              "Full includes device creation, normalization, IVF cluster assignment, packing/upload and all searches. "
              "Dataset loading/conversion precedes the window. Searches sums the recorded search intervals, "
              "including grouping, query repacking and upload. The energy workload has no warmup or final result readback. "
              "Kernel-cache preparation, when enabled, runs outside the measurements.", "",
              "Host energy counts one RAPL counter per package prefix X (one X-die-Y reading per socket). "
              "System means TT board + those host packages. "
              "It is not wall-socket energy. Telemetry and other host activity are included; no idle subtraction is applied. "
              "RSD and IQR describe independent fresh-process sessions, not the repetitions within one process.", "",
              "| NList/NProbe | Grouping | Sessions | Median TT (J) | Median host (J) | Median system (J) | System mJ/query | System RSD (%) | System IQR (J) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        group = grouped[(summary.grouping, summary.nlist, summary.nprobe)]
        host = statistics.median(r.host.energy_j for r in group if r.host) if any(r.host for r in group) else math.nan
        report.append(f"| {summary.nlist}/{summary.nprobe} | {summary.grouping} | {summary.session_count} | "
                      f"{summary.tt_energy.median:.3f} | {host:.3f} | {summary.total_energy.median:.3f} | "
                      f"{summary.median_total_energy_per_query_mj:.5f} | {summary.total_energy.rsd_percent:.3f} | "
                      f"{summary.total_energy.third_quartile - summary.total_energy.first_quartile:.3f} |")
    report += ["", "Component medians need not sum to the median system energy. "
               "System statistics are calculated from device + host energy for each session first. "
               "Sampling gaps and counts are retained in the per-session CSV. "
               "Energy savings must be reported alongside the separately validated recall change."]
    if rows:
        path = directory / "tt_energy_grouping_comparison.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report += ["", "| NList/NProbe | Candidate | TT energy saving (%) | System energy saving (%) |",
                   "|---|---|---:|---:|"]
        for row in rows:
            report.append(f"| {row['NList']}/{row['NProbe']} | {row['Candidate']} | "
                          f"{row['TTEnergySaving_percent']:.2f} | {row['SystemEnergySaving_percent']:.2f} |")
        print(f"Grouping comparison: {path}")
    report_path = directory / "tt_energy_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Energy report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory",
        nargs="?",
        type=Path,
        help="one tt_wormhole run directory; defaults to the newest run",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("energy_raw/tt_wormhole"),
        help="root used to discover the newest run",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="output directory; defaults to RUN_DIRECTORY/analysis",
    )
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=10,
        help="expected sessions per configuration",
    )
    parser.add_argument("--window", choices=["full", "searches", "initialization"], default="full",
                        help="full includes initialization and searches; searches sums only recorded search intervals")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.expected_sessions <= 0:
        parser.error("--expected-sessions must be positive")
    try:
        if args.run_directory is not None:
            run_directory = args.run_directory.resolve()
        else:
            root = args.root.resolve()
            if not root.is_dir():
                parser.error(f"energy root does not exist: {root}")
            run_directory = newest_run_directory(root)
        if not run_directory.is_dir():
            parser.error(f"run directory does not exist: {run_directory}")

        manifest = run_directory / "sessions.csv"
        specs = (
            read_manifest(run_directory)
            if manifest.is_file()
            else discover_without_manifest(run_directory)
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Run directory: {run_directory}")
    print(f"Discovered sessions: {len(specs)}")

    results: list[SessionResult] = []
    failures: list[str] = []
    for spec in specs:
        try:
            results.append(analyse_session(spec, args.window))
        except (OSError, ValueError) as error:
            failures.append(f"{spec.directory}: {error}")

    if failures:
        print("\nRejected sessions:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    if not results:
        parser.error("all sessions failed validation")

    try:
        summaries = summarize(results)
    except ValueError as error:
        parser.error(str(error))
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else run_directory / "analysis" if args.window == "full" else run_directory / "analysis" / args.window
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    sessions_path = output_directory / "tt_energy_sessions.csv"
    summary_path = output_directory / "tt_energy_configuration_summary.csv"
    write_session_csv(results, sessions_path, args.window)
    write_summary_csv(summaries, summary_path, args.window)

    print()
    print(f"Integration window: {args.window}")
    print_summary(summaries)
    try:
        write_grouping_comparison(summaries, results, output_directory, args.window)
    except ValueError as error:
        parser.error(str(error))

    incomplete = [
        summary
        for summary in summaries
        if summary.session_count != args.expected_sessions
    ]
    if incomplete:
        print("\nWARNING: unexpected session counts:", file=sys.stderr)
        for summary in incomplete:
            print(
                f"  {summary.grouping} {summary.nlist}/{summary.nprobe}: "
                f"{summary.session_count}, expected {args.expected_sessions}",
                file=sys.stderr,
            )

    if not args.no_plots:
        try:
            save_distribution_plot(
                results,
                output_directory / "tt_energy_per_query_distribution.png",
            )
            save_rsd_plot(summaries, output_directory / "tt_energy_rsd.png")
        except RuntimeError as error:
            parser.error(str(error))

    print()
    print(f"Per-session results: {sessions_path}")
    print(f"Configuration summary: {summary_path}")
    if not args.no_plots:
        print(
            "Energy distribution: "
            f"{output_directory / 'tt_energy_per_query_distribution.png'}"
        )
        print(f"Energy RSD: {output_directory / 'tt_energy_rsd.png'}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
