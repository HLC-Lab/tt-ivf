"""RAPL package counters and boundary integration shared by energy tools."""
from __future__ import annotations

import bisect
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RaplZone:
    column: str
    name: str
    path: Path
    maximum_uj: float


def finite(value: str, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: nonfinite value")
    return result


def package_identity(name: str) -> tuple[int, int | None] | None:
    """Accept package-X, package-X-die-Y and X-die-Y sensor names."""
    match = re.fullmatch(r'(?:package-)?(\d+)(?:-die-(\d+))?', name.strip().lower())
    if match is None:
        return None
    return int(match[1]), int(match[2]) if match[2] is not None else None


def unique_package_indices(names: list[str]) -> list[int]:
    # The target host exposes repeated package readings as X-die-Y. Follow
    # its socket prefix, not the global intel-rapl:N index or number of dies.
    # Prefer a package-X aggregate; otherwise select the lowest die number.
    packages = {}
    for index, name in enumerate(names):
        identity = package_identity(name)
        if identity is None:
            continue
        package, die = identity
        preference = (die is not None, die if die is not None else -1, index)
        if package not in packages or preference < packages[package][0]:
            packages[package] = (preference, index)
    return [packages[package][1] for package in sorted(packages)]


def discover_rapl(root: Path = Path('/sys/class/powercap')) -> list[RaplZone]:
    # Include immediate symlinks in sysfs; recursive glob alone misses those.
    paths = {*root.glob('*/energy_uj'), *root.glob('**/energy_uj')}
    candidates = []
    seen = set()
    for energy in sorted(paths):
        resolved = energy.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        name = (energy.parent / 'name').read_text().strip()
        # Package counters contain their child core/uncore domains. Exclude
        # those, psys, and all but one die-labelled reading per socket.
        if package_identity(name) is None:
            continue
        candidates.append((name, energy))
    zones = []
    for index in unique_package_indices([name for name, _ in candidates]):
        name, energy = candidates[index]
        package, _ = package_identity(name)
        maximum = finite((energy.parent / 'max_energy_range_uj').read_text(), str(energy))
        value = finite(energy.read_text(), str(energy))  # also checks permissions
        if maximum <= 0 or not 0 <= value <= maximum:
            raise ValueError(f'{energy}: invalid counter range or reading')
        zones.append(RaplZone(f'Package{package}_uJ', name, energy, maximum))
    return zones


def save_rapl(directory: Path, zones: list[RaplZone], samples: list[tuple]) -> None:
    with (directory / 'rapl_zones.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Column', 'ZoneName', 'EnergyPath', 'MaxEnergyRange_uJ'])
        writer.writerows((z.column, z.name, z.path, z.maximum_uj) for z in zones)
    with (directory / 'host_rapl.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Timestamp', *(z.column for z in zones)])
        writer.writerows(samples)


def unwrap(values: list[float], maximum: float, context: str) -> list[float]:
    if maximum <= 0 or any(not 0 <= value <= maximum for value in values):
        raise ValueError(f'{context}: counter outside its recorded range')
    result = []
    offset = 0.0
    for index, value in enumerate(values):
        if index and value < values[index - 1]:
            offset += maximum
        result.append(value + offset)
    return result


def interpolate(times: list[float], values: list[float], timestamp: float) -> float:
    if not times[0] <= timestamp <= times[-1]:
        raise ValueError('Samples do not bracket the measurement boundary')
    right = bisect.bisect_left(times, timestamp)
    if times[right] == timestamp:
        return values[right]
    left = right - 1
    weight = (timestamp - times[left]) / (times[right] - times[left])
    return values[left] + weight * (values[right] - values[left])


def rapl_energy(directory: Path, start: float, end: float) -> tuple[float, int, float]:
    with (directory / 'rapl_zones.csv').open(newline='') as handle:
        zones = list(csv.DictReader(handle))
    if not zones or len({z['Column'] for z in zones}) != len(zones):
        raise ValueError(f'{directory}: missing or duplicate RAPL zones')
    # Apply the same rule when reanalyzing captures that recorded all dies.
    zones = [zones[i] for i in unique_package_indices([z['ZoneName'] for z in zones])]
    if not zones:
        raise ValueError(f'{directory}: no recognized host package counters')
    paths = [Path(z['EnergyPath']).parent for z in zones]
    if any(a != b and a in b.parents for a in paths for b in paths):
        raise ValueError(f'{directory}: overlapping RAPL parent/child zones')
    with (directory / 'host_rapl.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    times = [finite(r['Timestamp'], str(directory)) for r in rows]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f'{directory}: insufficient or nonincreasing RAPL timestamps')
    if end <= start:
        raise ValueError(f'{directory}: nonpositive interval')
    energy = 0.0
    for zone in zones:
        maximum = finite(zone['MaxEnergyRange_uJ'], str(directory))
        values = unwrap([finite(r[zone['Column']], str(directory)) for r in rows], maximum, str(directory))
        energy += (interpolate(times, values, end) - interpolate(times, values, start)) / 1e6
    relevant_gaps = [b - a for a, b in zip(times, times[1:]) if a < end and b > start]
    return energy, sum(start <= t <= end for t in times), max(relevant_gaps)


def derived_host_power(zones: list[RaplZone], samples: list[tuple]) -> list[tuple[float, float]]:
    if len(samples) < 2:
        return []
    indices = unique_package_indices([zone.name for zone in zones])
    if not indices:
        raise ValueError('No recognized host package counters')
    columns = [unwrap([row[i + 1] for row in samples], zones[i].maximum_uj, zones[i].name)
               for i in indices]
    output = []
    for i in range(1, len(samples)):
        before, after = samples[i - 1][0], samples[i][0]
        if after <= before:
            raise ValueError('RAPL clock did not increase')
        delta = sum(column[i] - column[i - 1] for column in columns) / 1e6
        output.append(((before + after) / 2, delta / (after - before)))
    return output
