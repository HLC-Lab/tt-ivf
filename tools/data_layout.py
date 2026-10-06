"""Shared paths for datasets, trained centroids and generated measurements."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_PATTERNS = ('*_train.bin', '*_queries.bin', '*_neighbors.bin', '*_distances.bin', '*.hdf5')
CENTROID_PATTERNS = ('centroids-*.bin', 'centroids-*.npy')


def data_root() -> Path:
    return Path(os.environ.get('ANN_DATA_DIR') or PROJECT / 'data')


def results_root() -> Path:
    return Path(os.environ.get('ANN_RESULTS_DIR') or PROJECT / 'results')


def dataset_file(filename: str) -> Path:
    return data_root() / 'datasets' / filename


def centroid_file(dataset: str, nlist: int) -> Path:
    return data_root() / 'centroids' / f'centroids-{dataset}-{nlist}.npy'


def result_file(experiment: str, filename: str) -> Path:
    folder = results_root() / experiment
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def data_destination(project: Path, filename: str) -> Path:
    folder = 'centroids' if filename.startswith('centroids-') else 'datasets'
    return project / 'data' / folder / filename
