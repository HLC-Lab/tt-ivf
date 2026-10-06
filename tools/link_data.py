#!/usr/bin/env python3
"""Link ANN inputs into data/datasets/ and data/centroids/."""
from __future__ import annotations

import argparse
from pathlib import Path
from data_layout import DATA_PATTERNS, CENTROID_PATTERNS, data_destination

PROJECT = Path(__file__).resolve().parents[1]
PATTERNS = DATA_PATTERNS + CENTROID_PATTERNS


def link_data(source: Path, destination: Path = PROJECT) -> tuple[list[Path], list[Path]]:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir():
        raise ValueError(f'Dataset source directory does not exist: {source}')
    if not destination.is_dir():
        raise ValueError(f'Project directory does not exist: {destination}')
    files: dict[str, Path] = {}
    # Prefer organized inputs; older exports used the root or a flat data/.
    for folder in (source / 'data/datasets', source / 'data/centroids', source, source / 'data'):
        if folder.is_dir():
            for pattern in PATTERNS:
                for path in sorted(folder.glob(pattern)):
                    if path.is_file():
                        files.setdefault(path.name, path.resolve())
    if not files:
        raise ValueError(f'No ANN dataset or centroid files found in {source} or its data/ folder')
    pending, present = [], []
    for name, original in sorted(files.items()):
        target = data_destination(destination, name)
        for parent in (target.parent, target.parent.parent):
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(f'Data destination is not a local directory: {parent}')
        if target.exists() or target.is_symlink():
            if target.resolve() != original:
                raise ValueError(f'Destination already exists and points to different data: {target}')
            present.append(target)
        else:
            pending.append((target, original))
    # Validate all destinations before creating links, avoiding partial
    # setup when an existing file conflicts with the selected dataset.
    for target, original in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(original)
    return [target for target, _ in pending], present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Existing TT-Metal checkout or dataset directory')
    args = parser.parse_args()
    try:
        linked, present = link_data(args.source)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for path in linked:
        print(f'{path.relative_to(PROJECT)} -> {path.resolve()}')
    print(f'Dataset setup: {len(linked)} new links; {len(present)} already present in {PROJECT}')


if __name__ == '__main__':
    main()
