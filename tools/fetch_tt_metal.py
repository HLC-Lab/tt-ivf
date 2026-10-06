#!/usr/bin/env python3
"""Create this project's own TT-Metal checkout at the locked revision.

The default location is third_party/tt-metal inside this project, so the
experiments do not depend on any other TT-Metal folder on the host. Runtime
edits recorded in the lock (dependency_state/*.patch) are applied after the
checkout and verified against their recorded hashes. An existing directory is
only verified, never reset or replaced.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from check_tt_metal import PROJECT, check, source_changes

DEFAULT_DIRECTORY = PROJECT / 'third_party' / 'tt-metal'


def run(*args: str) -> None:
    subprocess.run(list(args), check=True)


def apply_patch(root: Path, record: dict | None, project: Path = PROJECT) -> None:
    if record is None:
        return
    patch = project / record['patch']
    if not patch.is_file():
        raise ValueError(f'Missing recorded runtime patch: {patch}')
    # --index also stages new/renamed files: the recorded patch is `git diff HEAD`
    # of the tested checkout, which includes its staged additions.
    run('git', '-C', str(root), 'apply', '--index', '--binary', '--whitespace=nowarn', str(patch))


def fetch(directory: Path, lock: dict, project: Path = PROJECT) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    run('git', 'init', '-q', str(directory))
    run('git', '-C', str(directory), 'remote', 'add', 'origin', lock['repository'])
    run('git', '-C', str(directory), 'fetch', '--depth=1', 'origin', lock['commit'])
    run('git', '-C', str(directory), 'checkout', '-q', '--detach', 'FETCH_HEAD')
    # Preserve the original version string when tags aren't available in a
    # shallow fetch. The checkout stays at the exact recorded commit.
    run('git', '-C', str(directory), '-c', 'user.name=ANN dependency setup',
        '-c', 'user.email=ann-dependency@example.invalid', 'tag', '-a', lock['describe'],
        '-m', 'Preserve the source checkout version for the pinned dependency')
    run('git', '-C', str(directory), 'submodule', 'update', '--init', '--recursive', '--',
        *[s['path'] for s in lock['submodules']])
    local = lock.get('local_changes', {})
    for submodule in lock['submodules']:
        path = directory / submodule['path']
        # A recorded submodule HEAD can differ from the superproject gitlink.
        if submodule.get('gitlink_commit', submodule['commit']) != submodule['commit']:
            run('git', '-C', str(path), 'fetch', 'origin', submodule['commit'])
            run('git', '-C', str(path), 'checkout', '-q', '--detach', submodule['commit'])
            run('git', '-C', str(path), 'submodule', 'update', '--init', '--recursive')
        apply_patch(path, local.get('submodules', {}).get(submodule['path']), project)
    apply_patch(directory, local.get('runtime'), project)


def report_differences(directory: Path, lock: dict) -> None:
    """Name the files whose state differs from the recorded runtime patch."""
    record = lock.get('local_changes', {}).get('runtime')
    if record is None or not (directory / '.git').exists():
        return
    try:
        _, names = source_changes(directory, runtime=True)
    except (OSError, subprocess.CalledProcessError):
        return
    recorded, actual = set(record.get('files', [])), set(names)
    for label, files in (('Recorded but not reproduced', recorded - actual),
                         ('Changed but not recorded', actual - recorded)):
        if files:
            print(f'{label}:\n  ' + '\n  '.join(sorted(files)), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    directory = args.directory.resolve()
    lock = json.loads((PROJECT / 'tt_metal.lock.json').read_text())
    try:
        if not directory.exists():
            fetch(directory, lock)
        check(directory, lock)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        report_differences(directory, lock)
        raise SystemExit(1)
    print(f'Pinned TT-Metal ready: {directory}')


if __name__ == '__main__':
    main()
