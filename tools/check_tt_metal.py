#!/usr/bin/env python3
"""Require the exact source and submodule revisions recorded by the exporter."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = ['CMakeLists.txt', '.gitmodules', 'cmake', 'third_party', 'tt_metal',
                 'tt_stl', 'tools', 'ttnn', ':!tt_metal/programming_examples']


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                   stderr=subprocess.PIPE).strip()


def source_changes(root: Path, runtime: bool = False) -> tuple[bytes, list[str]]:
    paths = RUNTIME_PATHS if runtime else []
    # Ignore gitlink noise in the parent: each runtime submodule's commit and
    # tracked changes are verified separately below. Include staged edits too.
    base = ['git', '-C', str(root), 'diff', '--no-ext-diff', '--no-textconv',
            '--ignore-submodules=all', 'HEAD']
    patch = subprocess.check_output([*base, '--binary', '--full-index', '--no-color',
                                     '--src-prefix=a/', '--dst-prefix=b/', '--', *paths],
                                    stderr=subprocess.PIPE)
    names = subprocess.check_output([*base, '--name-only', '-z', '--', *paths],
                                    stderr=subprocess.PIPE)
    return patch, [name.decode('utf-8', errors='surrogateescape')
                   for name in names.split(b'\0') if name]


def verify_changes(root: Path, recorded: dict | None, runtime: bool = False) -> None:
    patch, names = source_changes(root, runtime)
    if recorded is None:
        if patch:
            files = '\n  '.join(names)
            raise ValueError(f'TT-Metal runtime/build sources have local changes in {root}:\n  {files}\n'
                             'Record the current sources with: python3 tools/use_tt_metal.py '
                             f'{shlex.quote(str(root if runtime else root.parents[2]))} --record-local-changes')
    elif hashlib.sha256(patch).hexdigest() != recorded['sha256']:
        raise ValueError(f'TT-Metal sources changed after they were recorded: {root}. '
                         'Run tools/use_tt_metal.py again with --record-local-changes to record the new state.')


def check(root: Path, lock: dict, metadata_only: bool = False) -> None:
    root = root.resolve()
    if not (root / 'CMakeLists.txt').is_file():
        raise ValueError(f'Missing TT-Metal checkout: {root}. Run: bash setup_tt_metal.sh')
    if Path(git(root, 'rev-parse', '--show-toplevel')).resolve() != root:
        raise ValueError(f'{root} is not the root of a TT-Metal Git checkout')
    actual = git(root, 'rev-parse', 'HEAD')
    if actual != lock['commit']:
        raise ValueError(f'TT-Metal revision mismatch: expected {lock["commit"]}, found {actual}. '
                         'To use this existing version, run: python3 tools/use_tt_metal.py '
                         f'{shlex.quote(str(root))}. Alternatively: bash setup_tt_metal.sh')
    local = lock.get('local_changes', {})
    verify_changes(root, local.get('runtime'), runtime=True)
    if metadata_only:
        return
    for submodule in lock['submodules']:
        path = root / submodule['path']
        needs_cmake = path.name in {'umd', 'tracy'}
        if not (path / '.git').exists() or (needs_cmake and not (path / 'CMakeLists.txt').is_file()):
            raise ValueError(f'Missing submodule {submodule["path"]}. Run: '
                             f'git -C "{root}" submodule update --init --recursive -- '
                             + shlex.join(entry['path'] for entry in lock['submodules']))
        if Path(git(path, 'rev-parse', '--show-toplevel')).resolve() != path.resolve():
            raise ValueError(f'Uninitialized Git checkout for {submodule["path"]}')
        if git(path, 'rev-parse', 'HEAD') != submodule['commit']:
            raise ValueError(f'Unexpected revision for {submodule["path"]}; initialize it at the pinned commit')
        verify_changes(path, local.get('submodules', {}).get(submodule['path']))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--metadata-only', action='store_true', help='Skip initialized submodule checks')
    args = parser.parse_args()
    try:
        lock = json.loads((PROJECT / 'tt_metal.lock.json').read_text())
        check(args.checkout, lock, args.metadata_only)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
    print(f'TT-Metal {lock["describe"]} at {lock["commit"]} verified')
    if lock.get('local_changes'):
        print('Recorded tracked runtime changes also verified')


if __name__ == '__main__':
    main()
