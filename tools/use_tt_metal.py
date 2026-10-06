#!/usr/bin/env python3
"""Pin this ANN project to an existing TT-Metal checkout, without changing it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_tt_metal import PROJECT, check, git, source_changes


def use_checkout(checkout: Path, project: Path = PROJECT, record_local_changes: bool = False) -> dict:
    checkout = checkout.resolve()
    lock_path = project / 'tt_metal.lock.json'
    previous_text = lock_path.read_text()
    previous = json.loads(previous_text)
    if not (checkout / 'CMakeLists.txt').is_file():
        raise ValueError(f'Missing TT-Metal checkout: {checkout}')
    if Path(git(checkout, 'rev-parse', '--show-toplevel')).resolve() != checkout:
        raise ValueError(f'{checkout} is not the root of a TT-Metal Git checkout')

    # Read gitlinks from the chosen revision. Older versions also need tt_llk;
    # retaining the exported UMD/Tracy hashes would still reject this checkout.
    submodules = []
    for line in git(checkout, 'ls-tree', '-r', 'HEAD', '--', 'tt_metal/third_party').splitlines():
        metadata, path = line.split('\t', 1)
        mode, kind, commit = metadata.split()
        if mode == '160000' and kind == 'commit':
            submodules.append({'path': path, 'commit': commit})
    required = {'tt_metal/third_party/umd', 'tt_metal/third_party/tracy'}
    if not required.issubset({entry['path'] for entry in submodules}):
        raise ValueError('Checkout is missing the TT-Metal UMD/Tracy submodule entries')
    lock = dict(previous, commit=git(checkout, 'rev-parse', 'HEAD'),
                describe=git(checkout, 'describe', '--tags', '--always'), submodules=submodules)
    lock.pop('local_changes', None)
    try:
        lock['repository'] = git(checkout, 'remote', 'get-url', 'origin')
    except subprocess.CalledProcessError:
        pass  # Keep the previous fetch URL when the local checkout has no remote.

    snapshots = {}
    if record_local_changes:
        # Use the effective runtime already on the host, recording a changed
        # submodule HEAD separately from its superproject gitlink.
        for entry in submodules:
            module = checkout / entry['path']
            if not (module / '.git').exists():
                continue  # check() gives the actionable initialization error.
            actual = git(module, 'rev-parse', 'HEAD')
            if actual != entry['commit']:
                entry['gitlink_commit'] = entry['commit']
                entry['commit'] = actual
        patch, names = source_changes(checkout, runtime=True)
        local = {}

        def describe_patch(payload: bytes, files: list[str]) -> dict:
            digest = hashlib.sha256(payload).hexdigest()
            relative = f'dependency_state/{digest}.patch'
            snapshots[relative] = payload
            return {'sha256': digest, 'files': files, 'patch': relative}

        if patch:
            local['runtime'] = describe_patch(patch, names)
        modules = {}
        for entry in submodules:
            module = checkout / entry['path']
            if (module / '.git').exists():
                patch, names = source_changes(module)
                if patch:
                    modules[entry['path']] = describe_patch(patch, names)
        if modules:
            local['submodules'] = modules
        if local:
            lock['local_changes'] = local

    # Validate before writing or mutating anything in the project. The runtime
    # remains read only; subsequent checks require the recorded patch hashes.
    check(checkout, lock)
    new_text = json.dumps(lock, indent=2) + '\n'
    if new_text == previous_text:
        return lock
    backup = lock_path.with_name(lock_path.name + '.before-' + previous['commit'])
    if backup.exists() and backup.read_text() != previous_text:
        backup = backup.with_name(backup.name + '-' + hashlib.sha256(previous_text.encode()).hexdigest()[:12])
    if backup.exists() and backup.read_text() != previous_text:
        raise ValueError(f'Conflicting lock backup: {backup}')
    for name, payload in snapshots.items():
        target = project / name
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f'Conflicting runtime patch: {target}')
    for name, payload in snapshots.items():
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(payload)
    if not backup.exists():
        with backup.open('x') as handle:
            handle.write(previous_text)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=project, prefix='.tt-metal-lock-',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(new_text)
        temporary.replace(lock_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--record-local-changes', action='store_true',
                        help='Record tracked runtime/build edits and actual runtime submodule revisions')
    args = parser.parse_args()
    try:
        lock = use_checkout(args.checkout, record_local_changes=args.record_local_changes)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
    print(f'ANN dependency pinned to {lock["describe"]} at {lock["commit"]}')
    print('The TT-Metal checkout was not modified.')
    if lock.get('local_changes'):
        entries = [('runtime', lock['local_changes'].get('runtime'))]
        entries.extend(lock['local_changes'].get('submodules', {}).items())
        for name, state in entries:
            if state:
                print(f'Recorded {name}: {len(state["files"])} tracked files; patch: {state["patch"]}')
    print('This verifies source revisions; API compatibility still requires a build.')


if __name__ == '__main__':
    main()
