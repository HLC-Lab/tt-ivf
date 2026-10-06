"""Connect the ANN build to a built TT-Metal (third_party/tt-metal or an external checkout).

No checkout, library or kernel is copied or built. Temporarily add excluded
consumer probes while reconfiguring the existing build, read their resolved
requirements, then restore its original CMake project hooks. Only small import
files remain in the ANN project's .runtime directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from check_tt_metal import PROJECT, check
from snapshot_runtime import literal, snapshot


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_values(build: Path) -> dict[str, str]:
    values = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if not line or line.startswith(('#', '//')) or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.split(':', 1)[0]] = value
    return values


def check_attachment(runtime: Path, output: Path, expected_build: Path | None = None) -> dict:
    state = json.loads((output / 'external_runtime.json').read_text())
    if str(runtime.resolve()) != state['runtime']:
        raise ValueError('The selected external runtime changed')
    if expected_build is not None and str(expected_build.resolve()) != state['build']:
        raise ValueError('The selected external runtime build changed')
    if digest(PROJECT / 'tt_metal.lock.json') != state['lock_sha256']:
        raise ValueError('The runtime lock changed')
    if digest(output / 'ready.lock.json') != state['lock_sha256']:
        raise ValueError('The external runtime attachment is incomplete')
    if digest(Path(state['build']) / 'CMakeCache.txt') != state['cache_sha256']:
        raise ValueError('External runtime build options changed')
    # Reject edited/stale import files and deleted transitive libraries before
    # announcing that the attachment can be reused.
    for name, expected in state['import_sha256'].items():
        if digest(output / name) != expected:
            raise ValueError(f'External runtime import changed: {name}')
    for path in state['libraries']:
        if not Path(path).is_file():
            raise ValueError(f'External runtime dependency is missing: {path}')
    return state


def configure(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('External TT-Metal configure failed:\n' + result.stdout + result.stderr)


def inside_sources(path: Path) -> bool:
    """Only the git-ignored third_party/ folder may hold a runtime in this project."""
    project = PROJECT.resolve()
    return path.is_relative_to(project) and not path.is_relative_to(project / 'third_party')


def attach(runtime: Path, build: Path, output: Path) -> None:
    runtime, build, output = runtime.resolve(), build.resolve(), output.resolve()
    if inside_sources(runtime) or inside_sources(build):
        raise ValueError('TT-Metal must be outside the ANN sources: use third_party/tt-metal '
                         '(bash setup_tt_metal.sh) or an external checkout')
    lock = json.loads((PROJECT / 'tt_metal.lock.json').read_text())
    check(runtime, lock)
    if not (build / 'CMakeCache.txt').is_file():
        raise ValueError(f'Build TT-Metal first in {runtime}, using its build_metal.sh. '
                         f'No existing CMake build at {build}')
    cache = cache_values(build)
    if Path(cache['CMAKE_HOME_DIRECTORY']).resolve() != runtime:
        raise ValueError('The supplied build belongs to a different TT-Metal source tree')
    if not cache.get('CMAKE_BUILD_TYPE') or cache.get('CMAKE_CONFIGURATION_TYPES'):
        raise ValueError('Use an existing single-configuration TT-Metal build (normally Release)')
    configuration = cache['CMAKE_BUILD_TYPE']
    # A repeated attachment with unchanged sources/options needs no configure.
    try:
        check_attachment(runtime, output, build)
    except (OSError, ValueError, KeyError):
        pass
    else:
        print(f'Using existing external runtime attachment: {build}')
        return

    output.mkdir(parents=True, exist_ok=True)
    (output / 'ready.lock.json').unlink(missing_ok=True)
    query = build / '.cmake/api/v1/query/codemodel-v2'
    query.parent.mkdir(parents=True, exist_ok=True)
    query.touch()
    hook = output / 'consumer_hook.cmake'
    hook.write_text(
        '# Temporary configure-only hook. No probe is ever compiled.\n'
        'function(ann_record_external_requirements)\n'
        f'    set(ANN_PROJECT_ROOT {literal(str(PROJECT.resolve()))})\n'
        f'    set(TT_METAL_HOME {literal(str(runtime))})\n'
        f'    set(ANN_RUNTIME_EXPORT_DIR {literal(str(output))})\n'
        f'    include({literal(str(PROJECT / "cmake/ann_export_runtime.cmake"))})\n'
        'endfunction()\n'
        'cmake_language(DEFER CALL ann_record_external_requirements)\n')
    previous_hooks = cache.get('CMAKE_PROJECT_TOP_LEVEL_INCLUDES', '')
    hooks = ';'.join(value for value in (previous_hooks, str(hook)) if value)
    command = ['cmake', '-S', str(runtime), '-B', str(build)]
    print(f'Reading public headers and link libraries from external build: {build}', flush=True)
    try:
        # Retain the existing toolchain, cache, third-party versions and all
        # build options. CMAKE_PROJECT_TOP_LEVEL_INCLUDES applies only at root.
        configure(command + [f'-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES:STRING={hooks}'])
        libraries = snapshot(build, configuration, output)
    finally:
        # The external build must not retain a dependency on the ANN project.
        # Restore even when configuration/snapshot generation fails.
        configure(command + [f'-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES:STRING={previous_hooks}'])
        hook.unlink(missing_ok=True)
    check(runtime, lock)

    # All runtime libraries must remain outside this application. The compiled
    # requirements contain absolute paths; retain them for preflight checks.
    for path in libraries:
        if inside_sources(Path(path).resolve()):
            raise ValueError(f'Runtime dependency resides inside the ANN sources: {path}')
    imports = ('ann-metalium-config.cmake', 'ann-metalium-targets.cmake', 'ann-runtime-library.txt')
    state = {
        'runtime': str(runtime), 'build': str(build), 'configuration': configuration,
        'lock_sha256': digest(PROJECT / 'tt_metal.lock.json'),
        'cache_sha256': digest(build / 'CMakeCache.txt'),
        'import_sha256': {name: digest(output / name) for name in imports},
        'libraries': libraries,
        'c_compiler': cache_values(build).get('CMAKE_C_COMPILER'),
        'cxx_compiler': cache_values(build)['CMAKE_CXX_COMPILER'],
    }
    (output / 'external_runtime.json').write_text(json.dumps(state, indent=2) + '\n')
    (output / 'ready.lock.json').write_bytes((PROJECT / 'tt_metal.lock.json').read_bytes())
    print(f'Connected to {runtime}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--output', type=Path, default=PROJECT / '.runtime')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compiler', choices=('C', 'CXX'))
    args = parser.parse_args()
    try:
        if args.check or args.compiler:
            state = check_attachment(args.runtime, args.output, args.build)
            if args.compiler:
                value = state['c_compiler' if args.compiler == 'C' else 'cxx_compiler']
                if not value:
                    raise ValueError(f'The runtime did not record a {args.compiler} compiler')
                print(value)
        else:
            attach(args.runtime, args.build or args.runtime / 'build_Release', args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
