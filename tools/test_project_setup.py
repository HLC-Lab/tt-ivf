"""Check dependency pin enforcement, the external CMake graph, and shell argument handling."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock

from check_tt_metal import PROJECT, check, git, source_changes
from link_data import link_data
from use_tt_metal import use_checkout
from fetch_tt_metal import fetch
from snapshot_runtime import snapshot


class ProjectSetupTests(unittest.TestCase):
    def test_external_default_reuses_libraries_and_preserves_runtime_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            project, runtime = root / 'ann project', root / 'external metal'
            project.mkdir()
            runtime.mkdir()
            names = ('CMakeLists.txt', 'env.sh', 'build.sh', 'run.sh', 'prepare_runtime.sh',
                     'dependencies/CMakeLists.txt', 'cmake/ann_import_runtime.cmake',
                     'cmake/ann_export_runtime.cmake', 'cmake/ann_examples.cmake',
                     'tools/check_tt_metal.py', 'tools/attach_runtime.py', 'tools/snapshot_runtime.py',
                     'ann_ivf_energy/CMakeLists.txt')
            for name in names:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(PROJECT / name, target)
            (project / 'cmake/ann_examples.cmake').write_text('set(ANN_DEFAULT_EXAMPLES ann_ivf_energy)\n')
            (runtime / 'tt_metal').mkdir()
            (runtime / 'api.hpp').write_text('int fixture_runtime();\n')
            (runtime / 'forced.hpp').write_text('#define ANN_FORCED_HEADER 7\n')
            (runtime / 'metal.cpp').write_text('int fixture_runtime() { return 37; }\n')
            (runtime / 'stl.cpp').write_text('int fixture_stl() { return 0; }\n')
            (runtime / 'dependency.cpp').write_text('int fixture_dependency() { return 5; }\n')
            (runtime / 'unrelated.cpp').write_text('#error unrelated library must not build\n')
            (runtime / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.24)
project(FixtureRuntime LANGUAGES C CXX)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/do-not-configure")
    message(FATAL_ERROR "The ANN build reconfigured runtime sources")
endif()
set(ANN_FIXTURE_SETTING original CACHE STRING "Preserve this existing option")
set(TT_ENABLE_LIGHT_METAL_TRACE ON CACHE BOOL "")
set(ENABLE_LIBCXX OFF CACHE BOOL "")
add_library(public_headers INTERFACE)
target_compile_definitions(public_headers INTERFACE ANN_RUNTIME_SETTING=19)
target_compile_options(public_headers INTERFACE "SHELL:-include '${CMAKE_CURRENT_SOURCE_DIR}/forced.hpp'")
target_include_directories(public_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
add_library(fixture_dependency STATIC dependency.cpp)
set_target_properties(fixture_dependency PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/public libraries")
add_library(tt_stl SHARED stl.cpp)
add_library(TT::STL ALIAS tt_stl)
target_link_libraries(tt_stl INTERFACE public_headers)
add_library(tt_metal SHARED metal.cpp)
add_library(TT::Metalium ALIAS tt_metal)
target_link_libraries(tt_metal PUBLIC tt_stl public_headers fixture_dependency)
set_target_properties(tt_metal PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE YES)
add_library(unrelated EXCLUDE_FROM_ALL STATIC unrelated.cpp)
''')
            subprocess.run(['git', 'init', '-q', str(runtime)], check=True)
            subprocess.run(['git', '-C', str(runtime), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(runtime), '-c', 'user.name=Fixture',
                            '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture'], check=True)
            (project / 'tt_metal.lock.json').write_text(json.dumps({
                'commit': git(runtime, 'rev-parse', 'HEAD'), 'describe': 'fixture', 'submodules': []}))
            example = project / 'ann_ivf_energy'
            (example / 'kernels').mkdir()
            (example / 'kernels/fixture.cpp').write_text('// kernel lookup fixture\n')
            (example / 'ivf_tt.cpp').write_text('// fixture\n')
            (example / 'main.cpp').write_text('''#include "api.hpp"
#include <cstdlib>
#include <iostream>
#include <filesystem>
#if ANN_RUNTIME_SETTING != 19 || ANN_FORCED_HEADER != 7 || TT_ENABLE_LIGHT_METAL_TRACE != 1
#error Lost external runtime requirements
#endif
int fixture_dependency();
int main() {
    std::cout << std::getenv("TT_METAL_RUNTIME_ROOT") << "\\n"
              << std::getenv("TT_METAL_KERNEL_PATH") << "\\n";
    const auto kernel = std::filesystem::path(std::getenv("TT_METAL_KERNEL_PATH")) /
                        "ann_ivf_energy/kernels/fixture.cpp";
    return fixture_runtime() == 37 && fixture_dependency() == 5 &&
           std::filesystem::is_regular_file(kernel) ? 0 : 1;
}
''')
            # Retain a pre-existing CMake project hook, including a path with
            # spaces. Attachment must append its probe hook and restore ours.
            original_hook = root / 'original hook.cmake'
            original_hook.write_text('set(ANN_ORIGINAL_HOOK loaded CACHE STRING "fixture")\n')
            build = runtime / 'build_Release'
            environment = dict(os.environ, CC='cc', CXX='c++', ANN_CMAKE_GENERATOR='Unix Makefiles')
            subprocess.run(['cmake', '-S', str(runtime), '-B', str(build), '-G', 'Unix Makefiles',
                            '-DCMAKE_BUILD_TYPE=Release', '-DANN_FIXTURE_SETTING=keep-me',
                            f'-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={original_hook}'],
                           env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['cmake', '--build', str(build), '--target', 'tt_metal'],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            libraries = [path for path in build.rglob('*') if path.suffix in ('.a', '.so', '.dylib')]
            import hashlib
            library_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries}
            original_status = git(runtime, 'status', '--porcelain', '--untracked-files=no')
            # An old internal copy must not override the explicit external env.
            old_internal = project / 'dependencies/tt-metal'
            old_internal.mkdir()
            (old_internal / 'keep-until-migrated').write_text('old checkout fixture')
            (project / 'glove_train.bin').write_bytes(b'keep dataset')
            (project / 'results.csv').write_text('keep measurements\n')
            # A login-shell TT_METAL_HOME must not select the runtime; only an
            # explicit env.sh choice (ANN_TT_METAL_HOME) or third_party/ does.
            environment['TT_METAL_HOME'] = str(root / 'unrelated login-shell metal')
            environment['ANN_TT_METAL_HOME'] = str(runtime)
            prepared = subprocess.run(['bash', str(project / 'prepare_runtime.sh')],
                                      env=environment, capture_output=True, text=True)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            from attach_runtime import cache_values
            values = cache_values(build)
            self.assertEqual(values['CMAKE_PROJECT_TOP_LEVEL_INCLUDES'], str(original_hook))
            self.assertEqual(values['ANN_FIXTURE_SETTING'], 'keep-me')
            self.assertEqual(values['ANN_ORIGINAL_HOOK'], 'loaded')
            self.assertEqual(git(runtime, 'status', '--porcelain', '--untracked-files=no'), original_status)
            for path, expected in library_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            self.assertFalse(list(build.rglob('ann-consumer-probe.cpp.o')))
            self.assertFalse((project / 'dependencies/metalium-build').exists())
            requirements = (project / '.runtime/ann-metalium-targets.cmake').read_text()
            self.assertNotIn(str(old_internal), requirements)
            self.assertNotIn('unrelated', requirements)
            help_text = subprocess.check_output(['cmake', '--build', str(build), '--target', 'help'], text=True)
            self.assertNotIn('ann_probe_', help_text)

            # Remove the old internal copy and prohibit further runtime source
            # configuration. Default build.sh must compile only ANN sources,
            # with the external build's compiler selected automatically.
            shutil.rmtree(old_internal)
            (runtime / 'do-not-configure').touch()
            environment.pop('CC')
            environment.pop('CXX')
            environment.pop('ANN_METALIUM_MODE', None)
            built = subprocess.run(['bash', str(project / 'build.sh'), 'ann_ivf_energy'],
                                   env=environment, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            cache = project / 'build_Release/CMakeCache.txt'
            with cache.open('a') as handle:
                handle.write('\nANN_EXAMPLES:STRING=ann_ivf;ann_ivf_energy\n')
                handle.write('ANN_SOURCE_LAYOUT:INTERNAL=nested\n')
            migrated = subprocess.run(['bash', str(project / 'build.sh'), 'ann_ivf_energy'],
                                      env=environment, capture_output=True, text=True)
            self.assertEqual(migrated.returncode, 0, migrated.stdout + migrated.stderr)
            self.assertEqual(cache_values(project / 'build_Release')['ANN_EXAMPLES'], 'ann_ivf_energy')
            self.assertEqual(cache_values(project / 'build_Release')['ANN_SOURCE_LAYOUT'], 'flat')
            self.assertTrue((project / 'build_Release/bin/ann_ivf_energy').is_file())
            run = subprocess.run(['bash', str(project / 'run.sh'), 'ann_ivf_energy'],
                                 cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertTrue(run.stdout.endswith(f'{runtime}\n{project}\n'), run.stdout)
            self.assertEqual((project / 'glove_train.bin').read_bytes(), b'keep dataset')
            self.assertEqual((project / 'results.csv').read_text(), 'keep measurements\n')
            self.assertFalse(old_internal.exists())
            self.assertFalse((project / 'build_Release/metalium').exists())
            for path, expected in library_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            # A changed external cache must fail closed until reattachment.
            with (build / 'CMakeCache.txt').open('a') as handle:
                handle.write('\nANN_FIXTURE_CHANGED:BOOL=ON\n')
            rejected = subprocess.run(['cmake', '-S', str(project), '-B', str(project / 'build_Release')],
                                      env=environment, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('External runtime build options changed', rejected.stderr)
            # A missing prebuilt library must not trigger a runtime rebuild or
            # leave our project hook installed in the external build.
            (runtime / 'do-not-configure').unlink()
            next(path for path in libraries if path.suffix == '.a').unlink()
            failed = subprocess.run(['bash', str(project / 'prepare_runtime.sh')],
                                    env=environment, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Consumer library was not built', failed.stderr)
            self.assertEqual(cache_values(build)['CMAKE_PROJECT_TOP_LEVEL_INCLUDES'], str(original_hook))
            self.assertFalse((project / '.runtime/ready.lock.json').exists())
            self.assertNotIn('ann_probe_', subprocess.check_output(
                ['cmake', '--build', str(build), '--target', 'help'], text=True))

    def test_snapshot_preserves_consumer_flags_when_baseline_is_not_a_prefix(self):
        # These are CMake file API command fragments: its synthesized PIC/PIE
        # flags can split the default flag prefix before the standard flag.
        # Link flags may also appear before or between configuration defaults.
        # Test the full snapshot, including ordered flags with path operands.
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve()
            reply = build / '.cmake/api/v1/reply'
            reply.mkdir(parents=True)
            library = build / 'libtt_metal.so'
            library.write_bytes(b'fixture shared library')
            header = build / 'header with spaces.hpp'
            header.write_text('// fixture\n')

            def write_json(name, value):
                (reply / name).write_text(json.dumps(value))

            write_json('index-fixture.json', {'objects': [
                {'kind': 'codemodel', 'version': {'major': 2}, 'jsonFile': 'codemodel.json'}]})
            write_json('codemodel.json', {'configurations': [{'name': 'Release', 'targets': [
                {'name': f'ann_probe_{api}', 'jsonFile': f'{api}.json'} for api in ('Baseline', 'Metalium', 'STL')]}]})
            write_json('Baseline.json', {'name': 'ann_probe_Baseline', 'link': {'commandFragments': [
                {'role': 'flags', 'fragment': '-O3 -DNDEBUG'}]}})
            for api in ('Metalium', 'STL'):
                write_json(f'{api}.json', {'name': f'ann_probe_{api}',
                    'compileGroups': [{'language': 'CXX', 'languageStandard': {'standard': '20'},
                        'compileCommandFragments': [{'fragment': '-O3 -DNDEBUG -fPIE -std=c++20'},
                                                    {'fragment': f'-include "{header}" -fno-omit-frame-pointer'}]}],
                    'link': {'commandFragments': [
                        {'role': 'flags', 'fragment': '-fuse-ld=lld -O3 -Wl,--as-needed -DNDEBUG'},
                        {'role': 'libraries', 'fragment': f'"{library}"'},
                        {'role': 'libraries', 'fragment': '-lm'}]}})
            (build / 'ann-runtime-metadata.cmake').write_text('# fixture\n')
            (build / 'ann-runtime-library-Release.txt').write_text(str(library)+'\n')
            snapshot(build, 'Release')
            output = (build / 'ann-metalium-targets.cmake').read_text()
            self.assertIn('-O3 -DNDEBUG -fPIE -std=c++20 -include', output)
            self.assertIn(f"'{header}' -fno-omit-frame-pointer", output)
            self.assertIn('-fuse-ld=lld -O3 -Wl,--as-needed -DNDEBUG', output)
            self.assertIn('INTERFACE_LINK_LIBRARIES', output)
            self.assertIn(str(library)+';-lm', output)
            self.assertEqual((build / 'ann-runtime-library.txt').read_text(), str(library)+'\n')





    def test_adopt_existing_revision_and_runtime_submodules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency, project = root / 'existing metal', root / 'ann project'
            dependency.mkdir()
            project.mkdir()
            subprocess.run(['git', 'init', '-q', str(dependency)], check=True)

            def commit(repository, message):
                subprocess.run(['git', '-C', str(repository), '-c', 'user.name=Fixture',
                                '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', message], check=True)

            expected_submodules = []
            modules = ''
            for name in ('umd', 'tracy', 'tt_llk'):
                path = f'tt_metal/third_party/{name}'
                module = dependency / path
                module.mkdir(parents=True)
                # The older tt_llk submodule supplies headers, without CMake.
                module_file = 'llk_fixture.h' if name == 'tt_llk' else 'CMakeLists.txt'
                (module / module_file).write_text('// fixture module\n')
                subprocess.run(['git', 'init', '-q', str(module)], check=True)
                subprocess.run(['git', '-C', str(module), 'add', module_file], check=True)
                commit(module, f'{name} fixture')
                revision = git(module, 'rev-parse', 'HEAD')
                subprocess.run(['git', '-C', str(dependency), 'update-index', '--add',
                                '--cacheinfo', f'160000,{revision},{path}'], check=True)
                expected_submodules.append({'path': path, 'commit': revision})
                modules += f'[submodule "{path}"]\n\tpath = {path}\n\turl = example.invalid/{name}\n'
            (dependency / '.gitmodules').write_text(modules)
            (dependency / 'CMakeLists.txt').write_text('# fixture runtime\n')
            runtime = dependency / 'tt_metal/runtime_marker.cpp'
            runtime.write_text('// runtime\n')
            subprocess.run(['git', '-C', str(dependency), 'add', '.gitmodules',
                            'CMakeLists.txt', 'tt_metal/runtime_marker.cpp'], check=True)
            commit(dependency, 'Existing TT-Metal version')
            subprocess.run(['git', '-C', str(dependency), 'tag', 'v0.66.0-fixture'], check=True)
            actual = git(dependency, 'rev-parse', 'HEAD')
            previous = {'commit': '0' * 40, 'describe': 'export fixture',
                        'repository': 'https://example.invalid/tt-metal.git', 'submodules': []}
            lock_path = project / 'tt_metal.lock.json'
            old_text = json.dumps(previous, indent=2) + '\n'
            lock_path.write_text(old_text)
            before_status = git(dependency, 'status', '--porcelain')
            lock = use_checkout(dependency, project)
            self.assertEqual(lock['commit'], actual)
            self.assertEqual(lock['describe'], 'v0.66.0-fixture')
            self.assertCountEqual(lock['submodules'], expected_submodules)
            self.assertEqual(git(dependency, 'rev-parse', 'HEAD'), actual)
            self.assertEqual(git(dependency, 'status', '--porcelain'), before_status)
            self.assertEqual(lock_path.with_name(lock_path.name + '.before-' + previous['commit']).read_text(), old_text)
            self.assertEqual(json.loads(lock_path.read_text()), lock)
            check(dependency, lock)
            self.assertEqual(use_checkout(dependency, project), lock)

            # A source edit and a wrong submodule must fail before changing any lock.
            accepted_text = lock_path.read_text()
            runtime.write_text('// uncommitted change\n')
            with self.assertRaisesRegex(ValueError, 'local changes'):
                use_checkout(dependency, project)
            self.assertEqual(lock_path.read_text(), accepted_text)
            # Explicitly recording edits preserves the runtime and stores its
            # patch. Repeated builds verify that same state, rather than bypassing checks.
            dirty_status = git(dependency, 'status', '--porcelain')
            recorded = use_checkout(dependency, project, record_local_changes=True)
            self.assertEqual(git(dependency, 'status', '--porcelain'), dirty_status)
            self.assertEqual(runtime.read_text(), '// uncommitted change\n')
            state = recorded['local_changes']['runtime']
            self.assertEqual(state['files'], ['tt_metal/runtime_marker.cpp'])
            self.assertIn(b'uncommitted change', (project / state['patch']).read_bytes())
            check(dependency, recorded)
            self.assertEqual(use_checkout(dependency, project, record_local_changes=True), recorded)
            runtime.write_text('// a later edit\n')
            with self.assertRaisesRegex(ValueError, 'changed after'):
                check(dependency, recorded)
            # Re-recording at the same commit creates a distinct lock backup.
            rerecorded = use_checkout(dependency, project, record_local_changes=True)
            self.assertNotEqual(rerecorded['local_changes']['runtime']['sha256'], state['sha256'])
            check(dependency, rerecorded)

            runtime.write_text('// runtime\n')
            accepted_text = lock_path.read_text()
            module = dependency / 'tt_metal/third_party/umd'
            (module / 'CMakeLists.txt').write_text('# different module revision\n')
            subprocess.run(['git', '-C', str(module), 'add', 'CMakeLists.txt'], check=True)
            commit(module, 'Wrong module revision')
            with self.assertRaises(ValueError):
                use_checkout(dependency, project)
            self.assertEqual(lock_path.read_text(), accepted_text)
            # Explicit mode also records the effective submodule commit and its edits.
            (module / 'CMakeLists.txt').write_text('# local submodule edit\n')
            subprocess.run(['git', '-C', str(module), 'add', 'CMakeLists.txt'], check=True)
            adopted = use_checkout(dependency, project, record_local_changes=True)
            entry = next(entry for entry in adopted['submodules'] if entry['path'].endswith('/umd'))
            self.assertEqual(entry['commit'], git(module, 'rev-parse', 'HEAD'))
            self.assertEqual(entry['gitlink_commit'], expected_submodules[0]['commit'])
            self.assertNotIn('runtime', adopted['local_changes'])
            state = adopted['local_changes']['submodules'][entry['path']]
            self.assertIn(b'local submodule edit', (project / state['patch']).read_bytes())
            check(dependency, adopted)
            (module / 'CMakeLists.txt').write_text('# another edit\n')
            with self.assertRaisesRegex(ValueError, 'changed after'):
                check(dependency, adopted)

    def test_fetch_reproduces_recorded_runtime_without_the_original_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            upstream, project = root / 'upstream metal', root / 'ann project'
            project.mkdir()

            def commit(repository, message):
                subprocess.run(['git', '-C', str(repository), '-c', 'user.name=Fixture',
                                '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', message], check=True)

            # Submodule remotes: umd's working HEAD is one commit past its gitlink.
            gitlinks, modules = {}, ''
            for name in ('umd', 'tracy'):
                remote = root / f'{name} remote'
                remote.mkdir()
                subprocess.run(['git', 'init', '-q', str(remote)], check=True)
                (remote / 'CMakeLists.txt').write_text(f'# {name}\n')
                subprocess.run(['git', '-C', str(remote), 'add', 'CMakeLists.txt'], check=True)
                commit(remote, name)
                gitlinks[name] = git(remote, 'rev-parse', 'HEAD')
                modules += (f'[submodule "tt_metal/third_party/{name}"]\n'
                            f'\tpath = tt_metal/third_party/{name}\n\turl = {remote.as_uri()}\n')
            subprocess.run(['git', 'init', '-q', str(upstream)], check=True)
            (upstream / '.gitmodules').write_text(modules)
            (upstream / 'CMakeLists.txt').write_text('# fixture runtime\n')
            (upstream / 'tt_metal').mkdir()
            (upstream / 'tt_metal/runtime.cpp').write_text('// runtime\n')
            subprocess.run(['git', '-C', str(upstream), 'add', '.'], check=True)
            for name, revision in gitlinks.items():
                subprocess.run(['git', '-C', str(upstream), 'update-index', '--add', '--cacheinfo',
                                f'160000,{revision},tt_metal/third_party/{name}'], check=True)
            commit(upstream, 'Runtime release')
            umd_remote = root / 'umd remote'
            (umd_remote / 'CMakeLists.txt').write_text('# umd newer\n')
            subprocess.run(['git', '-C', str(umd_remote), 'add', 'CMakeLists.txt'], check=True)
            commit(umd_remote, 'newer umd')

            # The tested host checkout: a newer umd HEAD plus uncommitted edits.
            environment = {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'protocol.file.allow',
                           'GIT_CONFIG_VALUE_0': 'always'}
            original = root / 'host metal'
            with unittest.mock.patch.dict(os.environ, environment):
                subprocess.run(['git', 'clone', '-q', '--recurse-submodules', upstream.as_uri(),
                                str(original)], check=True)
                umd = original / 'tt_metal/third_party/umd'
                subprocess.run(['git', '-C', str(umd), 'checkout', '-q', git(umd_remote, 'rev-parse', 'HEAD')],
                               check=True)
                (umd / 'CMakeLists.txt').write_text('# umd local edit\n')
                (original / 'tt_metal/runtime.cpp').write_text('// host runtime edit\n')
                # A new file staged on the host is part of `git diff HEAD` there.
                (original / 'tt_metal/added.cpp').write_text('// staged new file\n')
                subprocess.run(['git', '-C', str(original), 'add', 'tt_metal/added.cpp'], check=True)
                (project / 'tt_metal.lock.json').write_text(json.dumps(
                    {'commit': '0' * 40, 'describe': 'export', 'submodules': []}))
                lock = use_checkout(original, project, record_local_changes=True)
                self.assertEqual(lock['repository'], upstream.as_uri())

                copy = project / 'third_party/tt-metal'
                fetch(copy, lock, project)
            check(copy, lock)
            self.assertEqual((copy / 'tt_metal/runtime.cpp').read_text(), '// host runtime edit\n')
            self.assertEqual((copy / 'tt_metal/added.cpp').read_text(), '// staged new file\n')
            self.assertEqual((copy / 'tt_metal/third_party/umd/CMakeLists.txt').read_text(), '# umd local edit\n')
            # The original host checkout is not needed after the fetch.
            shutil.rmtree(original)
            check(copy, lock)

    def test_dataset_links_and_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / 'old checkout', root / 'tt-ann-ivf'
            source.mkdir()
            destination.mkdir()
            (source / 'data').mkdir()
            for name in ('glove-100-angular_train.bin', 'glove-100-angular_queries.bin',
                         'glove-100-angular_neighbors.bin', 'centroids-glove-100-angular-512.bin'):
                (source / name).write_bytes(b'fixture')
            # Prefer the existing root file, as the original executables did.
            (source / 'data/glove-100-angular_train.bin').write_bytes(b'different')
            (source / 'data/glove-100-angular.hdf5').write_bytes(b'hdf5')
            (source / 'unrelated.txt').write_text('unrelated')
            linked, present = link_data(source, destination)
            self.assertEqual(len(linked), 5)
            self.assertEqual(present, [])
            self.assertEqual((destination / 'data/datasets/glove-100-angular_train.bin').resolve(),
                             (source / 'glove-100-angular_train.bin').resolve())
            self.assertFalse((destination / 'unrelated.txt').exists())
            self.assertEqual(link_data(source, destination), ([], sorted(linked)))
            # A collision fails before any new links are written.
            (source / 'a_new_queries.bin').write_bytes(b'new')
            target = destination / 'data/datasets/z_conflict_train.bin'
            target.write_bytes(b'keep mine')
            (source / target.name).write_bytes(b'other')
            with self.assertRaisesRegex(ValueError, 'different data'):
                link_data(source, destination)
            self.assertEqual(target.read_bytes(), b'keep mine')
            self.assertFalse((destination / 'data/datasets/a_new_queries.bin').exists())

    def test_pin_rejects_wrong_revision_changes_and_nested_checkouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / 'metal dependency'
            dependency.mkdir()
            (dependency / 'tt_metal').mkdir()
            (dependency / 'tt_metal/runtime_marker.cpp').write_text('// original\n')
            (dependency / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.24)
project(FakeMetalium LANGUAGES CXX)
if(BUILD_PROGRAMMING_EXAMPLES OR WITH_PYTHON_BINDINGS OR TT_METAL_BUILD_TESTS OR TTNN_BUILD_TESTS)
    message(FATAL_ERROR "ANN must disable unrelated upstream components")
endif()
add_library(fake_metal INTERFACE)
add_library(TT::Metalium ALIAS fake_metal)
add_library(fake_stl INTERFACE)
add_library(TT::STL ALIAS fake_stl)
set(TT_ENABLE_LIGHT_METAL_TRACE OFF CACHE BOOL "")
add_executable(unrelated_ttnn unrelated.cpp)
''')
            (dependency / 'unrelated.cpp').write_text('#error Unrelated TTNN target must not build\n')
            subprocess.run(['git', 'init', '-q', str(dependency)], check=True)
            subprocess.run(['git', '-C', str(dependency), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(dependency), '-c', 'user.name=Fixture',
                            '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture'], check=True)
            lock = {'commit': git(dependency, 'rev-parse', 'HEAD'), 'describe': 'fixture', 'submodules': []}
            check(dependency, lock)
            with self.assertRaisesRegex(ValueError, 'revision mismatch'):
                check(dependency, dict(lock, commit='0' * 40))
            (dependency / 'tt_metal/runtime_marker.cpp').write_text('// modified runtime\n')
            with self.assertRaisesRegex(ValueError, 'local changes'):
                check(dependency, lock)
            (dependency / 'tt_metal/runtime_marker.cpp').write_text('// original\n')
            # A folder nested in some other Git checkout is not itself a dependency.
            nested = dependency / 'nested'
            nested.mkdir()
            (nested / 'CMakeLists.txt').write_text('')
            with self.assertRaisesRegex(ValueError, 'not the root'):
                check(nested, lock)
            locked_submodule = dict(lock, submodules=[{'path': 'tt_metal/third_party/umd', 'commit': '0' * 40}])
            with self.assertRaisesRegex(ValueError, 'Missing submodule'):
                check(dependency, locked_submodule)



if __name__ == '__main__':
    unittest.main()
