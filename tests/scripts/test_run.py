"""Exercise the local Linux runner with fake build tools and no GPU runtime."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]

FAKE_BACKEND = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
config = Path(arguments[arguments.index('--config') + 1])
with open(os.environ['RUN_TEST_EVENTS'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps({'tool': 'backend', 'argv': arguments,
        'executable': str(Path(sys.argv[0]).resolve()),
        'configText': config.read_text(encoding='utf-8')}) + '\\n')
print('fake backend stdout', flush=True)
print('fake backend stderr', file=sys.stderr, flush=True)
raise SystemExit(int(os.environ.get('RUN_TEST_BACKEND_EXIT', '0')))
'''

FAKE_CMAKE = '''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
stage = 'build' if '--build' in arguments else 'configure'
with open(os.environ['RUN_TEST_EVENTS'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps({'tool': 'cmake', 'stage': stage, 'argv': arguments}) + '\\n')
if os.environ.get('RUN_TEST_FAIL_STAGE') == stage:
    raise SystemExit(int(os.environ['RUN_TEST_FAIL_CODE']))
if stage == 'build':
    directory = Path(arguments[arguments.index('--build') + 1])
    backend = directory / 'bin/triton_backend'
    backend.parent.mkdir(parents=True, exist_ok=True)
    backend.write_text(os.environ['RUN_TEST_BACKEND_SOURCE'], encoding='utf-8')
    backend.chmod(0o755)
'''


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'The runner requires Linux/WSL Bash')
class LocalRunScriptTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='triton run script ')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.checkout = self.directory / 'checkout with spaces'
        (self.checkout / 'src').mkdir(parents=True)
        (self.checkout / 'config').mkdir()
        self.script = self.checkout / 'src/run.sh'
        shutil.copy2(PROJECT / 'src/run.sh', self.script)
        self.config = self.checkout / 'config/server.yaml'
        self.config.write_text('server:\n  port: 6577\n', encoding='utf-8')
        (self.checkout / 'CMakeLists.txt').write_text('project(Fake)\n', encoding='utf-8')
        self.caller = self.directory / 'unrelated caller directory'
        self.caller.mkdir()
        self.triton = self.directory / 'fake Triton runtime'
        self.triton.mkdir()
        self.tools = self.directory / 'fake tools'
        self.tools.mkdir()
        cmake = self.tools / 'cmake'
        cmake.write_text(FAKE_CMAKE, encoding='utf-8')
        cmake.chmod(0o755)
        ninja = self.tools / 'ninja'
        ninja.write_text('#!/bin/sh\nexit 99\n', encoding='utf-8')
        ninja.chmod(0o755)
        self.events_path = self.directory / 'events.jsonl'
        self.bash = shutil.which('bash')
        self.environment = dict(os.environ, PATH=str(self.tools) + os.pathsep + os.environ.get('PATH', ''),
            TRITON_ROOT=str(self.triton), RUN_TEST_EVENTS=str(self.events_path),
            RUN_TEST_BACKEND_SOURCE=FAKE_BACKEND)

    def run_script(self, *arguments, environment=None):
        self.events_path.unlink(missing_ok=True)
        return subprocess.run([self.bash, str(self.script), *arguments], cwd=self.caller,
            env={**self.environment, **(environment or {})}, capture_output=True,
            text=True, encoding='utf-8', timeout=15)

    def events(self):
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text(encoding='utf-8').splitlines()]

    def assert_build_and_run(self, mode, jobs, config, selection=()):
        events = self.events()
        self.assertEqual([event['tool'] for event in events], ['cmake', 'cmake', 'backend'])
        configure, build, backend = events
        build_directory = self.checkout / 'build' / ('dev-release' if mode == 'Release' else 'dev-debug')
        arguments = configure['argv']
        self.assertEqual(arguments[arguments.index('-S') + 1], str(self.checkout))
        self.assertEqual(arguments[arguments.index('-B') + 1], str(build_directory))
        self.assertEqual(arguments[arguments.index('-G') + 1], 'Ninja')
        for value in ('-DCMAKE_BUILD_TYPE=' + mode, '-DBUILD_TESTING=OFF',
                      '-DTRITON_ROOT=' + str(self.triton.resolve()),
                      '-UTRITON_INCLUDE_DIR', '-UTRITON_SERVER_LIBRARY'):
            self.assertIn(value, arguments)
        arguments = build['argv']
        self.assertEqual(arguments[arguments.index('--build') + 1], str(build_directory))
        self.assertEqual(arguments[arguments.index('--target') + 1], 'triton_backend')
        self.assertEqual(arguments[arguments.index('--parallel') + 1], str(jobs))
        self.assertEqual(backend['executable'], str(build_directory / 'bin/triton_backend'))
        self.assertEqual(backend['argv'], ['--config', str(config.resolve()), *selection])
        self.assertEqual(backend['configText'], config.read_text(encoding='utf-8'))

    def test_default_debug_build_runs_from_any_working_directory_and_inherits_output(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_build_and_run('Debug', 4, self.config)
        self.assertIn('fake backend stdout', result.stdout)
        self.assertIn('fake backend stderr', result.stderr)

    def test_release_relative_configuration_models_job_boundaries_and_backend_exit_code(self):
        relative = Path('custom configs/camera settings.yaml')
        custom = self.caller / relative
        custom.parent.mkdir()
        custom.write_text('local camera settings\n', encoding='utf-8')
        selection = ('--model-id', '光伏 model', '--model-version', '3.0.0')
        for jobs in (1, 16):
            with self.subTest(jobs=jobs):
                result = self.run_script('--release', '--config', str(relative), '--jobs', str(jobs),
                    *selection, environment={'TRITON_ROOT': os.path.relpath(self.triton, self.caller),
                                             'RUN_TEST_BACKEND_EXIT': '17'})
                self.assertEqual(result.returncode, 17, result.stderr)
                self.assert_build_and_run('Release', jobs, custom, selection)
                self.assertIn('fake backend stdout', result.stdout)
                self.assertIn('fake backend stderr', result.stderr)

    def test_invalid_or_incomplete_arguments_fail_before_build(self):
        cases = [('--unknown',), ('unexpected-positional',), ('--config',), ('--jobs',),
            ('--model-id',), ('--model-version',), ('--model-id', 'best'),
            ('--model-version', '3.0.0'), ('--model-id', 'best', '--model-version'),
            ('--model-id', '', '--model-version', '3.0.0'),
            ('--jobs', '0'), ('--jobs', '17'), ('--jobs', '-1'), ('--jobs', '1.5'), ('--jobs', '08'),
            ('--jobs', 'many'), ('--config', 'missing.yaml'), ('--config', '.')]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_script(*arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(self.events(), [], 'Invalid arguments reached the build or backend')

    def test_missing_runtime_fails_before_build(self):
        result = self.run_script(environment={'TRITON_ROOT': str(self.directory / 'missing runtime')})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events(), [])

    def test_help_needs_neither_runtime_nor_configuration_and_never_builds(self):
        self.config.unlink()
        basic_tools = self.directory / 'basic shell tools only'
        basic_tools.mkdir()
        for name in ('dirname', 'cat'):
            (basic_tools / name).symlink_to(shutil.which(name))
        for argument in ('-h', '--help'):
            with self.subTest(argument=argument):
                result = self.run_script(argument,
                    environment={'TRITON_ROOT': str(self.directory / 'missing runtime'), 'PATH': str(basic_tools)})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip())
                self.assertEqual(self.events(), [])

    def test_configuration_and_compilation_errors_preserve_exit_codes_without_starting_backend(self):
        for stage, code, expected_calls in (('configure', 23, 1), ('build', 37, 2)):
            with self.subTest(stage=stage):
                result = self.run_script(environment={'RUN_TEST_FAIL_STAGE': stage, 'RUN_TEST_FAIL_CODE': str(code)})
                self.assertEqual(result.returncode, code, result.stderr)
                events = self.events()
                self.assertEqual(len(events), expected_calls)
                self.assertTrue(all(event['tool'] == 'cmake' for event in events))
                self.assertNotIn('fake backend stdout', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
