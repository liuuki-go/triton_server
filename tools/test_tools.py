#!/usr/bin/env python3
"""CPU-only regression checks for archive updates and owned process control."""
import io
import json
from pathlib import Path
import signal
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import call, patch

import manage
import sync_wsl


def small_project_archive(extra_files=None):
    files = {'CMakeLists.txt': b'project(Test)', 'src/app/main.cpp': b'int main() {}',
             'src/api/BackendHttpServer.cpp': b'new HTTP source',
             'config/server.yaml': b'server:\n  port: 6577\n',
             'models/manifest.json': b'{"models": []}',
             'scripts/wsl/build.sh': b'#!/bin/bash\n',
             'scripts/wsl/run.sh': b'#!/bin/bash\n', 'tools/manage.py': b'# helper\n'}
    files.update(extra_files or {})
    manifest = {'project': 'triton_back', 'files': {
        name: manage.hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files['.project-sync-manifest.json'] = json.dumps(manifest).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    return stream.getvalue()


class ArchiveSafetyTest(unittest.TestCase):
    def test_real_project_roundtrip_uses_plaintext_lengths_and_hashes(self):
        root = Path(__file__).resolve().parents[1]
        data = sync_wsl.build_archive(root)
        files, manifest = manage.read_archive(data)
        for name in ('CMakeLists.txt', 'src/app/main.cpp', 'src/api/BackendHttpServer.cpp',
                     'models/best/1/model.plan', 'scripts/wsl/build.sh'):
            with self.subTest(name=name):
                self.assertEqual(files[name], (root / name).read_bytes())
                self.assertEqual(manifest['files'][name], manage.hashlib.sha256(files[name]).hexdigest())
        self.assertFalse(any(part in manage.EXCLUDED for name in files for part in Path(name).parts))
        self.assertTrue(files['src/app/main.cpp'].startswith(b'#include'))
        self.assertFalse(any(Path(name).parts[0] in manage.LEGACY_SOURCE_DIRECTORIES for name in files))

    def test_legacy_root_source_directories_are_rejected_in_new_archives(self):
        for directory in manage.LEGACY_SOURCE_DIRECTORIES:
            name = directory + '/old.cpp'
            with self.subTest(name=name):
                self.assertFalse(manage.allowed_path(name))
                with self.assertRaisesRegex(ValueError, 'outside the project sync allowlist'):
                    manage.read_archive(small_project_archive({name: b'old source'}))

    def test_archives_require_the_new_source_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files, _ = manage.read_archive(small_project_archive())
            for name, data in files.items():
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (root / 'src/app/main.cpp').unlink()
            (root / 'app').mkdir()
            (root / 'app/main.cpp').write_bytes(b'old layout')
            with self.assertRaisesRegex(ValueError, 'src/app/main.cpp'):
                sync_wsl.build_archive(root)

    def test_traversal_absolute_and_noncanonical_paths_are_rejected(self):
        for name in ('../outside', '/etc/passwd', 'config/../../outside', './config/file',
                     'config//file', 'config\\file', 'C:/file'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                manage.allowed_path(name)

    def test_symbolic_and_hard_links_are_rejected_before_writes(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w:gz') as archive:
                entry = tarfile.TarInfo('src/api/outside')
                entry.type, entry.linkname = kind, '/etc/passwd'
                archive.addfile(entry)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                manage.read_archive(stream.getvalue())

    def test_model_hash_mismatch_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        files, _ = manage.read_archive(sync_wsl.build_archive(root))
        files['models/best/1/model.plan'] = b'wrong-engine'
        manifest = {'project': 'triton_back', 'files': {
            name: manage.hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        files['.project-sync-manifest.json'] = json.dumps(manifest).encode()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, data in files.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, 'Model checksum mismatch'):
            manage.read_archive(stream.getvalue())

    def test_existing_configuration_is_preserved_with_review_candidate(self):
        root = Path(__file__).resolve().parents[1]
        data = sync_wsl.build_archive(root)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'project'
            manage.sync_archive(target, data)
            config = target / 'config/server.yaml'
            config.write_bytes(b'local configuration change\n')
            report = manage.sync_archive(target, data)
            self.assertEqual(config.read_bytes(), b'local configuration change\n')
            entry = next(item for item in report['preservedConfig'] if item['path'] == 'config/server.yaml')
            self.assertEqual(Path(entry['incoming']).read_bytes(), (root / 'config/server.yaml').read_bytes())
            replaced = manage.sync_archive(target, data, replace_config=True)
            self.assertTrue(replaced['replaceConfig'])
            self.assertEqual(config.read_bytes(), (root / 'config/server.yaml').read_bytes())

    def write_legacy_manifest(self, target, files):
        for name, data in files.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        manage.atomic_write(target / 'runtime/sync-manifest.json', json.dumps({
            'sourceFiles': {name: manage.hashlib.sha256(data).hexdigest()
                            for name, data in files.items()}}).encode())

    def test_layout_migration_removes_only_unchanged_managed_source_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'project'
            self.write_legacy_manifest(target, {'app/main.cpp': b'old main',
                'api/BackendHttpServer.cpp': b'old HTTP source', 'api/local.cpp': b'managed local source',
                'domain/removed.h': b'no replacement'})
            (target / 'api/local.cpp').write_bytes(b'local edits')
            (target / 'api/user.cpp').write_bytes(b'unmanaged user file')
            report = manage.sync_archive(target, small_project_archive({
                'src/api/local.cpp': b'incoming local source'}))
            self.assertEqual(set(report['removedLegacySources']), {'app/main.cpp', 'api/BackendHttpServer.cpp'})
            self.assertFalse((target / 'app').exists())
            self.assertEqual((target / 'api/local.cpp').read_bytes(), b'local edits')
            self.assertEqual((target / 'api/user.cpp').read_bytes(), b'unmanaged user file')
            self.assertEqual((target / 'domain/removed.h').read_bytes(), b'no replacement')
            self.assertEqual(report['preservedLegacySources'], [
                {'path': 'api/local.cpp', 'reason': 'Local changes'},
                {'path': 'domain/removed.h', 'reason': 'No replacement in incoming src'}])
            self.assertEqual((target / 'src/app/main.cpp').read_bytes(), b'int main() {}')
            self.assertEqual(json.loads((target / 'runtime/sync-manifest.json').read_text()), report)
            self.assertEqual(manage.sync_archive(target, small_project_archive())['removedLegacySources'], [])
            self.assertEqual((target / 'api/local.cpp').read_bytes(), b'local edits')

    def test_layout_migration_without_previous_manifest_preserves_old_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'project'
            (target / 'app').mkdir(parents=True)
            (target / 'app/main.cpp').write_bytes(b'unmanaged source')
            report = manage.sync_archive(target, small_project_archive())
            self.assertEqual(report['removedLegacySources'], [])
            self.assertEqual((target / 'app/main.cpp').read_bytes(), b'unmanaged source')

    def test_layout_migration_never_removes_files_outside_legacy_source_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'project'
            self.write_legacy_manifest(target, {'docs/local.md': b'local document',
                'tests/local.cpp': b'local test', 'build/local': b'local build file'})
            report = manage.sync_archive(target, small_project_archive())
            self.assertEqual(report['removedLegacySources'], [])
            for name, expected in (('docs/local.md', b'local document'),
                    ('tests/local.cpp', b'local test'), ('build/local', b'local build file')):
                self.assertEqual((target / name).read_bytes(), expected)

    def test_layout_migration_preserves_symbolic_links_and_their_targets(self):
        if sys.platform == 'win32':
            self.skipTest('Linux deployment checks this without Windows symlink privileges')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'project'
            self.write_legacy_manifest(target, {'app/main.cpp': b'old main',
                'api/BackendHttpServer.cpp': b'old HTTP source'})
            outside_file = root / 'outside.cpp'
            outside_file.write_bytes(b'old main')
            (target / 'app/main.cpp').unlink()
            (target / 'app/main.cpp').symlink_to(outside_file)
            outside_directory = root / 'outside-api'
            (target / 'api').rename(outside_directory)
            (target / 'api').symlink_to(outside_directory, target_is_directory=True)
            report = manage.sync_archive(target, small_project_archive())
            self.assertEqual(report['removedLegacySources'], [])
            self.assertEqual(outside_file.read_bytes(), b'old main')
            self.assertEqual((outside_directory / 'BackendHttpServer.cpp').read_bytes(), b'old HTTP source')
            self.assertEqual(report['preservedLegacySources'], [
                {'path': 'app/main.cpp', 'reason': 'Symbolic link'},
                {'path': 'api/BackendHttpServer.cpp', 'reason': 'Symbolic link'}])

    def test_atomic_write_refuses_symlink(self):
        if sys.platform == 'win32':
            self.skipTest('Linux deployment checks this without Windows symlink privileges')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / 'outside'
            outside.write_bytes(b'preserved')
            (root / 'link').symlink_to(outside)
            with self.assertRaises(ValueError):
                manage.atomic_write(root / 'link', b'replacement')
            self.assertEqual(outside.read_bytes(), b'preserved')

    def test_unrelated_native_targets_are_rejected(self):
        for value in ('/opt', '/opt/triton', '/opt/triton_backend', '/etc', '/mnt/d/codes/triton_back'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                manage.managed_target(value)


class ProcessOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.process = self.root / 'proc/1234'
        self.process.mkdir(parents=True)
        self.pid_file = self.root / 'service.pid.json'
        self.record = {'pid': 1234, 'executable': '/opt/triton_back/bin/triton_backend',
            'config': '/opt/triton_back/config/server.yaml', 'linuxStartTicks': '777',
            'ownerToken': 'test-owner'}
        self.pid_file.write_text(json.dumps(self.record))
        self.process.joinpath('cmdline').write_bytes(b'/opt/triton_back/bin/triton_backend\0--config\0/opt/triton_back/config/server.yaml\0')
        self.process.joinpath('environ').write_bytes(b'TRITON_BACK_OWNER_TOKEN=test-owner\0')
        self.write_stat('S', '777')
        original = Path
        mocked = patch.object(manage, 'Path', side_effect=lambda value: self.root / 'proc'
            if str(value) == '/proc' else original(value))
        mocked.start()
        self.addCleanup(mocked.stop)

    def write_stat(self, state, ticks):
        self.process.joinpath('stat').write_text(' '.join(['1234', '(triton backend)', state] + ['0'] * 18 + [ticks, '0']))

    def test_exact_identity_is_accepted_including_stat_command_spaces(self):
        self.assertEqual(manage.running_process(self.pid_file), self.record)

    def test_different_command_is_rejected(self):
        self.process.joinpath('cmdline').write_bytes(b'/opt/other/server\0')
        with self.assertRaises(ValueError):
            manage.running_process(self.pid_file)

    def test_startup_model_is_part_of_the_verified_command(self):
        self.record['startupModel'] = {'modelId': 'best', 'version': '2.0.0'}
        self.pid_file.write_text(json.dumps(self.record))
        command = manage.backend_command(self.record['executable'], self.record['config'], 'best', '2.0.0')
        self.process.joinpath('cmdline').write_bytes(b'\0'.join(value.encode() for value in command) + b'\0')
        self.assertEqual(manage.running_process(self.pid_file), self.record)
        self.process.joinpath('cmdline').write_bytes(b'\0'.join(value.encode() for value in command[:-1] + ['1.0.0']) + b'\0')
        with self.assertRaises(ValueError):
            manage.running_process(self.pid_file)

    def test_startup_model_requires_both_nonempty_arguments(self):
        for model_id, version in (('best', None), (None, '1.0.0'), ('', ''), (' ', '1.0.0')):
            with self.subTest(model_id=model_id, version=version), self.assertRaises(ValueError):
                manage.backend_command('/binary', '/config', model_id, version)

    def test_reused_pid_is_rejected(self):
        self.write_stat('S', '999')
        with self.assertRaises(ValueError):
            manage.running_process(self.pid_file)

    def test_different_owner_is_rejected(self):
        self.process.joinpath('environ').write_bytes(b'TRITON_BACK_OWNER_TOKEN=other-owner\0')
        with self.assertRaises(ValueError):
            manage.running_process(self.pid_file)

    def test_caller_owner_mismatch_prevents_any_signal(self):
        with patch.object(manage.os, 'kill') as kill_mock:
            with self.assertRaises(ValueError):
                manage.stop_process(self.pid_file, expected_owner='another-caller')
        kill_mock.assert_not_called()

    def test_zombie_is_treated_as_exited(self):
        self.write_stat('Z', '777')
        self.assertIsNone(manage.running_process(self.pid_file))

    def test_escalation_rechecks_identity_and_reports_forced(self):
        def kill(pid, requested_signal):
            if requested_signal == signal.SIGKILL:
                self.write_stat('Z', '777')
        with patch.object(signal, 'SIGKILL', 9, create=True), \
             patch.object(manage.time, 'monotonic', side_effect=(0, 36, 37, 38)), \
             patch.object(manage.os, 'kill', side_effect=kill) as kill_mock:
            result = manage.stop_process(self.pid_file)
            self.assertTrue(result['forced'])
            self.assertEqual(kill_mock.call_args_list, [call(1234, signal.SIGTERM), call(1234, signal.SIGKILL)])

    def test_identity_change_prevents_escalation(self):
        with patch.object(manage, 'running_process', side_effect=(self.record, ValueError('PID changed'))), \
             patch.object(manage.time, 'monotonic', side_effect=(0, 36)), \
             patch.object(manage.os, 'kill') as kill_mock:
            with self.assertRaises(ValueError):
                manage.stop_process(self.pid_file)
        kill_mock.assert_called_once_with(1234, signal.SIGTERM)

    def test_already_stopped_is_idempotent(self):
        self.pid_file.unlink()
        with patch.object(manage.os, 'kill') as kill_mock:
            self.assertTrue(manage.stop_process(self.pid_file)['alreadyStopped'])
        kill_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
