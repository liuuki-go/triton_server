#!/usr/bin/env python3
"""Native WSL project updates and strictly owned C++ process management."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import sys
import tarfile
import time
from uuid import uuid4

TARGET = Path('/opt/triton_back')
MARKER = '.triton-back-project'
OWNER_ENV = 'TRITON_BACK_OWNER_TOKEN'
DIRECTORIES = frozenset(('src', 'config', 'models', 'model_sources', 'tests', 'docs', 'scripts', 'tools'))
LEGACY_SOURCE_DIRECTORIES = frozenset(('api', 'app', 'services', 'configuration', 'domain', 'inference'))
ROOT_FILES = frozenset(('CMakeLists.txt', '.gitignore', 'README.md'))
EXCLUDED = frozenset(('__pycache__', 'runtime', 'build', 'bin', '.git', '.venv', 'node_modules'))


def managed_target(value):
    target = Path(value)
    if str(target) != str(TARGET) or target.is_symlink() or target.resolve() != TARGET:
        raise ValueError('The managed project target must be /opt/triton_back without symlinks')
    marker = target / MARKER
    if target.exists() and any(target.iterdir()):
        if not marker.is_file() or marker.is_symlink():
            raise ValueError('Refusing to modify an unrelated /opt/triton_back directory')
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record != {'project': 'triton_back', 'schemaVersion': 1}:
            raise ValueError('The native project marker is invalid')
    return target


def allowed_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ('..', '.') for part in path.parts):
        raise ValueError('Unsafe archive path: ' + value)
    if '\\' in value or ':' in value or value != path.as_posix():
        raise ValueError('Archive paths must be canonical relative POSIX paths: ' + value)
    if any(part in EXCLUDED for part in path.parts) or path.suffix in ('.pyc', '.pyo'):
        return False
    return (len(path.parts) == 1 and value in ROOT_FILES) or path.parts[0] in DIRECTORIES


def read_archive(data):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name in files:
                raise ValueError('Archive must contain unique regular files only')
            if member.name != '.project-sync-manifest.json' and not allowed_path(member.name):
                raise ValueError('File outside the project sync allowlist: ' + member.name)
            stream = archive.extractfile(member)
            files[member.name] = stream.read()
    manifest = json.loads(files.pop('.project-sync-manifest.json'))
    if manifest['project'] != 'triton_back' or set(manifest['files']) != set(files):
        raise ValueError('Archive file list does not match its manifest')
    for name, content in files.items():
        if hashlib.sha256(content).hexdigest() != manifest['files'][name]:
            raise ValueError('Archive checksum mismatch: ' + name)
    for required in ('CMakeLists.txt', 'src/app/main.cpp', 'config/server.yaml', 'models/manifest.json',
                     'scripts/wsl/build.sh', 'scripts/wsl/run.sh', 'tools/manage.py'):
        if required not in files:
            raise ValueError('Incomplete project archive: ' + required)
    model_manifest = json.loads(files['models/manifest.json'])
    for model in model_manifest['models']:
        for path_key, hash_key in (('enginePath', 'sha256'), ('onnxPath', 'onnxSha256')):
            name = model[path_key]
            if not allowed_path(name) or name not in files:
                raise ValueError('Model file is missing from the standalone project: ' + name)
            if hashlib.sha256(files[name]).hexdigest() != model[hash_key].lower():
                raise ValueError('Model checksum mismatch: ' + name)
    return files, manifest


def atomic_write(path, content):
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Refusing to write through a symbolic link: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.updating-' + uuid4().hex)
    temporary.write_bytes(content)
    temporary.replace(path)


def process_stat(process):
    text = (process / 'stat').read_text()
    fields = text.rpartition(') ')[2].split()
    return fields[0], fields[19]


def backend_command(executable, config, model_id=None, model_version=None):
    command = [str(executable), '--config', str(config)]
    if model_id is not None or model_version is not None:
        if any(not isinstance(value, str) or not value.strip() or len(value) > 256
               for value in (model_id, model_version)):
            raise ValueError('Model ID and version must both be nonempty strings of at most 256 characters')
        command.extend(['--model-id', model_id, '--model-version', model_version])
    return command


def running_process(pid_file):
    if not pid_file.is_file():
        return None
    record = json.loads(pid_file.read_text(encoding='utf-8'))
    pid = record['pid']
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise ValueError('Invalid managed PID')
    process = Path('/proc') / str(pid)
    try:
        state, ticks = process_stat(process)
        if state == 'Z':
            return None
        command = (process / 'cmdline').read_bytes().split(b'\0')
        selection = record.get('startupModel', {})
        expected = [value.encode() for value in backend_command(record['executable'], record['config'],
                    selection.get('modelId'), selection.get('version'))]
        if command != expected + [b'']:
            raise ValueError('The recorded PID does not belong to this backend command')
        if ticks != record['linuxStartTicks']:
            raise ValueError('The recorded PID has been reused')
        owner = (OWNER_ENV + '=' + record['ownerToken']).encode()
        if owner not in (process / 'environ').read_bytes().split(b'\0'):
            raise ValueError('The recorded PID has a different owner token')
        if record['executable'] != '/opt/triton_back/bin/triton_backend' or \
                record['config'] != '/opt/triton_back/config/server.yaml':
            raise ValueError('The process record does not belong to this project')
        return record
    except FileNotFoundError:
        return None


def stop_process(pid_file, timeout=35, expected_owner=None):
    record = running_process(pid_file)
    if record is None:
        return {'running': False, 'alreadyStopped': True, 'forced': False}
    if expected_owner is not None and record['ownerToken'] != expected_owner:
        raise ValueError('The caller does not own this managed process')
    os.kill(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if running_process(pid_file) is None:
            pid_file.unlink(missing_ok=True)
            return {'running': False, 'stoppedPid': record['pid'], 'forced': False}
        time.sleep(0.1)
    verified = running_process(pid_file)
    if verified is None:
        pid_file.unlink(missing_ok=True)
        return {'running': False, 'stoppedPid': record['pid'], 'forced': False}
    if verified != record:
        raise ValueError('The managed process record changed during graceful shutdown')
    os.kill(verified['pid'], signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if running_process(pid_file) is None:
            pid_file.unlink(missing_ok=True)
            return {'running': False, 'stoppedPid': record['pid'], 'forced': True}
        time.sleep(0.1)
    raise TimeoutError('The managed process did not exit after SIGKILL')


def settings(target):
    source = (target / 'config/server.yaml').read_text(encoding='utf-8')
    # This reads the public listening port only; C++ validates the complete YAML.
    match = re.search(r'^  port:\s*([0-9]+)\s*(?:#.*)?$', source, re.MULTILINE)
    if match is None or not 1 <= int(match[1]) <= 65535:
        raise ValueError('server.port must be an unquoted integer in config/server.yaml')
    return {'target': str(target), 'port': int(match[1]),
            'url': 'http://127.0.0.1:' + match[1], 'config': str(target / 'config/server.yaml')}


def sync_archive(target, data, replace_config=False):
    files, manifest = read_archive(data)
    pid_file = target / 'runtime/service.pid.json'
    if running_process(pid_file):
        raise ValueError('Stop this managed backend before updating project files')
    previous_sources = {}
    previous_manifest = target / 'runtime/sync-manifest.json'
    if previous_manifest.is_symlink() or any(parent.is_symlink() for parent in previous_manifest.parents):
        raise ValueError('Refusing to read a sync manifest through a symbolic link')
    if previous_manifest.is_file():
        previous_sources = json.loads(previous_manifest.read_text(encoding='utf-8'))['sourceFiles']
        if not isinstance(previous_sources, dict):
            raise ValueError('The previous sync manifest file list is invalid')
    target.mkdir(parents=True, exist_ok=True)
    atomic_write(target / MARKER, json.dumps({'project': 'triton_back', 'schemaVersion': 1}).encode())
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    candidate_root = target / 'runtime/update-candidates' / timestamp
    copied, preserved = [], []
    for name, content in files.items():
        destination = target / name
        if name.startswith('config/') and destination.is_file() and not replace_config:
            if destination.read_bytes() != content:
                atomic_write(candidate_root / name, content)
                preserved.append({'path': name, 'incoming': str(candidate_root / name)})
            continue
        if destination.is_file() and destination.read_bytes() == content:
            continue
        atomic_write(destination, content)
        copied.append(name)
    removed_legacy, preserved_legacy = [], []
    for name, digest in previous_sources.items():
        parts = PurePosixPath(name).parts
        if not parts or parts[0] not in LEGACY_SOURCE_DIRECTORIES:
            continue
        if not allowed_path('src/' + name):
            continue
        destination = target / name
        reason = None
        if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
            reason = 'Symbolic link'
        elif not destination.exists():
            continue
        elif not destination.is_file():
            reason = 'Not a regular file'
        elif 'src/' + name not in files:
            reason = 'No replacement in incoming src'
        elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            reason = 'Local changes'
        if reason:
            preserved_legacy.append({'path': name, 'reason': reason})
            continue
        destination.unlink()
        removed_legacy.append(name)
        for directory in destination.parents:
            if directory == target:
                break
            try:
                directory.rmdir()
            except OSError:
                break
    report = {'ok': True, 'target': str(target), 'copiedFiles': copied,
              'preservedConfig': preserved, 'sourceFiles': manifest['files'],
              'removedLegacySources': removed_legacy, 'preservedLegacySources': preserved_legacy,
              'replaceConfig': replace_config, 'syncedAtUtc': timestamp}
    atomic_write(target / 'runtime/sync-manifest.json', json.dumps(report, indent=2).encode())
    return report


def foreground(target, model_id=None, model_version=None):
    config = settings(target)
    pid_file = target / 'runtime/service.pid.json'
    if running_process(pid_file):
        raise ValueError('This managed backend is already running')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', config['port'])) == 0:
            raise ValueError('The backend port is occupied; existing services are untouched')
    binary = target / 'bin/triton_backend'
    command = backend_command(binary, config['config'], model_id, model_version)
    if not binary.is_file():
        raise ValueError('Build the native project before starting it')
    environment = os.environ.copy()
    owner = environment.get(OWNER_ENV) or uuid4().hex
    environment[OWNER_ENV] = owner
    environment['LD_LIBRARY_PATH'] = ':'.join(('/opt/triton/2.71.0/lib',
        '/usr/local/cuda-13.3/targets/x86_64-linux/lib', '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/wsl/lib', environment.get('LD_LIBRARY_PATH', ''))).rstrip(':')
    _, ticks = process_stat(Path('/proc') / str(os.getpid()))
    record = {'pid': os.getpid(), 'linuxStartTicks': ticks, 'ownerToken': owner,
              'executable': str(binary), 'config': config['config'], 'port': config['port'],
              'startedAtUtc': datetime.now(timezone.utc).isoformat(), 'mode': 'foreground'}
    if model_id is not None:
        record['startupModel'] = {'modelId': model_id, 'version': model_version}
    atomic_write(pid_file, json.dumps(record, indent=2).encode())
    os.chdir(target)
    try:
        log_path = target / 'runtime/logs/service.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.is_symlink():
            raise ValueError('The native service log cannot be a symbolic link')
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            # The C++ process writes directly to Linux files, independent of host pipes.
            os.dup2(descriptor, 1)
            os.dup2(descriptor, 2)
        finally:
            os.close(descriptor)
        os.execve(binary, command, environment)
    except Exception:
        pid_file.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sync', 'settings', 'status', 'stop', 'foreground'))
    parser.add_argument('--target', default=str(TARGET))
    parser.add_argument('--stdin-archive', action='store_true')
    parser.add_argument('--replace-config', action='store_true')
    parser.add_argument('--owner-token', help='Stop only the process belonging to this caller')
    parser.add_argument('--model-id', help='Load this business model at process startup')
    parser.add_argument('--model-version', help='Business version to load with --model-id')
    args = parser.parse_args()
    if os.name == 'nt' or os.geteuid() != 0:
        parser.error('Run this native helper as root inside WSL')
    target = managed_target(args.target)
    if args.action == 'sync':
        if not args.stdin_archive:
            parser.error('sync requires --stdin-archive')
        result = sync_archive(target, sys.stdin.buffer.read(), args.replace_config)
    elif args.action == 'foreground':
        foreground(target, args.model_id, args.model_version)
        return
    elif args.action == 'stop':
        result = stop_process(target / 'runtime/service.pid.json', expected_owner=args.owner_token)
    elif args.action == 'status':
        result = {'process': running_process(target / 'runtime/service.pid.json')}
    else:
        result = settings(target)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
