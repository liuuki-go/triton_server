#!/usr/bin/env python3
"""Read authorized Windows project files and sync a verified native WSL copy."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from manage import allowed_path, read_archive


def build_archive(source):
    source = Path(source).resolve()
    content = {}
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source).as_posix()
        if not allowed_path(relative):
            continue
        if path.is_symlink():
            raise ValueError('Project sync does not follow symbolic links: ' + relative)
        if path.is_file():
            # Stat may describe encrypted storage; actual authorized bytes determine size.
            content[relative] = path.read_bytes()
    manifest = {'project': 'triton_back', 'files': {
        name: hashlib.sha256(data).hexdigest() for name, data in content.items()}}
    content['.project-sync-manifest.json'] = json.dumps(manifest).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, data in content.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    data = output.getvalue()
    read_archive(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--distribution', default='template-clone')
    parser.add_argument('--replace-config', action='store_true')
    parser.add_argument('--archive-only', help='Write a verified archive without changing WSL')
    parser.add_argument('--report')
    args = parser.parse_args()
    archive = build_archive(args.source)
    if args.archive_only:
        Path(args.archive_only).write_bytes(archive)
        print(json.dumps({'ok': True, 'archive': args.archive_only, 'bytes': len(archive)}))
        return
    bootstrap = ('import io,sys,tarfile; data=sys.stdin.buffer.read(); '
        'archive=tarfile.open(fileobj=io.BytesIO(data),mode="r:gz"); '
        'code=archive.extractfile("tools/manage.py").read().decode("utf-8"); '
        'sys.stdin=io.TextIOWrapper(io.BytesIO(data)); exec(compile(code,"tools/manage.py","exec"))')
    command = ['wsl.exe', '-d', args.distribution, '-u', 'root', '--exec', 'python3', '-c',
               bootstrap, 'sync', '--target', '/opt/triton_back', '--stdin-archive']
    if args.replace_config:
        command.append('--replace-config')
    result = subprocess.run(command, input=archive, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    report = json.loads(result.stdout.decode('utf-8'))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
