#!/usr/bin/env python3
"""Measure the real PNG HTTP path with preloaded images and closed-loop clients."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import http.client
import json
import math
import os
from pathlib import Path
import platform
import statistics
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4


STAGES = ('validation', 'preprocess', 'tritonRequest', 'postprocess', 'total')


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {'count': len(values), 'mean': statistics.fmean(values),
            'p50': percentile(0.5), 'p95': percentile(0.95),
            'p99': percentile(0.99), 'min': ordered[0], 'max': ordered[-1]}


def control(endpoint, path):
    connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port or 80, timeout=30)
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError((response.status, result))
        return result
    finally:
        connection.close()


def process_cpu(pid):
    if pid is None:
        return None
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rpartition(') ')[2].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')


def measure(images, endpoint, concurrency, seconds, model_id, version, server_pid):
    gate = threading.Barrier(concurrency + 1)
    index_lock = threading.Lock()
    next_index = 0
    started = 0.0
    captured = utc_now()
    run_id = uuid4().hex
    boundary = 'benchmark-' + run_id
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n'
            'Content-Type: application/json\r\n\r\n').encode()
    middle = (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame.png"\r\n'
              'Content-Type: image/png\r\n\r\n').encode()
    tail = f'\r\n--{boundary}--\r\n'.encode()

    def worker(worker_id):
        nonlocal next_index
        connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port or 80, timeout=30)
        rows, errors = [], []
        gate.wait()
        try:
            while True:
                with index_lock:
                    if next_index >= len(images) and time.perf_counter() >= started + seconds:
                        break
                    sequence = next_index
                    next_index += 1
                image_index = sequence % len(images)
                description, png = images[image_index]
                request_id = f'benchmark-{run_id}-{sequence}'
                metadata = {'SchemaVersion': '1.0', 'RequestId': request_id,
                            'Sequence': sequence, 'FrameId': sequence,
                            'CameraId': f'benchmark-{worker_id}', 'ChannelId': 'pv',
                            'CapturedAtUtc': captured, 'Width': description['width'],
                            'Height': description['height'], 'SourcePixelFormat': 'BGR8Packed',
                            'ImageEncoding': 'Png', 'ModelId': model_id,
                            'RequestedModelVersion': version, 'ConfidenceThreshold': 0.25}
                request_started = time.perf_counter()
                row = {'sequence': sequence, 'imageIndex': image_index, 'ok': False,
                       'httpStatus': 0, 'requestBytes': 0, 'responseBytes': 0,
                       'connectionClosed': False}
                try:
                    encoded = json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode()
                    body = head + encoded + middle + png + tail
                    row['requestBytes'] = len(body)
                    connection.request('POST', '/infer', body=body,
                                       headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
                    response = connection.getresponse()
                    raw = response.read()
                    row.update(httpStatus=response.status, responseBytes=len(raw),
                               connectionClosed=response.will_close)
                    result = json.loads(raw)
                    if response.status != 200 or result.get('ok') is not True:
                        raise RuntimeError(result.get('error', f'HTTP_{response.status}'))
                    for field in ('RequestId', 'FrameId', 'ModelId', 'RequestedModelVersion'):
                        if result.get(field) != metadata[field]:
                            raise RuntimeError('RESPONSE_IDENTITY_MISMATCH')
                    count = result['detectionCount']
                    if count != len(result['detections']) or not 0 <= count <= 300:
                        raise RuntimeError('INVALID_DETECTION_COUNT')
                    for stage in STAGES:
                        value = result['timingsMs'][stage]
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                            raise RuntimeError('INVALID_STAGE_TIMING')
                        row[stage] = value
                    row.update(ok=True, detectionCount=count)
                except Exception as error:
                    row['error'] = str(error)[:500]
                    errors.append({'sequence': sequence, 'imageIndex': image_index,
                                   'httpStatus': row['httpStatus'], 'error': row['error']})
                    connection.close()
                row['clientMs'] = (time.perf_counter() - request_started) * 1000
                rows.append(row)
        finally:
            connection.close()
        return rows, errors

    server_before = process_cpu(server_pid)
    client_before = time.process_time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, worker_id) for worker_id in range(concurrency)]
        started = time.perf_counter()
        started_utc = utc_now()
        gate.wait()
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    client_cpu = time.process_time() - client_before
    server_after = process_cpu(server_pid)
    rows = [row for result, _ in results for row in result]
    errors = [error for _, failures in results for error in failures]
    successful = [row for row in rows if row['ok']]
    result = {'startedAtUtc': started_utc, 'concurrency': concurrency, 'targetSeconds': seconds,
              'elapsedSeconds': elapsed, 'attempts': len(rows), 'successful': len(successful),
              'failed': len(errors), 'successRate': len(successful) / len(rows),
              'successfulImagesPerSecond': len(successful) / elapsed,
              'attemptsPerSecond': len(rows) / elapsed,
              'distinctImagesAttempted': len({row['imageIndex'] for row in rows}),
              'distinctImagesSucceeded': len({row['imageIndex'] for row in successful}),
              'clientLatencyMs': distribution([row['clientMs'] for row in successful]),
              'stagesMs': {stage: distribution([row[stage] for row in successful]) for stage in STAGES},
              'clientCpuPercentOfOneCore': client_cpu / elapsed * 100,
              'serverCpuPercentOfOneCore': None if server_pid is None else (server_after - server_before) / elapsed * 100,
              'requestMiBPerSecond': sum(row['requestBytes'] for row in rows) / elapsed / 1024**2,
              'responseMiBPerSecond': sum(row['responseBytes'] for row in rows) / elapsed / 1024**2,
              'httpStatuses': dict(Counter(str(row['httpStatus']) for row in rows)),
              'closedConnections': sum(row['connectionClosed'] for row in rows),
              'errorTypes': dict(Counter(row['error'] for row in rows if not row['ok'])),
              'errorExamples': errors[:20]}
    return result, rows


def write_samples(path, rows):
    columns = ['sequence', 'imageIndex', 'ok', 'httpStatus', 'clientMs', *STAGES,
               'detectionCount', 'requestBytes', 'responseBytes', 'connectionClosed', 'error']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Prepared dataset.json')
    parser.add_argument('--url', default='http://127.0.0.1:16577')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', default='1,2,4,8')
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup-seconds', type=float, default=10)
    parser.add_argument('--model-id', default='best')
    parser.add_argument('--model-version', default='1.0.0')
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--label', default='WSL loopback')
    args = parser.parse_args()
    levels = [int(value) for value in args.concurrency.split(',')]
    if not levels or any(value < 1 for value in levels) or args.seconds <= 0 or args.repeats < 1 or args.warmup_seconds < 0:
        parser.error('Invalid concurrency, duration or repeat count')
    endpoint = urlsplit(args.url)
    if endpoint.scheme != 'http' or not endpoint.hostname or endpoint.path not in ('', '/'):
        parser.error('The URL must be an HTTP origin')
    if args.output.exists():
        parser.error('Output already exists; use a new report filename')
    document = json.loads(args.dataset.read_text(encoding='utf-8'))
    images = [(image, (args.dataset.parent / image['pngRelativePath']).read_bytes()) for image in document['images']]
    if not images:
        parser.error('The dataset contains no images')
    for image, data in images:
        if len(data) != image['pngBytes'] or not data.startswith(b'\x89PNG\r\n\x1a\n'):
            parser.error('The prepared PNG payload does not match its manifest')
    initial = control(endpoint, '/api/status')
    active = initial['activeModel'] or {}
    if not initial['modelReady'] or active.get('id') != args.model_id or active.get('version') != args.model_version:
        raise RuntimeError('The requested model/version must already be active')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output.parent / (args.output.stem + '-samples')
    sample_dir.mkdir(exist_ok=False)
    report = {'complete': False, 'label': args.label, 'url': args.url,
              'clientPlatform': platform.platform(), 'python': platform.python_version(),
              'dataset': document['summary'], 'datasetManifest': str(args.dataset),
              'initialStatus': initial, 'measurement': {
                  'mode': 'closed-loop; one outstanding request per worker; no retries',
                  'input': 'full-size PNG; single image per request; confidence threshold 0.25',
                  'excluded': 'source image decode, PNG encoding and disk reads; all PNGs preloaded',
                  'clientLatency': 'metadata JSON/multipart construction through response JSON parse and identity checks',
                  'throughput': 'successful completed images / wall time including final in-flight drain',
                  'timingWarning': 'tritonRequest includes request setup, queueing, copies, GPU and callbacks; not pure GPU time'},
              'warmups': [], 'trials': [], 'summary': []}

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    for concurrency in levels:
        warmup, _ = measure(images, endpoint, concurrency, args.warmup_seconds,
                            args.model_id, args.model_version, args.server_pid)
        report['warmups'].append(warmup)
        save()
        if warmup['failed']:
            raise RuntimeError(f'Warmup failed: {warmup["errorTypes"]}')
        for repeat in range(args.repeats):
            trial, rows = measure(images, endpoint, concurrency, args.seconds,
                                  args.model_id, args.model_version, args.server_pid)
            trial['repeat'] = repeat + 1
            csv_path = sample_dir / f'c{concurrency}-r{repeat + 1}.csv'
            write_samples(csv_path, rows)
            trial['samples'] = str(csv_path)
            report['trials'].append(trial)
            save()
            print(json.dumps({'label': args.label, 'concurrency': concurrency, 'repeat': repeat + 1,
                              'imagesPerSecond': trial['successfulImagesPerSecond'],
                              'p50Ms': trial['clientLatencyMs']['p50'] if trial['successful'] else None,
                              'p95Ms': trial['clientLatencyMs']['p95'] if trial['successful'] else None,
                              'successful': trial['successful'], 'failed': trial['failed'],
                              'clientCpuPercentOfOneCore': trial['clientCpuPercentOfOneCore']}), flush=True)
        trials = [trial for trial in report['trials'] if trial['concurrency'] == concurrency]
        successful_trials = [trial for trial in trials if trial['successful']]
        report['summary'].append({'concurrency': concurrency,
                                  'medianImagesPerSecond': statistics.median(trial['successfulImagesPerSecond'] for trial in trials),
                                  'minimumImagesPerSecond': min(trial['successfulImagesPerSecond'] for trial in trials),
                                  'maximumImagesPerSecond': max(trial['successfulImagesPerSecond'] for trial in trials),
                                  'successful': sum(trial['successful'] for trial in trials),
                                  'failed': sum(trial['failed'] for trial in trials),
                                  'meanTrialP50Ms': statistics.fmean(trial['clientLatencyMs']['p50'] for trial in successful_trials) if successful_trials else None,
                                  'meanTrialP95Ms': statistics.fmean(trial['clientLatencyMs']['p95'] for trial in successful_trials) if successful_trials else None})
        save()
    report['finalStatus'] = control(endpoint, '/api/status')
    if report['finalStatus']['activeModel']['generation'] != active['generation']:
        raise RuntimeError('The active generation changed during measurement')
    report['complete'] = True
    report['passed'] = all(trial['failed'] == 0 and trial['distinctImagesSucceeded'] == len(images)
                           for trial in report['trials'])
    save()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
