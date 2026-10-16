#!/usr/bin/env python3
"""Compare single-image HTTP requests with real Triton dynamic-batch counters."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
from pathlib import Path
import statistics
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

import benchmark_http as common


IDENTITY_FIELDS = ('RequestId', 'Sequence', 'FrameId', 'CameraId', 'ModelId', 'RequestedModelVersion')
BOX_FIELDS = ('x1', 'y1', 'x2', 'y2')


def load_dataset(path):
    document = json.loads(path.read_text(encoding='utf-8'))
    images, identity = [], []
    for item in document['images']:
        png = (path.parent / item['pngRelativePath']).read_bytes()
        if len(png) != item['pngBytes'] or not png.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('PNG payload does not match the dataset manifest')
        digest = hashlib.sha256(png).hexdigest()
        images.append((item, png))
        identity.append({'imageIndex': len(images) - 1, 'pngRelativePath': item['pngRelativePath'],
                         'width': item['width'], 'height': item['height'], 'pngSha256': digest})
    if not images:
        raise ValueError('The dataset contains no images')
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return document, images, identity, digest


def statistics_snapshot(status):
    active = status.get('activeModel') or {}
    raw = active.get('executionStatistics')
    if not isinstance(raw, dict) or len(raw.get('model_stats', [])) != 1:
        raise ValueError('Missing activeModel.executionStatistics.model_stats; batching is unverified')
    model = raw['model_stats'][0]
    batches = {}
    for item in model['batch_stats']:
        size, count = int(item['batch_size']), int(item['compute_infer']['count'])
        if size < 1 or count < 0 or size in batches:
            raise ValueError('Invalid Triton batch statistics')
        batches[size] = count
    return {'inferenceCount': int(model['inference_count']),
            'executionCount': int(model['execution_count']), 'batches': batches}


def statistics_delta(before, after):
    batches = {size: after['batches'].get(size, 0) - before['batches'].get(size, 0)
               for size in sorted(set(before['batches']) | set(after['batches']))}
    inference = after['inferenceCount'] - before['inferenceCount']
    executions = after['executionCount'] - before['executionCount']
    if inference < 0 or executions < 0 or any(count < 0 for count in batches.values()):
        raise ValueError('Triton counters decreased during a stable generation')
    return {'inferenceCount': inference, 'executionCount': executions,
            'batchExecutions': {str(size): count for size, count in batches.items() if count},
            'imagesAccountedFor': sum(size * count for size, count in batches.items()),
            'executionsAccountedFor': sum(batches.values())}


def checked_status(endpoint, generation):
    status = common.control(endpoint, '/api/status')
    if (status.get('activeModel') or {}).get('generation') != generation:
        raise RuntimeError('The active model generation changed during measurement')
    statistics_snapshot(status)
    return status


def idle_snapshot(endpoint, generation, timeout=5):
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        status = checked_status(endpoint, generation)
        snapshot = statistics_snapshot(status)
        if status['inFlight'] == 0 and previous == snapshot:
            return status, snapshot
        previous = snapshot if status['inFlight'] == 0 else None
        if time.monotonic() >= deadline:
            raise RuntimeError('The model did not reach an idle, stable statistics snapshot')
        time.sleep(0.03)


def settled_statistics(endpoint, generation, before, successful, timeout=5):
    deadline = time.monotonic() + timeout
    while True:
        status = checked_status(endpoint, generation)
        after = statistics_snapshot(status)
        delta = statistics_delta(before, after)
        settled = (status['inFlight'] == 0 and delta['imagesAccountedFor'] == successful and
                   delta['inferenceCount'] == successful and
                   delta['executionsAccountedFor'] == delta['executionCount'])
        if settled or time.monotonic() >= deadline:
            return {'settled': settled, 'before': before, 'after': after, 'delta': delta,
                    'expectedSuccessfulImages': successful, 'finalInFlight': status['inFlight']}
        time.sleep(0.03)


def detection_difference(expected, actual, confidence_tolerance, coordinate_tolerance):
    edges = []
    for detection in expected:
        candidates = []
        for index, other in enumerate(actual):
            if (detection['classId'], detection['className']) != (other['classId'], other['className']):
                continue
            confidence = abs(detection['confidence'] - other['confidence'])
            coordinate = max(abs(detection['box'][field] - other['box'][field]) for field in BOX_FIELDS)
            if confidence <= confidence_tolerance and coordinate <= coordinate_tolerance:
                candidates.append((coordinate, confidence, index))
        edges.append([index for _, _, index in sorted(candidates)])
    assigned = {}

    def assign(index, seen):
        for candidate in edges[index]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in assigned or assign(assigned[candidate], seen):
                assigned[candidate] = index
                return True
        return False

    for index in range(len(expected)):
        assign(index, set())
    pairs = [(expected[index], actual[candidate]) for candidate, index in assigned.items()]
    return {'passed': len(pairs) == len(expected) == len(actual),
            'expectedCount': len(expected), 'actualCount': len(actual), 'matchedCount': len(pairs),
            'maxMatchedConfidenceDifference': max((abs(a['confidence'] - b['confidence']) for a, b in pairs), default=0),
            'maxMatchedCoordinateDifference': max((abs(a['box'][field] - b['box'][field])
                                                  for a, b in pairs for field in BOX_FIELDS), default=0)}


def compare_reference(reference, responses, confidence_tolerance, coordinate_tolerance):
    rows = reference['images']
    failures, mismatched, maximum_confidence, maximum_coordinate = [], set(), 0, 0
    for image_index, detections in responses:
        difference = detection_difference(rows[image_index]['detections'], detections,
                                          confidence_tolerance, coordinate_tolerance)
        maximum_confidence = max(maximum_confidence, difference['maxMatchedConfidenceDifference'])
        maximum_coordinate = max(maximum_coordinate, difference['maxMatchedCoordinateDifference'])
        if not difference['passed']:
            mismatched.add(image_index)
            if len(failures) < 10:
                failures.append({'imageIndex': image_index, **difference,
                                 'expectedExample': rows[image_index]['detections'][:5],
                                 'actualExample': detections[:5]})
    return {'passed': not mismatched, 'comparisons': len(responses),
            'mismatchedImageIndices': sorted(mismatched), 'examples': failures,
            'confidenceTolerance': confidence_tolerance, 'coordinateTolerancePixels': coordinate_tolerance,
            'maxMatchedConfidenceDifference': maximum_confidence,
            'maxMatchedCoordinateDifference': maximum_coordinate,
            'matching': 'complete bipartite matching by class and numerical tolerances; output order ignored'}


def send_image(connection, image, sequence, camera, run_id, model_id, version):
    description, png = image
    metadata = {'SchemaVersion': '1.0', 'RequestId': f'dynamic-{run_id}-{sequence}',
                'Sequence': sequence, 'FrameId': sequence, 'CameraId': f'camera-{camera}',
                'ChannelId': 'pv', 'CapturedAtUtc': common.utc_now(),
                'Width': description['width'], 'Height': description['height'],
                'SourcePixelFormat': 'BGR8Packed', 'ImageEncoding': 'Png', 'ModelId': model_id,
                'RequestedModelVersion': version, 'ConfidenceThreshold': 0.25}
    row = {'sequence': sequence, 'ok': False, 'httpStatus': 0, 'requestBytes': 0,
           'responseBytes': 0, 'connectionClosed': False}
    detections = None
    started = time.perf_counter()
    try:
        boundary = 'dynamic-' + run_id
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n'
                'Content-Type: application/json\r\n\r\n').encode()
        body += json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode()
        body += (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame.png"\r\n'
                 'Content-Type: image/png\r\n\r\n').encode() + png + f'\r\n--{boundary}--\r\n'.encode()
        row['requestBytes'] = len(body)
        connection.request('POST', '/infer', body=body,
                           headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
        response = connection.getresponse()
        raw = response.read()
        row.update(httpStatus=response.status, responseBytes=len(raw), connectionClosed=response.will_close)
        result = json.loads(raw)
        if response.status != 200 or result.get('ok') is not True:
            raise RuntimeError(result.get('error', f'HTTP_{response.status}'))
        if any(result.get(field) != metadata[field] for field in IDENTITY_FIELDS):
            raise RuntimeError('RESPONSE_IDENTITY_MISMATCH')
        detections = result['detections']
        if result['detectionCount'] != len(detections) or not 0 <= len(detections) <= 300:
            raise RuntimeError('INVALID_DETECTION_COUNT')
        for detection in detections:
            values = [detection['confidence'], *(detection['box'][key] for key in BOX_FIELDS)]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise RuntimeError('INVALID_DETECTION_VALUES')
            if (not isinstance(detection['classId'], int) or detection['classId'] < 0 or
                    not isinstance(detection['className'], str) or
                    not 0.25 <= detection['confidence'] <= 1):
                raise RuntimeError('INVALID_DETECTION_VALUES')
        for stage in common.STAGES:
            value = result['timingsMs'][stage]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError('INVALID_STAGE_TIMING')
            row[stage] = value
        row.update(ok=True, detectionCount=len(detections))
    except Exception as error:
        row['error'] = str(error)[:500]
        connection.close()
        detections = None
    row['clientMs'] = (time.perf_counter() - started) * 1000
    return row, detections


def measure_rounds(images, endpoint, concurrency, seconds, model_id, version, serial=False):
    connections = [http.client.HTTPConnection(endpoint.hostname, endpoint.port or 80, timeout=30)
                   for _ in range(concurrency)]
    rows, responses, sequence = [], [], 0
    run_id, started_utc = uuid4().hex, common.utc_now()
    started = time.perf_counter()
    cpu_before = time.process_time()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while sequence < len(images) or (not serial and time.perf_counter() - started < seconds):
                gate = threading.Barrier(concurrency + 1)

                def request(slot, current):
                    gate.wait()
                    index = current % len(images)
                    row, detections = send_image(connections[slot], images[index], current,
                                                  slot, run_id, model_id, version)
                    row['imageIndex'] = index
                    return row, detections

                futures = [pool.submit(request, slot, sequence + slot) for slot in range(concurrency)]
                gate.wait()
                for future in futures:
                    row, detections = future.result()
                    rows.append(row)
                    if row['ok']:
                        responses.append((row['imageIndex'], detections))
                sequence += concurrency
    finally:
        for connection in connections:
            connection.close()
    elapsed = time.perf_counter() - started
    successful = [row for row in rows if row['ok']]
    result = {'startedAtUtc': started_utc, 'concurrency': concurrency,
              'elapsedSeconds': elapsed, 'attempts': len(rows), 'successful': len(successful),
              'failed': len(rows) - len(successful), 'successfulImagesPerSecond': len(successful) / elapsed,
              'distinctImagesSucceeded': len({row['imageIndex'] for row in successful}),
              'clientLatencyMs': common.distribution([row['clientMs'] for row in successful]),
              'stagesMs': {stage: common.distribution([row[stage] for row in successful]) for stage in common.STAGES},
              'httpStatuses': dict(Counter(str(row['httpStatus']) for row in rows)),
              'errorTypes': dict(Counter(row['error'] for row in rows if not row['ok'])),
              'clientCpuPercentOfOneCore': (time.process_time() - cpu_before) / elapsed * 100}
    return result, rows, responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:16577')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('baseline', 'serial', 'closed-loop', 'synchronized-round'), required=True)
    parser.add_argument('--concurrency', default='1,2,4,8')
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup-seconds', type=float, default=0)
    parser.add_argument('--model-id', default='best')
    parser.add_argument('--model-version', default='1.0.0')
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--baseline', type=Path, action='append', default=[])
    parser.add_argument('--baseline-output', type=Path)
    parser.add_argument('--confidence-tolerance', type=float, default=1e-5)
    parser.add_argument('--coordinate-tolerance', type=float, default=0.001)
    parser.add_argument('--require-batch-size', type=int)
    args = parser.parse_args()
    levels = [int(value) for value in args.concurrency.split(',')]
    if (not levels or any(value < 1 for value in levels) or args.seconds <= 0 or args.repeats < 1 or
            args.warmup_seconds < 0 or args.confidence_tolerance < 0 or args.coordinate_tolerance < 0):
        parser.error('Invalid concurrency, durations, repeat count or tolerances')
    if args.mode == 'closed-loop' and (args.baseline or args.baseline_output):
        parser.error('Use serial or synchronized-round for numerical comparison; closed-loop measures existing accounting')
    if args.mode == 'baseline' and args.baseline_output is None:
        parser.error('baseline mode requires --baseline-output')
    if args.baseline_output and args.mode not in ('baseline', 'serial'):
        parser.error('Baseline export requires a serial pass')
    endpoint = urlsplit(args.url)
    if endpoint.scheme != 'http' or not endpoint.hostname or endpoint.path not in ('', '/'):
        parser.error('The URL must be an HTTP origin')
    if args.output.exists() or (args.baseline_output and args.baseline_output.exists()):
        parser.error('Output already exists; use new report/baseline filenames')
    document, images, identity, dataset_hash = load_dataset(args.dataset)
    references = {}
    for path in args.baseline:
        reference = json.loads(path.read_text(encoding='utf-8'))
        if (not reference.get('complete') or reference['datasetIdentitySha256'] != dataset_hash or
                len(reference['images']) != len(images) or
                any({key: item[key] for key in identity[index]} != identity[index]
                    for index, item in enumerate(reference['images']))):
            raise ValueError('Baseline dataset identity mismatch: ' + str(path))
        references[str(path)] = reference
    initial = common.control(endpoint, '/api/status')
    active = initial.get('activeModel') or {}
    if not initial['modelReady'] or active.get('id') != args.model_id or active.get('version') != args.model_version:
        raise RuntimeError('The requested model/version must already be active')
    statistics_snapshot(initial)
    generation = active['generation']
    report = {'complete': False, 'mode': args.mode, 'url': args.url, 'initialStatus': initial,
              'dataset': document['summary'], 'datasetIdentitySha256': dataset_hash,
              'measurement': {'input': 'one PNG per HTTP request; all PNGs preloaded',
                  'synchronizedRound': 'one request per camera, simultaneous release, next round waits for all responses',
                  'throughput': 'successful completed images / wall time including final drain',
                  'numericalComparison': 'performed after timed requests; does not enter FPS or latency',
                  'timingWarning': 'tritonRequest includes queueing, copies, GPU execution and callbacks'},
              'warmups': [], 'trials': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output.parent / (args.output.stem + '-samples')
    sample_dir.mkdir(exist_ok=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def run(level, seconds, serial):
        _, before = idle_snapshot(endpoint, generation)
        if args.mode == 'closed-loop':
            result, rows = common.measure(images, endpoint, level, seconds, args.model_id,
                                          args.model_version, args.server_pid)
            responses = []
        else:
            result, rows, responses = measure_rounds(images, endpoint, level, seconds,
                                                      args.model_id, args.model_version, serial)
        result['statistics'] = settled_statistics(endpoint, generation, before, result['successful'])
        result['correctness'] = {name: compare_reference(reference, responses,
            args.confidence_tolerance, args.coordinate_tolerance) for name, reference in references.items()}
        return result, rows, responses

    save()
    try:
        serial = args.mode in ('baseline', 'serial')
        for level in ([1] if serial else levels):
            if args.warmup_seconds:
                warmup, _, _ = run(level, args.warmup_seconds, False)
                report['warmups'].append(warmup)
                save()
                if warmup['failed'] or not warmup['statistics']['settled']:
                    raise RuntimeError('Warmup requests or Triton statistics failed')
            for repeat in range(1 if serial else args.repeats):
                trial, rows, responses = run(level, args.seconds, serial)
                trial['repeat'] = repeat + 1
                samples = sample_dir / f'c{level}-r{repeat + 1}.csv'
                common.write_samples(samples, rows)
                trial['samples'] = str(samples)
                report['trials'].append(trial)
                if (args.baseline_output and not trial['failed'] and trial['statistics']['settled'] and
                        trial['distinctImagesSucceeded'] == len(images)):
                    mapped = dict(responses)
                    baseline = {'complete': True, 'datasetIdentitySha256': dataset_hash,
                        'model': active, 'sourceReport': str(args.output), 'confidenceThreshold': 0.25,
                        'images': [{**item, 'detections': mapped[index]} for index, item in enumerate(identity)]}
                    args.baseline_output.parent.mkdir(parents=True, exist_ok=True)
                    args.baseline_output.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                    report['baselineOutput'] = str(args.baseline_output)
                save()
                print(json.dumps({'mode': args.mode, 'concurrency': level, 'repeat': repeat + 1,
                    'successfulImagesPerSecond': trial['successfulImagesPerSecond'],
                    'p95Ms': trial['clientLatencyMs']['p95'] if trial['clientLatencyMs'] else None,
                    'failed': trial['failed'], 'statistics': trial['statistics']['delta'],
                    'correctnessPassed': all(item['passed'] for item in trial['correctness'].values())}), flush=True)
        report['finalStatus'] = checked_status(endpoint, generation)
        observed = {int(size) for trial in report['trials']
                    for size, count in trial['statistics']['delta']['batchExecutions'].items() if count}
        report['observedBatchSizes'] = sorted(observed)
        report['requiredBatchObserved'] = args.require_batch_size is None or args.require_batch_size in observed
        report['summary'] = []
        for level in sorted({trial['concurrency'] for trial in report['trials']}):
            trials = [trial for trial in report['trials'] if trial['concurrency'] == level]
            report['summary'].append({'concurrency': level,
                'medianSuccessfulImagesPerSecond': statistics.median(trial['successfulImagesPerSecond'] for trial in trials),
                'meanTrialP95Ms': statistics.fmean(trial['clientLatencyMs']['p95'] for trial in trials
                                                 if trial['clientLatencyMs']) if any(trial['successful'] for trial in trials) else None})
        report['passed'] = report['requiredBatchObserved'] and all(
            trial['failed'] == 0 and trial['distinctImagesSucceeded'] == len(images) and
            trial['statistics']['settled'] and all(item['passed'] for item in trial['correctness'].values())
            for trial in report['trials'])
        report['complete'] = True
        save()
        return 0 if report['passed'] else 1
    except Exception as error:
        report['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
