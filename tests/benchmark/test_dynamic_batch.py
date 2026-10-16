"""CPU-only checks for synchronized cameras, batch accounting and frame mapping."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

import benchmark_dynamic_batch as dynamic


def detection(x):
    return {'classId': 0, 'className': 'defect', 'confidence': 0.9,
            'box': {'x1': x, 'y1': 1, 'x2': x + 1, 'y2': 2}}


def state(inferences=0, batches=None):
    batches = batches or {}
    return {'modelReady': True, 'inFlight': 0, 'activeModel': {
        'id': 'best', 'version': '2.0.0', 'generation': 7,
        'batching': {'maxBatchSize': 4, 'preferredBatchSizes': [4],
                     'maxQueueDelayMicroseconds': 1000}, 'instanceCount': 1,
        'executionStatistics': {'model_stats': [{'inference_count': str(inferences),
            'execution_count': str(sum(batches.values())),
            'batch_stats': [{'batch_size': str(size), 'compute_infer': {'count': str(count)}}
                            for size, count in batches.items()]}]}}}


def png_bytes(red):
    def chunk(name, data):
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data))
    pixels = (bytes([0]) + bytes([red, 0, 0]) * 32) * 32
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 32, 32, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


class StatisticsAndMatchingTest(unittest.TestCase):
    def test_statistics_weight_batches_by_their_actual_size(self):
        before = dynamic.statistics_snapshot(state(3, {1: 3}))
        after = dynamic.statistics_snapshot(state(15, {1: 3, 4: 3}))
        delta = dynamic.statistics_delta(before, after)
        self.assertEqual(delta['imagesAccountedFor'], 12)
        self.assertEqual(delta['inferenceCount'], 12)
        self.assertEqual(delta['executionCount'], 3)
        self.assertEqual(delta['batchExecutions'], {'4': 3})

    def test_missing_statistics_are_an_error_instead_of_no_batching(self):
        status = state()
        del status['activeModel']['executionStatistics']
        with self.assertRaisesRegex(ValueError, 'unverified'):
            dynamic.statistics_snapshot(status)

    def test_counter_resets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'decreased'):
            dynamic.statistics_delta(dynamic.statistics_snapshot(state(4, {4: 1})),
                                     dynamic.statistics_snapshot(state(0)))

    def test_delayed_batch_counters_are_retried_after_request_release(self):
        before = dynamic.statistics_snapshot(state())
        delayed = state(4)
        settled = state(4, {4: 1})
        with patch.object(dynamic.common, 'control', side_effect=[delayed, settled]) as control, \
                patch.object(dynamic.time, 'sleep') as sleep:
            result = dynamic.settled_statistics(None, 7, before, 4)
        self.assertTrue(result['settled'])
        self.assertEqual(control.call_count, 2)
        sleep.assert_called_once()

    def test_statistics_cannot_cross_a_generation_change(self):
        with patch.object(dynamic.common, 'control', return_value=state()):
            with self.assertRaisesRegex(RuntimeError, 'generation changed'):
                dynamic.checked_status(None, 6)

    def test_matching_ignores_detection_order_and_uses_complete_assignment(self):
        expected = [detection(0), detection(1)]
        actual = [detection(0.6), detection(-0.6)]
        self.assertTrue(dynamic.detection_difference(expected, actual, 1e-5, 0.7)['passed'])
        self.assertTrue(dynamic.detection_difference(expected, list(reversed(expected)), 1e-5, 0.001)['passed'])

    def test_matching_catches_wrong_images_and_strict_numerical_changes(self):
        self.assertFalse(dynamic.detection_difference([detection(0)], [detection(8)], 1e-5, 0.001)['passed'])
        other = detection(0)
        other['confidence'] += 0.00002
        self.assertFalse(dynamic.detection_difference([detection(0)], [other], 1e-5, 0.001)['passed'])
        self.assertFalse(dynamic.detection_difference([detection(0)], [], 1e-5, 0.001)['passed'])


class DynamicBatchHttpTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dynamic-batch-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = []
        for index in range(4):
            png = png_bytes(80 + index)
            description = {'width': 32, 'height': 32, 'pngBytes': len(png),
                           'pngRelativePath': f'image-{index}.png'}
            (self.root / description['pngRelativePath']).write_bytes(png)
            self.images.append((description, png))
        self.dataset = self.root / 'dataset.json'
        self.dataset.write_text(json.dumps({'summary': {'imageCount': 4},
            'images': [item for item, _ in self.images]}), encoding='utf-8')
        self.lock = threading.Lock()
        self.pending = []
        self.batches = {}
        self.inferences = 0
        self.swap_outputs = False
        self.failure = False
        self.identities = []
        case = self

        def release_pending():
            group, case.pending = case.pending, []
            case.inferences += len(group)
            case.batches[len(group)] = case.batches.get(len(group), 0) + 1
            for event in group:
                event.set()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass

            def send_json(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                with case.lock:
                    status = state(case.inferences, dict(case.batches))
                    status['inFlight'] = len(case.pending)
                self.send_json(200, status)

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                metadata = json.loads(body.split(b'\r\n\r\n', 1)[1].split(b'\r\n--', 1)[0])
                image_index = next(index for index, (_, png) in enumerate(case.images) if png in body)
                if case.failure:
                    self.send_json(503, {'ok': False, 'error': 'SERVICE_BUSY'})
                    return
                event = threading.Event()
                with case.lock:
                    case.identities.append((metadata['RequestId'], metadata['CameraId']))
                    case.pending.append(event)
                    if len(case.pending) == 4:
                        release_pending()
                if not event.wait(0.025):
                    with case.lock:
                        if not event.is_set():
                            release_pending()
                if case.swap_outputs:
                    image_index = (image_index + 1) % len(case.images)
                result = {key: metadata[key] for key in dynamic.IDENTITY_FIELDS}
                result.update(ok=True, detectionCount=1, detections=[detection(image_index + 1)],
                              timingsMs={stage: 0.1 for stage in dynamic.common.STAGES})
                self.send_json(200, result)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def cli(self, name, mode, *extra):
        output = self.root / (name + '.json')
        process = subprocess.run([sys.executable, str(Path(dynamic.__file__).resolve()),
            '--dataset', str(self.dataset), '--url', self.url, '--output', str(output),
            '--mode', mode, '--model-version', '2.0.0', '--seconds', '0.02',
            '--repeats', '1', *map(str, extra)], capture_output=True, text=True, timeout=20)
        return process, json.loads(output.read_text(encoding='utf-8'))

    def test_serial_baseline_and_different_synchronized_frames_with_real_counter_accounting(self):
        baseline = self.root / 'baseline.json'
        process, serial = self.cli('serial', 'baseline', '--baseline-output', baseline)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(serial['passed'])
        saved = json.loads(baseline.read_text())
        self.assertEqual([item['detections'][0]['box']['x1'] for item in saved['images']], [1, 2, 3, 4])
        process, rounds = self.cli('rounds', 'synchronized-round', '--concurrency', '4',
                                   '--baseline', baseline, '--require-batch-size', '4')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(rounds['passed'])
        trial = rounds['trials'][0]
        self.assertEqual(set(trial['statistics']['delta']['batchExecutions']), {'4'})
        self.assertGreater(trial['statistics']['delta']['batchExecutions']['4'], 0)
        self.assertEqual(trial['statistics']['delta']['imagesAccountedFor'], trial['successful'])
        self.assertEqual(trial['distinctImagesSucceeded'], 4)
        self.assertGreater(trial['successfulImagesPerSecond'], 0)
        self.assertEqual(len(set(request for request, _ in self.identities)), len(self.identities))
        self.assertEqual(len({camera for _, camera in self.identities}), 4)

    def test_swapped_frame_outputs_fail_even_when_identity_and_http_succeed(self):
        baseline = self.root / 'baseline.json'
        process, _ = self.cli('serial', 'baseline', '--baseline-output', baseline)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.swap_outputs = True
        process, report = self.cli('swapped', 'synchronized-round', '--concurrency', '4', '--baseline', baseline)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertTrue(report['complete'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['trials'][0]['failed'], 0)
        correctness = report['trials'][0]['correctness'][str(baseline)]
        self.assertEqual(correctness['mismatchedImageIndices'], [0, 1, 2, 3])

    def test_closed_loop_reuses_existing_measurement_and_reconciles_statistics(self):
        process, report = self.cli('closed', 'closed-loop', '--concurrency', '4')
        self.assertEqual(process.returncode, 0, process.stderr)
        trial = report['trials'][0]
        self.assertTrue(trial['statistics']['settled'])
        self.assertEqual(trial['statistics']['delta']['inferenceCount'], trial['successful'])

    def test_required_batch_is_not_inferred_from_client_concurrency(self):
        process, report = self.cli('serial-requires-four', 'serial', '--require-batch-size', '4')
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertTrue(report['complete'])
        self.assertFalse(report['requiredBatchObserved'])
        self.assertEqual(report['observedBatchSizes'], [1])

    def test_all_failed_requests_keep_fps_zero_and_finish_report(self):
        self.failure = True
        process, report = self.cli('failed', 'synchronized-round', '--concurrency', '4')
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertTrue(report['complete'])
        trial = report['trials'][0]
        self.assertEqual(trial['successfulImagesPerSecond'], 0)
        self.assertIsNone(trial['clientLatencyMs'])
        self.assertGreater(trial['failed'], 0)
        self.assertTrue(trial['statistics']['settled'])


if __name__ == '__main__':
    unittest.main()
