"""Exercise benchmark accounting against a local HTTP server without a GPU."""
import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from urllib.parse import urlsplit
import zlib

import benchmark_http


def png_bytes(red):
    def chunk(name, data):
        return (struct.pack('>I', len(data)) + name + data +
                struct.pack('>I', zlib.crc32(name + data)))

    return (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(bytes([0, red, 0, 0]))) + chunk(b'IEND', b''))


class BenchmarkHttpTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='benchmark-http-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.images = []
        for index in range(2):
            data = png_bytes(40 + index)
            description = {'width': 1, 'height': 1, 'pngBytes': len(data),
                           'pngRelativePath': f'image-{index}.png'}
            (self.directory / description['pngRelativePath']).write_bytes(data)
            self.images.append((description, data))
        self.attempts = []
        self.attempts_lock = threading.Lock()
        self.fail_every = 0
        self.fail_after = 0
        case = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass

            def send_json(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.send_json(200, {'modelReady': True, 'activeModel': {
                    'id': 'best', 'version': '1.0.0', 'generation': 1}})

            def do_POST(self):
                payload = self.rfile.read(int(self.headers['Content-Length']))
                metadata = json.loads(payload.split(b'\r\n\r\n', 1)[1].split(b'\r\n--', 1)[0])
                with case.attempts_lock:
                    case.attempts.append(metadata['RequestId'])
                    attempt = len(case.attempts)
                if ((case.fail_every and attempt % case.fail_every == 0) or
                        (case.fail_after and attempt > case.fail_after)):
                    self.send_json(503, {'ok': False, 'error': 'SERVICE_BUSY'})
                    return
                result = {field: metadata[field] for field in (
                    'RequestId', 'FrameId', 'ModelId', 'ModelVersion')}
                result.update(ok=True, detectionCount=0, detections=[],
                              timingsMs={stage: 0.1 for stage in benchmark_http.STAGES})
                self.send_json(200, result)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.close_server)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()

    def measure(self):
        return benchmark_http.measure(self.images, urlsplit(self.url), 2, 0.03,
                                      'best', '1.0.0', None)

    def assert_samples(self, result, rows):
        samples = self.directory / 'samples.csv'
        benchmark_http.write_samples(samples, rows)
        with samples.open(newline='', encoding='utf-8') as stream:
            stored = list(csv.DictReader(stream))
        self.assertEqual(len(stored), result['attempts'])
        self.assertEqual(sum(row['ok'] == 'True' for row in stored), result['successful'])
        self.assertEqual(len(rows), result['successful'] + result['failed'])
        self.assertEqual(len(self.attempts), len(rows))
        self.assertEqual(len(set(self.attempts)), len(rows))

    def test_successful_throughput_counts_and_csv(self):
        result, rows = self.measure()
        self.assertGreater(result['successful'], 0)
        self.assertEqual(result['failed'], 0)
        self.assertEqual(result['distinctImagesSucceeded'], 2)
        self.assertGreaterEqual(result['elapsedSeconds'], 0.03)
        self.assertAlmostEqual(result['successfulImagesPerSecond'],
                               result['successful'] / result['elapsedSeconds'])
        self.assert_samples(result, rows)

    def test_mixed_503_counts_failed_attempts_without_retry(self):
        self.fail_every = 2
        result, rows = self.measure()
        self.assertGreater(result['successful'], 0)
        self.assertGreater(result['failed'], 0)
        self.assertEqual(result['httpStatuses']['503'], result['failed'])
        self.assertEqual(result['errorTypes'], {'SERVICE_BUSY': result['failed']})
        self.assertEqual(result['distinctImagesAttempted'], 2)
        self.assertAlmostEqual(result['successfulImagesPerSecond'],
                               result['successful'] / result['elapsedSeconds'])
        self.assert_samples(result, rows)

    def test_all_failed_trial_finishes_cli_report(self):
        self.fail_after = len(self.images)
        manifest = self.directory / 'dataset.json'
        manifest.write_text(json.dumps({'summary': {'images': len(self.images)},
                                       'images': [image for image, _ in self.images]}), encoding='utf-8')
        output = self.directory / 'result.json'
        process = subprocess.run([
            sys.executable, str(Path(benchmark_http.__file__).resolve()),
            '--dataset', str(manifest), '--url', self.url, '--output', str(output),
            '--concurrency', '2', '--seconds', '0.03', '--repeats', '1', '--warmup-seconds', '0'],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(process.stderr, '')
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertTrue(report['complete'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['warmups'][0]['failed'], 0)
        trial = report['trials'][0]
        self.assertGreater(trial['failed'], 0)
        self.assertEqual(trial['successful'], 0)
        self.assertEqual(trial['attempts'], trial['failed'])
        self.assertEqual(trial['successfulImagesPerSecond'], 0)
        self.assertIsNone(report['summary'][0]['meanTrialP50Ms'])
        self.assertIsNone(report['summary'][0]['meanTrialP95Ms'])
        with Path(trial['samples']).open(newline='', encoding='utf-8') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), trial['attempts'])


if __name__ == '__main__':
    unittest.main()
