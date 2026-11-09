#!/usr/bin/env python3
"""Real HTTP regression for the single-port C++ Triton backend; standard library only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import http.client as http_client
import itertools
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from urllib.parse import urlsplit
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from png_helpers import encode_png


FIELDS = (
    "SchemaVersion", "RequestId", "Sequence", "FrameId", "CameraId", "ChannelId",
    "CapturedAtUtc", "Width", "Height", "SourcePixelFormat", "ImageEncoding",
    "ModelId", "ModelVersion", "PixelFormat", "StrideBytes", "RowOrder",
    "ConfidenceThreshold",
)
PIXELS = b"\x00\xff\x80\x01\x02\x03\r\n\x00\x07\x08\x09"
COUNTER = itertools.count(1)
OPTIONS = argparse.Namespace(base_url="http://127.0.0.1:6577", timeout=30.0,
                             max_body_bytes=67108864, max_metadata_bytes=16384,
                             load_workers=8, load_requests=16, status_budget_seconds=2.0,
                             skip_load=False, skip_lifecycle=False)


def metadata(model="best", **changes):
    sequence = next(COUNTER)
    result = {
        "SchemaVersion": "1.0", "RequestId": f"protocol-{sequence}",
        "Sequence": sequence, "FrameId": 34, "CameraId": "camera-01",
        "ChannelId": "channel-01", "CapturedAtUtc": "2026-10-14T01:02:03.123Z",
        "Width": 2, "Height": 2, "SourcePixelFormat": "Mono12Packed",
        "ImageEncoding": "Raw", "ModelId": model, "ModelVersion": "1.0.0",
        "PixelFormat": "RGB8Packed", "StrideBytes": 6, "RowOrder": "TopDown",
    }
    result.update(changes)
    return result


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def multipart(request, image=PIXELS, parts=None, metadata_bytes=None,
              metadata_type="application/json", image_type="application/octet-stream"):
    boundary = "triton-protocol-" + uuid.uuid4().hex
    if parts is None:
        parts = [("metadata", metadata_type, json_bytes(request) if metadata_bytes is None else metadata_bytes),
                 ("image", image_type, image)]
    body = bytearray()
    for name, content_type, content in parts:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{name}.raw\"\r\n".encode())
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), "multipart/form-data; boundary=" + boundary


@dataclass
class Response:
    status: int
    headers: dict
    body: bytes
    elapsed: float

    def json(self):
        try:
            value = json.loads(self.body)
        except (ValueError, UnicodeError) as error:
            raise AssertionError(f"HTTP {self.status} did not return JSON: {self.body[:500]!r}") from error
        if not isinstance(value, dict):
            raise AssertionError(f"Expected JSON object: {value!r}")
        return value


def endpoint():
    parsed = urlsplit(OPTIONS.base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.path not in ("", "/"):
        raise ValueError("--base-url must be an http://host:port endpoint without a path")
    return parsed.hostname, parsed.port or 80


def http(method, path, body=None, content_type=None, sent=None, extra_headers=None):
    host, port = endpoint()
    connection = http_client.HTTPConnection(host, port, timeout=OPTIONS.timeout)
    headers = {"Connection": "close"}
    if content_type:
        headers["Content-Type"] = content_type
    if extra_headers:
        headers.update(extra_headers)
    started = time.perf_counter()
    try:
        connection.request(method, path, body=body, headers=headers)
        if sent is not None:
            sent.set()
        response = connection.getresponse()
        data = response.read()
        return Response(response.status, {key.lower(): value for key, value in response.getheaders()},
                        data, time.perf_counter() - started)
    finally:
        connection.close()


def raw_http(body, content_type, framing_headers):
    host, port = endpoint()
    headers = [("Host", f"{host}:{port}"), ("Connection", "close"), ("Content-Type", content_type)]
    headers.extend(framing_headers)
    wire = b"POST /infer HTTP/1.1\r\n" + b"".join(f"{key}: {value}\r\n".encode("ascii") for key, value in headers)
    started = time.perf_counter()
    with socket.create_connection((host, port), timeout=OPTIONS.timeout) as connection:
        connection.sendall(wire + b"\r\n" + body)
        response = http_client.HTTPResponse(connection)
        response.begin()
        data = response.read()
        return Response(response.status, {key.lower(): value for key, value in response.getheaders()},
                        data, time.perf_counter() - started)


def infer(request, image=PIXELS, **parts_options):
    body, content_type = multipart(request, image, **parts_options)
    return http("POST", "/infer", body, content_type)


def raw_image(pixel_format, bottom_up=False, padding=0, grayscale=False):
    width, height = 8, 6
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            gray = (x * 17 + y * 31) % 256
            rgb = (gray, gray, gray) if grayscale else ((y * 41 + x * 17) % 256,
                  (y * 23 + x * 29) % 256, (y * 11 + x * 37) % 256)
            row.extend((gray,) if pixel_format == "Mono8" else rgb[::-1] if pixel_format == "BGR8Packed" else rgb)
        rows.append(bytes(row) + b"\xa5" * padding)
    if bottom_up:
        rows.reverse()
    stride = width * (1 if pixel_format == "Mono8" else 3) + padding
    return b"".join(rows), {"Width": width, "Height": height, "PixelFormat": pixel_format,
                            "StrideBytes": stride, "RowOrder": "BottomUp" if bottom_up else "TopDown"}


def configured_models():
    path = Path(__file__).resolve().parents[2] / "config/models.json"
    return json.loads(path.read_text(encoding="utf-8"))["models"]


class ProtocolTests(unittest.TestCase):
    observations = {}

    @classmethod
    def setUpClass(cls):
        response = http("POST", "/api/models/select", json_bytes({"ModelId": "best", "ModelVersion": "1.0.0"}), "application/json")
        if response.status != 200 or not response.json().get("ok"):
            raise AssertionError(f"Could not start an already deployed backend: HTTP {response.status} {response.body[:500]!r}")
        cls.wait_ready()

    def setUp(self):
        self.select_model("best")

    def select_model(self, model, version="1.0.0"):
        selection = {"ModelId": model, "ModelVersion": version}
        body = self.assert_json(http("POST", "/api/models/select", json_bytes(selection), "application/json"), 200)
        self.assertEqual(set(body), {"ok", "modelReady", "activeModel"})
        self.assertEqual(set(body["activeModel"]), {"id", "version"})
        self.assertEqual(body["activeModel"]["id"], model)
        self.assertEqual(body["activeModel"]["version"], version)
        self.assertTrue(body["modelReady"])
        return body

    @classmethod
    def wait_ready(cls):
        deadline = time.monotonic() + OPTIONS.timeout
        last = None
        while time.monotonic() < deadline:
            last = http("GET", "/api/status")
            if last.status == 200 and last.json().get("modelReady") is True:
                return
            time.sleep(0.05)
        raise AssertionError(f"Backend did not become ready: {last}")

    def assert_json(self, response, status):
        self.assertEqual(response.status, status, response.body[:1000])
        self.assertTrue(response.headers.get("content-type", "").lower().startswith("application/json"))
        return response.json()

    def assert_echo(self, body, request, skip=()):
        for key in FIELDS:
            if key in request and key not in skip:
                self.assertIn(key, body, key)
                self.assertEqual(body[key], request[key], key)
                self.assertIs(type(body[key]), type(request[key]), key)

    def assert_error(self, response, status, code=None, request=None, skip_echo=()):
        body = self.assert_json(response, status)
        self.assertIs(body.get("ok"), False)
        self.assertIsInstance(body.get("message"), str)
        self.assertTrue(body["message"])
        self.assertIsInstance(body.get("error"), str)
        if code:
            self.assertEqual(body["error"], code)
        if request is not None:
            self.assert_echo(body, request, skip_echo)
        return body

    def assert_success(self, response, request):
        body = self.assert_json(response, 200)
        self.assertIs(body.get("ok"), True)
        self.assert_echo(body, request)
        self.assertAlmostEqual(body["appliedConfidenceThreshold"], request.get("ConfidenceThreshold", 0.25), places=6)
        self.assertEqual(body["detectionCount"], len(body["detections"]))
        self.assertIsInstance(body["timingsMs"], dict)
        self.assertIn("tritonRequest", body["timingsMs"])
        for key in ("imagePath", "sourceImagePath", "poolSlot", "imageUrl", "confidenceThreshold"):
            self.assertNotIn(key, body)
        for detection in body["detections"]:
            self.assertIsInstance(detection["classId"], int)
            self.assertGreaterEqual(detection["classId"], 0)
            self.assertIsInstance(detection["className"], str)
            self.assertTrue(detection["className"])
            confidence = detection["confidence"]
            self.assertTrue(math.isfinite(confidence))
            self.assertGreaterEqual(confidence, request.get("ConfidenceThreshold", 0.25))
            self.assertLessEqual(confidence, 1)
            box = detection["box"]
            for name in ("x1", "y1", "x2", "y2"):
                self.assertTrue(math.isfinite(box[name]))
            self.assertGreater(box["x2"], box["x1"])
            self.assertGreater(box["y2"], box["y1"])
            self.assertGreaterEqual(box["x1"], 0)
            self.assertGreaterEqual(box["y1"], 0)
            self.assertLessEqual(box["x2"], request["Width"] - 1)
            self.assertLessEqual(box["y2"], request["Height"] - 1)
        return body

    def assert_equivalent(self, first, second):
        self.assertEqual(first["detectionCount"], second["detectionCount"])
        for expected, actual in zip(first["detections"], second["detections"]):
            self.assertEqual(actual["classId"], expected["classId"])
            self.assertEqual(actual["className"], expected["className"])
            self.assertAlmostEqual(actual["confidence"], expected["confidence"], delta=1e-6)
            for name in ("x1", "y1", "x2", "y2"):
                self.assertAlmostEqual(actual["box"][name], expected["box"][name], delta=1e-3)

    def assert_model_catalog(self, status):
        configured = configured_models()
        expected = {(model["id"], model["businessVersion"]): model for model in configured}
        self.assertEqual(len(expected), len(configured), "Duplicate configured model versions")
        self.assertEqual({model["id"] for model in status["models"]}, {key[0] for key in expected})
        versions = [(model["id"], version) for model in status["models"] for version in model["versions"]]
        self.assertEqual({(model_id, version["version"]) for model_id, version in versions}, set(expected))
        self.assertEqual(len(versions), len(expected), "Duplicate catalog model versions")
        active = status["activeModel"]
        for model_id, version in versions:
            key = model_id, version["version"]
            configuration = expected[key]
            batching = {"maxBatchSize": 0, "preferredBatchSizes": [], "maxQueueDelayMicroseconds": 0}
            batching.update(configuration.get("batching", {}))
            with self.subTest(model=model_id, version=version["version"]):
                selected = key == (active["id"], active["version"])
                self.assertIs(version["active"], selected)
                self.assertIs(version["ready"], selected)
                self.assertIs(version["available"], True)
                self.assertEqual(version["tritonVersion"], configuration["tritonVersion"])
                self.assertEqual(version["backend"], "tensorrt")
                self.assertEqual(version["batchSize"], 1)
                self.assertEqual(version["batching"], batching)
                self.assertEqual(version["instanceCount"], configuration.get("instanceCount", 1))
                batch_axis = -1 if batching["maxBatchSize"] > 0 else 1
                size = configuration["inputSize"]
                self.assertEqual(version["inputs"], [{"name": "images", "datatype": "FP32",
                                                      "shape": [batch_axis, 3, size, size]}])
                self.assertEqual(version["outputs"], [{"name": "output0", "datatype": "FP32",
                                                       "shape": [batch_axis, 300, 6]}])

    def test_010_status_combines_readiness_catalog_info_and_capabilities(self):
        info = self.assert_json(http("GET", "/api/status"), 200)
        self.assertIs(info["ok"], True)
        self.assertIs(info["serviceAlive"], True)
        self.assertIs(info["modelReady"], True)
        self.assertEqual(info["state"], "running")
        self.assertEqual(info["activeModel"]["id"], "best")
        self.assertEqual(info["activeModel"]["version"], "1.0.0")
        self.assertGreaterEqual(info["uptimeMs"], 0)
        self.assertEqual(info["schemaVersion"], "1.0")
        self.assertEqual(info["apiVersion"]["major"], 1)
        self.assertIn("version", info["triton"])
        self.assertIn("switching", info)
        self.assertIn("retiringModels", info)
        self.assertEqual(set(info["pixelFormats"]), {"Mono8", "RGB8Packed", "BGR8Packed"})
        self.assertEqual(set(info["imageEncodings"]), {"Png", "Raw"})
        self.assert_model_catalog(info)
        execution = info["activeModel"]["executionStatistics"]["model_stats"]
        self.assertEqual(len(execution), 1)
        self.assertEqual(str(execution[0]["version"]), str(info["activeModel"]["tritonVersion"]))
        self.assertIn("batch_stats", execution[0])
        for path in ("/health", "/api/health", "/api/info", "/api/models"):
            self.assert_error(http("GET", path), 404, "NOT_FOUND")
        for path in ("/api/start", "/api/stop"):
            self.assert_error(http("POST", path, b"{}", "application/json"), 404, "NOT_FOUND")
        for origin in ("http://127.0.0.1:8090", "http://localhost:8090"):
            with self.subTest(cors_origin=origin):
                response = http("GET", "/api/status", extra_headers={"Origin": origin})
                self.assert_json(response, 200)
                self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
                self.assertIn("origin", response.headers.get("vary", "").lower())
                response = http("OPTIONS", "/infer", extra_headers={"Origin": origin,
                    "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
                self.assertEqual(response.status, 204)
                self.assertEqual(response.body, b"")
                self.assertEqual(response.headers.get("access-control-allow-origin"), origin)
                self.assertIn("POST", response.headers.get("access-control-allow-methods", ""))
                self.assertEqual(response.headers.get("access-control-allow-headers", "").lower(), "content-type")
        self.assert_error(http("GET", "/api/status", extra_headers={"Origin": "http://example.invalid"}),
                          403, "INVALID_ORIGIN")
        invalid_preflights = [
            ("/infer", {"Origin": "http://localhost:8090", "Access-Control-Request-Method": "GET"}),
            ("/api/infer", {"Origin": "http://localhost:8090", "Access-Control-Request-Method": "POST"}),
            ("/infer", {"Origin": "http://localhost:8090", "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "x-custom"}),
            ("/infer", {"Access-Control-Request-Method": "POST"}),
        ]
        for path, headers in invalid_preflights:
            with self.subTest(invalid_preflight=headers):
                self.assert_error(http("OPTIONS", path, extra_headers=headers), 403, "INVALID_PREFLIGHT")

    def test_015_catalog_and_switching_are_explicit_and_missing_targets_preserve_selection(self):
        catalog = self.assert_json(http("GET", "/api/status"), 200)
        self.assert_model_catalog(catalog)
        for model in catalog["models"]:
            for version in model["versions"]:
                selected = self.select_model(model["id"], version["version"])
                status = self.assert_json(http("GET", "/api/status"), 200)
                self.assert_model_catalog(status)
                self.assertEqual(status["activeModel"]["id"], selected["activeModel"]["id"])
                self.assertEqual(status["activeModel"]["version"], selected["activeModel"]["version"])
                request = metadata(model["id"], ModelVersion=version["version"])
                self.assert_success(infer(request), request)
        self.select_model("yolo26n")
        status = self.assert_json(http("GET", "/api/status"), 200)
        self.assertEqual(sum(version["ready"] for model in status["models"] for version in model["versions"]), 1)
        self.assertEqual(status["activeModel"]["id"], "yolo26n")
        request = metadata("best")
        self.assert_error(infer(request), 409, "MODEL_NOT_ACTIVE", request)
        for selection, code in (({"ModelId": "unknown", "ModelVersion": "1.0.0"}, "MODEL_NOT_FOUND"),
                                ({"ModelId": "best", "ModelVersion": "missing"}, "MODEL_VERSION_NOT_FOUND")):
            for path in ("/api/models/select",):
                body = self.assert_error(http("POST", path, json_bytes(selection), "application/json"), 404, code)
                self.assertEqual(body["activeModel"]["id"], "yolo26n")
                self.assertTrue(body["modelReady"])
        for selection in ({}, {"ModelId": "best"}, {"ModelId": "best", "ModelVersion": 1}):
            self.assert_error(http("POST", "/api/models/select", json_bytes(selection), "application/json"), 400)
        request = metadata("yolo26n")
        self.assert_success(infer(request), request)

    def test_020_raw_success_two_models_full_uint64_and_unknown_field(self):
        for model in ("best", "yolo26n"):
            with self.subTest(model=model):
                self.select_model(model)
                request = metadata(model, Sequence=2**64 - 1, FrameId=2**64 - 1, ConfidenceThreshold=0.5)
                request["ok"] = False
                self.assert_success(infer(request), request)

    def test_030_formats_stride_and_row_order_preserve_pixels(self):
        for model in ("best", "yolo26n"):
            self.select_model(model)
            for grayscale in (False, True):
                canonical, fields = raw_image("RGB8Packed", grayscale=grayscale)
                request = metadata(model, **fields)
                baseline = self.assert_success(infer(request, canonical), request)
                formats = ("Mono8", "RGB8Packed", "BGR8Packed") if grayscale else ("RGB8Packed", "BGR8Packed")
                for pixel_format in formats:
                    for bottom_up in (False, True):
                        with self.subTest(model=model, grayscale=grayscale, pixel_format=pixel_format, bottom_up=bottom_up):
                            image, fields = raw_image(pixel_format, bottom_up, padding=5, grayscale=grayscale)
                            request = metadata(model, **fields)
                            actual = self.assert_success(infer(request, image), request)
                            self.assert_equivalent(baseline, actual)

    def test_035_png_matches_raw_for_gray_color_and_alpha(self):
        for model in ("best", "yolo26n"):
            self.select_model(model)
            for channels in (1, 3, 4):
                with self.subTest(model=model, channels=channels):
                    canonical, fields = raw_image("RGB8Packed", grayscale=channels == 1)
                    request = metadata(model, **fields)
                    baseline = self.assert_success(infer(request, canonical), request)
                    pixels = canonical[0::3] if channels == 1 else canonical
                    if channels == 4:
                        pixels = b"".join(canonical[index:index + 3] + b"\0" for index in range(0, len(canonical), 3))
                    png = encode_png(pixels, fields["Width"], fields["Height"], channels)
                    request["ImageEncoding"] = "Png"
                    for key in ("PixelFormat", "StrideBytes", "RowOrder"):
                        del request[key]
                    for media_type in ("image/png", "application/octet-stream"):
                        actual = self.assert_success(infer(request, png, image_type=media_type), request)
                        self.assert_equivalent(baseline, actual)

    def test_036_png_rejects_invalid_files_dimensions_and_16_bit(self):
        request = metadata(ImageEncoding="Png")
        for key in ("PixelFormat", "StrideBytes", "RowOrder"):
            del request[key]
        png = encode_png(PIXELS, 2, 2)
        for image in (b"", PIXELS, b"\xff\xd8\xff\xe0jpeg", png[:20], png[:33]):
            with self.subTest(invalid_bytes=len(image)):
                self.assert_error(infer(request, image, image_type="image/png"), 400, "INVALID_PNG_IMAGE", request)
        for changes in ({"Width": 3}, {"Height": 3}, {"Width": 0}, {"Height": 2147483648}):
            with self.subTest(dimensions=changes):
                invalid = dict(request, **changes)
                self.assert_error(infer(invalid, png), 400, "INVALID_PNG_IMAGE", invalid)
        image16 = encode_png(b"\x04\0" * 4, 2, 2, channels=1, bit_depth=16)
        self.assert_error(infer(request, image16), 400, "INVALID_PNG_IMAGE", request)
        oversized = bytearray(png)
        oversized[16:24] = (8192).to_bytes(4, "big") * 2
        invalid = dict(request, Width=8192, Height=8192)
        self.assert_error(infer(invalid, oversized), 400, "INVALID_PNG_IMAGE", invalid)
        self.assert_success(infer(request, png), request)

    def test_040_media_types_and_raw_encoded_input_rejection(self):
        request = metadata()
        self.assert_success(infer(request, metadata_type='Application/JSON; charset="UTF-8"',
                                  image_type="Application/Octet-Stream; foo=bar"), request)
        for part_options in ({"metadata_type": "application/json; charset=latin-1"},
                             {"image_type": "image/png"}, {"metadata_type": "text/plain"}):
            with self.subTest(options=part_options):
                self.assert_error(infer(request, **part_options), 415, "UNSUPPORTED_CONTENT_TYPE", request)
        for payload in ({"imagePath": "/tmp/image.png"}, {"imageBase64": "iVBORw0KGgo="}):
            with self.subTest(payload=payload):
                self.assert_error(http("POST", "/infer", json_bytes(payload), "application/json"),
                                  415, "UNSUPPORTED_CONTENT_TYPE")
        payload, content_type = multipart(request)
        self.assert_error(http("POST", "/api/infer", payload, content_type), 404, "NOT_FOUND")
        self.assert_error(infer(request, b"\x89PNG\r\n\x1a\n" + b"encoded-file-bytes"), 400, "INVALID_RAW_IMAGE", request)

    def test_050_multipart_parts_and_json_errors(self):
        request = metadata()
        normal = [("metadata", "application/json", json_bytes(request)),
                  ("image", "application/octet-stream", PIXELS)]
        for parts in (normal[:1], normal[1:], normal + [normal[1]], normal + [normal[0]],
                      normal + [("extra", "text/plain", b"x")]):
            with self.subTest(parts=[item[0] for item in parts]):
                self.assert_error(infer(request, parts=parts), 400, "INVALID_MULTIPART")
        invalid_json = [b"not-json", b"{", b'{"bad":"\xff"}',
                        json_bytes(request)[:-1] + b',"ConfidenceThreshold":1e999}']
        for value in invalid_json:
            with self.subTest(json=value[:80]):
                self.assert_error(infer(request, metadata_bytes=value), 400, "INVALID_JSON")
        for value in ([], None, 12, "text"):
            with self.subTest(non_object=value):
                self.assert_error(infer(request, metadata_bytes=json_bytes(value)), 400, "INVALID_METADATA")

    def test_060_all_required_fields_and_strings(self):
        complete = metadata()
        for key in complete:
            with self.subTest(missing=key):
                request = complete.copy()
                del request[key]
                self.assert_error(infer(request), 400, request=request)
        for key in ("RequestId", "CameraId", "ChannelId", "SourcePixelFormat", "ModelId", "ModelVersion"):
            for value in ("", " \t\r\n", 42, None, "x" * 257):
                with self.subTest(field=key, invalid=value):
                    request = metadata(**{key: value})
                    self.assert_error(infer(request), 400, request=request)
        request = metadata(RequestId="x" * 256, CameraId="相" * 85)
        self.assert_success(infer(request), request)
        request = metadata(CameraId="相" * 86)
        self.assert_error(infer(request), 400, request=request)

    def test_070_numeric_bounds_raw_layout_and_thresholds(self):
        changes = [({"Sequence": -1}, "INVALID_METADATA"), ({"FrameId": 1.5}, "INVALID_METADATA"),
                   ({"Sequence": "12"}, "INVALID_METADATA"), ({"Sequence": True}, "INVALID_METADATA"),
                   ({"FrameId": 2**64}, "INVALID_METADATA"), ({"Width": 0}, "INVALID_RAW_IMAGE"),
                   ({"Height": 2147483648}, "INVALID_RAW_IMAGE"), ({"StrideBytes": 5}, "INVALID_RAW_IMAGE"),
                   ({"Width": 2.0}, "INVALID_METADATA"), ({"StrideBytes": -1}, "INVALID_METADATA"),
                   ({"PixelFormat": "Mono16"}, "UNSUPPORTED_PIXEL_FORMAT"),
                   ({"RowOrder": "Unknown"}, "INVALID_RAW_IMAGE"),
                   ({"ImageEncoding": "Jpeg"}, "UNSUPPORTED_IMAGE_ENCODING")]
        for change, code in changes:
            with self.subTest(change=change):
                request = metadata(**change)
                skip = ("FrameId",) if change.get("FrameId") == 2**64 else ()
                self.assert_error(infer(request), 400, code, request, skip)
        for value in (1.1, 1.00000001, -1e-50, "0.25", True, None):
            with self.subTest(threshold=value):
                request = metadata(ConfidenceThreshold=value)
                self.assert_error(infer(request), 400, "INVALID_CONFIDENCE_THRESHOLD", request)
        for value in (0, 1, 0.25):
            with self.subTest(valid_threshold=value):
                request = metadata(ConfidenceThreshold=value)
                self.assert_success(infer(request), request)
        request = metadata()
        for image in (b"", b"\0" * 11, b"\0" * 13):
            with self.subTest(pixel_bytes=len(image)):
                self.assert_error(infer(request, image), 400, "INVALID_RAW_IMAGE", request)

    def test_080_utc_calendar_validation_and_valid_offsets(self):
        invalid = ("2026-02-29T01:02:03Z", "2026-10-14T25:02:03Z", "2026-10-14T01:60:03Z",
                   "2026-10-14T01:02:60Z", "2026-10-14T01:02:03+08:00", "0000-01-01T00:00:00Z",
                   "2026-13-01T00:00:00Z", "2026-04-31T00:00:00Z", "2026-10-14T01:02:03z",
                   "2026-10-14T01:02:03.1234567890Z")
        for value in invalid:
            with self.subTest(invalid_utc=value):
                request = metadata(CapturedAtUtc=value)
                self.assert_error(infer(request), 400, "INVALID_METADATA", request)
        for value in ("2024-02-29T01:02:03Z", "2026-10-14T01:02:03.1+00:00",
                      "2026-10-14T01:02:03.123456789Z"):
            with self.subTest(valid_utc=value):
                request = metadata(CapturedAtUtc=value)
                self.assert_success(infer(request), request)

    def test_090_schema_model_and_version_mismatch(self):
        for change, status, code in (({"SchemaVersion": "2.0"}, 400, "UNSUPPORTED_SCHEMA_VERSION"),
                                     ({"ModelId": "other"}, 409, "MODEL_MISMATCH"),
                                     ({"ModelVersion": "missing"}, 409, "MODEL_VERSION_MISMATCH"),
                                     ({"ModelVersion": "2.0.0"}, 409, "MODEL_NOT_ACTIVE")):
            with self.subTest(change=change):
                request = metadata(**change)
                self.assert_error(infer(request), status, code, request)

    def test_100_content_length_and_chunked_rejection(self):
        body, content_type = multipart(metadata())
        malformed = [([], body), ([("Content-Length", "abc")], body),
                     ([("Content-Length", str(len(body))), ("Content-Length", str(len(body)))], body)]
        for headers, wire_body in malformed:
            with self.subTest(headers=headers):
                self.assert_error(raw_http(wire_body, content_type, headers), 400)
        chunks = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
        self.assert_error(raw_http(chunks, content_type, [("Transfer-Encoding", "chunked")]),
                          400, "INVALID_CONTENT_LENGTH")

    def test_110_total_and_metadata_size_limits(self):
        body, content_type = multipart(metadata())
        response = raw_http(b"", content_type, [("Content-Length", str(OPTIONS.max_body_bytes + 1))])
        self.assert_error(response, 413, "PAYLOAD_TOO_LARGE")
        request = metadata()
        raw = json_bytes(request)
        request["ExtraPadding"] = "x" * (OPTIONS.max_metadata_bytes - len(raw) - len(b',"ExtraPadding":""'))
        self.assertEqual(len(json_bytes(request)), OPTIONS.max_metadata_bytes)
        self.assert_success(infer(request), request)
        request["ExtraPadding"] += "x"
        self.assert_error(infer(request), 413, "METADATA_TOO_LARGE")

    def test_120_control_rejects_invalid_json_without_changing_readiness(self):
        for path in ("/api/models/select",):
            for payload in (b"[]", b"null", b'"text"', b'{"force":true}', b"not-json"):
                with self.subTest(path=path, payload=payload):
                    self.assert_error(http("POST", path, payload, "application/json"), 400)
                    self.assertTrue(self.assert_json(http("GET", "/api/status"), 200)["modelReady"])
            self.assert_error(http("POST", path, b"{}", "text/plain"), 415)
        self.select_model("best")

    def test_130_hot_switch_keeps_old_model_serving_and_retires_it_after_cutover(self):
        if OPTIONS.skip_lifecycle:
            self.skipTest("Switching regression disabled explicitly")
        successes_while_loading = 0
        observed_loading = False
        selection = {"ModelId": "yolo26n", "ModelVersion": "1.0.0"}
        with ThreadPoolExecutor(max_workers=1) as pool:
            switching = pool.submit(http, "POST", "/api/models/select", json_bytes(selection), "application/json")
            while not switching.done():
                status = self.assert_json(http("GET", "/api/status"), 200)
                self.assertTrue(status["modelReady"])
                self.assertEqual(status["state"], "running")
                if status["switching"]:
                    observed_loading = True
                    self.assertEqual(status["activeModel"]["id"], "best")
                    self.assertEqual(status["pendingModel"]["id"], "yolo26n")
                    request = metadata("best")
                    response = infer(request)
                    if response.status == 200:
                        self.assert_success(response, request)
                        successes_while_loading += 1
                    else:
                        self.assert_error(response, 409, "MODEL_NOT_ACTIVE", request)
            selected = self.assert_json(switching.result(), 200)
        self.assertTrue(observed_loading)
        self.assertGreater(successes_while_loading, 0, "Old model did not serve a request during candidate loading")
        self.assertEqual(selected["activeModel"]["id"], "yolo26n")
        request = metadata("yolo26n")
        self.assert_success(infer(request), request)
        deadline = time.monotonic() + OPTIONS.timeout
        while True:
            status = self.assert_json(http("GET", "/api/status"), 200)
            self.assertTrue(status["modelReady"])
            if not status["retiringModels"]:
                break
            self.assertLess(time.monotonic(), deadline, "Old model was not fully unloaded")
            time.sleep(0.01)
        self.observations["hot_switch"] = {"old_model_requests_during_loading": successes_while_loading}

    def test_140_status_responds_during_concurrent_inference(self):
        if OPTIONS.skip_load:
            self.skipTest("Load regression disabled explicitly")
        sent = threading.Event()
        image, fields = raw_image("RGB8Packed", padding=5)

        def submit(index):
            request = metadata("best", **fields)
            payload, content_type = multipart(request, image)
            return request, http("POST", "/infer", payload, content_type, sent)

        samples = []
        with ThreadPoolExecutor(max_workers=OPTIONS.load_workers) as pool:
            futures = [pool.submit(submit, index) for index in range(OPTIONS.load_requests)]
            self.assertTrue(sent.wait(OPTIONS.timeout), "No inference upload began")
            observed_pending = False
            while not all(future.done() for future in futures):
                observed_pending = True
                response = http("GET", "/api/status")
                body = self.assert_json(response, 200)
                self.assertIs(body["serviceAlive"], True)
                self.assertIs(body["modelReady"], True)
                samples.append(response.elapsed)
            self.assertTrue(observed_pending, "No overlapping status and inference interval was observed")
            successful = 0
            for future in futures:
                request, response = future.result()
                if response.status == 200:
                    self.assert_success(response, request)
                    successful += 1
                else:
                    self.assert_error(response, 503, "SERVICE_BUSY", request)
            self.assertGreater(successful, 0)
        self.assertTrue(samples)
        self.assertLess(max(samples), OPTIONS.status_budget_seconds,
                        "Status endpoint was blocked by inference work")
        self.observations["status_under_load"] = {
            "requests": OPTIONS.load_requests, "workers": OPTIONS.load_workers,
            "successful_inferences": successful, "status_samples": len(samples),
            "maximum_status_seconds": max(samples), "budget_seconds": OPTIONS.status_budget_seconds,
        }


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.cases.append({"test": test.id(), "status": "passed"})

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        self.cases.append({"test": subtest.id(), "status": "passed" if error is None else "failed"})


def main():
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=OPTIONS.base_url)
    parser.add_argument("--timeout", type=float, default=OPTIONS.timeout)
    parser.add_argument("--max-body-bytes", type=int, default=OPTIONS.max_body_bytes)
    parser.add_argument("--max-metadata-bytes", type=int, default=OPTIONS.max_metadata_bytes)
    parser.add_argument("--load-workers", type=int, default=OPTIONS.load_workers)
    parser.add_argument("--load-requests", type=int, default=OPTIONS.load_requests)
    parser.add_argument("--status-budget-seconds", type=float, default=OPTIONS.status_budget_seconds)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-lifecycle", action="store_true")
    parser.add_argument("--match", help="Run only test names containing this substring")
    parser.add_argument("--output", type=Path, required=True)
    OPTIONS = parser.parse_args()
    endpoint()
    if OPTIONS.max_metadata_bytes < 1024 or OPTIONS.load_workers < 1 or OPTIONS.load_requests < OPTIONS.load_workers:
        parser.error("metadata limit must be >=1024 and load requests >= workers >=1")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    if OPTIONS.match:
        suite = unittest.TestSuite(test for test in suite if OPTIONS.match in test.id())
    if suite.countTestCases() == 0:
        parser.error("--match selected no tests")
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    report = {
        "base_url": OPTIONS.base_url, "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "cases": result.cases, "observations": ProtocolTests.observations,
        "failures": [{"test": test.id(), "traceback": traceback} for test, traceback in result.failures],
        "errors": [{"test": test.id(), "traceback": traceback} for test, traceback in result.errors],
        "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "limits": ["Synthetic raw pixels test protocol and equivalent layouts; they are not labeled accuracy data.",
                   "Concurrent status checks are a regression bound, not a throughput benchmark."],
    }
    OPTIONS.output.parent.mkdir(parents=True, exist_ok=True)
    OPTIONS.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
