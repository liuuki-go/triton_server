#!/usr/bin/env python3
"""Verify PNG and raw HTTP inputs against saved TensorRT references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit
from uuid import uuid4


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tests"))
from png_helpers import encode_png

FP32_REFERENCE_VERSIONS = frozenset((("best", "1.0.0"), ("best", "2.0.0"), ("yolo26n", "1.0.0")))
TF32_VERSION = ("best", "3.0.0")


def multipart(metadata, pixels):
    boundary = "triton-back-" + uuid4().hex
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"; filename="metadata.json"\r\n'
            'Content-Type: application/json; charset=utf-8\r\n\r\n').encode()
    png = metadata["ImageEncoding"] == "Png"
    filename, media_type = ("frame.png", "image/png") if png else ("frame.raw", "application/octet-stream")
    middle = (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
              f'Content-Type: {media_type}\r\n\r\n').encode()
    return (head + json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode() + middle +
            pixels + f"\r\n--{boundary}--\r\n".encode(), "multipart/form-data; boundary=" + boundary)


class Client:
    def __init__(self, url):
        endpoint = urlsplit(url)
        assert endpoint.scheme == "http" and endpoint.hostname
        self.connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port or 80, timeout=90)

    def close(self):
        self.connection.close()

    def call(self, path, metadata=None, pixels=None):
        if pixels is not None:
            body, media_type = multipart(metadata, pixels)
        elif metadata is not None:
            body, media_type = json.dumps(metadata, allow_nan=False).encode(), "application/json"
        else:
            body, media_type = None, "application/json"
        self.connection.request("GET" if body is None else "POST", path, body,
                                {"Content-Type": media_type})
        response = self.connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 200 and result["ok"], (path, response.status, result)
        return result


def metadata_for(fixture, model, index, encoding):
    result = {"SchemaVersion": "1.0", "RequestId": "real-frame-" + uuid4().hex,
            "Sequence": index, "FrameId": 10000 + index, "CameraId": f"camera-{index}",
            "ChannelId": f"channel-{index}",
            "CapturedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "Width": fixture["width"], "Height": fixture["height"],
            "SourcePixelFormat": fixture["pixelFormat"], "ImageEncoding": encoding,
            "ModelId": model["id"], "ModelVersion": model["businessVersion"],
            "ConfidenceThreshold": 0.25}
    if encoding == "Raw":
        result.update(PixelFormat=fixture["pixelFormat"], StrideBytes=fixture["strideBytes"], RowOrder=fixture["rowOrder"])
    return result


def check_result(result, sent, model, reference, confidence_tolerance, coordinate_tolerance):
    for field, value in sent.items():
        assert result[field] == value, (field, result.get(field), value)
    detections = result["detections"]
    assert len(detections) == result["detectionCount"] == reference["detectionCount"] > 0
    max_confidence_difference, max_coordinate_difference = 0.0, 0.0
    for actual, expected in zip(detections, reference["detections"]):
        assert actual["classId"] == expected["classId"]
        assert actual["className"] == model["classNames"][str(actual["classId"])]
        assert math.isfinite(actual["confidence"])
        delta = abs(actual["confidence"] - expected["confidence"])
        assert delta <= confidence_tolerance
        max_confidence_difference = max(max_confidence_difference, delta)
        for field in ("x1", "y1", "x2", "y2"):
            value = actual["box"][field]
            assert math.isfinite(value)
            delta = abs(value - expected["box"][field])
            assert delta <= coordinate_tolerance, (field, value, expected["box"][field])
            max_coordinate_difference = max(max_coordinate_difference, delta)
        assert 0 <= actual["box"]["x1"] < actual["box"]["x2"] < sent["Width"]
        assert 0 <= actual["box"]["y1"] < actual["box"]["y2"] < sent["Height"]
    return {"detectionCount": len(detections), "maxConfidenceDifference": max_confidence_difference,
            "maxCoordinateDifference": max_coordinate_difference}


def load_tf32_reference(directory, model, fixture, expected):
    reference = json.loads((directory / "expected-best-tf32.json").read_text(encoding="utf-8"))
    assert (reference["modelId"], reference["businessVersion"]) == TF32_VERSION
    assert reference["tritonVersion"] == model["tritonVersion"] == 3
    manifest = json.loads((PROJECT / "models/manifest.json").read_text(encoding="utf-8"))
    entries = [item for item in manifest["models"] if item["id"] == model["id"] and
               item["tritonVersion"] == model["tritonVersion"]]
    assert len(entries) == 1, "The versioned Engine manifest entry must be unique"
    assert reference["engineSha256"] == entries[0]["sha256"]
    engine = PROJECT / entries[0]["enginePath"]
    assert engine.resolve() == (PROJECT / "models" / model["tritonName"] / str(model["tritonVersion"]) / "model.plan").resolve()
    assert hashlib.sha256(engine.read_bytes()).hexdigest() == reference["engineSha256"]
    assert reference["rawSha256"] == fixture["sha256"]
    assert (reference["width"], reference["height"]) == (fixture["width"], fixture["height"])
    assert reference["inputSize"] == model["inputSize"]
    assert reference["confidenceThreshold"] == 0.25
    assert reference["confidenceTolerance"] == expected["confidenceTolerance"] == 1e-5
    assert reference["coordinateTolerance"] == expected["coordinateTolerance"] == 0.001
    maximum = model["batching"]["maxBatchSize"]
    assert maximum == 8
    entries = reference["references"]
    assert len(entries) == maximum * (maximum + 1) // 2
    assert {(item["batchSize"], item["batchPosition"]) for item in entries} == {
        (size, position) for size in range(1, maximum + 1) for position in range(size)}
    assert all(item["detectionCount"] == len(item["detections"]) > 0 for item in entries)
    return reference


def fp32_difference(result, reference, confidence_tolerance, coordinate_tolerance):
    actual, expected = result["detections"], reference["detections"]
    same_count = len(actual) == len(expected)
    same_classes = [item["classId"] for item in actual] == [item["classId"] for item in expected]
    confidence = max((abs(a["confidence"] - b["confidence"]) for a, b in zip(actual, expected)), default=0)
    coordinate = max((abs(a["box"][field] - b["box"][field]) for a, b in zip(actual, expected)
                      for field in ("x1", "y1", "x2", "y2")), default=0)
    return {"reference": "saved original FP32 TensorRT baseline; numerical comparison only",
            "strictThresholdsPassed": same_count and same_classes and confidence <= confidence_tolerance and coordinate <= coordinate_tolerance,
            "detectionCountMatches": same_count, "orderedClassIdsMatch": same_classes,
            "maxConfidenceDifference": confidence, "maxCoordinateDifference": coordinate,
            "confidenceTolerance": confidence_tolerance, "coordinateTolerance": coordinate_tolerance,
            "comparison": "ordered pairs; count and class agreement are required for strict parity"}


def check_tf32_result(result, sent, model, reference, original):
    matches = []
    for candidate in reference["references"]:
        try:
            comparison = check_result(result, sent, model, candidate,
                                      reference["confidenceTolerance"], reference["coordinateTolerance"])
        except AssertionError:
            continue
        matches.append({"batchSize": candidate["batchSize"], "batchPosition": candidate["batchPosition"], **comparison})
    assert matches, "The TF32 response matches no direct TensorRT reference for this Engine and raw image at batch 1..8"
    closest = min(matches, key=lambda item: (item["maxCoordinateDifference"], item["maxConfidenceDifference"]))
    return {"detectionCount": closest["detectionCount"],
            "maxConfidenceDifference": closest["maxConfidenceDifference"],
            "maxCoordinateDifference": closest["maxCoordinateDifference"],
            "matchedReference": {"batchSize": closest["batchSize"], "batchPosition": closest["batchPosition"]},
            "referenceKind": "same TF32 Engine direct TensorRT; functional transport and concurrency verification",
            "matchedDirectReferences": matches, "engineSha256": reference["engineSha256"],
            "rawSha256": reference["rawSha256"],
            "fp32NumericalComparison": fp32_difference(result, original,
                reference["confidenceTolerance"], reference["coordinateTolerance"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6577")
    parser.add_argument("--report", type=Path, default=PROJECT / "build/verification/business-report.json")
    args = parser.parse_args()
    configurations = json.loads((PROJECT / "config/models.json").read_text(encoding="utf-8"))["models"]
    models = {(item["id"], item["businessVersion"]): item for item in configurations}
    assert len(models) == len(configurations), "Duplicate configured model versions"
    unknown = set(models) - FP32_REFERENCE_VERSIONS - {TF32_VERSION}
    assert not unknown, ("A version-specific numeric reference is required", sorted(unknown))
    directory = PROJECT / "tests/fixtures"
    fixtures = json.loads((directory / "fixtures.json").read_text(encoding="utf-8"))["fixtures"]
    expected = json.loads((directory / "expected-baseline.json").read_text(encoding="utf-8"))
    assert {item["modelId"] for item in fixtures} == {key[0] for key in models}
    assert ("best", "1.0.0") in models, "The original best/1.0.0 baseline must remain covered"
    for fixture in fixtures:
        fixture["pixels"] = (directory / fixture["rawFile"]).read_bytes()
        assert len(fixture["pixels"]) == fixture["bytes"] == fixture["strideBytes"] * fixture["height"]
        assert hashlib.sha256(fixture["pixels"]).hexdigest() == fixture["sha256"]
        assert expected["models"][fixture["modelId"]]["rawSha256"] == fixture["sha256"]
        assert fixture["pixelFormat"] == "BGR8Packed" and fixture["rowOrder"] == "TopDown"
        assert fixture["strideBytes"] == fixture["width"] * 3
        rgb = bytearray(fixture["pixels"])
        rgb[0::3], rgb[2::3] = fixture["pixels"][2::3], fixture["pixels"][0::3]
        fixture["png"] = encode_png(rgb, fixture["width"], fixture["height"])
    tf32_reference = None
    if TF32_VERSION in models:
        fixture = next(item for item in fixtures if item["modelId"] == TF32_VERSION[0])
        tf32_reference = load_tf32_reference(directory, models[TF32_VERSION], fixture, expected)
    report = {"ok": False, "url": args.url, "models": {}, "concurrentRequests": [],
              "limits": ["Numeric parity with a saved direct TensorRT baseline; not labeled accuracy validation.",
                         "TF32 functional success uses that Engine's batch/position references and does not assert FP32 numerical parity.",
                         "Single-image requests across configured model versions; not a hardware-throughput benchmark."]}
    report["validationPolicy"] = {
        "fp32ReferenceVersions": [list(key) for key in sorted(FP32_REFERENCE_VERSIONS)],
        "tf32ReferenceVersion": list(TF32_VERSION) if tf32_reference else None,
        "confidenceTolerance": expected["confidenceTolerance"], "coordinateTolerance": expected["coordinateTolerance"],
        "tf32ReferenceProvenance": {key: tf32_reference[key] for key in (
            "modelId", "businessVersion", "tritonVersion", "engineSha256", "rawSha256")} if tf32_reference else None}

    def request(fixture, model, index, encoding, barrier=None):
        sent = metadata_for(fixture, model, index, encoding)
        client = Client(args.url)
        try:
            if barrier is not None:
                barrier.wait(timeout=20)
            result = client.call("/infer", sent, fixture["png"] if encoding == "Png" else fixture["pixels"])
            if (model["id"], model["businessVersion"]) == TF32_VERSION:
                comparison = check_tf32_result(result, sent, model, tf32_reference, expected["models"][model["id"]])
            else:
                comparison = check_result(result, sent, model, expected["models"][model["id"]],
                                          expected["confidenceTolerance"], expected["coordinateTolerance"])
                comparison["referenceKind"] = "strict saved original FP32 TensorRT baseline"
            return {"ModelId": model["id"], "ModelVersion": model["businessVersion"],
                    "ImageEncoding": encoding, "RequestId": sent["RequestId"],
                    "FrameId": sent["FrameId"], "CameraId": sent["CameraId"], **comparison}
        finally:
            client.close()

    control = Client(args.url)
    try:
        info = control.call("/api/status")
        assert info["serviceAlive"]
        assert info["apiVersion"]["major"] == 1 and info["apiVersion"]["minor"] >= 34
        cases = [(fixture, model) for fixture in fixtures for (model_id, _), model in models.items()
                 if model_id == fixture["modelId"]]
        for index, (fixture, model) in enumerate(cases):
            selected = control.call("/api/models/select", {"ModelId": model["id"], "ModelVersion": model["businessVersion"]})
            assert set(selected) == {"ok", "modelReady", "activeModel"}
            assert set(selected["activeModel"]) == {"id", "version"}
            assert selected["activeModel"]["id"] == model["id"]
            assert selected["activeModel"]["version"] == model["businessVersion"]
            assert selected["modelReady"]
            status = control.call("/api/status")
            assert status["modelReady"] and status["activeModel"]["ready"]
            assert status["activeModel"]["id"] == model["id"]
            assert status["activeModel"]["version"] == model["businessVersion"]
            versions = [version for group in status["models"] for version in group["versions"]]
            assert sum(version["ready"] for version in versions) == 1
            assert sum(version["active"] for version in versions) == 1
            report["models"].setdefault(model["id"], {})[model["businessVersion"]] = {
                encoding: request(fixture, model, index, encoding) for encoding in ("Png", "Raw")}
            barrier = threading.Barrier(8)
            with ThreadPoolExecutor(max_workers=8) as pool:
                report["concurrentRequests"].extend(pool.map(
                    lambda number: request(fixture, model, 100 + number,
                                           "Png" if number % 2 == 0 else "Raw", barrier), range(8)))
        assert len({item["RequestId"] for item in report["concurrentRequests"]}) == 8 * len(cases)
        assert {(model_id, version) for model_id, versions in report["models"].items()
                for version in versions} == set(models)
        report["finalStatus"] = control.call("/api/status")
        assert report["finalStatus"]["modelReady"] and report["finalStatus"]["inFlight"] == 0
        report["ok"] = True
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        control.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "models": report["models"], "report": str(args.report)}))


if __name__ == "__main__":
    main()
