"""CPU-only guards for versioned TF32 references and unchanged FP32 precision."""
import copy
import json
import unittest

import test_business as business


class ModelRepositoryTest(unittest.TestCase):
    def test_tf32_reference_uses_the_engine_in_the_model_repository(self):
        directory = business.PROJECT / "tests/fixtures"
        models = json.loads((business.PROJECT / "config/models.json").read_text(encoding="utf-8"))["models"]
        model = next(item for item in models if (item["id"], item["businessVersion"]) == business.TF32_VERSION)
        fixtures = json.loads((directory / "fixtures.json").read_text(encoding="utf-8"))["fixtures"]
        fixture = next(item for item in fixtures if item["modelId"] == model["id"])
        expected = json.loads((directory / "expected-baseline.json").read_text(encoding="utf-8"))

        reference = business.load_tf32_reference(directory, model, fixture, expected)

        self.assertEqual(len(reference["references"]), 36)


class BusinessReferenceTest(unittest.TestCase):
    def setUp(self):
        self.sent = {"RequestId": "frame-1", "Width": 800, "Height": 800}
        self.model = {"classNames": {"15": "scratch"}}
        self.original = {"detectionCount": 1, "detections": [{"classId": 15,
            "className": "scratch", "confidence": 0.9,
            "box": {"x1": 100.0, "y1": 200.0, "x2": 110.0, "y2": 210.0}}]}
        direct = copy.deepcopy(self.original)
        direct["detections"][0]["confidence"] += 0.004
        direct["detections"][0]["box"]["x1"] += 0.076
        self.result = {**self.sent, **copy.deepcopy(direct)}
        self.reference = {"engineSha256": "verified-versioned-engine-sha256",
            "rawSha256": "verified-fixture-sha256", "confidenceTolerance": 1e-5,
            "coordinateTolerance": 0.001,
            "references": [{"batchSize": 4, "batchPosition": 2, **direct}]}

    def check_tf32(self, result):
        return business.check_tf32_result(result, self.sent, self.model, self.reference, self.original)

    def test_original_fp32_assertions_still_reject_tf32_sized_drift(self):
        with self.assertRaises(AssertionError):
            business.check_result(self.result, self.sent, self.model, self.original, 1e-5, 0.001)
        unchanged = {**self.sent, **self.original}
        comparison = business.check_result(unchanged, self.sent, self.model, self.original, 1e-5, 0.001)
        self.assertEqual(comparison["maxConfidenceDifference"], 0)

    def test_same_engine_reference_passes_while_fp32_parity_is_explicitly_false(self):
        comparison = self.check_tf32(self.result)
        self.assertEqual(comparison["matchedReference"], {"batchSize": 4, "batchPosition": 2})
        self.assertEqual(comparison["maxConfidenceDifference"], 0)
        old = comparison["fp32NumericalComparison"]
        self.assertFalse(old["strictThresholdsPassed"])
        self.assertAlmostEqual(old["maxConfidenceDifference"], 0.004)
        self.assertAlmostEqual(old["maxCoordinateDifference"], 0.076)
        self.assertEqual((old["confidenceTolerance"], old["coordinateTolerance"]), (1e-5, 0.001))

    def test_matching_reference_does_not_hide_wrong_frame_identity(self):
        result = copy.deepcopy(self.result)
        result["RequestId"] = "another-frame"
        with self.assertRaises(AssertionError):
            self.check_tf32(result)

    def test_tf32_own_reference_keeps_strict_confidence_and_coordinate_tolerances(self):
        for field in ("confidence", "coordinate"):
            with self.subTest(field=field):
                result = copy.deepcopy(self.result)
                if field == "confidence":
                    result["detections"][0]["confidence"] += 0.00002
                else:
                    result["detections"][0]["box"]["x1"] += 0.002
                with self.assertRaises(AssertionError):
                    self.check_tf32(result)

    def test_tf32_own_reference_still_checks_class_names_finite_values_and_boxes(self):
        for field in ("class", "finite", "box"):
            with self.subTest(field=field):
                result = copy.deepcopy(self.result)
                if field == "class":
                    result["detections"][0]["className"] = "wrong-class"
                elif field == "finite":
                    result["detections"][0]["confidence"] = float("nan")
                else:
                    result["detections"][0]["box"]["x2"] = 900
                with self.assertRaises(AssertionError):
                    self.check_tf32(result)


if __name__ == "__main__":
    unittest.main()
