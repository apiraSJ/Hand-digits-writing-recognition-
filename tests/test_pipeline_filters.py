import cv2
import numpy as np
from unittest import TestCase

from src.pipeline import DigitPredictor, run_pipeline
from src.segmentation import filter_digit_boxes


class FixedRecognizer:
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self.calls = 0

    def predict_canonical(
        self,
        image: np.ndarray,
        use_tta: bool | None = None,
    ) -> tuple[int, np.ndarray]:
        self.calls += 1
        probabilities = np.zeros(10, dtype=np.float32)
        probabilities[3] = self.confidence
        probabilities[4] = 1.0 - self.confidence
        return 3, probabilities


class DigitFilterTests(TestCase):
    def test_filter_digit_boxes_rejects_small_large_and_wide_components(self):
        binary = np.zeros((120, 220), dtype=np.uint8)
        binary[10:40, 10:25] = 255
        binary[10:40, 40:43] = 255
        binary[10:100, 60:150] = 255
        binary[10:40, 155:205] = 255
        boxes = [
            (10, 10, 15, 30),
            (40, 10, 3, 30),
            (60, 10, 90, 90),
            (155, 10, 50, 30),
        ]

        filtered = filter_digit_boxes(
            binary,
            boxes,
            min_area=100,
            max_area=5000,
            min_aspect=0.1,
            max_aspect=1.2,
        )

        self.assertEqual(filtered, [(10, 10, 15, 30)])

    def test_pipeline_splits_before_aspect_filter_and_drops_low_confidence(self):
        image = np.full((100, 150, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10, 20), (24, 49), (0, 0, 0), -1)
        cv2.rectangle(image, (60, 25), (109, 49), (0, 0, 0), -1)
        recognizer: DigitPredictor = FixedRecognizer(confidence=0.79)

        result = run_pipeline(
            image,
            recognizer,
            min_area=100,
            max_area=5000,
            merge_gap=10,
            padding=4,
            max_digits=8,
            min_confidence=0.8,
            use_tta=None,
            min_aspect=0.1,
            max_aspect=1.2,
        )

        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(result.digits, [])
        self.assertEqual(result.full_number, "")

        confident_recognizer: DigitPredictor = FixedRecognizer(confidence=0.8)
        confident_result = run_pipeline(
            image,
            confident_recognizer,
            min_area=100,
            max_area=5000,
            merge_gap=10,
            padding=4,
            max_digits=8,
            min_confidence=0.8,
            use_tta=None,
            min_aspect=0.1,
            max_aspect=1.2,
        )

        self.assertEqual(confident_recognizer.calls, 1)
        self.assertEqual(confident_result.full_number, "3")
        self.assertEqual(len(confident_result.digits), 1)
        self.assertEqual(confident_result.digits[0].box, (10, 20, 15, 30))
