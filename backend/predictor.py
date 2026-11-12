"""Model loading and prediction helpers for the handwritten digit backend."""

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import run_pipeline
from src.recognizer import DigitRecognizer

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_model.keras")


class Predictor:
    """Thin wrapper around :class:`DigitRecognizer` and the recognition pipeline."""

    def __init__(self, model_path=None, use_tta=True):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.use_tta_default = bool(use_tta)
        self.recognizer = None
        self.loaded = False
        self.error = None

    def load(self):
        """Load the Keras model and warm it up. Safe to call multiple times."""
        try:
            self.recognizer = DigitRecognizer(
                self.model_path, use_tta=self.use_tta_default
            )
            # Warm up both prediction paths so the first live request is not slow.
            blank = np.zeros((28, 28), dtype=np.uint8)
            self.recognizer.predict_canonical(blank, use_tta=False)
            if self.use_tta_default:
                self.recognizer.predict_canonical(blank, use_tta=True)
            self.loaded = True
            self.error = None
        except Exception as exc:  # pragma: no cover - depends on environment
            self.recognizer = None
            self.loaded = False
            self.error = str(exc)

    def predict(
        self,
        image,
        min_area=100,
        max_area=5000,
        merge_gap=10,
        padding=4,
        max_digits=8,
        min_confidence=0.8,
        use_tta=None,
        min_aspect=0.1,
        max_aspect=1.2,
    ):
        """Run the full pipeline on ``image`` and return a ``PipelineResult``."""
        if not self.loaded or self.recognizer is None:
            raise RuntimeError(self.error or "Model is not loaded")
        if use_tta is None:
            use_tta = self.use_tta_default

        return run_pipeline(
            image,
            self.recognizer,
            min_area=min_area,
            max_area=max_area,
            merge_gap=merge_gap,
            padding=padding,
            max_digits=max_digits,
            min_confidence=min_confidence,
            use_tta=use_tta,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
        )