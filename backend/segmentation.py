"""OpenCV-based segmentation helpers for the handwritten digit pipeline.

This module wraps the existing ``src`` segmentation implementation and exposes
a small, stable API that the FastAPI backend (and any tests) can use directly.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np

from src.preprocessing import mnist_canonical, to_binary
from src.segmentation import (
    extract_roi,
    filter_digit_boxes,
    find_digit_contours,
    group_boxes_into_lines,
    merge_overlapping_boxes,
    split_wide_boxes,
)

# Re-export the underlying helpers so callers can import them from here too.
__all__ = [
    "to_binary",
    "mnist_canonical",
    "find_digit_contours",
    "filter_digit_boxes",
    "merge_overlapping_boxes",
    "split_wide_boxes",
    "group_boxes_into_lines",
    "extract_roi",
    "segment_digits",
    "segment_image",
    "run_segmentation",
]


def _ensure_binary(image):
    """Convert an arbitrary image (BGR/gray/binary) into (gray, binary)."""
    if image is None:
        raise ValueError("image must not be None")
    if image.ndim == 3:
        gray, binary = to_binary(image)
        return gray, binary
    # Already single-channel: treat as either gray or binary and re-threshold
    # through to_binary, which accepts grayscale input.
    gray, binary = to_binary(image)
    return gray, binary


def segment_digits(
    image,
    min_area=100,
    max_area=5000,
    min_height=10,
    merge_gap=10,
    max_digits=None,
    min_aspect=0.1,
    max_aspect=1.2,
    max_area_ratio=0.95,
):
    """Return a list of ``(x, y, w, h)`` bounding boxes for the digits.

    ``image`` may be a BGR image, a grayscale image, or a binary image.
    """
    gray, binary = _ensure_binary(image)

    raw_boxes = find_digit_contours(
        binary,
        min_area=min_area,
        min_height=min_height,
        min_aspect=min_aspect,
        max_aspect=max(8.0, max_aspect),
        max_area_ratio=max_area_ratio,
    )
    boxes = merge_overlapping_boxes(raw_boxes, gap=merge_gap)
    boxes = split_wide_boxes(binary, boxes)
    boxes = filter_digit_boxes(
        binary,
        boxes,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )
    if max_digits is not None and max_digits > 0:
        boxes = boxes[:max_digits]
    return boxes, binary, gray


def segment_image(image, **kwargs):
    """Convenience wrapper returning a dict with boxes and intermediate images."""
    boxes, binary, gray = segment_digits(image, **kwargs)
    return {
        "boxes": boxes,
        "binary": binary,
        "gray": gray,
        "count": len(boxes),
    }


def run_segmentation(image, **kwargs):
    """Alias for :func:`segment_image`."""
    return segment_image(image, **kwargs)


def annotate_image(image, boxes):
    """Draw bounding boxes on a copy of ``image`` and return it."""
    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
    return annotated