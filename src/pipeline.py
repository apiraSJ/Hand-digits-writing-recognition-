"""
pipeline.py
-----------
รวม Preprocessing → Segmentation → Recognition เข้าด้วยกัน
เป็น API กลางที่ app.py เรียกใช้
"""

from dataclasses import dataclass, field
from typing import Protocol

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


class DigitPredictor(Protocol):
    def predict_canonical(
        self,
        canonical: np.ndarray,
        use_tta: bool | None = None,
    ) -> tuple[int, np.ndarray]: ...


@dataclass
class DigitResult:
    """ผลลัพธ์ของตัวเลขหนึ่งหลัก"""

    digit: int
    probabilities: np.ndarray          # shape (10,)
    box: tuple[int, int, int, int]     # (x, y, w, h) บนภาพต้นฉบับ
    confidence: float                  # ความน่าจะเป็นของคลาสที่ทำนาย (0–1)
    canonical_28: np.ndarray           # ภาพ 28×28 "สิ่งที่โมเดลเห็นจริง"


@dataclass
class PipelineResult:
    """ผลลัพธ์ทั้งหมดของ Pipeline"""

    full_number: str                   # เลขที่รวมแล้ว เช่น "542" (หลายบรรทัดคั่นด้วย \n)
    digits: list[DigitResult] = field(default_factory=list)
    lines: list[list[DigitResult]] = field(default_factory=list)  # แยกตามบรรทัด
    binary_image: np.ndarray | None = None       # ภาพ Binary Threshold
    annotated_image: np.ndarray | None = None    # ภาพต้นฉบับ + กรอบ + ป้ายตัวเลข
    gray_image: np.ndarray | None = None         # ภาพ Grayscale ก่อน Threshold
    threshold_value: float | None = None         # ค่า Otsu threshold ที่ใช้จริง
    threshold_mode: str = "OTSU"                 # โหมด threshold ที่ใช้


def _draw_annotation(
    image: np.ndarray,
    digits: list[DigitResult],
) -> np.ndarray:
    """วาดกรอบสี่เหลี่ยมและป้ายตัวเลขบนภาพต้นฉบับ"""
    annotated = image.copy()
    h_img = annotated.shape[0]

    # ความหนา/ขนาดตัวอักษรปรับตามขนาดภาพ (ภาพเล็กจะได้ไม่ทับกัน)
    thickness = max(1, int(round(h_img / 300)))
    base_scale = max(0.45, min(1.2, h_img / 700.0))

    for dr in digits:
        x, y, w, h = dr.box
        cv2.rectangle(
            annotated, (x, y), (x + w, y + h), (0, 200, 0), thickness
        )

        # ตัวเลขสีเขียวเข้มพาสเทล (แดงต่ำ) เพื่อให้อ่านง่ายบนภาพส่วนใหญ่
        label = str(dr.digit)
        font_scale = max(base_scale, h / 90.0)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness + 1
        )

        top = max(0, y - text_h - baseline - 4)
        cv2.rectangle(
            annotated,
            (x, top),
            (x + text_w + 6, top + text_h + baseline + 4),
            (0, 200, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 3, top + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 1,
        )

    return annotated


def run_pipeline(
    image: np.ndarray,
    recognizer: DigitPredictor,
    min_area: int = 100,
    max_area: int = 5000,
    merge_gap: int = 10,
    padding: int = 4,
    max_digits: int = 8,
    min_confidence: float = 0.8,
    use_tta: bool | None = None,
    min_aspect: float = 0.1,
    max_aspect: float = 1.2,
) -> PipelineResult:
    """
    รันกระบวนการทั้งหมดบนภาพที่รับเข้ามา

    Parameters
    ----------
    image          : np.ndarray – ภาพ BGR (หรือ Grayscale) จาก OpenCV
    recognizer     : DigitPredictor – โมเดลที่โหลดแล้ว
    min_area       : int – พื้นที่ขั้นต่ำของ Contour (ตัดสัญญาณรบกวน)
    max_area       : int – พื้นที่พิกเซลตัวเลขสูงสุดหลังแยกหลักแล้ว
    merge_gap      : int – ระยะห่างแนวตั้งสูงสุดของชิ้นส่วนตัวเลขเดียวกัน
    padding        : int – Padding รอบ ROI ก่อนเข้ากระบวนการ MNIST canonical
    max_digits     : int – จำนวนหลักสูงสุดที่ยอมรับ
    min_confidence : float – ทิ้งผลทำนายที่มีความมั่นใจต่ำกว่านี้ (ค่าเริ่มต้น 0.8)
    use_tta        : bool | None – เปิด Test-Time Augmentation (None = ตามค่าของ recognizer)
    min_aspect     : float – สัดส่วน กว้าง/สูง ต่ำสุดของ contour ที่ถือว่าเป็นตัวเลข
    max_aspect     : float – สัดส่วน กว้าง/สูง สูงสุดของ contour ที่ถือว่าเป็นตัวเลข
                      คัดกรองหลังแยกกล่องกว้าง เพื่อให้แยกเลขที่เขียนติดกันได้ก่อน

    Returns
    -------
    PipelineResult
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("ไม่สามารถประมวลผลภาพว่างเปล่าได้")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    # --- Preprocessing ---
    gray, binary, threshold_value = to_binary(bgr, return_threshold=True)

    # --- Segmentation ---
    raw_boxes = find_digit_contours(
        binary,
        min_area=min_area,
        min_aspect=min_aspect,
        max_aspect=max(8.0, max_aspect),
    )
    boxes = merge_overlapping_boxes(raw_boxes, gap=merge_gap)
    # ตัวเลขที่เขียนติดกันจะถูกแยกด้วย vertical projection profile
    boxes = split_wide_boxes(binary, boxes)
    boxes = filter_digit_boxes(
        binary,
        boxes,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )[:max_digits]

    # --- Recognition ---
    digit_by_box: dict[tuple[int, int, int, int], DigitResult] = {}
    for box in boxes:
        roi = extract_roi(binary, box, padding=padding, neighbors=boxes)
        if roi.size == 0:
            continue

        # ภาพ 28×28 แบบ MNIST ที่จะถูกส่งเข้าโมเดล (ใช้แสดงใน UI ด้วย)
        canonical = mnist_canonical(roi)
        digit, probs = recognizer.predict_canonical(canonical, use_tta=use_tta)

        confidence = float(probs[digit])
        if confidence < min_confidence:
            continue

        digit_by_box[box] = DigitResult(
            digit=digit,
            probabilities=probs,
            box=box,
            confidence=confidence,
            canonical_28=canonical,
        )

    # --- จัดกลุ่มเป็นบรรทัด (บน → ล่าง, ซ้าย → ขวา) ---
    lines: list[list[DigitResult]] = []
    for line_boxes in group_boxes_into_lines(boxes):
        line_results = [digit_by_box[b] for b in line_boxes if b in digit_by_box]
        if line_results:
            lines.append(line_results)

    digits = [digit for line in lines for digit in line]
    annotated = _draw_annotation(bgr, digits)
    full_number = "\n".join(
        "".join(str(d.digit) for d in line) for line in lines
    )

    return PipelineResult(
        full_number=full_number,
        digits=digits,
        lines=lines,
        binary_image=binary,
        annotated_image=annotated,
        gray_image=gray,
        threshold_value=threshold_value,
        threshold_mode="OTSU",
    )
