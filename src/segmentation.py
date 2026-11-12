"""
segmentation.py
---------------
ตัดแบ่งตัวเลขแต่ละหลักออกจากภาพ Binary
คืน Bounding Boxes เรียงจากซ้ายไปขวา

หลักการสำคัญ: การรวม Box (merge) ต้อง "ไม่รวมตัวเลขคนละหลักเข้าด้วยกัน"
จึงยอมให้รวมเฉพาะ Box ที่ซ้อนทับกันจริงในแนวแกน x (ชิ้นส่วนของหลักเดียวกัน)
ไม่ได้อิงระยะห่างแนวนอนเพียงอย่างเดียวเหมือนเวอร์ชันก่อน
"""

import cv2
import numpy as np


def _horizontal_overlap(a: tuple, b: tuple) -> float:
    """สัดส่วนการซ้อนทับในแนวแกน x เทียบกับความกว้างที่น้อยกว่า (0.0 – 1.0)"""
    a_left, a_right = a[0], a[0] + a[2]
    b_left, b_right = b[0], b[0] + b[2]
    inter = min(a_right, b_right) - max(a_left, b_left)
    return max(0, inter) / max(1, min(a[2], b[2]))


def _vertical_overlap(a: tuple, b: tuple) -> float:
    """สัดส่วนการซ้อนทับในแนวแกน y เทียบกับความสูงที่น้อยกว่า (0.0 – 1.0)"""
    a_top, a_bottom = a[1], a[1] + a[3]
    b_top, b_bottom = b[1], b[1] + b[3]
    inter = min(a_bottom, b_bottom) - max(a_top, b_top)
    return max(0, inter) / max(1, min(a[3], b[3]))


def _vertical_gap(a: tuple, b: tuple) -> int:
    """ระยะห่างแนวตั้งระหว่าง Box (ติดลบ = ซ้อนทับกัน)"""
    a_top, a_bottom = a[1], a[1] + a[3]
    b_top, b_bottom = b[1], b[1] + b[3]
    if a_bottom < b_top:
        return b_top - a_bottom
    if b_bottom < a_top:
        return a_top - b_bottom
    return 0


def find_digit_contours(
    binary: np.ndarray,
    min_area: int = 100,
    min_height: int = 10,
    min_area_ratio: float = 0.0,
    min_aspect: float = 0.15,
    max_aspect: float = 8.0,
    max_area_ratio: float = 0.95,
    max_digits: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    หา Bounding Box ของตัวเลขแต่ละหลักในภาพ Binary

    Parameters
    ----------
    binary        : np.ndarray – ภาพ Binary (ตัวเลขขาว พื้นดำ)
    min_area      : int – พื้นที่ขั้นต่ำของ Contour (กรองสัญญาณรบกวนออก)
    min_height    : int – ความสูงขั้นต่ำของ Bounding Box
    min_area_ratio: float – สัดส่วนพื้นที่ขั้นต่ำเทียบกับภาพทั้งใบ (0–1)
    min_aspect    : float – สัดส่วน กว้าง/สูง ต่ำสุด (กันเส้นบางผิดรูป)
    max_aspect    : float – สัดส่วน กว้าง/สูง สูงสุด
    max_area_ratio: float – สัดส่วนพื้นที่สูงสุดเทียบกับภาพทั้งใบ
                      (กันกรอบกระดาษ/พื้นหลังหลุดเข้ามาเป็น "ตัวเลข")
    max_digits    : int | None – จำนวนหลักสูงสุดที่ยอมรับ (None = ไม่จำกัด)

    Returns
    -------
    list of (x, y, w, h) เรียงจากซ้ายไปขวา
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = binary.shape[:2]
    image_area = float(h_img * w_img) if h_img and w_img else 1.0

    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        if h < min_height:
            continue

        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        if area / image_area < min_area_ratio:
            continue
        if area / image_area > max_area_ratio:
            continue

        boxes.append((int(x), int(y), int(w), int(h)))

    # เรียงจากซ้ายไปขวาตามค่า x
    boxes.sort(key=lambda b: b[0])

    if max_digits is not None and max_digits > 0:
        boxes = boxes[:max_digits]

    return boxes


def filter_digit_boxes(
    binary: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    min_area: int = 100,
    max_area: int = 5000,
    min_aspect: float = 0.1,
    max_aspect: float = 1.2,
) -> list[tuple[int, int, int, int]]:
    """Keep boxes whose foreground area and width/height ratio fit a digit."""
    h_img, w_img = binary.shape[:2]
    filtered: list[tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        contours, _ = cv2.findContours(
            binary[y1:y2, x1:x2],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        area = sum(cv2.contourArea(contour) for contour in contours)
        aspect = (x2 - x1) / float(y2 - y1)
        if (
            min_area <= area <= max_area
            and min_aspect <= aspect <= max_aspect
        ):
            filtered.append((x1, y1, x2 - x1, y2 - y1))

    return filtered


def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    gap: int = 10,
) -> list[tuple[int, int, int, int]]:
    """
    รวม Bounding Box ที่เป็น "ชิ้นส่วนของตัวเลขหลักเดียวกัน"

    เงื่อนไขการรวม (ต้องเข้าเงื่อนไขใดเงื่อนไขหนึ่ง)
      1. Box ซ้อนทับกันในแนวแกน x จริง (horizontal overlap > 0)
         และซ้อนทับกันในแนวแกน y อย่างน้อย 30%
         → เช่น เส้นของเลข "5" ที่ขาดออกเป็นสองส่วนในคอลัมน์เดียวกัน
      2. Box วางซ้อนกันในแนวตั้ง (horizontal overlap ≥ 50%)
         และระยะห่างแนวตั้งไม่เกิน `gap`
         → เช่น จุด/ขีดที่แยกจากตัวหลัก

    ตัวเลขคนละหลักที่วางเรียงกัน (horizontal overlap = 0) จะ "ไม่ถูก merge"
    แม้จะอยู่ห่างกันเพียง 2 พิกเซล ซึ่งเป็นปัญหาของเวอร์ชันก่อน

    Parameters
    ----------
    boxes : list of (x, y, w, h) เรียงจากซ้ายไปขวา
    gap   : int – ระยะห่างแนวตั้งสูงสุดที่ถือว่าเป็นชิ้นส่วนเดียวกัน

    Returns
    -------
    list of (x, y, w, h) ที่รวมแล้ว เรียงจากซ้ายไปขวา
    """
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: b[0])
    merged: list[list[int]] = [list(ordered[0])]

    for box in ordered[1:]:
        prev = merged[-1]
        h_overlap = _horizontal_overlap(tuple(prev), box)
        v_overlap = _vertical_overlap(tuple(prev), box)
        v_gap = _vertical_gap(tuple(prev), box)

        same_side_by_side_piece = h_overlap > 0 and v_overlap >= 0.30
        same_stacked_piece = h_overlap >= 0.50 and v_gap <= max(0, gap)

        if same_side_by_side_piece or same_stacked_piece:
            prev_right = prev[0] + prev[2]
            prev_bottom = prev[1] + prev[3]
            new_x = min(prev[0], box[0])
            new_y = min(prev[1], box[1])
            new_right = max(prev_right, box[0] + box[2])
            new_bottom = max(prev_bottom, box[1] + box[3])
            merged[-1] = [new_x, new_y, new_right - new_x, new_bottom - new_y]
        else:
            merged.append(list(box))

    return [tuple(int(v) for v in b) for b in merged]


def split_wide_boxes(
    binary: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    max_aspect: float = 1.1,
    min_piece_width: int = 8,
    min_ink_ratio: float = 0.10,
    valley_ratio: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """
    แยก Box ที่กว้างผิดปกติ (ตัวเลขหลายหลักเขียนติดกันเป็นเนื้อเดียว)

    ใช้ Vertical Projection Profile: หาคอลัมน์ที่มีหมึกน้อยที่สุดในช่วงกลาง
    แล้วตัดออกเป็นสองส่วน ทำซ้ำจนกว่าแต่ละ Box จะมีสัดส่วนปกติ

    Parameters
    ----------
    binary         : np.ndarray – ภาพ Binary
    boxes          : list of (x, y, w, h)
    max_aspect     : float – ถ้า w/h เกินค่านี้จะถือว่าอาจมีหลายหลัก
    min_piece_width: int – ความกว้างขั้นต่ำของชิ้นส่วน (กันการตัดชิ้นจิ๋ว)
    min_ink_ratio  : float – แต่ละฝั่งต้องมีหมึกอย่างน้อยสัดส่วนนี้ของทั้งหมด
    valley_ratio   : float – จุดตัดต้องมีหมึกน้อยกว่าคอลัมน์ที่หนึกว่า 'valley_ratio' เท่า
                     (กันการตัดตัวเลขที่กว้างจริง เช่น 7 หัวกว้าง)

    Returns
    -------
    list of (x, y, w, h) เรียงจากซ้ายไปขวา
    """
    if not boxes:
        return []

    result: list[tuple[int, int, int, int]] = []
    queue = list(boxes)
    guard = 0
    max_iterations = 200

    while queue and guard < max_iterations:
        guard += 1
        x, y, w, h = queue.pop(0)

        if h <= 0 or w <= 0:
            continue
        if w / float(h) <= max_aspect or w < 2 * min_piece_width:
            result.append((x, y, w, h))
            continue

        roi = binary[y: y + h, x: x + w]
        ink = roi > 0
        total_ink = int(ink.sum())
        if total_ink == 0:
            result.append((x, y, w, h))
            continue

        column_ink = ink.sum(axis=0)
        mid_start = max(1, int(w * 0.25))
        mid_end = min(w - 1, int(w * 0.75))
        if mid_end <= mid_start:
            result.append((x, y, w, h))
            continue

        middle = column_ink[mid_start:mid_end]
        strongest_column = float(column_ink.max())
        split_offset = int(np.argmin(middle))
        split = mid_start + split_offset

        # ไม่มี "หุบเขา" ของหมึก → ไม่ใช่ตัวเลขติดกัน ให้คงเดิม
        if strongest_column > 0 and middle[split_offset] > valley_ratio * strongest_column:
            result.append((x, y, w, h))
            continue

        left_ink = int(ink[:, :split].sum())
        right_ink = int(ink[:, split:].sum())
        if (
            left_ink < total_ink * min_ink_ratio
            or right_ink < total_ink * min_ink_ratio
        ):
            result.append((x, y, w, h))
            continue

        # ตัดแล้วคำนวณ Bounding Box ใหม่ให้ชิดขอบของแต่ละฝั่ง
        for piece, x_offset in ((ink[:, :split], x), (ink[:, split:], x + split)):
            ys, xs = np.nonzero(piece)
            if len(xs) == 0:
                continue
            queue.append(
                (
                    int(x_offset + xs.min()),
                    int(y + ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                )
            )

    result.sort(key=lambda b: b[0])
    return result


def group_boxes_into_lines(
    boxes: list[tuple[int, int, int, int]],
    y_overlap_threshold: float = 0.5,
) -> list[list[tuple[int, int, int, int]]]:
    """
    จัดกลุ่ม Bounding Box เป็น "บรรทัด" ตามการซ้อนทับในแนวตั้ง

    Parameters
    ----------
    boxes               : list of (x, y, w, h)
    y_overlap_threshold : float – สัดส่วนการซ้อนทับแนวตั้งขั้นต่ำ
                          ที่ถือว่าอยู่บรรทัดเดียวกัน (0–1)

    Returns
    -------
    list of lines (บน → ล่าง) โดยแต่ละบรรทัดเรียงซ้าย → ขวา
    """
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: b[1])
    lines: list[dict] = []

    for box in ordered:
        top = box[1]
        bottom = box[1] + box[3]
        placed = False

        for line in lines:
            overlap = min(line["bottom"], bottom) - max(line["top"], top)
            if overlap <= 0:
                continue

            shorter = max(1, min(line["bottom"] - line["top"], box[3]))
            if overlap / shorter >= y_overlap_threshold:
                line["boxes"].append(box)
                line["top"] = min(line["top"], top)
                line["bottom"] = max(line["bottom"], bottom)
                placed = True
                break

        if not placed:
            lines.append({"top": top, "bottom": bottom, "boxes": [box]})

    lines.sort(key=lambda line: line["top"])
    return [
        sorted(line["boxes"], key=lambda b: b[0])
        for line in lines
    ]


def extract_roi(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int = 4,
    neighbors: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    ตัด ROI ของตัวเลขหนึ่งหลักออกจากภาพ Binary พร้อม Padding

    Parameters
    ----------
    binary    : np.ndarray – ภาพ Binary
    box       : (x, y, w, h) – Bounding Box
    padding   : int – จำนวนพิกเซลที่เพิ่มรอบ ROI
    neighbors : list of (x, y, w, h) | None – Box อื่น ๆ ในภาพ
                ใช้จำกัด Padding แนวนอนไม่ให้ล้ำเข้าไปในตัวเลขหลักข้างเคียง
                (สำคัญมากเมื่อเขียนตัวเลขชิดกัน เพราะเศษเส้นของเพื่อนบ้าน
                จะทำให้โมเดลทำนายผิด)

    Returns
    -------
    np.ndarray – ภาพ ROI ของตัวเลข (อาจเป็นอาร์เรย์ว่างถ้าตัดไม่ได้)
    """
    h_img, w_img = binary.shape[:2]
    x, y, w, h = box

    left_pad = padding
    right_pad = padding

    if neighbors:
        for other in neighbors:
            if other == box:
                continue
            ox, oy, ow, oh = other

            # ต้องมีส่วนซ้อนทับในแนวตั้งจึงจะถือว่าเป็นเพื่อนบ้านข้าง ๆ
            vertical_overlap = min(y + h, oy + oh) - max(y, oy)
            if vertical_overlap <= 0:
                continue

            if ox + ow <= x:            # เพื่อนบ้านอยู่ทางซ้าย
                left_pad = min(left_pad, max(0, x - (ox + ow)))
            elif ox >= x + w:           # เพื่อนบ้านอยู่ทางขวา
                right_pad = min(right_pad, max(0, ox - (x + w)))

    x1 = max(0, x - left_pad)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + right_pad)
    y2 = min(h_img, y + h + padding)
    return binary[y1:y2, x1:x2]
