"""
preprocessing.py
----------------
1. แปลงภาพ (BGR หรือ Grayscale) → ภาพ Binary Threshold สำหรับ Segmentation
2. แปลง ROI ของตัวเลขหนึ่งหลัก → รูปแบบเดียวกับ MNIST (canonical 28×28)

รูปแบบ MNIST มาตรฐาน
    - ครอบตัดตัวเลขให้ชิดขอบ (tight crop)
    - ย่อ/ขยายให้ด้านที่ยาวที่สุด = 20 พิกเซล (คงสัดส่วนเดิม)
    - วางกึ่งกลางบนผืนภาพขนาด 28×28 (เหลือขอบรอบละ 4 พิกเซล)
    - ถ้าลายเส้นบางเกินไป (สัดส่วนพิกเซลขาว < 13%) จะขยายเส้น 1 พิกเซล
      แล้วจัดวางใหม่ เพื่อให้ความหนาเส้นใกล้เคียงข้อมูลที่โมเดลถูกฝึกมา
"""

import cv2
import numpy as np

MNIST_SIZE = 28
MNIST_BOX = 20
THIN_FOREGROUND_RATIO = 0.13


def _detect_digit_polarity(blurred: np.ndarray) -> bool:
    """
    เดาว่า "ตัวเลข" เป็นสีสว่างหรือสีเข้ม

    Returns
    -------
    True  – ตัวเลขสว่างบนพื้นมืด (ใช้ THRESH_BINARY_INV เพื่อให้ตัวเลขเป็นสีขาว)
    False – ตัวเลขเข้มบนพื้นสว่าง (ใช้ THRESH_BINARY ตรง ๆ เพื่อให้ตัวเลขเป็นสีขาว)

    หลักการ: ดูค่าเฉลี่ยของ 4 มุมภาพ (ให้ผลที่แม่นยำกว่าขอบทั้งวง)
      • ถ้ามุมมากกว่า threshold → พื้นหลังสว่าง → ตัวเลขเข้ม → ให้กลับสี (False)
      • ถ้ามุมน้อยกว่า/ใกล้เคียง → พื้นหลังมืด → ตัวเลขสว่าง → ไม่ต้องกลับสี (True)
    """
    h, w = blurred.shape[:2]
    margin = max(1, int(min(h, w) * 0.08))

    # ดึงค่าเฉลี่ยของ 4 มุมมากกว่า 1px
    tl = blurred[:margin, :margin].mean()
    tr = blurred[:margin, -margin:].mean()
    bl = blurred[-margin:, :margin].mean()
    br = blurred[-margin:, -margin:].mean()
    corner_mean = (tl + tr + bl + br) / 4.0

    threshold = float(cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

    # ถ้ามุมเฉลี่ยมากกว่า threshold → พื้นหลังสว่าง → ตัวเลขเข้ม → ให้กลับสี (False)
    # ถ้ามุมเฉลี่ยน้อยกว่า/ใกล้เคียง → พื้นหลังมืด → ตัวเลขสว่าง → ไม่ต้องกลับสี (True)
    if abs(corner_mean - threshold) >= 8:
        return corner_mean < threshold

    # fallback: ใช้สัดส่วนพิกเซลสว่างของภาพทั้งใบ
    bright_ratio = float((blurred > threshold).mean())
    return bright_ratio < 0.5


def to_binary(
    image: np.ndarray,
    blur_kernel: int = 5,
    use_clahe: bool = True,
    return_threshold: bool = False,
):
    """
    แปลงภาพเป็น Binary (ขาว-ดำ) สำหรับใช้ตรวจหาตัวเลข

    รองรับทั้งภาพ "ตัวเลขเข้มบนพื้นสว่าง" และ "ตัวเลขสว่างบนพื้นมืด"
    โดยตรวจทิศทางสีอัตโนมัติ (ตัวเลขจะถูกแปลงเป็นสีขาวบนพื้นดำเสมอ)

    Parameters
    ----------
    image       : np.ndarray – ภาพ BGR หรือ Grayscale
    blur_kernel : int – ขนาด Kernel สำหรับ Gaussian Blur (ต้องเป็นเลขคี่)
    use_clahe   : bool – ปรับความต่างแสง-เงา (CLAHE) ก่อน Threshold
                  ช่วยให้ภาพถ่าย/ภาพจากเว็บแคมที่แสงไม่สม่ำเสมอแยกตัวเลขได้ดีขึ้น
    return_threshold : bool – ถ้า True จะคืนค่า Otsu threshold ที่ใช้จริงเพิ่มอีกค่า

    Returns
    -------
    gray   : np.ndarray – ภาพ Grayscale ก่อน Threshold
    binary : np.ndarray – ภาพ Binary หลัง Threshold (ตัวเลขเป็นสีขาว พื้นหลังดำ)
    threshold : float – ค่า Otsu threshold (คืนเฉพาะเมื่อ return_threshold=True)
    """
    # 1. แปลงเป็น Grayscale
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("ไม่สามารถประมวลผลภาพว่างเปล่าได้")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. ปรับแสงให้สม่ำเสมอ (ลดผลจากเงา/แสงไม่เท่ากันในภาพถ่าย)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_for_threshold = clahe.apply(gray)
    else:
        gray_for_threshold = gray

    # 3. ลด Noise ด้วย Gaussian Blur
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray_for_threshold, (k, k), 0)

    # 4. Otsu's Thresholding + ตรวจทิศทางสีของตัวเลข
    threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if not _detect_digit_polarity(blurred):
        # ตัวเลขเข้มบนพื้นสว่าง → กลับสีให้ตัวเลขเป็นสีขาว
        binary = cv2.bitwise_not(binary)

    if return_threshold:
        return gray, binary, float(threshold)
    return gray, binary


def _tight_crop(digit: np.ndarray, threshold: int = 30) -> np.ndarray:
    """ครอบตัดให้ชิดขอบตัวเลขจริง (ตัดพื้นดำส่วนเกินออก)"""
    ys, xs = np.nonzero(digit > threshold)
    if len(xs) == 0:
        return digit
    return digit[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _center_in_canvas(
    digit: np.ndarray,
    out_size: int = MNIST_SIZE,
    box_size: int = MNIST_BOX,
) -> np.ndarray:
    """ย่อ/ขยายแบบคงสัดส่วนให้อยู่กลางผืนภาพ out_size×out_size"""
    h, w = digit.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    scale = box_size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        digit, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    x0 = (out_size - new_w) // 2
    y0 = (out_size - new_h) // 2
    canvas[y0: y0 + new_h, x0: x0 + new_w] = resized
    return canvas


def mnist_canonical(
    roi: np.ndarray,
    out_size: int = MNIST_SIZE,
    box_size: int = MNIST_BOX,
    thin_ratio: float = THIN_FOREGROUND_RATIO,
) -> np.ndarray:
    """
    แปลง ROI ของตัวเลขหนึ่งหลัก → ภาพขนาด 28×28 รูปแบบเดียวกับ MNIST

    Parameters
    ----------
    roi       : np.ndarray – ภาพ ROI (ตัวเลขสีขาวบนพื้นดำ)
    out_size  : int – ขนาดภาพปลายทาง (default 28)
    box_size  : int – ขนาดกล่องตัวเลขภายในภาพปลายทาง (default 20)
    thin_ratio: float – ถ้าสัดส่วนพิกเซลขาวต่ำกว่านี้ จะขยายเส้นก่อนจัดวาง

    Returns
    -------
    np.ndarray – ภาพ Grayscale uint8 ขนาด (out_size, out_size)
    """
    if roi is None or getattr(roi, "size", 0) == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    work = roi
    # รองรับกรณีผู้เรียกส่งภาพโทนสว่าง (ตัวเลขดำบนพื้นขาว) เข้ามา
    if float(work.mean()) > 127:
        work = 255 - work

    digit = _tight_crop(work)
    canvas = _center_in_canvas(digit, out_size=out_size, box_size=box_size)

    # ลายเส้นบางเกินไป → ขยาย 1 พิกเซล แล้วจัดวางใหม่
    if float((canvas > 60).mean()) < thin_ratio:
        thickened = cv2.dilate(digit, np.ones((2, 2), dtype=np.uint8))
        canvas = _center_in_canvas(thickened, out_size=out_size, box_size=box_size)

    return canvas


def normalize_digit(roi: np.ndarray) -> np.ndarray:
    """
    รับภาพ ROI (ตัวเลขหนึ่งหลัก) แล้วแปลงเป็น Tensor รูปแบบที่โมเดลต้องการ

    Returns
    -------
    np.ndarray shape (1, 28, 28, 1) ค่า Float32 [0.0, 1.0]
    """
    canonical = mnist_canonical(roi)
    normalized = canonical.astype(np.float32) / 255.0
    return normalized.reshape(1, MNIST_SIZE, MNIST_SIZE, 1)