"""
recognizer.py
-------------
โหลดโมเดล CNN และทำนายตัวเลขจาก ROI ที่ผ่านการ Preprocess แล้ว

รองรับ Test-Time Augmentation (TTA) แบบเบา ๆ:
ทำนายจากภาพ 28×28 สองมุมมอง (ภาพเดิม + ภาพขยายเส้น 1 พิกเซล)
แล้วเฉลี่ยความน่าจะเป็น ช่วยให้ตัวเลขที่เขียนสไตล์แปลก ๆ (เช่น 4 ที่ด้ามยาว)
มีความมั่นใจและความถูกต้องดีขึ้น
"""

import os

import cv2
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402


class DigitRecognizer:
    """
    โหลดโมเดล Keras และทำนายตัวเลขจากภาพขนาด 28×28 พิกเซล
    """

    def __init__(self, model_path: str, use_tta: bool = True):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model: tf.keras.Model = tf.keras.models.load_model(model_path)
        self.use_tta = use_tta

    # ------------------------------------------------------------------
    #  การทำนาย
    # ------------------------------------------------------------------
    def _to_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """แปลงภาพ 28×28 (uint8) หลายภาพ → batch tensor (N, 28, 28, 1)"""
        arrays = [img.astype(np.float32) / 255.0 for img in images]
        stacked = np.stack(arrays)
        return stacked.reshape(-1, 28, 28, 1)

    def _tta_batch(self, canonical: np.ndarray) -> np.ndarray:
        """สร้างชุดภาพสำหรับ TTA: ภาพเดิม + ภาพที่ขยายเส้นขึ้น 1 พิกเซล"""
        thickened = cv2.dilate(canonical, np.ones((3, 3), np.uint8))
        return self._to_batch([canonical, thickened])

    def predict_canonical(
        self,
        canonical: np.ndarray,
        use_tta: bool | None = None,
    ) -> tuple[int, np.ndarray]:
        """
        ทำนายจากภาพ 28×28 แบบ MNIST canonical (uint8)

        Parameters
        ----------
        canonical : np.ndarray – ภาพ 28×28 (ตัวเลขขาวบนพื้นดำ)
        use_tta   : bool | None – เปิด TTA หรือไม่ (None = ใช้ค่าที่ตั้งไว้ตอนโหลด)

        Returns
        -------
        digit         : int – ตัวเลขที่ทำนายได้ (0–9)
        probabilities : np.ndarray shape (10,) – ความน่าจะเป็นของแต่ละคลาส
        """
        if use_tta is None:
            use_tta = self.use_tta

        if use_tta:
            batch = self._tta_batch(canonical)
            probs = self.model.predict(batch, verbose=0).mean(axis=0)
        else:
            probs = self.model.predict(
                self._to_batch([canonical]), verbose=0
            )[0]

        digit = int(np.argmax(probs))
        return digit, probs.astype(np.float32)

    def predict(self, tensor: np.ndarray) -> tuple[int, np.ndarray]:
        """
        ทำนายตัวเลขจาก Tensor รูปแบบ (1, 28, 28, 1)

        Returns
        -------
        digit         : int – ตัวเลขที่ทำนายได้ (0–9)
        probabilities : np.ndarray shape (10,) – ความน่าจะเป็นของแต่ละคลาส
        """
        probs = self.model.predict(tensor, verbose=0)[0]
        digit = int(np.argmax(probs))
        return digit, probs.astype(np.float32)
