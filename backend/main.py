"""FastAPI application for multi-digit handwritten number recognition."""

import base64
import io
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.predictor import Predictor

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "models", "mnist_model.keras")
)

predictor = Predictor(model_path=MODEL_PATH, use_tta=True)
predictor.load()

app = FastAPI(
    title="Multi-Digit Handwritten Number Recognition",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(data: bytes):
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        from PIL import Image as PILImage

        pil = PILImage.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def encode_image(image):
    if image is None:
        return None
    try:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            return None
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def _image_size(*images):
    for image in images:
        if image is not None and getattr(image, "shape", None):
            return {"width": int(image.shape[1]), "height": int(image.shape[0])}
    return None


def serialize_result(
    result,
    elapsed_ms,
    frame_id=0,
    include_annotated=True,
    include_canonical=True,
    include_binary=True,
    include_gray=True,
):
    digits = []
    for idx, dr in enumerate(result.digits):
        digits.append(
            {
                "index": idx,
                "digit": int(dr.digit),
                "confidence": float(dr.confidence),
                "box": {
                    "x": int(dr.box[0]),
                    "y": int(dr.box[1]),
                    "w": int(dr.box[2]),
                    "h": int(dr.box[3]),
                },
                "probabilities": [float(p) for p in dr.probabilities],
                "canonical_image": encode_image(dr.canonical_28)
                if include_canonical
                else None,
            }
        )

    image_size = _image_size(
        result.binary_image, result.gray_image, result.annotated_image
    )

    return {
        "frame_id": int(frame_id),
        "server_time_ms": round(time.time() * 1000.0, 2),
        "full_number": result.full_number or "",
        "digits": digits,
        "lines": [line for line in (result.full_number or "").split("\n")],
        "annotated_image": encode_image(result.annotated_image)
        if include_annotated
        else None,
        "binary_image": encode_image(result.binary_image) if include_binary else None,
        "gray_image": encode_image(result.gray_image) if include_gray else None,
        "threshold": {
            "mode": getattr(result, "threshold_mode", "OTSU"),
            "value": float(result.threshold_value)
            if getattr(result, "threshold_value", None) is not None
            else None,
        },
        "image_size": image_size,
        "digit_count": len(result.digits),
        "average_confidence": float(np.mean([d.confidence for d in result.digits]))
        if result.digits
        else 0.0,
        "processing_time_ms": round(elapsed_ms, 2),
        "inference_ms": round(elapsed_ms, 2),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor.loaded,
        "model_path": MODEL_PATH,
        "error": predictor.error,
        "api_version": app.version,
    }


async def _handle_predict(
    file: UploadFile,
    min_area: int,
    max_area: int,
    merge_gap: int,
    padding: int,
    max_digits: int,
    min_confidence: float,
    use_tta: bool,
    min_aspect: float,
    max_aspect: float,
    frame_id: int,
    include_annotated: bool,
    include_canonical: bool,
    include_binary: bool,
    include_gray: bool,
):
    data = await file.read()
    image = decode_image(data)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    # Accept either a 0-1 fraction or a 0-100 percentage for convenience.
    mc = min_confidence / 100.0 if min_confidence > 1.0 else min_confidence
    if max_aspect <= min_aspect:
        max_aspect = min_aspect + 0.01

    started = time.perf_counter()
    try:
        result = predictor.predict(
            image,
            min_area=min_area,
            max_area=max_area,
            merge_gap=merge_gap,
            padding=padding,
            max_digits=max_digits,
            min_confidence=mc,
            use_tta=use_tta,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return JSONResponse(
        content=serialize_result(
            result,
            elapsed_ms,
            frame_id=frame_id,
            include_annotated=include_annotated,
            include_canonical=include_canonical,
            include_binary=include_binary,
            include_gray=include_gray,
        )
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    min_area: int = Form(100),
    max_area: int = Form(5000),
    merge_gap: int = Form(10),
    padding: int = Form(4),
    max_digits: int = Form(8),
    min_confidence: float = Form(0.8),
    use_tta: bool = Form(True),
    min_aspect: float = Form(0.1),
    max_aspect: float = Form(1.2),
    frame_id: int = Form(0),
    include_annotated: bool = Form(True),
    include_canonical: bool = Form(True),
    include_binary: bool = Form(True),
    include_gray: bool = Form(True),
):
    return await _handle_predict(
        file,
        min_area,
        max_area,
        merge_gap,
        padding,
        max_digits,
        min_confidence,
        use_tta,
        min_aspect,
        max_aspect,
        frame_id,
        include_annotated,
        include_canonical,
        include_binary,
        include_gray,
    )


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    min_area: int = Form(100),
    max_area: int = Form(5000),
    merge_gap: int = Form(10),
    padding: int = Form(4),
    max_digits: int = Form(8),
    min_confidence: float = Form(0.8),
    use_tta: bool = Form(True),
    min_aspect: float = Form(0.1),
    max_aspect: float = Form(1.2),
    frame_id: int = Form(0),
    include_annotated: bool = Form(True),
    include_canonical: bool = Form(True),
    include_binary: bool = Form(True),
    include_gray: bool = Form(True),
):
    return await _handle_predict(
        file,
        min_area,
        max_area,
        merge_gap,
        padding,
        max_digits,
        min_confidence,
        use_tta,
        min_aspect,
        max_aspect,
        frame_id,
        include_annotated,
        include_canonical,
        include_binary,
        include_gray,
    )


# Serve the frontend (must be registered after API routes).
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:

    @app.get("/")
    def root():
        return {"message": "Frontend not found", "health": "/health"}
