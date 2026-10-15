from __future__ import annotations

from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_IMAGE_SUFFIXES, ALLOWED_IMAGE_TYPES, MAX_UPLOAD_BYTES
from .inference import runtime
from .schemas import DetectResponse, HealthResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime.load()
    yield


app = FastAPI(
    title="Age and Gender Detection API",
    description="Detect faces and predict age and gender using the existing trained models.",
    lifespan=lifespan,
)

# Allow the frontend (served from any localhost port or file://) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _decode_upload(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a valid JPEG, PNG, BMP, or WebP file.")
    return frame


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = runtime.ready
    return HealthResponse(
        status="ok" if ready else "error",
        gender_model=runtime.gender_ok,
        age_model=runtime.age_ok,
        face_detector=runtime.detector_ok,
        details=runtime.details,
    )


@app.post("/api/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(..., description="Image containing one or more faces")) -> DetectResponse:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="Models are not loaded. Check GET /health.")

    content_type = (file.content_type or "").lower()
    suffix = ""
    if file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    type_ok = content_type in ALLOWED_IMAGE_TYPES if content_type else False
    suffix_ok = suffix in ALLOWED_IMAGE_SUFFIXES if suffix else False
    if content_type and not type_ok and not suffix_ok:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a JPEG, PNG, BMP, or WebP image.",
        )
    if suffix and not suffix_ok and not type_ok:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Use .jpg, .jpeg, .png, .bmp, or .webp.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    frame = _decode_upload(data)
    faces = runtime.predict_image(frame)
    height, width = frame.shape[:2]
    return DetectResponse(image_width=width, image_height=height, faces=faces)
