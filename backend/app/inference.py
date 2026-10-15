from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import tensorflow as tf

from .config import (
    AGE_MODEL_PATH,
    FACE_BLOB_MEAN,
    FACE_BLOB_SIZE,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_PROTO_PATH,
    FACE_WEIGHTS_PATH,
    GENDER_LABELS,
    GENDER_MODEL_PATH,
    IMG_SIZE,
)
from .schemas import BoundingBox, FacePrediction


@dataclass
class ModelRuntime:
    gender_model: object | None = None
    age_model: object | None = None
    face_net: object | None = None
    gender_ok: bool = False
    age_ok: bool = False
    detector_ok: bool = False
    details: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.gender_ok and self.age_ok and self.detector_ok

    def load(self) -> None:
        try:
            if not GENDER_MODEL_PATH.is_file():
                raise FileNotFoundError(f"Missing file: {GENDER_MODEL_PATH}")
            self.gender_model = tf.keras.models.load_model(
                GENDER_MODEL_PATH, compile=False
            )
            self.gender_ok = True
            self.details["gender_model"] = "loaded"
        except Exception as exc:
            self.gender_ok = False
            self.gender_model = None
            self.details["gender_model"] = str(exc)

        try:
            if not AGE_MODEL_PATH.is_file():
                raise FileNotFoundError(f"Missing file: {AGE_MODEL_PATH}")
            self.age_model = tf.keras.models.load_model(
                AGE_MODEL_PATH, compile=False
            )
            self.age_ok = True
            self.details["age_model"] = "loaded"
        except Exception as exc:
            self.age_ok = False
            self.age_model = None
            self.details["age_model"] = str(exc)

        try:
            if not FACE_WEIGHTS_PATH.is_file():
                raise FileNotFoundError(f"Missing file: {FACE_WEIGHTS_PATH}")
            if not FACE_PROTO_PATH.is_file():
                raise FileNotFoundError(f"Missing file: {FACE_PROTO_PATH}")
            face_net = cv2.dnn.readNet(str(FACE_WEIGHTS_PATH), str(FACE_PROTO_PATH))
            if face_net.empty():
                raise RuntimeError("cv2.dnn.readNet returned an empty network")
            self.face_net = face_net
            self.detector_ok = True
            self.details["face_detector"] = "loaded"
        except Exception as exc:
            self.detector_ok = False
            self.face_net = None
            self.details["face_detector"] = str(exc)

    def predict_image(self, frame: np.ndarray) -> list[FacePrediction]:
        if not self.ready:
            raise RuntimeError("Models are not loaded")

        h, w = frame.shape[:2]
        with self._lock:
            blob = cv2.dnn.blobFromImage(
                frame,
                1.0,
                FACE_BLOB_SIZE,
                FACE_BLOB_MEAN,
                swapRB=False,
            )
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            faces: list[FacePrediction] = []
            for i in range(detections.shape[2]):
                face_conf = float(detections[0, 0, i, 2])
                if face_conf <= FACE_CONFIDENCE_THRESHOLD:
                    continue

                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                crop = frame[y1_c:y2_c, x1_c:x2_c]
                if crop.size == 0:
                    continue

                gender, gender_conf, age = self._predict_gender_age(crop)
                faces.append(
                    FacePrediction(
                        box=BoundingBox(x1=x1_c, y1=y1_c, x2=x2_c, y2=y2_c),
                        age=age,
                        gender=gender,
                        gender_confidence=round(gender_conf, 4),
                        face_confidence=round(face_conf, 4),
                    )
                )
            return faces

    def _predict_gender_age(self, face_img: np.ndarray) -> tuple[str, float, int]:
        img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        gender_prob = float(self.gender_model.predict(img, verbose=0)[0][0])
        gender = GENDER_LABELS[int(gender_prob > 0.5)]
        gender_conf = gender_prob if gender_prob > 0.5 else 1.0 - gender_prob
        age = int(self.age_model.predict(img, verbose=0)[0][0] * 100)
        age = max(1, min(100, age))
        return gender, gender_conf, age


runtime = ModelRuntime()
