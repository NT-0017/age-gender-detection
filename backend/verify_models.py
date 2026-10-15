"""Load the existing age/gender CNNs and OpenCV DNN face detector.

Does not retrain, convert, or modify any model files.
Run from the project after installing backend/requirements.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENDER_MODEL_PATH = PROJECT_ROOT / "gender_model.h5"
AGE_MODEL_PATH = PROJECT_ROOT / "age_model.h5"
FACE_PROTO_PATH = PROJECT_ROOT / "opencv_face_detector.pbtxt"
FACE_WEIGHTS_PATH = PROJECT_ROOT / "opencv_face_detector_uint8.pb"


def _print_result(name: str, ok: bool, detail: str) -> None:
    status = "SUCCESS" if ok else "FAILURE"
    print(f"[{status}] {name}: {detail}")


def main() -> int:
    print(f"Python {sys.version}")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    failures = 0

    try:
        import tensorflow as tf

        print(f"TensorFlow {tf.__version__}")
        print(f"Keras {tf.keras.__version__}")
    except Exception as exc:
        _print_result("TensorFlow/Keras import", False, str(exc))
        return 1

    try:
        import cv2

        print(f"OpenCV {cv2.__version__}")
    except Exception as exc:
        _print_result("OpenCV import", False, str(exc))
        return 1

    print()

    try:
        if not GENDER_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Missing file: {GENDER_MODEL_PATH}")
        gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH, compile=False)
        shape = getattr(gender_model, "input_shape", None)
        _print_result(
            "gender_model.h5",
            True,
            f"loaded with compile=False (input_shape={shape})",
        )
    except Exception as exc:
        failures += 1
        _print_result("gender_model.h5", False, str(exc))

    try:
        if not AGE_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Missing file: {AGE_MODEL_PATH}")
        age_model = tf.keras.models.load_model(AGE_MODEL_PATH, compile=False)
        shape = getattr(age_model, "input_shape", None)
        _print_result(
            "age_model.h5",
            True,
            f"loaded with compile=False (input_shape={shape})",
        )
    except Exception as exc:
        failures += 1
        _print_result("age_model.h5", False, str(exc))

    try:
        if not FACE_WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"Missing file: {FACE_WEIGHTS_PATH}")
        if not FACE_PROTO_PATH.is_file():
            raise FileNotFoundError(f"Missing file: {FACE_PROTO_PATH}")
        face_net = cv2.dnn.readNet(str(FACE_WEIGHTS_PATH), str(FACE_PROTO_PATH))
        if face_net.empty():
            raise RuntimeError("cv2.dnn.readNet returned an empty network")
        _print_result(
            "OpenCV face detector",
            True,
            "loaded opencv_face_detector_uint8.pb + opencv_face_detector.pbtxt",
        )
    except Exception as exc:
        failures += 1
        _print_result("OpenCV face detector", False, str(exc))

    print()
    if failures:
        print(f"Verification finished with {failures} failure(s).")
        return 1

    print("Verification finished: all models loaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
