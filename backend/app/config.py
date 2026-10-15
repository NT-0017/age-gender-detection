from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

GENDER_MODEL_PATH = PROJECT_ROOT / "gender_model.h5"
AGE_MODEL_PATH = PROJECT_ROOT / "age_model.h5"
FACE_PROTO_PATH = PROJECT_ROOT / "opencv_face_detector.pbtxt"
FACE_WEIGHTS_PATH = PROJECT_ROOT / "opencv_face_detector_uint8.pb"

IMG_SIZE = 64
FACE_CONFIDENCE_THRESHOLD = 0.7
FACE_BLOB_SIZE = (300, 300)
FACE_BLOB_MEAN = (104.0, 117.0, 123.0)
GENDER_LABELS = ("Male", "Female")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
