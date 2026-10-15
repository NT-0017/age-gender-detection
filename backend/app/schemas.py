from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class FacePrediction(BaseModel):
    box: BoundingBox
    age: int
    gender: str
    gender_confidence: float = Field(description="Confidence of the predicted gender, 0–1")
    face_confidence: float = Field(description="OpenCV DNN face detection score, 0–1")


class DetectResponse(BaseModel):
    image_width: int
    image_height: int
    faces: list[FacePrediction]


class HealthResponse(BaseModel):
    status: str
    gender_model: bool
    age_model: bool
    face_detector: bool
    details: dict[str, str] = Field(default_factory=dict)
