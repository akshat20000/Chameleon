from pathlib import Path
from typing import Tuple, Optional
import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "BodySwap Inference Service"
    environment: str = "development"
    debug: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads: int = 4
    fp16_enabled: bool = torch.cuda.is_available()
    detector_backend: str = "yunet"
    detection_threshold: float = 0.5
    nms_threshold: float = 0.3
    max_faces: int = 10
    tracker_backend: str = "kalman_iou"
    min_iou_threshold: float = 0.2
    max_age: int = 30
    min_hits: int = 3
    landmark_backend: str = "mediapipe"
    dense_landmarks: bool = True
    landmark_model_path: Path = Path("models/face_landmarker.task")
    landmark_min_detection_confidence: float = 0.5
    landmark_min_presence_confidence: float = 0.5
    landmark_min_tracking_confidence: float = 0.5
    segmenter_backend: str = "mediapipe"
    segmentation_threshold: float = 0.5
    segmentation_model_path: Path = Path("models/selfie_multiclass_256x256.tflite")
    identity_backend: str = "onnx"
    identity_model_path: Path = Path("models/w600k_mbf.onnx")
    input_resolution: Tuple[int, int] = (640, 480)
    target_fps: int = 30
    models_dir: Path = Path("models")

    class Config:
        env_prefix = "BODYSWAP_"

_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
