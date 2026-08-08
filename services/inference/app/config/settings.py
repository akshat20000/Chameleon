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
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    match_threshold: float = 0.8
    landmark_backend: str = "mediapipe"
    dense_landmarks: bool = True
    segmenter_backend: str = "mediapipe"
    segmentation_threshold: float = 0.5
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
