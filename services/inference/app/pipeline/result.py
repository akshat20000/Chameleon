from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass
class FaceDetection:
    bbox: BoundingBox
    confidence: float
    landmarks_5pt: Optional[np.ndarray] = None
    class_id: int = 0
@dataclass
class TrackedFace:
    track_id: int
    bbox: BoundingBox
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    is_observed: bool = True

@dataclass
class LandmarkResult:
    points_2d: np.ndarray
    points_3d: Optional[np.ndarray] = None
    confidence: float = 1.0
    landmarks_type: str = "478_pt"

@dataclass
class PoseResult:
    pitch: float
    yaw: float
    roll: float
    transformation_matrix: np.ndarray
    blendshapes: Dict[str, float]

@dataclass
class SegmentationResult:
    face_mask: np.ndarray
    hair_mask: Optional[np.ndarray] = None
    skin_mask: Optional[np.ndarray] = None
    class_mask: Optional[np.ndarray] = None

@dataclass
class PipelineResult:
    frame_id: int
    timestamp: float
    output_image: np.ndarray
    detections: List[FaceDetection] = field(default_factory=list)
    tracks: List[TrackedFace] = field(default_factory=list)
    landmarks: Dict[int, LandmarkResult] = field(default_factory=dict)
    pose: Dict[int, PoseResult] = field(default_factory=dict)
    segmentation: Optional[SegmentationResult] = None
    timings: Dict[str, float] = field(default_factory=dict)
