from abc import ABC, abstractmethod
from typing import List
import numpy as np
import cv2
from app.pipeline.result import BoundingBox, FaceDetection
from app.config.settings import get_settings

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        pass
class YuNetDetector(BaseDetector):
    def __init__(self, model_path: str = "models/face_detection_yunet_2023mar.onnx", score_threshold: float = 0.5, nms_threshold: float = 0.3):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.detector = None
        self.model_path = model_path
        self._init_detector()

    def _init_detector(self):
        import os
        if os.path.exists(self.model_path):
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold
            )
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        if self.detector is not None:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
            results = []
            if faces is not None:
                for face in faces:
                    x, y, bw, bh = face[0:4]
                    score = float(face[14])
                    lm = face[4:14].reshape((5, 2))
                    bbox = BoundingBox(x_min=float(x), y_min=float(y), x_max=float(x+bw), y_max=float(y+bh))
                    results.append(FaceDetection(bbox=bbox, confidence=score, landmarks_5pt=lm))
            return results
        return []

class MediaPipeDetector(BaseDetector):
    def __init__(self, model_path: str = "services/inference/models/blaze_face_short_range.tflite", min_detection_confidence: float = 0.5):
        import os
        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision
        self.detector = None
        if os.path.exists(model_path):
            base_options = tasks.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_detection_confidence)
            self.detector = vision.FaceDetector.create_from_options(options)
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if image is None or image.size == 0 or self.detector is None:
            return []
        import mediapipe as mp
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        results = []
        if res.detections:
            for det in res.detections:
                b = det.bounding_box
                score = float(det.categories[0].score) if det.categories else 1.0
                x1 = max(0.0, float(b.origin_x))
                y1 = max(0.0, float(b.origin_y))
                x2 = min(float(w), x1 + float(b.width))
                y2 = min(float(h), y1 + float(b.height))
                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
                results.append(FaceDetection(bbox=bbox, confidence=score))
        return results
