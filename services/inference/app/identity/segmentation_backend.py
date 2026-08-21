"""
Segmentation Backend Abstraction & Provenance Tracking.

Spec: docs/architecture/IDENTITY_ASSET.md (Revision 3)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

import cv2
import numpy as np

from app.identity.identity_asset import SemanticSegmentationResult

logger = logging.getLogger(__name__)


class SegmentationBackend(ABC):
    """
    Abstract Interface for Semantic Segmentation Backends.
    """

    @abstractmethod
    def segment(self, bgr_image: np.ndarray) -> SemanticSegmentationResult:
        """
        Segment a BGR reference image and return SemanticSegmentationResult with masks and provenance.
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @property
    @abstractmethod
    def backend_version(self) -> str:
        pass


class DummySegmentationBackend(SegmentationBackend):
    """
    Fallback/Mock Segmentation Backend for testing or when neural models are uninitialized.
    """

    def __init__(self, supported_classes: Optional[Set[str]] = None):
        self._classes = supported_classes or {"face", "hair", "clothing", "background"}

    @property
    def backend_name(self) -> str:
        return "DummySegmentationBackend"

    @property
    def backend_version(self) -> str:
        return "1.0.0"

    def segment(self, bgr_image: np.ndarray) -> SemanticSegmentationResult:
        h, w = bgr_image.shape[:2]
        masks = {}
        confidence = {}

        for cls_name in self._classes:
            # Create synthetic oval/box masks for dummy testing
            m = np.zeros((h, w), dtype=np.uint8)
            if cls_name == "face":
                cv2.ellipse(m, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 255, -1)
            elif cls_name == "hair":
                cv2.ellipse(m, (w // 2, h // 3), (w // 4, h // 6), 0, 0, 360, 255, -1)
            elif cls_name == "clothing":
                cv2.rectangle(m, (w // 4, h // 2), (3 * w // 4, h), 255, -1)
            elif cls_name == "background":
                m.fill(255)
                cv2.rectangle(m, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 0, -1)

            masks[cls_name] = m
            confidence[cls_name] = 0.95

        return SemanticSegmentationResult(
            masks=masks,
            available_classes=set(self._classes),
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            confidence_metadata=confidence,
        )


class MediaPipeSegmentationBackend(SegmentationBackend):
    """
    MediaPipe Multiclass ImageSegmenter Backend.
    """

    def __init__(self, model_asset_path: Optional[Union[str, Path]] = None):
        self._model_path = Path(model_asset_path) if model_asset_path else None
        self._segmenter = None
        self._initialized = False
        self._init_segmenter()

    def _init_segmenter(self):
        if self._model_path and not self._model_path.exists():
            logger.warning("MediaPipe ImageSegmenter model path not found: %s", self._model_path)
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision

            if not self._model_path:
                # Default search location
                models_dir = Path(__file__).resolve().parent.parent.parent / "models"
                candidate = models_dir / "selfie_multiclass_256x256.tflite"
                if candidate.exists():
                    self._model_path = candidate

            if self._model_path and self._model_path.exists():
                opts = vision.ImageSegmenterOptions(
                    base_options=tasks.BaseOptions(model_asset_path=str(self._model_path)),
                    output_category_mask=True,
                    output_confidence_masks=True,
                )
                self._segmenter = vision.ImageSegmenter.create_from_options(opts)
                self._initialized = True
                logger.info("Initialized MediaPipeSegmentationBackend with model %s", self._model_path.name)
        except Exception as e:
            logger.warning("Failed to initialize MediaPipe ImageSegmenter: %s", e)
            self._initialized = False

    @property
    def backend_name(self) -> str:
        return "MediaPipeImageSegmenter"

    @property
    def backend_version(self) -> str:
        return "1.0.0"

    def segment(self, bgr_image: np.ndarray) -> SemanticSegmentationResult:
        if not self._initialized or self._segmenter is None:
            # Fallback to DummySegmentationBackend if neural model asset is unavailable
            logger.debug("MediaPipe ImageSegmenter uninitialized, falling back to dummy backend")
            return DummySegmentationBackend().segment(bgr_image)

        import mediapipe as mp
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        seg_res = self._segmenter.segment(mp_img)

        # MediaPipe Selfie Multiclass labels:
        # 0: background, 1: hair, 2: body/skin, 3: face/skin, 4: clothes, 5: accessories
        label_map = {
            0: "background",
            1: "hair",
            2: "body",
            3: "face",
            4: "clothing",
        }

        masks = {}
        available_classes = set()
        confidence_meta = {}

        if seg_res.category_mask is not None:
            cat_mask = seg_res.category_mask.numpy_view()
            cat_mask_resized = cv2.resize(cat_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            for class_idx, class_name in label_map.items():
                binary_mask = (cat_mask_resized == class_idx).astype(np.uint8) * 255
                pixel_count = int(np.count_nonzero(binary_mask))

                if pixel_count > 50:  # Only report class as available if at least 50 pixels present
                    masks[class_name] = binary_mask
                    available_classes.add(class_name)
                    confidence_meta[class_name] = float(pixel_count / (w * h))

        return SemanticSegmentationResult(
            masks=masks,
            available_classes=available_classes,
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            confidence_metadata=confidence_meta,
        )
