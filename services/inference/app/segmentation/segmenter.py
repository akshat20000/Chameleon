"""
Semantic face and body segmentation using MediaPipe Tasks API (mediapipe 1.0.0).

Model Class Mapping
-------------------
The selfie_multiclass_256x256.tflite model metadata exposes the following exact 6 category labels:

    Index 0: 'background'
    Index 1: 'hair'       -> hair_mask (bool)
    Index 2: 'body-skin'  -> skin_mask (bool)
    Index 3: 'face-skin'  -> face_mask (bool)
    Index 4: 'clothes'    -> accessible via class_mask (uint8 category index 4)
    Index 5: 'others'     -> accessible via class_mask (uint8 category index 5)

Mask Semantics
--------------
class_mask : np.ndarray, shape (H, W), dtype uint8
    Raw multi-class category label map where pixel values are in range [0, 5].

face_mask : np.ndarray, shape (H, W), dtype bool
    Boolean mask indicating pixels corresponding to face skin (index 3).

hair_mask : np.ndarray, shape (H, W), dtype bool
    Boolean mask indicating pixels corresponding to hair (index 1).

skin_mask : np.ndarray, shape (H, W), dtype bool
    Boolean mask indicating pixels corresponding to body skin (index 2).

Deferred Capabilities
---------------------
Eyes, mouth, and neck segmentation are not independent classes in the
selfie_multiclass_256x256 model and are intentionally deferred.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config.settings import get_settings
from app.pipeline.result import SegmentationResult

logger = logging.getLogger(__name__)


class BaseSegmenter(ABC):
    """Abstract base class for semantic image segmenters."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the segmenter is initialized and ready for inference."""
        pass

    @abstractmethod
    def segment(self, image: np.ndarray) -> Optional[SegmentationResult]:
        """
        Run semantic segmentation on a BGR image frame.

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.

        Returns
        -------
        Optional[SegmentationResult]
            Populated SegmentationResult containing category mask and boolean
            sub-masks, or None if segmentation fails or the component is not ready.
        """
        pass


class MediaPipeSegmenter(BaseSegmenter):
    """
    Semantic segmenter backed by MediaPipe Tasks Image Segmenter API.

    Parameters
    ----------
    model_path : Optional[str]
        Path to the selfie_multiclass_256x256.tflite model file.
        Defaults to settings.segmentation_model_path.
    """

    CLASS_BACKGROUND = 0
    CLASS_HAIR = 1
    CLASS_BODY_SKIN = 2
    CLASS_FACE_SKIN = 3
    CLASS_CLOTHES = 4
    CLASS_OTHERS = 5

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(get_settings().segmentation_model_path)

        self._model_path = model_path
        self._segmenter = None
        self._is_ready = False

        if not os.path.exists(self._model_path):
            logger.warning(f"Segmentation model not found at path: {self._model_path}")
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self._model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True,
                output_confidence_masks=False,
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            self._is_ready = True
        except Exception as err:
            logger.warning(
                f"Failed to initialize MediaPipe ImageSegmenter from {self._model_path}: {err}"
            )
            self._segmenter = None
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def segment(self, image: np.ndarray) -> Optional[SegmentationResult]:
        """
        Run semantic segmentation on the input BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.

        Returns
        -------
        Optional[SegmentationResult]
            SegmentationResult or None if unready or input is invalid.
        """
        if not self._is_ready or self._segmenter is None:
            return None

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None

        if len(image.shape) != 3 or image.shape[2] != 3:
            return None

        h_orig, w_orig = image.shape[:2]

        import mediapipe as mp

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = self._segmenter.segment(mp_image)
        if result is None or result.category_mask is None:
            return None

        category_mask_np = result.category_mask.numpy_view()
        if category_mask_np.ndim == 3:
            category_mask_np = category_mask_np[:, :, 0]

        category_mask_np = category_mask_np.astype(np.uint8)
        h_mask, w_mask = category_mask_np.shape[:2]

        if (h_mask, w_mask) != (h_orig, w_orig):
            class_mask = cv2.resize(
                category_mask_np,
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            class_mask = category_mask_np

        hair_mask = (class_mask == self.CLASS_HAIR)
        skin_mask = (class_mask == self.CLASS_BODY_SKIN)
        face_mask = (class_mask == self.CLASS_FACE_SKIN)

        return SegmentationResult(
            face_mask=face_mask,
            hair_mask=hair_mask,
            skin_mask=skin_mask,
            class_mask=class_mask,
        )

    def close(self) -> None:
        """Close the underlying MediaPipe segmenter instance."""
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:
                pass
            self._segmenter = None
            self._is_ready = False
