"""
Target identity representation & facial feature encoder using ONNX Runtime.

Model & Preprocessing Contract
------------------------------
Model Architecture : ArcFace (MobileFaceNet / ResNet family)
Default Model File : models/w600k_mbf.onnx (~13.6 MB)
Input Format       : RGB image tensor, shape (1, 3, 112, 112), dtype float32
Input Preprocessing: (x_rgb - 127.5) / 127.5  --> range [-1.0, 1.0]
Output Tensor      : shape (1, 512), dtype float32
Output Normalized  : L2-normalized 512-dimensional embedding vector (norm == 1.0)

Five-Point MediaPipe Landmark Mapping
--------------------------------------
MediaPipe 478 landmark indices used for 5-point face alignment:
    - Left eye center  : Index 468 (iris center)
    - Right eye center : Index 473 (iris center)
    - Nose tip         : Index 1
    - Left mouth corner: Index 61
    - Right mouth corner: Index 291
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Standard ArcFace 5-point destination template for 112x112 aligned face chip
ARCFACE_TARGET_5PTS = np.array(
    [
        [38.2946, 51.6963],  # left eye center
        [73.5318, 51.5014],  # right eye center
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

MEDIAPIPE_5PT_INDICES = [468, 473, 1, 61, 291]


def normalize_embedding(vector: np.ndarray) -> Optional[np.ndarray]:
    """
    Safely L2-normalize a 1D feature vector into float32 precision.

    Parameters
    ----------
    vector : np.ndarray
        1D feature vector.

    Returns
    -------
    Optional[np.ndarray]
        L2-normalized float32 array with unit norm (1.0), or None if input is
        invalid, non-finite, empty, or zero.
    """
    if vector is None or not isinstance(vector, np.ndarray) or vector.size == 0:
        return None

    vec_1d = vector.flatten()
    if not np.all(np.isfinite(vec_1d)):
        return None

    vec_f32 = vec_1d.astype(np.float32)
    norm = float(np.linalg.norm(vec_f32))

    if norm <= 1e-12:
        return None

    normalized = vec_f32 / norm
    return normalized.astype(np.float32)


def fuse_embeddings(
    embeddings: List[np.ndarray],
    weights: Optional[List[float]] = None,
    min_quality_weight: float = 1e-3,
) -> Optional[np.ndarray]:
    """
    Compute multi-reference quality-weighted normalized fused identity embedding.

    Parameters
    ----------
    embeddings : List[np.ndarray]
        List of 1D feature vectors to fuse.
    weights : Optional[List[float]]
        Optional normalized quality weights q_i in [0, 1] for each vector.
    min_quality_weight : float
        Minimum weight threshold; vectors with weight below this epsilon are omitted.

    Returns
    -------
    Optional[np.ndarray]
        L2-normalized float32 fused embedding vector (norm == 1.0), or None if invalid.
    """
    if not embeddings or not isinstance(embeddings, list):
        return None

    if weights is not None and len(weights) != len(embeddings):
        logger.warning(f"Weights length mismatch: got {len(weights)}, expected {len(embeddings)}")
        return None

    valid_embeddings: List[np.ndarray] = []
    valid_weights: List[float] = []
    expected_dim: Optional[int] = None

    for idx, emb in enumerate(embeddings):
        w = float(weights[idx]) if weights is not None else 1.0
        if w < min_quality_weight:
            logger.debug("Omitting view %d from fusion: quality weight %.4f < min %.4f", idx, w, min_quality_weight)
            continue

        norm_emb = normalize_embedding(emb)
        if norm_emb is None:
            continue

        if expected_dim is None:
            expected_dim = norm_emb.shape[0]
        elif norm_emb.shape[0] != expected_dim:
            logger.warning(
                f"Embedding dimension mismatch in fusion: expected {expected_dim}, got {norm_emb.shape[0]}"
            )
            return None

        valid_embeddings.append(norm_emb)
        valid_weights.append(w)

    if not valid_embeddings:
        return None

    emb_arr = np.array(valid_embeddings, dtype=np.float32)  # (N, D)
    w_arr = np.array(valid_weights, dtype=np.float32).reshape(-1, 1)  # (N, 1)

    weighted_sum = np.sum(emb_arr * w_arr, axis=0)
    return normalize_embedding(weighted_sum)


def extract_5pt_landmarks_from_478(points_2d: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract 5 key points from MediaPipe 478 2D landmark pixel array.

    Parameters
    ----------
    points_2d : np.ndarray
        Shape (478, 2), float32.

    Returns
    -------
    Optional[np.ndarray]
        Shape (5, 2), float32 array [left_eye, right_eye, nose, left_mouth, right_mouth].
    """
    if (
        points_2d is None
        or not isinstance(points_2d, np.ndarray)
        or points_2d.shape[0] < 478
        or points_2d.shape[1] < 2
    ):
        return None

    return points_2d[MEDIAPIPE_5PT_INDICES, :2].astype(np.float32)


def align_face_5pt(
    image: np.ndarray,
    landmarks_5pt: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """
    Perform 5-point similarity-transform face alignment to crop face chip.

    Parameters
    ----------
    image : np.ndarray
        BGR full image frame, shape (H, W, 3), dtype uint8.
    landmarks_5pt : np.ndarray
        5 key points, shape (5, 2), dtype float32.
    target_size : Tuple[int, int]
        Output resolution (W, H). Defaults to (112, 112).

    Returns
    -------
    Optional[np.ndarray]
        Aligned BGR face chip, shape (H_target, W_target, 3), dtype uint8,
        or None if input is invalid or transformation estimation fails.
    """
    if (
        image is None
        or not isinstance(image, np.ndarray)
        or image.size == 0
        or len(image.shape) != 3
        or image.shape[2] != 3
    ):
        return None

    if (
        landmarks_5pt is None
        or not isinstance(landmarks_5pt, np.ndarray)
        or landmarks_5pt.shape != (5, 2)
        or not np.all(np.isfinite(landmarks_5pt))
    ):
        return None

    dst_pts = ARCFACE_TARGET_5PTS.copy()
    if target_size != (112, 112):
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        dst_pts[:, 0] *= scale_x
        dst_pts[:, 1] *= scale_y

    src_pts = landmarks_5pt.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return None

    aligned = cv2.warpAffine(
        image,
        M,
        target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


class BaseIdentityEncoder(ABC):
    """Abstract base class for facial identity encoders."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the identity encoder is loaded and ready."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output embedding vector dimension (e.g. 512)."""
        pass

    @abstractmethod
    def extract_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract L2-normalized identity embedding from an aligned face chip.

        Parameters
        ----------
        aligned_face : np.ndarray
            Aligned BGR face image chip, shape (112, 112, 3), dtype uint8.

        Returns
        -------
        Optional[np.ndarray]
            L2-normalized float32 1D embedding vector, or None if extraction fails.
        """
        pass


class ONNXIdentityEncoder(BaseIdentityEncoder):
    """
    Facial identity encoder backed by ArcFace ONNX model executed via ONNX Runtime.

    Parameters
    ----------
    model_path : Optional[str]
        Filesystem path to the ArcFace .onnx model asset.
        Defaults to settings.identity_model_path.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(get_settings().identity_model_path)

        self._model_path = model_path
        self._session = None
        self._is_ready = False
        self._input_name = ""
        self._output_name = ""
        self._embedding_dim = 512

        if not os.path.exists(self._model_path):
            logger.warning(f"Identity model not found at path: {self._model_path}")
            return

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            inp_info = self._session.get_inputs()[0]
            out_info = self._session.get_outputs()[0]

            self._input_name = inp_info.name
            self._output_name = out_info.name

            if out_info.shape and len(out_info.shape) >= 2:
                self._embedding_dim = int(out_info.shape[1])

            self._is_ready = True
        except Exception as err:
            logger.warning(
                f"Failed to initialize ONNX IdentityEncoder from {self._model_path}: {err}"
            )
            self._session = None
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def extract_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract L2-normalized identity embedding vector from an aligned BGR face chip.

        Parameters
        ----------
        aligned_face : np.ndarray
            Aligned BGR face chip, shape (112, 112, 3), dtype uint8.

        Returns
        -------
        Optional[np.ndarray]
            L2-normalized float32 1D embedding vector, shape (512,), or None if failed.
        """
        if not self._is_ready or self._session is None:
            return None

        if (
            aligned_face is None
            or not isinstance(aligned_face, np.ndarray)
            or aligned_face.size == 0
            or len(aligned_face.shape) != 3
            or aligned_face.shape[2] != 3
        ):
            return None

        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        img_f32 = (rgb.astype(np.float32) - 127.5) / 127.5
        nchw = np.transpose(img_f32, (2, 0, 1))
        tensor_input = np.expand_dims(nchw, axis=0)

        outputs = self._session.run([self._output_name], {self._input_name: tensor_input})
        if not outputs or outputs[0] is None:
            return None

        raw_vec = outputs[0][0]
        return normalize_embedding(raw_vec)
