"""
Reference Quality Analyzer, FacePoseEstimator, and Stage 2 Pose Diversity Filter.

Spec: docs/architecture/IDENTITY_ASSET.md
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe 478 Landmark indices for face pose estimation
# Left eye center: 468 (iris) or 33 (outer corner)
# Right eye center: 473 (iris) or 263 (outer corner)
# Chin tip: 152
# Nose tip: 1
LM_LEFT_EYE = 468
LM_RIGHT_EYE = 473
LM_NOSE_TIP = 1
LM_CHIN = 152


@dataclass(frozen=True)
class ReferenceQualityThresholds:
    min_face_size_px: int = 112
    min_blur_score: float = 100.0
    min_detection_confidence: float = 0.8
    max_yaw_deg: float = 30.0
    max_pitch_deg: float = 30.0
    min_quality_weight: float = 1e-3


@dataclass
class SelectedReferenceView:
    view_index: int
    image_path: Path
    image_bgr: np.ndarray
    face_landmarks_3d: np.ndarray        # (478, 3) float32
    face_bbox: Tuple[int, int, int, int] # (x, y, w, h)
    quality_score: float                  # normalized q_i in [0, 1]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


class FacePoseEstimator:
    """
    Mathematical Face Pose Estimator using MediaPipe 3D face landmarks.

    Basis Frame Construction:
    -------------------------
    1. eye_center = 0.5 * (left_eye + right_eye)
    2. x_face = normalize(right_eye - left_eye)
    3. z_face = normalize(cross(x_face, chin - eye_center))   [pointing forward out of face]
    4. y_face = normalize(cross(z_face, x_face))               [pointing up along face]
    5. R_face = column_stack([x_face, y_face, z_face])

    Euler angles (camera-relative, degrees):
    -----------------------------------------
    yaw   = arctan2(R_face[0, 2], R_face[2, 2])
    pitch = arctan2(-R_face[1, 2], sqrt(R_face[0, 2]^2 + R_face[2, 2]^2))
    roll  = arctan2(R_face[1, 0], R_face[1, 1])
    """

    @staticmethod
    def estimate_pose(landmarks_3d: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate camera-relative yaw, pitch, roll in degrees from 3D face landmarks.

        Parameters
        ----------
        landmarks_3d : np.ndarray (478, 3) or (N, 3) float32

        Returns
        -------
        Tuple[float, float, float]
            (yaw_deg, pitch_deg, roll_deg)
        """
        if landmarks_3d is None or len(landmarks_3d) < 474:
            return 0.0, 0.0, 0.0

        # Fallback to secondary landmarks if iris centers (468, 473) are absent
        l_eye_idx = LM_LEFT_EYE if len(landmarks_3d) > LM_LEFT_EYE else 33
        r_eye_idx = LM_RIGHT_EYE if len(landmarks_3d) > LM_RIGHT_EYE else 263
        chin_idx = LM_CHIN if len(landmarks_3d) > LM_CHIN else 152

        left_eye = landmarks_3d[l_eye_idx, :3]
        right_eye = landmarks_3d[r_eye_idx, :3]
        chin = landmarks_3d[chin_idx, :3]

        eye_center = 0.5 * (left_eye + right_eye)

        # 1. Primary lateral vector x_face (pointing from left eye to right eye)
        x_vec = right_eye - left_eye
        norm_x = float(np.linalg.norm(x_vec))
        if norm_x < 1e-6:
            return 0.0, 0.0, 0.0
        x_face = x_vec / norm_x

        # 2. Vector from eye center to chin (pointing down along face)
        down_vec = chin - eye_center
        
        # 3. Normal vector z_face (pointing out of face towards camera +Z)
        z_vec = np.cross(down_vec, x_face)
        norm_z = float(np.linalg.norm(z_vec))
        if norm_z < 1e-6:
            return 0.0, 0.0, 0.0
        z_face = z_vec / norm_z

        # 4. Vertical vector y_face (up along face +Y)
        y_face = np.cross(z_face, x_face)
        y_face = y_face / (np.linalg.norm(y_face) + 1e-9)

        # 5. SO(3) Rotation matrix
        R_face = np.column_stack([x_face, y_face, z_face])

        # Camera-relative Euler angles in degrees
        yaw_rad = math.atan2(R_face[0, 2], R_face[2, 2])
        pitch_rad = math.atan2(-R_face[1, 2], math.sqrt(R_face[0, 2] ** 2 + R_face[2, 2] ** 2))
        roll_rad = math.atan2(R_face[1, 0], R_face[1, 1])

        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)
        roll_deg = math.degrees(roll_rad)

        return yaw_deg, pitch_deg, roll_deg


class ReferenceQualityChecker:
    """
    Reference Quality Analyzer & Stage 2 Pose Diversity Filter.
    """

    def __init__(self, thresholds: Optional[ReferenceQualityThresholds] = None):
        self.thresholds = thresholds or ReferenceQualityThresholds()

    def compute_blur_score(self, bgr_image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> float:
        """Compute Laplacian variance blur score within face bounding box."""
        x, y, w, h = face_bbox
        img_h, img_w = bgr_image.shape[:2]

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img_w, x + w), min(img_h, y + h)

        if x1 - x0 < 10 or y1 - y0 < 10:
            crop = bgr_image
        else:
            crop = bgr_image[y0:y1, x0:x1]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(np.var(laplacian))
        return var

    def evaluate_view(
        self,
        view_index: int,
        image_path: Path,
        image_bgr: np.ndarray,
        landmarks_3d: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        detection_confidence: float,
    ) -> Optional[SelectedReferenceView]:
        """
        Evaluate quality of a single reference view and return SelectedReferenceView if accepted.
        """
        w_bbox, h_bbox = face_bbox[2], face_bbox[3]
        if min(w_bbox, h_bbox) < self.thresholds.min_face_size_px:
            logger.debug("View %d rejected: face size %dx%d < min %d",
                         view_index, w_bbox, h_bbox, self.thresholds.min_face_size_px)
            return None

        if detection_confidence < self.thresholds.min_detection_confidence:
            logger.debug("View %d rejected: confidence %.2f < min %.2f",
                         view_index, detection_confidence, self.thresholds.min_detection_confidence)
            return None

        blur_score = self.compute_blur_score(image_bgr, face_bbox)
        if blur_score < self.thresholds.min_blur_score:
            logger.debug("View %d rejected: blur score %.1f < min %.1f",
                         view_index, blur_score, self.thresholds.min_blur_score)
            return None

        yaw, pitch, roll = FacePoseEstimator.estimate_pose(landmarks_3d)

        if abs(yaw) > self.thresholds.max_yaw_deg:
            logger.debug("View %d rejected: yaw %.1f° > max %.1f°",
                         view_index, yaw, self.thresholds.max_yaw_deg)
            return None

        if abs(pitch) > self.thresholds.max_pitch_deg:
            logger.debug("View %d rejected: pitch %.1f° > max %.1f°",
                         view_index, pitch, self.thresholds.max_pitch_deg)
            return None

        # Normalized component scores in [0, 1]
        q_res = float(np.clip(min(w_bbox, h_bbox) / 256.0, 0.0, 1.0))
        q_sharp = float(np.clip(blur_score / 500.0, 0.0, 1.0))
        q_front = float(np.clip(1.0 - (math.sqrt(yaw**2 + pitch**2) / 45.0), 0.0, 1.0))
        q_conf = float(np.clip(detection_confidence, 0.0, 1.0))

        # Composite quality score
        q_composite = q_res * q_sharp * q_front * q_conf

        if q_composite < self.thresholds.min_quality_weight:
            logger.debug("View %d rejected: composite score %.4f < min weight %.4f",
                         view_index, q_composite, self.thresholds.min_quality_weight)
            return None

        return SelectedReferenceView(
            view_index=view_index,
            image_path=image_path,
            image_bgr=image_bgr,
            face_landmarks_3d=landmarks_3d,
            face_bbox=face_bbox,
            quality_score=q_composite,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )

    def filter_pose_diversity(
        self,
        candidate_views: List[SelectedReferenceView],
        pose_similarity_deg: float = 3.0,
    ) -> List[SelectedReferenceView]:
        """
        Stage 2 Pose Diversity Filter: deduplicate views with near-identical head pose angles.
        """
        if not candidate_views:
            return []

        # Sort candidate views by quality score descending
        sorted_views = sorted(candidate_views, key=lambda v: v.quality_score, reverse=True)
        selected: List[SelectedReferenceView] = []

        for v in sorted_views:
            is_duplicate = False
            for s in selected:
                d_yaw = abs(v.yaw_deg - s.yaw_deg)
                d_pitch = abs(v.pitch_deg - s.pitch_deg)
                if d_yaw < pose_similarity_deg and d_pitch < pose_similarity_deg:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(v)

        return selected
