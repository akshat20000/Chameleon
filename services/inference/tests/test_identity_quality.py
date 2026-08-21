"""
Unit test suite for Step 3 — Reference Quality Analyzer, FacePoseEstimator matrix basis, and Stage 2 Pose Diversity Filter.
"""

import sys
import math
import cv2
import numpy as np
import pytest
from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.identity.quality_checker import (
    FacePoseEstimator,
    ReferenceQualityChecker,
    ReferenceQualityThresholds,
    SelectedReferenceView,
)


def _build_synthetic_face_landmarks(yaw_deg: float = 0.0, pitch_rad: float = 0.0) -> np.ndarray:
    """Helper to build 478 synthetic 3D face landmarks with specified head rotation."""
    lms = np.zeros((478, 3), dtype=np.float32)

    # Base frontal coordinates (centered at origin)
    left_eye = np.array([-0.035, 0.02, 0.0], dtype=np.float32)
    right_eye = np.array([0.035, 0.02, 0.0], dtype=np.float32)
    nose = np.array([0.0, 0.0, 0.03], dtype=np.float32)
    chin = np.array([0.0, -0.07, 0.0], dtype=np.float32)

    # Rotate around Y axis (yaw)
    yaw_rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)

    lms[468] = R_y @ left_eye
    lms[473] = R_y @ right_eye
    lms[1] = R_y @ nose
    lms[152] = R_y @ chin

    return lms


def test_face_pose_estimator_matrix_basis():
    # 1. Frontal pose
    lms_frontal = _build_synthetic_face_landmarks(yaw_deg=0.0)
    yaw, pitch, roll = FacePoseEstimator.estimate_pose(lms_frontal)
    assert abs(yaw) < 2.0
    assert abs(pitch) < 2.0
    assert abs(roll) < 2.0

    # 2. 20 degree yaw (head turned right)
    lms_yaw = _build_synthetic_face_landmarks(yaw_deg=20.0)
    yaw_20, pitch_20, roll_20 = FacePoseEstimator.estimate_pose(lms_yaw)
    assert abs(yaw_20 - 20.0) < 3.0


def test_blur_score_calculation():
    checker = ReferenceQualityChecker()

    # Sharp image: checkerboard pattern
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp_img[::20, :] = 255
    sharp_score = checker.compute_blur_score(sharp_img, (0, 0, 200, 200))
    assert sharp_score > 500.0

    # Blurry image: uniform gray
    blurry_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    blurry_score = checker.compute_blur_score(blurry_img, (0, 0, 200, 200))
    assert blurry_score < 10.0


def test_quality_checker_rejections(tmp_path):
    thresholds = ReferenceQualityThresholds(
        min_face_size_px=112,
        min_blur_score=100.0,
        min_detection_confidence=0.8,
        max_yaw_deg=30.0,
    )
    checker = ReferenceQualityChecker(thresholds)
    lms = _build_synthetic_face_landmarks(yaw_deg=0.0)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[::10, :] = 255

    # 1. Small face size rejection
    view_small = checker.evaluate_view(0, tmp_path / "v0.png", img, lms, (10, 10, 50, 50), 0.95)
    assert view_small is None

    # 2. Low confidence rejection
    view_low_conf = checker.evaluate_view(1, tmp_path / "v1.png", img, lms, (10, 10, 200, 200), 0.5)
    assert view_low_conf is None

    # 3. Blurry image rejection
    blur_img = np.ones((300, 300, 3), dtype=np.uint8) * 120
    view_blur = checker.evaluate_view(2, tmp_path / "v2.png", blur_img, lms, (10, 10, 200, 200), 0.95)
    assert view_blur is None

    # 4. Large yaw rejection
    lms_large_yaw = _build_synthetic_face_landmarks(yaw_deg=45.0)
    view_yaw = checker.evaluate_view(3, tmp_path / "v3.png", img, lms_large_yaw, (10, 10, 200, 200), 0.95)
    assert view_yaw is None

    # 5. Valid high quality view accepted
    view_valid = checker.evaluate_view(4, tmp_path / "v4.png", img, lms, (10, 10, 200, 200), 0.95)
    assert view_valid is not None
    assert view_valid.quality_score > 0.1


def test_stage_2_pose_diversity_filter(tmp_path):
    checker = ReferenceQualityChecker()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_lms = np.zeros((478, 3), dtype=np.float32)

    v1 = SelectedReferenceView(0, tmp_path / "1.png", dummy_img, dummy_lms, (0, 0, 150, 150), 0.9, yaw_deg=5.0, pitch_deg=0.0, roll_deg=0.0)
    v2 = SelectedReferenceView(1, tmp_path / "2.png", dummy_img, dummy_lms, (0, 0, 150, 150), 0.8, yaw_deg=5.2, pitch_deg=0.1, roll_deg=0.0) # Near duplicate of v1
    v3 = SelectedReferenceView(2, tmp_path / "3.png", dummy_img, dummy_lms, (0, 0, 150, 150), 0.85, yaw_deg=18.0, pitch_deg=-5.0, roll_deg=0.0) # Diverse pose

    filtered = checker.filter_pose_diversity([v1, v2, v3], pose_similarity_deg=3.0)

    assert len(filtered) == 2
    view_indices = {v.view_index for v in filtered}
    assert 0 in view_indices
    assert 2 in view_indices
    assert 1 not in view_indices  # v2 dropped as pose duplicate of higher quality v1
