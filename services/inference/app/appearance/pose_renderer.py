"""
Skeletal Pose Renderer for Phase 2.5 Appearance Conditioning.

Spec: docs/architecture/ADR/ADR-006-appearance-synthesis-architecture.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.motion.retargeted_actor_state import RetargetedActorState

logger = logging.getLogger(__name__)

# Standard skeleton bone connections for 2D pose map rendering: (joint_a_idx, joint_b_idx, color_bgr)
SKELETON_BONES_2D = [
    (0, 1, (0, 0, 255)),    # pelvis -> spine (Red)
    (1, 2, (0, 128, 255)),  # spine -> chest (Orange)
    (2, 3, (0, 255, 255)),  # chest -> neck (Yellow)
    (3, 4, (0, 255, 0)),    # neck -> head (Green)
    # Left Arm
    (2, 5, (255, 0, 0)),    # chest -> left_shoulder (Blue)
    (5, 6, (255, 128, 0)),  # left_shoulder -> left_elbow (Cyan)
    (6, 7, (255, 255, 0)),  # left_elbow -> left_wrist (Light Blue)
    # Right Arm
    (2, 8, (128, 0, 255)),  # chest -> right_shoulder (Purple)
    (8, 9, (255, 0, 255)),  # right_shoulder -> right_elbow (Magenta)
    (9, 10, (255, 128, 255)), # right_elbow -> right_wrist (Pink)
    # Left Leg
    (0, 11, (0, 255, 128)), # pelvis -> left_hip
    (11, 12, (0, 200, 200)),# left_hip -> left_knee
    (12, 13, (0, 150, 150)),# left_knee -> left_ankle
    # Right Leg
    (0, 14, (128, 255, 0)), # pelvis -> right_hip
    (14, 15, (200, 200, 0)),# right_hip -> right_knee
    (15, 16, (150, 150, 0)),# right_knee -> right_ankle
]


@dataclass
class CameraParameters:
    focal_length_px: float = 800.0
    principal_point_x: float = 256.0
    principal_point_y: float = 256.0
    image_width: int = 512
    image_height: int = 512


class SkeletalPoseRenderer:
    """
    Renders 2D skeletal pose conditioning maps and extracts keypoint arrays from RetargetedActorState.

    INVARIANT: RetargetedActorState is strictly read-only and never mutated.
    """

    def __init__(self, camera_params: Optional[CameraParameters] = None):
        self.camera = camera_params or CameraParameters()

    def project_joints_to_2d(self, actor_state: RetargetedActorState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D actor joints to 2D pixel coordinates and return (keypoints_2d, joint_confidence).

        Parameters
        ----------
        actor_state : RetargetedActorState
            Input retargeted actor state (READ-ONLY).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - keypoints_2d: (N, 2) float32 array in pixel coordinates (x, y)
            - joint_confidence: (N,) float32 array in [0, 1]
        """
        # Support both joints and joint_world_positions attribute access
        joints_dict = getattr(actor_state, "joints", None)
        if joints_dict is None:
            joints_dict = getattr(actor_state, "joint_world_positions", {})

        joint_names = list(joints_dict.keys())
        num_joints = len(joint_names)

        keypoints_2d = np.zeros((num_joints, 2), dtype=np.float32)
        confidence = np.ones((num_joints,), dtype=np.float32)

        f = self.camera.focal_length_px
        cx = self.camera.principal_point_x
        cy = self.camera.principal_point_y

        for idx, j_name in enumerate(joint_names):
            p3d = joints_dict[j_name]
            if p3d is None:
                confidence[idx] = 0.0
                continue
            x, y, z = float(p3d[0]), float(p3d[1]), float(p3d[2])

            # Pinhole projection: assume camera at origin looking along +Z
            # In Chameleon canonical coordinate system, +Z is forward
            z_eff = max(z, 0.1)  # avoid div by zero if joint is behind camera
            u = (x * f / z_eff) + cx
            v = (-y * f / z_eff) + cy  # flip Y for image pixel coordinates

            keypoints_2d[idx, 0] = float(u)
            keypoints_2d[idx, 1] = float(v)

        return keypoints_2d, confidence

    def render_pose_map(
        self,
        keypoints_2d: np.ndarray,
        joint_confidence: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Render color-coded 2D skeletal line map uint8 BGR image.

        Parameters
        ----------
        keypoints_2d : np.ndarray
            (N, 2) float32 pixel coordinates.
        joint_confidence : Optional[np.ndarray]
            (N,) float32 joint confidence scores.
        image_shape : Optional[Tuple[int, int]]
            (height, width). Defaults to camera parameters.

        Returns
        -------
        np.ndarray
            uint8 BGR image shape (H, W, 3).
        """
        h = image_shape[0] if image_shape else self.camera.image_height
        w = image_shape[1] if image_shape else self.camera.image_width

        pose_map = np.zeros((h, w, 3), dtype=np.uint8)
        num_joints = len(keypoints_2d)

        # Draw bone connections
        for idx_a, idx_b, color in SKELETON_BONES_2D:
            if idx_a < num_joints and idx_b < num_joints:
                if joint_confidence is not None:
                    if joint_confidence[idx_a] < 0.5 or joint_confidence[idx_b] < 0.5:
                        continue

                pt_a = (int(round(keypoints_2d[idx_a, 0])), int(round(keypoints_2d[idx_a, 1])))
                pt_b = (int(round(keypoints_2d[idx_b, 0])), int(round(keypoints_2d[idx_b, 1])))

                cv2.line(pose_map, pt_a, pt_b, color, thickness=3, lineType=cv2.LINE_AA)

        # Draw joint keypoint circles
        for i in range(num_joints):
            if joint_confidence is not None and joint_confidence[i] < 0.5:
                continue
            pt = (int(round(keypoints_2d[i, 0])), int(round(keypoints_2d[i, 1])))
            cv2.circle(pose_map, pt, radius=4, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        return pose_map
