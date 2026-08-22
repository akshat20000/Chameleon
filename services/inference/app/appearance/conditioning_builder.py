"""
Appearance Conditioning Builder for Phase 2.5A.

Spec: docs/architecture/ADR/ADR-006-appearance-synthesis-architecture.md
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.appearance.pose_renderer import CameraParameters, SkeletalPoseRenderer
from app.identity.identity_asset import IdentityAsset, SegmentedReferenceView
from app.motion.retargeted_actor_state import RetargetedActorState

logger = logging.getLogger(__name__)


@dataclass
class AppearanceTemporalState:
    previous_frame_bgr: Optional[np.ndarray] = None
    previous_synthetic_result: Optional[Any] = None
    optical_flow: Optional[np.ndarray] = None
    frame_index: int = 0


@dataclass
class AppearanceConditioningState:
    identity_embedding: np.ndarray          # (512,) float32 ArcFace vector
    reference_views: List[SegmentedReferenceView]
    pose_map_2d: np.ndarray                 # Rendered 2D skeletal line map uint8 BGR
    keypoints_2d: np.ndarray                # (N, 2) float32 normalized/pixel keypoints
    joint_confidence: np.ndarray            # (N,) float32 joint confidence
    camera_parameters: Optional[CameraParameters] = None
    region_guidance: Dict[str, np.ndarray] = field(default_factory=dict)
    frame_index: int = 0
    timestamp_s: float = 0.0


class AppearanceConditioningBuilder:
    """
    Converts IdentityAsset (Phase 2.3) + RetargetedActorState (Phase 2.4D) into AppearanceConditioningState.

    INVARIANT: RetargetedActorState and IdentityAsset are strictly read-only and never mutated.
    """

    def __init__(
        self,
        pose_renderer: Optional[SkeletalPoseRenderer] = None,
        camera_params: Optional[CameraParameters] = None,
    ):
        self.camera = camera_params or CameraParameters()
        self.pose_renderer = pose_renderer or SkeletalPoseRenderer(self.camera)

    def build_conditioning(
        self,
        identity_asset: IdentityAsset,
        actor_state: RetargetedActorState,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> AppearanceConditioningState:
        """
        Build AppearanceConditioningState from identity asset and retargeted actor state.

        Parameters
        ----------
        identity_asset : IdentityAsset
            Target reference identity package (READ-ONLY).
        actor_state : RetargetedActorState
            Current retargeted kinematic actor pose (READ-ONLY).
        frame_index : int
            Sequence frame index.
        timestamp_s : float
            Frame timestamp in seconds.

        Returns
        -------
        AppearanceConditioningState
        """
        # Validate fused identity embedding
        fused_emb = identity_asset.fused_identity_embedding
        if fused_emb is None or fused_emb.shape != (512,):
            raise ValueError(f"Invalid fused identity embedding in asset '{identity_asset.identity_id}'")

        # Project 3D actor joints to 2D using SkeletalPoseRenderer
        keypoints_2d, confidence = self.pose_renderer.project_joints_to_2d(actor_state)

        # Render 2D skeletal pose conditioning map
        pose_map_2d = self.pose_renderer.render_pose_map(keypoints_2d, confidence)

        # Extract available region guidance masks from reference views (face, hair, clothing, body)
        region_guidance: Dict[str, np.ndarray] = {}
        if identity_asset.segmented_views:
            for view in identity_asset.segmented_views:
                if view.segmentation and view.segmentation.masks:
                    for cls_name, mask_arr in view.segmentation.masks.items():
                        if cls_name not in region_guidance:
                            region_guidance[cls_name] = mask_arr.copy()

        return AppearanceConditioningState(
            identity_embedding=fused_emb.copy(),
            reference_views=list(identity_asset.segmented_views),
            pose_map_2d=pose_map_2d,
            keypoints_2d=keypoints_2d,
            joint_confidence=confidence,
            camera_parameters=self.camera,
            region_guidance=region_guidance,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
