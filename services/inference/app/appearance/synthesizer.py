"""
Appearance Synthesizer Interface & Deterministic Baseline Engine.

Spec: docs/architecture/ADR/ADR-006-appearance-synthesis-architecture.md
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.appearance.conditioning_builder import AppearanceConditioningState, AppearanceTemporalState

logger = logging.getLogger(__name__)


@dataclass
class SyntheticFrameResult:
    frame_bgr: np.ndarray                   # Synthesized output RGB/BGR frame uint8
    synthetic_mask: Optional[np.ndarray]    # Subject mask uint8 (0 or 255)
    latency_ms: float
    metadata: Dict                          # Backend provenance & telemetry
    valid: bool = True
    warnings: List[str] = field(default_factory=list)


class BaseAppearanceSynthesizer(ABC):
    """
    Abstract Base Interface for Appearance Synthesis Engines.
    """

    @abstractmethod
    def synthesize_frame(
        self,
        conditioning: AppearanceConditioningState,
        temporal_state: Optional[AppearanceTemporalState] = None,
        background_bgr: Optional[np.ndarray] = None,
    ) -> SyntheticFrameResult:
        """
        Synthesize a character frame from appearance conditioning state.
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass


def compute_2d_similarity_transform(
    src_pt_a: Tuple[float, float],
    src_pt_b: Tuple[float, float],
    tgt_pt_a: Tuple[float, float],
    tgt_pt_b: Tuple[float, float],
) -> np.ndarray:
    """
    Compute 2D Similarity Transformation Matrix M in R^(2x3) (translation, rotation, uniform scale).
    """
    sa = np.array(src_pt_a, dtype=np.float32)
    sb = np.array(src_pt_b, dtype=np.float32)
    ta = np.array(tgt_pt_a, dtype=np.float32)
    tb = np.array(tgt_pt_b, dtype=np.float32)

    v_src = sb - sa
    v_tgt = tb - ta

    len_src = float(np.linalg.norm(v_src))
    len_tgt = float(np.linalg.norm(v_tgt))

    scale = len_tgt / (len_src + 1e-6)
    ang_src = math.atan2(v_src[1], v_src[0])
    ang_tgt = math.atan2(v_tgt[1], v_tgt[0])
    theta = ang_tgt - ang_src

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = scale * np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    t = ta - R @ sa
    M = np.column_stack([R, t])
    return M.astype(np.float32)


def compute_2d_affine_transform_3pt(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
) -> np.ndarray:
    """Compute 2D Affine Transform Matrix M in R^(2x3) from 3 point correspondences."""
    return cv2.getAffineTransform(src_pts.astype(np.float32), tgt_pts.astype(np.float32))


class BaselineArticulatedSynthesizer(BaseAppearanceSynthesizer):
    """
    Deterministic Articulated Region Transformation Renderer.

    - Articulated Limb Regions: Transformed using 2D Similarity Transformations (translation, rotation, uniform scale).
    - Torso / Clothing Regions: Transformed using 2D Affine Transformations.
    - Graceful Fallback: Respects Phase 2.3 region mask availability. Missing optional masks produce warnings in SyntheticFrameResult without crashing.
    """

    def __init__(self, output_width: int = 512, output_height: int = 512):
        self.output_w = output_width
        self.output_h = output_height

    @property
    def backend_name(self) -> str:
        return "BaselineArticulatedSynthesizer"

    def synthesize_frame(
        self,
        conditioning: AppearanceConditioningState,
        temporal_state: Optional[AppearanceTemporalState] = None,
        background_bgr: Optional[np.ndarray] = None,
    ) -> SyntheticFrameResult:
        t0 = time.perf_counter()
        warnings: List[str] = []

        # 1. Canvas setup
        if background_bgr is not None:
            canvas = cv2.resize(background_bgr, (self.output_w, self.output_h))
        else:
            canvas = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)

        synthetic_mask = np.zeros((self.output_h, self.output_w), dtype=np.uint8)

        # 2. Check reference views
        if not conditioning.reference_views:
            warnings.append("No reference views available in conditioning state; returning baseline pose map render")
            pose_resized = cv2.resize(conditioning.pose_map_2d, (self.output_w, self.output_h))
            t1 = time.perf_counter()
            return SyntheticFrameResult(
                frame_bgr=pose_resized,
                synthetic_mask=None,
                latency_ms=(t1 - t0) * 1000.0,
                metadata={"backend": self.backend_name, "mode": "fallback_pose_map"},
                valid=True,
                warnings=warnings,
            )

        ref_view = conditioning.reference_views[0]
        region_masks = conditioning.region_guidance
        kpts = conditioning.keypoints_2d

        # Check optional region availability
        for req_region in ("hair", "clothing", "torso", "left_arm", "right_arm"):
            if req_region not in region_masks:
                warnings.append(f"Optional region mask '{req_region}' unavailable in identity asset")

        # Load reference BGR image if available
        ref_bgr = None
        if hasattr(ref_view, "image_bgr") and ref_view.image_bgr is not None:
            ref_bgr = ref_view.image_bgr
        elif hasattr(ref_view, "image_path") and Path(ref_view.image_path).exists():
            ref_bgr = cv2.imread(str(ref_view.image_path))

        # 3. Deterministic Z-Order Compositing: Torso/Clothing -> Limbs -> Head/Face
        regions_processed = []

        # (A) Torso / Clothing Region (Affine Transform)
        torso_mask_name = "torso" if "torso" in region_masks else ("clothing" if "clothing" in region_masks else ("body" if "body" in region_masks else None))
        if torso_mask_name and torso_mask_name in region_masks and len(kpts) > 14:
            m_torso = region_masks[torso_mask_name]
            if np.count_nonzero(m_torso) > 0:
                h_m, w_m = m_torso.shape[:2]
                src_3pts = np.array([[w_m * 0.3, h_m * 0.4], [w_m * 0.7, h_m * 0.4], [w_m * 0.5, h_m * 0.8]], dtype=np.float32)
                tgt_3pts = np.array([kpts[5], kpts[8], 0.5 * (kpts[11] + kpts[14])], dtype=np.float32)

                M_aff = compute_2d_affine_transform_3pt(src_3pts, tgt_3pts)
                self._warp_and_composite(canvas, synthetic_mask, ref_bgr, m_torso, M_aff, (100, 150, 200))
                regions_processed.append("torso")

        # (B) Articulated Arm Regions (Similarity Transform)
        if "left_arm" in region_masks and len(kpts) > 7:
            m_larm = region_masks["left_arm"]
            if np.count_nonzero(m_larm) > 0:
                ys, xs = np.where(m_larm > 0)
                src_a = (float(np.min(xs)), float(np.min(ys)))
                src_b = (float(np.max(xs)), float(np.max(ys)))
                M_sim_larm = compute_2d_similarity_transform(src_a, src_b, (kpts[5][0], kpts[5][1]), (kpts[7][0], kpts[7][1]))
                self._warp_and_composite(canvas, synthetic_mask, ref_bgr, m_larm, M_sim_larm, (255, 128, 0))
                regions_processed.append("left_arm")

        if "right_arm" in region_masks and len(kpts) > 10:
            m_rarm = region_masks["right_arm"]
            if np.count_nonzero(m_rarm) > 0:
                ys, xs = np.where(m_rarm > 0)
                src_a = (float(np.min(xs)), float(np.min(ys)))
                src_b = (float(np.max(xs)), float(np.max(ys)))
                M_sim_rarm = compute_2d_similarity_transform(src_a, src_b, (kpts[8][0], kpts[8][1]), (kpts[10][0], kpts[10][1]))
                self._warp_and_composite(canvas, synthetic_mask, ref_bgr, m_rarm, M_sim_rarm, (255, 0, 255))
                regions_processed.append("right_arm")

        # (C) Articulated Leg Regions (Similarity Transform)
        if "left_leg" in region_masks and len(kpts) > 13:
            m_lleg = region_masks["left_leg"]
            if np.count_nonzero(m_lleg) > 0:
                ys, xs = np.where(m_lleg > 0)
                src_a = (float(np.min(xs)), float(np.min(ys)))
                src_b = (float(np.max(xs)), float(np.max(ys)))
                M_sim_lleg = compute_2d_similarity_transform(src_a, src_b, (kpts[11][0], kpts[11][1]), (kpts[13][0], kpts[13][1]))
                self._warp_and_composite(canvas, synthetic_mask, ref_bgr, m_lleg, M_sim_lleg, (0, 200, 200))
                regions_processed.append("left_leg")

        if "right_leg" in region_masks and len(kpts) > 16:
            m_rleg = region_masks["right_leg"]
            if np.count_nonzero(m_rleg) > 0:
                ys, xs = np.where(m_rleg > 0)
                src_a = (float(np.min(xs)), float(np.min(ys)))
                src_b = (float(np.max(xs)), float(np.max(ys)))
                M_sim_rleg = compute_2d_similarity_transform(src_a, src_b, (kpts[14][0], kpts[14][1]), (kpts[16][0], kpts[16][1]))
                self._warp_and_composite(canvas, synthetic_mask, ref_bgr, m_rleg, M_sim_rleg, (200, 200, 0))
                regions_processed.append("right_leg")

        # (D) Face & Hair Region (Similarity Transform anchored to Head/Neck)
        if "face" in region_masks and len(kpts) > 4:
            m_face = region_masks["face"]
            if np.count_nonzero(m_face) > 0:
                ys, xs = np.where(m_face > 0)
                src_cx, src_cy = float(np.mean(xs)), float(np.mean(ys))
                tgt_hx, tgt_hy = float(kpts[4][0]), float(kpts[4][1])
                M_sim_face = np.array([[1.0, 0.0, tgt_hx - src_cx], [0.0, 1.0, tgt_hy - src_cy]], dtype=np.float32)
                self._warp_and_composite(canvas, synthetic_mask, ref_bgr, m_face, M_sim_face, (180, 200, 230))
                regions_processed.append("face")

        if "hair" in region_masks and len(kpts) > 4:
            m_hair = region_masks["hair"]
            if np.count_nonzero(m_hair) > 0:
                ys, xs = np.where(m_hair > 0)
                src_cx, src_cy = float(np.mean(xs)), float(np.mean(ys))
                tgt_hx, tgt_hy = float(kpts[4][0]), float(kpts[4][1])
                M_sim_hair = np.array([[1.0, 0.0, tgt_hx - src_cx], [0.0, 1.0, (tgt_hy - 30.0) - src_cy]], dtype=np.float32)
                self._warp_and_composite(canvas, synthetic_mask, ref_bgr, m_hair, M_sim_hair, (20, 20, 120))
                regions_processed.append("hair")

        # Overlay skeleton pose lines on top of character render for diagnostic inspection
        pose_resized = cv2.resize(conditioning.pose_map_2d, (self.output_w, self.output_h))
        non_zero = np.any(pose_resized > 0, axis=2)
        canvas[non_zero] = pose_resized[non_zero]

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return SyntheticFrameResult(
            frame_bgr=canvas,
            synthetic_mask=synthetic_mask,
            latency_ms=latency_ms,
            metadata={
                "backend": self.backend_name,
                "frame_index": conditioning.frame_index,
                "num_reference_views": len(conditioning.reference_views),
                "transformed_regions": regions_processed,
            },
            valid=True,
            warnings=warnings,
        )

    def _warp_and_composite(
        self,
        canvas: np.ndarray,
        synthetic_mask: np.ndarray,
        ref_bgr: Optional[np.ndarray],
        region_mask: np.ndarray,
        M_transform: np.ndarray,
        fallback_color: Tuple[int, int, int],
    ):
        """Warp and composite region mask and pixels onto canvas."""
        warped_mask = cv2.warpAffine(region_mask, M_transform, (self.output_w, self.output_h), flags=cv2.INTER_NEAREST)
        valid_px = warped_mask > 0

        if ref_bgr is not None and ref_bgr.shape == region_mask.shape:
            # Warp reference BGR pixels
            warped_bgr = cv2.warpAffine(ref_bgr, M_transform, (self.output_w, self.output_h), flags=cv2.INTER_LINEAR)
            canvas[valid_px] = warped_bgr[valid_px]
        else:
            # Fallback color fill
            canvas[valid_px] = fallback_color

        synthetic_mask[valid_px] = 255
