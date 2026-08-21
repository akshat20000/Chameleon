"""
anatomical_frame_builder.py — Build complete SO(3) anatomical frames per joint.

Problem addressed
-----------------
_bone_rotation() in mediapipe_adapter.py aligns one vector to another with
Rodrigues rotation.  That constrains only one DOF (bone direction) and leaves
one DOF unconstrained: twist around the bone axis.  Using R_parent^T @ R_child
from such rotations therefore does NOT produce a physically meaningful local
joint rotation.

Solution
--------
For each joint, build a complete orthonormal SO(3) frame using TWO
anatomically-meaningful directions derived from neighboring landmarks:

    e1 = normalize(primary_direction)          # bone axis
    e3 = normalize(cross(e1, secondary))       # perpendicular to plane of motion
    e2 = cross(e3, e1)                         # completes right-hand basis

    F_world(j) = column_stack(e1, e2, e3)     # (3,3) SO(3), no twist ambiguity

Degenerate case policy (fully extended limbs)
---------------------------------------------
When norm(cross(primary, secondary)) < DEGENERATE_THRESHOLD:
    Priority 1: hold previous_valid_frame for this joint (temporal hold).
    Priority 2: use canonical fallback secondary (only when no previous frame).
    Never: silently switch to canonical fallback without emitting a diagnostic.

This prevents the entry/exit discontinuity that would otherwise occur when a
limb passes through the degenerate region.

Important distinctions
----------------------
This module produces AnatomicalFrame objects only.
It does NOT compute BoneDirection (used for rendering) or JointMotionDelta
(produced by LocalRotationExtractor from these frames).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.motion.canonical_state import CanonicalMotionState

logger = logging.getLogger(__name__)

# Cross-product norm below this value → degenerate (arm nearly straight).
DEGENERATE_THRESHOLD: float = 1e-4

# Maximum frames to hold a previous valid frame before decaying to canonical.
MAX_HOLD_FRAMES: int = 10


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic event
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameDegenerateEvent:
    """Emitted (as a warning) when a joint frame enters the degenerate region."""
    joint: str
    frame_index: int
    reason: str          # 'held' | 'canonical_fallback'
    cross_norm: float    # actual norm(cross(primary, secondary))


# ──────────────────────────────────────────────────────────────────────────────
# Internal math helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def _build_frame(e1: np.ndarray, secondary: np.ndarray) -> Optional[np.ndarray]:
    """
    Build a right-handed SO(3) frame from primary axis e1 and a secondary
    (not necessarily orthogonal) reference vector.

    e1 → column 0  (primary bone axis)
    e3 = normalize(cross(e1, secondary)) → column 2
    e2 = cross(e3, e1) → column 1

    Returns None if cross-product magnitude < DEGENERATE_THRESHOLD.
    """
    e1 = _normalize(e1)
    cross = np.cross(e1, secondary)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < DEGENERATE_THRESHOLD:
        return None
    e3 = cross / cross_norm
    e2 = np.cross(e3, e1)
    F = np.column_stack([e1, e2, e3]).astype(np.float32)
    return F


def _joint_pos(state: CanonicalMotionState, name: str) -> Optional[np.ndarray]:
    """Return joint position or None if joint is missing / not visible."""
    joint = getattr(state.body, name, None)
    if joint is None or not joint.is_visible:
        return None
    return joint.position.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Canonical fallback secondary axes
# ──────────────────────────────────────────────────────────────────────────────

# Used only as last resort when no previous_valid_frame exists for a joint.
_CANONICAL_SECONDARY: Dict[str, np.ndarray] = {
    # Arms: secondary = frontal plane normal (+Z, toward camera)
    "left_shoulder":  np.array([0., 0., 1.], dtype=np.float32),
    "left_elbow":     np.array([0., 0., 1.], dtype=np.float32),
    "right_shoulder": np.array([0., 0., 1.], dtype=np.float32),
    "right_elbow":    np.array([0., 0., 1.], dtype=np.float32),
    # Legs: secondary = sagittal plane normal (-X, camera left)
    "left_hip":       np.array([-1., 0., 0.], dtype=np.float32),
    "left_knee":      np.array([-1., 0., 0.], dtype=np.float32),
    "right_hip":      np.array([-1., 0., 0.], dtype=np.float32),
    "right_knee":     np.array([-1., 0., 0.], dtype=np.float32),
    # Spine/head: secondary = lateral axis (+X)
    "pelvis":         np.array([1., 0., 0.], dtype=np.float32),
    "spine_mid":      np.array([1., 0., 0.], dtype=np.float32),
    "chest":          np.array([1., 0., 0.], dtype=np.float32),
    "neck":           np.array([1., 0., 0.], dtype=np.float32),
    "head":           np.array([1., 0., 0.], dtype=np.float32),
}


def build_canonical_tpose_frames() -> Dict[str, np.ndarray]:
    """
    Construct complete SO(3) anatomical frames for a canonical T-pose reference.

    These serve as the default performer rest-pose reference frames, ensuring that
    R_motion_local is computed relative to a standard T-pose rather than an
    arbitrary video frame.
    """
    from app.motion.actor_skeleton import REST_PRIMARY_DIRECTION

    frames = {}
    for joint, primary in REST_PRIMARY_DIRECTION.items():
        sec = _CANONICAL_SECONDARY.get(joint, np.array([0., 0., 1.], dtype=np.float32))
        F = _build_frame(primary, sec)
        if F is not None:
            frames[joint] = F
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# AnatomicalFrameBuilder
# ──────────────────────────────────────────────────────────────────────────────

class AnatomicalFrameBuilder:
    """
    Builds a complete SO(3) anatomical frame per joint from CanonicalMotionState
    landmark positions.

    Each frame uses two anatomically-meaningful directions, eliminating the
    bone-axis twist ambiguity of single-vector Rodrigues alignment.

    State
    -----
    previous_valid_frame : Dict[str, np.ndarray]
        Last successfully built (non-degenerate) frame per joint.
        Updated only when the cross-product norm >= DEGENERATE_THRESHOLD.
    held_frames : Dict[str, int]
        Number of consecutive frames this joint has been held.

    Thread safety: not thread-safe (one instance per pipeline).
    """

    def __init__(self) -> None:
        self.previous_valid_frame: Dict[str, np.ndarray] = {}
        self.held_frames: Dict[str, int] = {}
        self.degenerate_events: List[FrameDegenerateEvent] = []

    def reset(self) -> None:
        """Clear all state (use between independent video sequences)."""
        self.previous_valid_frame.clear()
        self.held_frames.clear()
        self.degenerate_events.clear()

    # ── Public API ──────────────────────────────────────────────────────────

    def build_frames(
        self,
        state: CanonicalMotionState,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Build SO(3) anatomical frames for all joints from the given state.

        Returns
        -------
        Dict[str, Optional[np.ndarray]]
            joint_name → (3,3) float32 SO(3) frame, or None if the joint
            position is unavailable AND no previous frame is held.
        """
        frames: Dict[str, Optional[np.ndarray]] = {}
        frame_index = state.frame_index

        frames["pelvis"]    = self._torso_pelvis(state, frame_index)
        frames["spine_mid"] = self._torso_spine_mid(state, frame_index)
        frames["chest"]     = self._torso_chest(state, frame_index)
        frames["neck"]      = self._torso_neck(state, frame_index)
        frames["head"]      = self._torso_head(state, frame_index)

        # Arms — share the arm-plane normal as secondary axis.
        # is_degenerate=True when the arm is nearly straight (upper_arm × forearm < DEGENERATE_THRESHOLD).
        l_arm_plane, l_arm_degen = self._arm_plane_normal(state, "left")
        r_arm_plane, r_arm_degen = self._arm_plane_normal(state, "right")
        frames["left_shoulder"]  = self._limb_frame(state, frame_index,
            "left_shoulder", "left_elbow", l_arm_plane, l_arm_degen)
        frames["left_elbow"]     = self._limb_frame(state, frame_index,
            "left_elbow",    "left_wrist", l_arm_plane, l_arm_degen)
        frames["right_shoulder"] = self._limb_frame(state, frame_index,
            "right_shoulder", "right_elbow", r_arm_plane, r_arm_degen)
        frames["right_elbow"]    = self._limb_frame(state, frame_index,
            "right_elbow",   "right_wrist", r_arm_plane, r_arm_degen)

        # Legs — share the leg-plane normal as secondary axis.
        l_leg_plane, l_leg_degen = self._leg_plane_normal(state, "left")
        r_leg_plane, r_leg_degen = self._leg_plane_normal(state, "right")
        frames["left_hip"]   = self._limb_frame(state, frame_index,
            "left_hip",  "left_knee",   l_leg_plane, l_leg_degen)
        frames["left_knee"]  = self._limb_frame(state, frame_index,
            "left_knee", "left_ankle",  l_leg_plane, l_leg_degen)
        frames["right_hip"]  = self._limb_frame(state, frame_index,
            "right_hip",  "right_knee",  r_leg_plane, r_leg_degen)
        frames["right_knee"] = self._limb_frame(state, frame_index,
            "right_knee", "right_ankle", r_leg_plane, r_leg_degen)

        # Leaf joints — no frame needed for retargeting
        for leaf in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
            frames[leaf] = None

        return frames

    # ── Internal: torso frames ───────────────────────────────────────────────

    def _torso_pelvis(self, state: CanonicalMotionState, fi: int) -> Optional[np.ndarray]:
        chest = _joint_pos(state, "chest")
        pelvis = _joint_pos(state, "pelvis")
        left_hip = _joint_pos(state, "left_hip")
        right_hip = _joint_pos(state, "right_hip")
        if pelvis is None:
            return self._hold_or_fallback("pelvis", fi, 0.0)
        primary = _normalize((chest - pelvis) if chest is not None else np.array([0., 1., 0.]))
        secondary = _normalize((left_hip - right_hip) if (left_hip is not None and right_hip is not None) else _CANONICAL_SECONDARY["pelvis"])
        return self._resolve("pelvis", fi, primary, secondary)

    def _torso_spine_mid(self, state: CanonicalMotionState, fi: int) -> Optional[np.ndarray]:
        spine_mid = _joint_pos(state, "spine_mid")
        chest = _joint_pos(state, "chest")
        pelvis = _joint_pos(state, "pelvis")
        ls = _joint_pos(state, "left_shoulder")
        rs = _joint_pos(state, "right_shoulder")
        if spine_mid is None or chest is None:
            return self._hold_or_fallback("spine_mid", fi, 0.0)
        primary = _normalize(chest - spine_mid)
        secondary = _normalize((ls - rs) if (ls is not None and rs is not None) else _CANONICAL_SECONDARY["spine_mid"])
        return self._resolve("spine_mid", fi, primary, secondary)

    def _torso_chest(self, state: CanonicalMotionState, fi: int) -> Optional[np.ndarray]:
        chest = _joint_pos(state, "chest")
        neck = _joint_pos(state, "neck")
        ls = _joint_pos(state, "left_shoulder")
        rs = _joint_pos(state, "right_shoulder")
        if chest is None or neck is None:
            return self._hold_or_fallback("chest", fi, 0.0)
        primary = _normalize(neck - chest)
        secondary = _normalize((ls - rs) if (ls is not None and rs is not None) else _CANONICAL_SECONDARY["chest"])
        return self._resolve("chest", fi, primary, secondary)

    def _torso_neck(self, state: CanonicalMotionState, fi: int) -> Optional[np.ndarray]:
        neck = _joint_pos(state, "neck")
        head = _joint_pos(state, "head")
        ls = _joint_pos(state, "left_shoulder")
        rs = _joint_pos(state, "right_shoulder")
        if neck is None or head is None:
            return self._hold_or_fallback("neck", fi, 0.0)
        primary = _normalize(head - neck)
        secondary = _normalize((ls - rs) if (ls is not None and rs is not None) else _CANONICAL_SECONDARY["neck"])
        return self._resolve("neck", fi, primary, secondary)

    def _torso_head(self, state: CanonicalMotionState, fi: int) -> Optional[np.ndarray]:
        neck = _joint_pos(state, "neck")
        head = _joint_pos(state, "head")
        ls = _joint_pos(state, "left_shoulder")
        rs = _joint_pos(state, "right_shoulder")
        if neck is None or head is None:
            return self._hold_or_fallback("head", fi, 0.0)
        primary = _normalize(head - neck)
        secondary = _normalize((ls - rs) if (ls is not None and rs is not None) else _CANONICAL_SECONDARY["head"])
        return self._resolve("head", fi, primary, secondary)

    # ── Internal: limb plane normals ─────────────────────────────────────────

    def _arm_plane_normal(
        self, state: CanonicalMotionState, side: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute arm plane normal from upper_arm × forearm.
        Returns (plane_normal, is_degenerate).
        is_degenerate=True when cross-product magnitude < DEGENERATE_THRESHOLD.
        Falls back to canonical +Z (frontal plane) and sets is_degenerate=True.
        """
        shoulder = _joint_pos(state, f"{side}_shoulder")
        elbow    = _joint_pos(state, f"{side}_elbow")
        wrist    = _joint_pos(state, f"{side}_wrist")
        if shoulder is None or elbow is None or wrist is None:
            return _CANONICAL_SECONDARY[f"{side}_shoulder"], True
        upper_arm = _normalize(elbow - shoulder)
        forearm   = _normalize(wrist  - elbow)
        cross = np.cross(upper_arm, forearm)
        cn = float(np.linalg.norm(cross))
        if cn < DEGENERATE_THRESHOLD:
            return _CANONICAL_SECONDARY[f"{side}_shoulder"], True   # degenerate
        return cross / cn, False

    def _leg_plane_normal(
        self, state: CanonicalMotionState, side: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute leg plane normal from thigh × shin.
        Returns (plane_normal, is_degenerate).
        is_degenerate=True when cross-product magnitude < DEGENERATE_THRESHOLD.
        Falls back to canonical +X (sagittal plane) and sets is_degenerate=True.
        """
        hip   = _joint_pos(state, f"{side}_hip")
        knee  = _joint_pos(state, f"{side}_knee")
        ankle = _joint_pos(state, f"{side}_ankle")
        if hip is None or knee is None or ankle is None:
            return _CANONICAL_SECONDARY[f"{side}_hip"], True
        thigh = _normalize(knee  - hip)
        shin  = _normalize(ankle - knee)
        cross = np.cross(thigh, shin)
        cn = float(np.linalg.norm(cross))
        if cn < DEGENERATE_THRESHOLD:
            return _CANONICAL_SECONDARY[f"{side}_hip"], True    # degenerate
        return cross / cn, False

    # ── Internal: generic limb joint frame ──────────────────────────────────

    def _limb_frame(
        self,
        state: CanonicalMotionState,
        fi: int,
        proximal_name: str,
        distal_name: str,
        plane_normal: np.ndarray,
        plane_is_degenerate: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Build frame for a limb joint from (proximal → distal) as primary axis
        and the pre-computed plane normal as secondary.

        If plane_is_degenerate=True, the underlying limb plane cross-product
        was below DEGENERATE_THRESHOLD, meaning this joint's anatomical frame
        is unreliable.  Force the temporal hold path rather than building a
        fresh frame from a canonical artificial secondary.
        """
        proximal = _joint_pos(state, proximal_name)
        distal   = _joint_pos(state, distal_name)
        if proximal is None or distal is None:
            return self._hold_or_fallback(proximal_name, fi, 0.0)
        if plane_is_degenerate:
            # Arm/leg is nearly straight: the frame cannot be reliably determined.
            # Apply temporal hold (not canonical fallback) to avoid discontinuity.
            return self._hold_or_fallback(proximal_name, fi, 0.0)
        primary = _normalize(distal - proximal)
        return self._resolve(proximal_name, fi, primary, plane_normal)

    # ── Internal: frame resolution with degenerate policy ───────────────────

    def _resolve(
        self,
        joint: str,
        fi: int,
        primary: np.ndarray,
        secondary: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Attempt to build a frame from primary+secondary.
        On degenerate cross-product: temporal hold → canonical fallback.
        """
        cross = np.cross(primary, secondary)
        cross_norm = float(np.linalg.norm(cross))

        if cross_norm >= DEGENERATE_THRESHOLD:
            # Success: build frame and update state
            e3 = cross / cross_norm
            e2 = np.cross(e3, primary)
            F = np.column_stack([primary, e2, e3]).astype(np.float32)
            self.previous_valid_frame[joint] = F
            self.held_frames[joint] = 0
            return F

        return self._hold_or_fallback(joint, fi, cross_norm)

    def _hold_or_fallback(
        self,
        joint: str,
        fi: int,
        cross_norm: float,
    ) -> Optional[np.ndarray]:
        """
        Degenerate / missing joint: apply temporal hold first, then fallback.
        """
        if joint in self.previous_valid_frame:
            # Priority 1: temporal hold of last valid frame
            count = self.held_frames.get(joint, 0) + 1
            self.held_frames[joint] = count
            evt = FrameDegenerateEvent(joint=joint, frame_index=fi,
                                       reason="held", cross_norm=cross_norm)
            self.degenerate_events.append(evt)
            logger.debug("AnatomicalFrameBuilder: %s frame held (frame %d, cross=%.2e)",
                         joint, fi, cross_norm)
            return self.previous_valid_frame[joint]

        # Priority 2: canonical fallback (last resort — no previous frame)
        canonical_sec = _CANONICAL_SECONDARY.get(joint)
        if canonical_sec is not None:
            # Use a synthetic primary from the canonical rest direction
            from app.motion.actor_skeleton import REST_PRIMARY_DIRECTION
            rest_dir = REST_PRIMARY_DIRECTION.get(joint, np.array([0., 1., 0.], dtype=np.float32))
            F = _build_frame(rest_dir, canonical_sec)
            if F is not None:
                evt = FrameDegenerateEvent(joint=joint, frame_index=fi,
                                           reason="canonical_fallback", cross_norm=cross_norm)
                self.degenerate_events.append(evt)
                logger.warning(
                    "AnatomicalFrameBuilder: %s using CANONICAL FALLBACK (frame %d, "
                    "cross=%.2e) — no previous valid frame available.",
                    joint, fi, cross_norm,
                )
                self.previous_valid_frame[joint] = F
                return F

        logger.warning("AnatomicalFrameBuilder: %s — cannot build frame (frame %d).", joint, fi)
        return None
