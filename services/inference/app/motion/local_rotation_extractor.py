"""
local_rotation_extractor.py — Extract performer joint motion deltas.

Sole responsibility
-------------------
Given a dict of anatomical frames F_world(j) (from AnatomicalFrameBuilder),
compute the parent-relative motion delta for each joint:

    R_local_performer(j) = F_world(parent)^T @ F_world(j)

    R_rest_local_performer(j) = F_world_rest(parent)^T @ F_world_rest(j)
        (initialized from the first valid frame set, or synthetic T-pose)

    R_motion_local(j) = R_rest_local_performer(j)^T @ R_local_performer(j)

R_motion_local encodes "how much the performer deviated from their own
neutral pose at this joint" and is the quantity transferred to the actor.

Missing frame policy (temporal hold)
-------------------------------------
    t == 0 and no frame: R_motion_local = I  (remain in rest pose)
    t > 0 and held_frames < MAX_HOLD_FRAMES:
        R_motion_local = R_motion_local_prev  (temporal hold)
        held_count += 1
    held_frames >= MAX_HOLD_FRAMES:
        alpha = (held_count - MAX_HOLD_FRAMES) / DECAY_FRAMES  (0→1)
        R_motion_local = slerp(R_motion_local_prev, I, alpha)
        (actor decays toward rest pose after extended occlusion)

This is architecturally consistent with TemporalStabilizer's policy.

Important distinction
---------------------
This module produces JointMotionDelta (LocalJointRotations) only.
It does NOT produce BoneDirection or AnatomicalFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from app.motion.actor_skeleton import JOINT_HIERARCHY, FK_JOINT_ORDER

logger = logging.getLogger(__name__)

MAX_HOLD_FRAMES: int = 6       # ~200ms at 30fps
DECAY_FRAMES:    int = 15      # frames to decay from held rotation → identity

_EPS = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# LocalJointRotations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalJointRotations:
    """
    Parent-relative SO(3) rotation per joint.

    For R_motion_local:  Identity = joint is in performer's rest pose.
    For R_current_local: Identity = actor is in its own rest pose.

    None = rotation unavailable (joint occluded for too long).
    """
    joints: Dict[str, Optional[np.ndarray]]   # joint_name → (3,3) float32 SO3 or None


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    m = R.astype(np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + _EPS)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
    q = q / (np.linalg.norm(q) + _EPS)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between two quaternions (w,x,y,z). t in [0,1]."""
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 1.0 - 1e-6:
        return (q0 + t * (q1 - q0)) / (np.linalg.norm(q0 + t * (q1 - q0)) + _EPS)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta   = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / (sin_theta_0 + _EPS)
    s1 = sin_theta / (sin_theta_0 + _EPS)
    return s0 * q0 + s1 * q1


_IDENTITY_QUAT = np.array([1., 0., 0., 0.], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# LocalRotationExtractor
# ──────────────────────────────────────────────────────────────────────────────

class LocalRotationExtractor:
    """
    Extracts performer joint motion deltas from anatomical frames.

    Usage
    -----
        extractor = LocalRotationExtractor()
        for frame in video:
            frames = builder.build_frames(state)
            motion_local = extractor.extract(frames, state.frame_index)
    """

    def __init__(self) -> None:
        # Performer rest pose: anatomical frames initialized to Canonical T-Pose reference by default.
        # This guarantees that R_motion_local is computed relative to standard T-pose geometry,
        # preventing arbitrary video Frame 0 from distorting the neutral pose mapping.
        from app.motion.anatomical_frame_builder import build_canonical_tpose_frames
        self._rest_frames: Dict[str, np.ndarray] = build_canonical_tpose_frames()
        self._rest_initialized: bool = True

        # Previous motion deltas for temporal hold policy.
        self._prev_motion_local: Dict[str, np.ndarray] = {}
        self._held_frames: Dict[str, int] = {}

    def reset(self) -> None:
        """Clear all state and re-initialize rest pose to Canonical T-Pose reference."""
        from app.motion.anatomical_frame_builder import build_canonical_tpose_frames
        self._rest_frames = build_canonical_tpose_frames()
        self._rest_initialized = True
        self._prev_motion_local.clear()
        self._held_frames.clear()

    def initialize_rest_pose(self, frames: Dict[str, Optional[np.ndarray]]) -> None:
        """
        Explicitly initialize the performer rest pose from a known T-pose frame.
        Call this before the first extract() call if a T-pose frame is available.
        If not called, rest pose is initialized automatically from the first
        complete set of valid frames.
        """
        self._rest_frames = {
            j: F.copy() for j, F in frames.items() if F is not None
        }
        self._rest_initialized = bool(self._rest_frames)
        logger.info("LocalRotationExtractor: rest pose initialized (%d joints).",
                    len(self._rest_frames))

    def extract(
        self,
        frames: Dict[str, Optional[np.ndarray]],
        frame_index: int,
    ) -> LocalJointRotations:
        """
        Extract parent-relative motion deltas R_motion_local for all 17 joints.

        Computes world-space motion deltas R_world_delta(j) = F_curr(j) @ F_rest(j)^T
        relative to the Canonical T-Pose reference rest pose, then converts to
        parent-relative local deltas R_motion_local(j) = R_world_delta(parent)^T @ R_world_delta(j).
        """
        # 1. Compute world motion deltas relative to canonical T-pose reference
        world_deltas: Dict[str, Optional[np.ndarray]] = {}

        for name, F_curr in frames.items():
            if F_curr is None:
                world_deltas[name] = None
                continue

            F_rest = self._rest_frames.get(name)
            if F_rest is not None:
                # World motion delta: R_world_delta = F_curr @ F_rest^T
                R_w = (F_curr @ F_rest.T).astype(np.float32)
                world_deltas[name] = R_w
            else:
                world_deltas[name] = np.eye(3, dtype=np.float32)

        # 2. Convert world motion deltas to parent-relative local motion deltas
        local_deltas: Dict[str, Optional[np.ndarray]] = {}

        for j in FK_JOINT_ORDER:
            R_w_j = world_deltas.get(j)
            if R_w_j is None:
                local_deltas[j] = self._missing(j, frame_index)
                continue

            parent = JOINT_HIERARCHY[j]
            if parent is None or parent not in world_deltas or world_deltas[parent] is None:
                # Root joint or no parent: local delta == world delta
                R_motion = R_w_j
            else:
                # Parent-relative motion delta: R_motion = R_world_delta(parent)^T @ R_world_delta(j)
                R_w_parent = world_deltas[parent]
                R_motion = (R_w_parent.T @ R_w_j).astype(np.float32)

            local_deltas[j] = self._update(j, R_motion, frame_index)

        return LocalJointRotations(joints=local_deltas)

    # ── Internal: temporal hold and decay ───────────────────────────────────

    def _update(self, joint: str, R: np.ndarray, fi: int) -> np.ndarray:
        """Record a valid motion delta and reset hold counter."""
        self._prev_motion_local[joint] = R.copy()
        self._held_frames[joint] = 0
        return R.astype(np.float32)

    def _missing(self, joint: str, fi: int) -> Optional[np.ndarray]:
        """Apply temporal hold / decay for a missing joint."""
        held = self._held_frames.get(joint, 0)
        prev = self._prev_motion_local.get(joint)

        if prev is None:
            # First frame with no data: identity delta (stay in rest)
            return np.eye(3, dtype=np.float32)

        if held < MAX_HOLD_FRAMES:
            self._held_frames[joint] = held + 1
            logger.debug("LocalRotationExtractor: %s held (frame %d, count=%d)",
                         joint, fi, held + 1)
            return prev.copy()

        # Decay toward identity via SLERP
        alpha = min(1.0, (held - MAX_HOLD_FRAMES) / max(DECAY_FRAMES, 1))
        q_prev = _mat_to_quat(prev)
        q_decayed = _slerp(q_prev, _IDENTITY_QUAT, alpha)
        R_decayed = _quat_to_mat(q_decayed)
        self._prev_motion_local[joint] = R_decayed
        self._held_frames[joint] = held + 1
        return R_decayed
