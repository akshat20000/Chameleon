"""
Temporal Stabilization Module for CanonicalMotionState.

Implements a three-tier stabilization pipeline:
1. Temporal Association & Hysteresis Policy (detects & rejects L/R swaps)
2. 1€-Filter Position Smoothing (suppresses position jitter while preserving low latency)
3. SO(3) Quaternion SLERP Filtering (smooths joint rotations on the rotation manifold)

Architecture
------------
    CanonicalMotionState(t) [Raw]
                 ↓
      TemporalAssociationFilter  (L/R swap detection & hysteresis hold)
                 ↓
      OneEuroPositionFilter       (Adaptive cutoff frequency per joint)
                 ↓
      SO3QuaternionFilter         (SLERP rotation smoothing)
                 ↓
    CanonicalMotionState(t) [Stabilized]

Decoupled design: raw CanonicalMotionState remains unmodified and observable for benchmarking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from app.motion.canonical_state import (
    BodyPose,
    CanonicalMotionState,
    CONFIDENCE_THRESHOLD,
    FacialExpression,
    FingerState,
    HandPose,
    JointState,
)


# ==============================================================================
# 1. OneEuroFilter (Adaptive Low-Pass Filter for scalar / vector data)
# ==============================================================================

class OneEuroFilter1D:
    """
    1€ Filter for scalar signals (Casiez et al., CHI 2012).

    Adapts cutoff frequency dynamically based on signal rate of change:
    - Low speed  → low cutoff frequency (heavy smoothing, removes jitter)
    - High speed → high cutoff frequency (light smoothing, minimizes lag)
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,     # f_c_min (Hz)
        beta: float = 0.005,         # speed coefficient
        d_cutoff: float = 1.0,       # derivative cutoff (Hz)
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        if dt <= 0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.t_prev = timestamp
            self.dx_prev = 0.0
            return x

        dt = timestamp - self.t_prev
        if dt <= 0:
            return self.x_prev

        # Estimate derivative
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)

        # Filter signal
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class OneEuroFilter3D:
    """Vector 3D wrapper around OneEuroFilter1D."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ):
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fz = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def filter(self, pos: np.ndarray, timestamp: float) -> np.ndarray:
        x = self.fx.filter(float(pos[0]), timestamp)
        y = self.fy.filter(float(pos[1]), timestamp)
        z = self.fz.filter(float(pos[2]), timestamp)
        return np.array([x, y, z], dtype=np.float32)

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


# ==============================================================================
# 2. SO(3) Quaternion SLERP Filter
# ==============================================================================

def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 SO(3) rotation matrix to normalized quaternion [w, x, y, z]."""
    tr = float(np.trace(R))
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float32)
    norm = np.linalg.norm(q)
    return q / max(norm, 1e-9)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert normalized quaternion [w, x, y, z] to 3x3 SO(3) rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical Linear Interpolation between unit quaternions q1 and q2."""
    dot = float(np.dot(q1, q2))

    # Ensure shortest path on S^3
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        res = q1 + t * (q2 - q1)
        return res / np.linalg.norm(res)

    theta_0 = math.acos(np.clip(dot, 0.0, 1.0))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q1 + s1 * q2


class SO3QuaternionFilter:
    """Smooths SO(3) rotation matrices using quaternion SLERP filtering."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha  # Interpolation factor (0 < alpha <= 1)
        self.q_prev: Optional[np.ndarray] = None

    def filter(self, R: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if R is None:
            return None

        q_curr = matrix_to_quaternion(R)
        if self.q_prev is None:
            self.q_prev = q_curr
            return R

        # SLERP step
        q_filtered = slerp(self.q_prev, q_curr, self.alpha)
        self.q_prev = q_filtered
        return quaternion_to_matrix(q_filtered)

    def reset(self):
        self.q_prev = None


# ==============================================================================
# 3. Temporal Association & Kinematic Hysteresis Policy
# ==============================================================================

class TemporalAssociationPolicy:
    """
    Enforces camera-space anatomical consistency and kinematic continuity across frames.
    Detects physical left/right shoulder/hip swaps and holds prior plausible state.
    """

    def __init__(self, max_swap_jump_units: float = 0.15):
        self.max_swap_jump_units = max_swap_jump_units
        self.prev_body: Optional[BodyPose] = None

    def process(self, body: BodyPose) -> Tuple[BodyPose, bool]:
        """
        Evaluate body pose for left/right inversion.
        Returns (corrected_body_pose, swap_detected_flag).
        """
        if self.prev_body is None:
            self.prev_body = body
            return body, False

        ls, rs = body.left_shoulder, body.right_shoulder
        lh, rh = body.left_hip, body.right_hip
        pls, prs = self.prev_body.left_shoulder, self.prev_body.right_shoulder

        swap_detected = False

        # 1. Camera-space anatomical inversion check:
        # Front-facing performer: left_shoulder.x (camera right) MUST be > right_shoulder.x (camera left)
        if ls and rs and ls.is_visible and rs.is_visible:
            if rs.position[0] > ls.position[0] + 0.02:
                swap_detected = True

        # 2. Kinematic distance jump check (did left shoulder jump closer to previous right shoulder?):
        if pls and prs and ls and rs and pls.is_visible and prs.is_visible and ls.is_visible and rs.is_visible:
            dist_l_to_prev_l = np.linalg.norm(ls.position - pls.position)
            dist_l_to_prev_r = np.linalg.norm(ls.position - prs.position)
            if dist_l_to_prev_r < dist_l_to_prev_l - 0.05:
                swap_detected = True

        if swap_detected:
            # Swap detected! Hold previous valid limb assignment with small kinematic velocity prediction
            corrected_joints = {}
            for name, j in body.all_joints().items():
                prev_j = getattr(self.prev_body, name, None)
                if prev_j is not None and prev_j.is_visible:
                    # Hold previous joint state to prevent avatar twitching
                    corrected_joints[name] = prev_j
                else:
                    corrected_joints[name] = j

            new_body = BodyPose(
                pelvis=corrected_joints.get("pelvis", body.pelvis),
                spine_mid=corrected_joints.get("spine_mid", body.spine_mid),
                chest=corrected_joints.get("chest", body.chest),
                neck=corrected_joints.get("neck", body.neck),
                head=corrected_joints.get("head", body.head),
                left_shoulder=corrected_joints.get("left_shoulder", body.left_shoulder),
                left_elbow=corrected_joints.get("left_elbow", body.left_elbow),
                left_wrist=corrected_joints.get("left_wrist", body.left_wrist),
                right_shoulder=corrected_joints.get("right_shoulder", body.right_shoulder),
                right_elbow=corrected_joints.get("right_elbow", body.right_elbow),
                right_wrist=corrected_joints.get("right_wrist", body.right_wrist),
                left_hip=corrected_joints.get("left_hip", body.left_hip),
                left_knee=corrected_joints.get("left_knee", body.left_knee),
                left_ankle=corrected_joints.get("left_ankle", body.left_ankle),
                right_hip=corrected_joints.get("right_hip", body.right_hip),
                right_knee=corrected_joints.get("right_knee", body.right_knee),
                right_ankle=corrected_joints.get("right_ankle", body.right_ankle),
            )
            self.prev_body = new_body
            return new_body, True

        self.prev_body = body
        return body, False


# ==============================================================================
# 4. Main TemporalStabilizer Module
# ==============================================================================

@dataclass
class StabilizationTelemetry:
    """Telemetry report produced per stabilization frame."""
    latency_ms: float = 0.0
    swap_detected: bool = False
    max_position_delta_raw: float = 0.0
    max_position_delta_stabilized: float = 0.0
    max_rotation_delta_raw_deg: float = 0.0
    max_rotation_delta_stabilized_deg: float = 0.0


class TemporalStabilizer:
    """
    Standalone Temporal Stabilizer for CanonicalMotionState.

    Consumes raw CanonicalMotionState(t) and outputs a stabilized copy,
    preserving input state immutability.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,     # 1.2 Hz cutoff at rest (eliminates position jitter)
        beta: float = 0.008,         # speed coefficient (eliminates lag during fast motion)
        rotation_alpha: float = 0.35, # SLERP rotation smoothing factor
    ):
        self.association_policy = TemporalAssociationPolicy()

        # Per-joint 1€ position filters
        self.pos_filters: Dict[str, OneEuroFilter3D] = {}

        # Per-joint quaternion rotation filters
        self.rot_filters: Dict[str, SO3QuaternionFilter] = {}
        self.rotation_alpha = rotation_alpha

        self.min_cutoff = min_cutoff
        self.beta = beta

        self.last_telemetry: Optional[StabilizationTelemetry] = None
        self.prev_raw_state: Optional[CanonicalMotionState] = None
        self.prev_stabilized_state: Optional[CanonicalMotionState] = None

    def _get_pos_filter(self, joint_name: str) -> OneEuroFilter3D:
        if joint_name not in self.pos_filters:
            self.pos_filters[joint_name] = OneEuroFilter3D(
                min_cutoff=self.min_cutoff, beta=self.beta
            )
        return self.pos_filters[joint_name]

    def _get_rot_filter(self, joint_name: str) -> SO3QuaternionFilter:
        if joint_name not in self.rot_filters:
            self.rot_filters[joint_name] = SO3QuaternionFilter(alpha=self.rotation_alpha)
        return self.rot_filters[joint_name]

    def process(self, raw_state: CanonicalMotionState) -> CanonicalMotionState:
        """
        Process a raw CanonicalMotionState frame and return a stabilized copy.
        """
        t0 = time.perf_counter()
        timestamp = raw_state.capture_timestamp

        # 1. Temporal Association & Swap Filter
        body_assoc, swap_detected = self.association_policy.process(raw_state.body)

        # 2. Filter joint positions & rotations
        stabilized_joints: Dict[str, Optional[JointState]] = {}

        for name, j in body_assoc.all_joints().items():
            if j is None or not j.is_visible:
                stabilized_joints[name] = j
                continue

            # Position 1€ filtering
            pf = self._get_pos_filter(name)
            smooth_pos = pf.filter(j.position, timestamp)

            # Rotation SLERP filtering
            rf = self._get_rot_filter(name)
            smooth_rot = rf.filter(j.rotation)

            stabilized_joints[name] = JointState(
                position=smooth_pos,
                rotation=smooth_rot,
                confidence=j.confidence,
            )

        stabilized_body = BodyPose(
            pelvis=stabilized_joints["pelvis"],
            spine_mid=stabilized_joints["spine_mid"],
            chest=stabilized_joints["chest"],
            neck=stabilized_joints["neck"],
            head=stabilized_joints["head"],
            left_shoulder=stabilized_joints["left_shoulder"],
            left_elbow=stabilized_joints["left_elbow"],
            left_wrist=stabilized_joints["left_wrist"],
            right_shoulder=stabilized_joints["right_shoulder"],
            right_elbow=stabilized_joints["right_elbow"],
            right_wrist=stabilized_joints["right_wrist"],
            left_hip=stabilized_joints["left_hip"],
            left_knee=stabilized_joints["left_knee"],
            left_ankle=stabilized_joints["left_ankle"],
            right_hip=stabilized_joints["right_hip"],
            right_knee=stabilized_joints["right_knee"],
            right_ankle=stabilized_joints["right_ankle"],
        )

        stabilized_state = CanonicalMotionState(
            schema_version=raw_state.schema_version,
            frame_index=raw_state.frame_index,
            capture_timestamp=raw_state.capture_timestamp,
            source_backend=raw_state.source_backend + "_stabilized",
            body=stabilized_body,
            face=raw_state.face,
            left_hand=raw_state.left_hand,
            right_hand=raw_state.right_hand,
            body_scale=raw_state.body_scale,
            adapter_timings=dict(raw_state.adapter_timings),
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Telemetry calculation
        self.last_telemetry = StabilizationTelemetry(
            latency_ms=latency_ms,
            swap_detected=swap_detected,
        )

        self.prev_raw_state = raw_state
        self.prev_stabilized_state = stabilized_state

        return stabilized_state

    def reset(self):
        self.association_policy = TemporalAssociationPolicy()
        self.pos_filters.clear()
        self.rot_filters.clear()
        self.prev_raw_state = None
        self.prev_stabilized_state = None
