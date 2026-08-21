"""
motion_retargeter.py — Kinematic Motion Retargeting Engine.

Pipeline (per frame)
--------------------
    StableCanonicalMotionState(t)
             │  (landmark positions only — JointState.rotation NOT used)
             ▼
    AnatomicalFrameBuilder.build_frames()
             │  F_world(j) — complete SO(3) per joint
             ▼
    LocalRotationExtractor.extract()
             │  R_motion_local(j) — performer's deviation from rest pose
             ▼
    KinematicRetargeter._compose()
             │  R_current_local_actor(j) = R_rest_actor(j) @ R_motion_local(j)
             │
             │  Critical invariant: R_rest_actor appears EXACTLY ONCE.
             │  It is NOT re-applied in the FK position step.
             ▼
    KinematicRetargeter._forward_kinematics()
             │  BFS traversal from root:
             │  R_world_actor(j) = R_world_actor(parent) @ R_current_local(j)
             │  P_actor(j) = P_actor(parent) + R_world_actor(parent) @ v_rest(j)
             │    where v_rest(j) = rest_primary_direction[j] * bone_lengths[j]
             ▼
    KinematicRetargeter._apply_constraints()
             │  Per joint: axis-angle decomposition → per-axis clamp → reconstruct
             ▼
    RetargetedActorState

Pose-only (Phase 2.4D)
-----------------------
    P_actor(pelvis) = (0, 0, 0)
    Root translation is NOT transferred.  CanonicalMotionState is pelvis-
    normalized and does not contain global root trajectory.  Root motion
    is deferred to a future milestone that introduces RootTrajectory.

Coordinate convention
---------------------
    Same as CanonicalMotionState: +Y up, +X camera-right, +Z toward camera.
    Actor positions are in body-height-normalized units (actor's body_scale).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.motion.canonical_state import CanonicalMotionState
from app.motion.actor_skeleton import (
    ActorSkeleton, ConstraintViolation,
    JOINT_HIERARCHY, FK_JOINT_ORDER,
)
from app.motion.anatomical_frame_builder import AnatomicalFrameBuilder
from app.motion.local_rotation_extractor import LocalRotationExtractor, LocalJointRotations
from app.motion.retargeted_actor_state import RetargetedActorState

logger = logging.getLogger(__name__)

_EPS = 1e-9
_DEG = math.pi / 180.0


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def _geodesic_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic angle between two SO(3) matrices, in degrees."""
    R = A.T @ B
    trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(trace))


def _axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Decompose 3×3 rotation matrix into (axis, angle_degrees).
    Returns (zero_vector, 0.0) for identity-like matrices.
    """
    R = R.astype(np.float64)
    trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(trace)          # radians
    if abs(angle) < 1e-7:
        return np.array([0., 0., 1.], dtype=np.float32), 0.0
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2.0 * math.sin(angle))
    n = np.linalg.norm(axis)
    if n > _EPS:
        axis = axis / n
    return axis.astype(np.float32), math.degrees(angle)


def _axis_angle_to_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Build 3×3 SO(3) from axis (unit vector) and angle (degrees)."""
    angle = angle_deg * _DEG
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis.astype(np.float64)
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# KinematicRetargeter
# ──────────────────────────────────────────────────────────────────────────────

class KinematicRetargeter:
    """
    Drives a target actor skeleton using a performer's motion deltas.

    Usage
    -----
        retargeter = KinematicRetargeter(actor_skeleton)
        for state in performer_states:
            result = retargeter.retarget(state)
    """

    def __init__(self, actor: ActorSkeleton) -> None:
        self.actor = actor
        self._frame_builder = AnatomicalFrameBuilder()
        self._extractor = LocalRotationExtractor()

    def reset(self) -> None:
        """Clear all state (use between independent video sequences)."""
        self._frame_builder.reset()
        self._extractor.reset()

    def initialize_rest_from_state(self, state: CanonicalMotionState) -> None:
        """
        Explicitly initialize the performer rest pose from a known T-pose frame.
        Optional — the extractor auto-initializes from the first valid frame.
        """
        frames = self._frame_builder.build_frames(state)
        self._extractor.initialize_rest_pose(frames)

    # ── Main entry point ────────────────────────────────────────────────────

    def retarget(self, state: CanonicalMotionState) -> RetargetedActorState:
        """
        Retarget one performer frame to the actor skeleton.

        Parameters
        ----------
        state : CanonicalMotionState (or StableCanonicalMotionState)
            Performer state.  Only joint POSITIONS are consumed;
            JointState.rotation is NOT used (it has twist ambiguity).

        Returns
        -------
        RetargetedActorState
        """
        # Step 1: build anatomical frames from landmark positions
        frames = self._frame_builder.build_frames(state)

        # Step 2: extract motion deltas
        motion_local = self._extractor.extract(frames, state.frame_index)

        # Step 3: compose R_current_local = R_rest_actor @ R_motion_local
        current_local = self._compose(motion_local)

        # Step 4: forward kinematics
        world_rotations, positions = self._forward_kinematics(current_local)

        # Step 5: apply axis-specific anatomical constraints (diagnostic)
        violations: List[ConstraintViolation] = []
        current_local, violations = self._apply_constraints(
            current_local, motion_local, state.frame_index
        )

        # Step 6: re-run FK with clamped rotations (constraints may have changed them)
        if violations:
            world_rotations, positions = self._forward_kinematics(current_local)

        return RetargetedActorState(
            frame_index=state.frame_index,
            capture_timestamp=state.capture_timestamp,
            actor_name=self.actor.name,
            joints=positions,
            world_rotations=world_rotations,
            local_rotations=current_local,
            motion_deltas=motion_local,
            actor_skeleton=self.actor,
            source_frame_index=state.frame_index,
            constraint_violations=violations,
        )

    # ── Step 3: Compose R_current_local ─────────────────────────────────────

    def _compose(self, motion_local: LocalJointRotations) -> LocalJointRotations:
        """
        R_current_local_actor(j) = R_rest_actor(j) @ R_motion_local(j)

        R_rest_actor appears EXACTLY ONCE here.
        It does NOT reappear in the FK position step.
        The FK position step uses v_rest (a fixed vector), not R_rest again.
        """
        current: Dict[str, Optional[np.ndarray]] = {}
        for joint in FK_JOINT_ORDER:
            R_motion = motion_local.joints.get(joint)
            R_rest = self.actor.rest_local_rotations.get(joint, np.eye(3, dtype=np.float32))
            if R_motion is None:
                current[joint] = R_rest.copy()  # no motion data → stay in rest
            else:
                current[joint] = (R_rest @ R_motion).astype(np.float32)
        return LocalJointRotations(joints=current)

    # ── Step 4: Forward Kinematics ───────────────────────────────────────────

    def _forward_kinematics(
        self,
        current_local: LocalJointRotations,
    ) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Optional[np.ndarray]]]:
        """
        BFS traversal from root (pelvis).

        World rotation:
            R_world_actor(pelvis) = R_current_local_actor(pelvis)
            R_world_actor(j) = R_world_actor(parent) @ R_current_local_actor(j)

        Position:
            P_actor(pelvis) = (0, 0, 0)   [pose-only: no root translation]
            P_actor(j) = P_actor(parent) + R_world_actor(parent) @ v_rest(j)

        v_rest(j) = rest_primary_direction[j] * bone_lengths[j]
        This is a FIXED rest-pose offset — R_rest does NOT appear here again.
        """
        world_rotations: Dict[str, Optional[np.ndarray]] = {}
        positions: Dict[str, Optional[np.ndarray]] = {}

        for joint in FK_JOINT_ORDER:
            parent = JOINT_HIERARCHY[joint]
            R_current = current_local.joints.get(joint)

            if parent is None:
                # Root joint
                positions[joint] = np.zeros(3, dtype=np.float32)
                if R_current is not None:
                    world_rotations[joint] = R_current.copy()
                else:
                    world_rotations[joint] = np.eye(3, dtype=np.float32)
                continue

            R_world_parent = world_rotations.get(parent)
            P_parent = positions.get(parent)

            if R_world_parent is None or P_parent is None:
                world_rotations[joint] = None
                positions[joint] = None
                continue

            if R_current is not None:
                R_world = (R_world_parent @ R_current).astype(np.float32)
            else:
                R_world = R_world_parent.copy()
            world_rotations[joint] = R_world

            # Position: parent_pos + R_world_parent @ v_rest(j)
            # Note: R_world_parent (not R_world) — v_rest is in parent-local frame
            v = self.actor.v_rest(joint)
            positions[joint] = (P_parent + R_world_parent @ v).astype(np.float32)

        return world_rotations, positions

    # ── Step 5: Axis-specific constraint enforcement ─────────────────────────

    def _apply_constraints(
        self,
        current_local: LocalJointRotations,
        motion_local: LocalJointRotations,
        frame_index: int,
    ) -> Tuple[LocalJointRotations, List[ConstraintViolation]]:
        """
        Apply axis-specific anatomical constraints to R_motion_local.

        Procedure per joint:
        1. Decompose R_motion_local via axis-angle.
        2. Project axis onto (e_flexion, e_abduction, e_twist) from actor frame.
        3. Clamp each component independently.
        4. Reconstruct R_motion_local_clamped = R_flex @ R_abd @ R_twist.
        5. Recompose R_current_local = R_rest_actor @ R_motion_clamped.
        6. Emit ConstraintViolation if any component clamped > 0.5°.

        This is a diagnostic approximation (axis projection), not full IK.
        """
        violations: List[ConstraintViolation] = []
        new_current = dict(current_local.joints)

        for joint, constraint in self.actor.constraints.items():
            R_motion = motion_local.joints.get(joint)
            if R_motion is None:
                continue

            axis, total_deg = _axis_angle(R_motion)

            # Project axis onto three anatomical directions.
            # We use the world-frame canonical basis as a proxy for joint-local axes:
            #   e_flexion   = primary bone direction (column 0 of rest frame)
            #   e_abduction = lateral direction (column 1)
            #   e_twist     = bone axis (column 0 = same as flexion axis for hinge joints)
            # For Phase 2.4D we use the actor's rest_primary_direction as e_flexion.
            e_primary = self.actor.rest_primary_direction.get(joint, np.array([0., 1., 0.]))
            e_primary = e_primary / (np.linalg.norm(e_primary) + _EPS)

            # Construct three orthogonal axes
            e_flex = e_primary
            # Abduction: perpendicular to primary in the frontal plane
            world_up = np.array([0., 1., 0.], dtype=np.float32)
            e_abd_raw = np.cross(e_flex, world_up)
            abd_norm = np.linalg.norm(e_abd_raw)
            if abd_norm < _EPS:
                e_abd = np.array([1., 0., 0.], dtype=np.float32)
            else:
                e_abd = e_abd_raw / abd_norm
            e_twist = np.cross(e_flex, e_abd)

            flex_raw = float(np.dot(axis, e_flex)) * total_deg
            abd_raw  = float(np.dot(axis, e_abd))  * total_deg
            twist_raw = float(np.dot(axis, e_twist)) * total_deg

            flex_c  = float(np.clip(flex_raw,  constraint.flexion_min_deg,   constraint.flexion_max_deg))
            abd_c   = float(np.clip(abd_raw,   constraint.abduction_min_deg, constraint.abduction_max_deg))
            twist_c = float(np.clip(twist_raw, constraint.twist_min_deg,     constraint.twist_max_deg))

            CLAMP_TOL = 0.5  # degrees

            any_clamped = False
            for ax_name, raw, clamped in [
                ("flexion",   flex_raw,  flex_c),
                ("abduction", abd_raw,   abd_c),
                ("twist",     twist_raw, twist_c),
            ]:
                if abs(raw - clamped) > CLAMP_TOL:
                    violations.append(ConstraintViolation(
                        joint=joint, axis=ax_name,
                        raw_deg=raw, clamped_deg=clamped,
                        frame_index=frame_index,
                    ))
                    any_clamped = True
                    logger.debug(
                        "Constraint: %s %s clamped %.1f° → %.1f° (frame %d)",
                        joint, ax_name, raw, clamped, frame_index,
                    )

            if any_clamped:
                # Reconstruct in Lie algebra so(3): exp(omega_clamped)
                # Multiplying separate matrix exponentials (R_flex @ R_abd @ R_twist) violates BCH formula
                # and introduces non-physical 70° rotation jumps. Exponentiating the clamped vector directly
                # guarantees continuous rotation changes exactly equal to the clamped angular delta.
                v_clamped = flex_c * e_flex + abd_c * e_abd + twist_c * e_twist
                norm_c = float(np.linalg.norm(v_clamped))
                if norm_c < _EPS:
                    R_motion_clamped = np.eye(3, dtype=np.float32)
                else:
                    R_motion_clamped = _axis_angle_to_matrix(v_clamped / norm_c, norm_c)

                R_rest = self.actor.rest_local_rotations.get(joint, np.eye(3, dtype=np.float32))
                new_current[joint] = (R_rest @ R_motion_clamped).astype(np.float32)

        return LocalJointRotations(joints=new_current), violations


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: geodesic angle (exposed for tests)
# ──────────────────────────────────────────────────────────────────────────────

def geodesic_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Public wrapper — geodesic angle between two SO(3) matrices, in degrees."""
    return _geodesic_angle_deg(A, B)
