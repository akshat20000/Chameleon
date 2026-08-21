"""
actor_skeleton.py — Actor skeleton definition for kinematic retargeting.

Defines:
  - JointConstraint     : axis-specific anatomical plausibility bounds on motion delta
  - ConstraintViolation : per-frame diagnostic record emitted when a bound is exceeded
  - ActorSkeleton       : rest pose, hierarchy, bone lengths, constraints
  - Predefined profiles : DEFAULT, TALL, PETITE, LONG_ARMS, SHORT_ARMS

Design contract
---------------
All bone lengths and rest positions are in body-height-normalized units
(1.0 = full standing height of this actor), matching the convention used
by CanonicalMotionState.

The hierarchy dict maps each joint name to its parent joint name.
The root joint (pelvis) maps to None.

Rest-local rotations encode the T-pose orientation of each joint relative
to its parent frame.  In a canonical T-pose (arms extended horizontally,
palms down, facing +Z), all rest-local rotations are identity (no
pre-existing tilt or twist).

This module has NO dependency on any tracker, CanonicalMotionState,
AnatomicalFrameBuilder, or retargeter module.  It is purely structural data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# JointConstraint
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JointConstraint:
    """
    Anatomical plausibility limits on the three axes of a joint's motion delta.

    All angles in degrees, signed:
        positive flexion   = anatomically forward (knee bends, arm raises forward)
        positive abduction = lateral (arm moves away from midline)
        positive twist     = external rotation

    Constraint enforcement procedure (in motion_retargeter.py):
        1. Decompose R_motion_local via axis-angle.
        2. Project onto (e_flexion, e_abduction, e_twist) axes.
        3. Clamp each component independently.
        4. Reconstruct R_motion_local_clamped = R_flex @ R_abd @ R_twist.
        5. Emit ConstraintViolation if any component was clamped > 0.5°.

    Phase 2.4D: these are DIAGNOSTIC bounds, not full biomechanical IK.
    """
    flexion_min_deg: float
    flexion_max_deg: float
    abduction_min_deg: float
    abduction_max_deg: float
    twist_min_deg: float
    twist_max_deg: float


# ──────────────────────────────────────────────────────────────────────────────
# ConstraintViolation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstraintViolation:
    """Diagnostic record emitted when a joint exceeds an anatomical constraint."""
    joint: str
    axis: str           # 'flexion' | 'abduction' | 'twist'
    raw_deg: float
    clamped_deg: float
    frame_index: int


# ──────────────────────────────────────────────────────────────────────────────
# Canonical hierarchy and traversal order (shared by all actors)
# ──────────────────────────────────────────────────────────────────────────────

# parent → child hierarchy.  pelvis is root (parent=None).
JOINT_HIERARCHY: Dict[str, Optional[str]] = {
    "pelvis":           None,
    "spine_mid":        "pelvis",
    "chest":            "spine_mid",
    "neck":             "chest",
    "head":             "neck",
    "left_hip":         "pelvis",
    "left_knee":        "left_hip",
    "left_ankle":       "left_knee",
    "right_hip":        "pelvis",
    "right_knee":       "right_hip",
    "right_ankle":      "right_knee",
    "left_shoulder":    "chest",
    "left_elbow":       "left_shoulder",
    "left_wrist":       "left_elbow",
    "right_shoulder":   "chest",
    "right_elbow":      "right_shoulder",
    "right_wrist":      "right_elbow",
}

# BFS traversal order (root first) for deterministic FK computation.
FK_JOINT_ORDER: List[str] = [
    "pelvis",
    "spine_mid", "left_hip", "right_hip",
    "chest", "left_knee", "right_knee",
    "neck", "left_ankle", "right_ankle",
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

# Rest-pose primary axis (in parent-local frame) pointing from parent toward child.
# This defines the direction of the bone in the canonical T-pose.
# Used in FK: v_rest(j) = REST_PRIMARY_DIRECTION[j] * bone_lengths[j]
# Then: P_child = P_parent + R_world(parent) @ v_rest(j)
REST_PRIMARY_DIRECTION: Dict[str, np.ndarray] = {
    "pelvis":           np.array([0.,  0., 0.], dtype=np.float32),   # root: no offset
    "spine_mid":        np.array([0.,  1., 0.], dtype=np.float32),   # upward
    "chest":            np.array([0.,  1., 0.], dtype=np.float32),
    "neck":             np.array([0.,  1., 0.], dtype=np.float32),
    "head":             np.array([0.,  1., 0.], dtype=np.float32),
    "left_hip":         np.array([-1., 0., 0.], dtype=np.float32),   # camera-right is performer-left
    "left_knee":        np.array([0., -1., 0.], dtype=np.float32),   # downward
    "left_ankle":       np.array([0., -1., 0.], dtype=np.float32),
    "right_hip":        np.array([1.,  0., 0.], dtype=np.float32),   # camera-left is performer-right
    "right_knee":       np.array([0., -1., 0.], dtype=np.float32),
    "right_ankle":      np.array([0., -1., 0.], dtype=np.float32),
    "left_shoulder":    np.array([1.,  0., 0.], dtype=np.float32),   # T-pose: arm extends camera-right
    "left_elbow":       np.array([1.,  0., 0.], dtype=np.float32),
    "left_wrist":       np.array([1.,  0., 0.], dtype=np.float32),
    "right_shoulder":   np.array([-1., 0., 0.], dtype=np.float32),   # T-pose: arm extends camera-left
    "right_elbow":      np.array([-1., 0., 0.], dtype=np.float32),
    "right_wrist":      np.array([-1., 0., 0.], dtype=np.float32),
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _identity_local_rotations() -> Dict[str, np.ndarray]:
    """Return identity (3×3) local rotations for all joints (canonical T-pose)."""
    return {j: np.eye(3, dtype=np.float32) for j in JOINT_HIERARCHY}


def _default_constraints() -> Dict[str, JointConstraint]:
    """
    Phase 2.4D anatomical constraint bounds.
    Loose — diagnostic priority, not restrictive simulation.
    """
    return {
        "left_shoulder":  JointConstraint(-90, 180, -90,  90, -90,  90),
        "right_shoulder": JointConstraint(-90, 180, -90,  90, -90,  90),
        "left_elbow":     JointConstraint(-10, 145, -20,  20, -30,  30),
        "right_elbow":    JointConstraint(-10, 145, -20,  20, -30,  30),
        "left_hip":       JointConstraint(-30, 120, -45,  45, -45,  45),
        "right_hip":      JointConstraint(-30, 120, -45,  45, -45,  45),
        "left_knee":      JointConstraint( -5, 145, -15,  15, -20,  20),
        "right_knee":     JointConstraint( -5, 145, -15,  15, -20,  20),
        "spine_mid":      JointConstraint(-45,  45, -30,  30, -60,  60),
        "chest":          JointConstraint(-45,  45, -30,  30, -60,  60),
        "neck":           JointConstraint(-60,  60, -45,  45, -45,  45),
    }


def _compute_rest_positions(
    bone_lengths: Dict[str, float],
    rest_local_rotations: Dict[str, np.ndarray],
    rest_primary_direction: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute world-space rest positions for all joints via FK on the T-pose.
    pelvis is at origin.  Uses the same FK equation as KinematicRetargeter.
    """
    positions: Dict[str, np.ndarray] = {}
    world_rotations: Dict[str, np.ndarray] = {}

    for joint in FK_JOINT_ORDER:
        parent = JOINT_HIERARCHY[joint]
        R_rest_local = rest_local_rotations.get(joint, np.eye(3, dtype=np.float32))

        if parent is None:
            positions[joint] = np.zeros(3, dtype=np.float32)
            world_rotations[joint] = R_rest_local.copy()
        else:
            R_world_parent = world_rotations[parent]
            world_rotations[joint] = R_world_parent @ R_rest_local

            v = rest_primary_direction.get(joint, np.zeros(3, dtype=np.float32))
            length = bone_lengths.get(joint, 0.0)
            positions[joint] = positions[parent] + R_world_parent @ (v * length)

    return positions


# ──────────────────────────────────────────────────────────────────────────────
# ActorSkeleton
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActorSkeleton:
    """
    Complete description of a target actor skeleton for kinematic retargeting.

    Fields
    ------
    name : str
    body_scale : float
        Actor's standing height as a fraction of the canonical reference
        (1.0 = standard proportions).  Used to normalize root translation.
    bone_lengths : Dict[str, float]
        Length of each bone in normalized body-height units.
        Key = child joint name (distal end of the bone).
    rest_local_rotations : Dict[str, np.ndarray]
        Per-joint rest-pose local rotation (3×3 SO3) relative to parent frame.
        Identity = T-pose canonical orientation.
    rest_positions : Dict[str, np.ndarray]
        World-space T-pose positions relative to pelvis=(0,0,0).
        Derived from bone_lengths via FK.  Used by Gate A / Gate C validation.
    hierarchy : Dict[str, Optional[str]]
        joint → parent joint.  pelvis → None.
    rest_primary_direction : Dict[str, np.ndarray]
        Per-joint rest-pose bone direction in parent-local frame.
        Defines v_rest used in FK position step.
    constraints : Dict[str, JointConstraint]
        Anatomical plausibility bounds per joint.
    """
    name: str
    body_scale: float
    bone_lengths: Dict[str, float]
    rest_local_rotations: Dict[str, np.ndarray]
    rest_positions: Dict[str, np.ndarray]
    hierarchy: Dict[str, Optional[str]]
    rest_primary_direction: Dict[str, np.ndarray]
    constraints: Dict[str, JointConstraint]

    def v_rest(self, joint: str) -> np.ndarray:
        """
        Rest-pose bone offset vector in parent-local frame.

            v_rest(j) = rest_primary_direction[j] * bone_lengths[j]

        Used in FK position step:
            P_actor(j) = P_actor(parent) + R_world_actor(parent) @ v_rest(j)

        For the root joint (pelvis), returns zero vector.
        """
        length = self.bone_lengths.get(joint, 0.0)
        return self.rest_primary_direction[joint] * length

    def fk_order(self) -> List[str]:
        """Return joints in BFS order (root first) for FK traversal."""
        return FK_JOINT_ORDER


# ──────────────────────────────────────────────────────────────────────────────
# Profile factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_actor(
    name: str,
    body_scale: float,
    arm_scale: float = 1.0,
) -> ActorSkeleton:
    """
    Build a synthetic ActorSkeleton with proportions derived from body_scale
    and arm_scale multipliers applied to the default reference lengths.

    Reference proportions (body_scale=1.0, arm_scale=1.0):
        spine segments : 0.12–0.14
        upper arm      : 0.16  forearm: 0.14
        thigh          : 0.25  shin:    0.24
        hip offset     : 0.09  shoulder offset: 0.09
    """
    s = body_scale
    a = arm_scale

    bone_lengths: Dict[str, float] = {
        "spine_mid":       0.12 * s,
        "chest":           0.14 * s,
        "neck":            0.06 * s,
        "head":            0.12 * s,
        "left_hip":        0.09 * s,
        "right_hip":       0.09 * s,
        "left_knee":       0.25 * s,
        "left_ankle":      0.24 * s,
        "right_knee":      0.25 * s,
        "right_ankle":     0.24 * s,
        "left_shoulder":   0.09 * s,
        "left_elbow":      0.16 * s * a,
        "left_wrist":      0.14 * s * a,
        "right_shoulder":  0.09 * s,
        "right_elbow":     0.16 * s * a,
        "right_wrist":     0.14 * s * a,
    }

    rest_local_rotations = _identity_local_rotations()
    rest_primary_direction = {k: v.copy() for k, v in REST_PRIMARY_DIRECTION.items()}
    rest_positions = _compute_rest_positions(
        bone_lengths, rest_local_rotations, rest_primary_direction
    )

    return ActorSkeleton(
        name=name,
        body_scale=body_scale,
        bone_lengths=bone_lengths,
        rest_local_rotations=rest_local_rotations,
        rest_positions=rest_positions,
        hierarchy=dict(JOINT_HIERARCHY),
        rest_primary_direction=rest_primary_direction,
        constraints=_default_constraints(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public predefined profiles
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ACTOR    = _make_actor("DEFAULT",    body_scale=1.00)
TALL_ACTOR       = _make_actor("TALL",       body_scale=1.12)
PETITE_ACTOR     = _make_actor("PETITE",     body_scale=0.88)
LONG_ARMS_ACTOR  = _make_actor("LONG_ARMS",  body_scale=1.00, arm_scale=1.15)
SHORT_ARMS_ACTOR = _make_actor("SHORT_ARMS", body_scale=1.00, arm_scale=0.85)

ACTOR_PROFILES: Dict[str, ActorSkeleton] = {
    "DEFAULT":    DEFAULT_ACTOR,
    "TALL":       TALL_ACTOR,
    "PETITE":     PETITE_ACTOR,
    "LONG_ARMS":  LONG_ARMS_ACTOR,
    "SHORT_ARMS": SHORT_ARMS_ACTOR,
}
