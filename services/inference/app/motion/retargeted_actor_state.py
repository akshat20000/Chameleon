"""
retargeted_actor_state.py — RetargetedActorState type definition.

This type is explicitly NOT a CanonicalMotionState.

CanonicalMotionState: performer geometry (pelvis-normalized, tracker units).
RetargetedActorState: actor geometry (actor proportions, actor skeleton).

Mixing these types is an architectural error.  The retargeter receives
CanonicalMotionState as input and produces RetargetedActorState as output.
No downstream consumer should treat these as interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.motion.local_rotation_extractor import LocalJointRotations
from app.motion.actor_skeleton import ActorSkeleton, ConstraintViolation


@dataclass
class RetargetedActorState:
    """
    Joint positions and orientations for a target actor after kinematic
    retargeting from a performer's StableCanonicalMotionState.

    This is NOT a CanonicalMotionState — it contains actor-proportioned
    geometry, not performer geometry.

    Fields
    ------
    frame_index : int
        Actor frame index (matches source_frame_index for deterministic replay).
    capture_timestamp : float
        Timestamp from the source performer frame.
    actor_name : str
        Identifier of the actor profile used.
    joints : Dict[str, Optional[np.ndarray]]
        World-space joint positions, shape (3,) float32.
        Origin = actor pelvis = (0, 0, 0) (pose-only, Phase 2.4D).
        None if the joint was not computed (missing input for too long).
    world_rotations : Dict[str, Optional[np.ndarray]]
        World-space SO(3) orientation per joint, shape (3,3) float32.
        R_world_actor(j) = R_world_actor(parent) @ R_current_local_actor(j)
    local_rotations : LocalJointRotations
        R_current_local_actor per joint = R_rest_actor @ R_motion_local.
        These are the rotations actually applied during FK.
    motion_deltas : LocalJointRotations
        R_motion_local per joint (performer's deviation from rest pose).
        Used for Gate G3 (local motion preservation) and debugging.
    actor_skeleton : ActorSkeleton
        The actor skeleton used for this retargeting result.
    source_frame_index : int
        Frame index of the performer CanonicalMotionState that was retargeted.
    constraint_violations : List[ConstraintViolation]
        Per-frame diagnostic records for joints that exceeded anatomical bounds.
        Empty list = no violations this frame.
    """
    frame_index: int
    capture_timestamp: float
    actor_name: str

    # Geometry (actor-proportioned)
    joints: Dict[str, Optional[np.ndarray]]           # world positions (3,) per joint
    world_rotations: Dict[str, Optional[np.ndarray]]  # world SO(3) (3,3) per joint
    local_rotations: LocalJointRotations               # R_current_local per joint
    motion_deltas: LocalJointRotations                 # R_motion_local per joint

    # Provenance
    actor_skeleton: ActorSkeleton
    source_frame_index: int

    # Diagnostics
    constraint_violations: List[ConstraintViolation] = field(default_factory=list)
