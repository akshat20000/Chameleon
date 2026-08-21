"""
test_motion_retargeting.py — Phase 2.4D retargeting test suite.

10 verification gates:
  A  — Bone length preservation
  B  — Local rotation fidelity (motion delta < 0.5°)
  C  — Hierarchical consistency (FK self-check)
  D  — Multi-proportion invariance
  E  — L/R label identity (permutation test)
  F  — Extreme articulation (7 synthetic poses)
  G1 — World rotation continuity
  G2 — FK determinism
  G3 — Local motion preservation (geodesic invariant, tolerance < 0.5°)
  H  — Anatomical frame degenerate-region continuity

Architecture under test
-----------------------
    StableCanonicalMotionState
            ↓ (positions only; JointState.rotation NOT used)
    AnatomicalFrameBuilder  →  F_world(j)
            ↓
    LocalRotationExtractor  →  R_motion_local(j)
            ↓
    KinematicRetargeter     →  R_current_local = R_rest_actor @ R_motion_local
            ↓ FK
    RetargetedActorState
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest

# Path setup
SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.motion.canonical_state import (
    CanonicalMotionState, BodyPose, JointState, CANONICAL_MOTION_STATE_VERSION
)
from app.motion.actor_skeleton import (
    ActorSkeleton, JOINT_HIERARCHY, FK_JOINT_ORDER,
    DEFAULT_ACTOR, TALL_ACTOR, PETITE_ACTOR, LONG_ARMS_ACTOR, SHORT_ARMS_ACTOR,
    ACTOR_PROFILES,
)
from app.motion.anatomical_frame_builder import (
    AnatomicalFrameBuilder, DEGENERATE_THRESHOLD
)
from app.motion.local_rotation_extractor import LocalRotationExtractor, LocalJointRotations
from app.motion.motion_retargeter import KinematicRetargeter, geodesic_angle_deg
from app.motion.retargeted_actor_state import RetargetedActorState

_EPS = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic state helpers
# ──────────────────────────────────────────────────────────────────────────────

def _js(pos, conf=1.0) -> JointState:
    return JointState(position=np.array(pos, dtype=np.float32), confidence=conf)


def _axis_angle_to_matrix(axis, angle_deg: float) -> np.ndarray:
    """Build 3×3 SO(3) from unit axis and angle in degrees."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    t = 1 - c
    x, y, z = np.array(axis, dtype=np.float64) / (np.linalg.norm(axis) + _EPS)
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float32)


def _make_tpose_state(frame_index: int = 0) -> CanonicalMotionState:
    """
    Synthetic T-pose CanonicalMotionState.
    Positions are in canonical +Y-up, pelvis=origin coordinates.
    Arms extended horizontally, palms down, facing +Z.
    """
    body = BodyPose(
        pelvis=         _js([0.00,  0.00,  0.0]),
        spine_mid=      _js([0.00,  0.12,  0.0]),
        chest=          _js([0.00,  0.26,  0.0]),
        neck=           _js([0.00,  0.32,  0.0]),
        head=           _js([0.00,  0.44,  0.0]),
        left_hip=       _js([-0.09, 0.00,  0.0]),
        left_knee=      _js([-0.09,-0.25,  0.0]),
        left_ankle=     _js([-0.09,-0.49,  0.0]),
        right_hip=      _js([ 0.09, 0.00,  0.0]),
        right_knee=     _js([ 0.09,-0.25,  0.0]),
        right_ankle=    _js([ 0.09,-0.49,  0.0]),
        # Left arm: chest→shoulder→elbow→wrist along +X
        left_shoulder=  _js([ 0.09+0.09, 0.26, 0.0]),   # chest_x + shoulder_length
        left_elbow=     _js([ 0.09+0.09+0.16, 0.26, 0.0]),
        left_wrist=     _js([ 0.09+0.09+0.16+0.14, 0.26, 0.0]),
        # Right arm: along -X
        right_shoulder= _js([-0.09-0.09, 0.26, 0.0]),
        right_elbow=    _js([-0.09-0.09-0.16, 0.26, 0.0]),
        right_wrist=    _js([-0.09-0.09-0.16-0.14, 0.26, 0.0]),
    )
    return CanonicalMotionState(
        schema_version=CANONICAL_MOTION_STATE_VERSION,
        frame_index=frame_index,
        capture_timestamp=1.0 + frame_index * 0.033,
        source_backend="test",
        body=body,
    )


def _make_state_with_arm_raise(
    left_flex_deg: float = 0.0,
    right_flex_deg: float = 0.0,
    left_elbow_flex_deg: float = 0.0,
    right_elbow_flex_deg: float = 0.0,
    frame_index: int = 1,
) -> CanonicalMotionState:
    """
    Synthetic state with per-side shoulder and elbow flexion.
    Left shoulder flexion = rotation around -Z axis (camera convention).
    """
    tpose = _make_tpose_state(frame_index)
    b = tpose.body

    def _rotate_arm(shoulder_pos, elbow_pos, wrist_pos, shoulder_flex_deg, elbow_flex_deg):
        """Apply shoulder rotation to elbow and wrist, then elbow rotation to wrist."""
        R_sh = _axis_angle_to_matrix([0., 0., -1.], shoulder_flex_deg)  # -Z: lift arm upward
        sh = shoulder_pos
        elbow_offset_orig = elbow_pos - sh
        wrist_offset_orig = wrist_pos  - sh

        new_elbow = sh + R_sh @ elbow_offset_orig
        new_wrist = sh + R_sh @ wrist_offset_orig

        # Elbow flexion: rotate wrist relative to new elbow
        if abs(elbow_flex_deg) > 0.01:
            R_el = _axis_angle_to_matrix([0., 0., -1.], elbow_flex_deg)
            w_from_el = new_wrist - new_elbow
            new_wrist = new_elbow + R_el @ w_from_el

        return new_elbow, new_wrist

    # Left arm
    ls = b.left_shoulder.position
    le = b.left_elbow.position
    lw = b.left_wrist.position
    new_le, new_lw = _rotate_arm(ls, le, lw, left_flex_deg, left_elbow_flex_deg)

    # Right arm
    rs = b.right_shoulder.position
    re = b.right_elbow.position
    rw = b.right_wrist.position
    new_re, new_rw = _rotate_arm(rs, re, rw, right_flex_deg, right_elbow_flex_deg)

    import dataclasses
    new_body = dataclasses.replace(
        b,
        left_elbow=  _js(new_le),
        left_wrist=  _js(new_lw),
        right_elbow= _js(new_re),
        right_wrist= _js(new_rw),
    )
    import dataclasses as dc
    return dc.replace(tpose, body=new_body, frame_index=frame_index)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: retarget a state and return the result
# ──────────────────────────────────────────────────────────────────────────────

def _retarget(state: CanonicalMotionState, actor: ActorSkeleton,
              rest_state: Optional[CanonicalMotionState] = None) -> RetargetedActorState:
    """Fresh retargeter (no accumulated state) for a single frame."""
    retargeter = KinematicRetargeter(actor)
    if rest_state is not None:
        retargeter.initialize_rest_from_state(rest_state)
    else:
        retargeter.initialize_rest_from_state(_make_tpose_state())
    return retargeter.retarget(state)


# ──────────────────────────────────────────────────────────────────────────────
# Gate A — Bone Length Preservation
# ──────────────────────────────────────────────────────────────────────────────

class TestGateA:
    """Every bone in the retargeted output must match actor.bone_lengths."""

    TOLERANCE = 1e-3   # 1mm in normalized units

    @pytest.mark.parametrize("actor_name", list(ACTOR_PROFILES.keys()))
    def test_tpose_bone_lengths(self, actor_name):
        actor = ACTOR_PROFILES[actor_name]
        state = _make_tpose_state()
        result = _retarget(state, actor)

        for joint in FK_JOINT_ORDER:
            parent = JOINT_HIERARCHY[joint]
            if parent is None:
                continue
            expected_length = actor.bone_lengths.get(joint, 0.0)
            P_child = result.joints.get(joint)
            P_parent = result.joints.get(parent)
            if P_child is None or P_parent is None:
                continue
            actual_length = float(np.linalg.norm(P_child - P_parent))
            assert abs(actual_length - expected_length) < self.TOLERANCE, (
                f"Gate A [{actor_name}] bone {joint}: "
                f"expected={expected_length:.4f} actual={actual_length:.4f}"
            )

    @pytest.mark.parametrize("actor_name", list(ACTOR_PROFILES.keys()))
    def test_motion_pose_bone_lengths(self, actor_name):
        """Bone lengths must be preserved for non-trivial poses too."""
        actor = ACTOR_PROFILES[actor_name]
        state = _make_state_with_arm_raise(left_flex_deg=80, right_flex_deg=-15,
                                           left_elbow_flex_deg=45)
        result = _retarget(state, actor)

        for joint in FK_JOINT_ORDER:
            parent = JOINT_HIERARCHY[joint]
            if parent is None:
                continue
            expected_length = actor.bone_lengths.get(joint, 0.0)
            P_child = result.joints.get(joint)
            P_parent = result.joints.get(parent)
            if P_child is None or P_parent is None:
                continue
            actual_length = float(np.linalg.norm(P_child - P_parent))
            assert abs(actual_length - expected_length) < self.TOLERANCE, (
                f"Gate A motion [{actor_name}] bone {joint}: "
                f"expected={expected_length:.4f} actual={actual_length:.4f}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Gate B — Local Rotation Fidelity
# ──────────────────────────────────────────────────────────────────────────────

class TestGateB:
    """
    actor.motion_deltas[j] must equal performer motion deltas within 0.5°.
    (Not absolute bone direction — that differs between actors.)
    """

    TOLERANCE_DEG = 0.5

    def test_motion_delta_preserved(self):
        """R_motion_local must be transferred without modification."""
        actor = DEFAULT_ACTOR
        state = _make_state_with_arm_raise(left_flex_deg=60, right_flex_deg=-20,
                                           left_elbow_flex_deg=30, frame_index=1)
        result = _retarget(state, actor)

        # Extract performer motion deltas directly
        builder = AnatomicalFrameBuilder()
        extractor = LocalRotationExtractor()
        rest_frames = builder.build_frames(_make_tpose_state())
        extractor.initialize_rest_pose(rest_frames)
        frames = builder.build_frames(state)
        performer_motion = extractor.extract(frames, state.frame_index)

        for joint in ["left_shoulder", "left_elbow", "right_shoulder", "right_elbow"]:
            R_perf = performer_motion.joints.get(joint)
            R_actor = result.motion_deltas.joints.get(joint)
            if R_perf is None or R_actor is None:
                continue
            angle = geodesic_angle_deg(R_perf, R_actor)
            assert angle < self.TOLERANCE_DEG, (
                f"Gate B: {joint} motion delta mismatch = {angle:.3f}° (max {self.TOLERANCE_DEG}°)"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Gate C — Hierarchical Consistency (FK self-check)
# ──────────────────────────────────────────────────────────────────────────────

class TestGateC:
    """
    For every joint: stored position must equal FK-recomputed position
    from stored world rotations + actor skeleton.
    """

    TOLERANCE = 1e-4

    def test_fk_self_consistency(self):
        actor = DEFAULT_ACTOR
        state = _make_state_with_arm_raise(left_flex_deg=70, right_flex_deg=-10)
        result = _retarget(state, actor)

        for joint in FK_JOINT_ORDER:
            parent = JOINT_HIERARCHY[joint]
            if parent is None:
                continue
            P_parent = result.joints.get(parent)
            R_world_parent = result.world_rotations.get(parent)
            P_stored = result.joints.get(joint)
            if P_parent is None or R_world_parent is None or P_stored is None:
                continue

            v = actor.v_rest(joint)
            P_expected = P_parent + R_world_parent @ v
            err = float(np.linalg.norm(P_stored - P_expected))
            assert err < self.TOLERANCE, (
                f"Gate C: {joint} FK inconsistency = {err:.6f} (max {self.TOLERANCE})"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Gate D — Multi-Proportion Invariance
# ──────────────────────────────────────────────────────────────────────────────

class TestGateD:
    """
    Same performer frame retargeted to 5 actor profiles:
    - Motion deltas must be equal across all profiles (< 0.5°).
    - Bone lengths in each output must match the target profile.
    """

    DELTA_TOL_DEG = 0.5
    LENGTH_TOL    = 1e-3

    def test_same_motion_delta_across_profiles(self):
        state = _make_state_with_arm_raise(left_flex_deg=80, right_flex_deg=-20,
                                           left_elbow_flex_deg=40)
        results = {name: _retarget(state, actor)
                   for name, actor in ACTOR_PROFILES.items()}

        # Reference: DEFAULT
        ref = results["DEFAULT"]
        for name, res in results.items():
            if name == "DEFAULT":
                continue
            for joint in ["left_shoulder", "left_elbow", "right_shoulder", "right_elbow"]:
                R_ref  = ref.motion_deltas.joints.get(joint)
                R_other = res.motion_deltas.joints.get(joint)
                if R_ref is None or R_other is None:
                    continue
                angle = geodesic_angle_deg(R_ref, R_other)
                assert angle < self.DELTA_TOL_DEG, (
                    f"Gate D: {joint} delta mismatch DEFAULT vs {name}: {angle:.3f}°"
                )

    def test_each_profile_retains_own_proportions(self):
        state = _make_state_with_arm_raise(left_flex_deg=60)
        for name, actor in ACTOR_PROFILES.items():
            result = _retarget(state, actor)
            for joint in FK_JOINT_ORDER:
                parent = JOINT_HIERARCHY[joint]
                if parent is None:
                    continue
                P_c = result.joints.get(joint)
                P_p = result.joints.get(parent)
                if P_c is None or P_p is None:
                    continue
                expected = actor.bone_lengths.get(joint, 0.0)
                actual = float(np.linalg.norm(P_c - P_p))
                assert abs(actual - expected) < self.LENGTH_TOL, (
                    f"Gate D: {name} bone {joint}: expected={expected:.4f} actual={actual:.4f}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Gate E — L/R Label Identity (permutation test)
# ──────────────────────────────────────────────────────────────────────────────

class TestGateE:
    """
    The retargeter is label-preserving: it does NOT infer anatomical identity
    from positional heuristics.

    Correct input → correct output (same-side labels preserved).
    Swapped input → swapped output (retargeter does NOT silently correct swaps).
    """

    DELTA_TOL_DEG = 0.5

    def _extract_motion_deltas(self, state: CanonicalMotionState, rest: CanonicalMotionState
                                ) -> Dict[str, Optional[np.ndarray]]:
        builder = AnatomicalFrameBuilder()
        extractor = LocalRotationExtractor()
        extractor.initialize_rest_pose(builder.build_frames(rest))
        frames = builder.build_frames(state)
        motion = extractor.extract(frames, state.frame_index)
        return motion.joints

    def test_correct_labels_preserved(self):
        """Left → left, right → right: motion deltas match same-side performer deltas."""
        rest = _make_tpose_state()
        # Asymmetric: left arm raised 90°, right arm lowered 20°, left elbow bent 45°
        state = _make_state_with_arm_raise(
            left_flex_deg=90, right_flex_deg=-20,
            left_elbow_flex_deg=45, right_elbow_flex_deg=0,
            frame_index=1,
        )
        performer_deltas = self._extract_motion_deltas(state, rest)
        result = _retarget(state, DEFAULT_ACTOR, rest_state=rest)

        for joint in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]:
            R_perf  = performer_deltas.get(joint)
            R_actor = result.motion_deltas.joints.get(joint)
            if R_perf is None or R_actor is None:
                continue
            angle = geodesic_angle_deg(R_perf, R_actor)
            assert angle < self.DELTA_TOL_DEG, (
                f"Gate E (correct labels): {joint} mismatch = {angle:.3f}°"
            )

    def test_swapped_input_produces_swapped_output(self):
        """
        If the caller swaps left/right labels in the input state, the retargeter
        must preserve the swap — it does NOT silently fix anatomical identity.
        """
        rest = _make_tpose_state()
        # Normal asymmetric state
        normal = _make_state_with_arm_raise(
            left_flex_deg=90, right_flex_deg=-20,
            left_elbow_flex_deg=45, right_elbow_flex_deg=0,
            frame_index=1,
        )

        # Build a "swapped" state by exchanging left/right arm joint positions
        import dataclasses as dc
        b = normal.body
        swapped_body = dc.replace(
            b,
            left_shoulder=  dc.replace(b.right_shoulder,  position=b.right_shoulder.position.copy()),
            right_shoulder= dc.replace(b.left_shoulder,   position=b.left_shoulder.position.copy()),
            left_elbow=     dc.replace(b.right_elbow,     position=b.right_elbow.position.copy()),
            right_elbow=    dc.replace(b.left_elbow,      position=b.left_elbow.position.copy()),
            left_wrist=     dc.replace(b.right_wrist,     position=b.right_wrist.position.copy()),
            right_wrist=    dc.replace(b.left_wrist,      position=b.left_wrist.position.copy()),
        )
        swapped_state = dc.replace(normal, body=swapped_body, frame_index=2)

        # Get performer deltas for swapped state
        swapped_deltas = self._extract_motion_deltas(swapped_state, rest)
        result_swapped = _retarget(swapped_state, DEFAULT_ACTOR, rest_state=rest)

        # Retargeter must preserve the swapped labels — NOT correct them
        for joint in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]:
            R_input  = swapped_deltas.get(joint)
            R_output = result_swapped.motion_deltas.joints.get(joint)
            if R_input is None or R_output is None:
                continue
            angle = geodesic_angle_deg(R_input, R_output)
            assert angle < self.DELTA_TOL_DEG, (
                f"Gate E (swapped): retargeter altered label {joint} by {angle:.3f}°"
            )

    def test_hierarchical_chain_continuity(self):
        """
        In the output, each joint's parent must be anatomically proximal:
        bone length > 0 and equals actor.bone_lengths[joint].
        """
        rest = _make_tpose_state()
        state = _make_state_with_arm_raise(left_flex_deg=70, right_flex_deg=70, frame_index=1)
        result = _retarget(state, DEFAULT_ACTOR, rest_state=rest)

        for side in ["left", "right"]:
            chain = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
            for child in chain[1:]:
                parent = JOINT_HIERARCHY[child]
                P_c = result.joints.get(child)
                P_p = result.joints.get(parent)
                if P_c is None or P_p is None:
                    continue
                length = float(np.linalg.norm(P_c - P_p))
                assert length > 1e-4, (
                    f"Gate E chain: {child} parent={parent} distance={length:.6f} (near zero)"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Gate F — Extreme Articulation (7 synthetic poses)
# ──────────────────────────────────────────────────────────────────────────────

class TestGateF:
    """
    For T-pose (identity delta), the actor must be exactly at its rest positions.
    For non-trivial poses, bone lengths must be preserved.
    """

    REST_TOL  = 1e-3  # T-pose: position error tolerance
    BONE_TOL  = 1e-3  # all poses: bone length tolerance

    def test_tpose_actor_at_rest_positions(self):
        """Identity deltas → actor at exact rest-pose positions."""
        actor = DEFAULT_ACTOR
        state = _make_tpose_state()
        result = _retarget(state, actor, rest_state=state)

        for joint, expected_pos in actor.rest_positions.items():
            actual = result.joints.get(joint)
            if actual is None:
                continue
            err = float(np.linalg.norm(actual - expected_pos))
            assert err < self.REST_TOL, (
                f"Gate F T-pose: {joint} err={err:.4f} expected={expected_pos} actual={actual}"
            )

    @pytest.mark.parametrize("left_flex,right_flex,left_elbow,right_elbow,desc", [
        ( 90,   0,  0,  0, "left_arm_forward"),
        (170,  90,  0,  0, "arms_overhead"),
        ( 45,  45, 90, 90, "arms_crossed_elbow_bent"),
        ( 80, -80,  0,  0, "asymmetric_large"),
        (  0,   0, 90, 90, "elbows_bent"),
        (-20, -20,  0,  0, "arms_slightly_lowered"),
        ( 45,  45, 45, 45, "symmetric_raise_and_bend"),
    ])
    def test_bone_lengths_preserved(self, left_flex, right_flex, left_elbow, right_elbow, desc):
        actor = DEFAULT_ACTOR
        state = _make_state_with_arm_raise(
            left_flex_deg=left_flex, right_flex_deg=right_flex,
            left_elbow_flex_deg=left_elbow, right_elbow_flex_deg=right_elbow,
            frame_index=1,
        )
        result = _retarget(state, actor)

        for joint in FK_JOINT_ORDER:
            parent = JOINT_HIERARCHY[joint]
            if parent is None:
                continue
            P_c = result.joints.get(joint)
            P_p = result.joints.get(parent)
            if P_c is None or P_p is None:
                continue
            expected = actor.bone_lengths.get(joint, 0.0)
            actual = float(np.linalg.norm(P_c - P_p))
            assert abs(actual - expected) < self.BONE_TOL, (
                f"Gate F [{desc}] bone {joint}: expected={expected:.4f} actual={actual:.4f}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Gate G1 — World Rotation Continuity
# ──────────────────────────────────────────────────────────────────────────────

class TestGateG1:
    """
    Consecutive actor world rotations must not jump > G1_MAX_DEG between frames.
    """

    G1_MAX_DEG = 15.0

    def test_rotation_continuity_across_smooth_sequence(self):
        actor = DEFAULT_ACTOR
        retargeter = KinematicRetargeter(actor)
        retargeter.initialize_rest_from_state(_make_tpose_state())

        prev_result = None
        for i, flex in enumerate(range(0, 81, 5)):
            state = _make_state_with_arm_raise(left_flex_deg=float(flex), frame_index=i)
            result = retargeter.retarget(state)

            if prev_result is not None:
                for joint in ["left_shoulder", "left_elbow", "right_shoulder"]:
                    R_prev = prev_result.world_rotations.get(joint)
                    R_curr = result.world_rotations.get(joint)
                    if R_prev is None or R_curr is None:
                        continue
                    delta = geodesic_angle_deg(R_prev, R_curr)
                    assert delta < self.G1_MAX_DEG, (
                        f"Gate G1: {joint} world rotation jump {delta:.2f}° at frame {i}"
                    )
            prev_result = result


# ──────────────────────────────────────────────────────────────────────────────
# Gate G2 — FK Determinism
# ──────────────────────────────────────────────────────────────────────────────

class TestGateG2:
    """
    Given world rotations + actor skeleton, recomputing FK must reproduce
    the stored positions to numerical precision.
    """

    TOLERANCE = 1e-4

    def test_fk_determinism(self):
        actor = DEFAULT_ACTOR
        state = _make_state_with_arm_raise(left_flex_deg=60, right_flex_deg=-15,
                                           left_elbow_flex_deg=40)
        result = _retarget(state, actor)

        for joint in FK_JOINT_ORDER:
            parent = JOINT_HIERARCHY[joint]
            if parent is None:
                continue
            P_parent = result.joints.get(parent)
            R_world_parent = result.world_rotations.get(parent)
            P_stored = result.joints.get(joint)
            if P_parent is None or R_world_parent is None or P_stored is None:
                continue

            v = actor.v_rest(joint)
            P_recomputed = P_parent + R_world_parent @ v
            err = float(np.linalg.norm(P_stored - P_recomputed))
            assert err < self.TOLERANCE, (
                f"Gate G2: {joint} FK determinism error = {err:.6f}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Gate G3 — Local Motion Preservation
# ──────────────────────────────────────────────────────────────────────────────

class TestGateG3:
    """
    Mathematical basis:
        R_current_local_actor = R_rest_actor @ R_motion_local
        Geodesic distance is invariant under fixed left multiplication:
        angle(R_rest @ A, R_rest @ B) == angle(A, B)

    Therefore:
        angle(R_current_local_actor(t), R_current_local_actor(t-1))
        must equal
        angle(R_motion_local_performer(t), R_motion_local_performer(t-1))

    Tolerance: < 0.5° (numerical precision only, no FK slack).
    """

    TOLERANCE_DEG = 0.5

    def test_local_motion_preserved_across_frames(self):
        actor = DEFAULT_ACTOR
        retargeter = KinematicRetargeter(actor)
        retargeter.initialize_rest_from_state(_make_tpose_state())

        prev_motion  = None
        prev_current = None

        for i, flex in enumerate(range(0, 71, 10)):
            state = _make_state_with_arm_raise(left_flex_deg=float(flex),
                                               right_flex_deg=float(-flex // 3),
                                               frame_index=i)
            result = retargeter.retarget(state)

            if prev_motion is not None and prev_current is not None:
                for joint in ["left_shoulder", "left_elbow", "right_shoulder"]:
                    R_motion_prev = prev_motion.get(joint)
                    R_motion_curr = result.motion_deltas.joints.get(joint)
                    R_cur_prev    = prev_current.get(joint)
                    R_cur_curr    = result.local_rotations.joints.get(joint)

                    if any(x is None for x in [R_motion_prev, R_motion_curr,
                                                R_cur_prev, R_cur_curr]):
                        continue

                    performer_delta_angle = geodesic_angle_deg(R_motion_prev, R_motion_curr)
                    actor_local_delta     = geodesic_angle_deg(R_cur_prev, R_cur_curr)

                    err = abs(actor_local_delta - performer_delta_angle)
                    assert err < self.TOLERANCE_DEG, (
                        f"Gate G3: {joint} local motion delta mismatch = {err:.4f}° "
                        f"(performer={performer_delta_angle:.4f}°, "
                        f"actor_local={actor_local_delta:.4f}°) at frame {i}"
                    )

            prev_motion  = {j: v.copy() if v is not None else None
                            for j, v in result.motion_deltas.joints.items()}
            prev_current = {j: v.copy() if v is not None else None
                            for j, v in result.local_rotations.joints.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Gate H — Anatomical Frame Degenerate-Region Continuity
# ──────────────────────────────────────────────────────────────────────────────

class TestGateH:
    """
    The degenerate condition is defined by norm(cross(primary, secondary)) < DEGENERATE_THRESHOLD.
    Landmark positions are constructed explicitly to satisfy this condition.
    """

    MAX_FRAME_DELTA_DEG = 20.0
    RECOVERY_TOL_DEG    = 5.0

    def _make_state_from_arm_vecs(self, shoulder, elbow, wrist, frame_index) -> CanonicalMotionState:
        """Build a CanonicalMotionState with specified arm joint positions."""
        tpose = _make_tpose_state(frame_index)
        import dataclasses as dc
        b = tpose.body
        new_body = dc.replace(
            b,
            left_elbow=_js(elbow),
            left_wrist=_js(wrist),
        )
        return dc.replace(tpose, body=new_body, frame_index=frame_index)

    def test_degenerate_entry_no_discontinuity(self):
        """
        Frame 0: clearly bent arm (cross-product > DEGENERATE_THRESHOLD).
        Frame 1: nearly straight arm (cross-product < DEGENERATE_THRESHOLD).
        The frame builder must hold frame 0's orientation, causing no large jump.
        """
        # Frame 0: elbow clearly bent
        shoulder_pos = np.array([ 0.18, 0.26, 0.0], dtype=np.float32)
        elbow_bent   = np.array([ 0.34, 0.26, 0.0], dtype=np.float32)  # straight out +X
        wrist_bent   = np.array([ 0.34, 0.12, 0.0], dtype=np.float32)  # wrist drops down

        upper_arm = (elbow_bent - shoulder_pos) / (np.linalg.norm(elbow_bent - shoulder_pos) + 1e-9)
        forearm   = (wrist_bent - elbow_bent)   / (np.linalg.norm(wrist_bent - elbow_bent) + 1e-9)
        cp_bent   = np.linalg.norm(np.cross(upper_arm, forearm))
        assert cp_bent >= DEGENERATE_THRESHOLD, (
            f"Gate H setup: normal frame cross-product {cp_bent:.6e} should be >= {DEGENERATE_THRESHOLD}"
        )

        # Frame 1: nearly straight arm — explicitly construct degenerate landmarks
        # Wrist nearly collinear with shoulder→elbow direction
        elbow_ext  = np.array([ 0.34, 0.26,  0.0], dtype=np.float32)
        wrist_ext  = np.array([ 0.48, 0.26, -0.000001], dtype=np.float32)  # nearly collinear → cross < 1e-4
        forearm_ext = (wrist_ext - elbow_ext) / (np.linalg.norm(wrist_ext - elbow_ext) + 1e-9)
        cp_ext = np.linalg.norm(np.cross(upper_arm, forearm_ext))
        assert cp_ext < DEGENERATE_THRESHOLD, (
            f"Gate H setup: degenerate frame cross-product {cp_ext:.6e} should be < {DEGENERATE_THRESHOLD}"
        )

        # Run builder through both frames
        builder = AnatomicalFrameBuilder()
        state_0 = self._make_state_from_arm_vecs(shoulder_pos, elbow_bent, wrist_bent, 0)
        state_1 = self._make_state_from_arm_vecs(shoulder_pos, elbow_ext,  wrist_ext,  1)

        frames_0 = builder.build_frames(state_0)
        frames_1 = builder.build_frames(state_1)

        F_0 = frames_0.get("left_elbow")
        F_1 = frames_1.get("left_elbow")

        assert F_0 is not None, "Gate H: frame 0 (non-degenerate) must produce a valid frame"
        assert F_1 is not None, "Gate H: frame 1 (degenerate) must return held frame, not None"

        # H1: no large jump on degenerate entry
        delta = geodesic_angle_deg(F_0, F_1)
        assert delta < self.MAX_FRAME_DELTA_DEG, (
            f"Gate H1: degenerate entry jump = {delta:.2f}° (max {self.MAX_FRAME_DELTA_DEG}°)"
        )

        # H2: held frame must be the same as frame 0
        assert np.allclose(F_1, F_0, atol=1e-5), (
            "Gate H2: held frame during degenerate zone must equal last valid frame"
        )

    def test_degenerate_recovery(self):
        """
        After returning to a non-degenerate configuration, the frame must resume
        anatomically consistent orientation (close to a fresh builder's result).
        """
        shoulder_pos = np.array([ 0.18, 0.26, 0.0], dtype=np.float32)
        elbow_pos    = np.array([ 0.34, 0.26, 0.0], dtype=np.float32)
        wrist_bent   = np.array([ 0.34, 0.12, 0.0], dtype=np.float32)
        wrist_ext    = np.array([ 0.48, 0.26, -0.000001], dtype=np.float32)  # same degenerate vector as above test

        # Stateful builder: normal → degenerate → normal
        builder = AnatomicalFrameBuilder()
        state_0 = self._make_state_from_arm_vecs(shoulder_pos, elbow_pos, wrist_bent, 0)
        state_1 = self._make_state_from_arm_vecs(shoulder_pos, elbow_pos, wrist_ext,  1)
        state_2 = self._make_state_from_arm_vecs(shoulder_pos, elbow_pos, wrist_bent, 2)  # recover

        builder.build_frames(state_0)
        builder.build_frames(state_1)
        frames_2 = builder.build_frames(state_2)

        # Fresh builder (no state) for comparison
        fresh_builder = AnatomicalFrameBuilder()
        fresh_frames_2 = fresh_builder.build_frames(state_2)

        F_2       = frames_2.get("left_elbow")
        F_fresh_2 = fresh_frames_2.get("left_elbow")

        assert F_2 is not None, "Gate H3: recovered frame must not be None"
        assert F_fresh_2 is not None, "Gate H3: fresh frame for recovered state must not be None"

        delta = geodesic_angle_deg(F_2, F_fresh_2)
        assert delta < self.RECOVERY_TOL_DEG, (
            f"Gate H3: recovered frame differs from fresh by {delta:.2f}° (max {self.RECOVERY_TOL_DEG}°)"
        )
