"""
MediaPipe → CanonicalMotionState adapter.

This is the ONLY module that is allowed to access MediaPipe-specific field
structures (PerformerState, FaceState, BodyPoseState, HandState, etc.).

All downstream modules receive only CanonicalMotionState.

MediaPipe Pose Landmark Index Reference
----------------------------------------
 0: nose                17: left_pinky
 1: left_eye_inner      18: right_pinky
 2: left_eye            19: left_index
 3: left_eye_outer      20: right_index
 4: right_eye_inner     21: left_thumb
 5: right_eye           22: right_thumb
 6: right_eye_outer     23: left_hip
 7: left_ear            24: right_hip
 8: right_ear           25: left_knee
 9: mouth_left          26: right_knee
10: mouth_right         27: left_ankle
11: left_shoulder       28: right_ankle
12: right_shoulder      29: left_heel
13: left_elbow          30: right_heel
14: right_elbow         31: left_foot_index
15: left_wrist          32: right_foot_index
16: right_wrist

MediaPipe Hand Landmark Index Reference (per hand)
---------------------------------------------------
 0: wrist               11: middle_pip
 1: thumb_cmc           12: middle_dip
 2: thumb_mcp           13: ring_mcp
 3: thumb_ip            14: ring_pip
 4: thumb_tip           15: ring_dip
 5: index_mcp           16: ring_tip
 6: index_pip           17: pinky_mcp
 7: index_dip           18: pinky_pip
 8: index_tip           19: pinky_dip
 9: middle_mcp          20: pinky_tip
10: middle_pip
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from app.motion.canonical_state import (
    CANONICAL_MOTION_STATE_VERSION,
    BodyPose,
    CanonicalMotionState,
    CONFIDENCE_THRESHOLD,
    FacialExpression,
    FingerState,
    HandPose,
    JointState,
)

# ── MediaPipe body landmark indices ──────────────────────────────────────────
_LM_NOSE          = 0
_LM_LEFT_SHOULDER = 11
_LM_RIGHT_SHOULDER = 12
_LM_LEFT_ELBOW    = 13
_LM_RIGHT_ELBOW   = 14
_LM_LEFT_WRIST    = 15
_LM_RIGHT_WRIST   = 16
_LM_LEFT_HIP      = 23
_LM_RIGHT_HIP     = 24
_LM_LEFT_KNEE     = 25
_LM_RIGHT_KNEE    = 26
_LM_LEFT_ANKLE    = 27
_LM_RIGHT_ANKLE   = 28

# ── MediaPipe hand landmark index groups ─────────────────────────────────────
_HAND_WRIST  = 0
_HAND_THUMB  = (1, 3, 4)    # CMC, IP, TIP
_HAND_INDEX  = (5, 6, 7)    # MCP, PIP, DIP (tip=8 excluded, DIP proxies tip)
_HAND_MIDDLE = (9, 10, 11)
_HAND_RING   = (13, 14, 15)
_HAND_PINKY  = (17, 18, 19)

_HAND_CONFIDENCE_THRESHOLD = 0.3


# ==============================================================================
# Internal helpers
# ==============================================================================

def _lm_pos(lms: np.ndarray, idx: int) -> np.ndarray:
    """Return xyz of landmark at index idx as float32 (3,)."""
    return lms[idx, :3].astype(np.float32)


def _lm_vis(lms: np.ndarray, idx: int) -> float:
    """Return visibility of landmark at index idx (column 3 of (33,4) array)."""
    if lms.shape[1] >= 4:
        return float(np.clip(lms[idx, 3], 0.0, 1.0))
    return 1.0


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a + b) * 0.5).astype(np.float32)


def _make_joint(
    pos: np.ndarray,
    confidence: float,
    rotation: Optional[np.ndarray] = None,
) -> JointState:
    return JointState(
        position=pos.astype(np.float32),
        rotation=rotation,
        confidence=confidence,
    )


def _make_joint_optional(
    lms: np.ndarray,
    idx: int,
    pelvis: np.ndarray,
    scale: float,
) -> Optional[JointState]:
    """
    Build a JointState from a MediaPipe world-landmark entry.

    Converts MediaPipe world coordinates (origin = hip midpoint, meters)
    to canonical coordinates (origin = pelvis, units = body height).
    Returns None when confidence < CONFIDENCE_THRESHOLD.
    """
    conf = _lm_vis(lms, idx)
    pos_metric = _lm_pos(lms, idx)
    # MediaPipe world landmarks are already hip-centred, but we subtract
    # our computed pelvis to be explicit about the origin.
    pos_canon = (pos_metric - pelvis) / max(scale, 1e-6)
    # MediaPipe world coordinates use +Y pointing DOWNWARD (toward ground)
    # and +Z pointing AWAY from camera (into screen).
    # CanonicalMotionState requires +Y UPWARD and +Z TOWARD camera.
    pos_canon[1] = -pos_canon[1]
    pos_canon[2] = -pos_canon[2]
    return _make_joint(pos_canon, conf)


def _estimate_body_scale(lms: np.ndarray) -> float:
    """
    Estimate body height in MediaPipe world-landmark units (approx. meters).

    Priority order:
    1. |left_shoulder - left_ankle|  (full body visible)
    2. |left_shoulder - left_hip| * 2  (lower body missing)
    3. |left_shoulder - right_shoulder| * 3.5  (torso/face-only; shoulder span ≈ 28% height)
    Returns 0.0 only if no landmarks are visible at all.
    """
    ls_conf = _lm_vis(lms, _LM_LEFT_SHOULDER)
    rs_conf = _lm_vis(lms, _LM_RIGHT_SHOULDER)
    la_conf = _lm_vis(lms, _LM_LEFT_ANKLE)
    lh_conf = _lm_vis(lms, _LM_LEFT_HIP)

    if ls_conf >= CONFIDENCE_THRESHOLD and la_conf >= CONFIDENCE_THRESHOLD:
        return float(np.linalg.norm(_lm_pos(lms, _LM_LEFT_SHOULDER) - _lm_pos(lms, _LM_LEFT_ANKLE)))

    if ls_conf >= CONFIDENCE_THRESHOLD and lh_conf >= CONFIDENCE_THRESHOLD:
        return float(np.linalg.norm(_lm_pos(lms, _LM_LEFT_SHOULDER) - _lm_pos(lms, _LM_LEFT_HIP))) * 2.0

    if ls_conf >= CONFIDENCE_THRESHOLD and rs_conf >= CONFIDENCE_THRESHOLD:
        shoulder_width = float(np.linalg.norm(_lm_pos(lms, _LM_LEFT_SHOULDER) - _lm_pos(lms, _LM_RIGHT_SHOULDER)))
        return shoulder_width * 3.5  # empirical: biacromial breadth ≈ 28% of height

    return 0.0


def _bone_rotation(
    parent_pos: np.ndarray,
    child_pos: np.ndarray,
    reference_direction: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32),
) -> Optional[np.ndarray]:
    """
    Estimate a 3×3 rotation matrix that rotates the reference_direction
    (canonical T-pose bone axis) to align with the actual bone direction
    (child_pos - parent_pos).

    Returns None if the bone has near-zero length.
    """
    bone_dir = child_pos - parent_pos
    length = float(np.linalg.norm(bone_dir))
    if length < 1e-6:
        return None
    bone_dir = bone_dir / length
    ref = reference_direction / (np.linalg.norm(reference_direction) + 1e-9)

    # Rodrigues' rotation formula
    cross = np.cross(ref, bone_dir)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.dot(ref, bone_dir))

    if cross_norm < 1e-9:
        # Vectors are parallel
        if dot > 0:
            return np.eye(3, dtype=np.float32)
        else:
            # 180° rotation around any perpendicular axis
            perp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(ref, perp))) > 0.9:
                perp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            perp = np.cross(ref, perp)
            perp /= np.linalg.norm(perp)
            # R = I - 2*ref*ref^T + 2*perp*perp^T  (special case)
            R = (
                np.eye(3, dtype=np.float32)
                - 2.0 * np.outer(ref, ref)
                + 2.0 * np.outer(perp, perp)
            )
            return R.astype(np.float32)

    axis = cross / cross_norm
    angle = float(np.arctan2(cross_norm, dot))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float32)
    R = (
        np.eye(3, dtype=np.float32)
        + np.sin(angle) * K
        + (1 - np.cos(angle)) * (K @ K)
    )
    return R.astype(np.float32)


def _build_finger_state(
    hand_lms_world: np.ndarray,
    indices: Tuple[int, int, int],
    wrist_world: np.ndarray,
    scale: float,
) -> FingerState:
    p_idx, m_idx, d_idx = indices

    def _fj(idx: int) -> JointState:
        pos_metric = hand_lms_world[idx, :3].astype(np.float32)
        pos_canon = (pos_metric - wrist_world) / max(scale, 1e-6)
        pos_canon[1] = -pos_canon[1]
        pos_canon[2] = -pos_canon[2]
        # MediaPipe hand landmark confidence: no per-landmark visibility,
        # use presence threshold of 1.0 as default
        return JointState(position=pos_canon, confidence=1.0)

    return FingerState(
        proximal=_fj(p_idx),
        middle=_fj(m_idx),
        distal=_fj(d_idx),
    )


# ==============================================================================
# Public adapter functions
# ==============================================================================

def adapt_body_pose(
    body_lms: np.ndarray,
    scale: float,
) -> BodyPose:
    """
    Convert a MediaPipe Pose world-landmarks array to a canonical BodyPose.

    Parameters
    ----------
    body_lms : np.ndarray, shape (33, 4)
        MediaPipe Pose world landmarks: columns are (x, y, z, visibility).
        World coordinates: hip-centred, Y up, Z toward camera (negated here).
    scale : float
        Estimated body height in MediaPipe world units. Used to normalize.

    Returns
    -------
    BodyPose with pelvis always populated.
    """
    # Pelvis = midpoint of hips in metric space (MediaPipe's world origin)
    left_hip_metric  = _lm_pos(body_lms, _LM_LEFT_HIP)
    right_hip_metric = _lm_pos(body_lms, _LM_RIGHT_HIP)
    pelvis_metric    = _midpoint(left_hip_metric, right_hip_metric)

    # In canonical space pelvis is at origin
    pelvis_canon = np.zeros(3, dtype=np.float32)
    pelvis_conf  = min(_lm_vis(body_lms, _LM_LEFT_HIP), _lm_vis(body_lms, _LM_RIGHT_HIP))

    def _joint(idx: int) -> Optional[JointState]:
        return _make_joint_optional(body_lms, idx, pelvis_metric, scale)

    def _joint_midpoint(idx_a: int, idx_b: int) -> Optional[JointState]:
        conf_a = _lm_vis(body_lms, idx_a)
        conf_b = _lm_vis(body_lms, idx_b)
        if conf_a < CONFIDENCE_THRESHOLD and conf_b < CONFIDENCE_THRESHOLD:
            return None
        pa = (_lm_pos(body_lms, idx_a) - pelvis_metric) / max(scale, 1e-6)
        pb = (_lm_pos(body_lms, idx_b) - pelvis_metric) / max(scale, 1e-6)
        pa[1] = -pa[1];  pa[2] = -pa[2]
        pb[1] = -pb[1];  pb[2] = -pb[2]
        mid_conf = (conf_a + conf_b) / 2.0
        return _make_joint(_midpoint(pa, pb), mid_conf)

    # Derived joints (not directly from landmarks)
    chest_joint    = _joint_midpoint(_LM_LEFT_SHOULDER, _LM_RIGHT_SHOULDER)
    left_hip_j     = _joint(_LM_LEFT_HIP)
    right_hip_j    = _joint(_LM_RIGHT_HIP)

    # spine_mid: midpoint of pelvis (origin) and chest
    spine_mid_joint = None
    if chest_joint is not None:
        spine_mid_pos = chest_joint.position * 0.5  # halfway between pelvis(0) and chest
        spine_mid_joint = _make_joint(spine_mid_pos, chest_joint.confidence)

    # neck: estimated above chest in proportion
    neck_joint = None
    head_j = _joint(_LM_NOSE)
    if chest_joint is not None and head_j is not None and head_j.is_visible:
        # neck ≈ 30% of the way from chest to head
        neck_pos = chest_joint.position + 0.3 * (head_j.position - chest_joint.position)
        neck_conf = min(chest_joint.confidence, head_j.confidence)
        neck_joint = _make_joint(neck_pos, neck_conf)

    # Add rotations to arm and leg bones
    def _with_rotation(
        parent_j: Optional[JointState],
        child_j: Optional[JointState],
    ) -> Optional[JointState]:
        """Return child_j with a rotation matrix computed from parent→child bone."""
        if parent_j is None or child_j is None:
            return child_j
        R = _bone_rotation(parent_j.position, child_j.position)
        return JointState(position=child_j.position, rotation=R, confidence=child_j.confidence)

    ls = _joint(_LM_LEFT_SHOULDER)
    le = _joint(_LM_LEFT_ELBOW)
    lw = _joint(_LM_LEFT_WRIST)
    rs = _joint(_LM_RIGHT_SHOULDER)
    re = _joint(_LM_RIGHT_ELBOW)
    rw = _joint(_LM_RIGHT_WRIST)
    lk = _joint(_LM_LEFT_KNEE)
    la = _joint(_LM_LEFT_ANKLE)
    rk = _joint(_LM_RIGHT_KNEE)
    ra = _joint(_LM_RIGHT_ANKLE)

    return BodyPose(
        pelvis        = _make_joint(pelvis_canon, pelvis_conf),
        spine_mid     = spine_mid_joint,
        chest         = chest_joint,
        neck          = neck_joint,
        head          = head_j,
        left_shoulder = _with_rotation(chest_joint, ls),
        left_elbow    = _with_rotation(ls, le),
        left_wrist    = _with_rotation(le, lw),
        right_shoulder = _with_rotation(chest_joint, rs),
        right_elbow   = _with_rotation(rs, re),
        right_wrist   = _with_rotation(re, rw),
        left_hip      = left_hip_j,
        left_knee     = _with_rotation(left_hip_j, lk),
        left_ankle    = _with_rotation(lk, la),
        right_hip     = right_hip_j,
        right_knee    = _with_rotation(right_hip_j, rk),
        right_ankle   = _with_rotation(rk, ra),
    )


def adapt_hand_pose(
    hand_lms_world: np.ndarray,
    handedness: str,
    body_scale: float,
    wrist_canon: Optional[np.ndarray] = None,
) -> HandPose:
    """
    Convert a MediaPipe Hand world-landmark array to a canonical HandPose.

    Parameters
    ----------
    hand_lms_world : np.ndarray, shape (21, 3)
        MediaPipe Hand world landmarks in world coordinates.
    handedness : str
        'Left' or 'Right'.
    body_scale : float
        Body height in world units, for normalisation.
    wrist_canon : np.ndarray or None
        Canonical wrist position from body pose. If None, estimated from
        hand landmarks.

    Returns
    -------
    HandPose in canonical world coordinates.
    """
    wrist_world = hand_lms_world[_HAND_WRIST, :3].astype(np.float32)
    scale = max(body_scale, 1e-6)

    # Convert wrist to canonical (relative to body pelvis is not available here,
    # so we express hand joints relative to the wrist and position the wrist
    # using the body-pose wrist joint if provided).
    if wrist_canon is not None:
        wrist_pos = wrist_canon.astype(np.float32)
    else:
        # Fallback: place wrist at hand-landmark origin (wrist = zero)
        wrist_pos = np.zeros(3, dtype=np.float32)

    wrist_joint = JointState(position=wrist_pos, confidence=1.0)

    def _finger(indices: Tuple[int, int, int]) -> FingerState:
        return _build_finger_state(hand_lms_world, indices, wrist_world, scale)

    confs = [1.0]  # MediaPipe Hand does not expose per-landmark visibility

    return HandPose(
        wrist=wrist_joint,
        thumb=_finger(_HAND_THUMB),
        index=_finger(_HAND_INDEX),
        middle=_finger(_HAND_MIDDLE),
        ring=_finger(_HAND_RING),
        pinky=_finger(_HAND_PINKY),
        handedness=handedness,
        overall_confidence=float(min(confs)),
    )


def adapt_facial_expression(
    blendshapes: dict,
    transformation_matrix: np.ndarray,
    confidence: float = 1.0,
) -> FacialExpression:
    """
    Build a FacialExpression from MediaPipe FaceLandmarker outputs.

    Parameters
    ----------
    blendshapes : dict
        52 ARKit blendshape coefficients (name → float).
    transformation_matrix : np.ndarray, shape (4, 4)
        MediaPipe facial transformation matrix.
    confidence : float
        Face detection confidence.

    Returns
    -------
    FacialExpression in canonical space.
    """
    mat = np.asarray(transformation_matrix, dtype=np.float32)
    head_rot = mat[:3, :3] if mat.shape == (4, 4) else np.eye(3, dtype=np.float32)

    return FacialExpression(
        blendshapes=dict(blendshapes),
        head_rotation=head_rot,
        eye_open_left=float(blendshapes.get("eyeBlinkLeft", 0.0)),
        eye_open_right=float(blendshapes.get("eyeBlinkRight", 0.0)),
        jaw_open=float(blendshapes.get("jawOpen", 0.0)),
        confidence=confidence,
    )


# ==============================================================================
# Top-level adapter entry point
# ==============================================================================

def adapt_performer_state(
    performer_state,
    frame_index: int,
    capture_timestamp: float,
) -> CanonicalMotionState:
    """
    Convert a PerformerState to a CanonicalMotionState.

    This is the ONLY function allowed to access PerformerState internals.
    All downstream modules receive only CanonicalMotionState.

    Parameters
    ----------
    performer_state : PerformerState
        Raw tracker output from the motion extraction pipeline.
    frame_index : int
        Monotonically increasing frame counter.
    capture_timestamp : float
        Unix timestamp at capture.

    Returns
    -------
    CanonicalMotionState — never raises; returns a degraded state on failure.
    """
    timings: dict = {}

    # ── Body pose ─────────────────────────────────────────────────────────────
    body_pose: Optional[BodyPose] = None
    body_scale = 0.0

    t = time.perf_counter()
    if performer_state.body is not None and performer_state.body.landmarks_3d is not None:
        lms = performer_state.body.landmarks_3d  # (33, 4)
        body_scale = _estimate_body_scale(lms)
        body_pose = adapt_body_pose(lms, body_scale)
    timings["body_adapt_ms"] = (time.perf_counter() - t) * 1000

    if body_pose is None:
        # Always need a pelvis — fallback to origin
        body_pose = BodyPose(pelvis=JointState(
            position=np.zeros(3, dtype=np.float32),
            confidence=0.0,
            is_visible=False,
        ))

    # ── Face expression ───────────────────────────────────────────────────────
    face_expr: Optional[FacialExpression] = None

    t = time.perf_counter()
    primary = performer_state.primary_face
    if primary is not None and primary.blendshapes:
        face_expr = adapt_facial_expression(
            blendshapes=primary.blendshapes,
            transformation_matrix=primary.head_rotation.transformation_matrix,
            confidence=primary.confidence,
        )
    timings["face_adapt_ms"] = (time.perf_counter() - t) * 1000

    # ── Hands ─────────────────────────────────────────────────────────────────
    left_hand_pose: Optional[HandPose] = None
    right_hand_pose: Optional[HandPose] = None

    t = time.perf_counter()
    for hand_attr, target_handedness in [
        ("left_hand", "Left"), ("right_hand", "Right")
    ]:
        hand = getattr(performer_state, hand_attr)
        if hand is None or hand.landmarks_3d is None:
            continue
        # Use body-pose wrist position as anchor when available
        wrist_canon = None
        if hand.handedness == "Left" and body_pose.left_wrist is not None:
            wrist_canon = body_pose.left_wrist.position
        elif hand.handedness == "Right" and body_pose.right_wrist is not None:
            wrist_canon = body_pose.right_wrist.position

        hp = adapt_hand_pose(
            hand_lms_world=hand.landmarks_3d,
            handedness=hand.handedness,
            body_scale=body_scale,
            wrist_canon=wrist_canon,
        )
        if hand.handedness == "Left":
            left_hand_pose = hp
        else:
            right_hand_pose = hp
    timings["hands_adapt_ms"] = (time.perf_counter() - t) * 1000

    return CanonicalMotionState(
        schema_version=CANONICAL_MOTION_STATE_VERSION,
        frame_index=frame_index,
        capture_timestamp=capture_timestamp,
        source_backend="mediapipe_tasks_v1.0.0",
        body=body_pose,
        face=face_expr,
        left_hand=left_hand_pose,
        right_hand=right_hand_pose,
        body_scale=body_scale,
        adapter_timings=timings,
    )
