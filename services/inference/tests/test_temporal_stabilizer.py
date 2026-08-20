import sys
from pathlib import Path

import numpy as np
import pytest

SERVICES_DIR = Path(__file__).resolve().parent.parent
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from app.motion.canonical_state import (
    BodyPose,
    CanonicalMotionState,
    JointState,
    CANONICAL_MOTION_STATE_VERSION,
)
from app.motion.temporal_stabilizer import (
    OneEuroFilter1D,
    OneEuroFilter3D,
    SO3QuaternionFilter,
    TemporalAssociationPolicy,
    TemporalStabilizer,
    matrix_to_quaternion,
    quaternion_to_matrix,
    slerp,
)


def _make_dummy_state(frame_idx: int, timestamp: float, head_y: float = 0.55) -> CanonicalMotionState:
    body = BodyPose(
        pelvis=JointState(position=np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        chest=JointState(position=np.array([0.0, 0.35, 0.0], dtype=np.float32)),
        neck=JointState(position=np.array([0.0, 0.45, 0.0], dtype=np.float32)),
        head=JointState(position=np.array([0.0, head_y, 0.0], dtype=np.float32)),
        left_shoulder=JointState(position=np.array([0.15, 0.35, 0.0], dtype=np.float32)),
        right_shoulder=JointState(position=np.array([-0.15, 0.35, 0.0], dtype=np.float32)),
    )
    return CanonicalMotionState(
        schema_version=CANONICAL_MOTION_STATE_VERSION,
        frame_index=frame_idx,
        capture_timestamp=timestamp,
        source_backend="test",
        body=body,
        body_scale=1.75,
    )


def test_one_euro_filter_1d_smoothing():
    f = OneEuroFilter1D(min_cutoff=1.0, beta=0.005)
    # Steady signal with small high-frequency noise
    t0 = 0.0
    val_clean = 10.0
    noisy_vals = [val_clean + (0.5 if i % 2 == 0 else -0.5) for i in range(10)]

    filtered_vals = []
    for i, v in enumerate(noisy_vals):
        filtered_vals.append(f.filter(v, t0 + i * 0.033))

    # Variance of filtered signal should be substantially smaller than raw noisy signal
    raw_var = np.var(noisy_vals)
    filt_var = np.var(filtered_vals[3:])
    assert filt_var < raw_var * 0.3, f"1€ filter did not reduce jitter: {filt_var} >= {raw_var * 0.3}"


def test_quaternion_so3_conversion():
    # Test round-trip conversion matrix -> quaternion -> matrix
    R_eye = np.eye(3, dtype=np.float32)
    q_eye = matrix_to_quaternion(R_eye)
    R_rec = quaternion_to_matrix(q_eye)
    assert np.allclose(R_eye, R_rec, atol=1e-5), "Identity matrix round-trip failed"

    # Test 90-deg Z rotation matrix
    R_z90 = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float32)
    q_z90 = matrix_to_quaternion(R_z90)
    R_rec_90 = quaternion_to_matrix(q_z90)
    assert np.allclose(R_z90, R_rec_90, atol=1e-4), "90-deg Z rotation matrix round-trip failed"
    assert abs(np.linalg.det(R_rec_90) - 1.0) < 1e-5, "SO3 determinant must be +1.0"


def test_quaternion_slerp():
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 0 deg
    q2 = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float32)  # 90 deg Z
    q_half = slerp(q1, q2, 0.5)
    # At t=0.5, rotation should be 45 deg Z (w = cos(22.5 deg) = 0.9238, z = sin(22.5 deg) = 0.3827)
    assert abs(q_half[0] - 0.9238) < 1e-3
    assert abs(q_half[3] - 0.3827) < 1e-3


def test_temporal_association_policy_side_swap():
    policy = TemporalAssociationPolicy()

    # Frame 1: valid upright standing
    state1 = _make_dummy_state(0, 0.0)
    body1, swap1 = policy.process(state1.body)
    assert not swap1

    # Frame 2: simulate severe tracker L/R shoulder swap (right shoulder x > left shoulder x)
    state2 = _make_dummy_state(1, 0.033)
    state2.body.left_shoulder.position = np.array([-0.15, 0.35, 0.0], dtype=np.float32)
    state2.body.right_shoulder.position = np.array([0.15, 0.35, 0.0], dtype=np.float32)

    body2, swap2 = policy.process(state2.body)
    assert swap2, "Failed to detect anatomical L/R shoulder swap"
    # Corrected body must hold previous valid left shoulder position
    assert body2.left_shoulder.position[0] > 0.0


def test_temporal_stabilizer_end_to_end():
    stabilizer = TemporalStabilizer()

    s1 = _make_dummy_state(0, 0.0)
    st1 = stabilizer.process(s1)
    assert st1.frame_index == 0
    assert "stabilized" in st1.source_backend

    # Telemetry should be populated
    assert stabilizer.last_telemetry is not None
    assert stabilizer.last_telemetry.latency_ms >= 0.0
