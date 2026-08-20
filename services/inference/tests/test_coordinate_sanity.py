"""
Coordinate System Sanity & Mathematical Contract Test

Validates the full 3D canonical coordinate system and 2D image-space projection contract:

1. Canonical Space Contract (+Y is UPWARD):
       head.y > neck.y > chest.y > pelvis.y (0.0)
       pelvis.y > left_knee.y > left_ankle.y
       pelvis.y > right_knee.y > right_ankle.y

2. Screen Image Space Contract (+y is DOWNWARD):
       pixel_y(head) < pixel_y(neck) < pixel_y(chest) < pixel_y(pelvis)
       pixel_y(pelvis) < pixel_y(knee) < pixel_y(ankle)

3. Left/Right Camera-Space Contract (+X is CAMERA-RIGHT):
       left_shoulder.x > right_shoulder.x (for front-facing standing subject)
       pixel_x(left_shoulder) > pixel_x(right_shoulder)
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Add services/inference to sys.path
SERVICES_DIR = Path(__file__).resolve().parent.parent
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from app.motion.canonical_state import (
    BodyPose,
    CanonicalMotionState,
    JointState,
    CANONICAL_MOTION_STATE_VERSION,
)
from scripts.validate_full_body import render_skeleton


def _make_standing_canonical_state() -> CanonicalMotionState:
    """Construct a clean standing pose CanonicalMotionState with +Y UPWARD."""
    pelvis = JointState(position=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    chest  = JointState(position=np.array([0.0, 0.35, 0.0], dtype=np.float32))
    neck   = JointState(position=np.array([0.0, 0.45, 0.0], dtype=np.float32))
    head   = JointState(position=np.array([0.0, 0.55, 0.0], dtype=np.float32))

    l_shoulder = JointState(position=np.array([0.15, 0.35, 0.0], dtype=np.float32))
    r_shoulder = JointState(position=np.array([-0.15, 0.35, 0.0], dtype=np.float32))
    l_elbow    = JointState(position=np.array([0.18, 0.18, 0.0], dtype=np.float32))
    r_elbow    = JointState(position=np.array([-0.18, 0.18, 0.0], dtype=np.float32))
    l_wrist    = JointState(position=np.array([0.20, 0.00, 0.0], dtype=np.float32))
    r_wrist    = JointState(position=np.array([-0.20, 0.00, 0.0], dtype=np.float32))

    l_hip   = JointState(position=np.array([0.08, 0.00, 0.0], dtype=np.float32))
    r_hip   = JointState(position=np.array([-0.08, 0.00, 0.0], dtype=np.float32))
    l_knee  = JointState(position=np.array([0.09, -0.30, 0.0], dtype=np.float32))
    r_knee  = JointState(position=np.array([-0.09, -0.30, 0.0], dtype=np.float32))
    l_ankle = JointState(position=np.array([0.10, -0.60, 0.0], dtype=np.float32))
    r_ankle = JointState(position=np.array([-0.10, -0.60, 0.0], dtype=np.float32))

    body = BodyPose(
        pelvis=pelvis,
        chest=chest,
        neck=neck,
        head=head,
        left_shoulder=l_shoulder,
        right_shoulder=r_shoulder,
        left_elbow=l_elbow,
        right_elbow=r_elbow,
        left_wrist=l_wrist,
        right_wrist=r_wrist,
        left_hip=l_hip,
        right_hip=r_hip,
        left_knee=l_knee,
        right_knee=r_knee,
        left_ankle=l_ankle,
        right_ankle=r_ankle,
    )

    return CanonicalMotionState(
        schema_version=CANONICAL_MOTION_STATE_VERSION,
        frame_index=0,
        capture_timestamp=1.0,
        source_backend="synthetic_test",
        body=body,
        body_scale=1.75,
    )


def test_canonical_space_vertical_hierarchy():
    """Assert 3D CanonicalSpace contract: +Y is UPWARD."""
    motion = _make_standing_canonical_state()
    body = motion.body

    # Validation check must pass with zero errors
    errors = motion.validate()
    assert errors == [], f"CanonicalMotionState validation failed: {errors}"

    # Vertical Y ordering (top to bottom)
    assert body.head.position[1] > body.neck.position[1]
    assert body.neck.position[1] > body.chest.position[1]
    assert body.chest.position[1] > body.pelvis.position[1]

    assert body.pelvis.position[1] > body.left_knee.position[1]
    assert body.left_knee.position[1] > body.left_ankle.position[1]

    assert body.pelvis.position[1] > body.right_knee.position[1]
    assert body.right_knee.position[1] > body.right_ankle.position[1]


def test_canonical_space_horizontal_chirality():
    """Assert 3D CanonicalSpace contract: +X is CAMERA-RIGHT (performer left)."""
    motion = _make_standing_canonical_state()
    body = motion.body

    assert body.left_shoulder.position[0] > body.right_shoulder.position[0]
    assert body.left_hip.position[0] > body.right_hip.position[0]


def test_renderer_image_space_projection_contract():
    """Assert 2D Renderer projection contract: screen +y is DOWNWARD."""
    motion = _make_standing_canonical_state()

    # Create dummy 500x500 BGR canvas
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)

    # Render skeleton onto canvas
    pixel_map = render_skeleton(
        motion,
        canvas,
        scale_px=200.0,
        origin_px=(250, 250),  # Pelvis origin at center (250, 250)
    )

    # All key joints projected
    head_px = pixel_map["head"]
    neck_px = pixel_map["neck"]
    chest_px = pixel_map["chest"]
    pelvis_px = pixel_map["pelvis"]
    l_knee_px = pixel_map["left_knee"]
    l_ankle_px = pixel_map["left_ankle"]

    assert head_px is not None
    assert neck_px is not None
    assert chest_px is not None
    assert pelvis_px is not None
    assert l_knee_px is not None
    assert l_ankle_px is not None

    # In image pixel space: smaller Y is HIGHER on screen
    assert head_px[1] < neck_px[1]
    assert neck_px[1] < chest_px[1]
    assert chest_px[1] < pelvis_px[1]
    assert pelvis_px[1] < l_knee_px[1]
    assert l_knee_px[1] < l_ankle_px[1]

    # In image pixel space: performer left shoulder is on camera right (larger pixel X)
    l_sh_px = pixel_map["left_shoulder"]
    r_sh_px = pixel_map["right_shoulder"]
    assert l_sh_px is not None and r_sh_px is not None
    assert l_sh_px[0] > r_sh_px[0]


def test_inverted_state_rejected_by_validate():
    """Assert that an upside-down state is explicitly rejected by validate()."""
    motion = _make_standing_canonical_state()
    # Invert Y of head (head below pelvis)
    motion.body.head.position[1] = -0.55
    errors = motion.validate()
    assert any("vertical hierarchy inversion" in e for e in errors), "Failed to detect inverted head"
