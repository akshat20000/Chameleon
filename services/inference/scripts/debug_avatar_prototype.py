"""
Phase 2.4A — Non-Photorealistic Motion Retargeting Prototype

Validates the full motion pipeline:

    Camera frame
         ↓
    Motion Extraction (MediaPipe: face, pose, hands, segmentation)
         ↓
    PerformerState
         ↓
    MediaPipe → CanonicalMotionState adapter
         ↓
    CanonicalMotionState
         ↓
    Debug Avatar Renderer (2D stick figure — no 3D engine required)
         ↓
    debug_avatar_frames/

The debug avatar is intentionally geometrically simple. It uses OpenCV 2D
drawing primitives (circles for joints, lines for bones). Visual realism is
irrelevant at this stage. The sole question being answered is:

    "Does the avatar coherently follow the performer's motion?"

Usage
-----
    python services/inference/scripts/debug_avatar_prototype.py [--image PATH]

Output (test_data/phase2_motion_retargeting/)
---------------------------------------------
    debug_avatar_overlay.png     — skeleton drawn on source image
    debug_avatar_clean.png       — skeleton on black background
    side_by_side.png             — source | overlay | clean, annotated
    canonical_state.json         — serialized CanonicalMotionState summary
    metrics.json                 — motion quality metrics + per-stage latency
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
SERVICES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SERVICES_DIR.parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.motion.canonical_state import CanonicalMotionState, BodyPose, JointState
from app.motion.mediapipe_adapter import adapt_performer_state

MODELS_DIR  = PROJECT_ROOT / "services" / "inference" / "models"
TEST_IMAGE  = PROJECT_ROOT / "test_data" / "2face_validation.png"
OUT_DIR     = PROJECT_ROOT / "test_data" / "phase2_motion_retargeting"

N_BENCHMARK_ITERS = 30

# ── avatar appearance constants ───────────────────────────────────────────────
JOINT_RADIUS   = 7
HEAD_RADIUS    = 20
BONE_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Color palette (BGR)
COLOR_SPINE       = (255, 255, 100)
COLOR_LEFT_ARM    = (80,  200, 255)   # blue-ish  (performer's left)
COLOR_RIGHT_ARM   = (100, 255, 100)   # green-ish (performer's right)
COLOR_LEFT_LEG    = (180, 130, 255)   # purple
COLOR_RIGHT_LEG   = (255, 160, 80)    # orange
COLOR_HEAD        = (255, 255, 255)
COLOR_HANDS       = (50,  220, 220)
COLOR_FACE_MARKER = (0,   180, 255)
COLOR_CONFIDENCE  = (80,  80,  80)

# ── bone definitions: (parent_attr, child_attr, color) ───────────────────────
BODY_BONES = [
    # Spine chain
    ("pelvis",       "left_hip",       COLOR_SPINE),
    ("pelvis",       "right_hip",      COLOR_SPINE),
    ("pelvis",       "spine_mid",      COLOR_SPINE),
    ("spine_mid",    "chest",          COLOR_SPINE),
    ("chest",        "neck",           COLOR_SPINE),
    ("neck",         "head",           COLOR_HEAD),
    # Left arm  (performer's left = appears on camera right for front-facing)
    ("chest",        "left_shoulder",  COLOR_LEFT_ARM),
    ("left_shoulder","left_elbow",     COLOR_LEFT_ARM),
    ("left_elbow",   "left_wrist",     COLOR_LEFT_ARM),
    # Right arm
    ("chest",        "right_shoulder", COLOR_RIGHT_ARM),
    ("right_shoulder","right_elbow",   COLOR_RIGHT_ARM),
    ("right_elbow",  "right_wrist",    COLOR_RIGHT_ARM),
    # Left leg
    ("left_hip",     "left_knee",      COLOR_LEFT_LEG),
    ("left_knee",    "left_ankle",     COLOR_LEFT_LEG),
    # Right leg
    ("right_hip",    "right_knee",     COLOR_RIGHT_LEG),
    ("right_knee",   "right_ankle",    COLOR_RIGHT_LEG),
]


# ==============================================================================
# PerformerState builder (re-uses motion_extraction_prototype logic)
# ==============================================================================

def _estimate_body_scale(lms: np.ndarray) -> float:
    """
    Estimate body height in MediaPipe world-landmark units (approx. meters).

    Priority order:
    1. |shoulder - ankle| (full body visible)
    2. |shoulder - hip| * 2  (lower body missing)
    3. |left_shoulder - right_shoulder| * 3.5 (rough torso-only estimate)
    Returns 0.0 only if nothing is detectable.
    """
    ls_conf = _lm_vis(lms, _LM_LEFT_SHOULDER)
    rs_conf = _lm_vis(lms, _LM_RIGHT_SHOULDER)
    la_conf = _lm_vis(lms, _LM_LEFT_ANKLE)
    lh_conf = _lm_vis(lms, _LM_LEFT_HIP)
    rh_conf = _lm_vis(lms, _LM_RIGHT_HIP)

    # Option 1: shoulder-to-ankle
    if ls_conf >= CONFIDENCE_THRESHOLD and la_conf >= CONFIDENCE_THRESHOLD:
        shoulder = _lm_pos(lms, _LM_LEFT_SHOULDER)
        ankle    = _lm_pos(lms, _LM_LEFT_ANKLE)
        return float(np.linalg.norm(shoulder - ankle))

    # Option 2: shoulder-to-hip * 2
    if ls_conf >= CONFIDENCE_THRESHOLD and lh_conf >= CONFIDENCE_THRESHOLD:
        shoulder = _lm_pos(lms, _LM_LEFT_SHOULDER)
        hip      = _lm_pos(lms, _LM_LEFT_HIP)
        return float(np.linalg.norm(shoulder - hip)) * 2.0

    # Option 3: shoulder-to-shoulder * 3.5 (torso-only estimate)
    if ls_conf >= CONFIDENCE_THRESHOLD and rs_conf >= CONFIDENCE_THRESHOLD:
        ls_pos = _lm_pos(lms, _LM_LEFT_SHOULDER)
        rs_pos = _lm_pos(lms, _LM_RIGHT_SHOULDER)
        shoulder_width = float(np.linalg.norm(ls_pos - rs_pos))
        return shoulder_width * 3.5  # empirical: shoulder span ~28% of body height

    return 0.0

def _build_performer_state_from_mediapipe(bgr: np.ndarray) -> Tuple[object, dict]:
    """
    Run MediaPipe and assemble a minimal PerformerState-compatible object
    that the adapter can consume, without importing the full production app.

    Returns (performer_state_like, timings_ms).
    This is a lightweight shim — the real PerformerState uses tracked faces.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as tasks
    from mediapipe.tasks.python import vision

    from app.pipeline.result import (
        BoundingBox, FaceDetection, TrackedFace, LandmarkResult, PoseResult,
    )
    from app.landmarks.landmarker import MediaPipeLandmarker
    from app.segmentation.segmenter import MediaPipeSegmenter

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timings = {}

    # ── Face Landmarker ───────────────────────────────────────────────────────
    t = time.perf_counter()
    fl_opts = vision.FaceLandmarkerOptions(
        base_options=tasks.BaseOptions(
            model_asset_path=str(MODELS_DIR / "face_landmarker.task")
        ),
        num_faces=4,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    fl = vision.FaceLandmarker.create_from_options(fl_opts)
    face_result = fl.detect(mp_img)
    timings["face_ms"] = (time.perf_counter() - t) * 1000

    # ── Pose Landmarker ───────────────────────────────────────────────────────
    t = time.perf_counter()
    pl_opts = vision.PoseLandmarkerOptions(
        base_options=tasks.BaseOptions(
            model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")
        ),
        num_poses=2,
        output_segmentation_masks=False,
    )
    pl = vision.PoseLandmarker.create_from_options(pl_opts)
    pose_result = pl.detect(mp_img)
    timings["pose_ms"] = (time.perf_counter() - t) * 1000

    # ── Hand Landmarker ───────────────────────────────────────────────────────
    t = time.perf_counter()
    hl_opts = vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(
            model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
        ),
        num_hands=2,
    )
    hl = vision.HandLandmarker.create_from_options(hl_opts)
    hand_result = hl.detect(mp_img)
    timings["hands_ms"] = (time.perf_counter() - t) * 1000

    # ── Assemble PerformerState-compatible shim ───────────────────────────────
    # We build a plain namespace object matching the fields the adapter reads.

    class _ShimBodyPose:
        def __init__(self):
            self.landmarks_3d = None

    class _ShimHandState:
        def __init__(self):
            self.landmarks_3d = None
            self.handedness = "Right"

    class _ShimHeadRotation:
        def __init__(self, mat):
            self.transformation_matrix = mat

    class _ShimFaceState:
        def __init__(self, blendshapes, mat, confidence):
            self.blendshapes = blendshapes
            self.head_rotation = _ShimHeadRotation(mat)
            self.confidence = confidence
            self.track_id = 0

    class _ShimPerformerState:
        def __init__(self):
            self.faces = []
            self.left_hand = None
            self.right_hand = None
            self.body = _ShimBodyPose()
            self.segmentation = None
            self.primary_face_track_id = 0

        @property
        def primary_face(self):
            return self.faces[0] if self.faces else None

    state = _ShimPerformerState()

    # Body pose (world landmarks = 3D in metres, hip-centred)
    if pose_result.pose_world_landmarks:
        lms = pose_result.pose_world_landmarks[0]
        arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms], dtype=np.float32)
        state.body.landmarks_3d = arr

    # Face
    if face_result.face_landmarks and face_result.face_blendshapes and face_result.facial_transformation_matrixes:
        bs_dict = {
            b.category_name: float(b.score)
            for b in face_result.face_blendshapes[0]
        }
        mat = np.array(face_result.facial_transformation_matrixes[0], dtype=np.float32)
        state.faces = [_ShimFaceState(bs_dict, mat, 1.0)]

    # Hands
    if hand_result.hand_landmarks and hand_result.hand_world_landmarks:
        for hand_lms_world, handedness_list in zip(
            hand_result.hand_world_landmarks, hand_result.handedness
        ):
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms_world], dtype=np.float32
            )
            h_name = handedness_list[0].category_name
            shim = _ShimHandState()
            shim.landmarks_3d = arr
            shim.handedness = h_name
            if h_name == "Left":
                state.left_hand = shim
            else:
                state.right_hand = shim

    return state, timings


# ==============================================================================
# Canonical-space → 2D projection
# ==============================================================================

def _project_joint_to_screen(
    joint: Optional[JointState],
    canvas_h: int,
    canvas_w: int,
    origin_px: Tuple[int, int],
    scale_px: float,
) -> Optional[Tuple[int, int]]:
    """
    Project a canonical 3D joint position to 2D screen coordinates.

    In canonical space:
        +X = right, +Y = up, +Z = toward camera.
    In image space:
        +x = right, +y = down (origin top-left).

    We project X → image_x, Y → image_y (flipped), ignore Z.
    The origin on screen is at origin_px (pelvis projected position).
    """
    if joint is None or not joint.is_visible:
        return None
    # X right, Y up in canonical → X right, Y down in image
    ix = int(origin_px[0] + joint.position[0] * scale_px)
    iy = int(origin_px[1] - joint.position[1] * scale_px)
    return (
        max(0, min(canvas_w - 1, ix)),
        max(0, min(canvas_h - 1, iy)),
    )


# ==============================================================================
# Debug avatar renderer
# ==============================================================================

def render_debug_avatar(
    motion: CanonicalMotionState,
    canvas: np.ndarray,
    person_idx: int = 0,
    scale_px: float = 180.0,
    origin_offset_x: int = 0,
) -> dict:
    """
    Render a 2D wireframe skeleton of the CanonicalMotionState onto canvas.

    Parameters
    ----------
    motion : CanonicalMotionState
    canvas : np.ndarray, BGR
        Image to draw onto (modified in-place).
    person_idx : int
        Which detected person (0-indexed), used to offset origin.
    scale_px : float
        Pixels per body-height unit. Tune to match the on-screen body size.
    origin_offset_x : int
        Horizontal pixel offset for the pelvis origin (for multiple people).

    Returns
    -------
    dict of joint pixel positions for metric calculation.
    """
    h, w = canvas.shape[:2]

    # Place pelvis origin at lower-middle of the canvas (or offset for multi-person)
    origin_px = (w // 4 + origin_offset_x, int(h * 0.75))

    joints = motion.body.all_joints()
    joint_pixels: Dict[str, Optional[Tuple[int, int]]] = {}

    for name, joint in joints.items():
        joint_pixels[name] = _project_joint_to_screen(
            joint, h, w, origin_px, scale_px
        )

    # ── Draw bones ────────────────────────────────────────────────────────────
    for parent_name, child_name, color in BODY_BONES:
        pp = joint_pixels.get(parent_name)
        cp = joint_pixels.get(child_name)
        if pp is not None and cp is not None:
            cv2.line(canvas, pp, cp, color, BONE_THICKNESS, cv2.LINE_AA)

    # ── Draw joint circles ────────────────────────────────────────────────────
    for name, px in joint_pixels.items():
        if px is None:
            continue
        joint = joints[name]
        color = COLOR_HEAD if name in ("head", "neck") else COLOR_SPINE
        if "left_arm" in name or "left_shoulder" in name or "left_elbow" in name or "left_wrist" in name:
            color = COLOR_LEFT_ARM
        if "right_shoulder" in name or "right_elbow" in name or "right_wrist" in name:
            color = COLOR_RIGHT_ARM
        if "left_knee" in name or "left_ankle" in name or "left_hip" in name:
            color = COLOR_LEFT_LEG
        if "right_knee" in name or "right_ankle" in name or "right_hip" in name:
            color = COLOR_RIGHT_LEG

        radius = HEAD_RADIUS if name == "head" else JOINT_RADIUS
        conf = joint.confidence if joint is not None else 0.0
        alpha_color = tuple(int(c * conf) for c in color)
        cv2.circle(canvas, px, radius, alpha_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, px, radius, (255, 255, 255), 1, cv2.LINE_AA)

        # Confidence label for key joints
        if name in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
                    "head", "left_ankle", "right_ankle"):
            label = f"{name.split('_')[-1][:3]} {conf:.2f}"
            cv2.putText(canvas, label, (px[0] + 8, px[1] - 4),
                        FONT, 0.33, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Draw head direction arrow ──────────────────────────────────────────────
    if motion.has_face and joint_pixels.get("head"):
        head_px = joint_pixels["head"]
        R = motion.face.head_rotation
        fwd = R[:, 2]   # Z column = forward direction of head
        arrow_len = 40
        arrow_end = (
            int(head_px[0] + fwd[0] * arrow_len),
            int(head_px[1] - fwd[1] * arrow_len),
        )
        cv2.arrowedLine(canvas, head_px, arrow_end, COLOR_FACE_MARKER, 2,
                        tipLength=0.3, line_type=cv2.LINE_AA)

    # ── Draw hands (wrist markers) ─────────────────────────────────────────────
    for hand_pose, side_label in [
        (motion.left_hand, "L"), (motion.right_hand, "R")
    ]:
        if hand_pose is not None:
            wrist_px = joint_pixels.get(
                "left_wrist" if side_label == "L" else "right_wrist"
            )
            if wrist_px:
                cv2.circle(canvas, wrist_px, 12, COLOR_HANDS, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"Hand-{side_label}", (wrist_px[0] + 14, wrist_px[1]),
                            FONT, 0.4, COLOR_HANDS, 1, cv2.LINE_AA)

    return joint_pixels


def render_debug_legend(canvas: np.ndarray) -> None:
    """Draw color key in the bottom-left corner."""
    entries = [
        ("Spine / Head",  COLOR_SPINE),
        ("Left Arm",      COLOR_LEFT_ARM),
        ("Right Arm",     COLOR_RIGHT_ARM),
        ("Left Leg",      COLOR_LEFT_LEG),
        ("Right Leg",     COLOR_RIGHT_LEG),
        ("Head direction", COLOR_FACE_MARKER),
    ]
    h, w = canvas.shape[:2]
    for i, (label, color) in enumerate(entries):
        y = h - 20 - i * 20
        cv2.circle(canvas, (14, y), 6, color, -1)
        cv2.putText(canvas, label, (24, y + 4), FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


# ==============================================================================
# Motion quality metrics
# ==============================================================================

def compute_metrics(motion: CanonicalMotionState) -> dict:
    """
    Compute motion quality metrics from a single CanonicalMotionState.

    For a single-frame prototype, temporal metrics (jitter, velocity) cannot
    be computed. Those limitations are documented explicitly.
    """
    joints = motion.body.all_joints()
    total_joints = len(joints)
    visible_joints = sum(1 for j in joints.values() if j is not None and j.is_visible)
    missing_joints = [
        name for name, j in joints.items()
        if j is None or not j.is_visible
    ]

    # Left/right inversion check
    ls = motion.body.left_shoulder
    rs = motion.body.right_shoulder
    inversion_flag = False
    if ls is not None and rs is not None and ls.is_visible and rs.is_visible:
        inversion_flag = bool(ls.position[0] > rs.position[0] + 0.05)

    # NaN check across all joint positions
    nan_joints = []
    for name, j in joints.items():
        if j is not None and np.any(np.isnan(j.position)):
            nan_joints.append(name)

    # Rotation orthogonality check
    non_orthogonal_joints = []
    for name, j in joints.items():
        if j is not None and j.rotation is not None:
            det = float(np.linalg.det(j.rotation))
            if abs(det - 1.0) > 0.01:
                non_orthogonal_joints.append({"joint": name, "det": round(det, 4)})

    # Validation
    validation_errors = motion.validate()

    return {
        "tracking_quality": motion.tracking_quality,
        "visible_joints": visible_joints,
        "total_joints": total_joints,
        "tracking_rate": round(visible_joints / total_joints, 3),
        "missing_joints": missing_joints,
        "left_right_inversion_detected": inversion_flag,
        "nan_joint_positions": nan_joints,
        "non_orthogonal_rotations": non_orthogonal_joints,
        "has_face": motion.has_face,
        "has_left_hand": motion.has_left_hand,
        "has_right_hand": motion.has_right_hand,
        "body_scale_m": round(motion.body_scale, 4),
        "blendshape_count": len(motion.face.blendshapes) if motion.face else 0,
        "schema_version": motion.schema_version,
        "source_backend": motion.source_backend,
        "validation_errors": validation_errors,
        "NOTE_temporal_metrics": (
            "Temporal metrics (jitter, velocity, dropped frames) require "
            "multi-frame video input. Cannot be computed from a single static image."
        ),
    }


def serialize_motion_state(motion: CanonicalMotionState) -> dict:
    """Serialize CanonicalMotionState to a JSON-compatible dict."""
    def _joint_dict(j: Optional[object]) -> Optional[dict]:
        if j is None:
            return None
        return {
            "position": j.position.tolist(),
            "confidence": round(j.confidence, 3),
            "is_visible": bool(j.is_visible),
            "has_rotation": j.rotation is not None,
        }

    joints_out = {}
    for name, joint in motion.body.all_joints().items():
        joints_out[name] = _joint_dict(joint)

    face_out = None
    if motion.face:
        face_out = {
            "head_rotation_det": round(float(np.linalg.det(motion.face.head_rotation)), 4),
            "eye_open_left": round(motion.face.eye_open_left, 3),
            "eye_open_right": round(motion.face.eye_open_right, 3),
            "jaw_open": round(motion.face.jaw_open, 3),
            "confidence": round(motion.face.confidence, 3),
            "blendshape_count": len(motion.face.blendshapes),
            "sample_blendshapes": dict(
                list({k: round(v, 3) for k, v in motion.face.blendshapes.items()}.items())[:8]
            ),
        }

    return {
        "schema_version": motion.schema_version,
        "frame_index": motion.frame_index,
        "source_backend": motion.source_backend,
        "tracking_quality": motion.tracking_quality,
        "body_scale_m": round(motion.body_scale, 4),
        "body_joints": joints_out,
        "face": face_out,
        "has_left_hand": motion.has_left_hand,
        "has_right_hand": motion.has_right_hand,
        "adapter_timings_ms": {k: round(v, 2) for k, v in motion.adapter_timings.items()},
    }


# ==============================================================================
# Benchmark loop
# ==============================================================================

def benchmark_pipeline(bgr: np.ndarray) -> Tuple[CanonicalMotionState, dict]:
    """Run extraction + adaptation N times and return last state + aggregated timings."""
    import time

    print(f"Benchmarking pipeline ({N_BENCHMARK_ITERS} iterations)...")

    performer_state_shim, extraction_timings = _build_performer_state_from_mediapipe(bgr)

    adapt_times = []
    canonical = None
    t0 = time.perf_counter()
    for i in range(N_BENCHMARK_ITERS):
        t = time.perf_counter()
        canonical = adapt_performer_state(
            performer_state_shim,
            frame_index=i,
            capture_timestamp=t0 + i * 0.033,
        )
        adapt_times.append((time.perf_counter() - t) * 1000)

    def _stats(times: list) -> dict:
        a = np.array(times)
        return {
            "mean_ms": round(float(np.mean(a)), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "min_ms": round(float(np.min(a)), 2),
            "max_ms": round(float(np.max(a)), 2),
        }

    timings_report = {
        "mediapipe_extraction": {k: round(v, 2) for k, v in extraction_timings.items()},
        "adapter_adapt_performer_state": _stats(adapt_times),
    }

    return canonical, timings_report


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2.4A debug avatar prototype")
    parser.add_argument("--image", default=str(TEST_IMAGE), help="Input image path")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Phase 2.4A — Debug Avatar Prototype")
    print("  PerformerState -> CanonicalMotionState -> 2D Skeleton Renderer")
    print("=" * 72)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Input image not found: {args.image}")
    h, w = bgr.shape[:2]
    print(f"Loaded: {args.image}  ({w}×{h})")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    print("\nStep 1: Extracting PerformerState from MediaPipe...")
    performer_state_shim, extraction_timings = _build_performer_state_from_mediapipe(bgr)

    print("Step 2: Adapting to CanonicalMotionState...")
    import time
    t = time.perf_counter()
    canonical = adapt_performer_state(
        performer_state_shim,
        frame_index=0,
        capture_timestamp=time.time(),
    )
    adapt_ms = (time.perf_counter() - t) * 1000

    print(f"  Adapter latency: {adapt_ms:.2f} ms")
    print(f"  Tracking quality: {canonical.tracking_quality}")
    print(f"  Visible joints: {canonical.body.visible_joint_count()} / {len(canonical.body.all_joints())}")
    print(f"  Body scale: {canonical.body_scale:.3f} m")
    print(f"  Face detected: {canonical.has_face}")
    print(f"  Left hand: {canonical.has_left_hand}")
    print(f"  Right hand: {canonical.has_right_hand}")

    # Validation
    errors = canonical.validate()
    if errors:
        print(f"\n  VALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    FAIL: {e}")
    else:
        print("  Validation: PASS")

    # ── Render debug avatar ───────────────────────────────────────────────────
    print("\nStep 3: Rendering debug avatar...")

    # Estimate per-person scale from image (body height in pixels approximation)
    scale_px = h * 0.5   # rough: body fills ~50% of frame height

    # Overlay on source image
    overlay_canvas = bgr.copy()
    render_debug_legend(overlay_canvas)
    pixel_map = render_debug_avatar(
        canonical, overlay_canvas, person_idx=0, scale_px=scale_px,
        origin_offset_x=w // 4,
    )
    cv2.putText(overlay_canvas, "CanonicalMotionState — Debug Avatar (Overlay)",
                (10, 20), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Clean black canvas
    clean_canvas = np.zeros_like(bgr)
    render_debug_legend(clean_canvas)
    render_debug_avatar(
        canonical, clean_canvas, person_idx=0, scale_px=scale_px,
        origin_offset_x=w // 2,
    )
    cv2.putText(clean_canvas, "CanonicalMotionState — Debug Avatar (Clean)",
                (10, 20), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Quality HUD on clean canvas
    hud = [
        f"Quality: {canonical.tracking_quality}",
        f"Joints: {canonical.body.visible_joint_count()}/{len(canonical.body.all_joints())}",
        f"Face: {'yes (' + str(len(canonical.face.blendshapes)) + ' blendshapes)' if canonical.has_face else 'no'}",
        f"Hands: {'L ' if canonical.has_left_hand else ''}{'R' if canonical.has_right_hand else ''}",
        f"Validation: {'PASS' if not errors else str(len(errors)) + ' ERRORS'}",
        f"Backend: {canonical.source_backend}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(clean_canvas, line, (10, h - 140 + i * 20),
                    FONT, 0.45, (180, 255, 180), 1, cv2.LINE_AA)

    # Side-by-side
    # Resize source for panel
    panel_source = cv2.resize(bgr, (w // 2, h // 2))
    panel_overlay = cv2.resize(overlay_canvas, (w // 2, h // 2))
    panel_clean = cv2.resize(clean_canvas, (w // 2, h // 2))
    empty = np.zeros((h // 2, w // 2, 3), dtype=np.uint8)
    top_row = np.hstack([panel_source, panel_overlay])
    bot_row = np.hstack([panel_clean, empty])
    side_by_side = np.vstack([top_row, bot_row])

    # Labels on side-by-side
    for (label, col) in [
        ("Source", (0, 0)),
        ("Overlay", (w // 2, 0)),
        ("Clean Avatar", (0, h // 2)),
    ]:
        cv2.putText(side_by_side, label, (col[0] + 6, col[1] + 20),
                    FONT, 0.6, (255, 220, 80), 2, cv2.LINE_AA)

    # ── Save outputs ─────────────────────────────────────────────────────────
    overlay_path     = str(OUT_DIR / "debug_avatar_overlay.png")
    clean_path       = str(OUT_DIR / "debug_avatar_clean.png")
    sbs_path         = str(OUT_DIR / "side_by_side.png")
    state_path       = str(OUT_DIR / "canonical_state.json")
    metrics_path     = str(OUT_DIR / "metrics.json")

    cv2.imwrite(overlay_path, overlay_canvas)
    cv2.imwrite(clean_path, clean_canvas)
    cv2.imwrite(sbs_path, side_by_side)

    state_dict = serialize_motion_state(canonical)
    with open(state_path, "w") as f:
        json.dump(state_dict, f, indent=2)

    metrics = compute_metrics(canonical)
    _, bench_timings = benchmark_pipeline(bgr)
    metrics["pipeline_benchmark"] = bench_timings
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  MOTION QUALITY METRICS")
    print("=" * 72)
    print(f"  Tracking quality:        {metrics['tracking_quality']}")
    print(f"  Visible joints:          {metrics['visible_joints']} / {metrics['total_joints']}  ({metrics['tracking_rate']*100:.0f}%)")
    inv = metrics['left_right_inversion_detected']
    print(f"  Left/right inversion:    {'WARN (see validation errors)' if inv else 'PASS'}")
    print(f"  NaN positions:           {metrics['nan_joint_positions'] or 'none - PASS'}")
    print(f"  Non-orthogonal rotations:{metrics['non_orthogonal_rotations'] or 'none - PASS'}")
    print(f"  Validation errors:       {metrics['validation_errors'] or 'none - PASS'}")
    print(f"  Blendshapes:             {metrics['blendshape_count']}")
    if metrics["missing_joints"]:
        print(f"  Missing joints:          {', '.join(metrics['missing_joints'])}")

    print("\n" + "=" * 72)
    print("  ADAPTER LATENCY (single pass)")
    print("=" * 72)
    for k, v in canonical.adapter_timings.items():
        print(f"  {k:<30} {v:.2f} ms")

    print("\n  Output files:")
    for p in [overlay_path, clean_path, sbs_path, state_path, metrics_path]:
        print(f"    {p}")


if __name__ == "__main__":
    main()
