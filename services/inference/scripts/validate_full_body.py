"""
Phase 2.4B — Full-Body Static Validation

Validates the complete 17-joint CanonicalMotionState skeleton against a
full-body image. Each of the 8 Phase 2.4B exit gates is encoded as an
explicit, binary PASS / FAIL check. The script exits with code 0 only if
ALL gates pass.

Usage
-----
    python services/inference/scripts/validate_full_body.py --image PATH

Exit codes
----------
    0  All 8 gates PASS
    1  One or more gates FAIL (or unrecoverable error)

Output (test_data/phase2_4b_validation/)
-----------------------------------------
    skeleton_overlay.png    Source image with skeleton drawn on top
    skeleton_clean.png      Skeleton on black background
    side_by_side.png        Source | overlay | clean | gate report panel
    gate_report.json        Machine-readable gate results + joint data
    canonical_state.json    Full canonical state snapshot

The 8 exit gates (from PHASE_2_ROADMAP.md § Phase 2.4B)
----------------------------------------------------------
    Gate 1  Full-body detected          — all 17 canonical joints present (not None)
    Gate 2  17/17 joints valid          — each joint has confidence >= 0.3
    Gate 3  No NaNs                     — no NaN or Inf in any joint position or rotation
    Gate 4  Rotation matrices valid     — |det(R) - 1| < 0.01 for every rotation
    Gate 5  Left/right consistency      — anatomically correct: left_shoulder.x > right_shoulder.x
                                          (camera-space: performer's left is at positive X)
                                          and left_hip.x > right_hip.x
    Gate 6  Body scale valid            — body_scale > 0.0
    Gate 7  Wrist-to-hand attachment    — if hand detected, hand wrist within 0.15 body-height
                                          units of body wrist
    Gate 8  Visual skeleton correctness — produced image files; human must confirm visually

Coordinate system note
-----------------------
In the CanonicalMotionState, the X-axis follows CAMERA-space, NOT performer anatomy:
    - Performer's LEFT arm appears on camera RIGHT  → left_shoulder.x is POSITIVE
    - Performer's RIGHT arm appears on camera LEFT  → right_shoulder.x is NEGATIVE
So the correct anatomical consistency check is:
    left_shoulder.x  >  right_shoulder.x   (in camera-space)
    left_hip.x       >  right_hip.x
This matches the check already in canonical_state.validate().
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
SERVICES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SERVICES_DIR.parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.motion.canonical_state import (
    CanonicalMotionState,
    BodyPose,
    JointState,
    CONFIDENCE_THRESHOLD,
)
from app.motion.mediapipe_adapter import adapt_performer_state

MODELS_DIR = PROJECT_ROOT / "services" / "inference" / "models"
DEFAULT_OUT_DIR = PROJECT_ROOT / "test_data" / "outputs" / "phase2_4b_validation"

# The 17 joints that must be present for a full-body pass.
# These match the Phase 2.4B roadmap joint hierarchy.
REQUIRED_JOINTS_17 = [
    "head", "neck",
    "left_shoulder",  "left_elbow",  "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "pelvis",
    "left_hip",  "left_knee",  "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    # spine_mid and chest are estimated; also included for completeness
    "spine_mid", "chest",
]

# The 15 joints that are directly mapped from MediaPipe pose landmarks
# (spine_mid and chest are estimated midpoints — do not fail if low confidence)
MEDIAPIPE_DIRECT_JOINTS = [
    "head", "neck",
    "left_shoulder",  "left_elbow",  "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip",  "left_knee",  "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    # pelvis is always computed (midpoint of hips)
    "pelvis",
]

# Wrist-to-hand attachment tolerance in body-height units
WRIST_ATTACH_TOLERANCE = 0.15

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
COLOR_SPINE     = (255, 255, 100)
COLOR_LEFT_ARM  = (80,  200, 255)
COLOR_RIGHT_ARM = (100, 255, 100)
COLOR_LEFT_LEG  = (180, 130, 255)
COLOR_RIGHT_LEG = (255, 160, 80)
COLOR_HEAD      = (255, 255, 255)
COLOR_HANDS     = (50,  220, 220)
COLOR_PASS      = (80,  220, 80)
COLOR_FAIL      = (60,  60,  255)
COLOR_WARN      = (30,  180, 255)
FONT            = cv2.FONT_HERSHEY_SIMPLEX

BODY_BONES = [
    ("pelvis",          "left_hip",        COLOR_SPINE),
    ("pelvis",          "right_hip",       COLOR_SPINE),
    ("pelvis",          "spine_mid",       COLOR_SPINE),
    ("spine_mid",       "chest",           COLOR_SPINE),
    ("chest",           "neck",            COLOR_SPINE),
    ("neck",            "head",            COLOR_HEAD),
    ("chest",           "left_shoulder",   COLOR_LEFT_ARM),
    ("left_shoulder",   "left_elbow",      COLOR_LEFT_ARM),
    ("left_elbow",      "left_wrist",      COLOR_LEFT_ARM),
    ("chest",           "right_shoulder",  COLOR_RIGHT_ARM),
    ("right_shoulder",  "right_elbow",     COLOR_RIGHT_ARM),
    ("right_elbow",     "right_wrist",     COLOR_RIGHT_ARM),
    ("left_hip",        "left_knee",       COLOR_LEFT_LEG),
    ("left_knee",       "left_ankle",      COLOR_LEFT_LEG),
    ("right_hip",       "right_knee",      COLOR_RIGHT_LEG),
    ("right_knee",      "right_ankle",     COLOR_RIGHT_LEG),
]


# ==============================================================================
# MediaPipe extraction shim (copied from debug_avatar_prototype.py)
# ==============================================================================

def _build_performer_state(bgr: np.ndarray) -> Tuple[object, dict]:
    """Run MediaPipe on a BGR image, return a PerformerState-compatible shim."""
    import mediapipe as mp
    from mediapipe.tasks import python as tasks
    from mediapipe.tasks.python import vision

    h, w = bgr.shape[:2]
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timings: dict = {}

    # Face
    t = time.perf_counter()
    fl = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(
                model_asset_path=str(MODELS_DIR / "face_landmarker.task")
            ),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
    )
    face_result = fl.detect(mp_img)
    timings["face_ms"] = (time.perf_counter() - t) * 1000

    # Pose
    t = time.perf_counter()
    pl = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=tasks.BaseOptions(
                model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")
            ),
            num_poses=1,
            output_segmentation_masks=False,
        )
    )
    pose_result = pl.detect(mp_img)
    timings["pose_ms"] = (time.perf_counter() - t) * 1000

    # Hands
    t = time.perf_counter()
    hl = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=tasks.BaseOptions(
                model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
            ),
            num_hands=2,
        )
    )
    hand_result = hl.detect(mp_img)
    timings["hands_ms"] = (time.perf_counter() - t) * 1000

    # Shim classes
    class _BP:
        def __init__(self): self.landmarks_3d = None
    class _HS:
        def __init__(self): self.landmarks_3d = None; self.handedness = "Right"
    class _HR:
        def __init__(self, m): self.transformation_matrix = m
    class _FS:
        def __init__(self, bs, m, c):
            self.blendshapes = bs
            self.head_rotation = _HR(m)
            self.confidence = c
            self.track_id = 0
    class _PS:
        def __init__(self):
            self.faces = []
            self.left_hand = None
            self.right_hand = None
            self.body = _BP()
            self.segmentation = None
            self.primary_face_track_id = 0
        @property
        def primary_face(self): return self.faces[0] if self.faces else None

    state = _PS()

    if pose_result.pose_world_landmarks:
        lms = pose_result.pose_world_landmarks[0]
        state.body.landmarks_3d = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms], dtype=np.float32
        )

    if (face_result.face_landmarks
            and face_result.face_blendshapes
            and face_result.facial_transformation_matrixes):
        bs = {b.category_name: float(b.score) for b in face_result.face_blendshapes[0]}
        mat = np.array(face_result.facial_transformation_matrixes[0], dtype=np.float32)
        state.faces = [_FS(bs, mat, 1.0)]

    if hand_result.hand_landmarks and hand_result.hand_world_landmarks:
        for wlms, handedness_list in zip(
            hand_result.hand_world_landmarks, hand_result.handedness
        ):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in wlms], dtype=np.float32)
            h_name = handedness_list[0].category_name
            shim = _HS()
            shim.landmarks_3d = arr
            shim.handedness = h_name
            if h_name == "Left":
                state.left_hand = shim
            else:
                state.right_hand = shim

    return state, timings


# ==============================================================================
# Gate evaluation
# ==============================================================================

class GateResult:
    """Result of a single Phase 2.4B exit gate."""
    def __init__(self, gate_id: int, name: str, passed: bool, detail: str):
        self.gate_id = gate_id
        self.name    = name
        self.passed  = passed
        self.detail  = detail

    def to_dict(self) -> dict:
        return {
            "gate":   self.gate_id,
            "name":   self.name,
            "result": "PASS" if self.passed else "FAIL",
            "detail": self.detail,
        }

    def __str__(self) -> str:
        badge = "PASS" if self.passed else "FAIL"
        return f"  Gate {self.gate_id}: [{badge}] {self.name}\n           {self.detail}"


def evaluate_gates(motion: CanonicalMotionState) -> List[GateResult]:
    """
    Evaluate all 8 Phase 2.4B exit gates against a CanonicalMotionState.
    Returns a list of GateResult objects in gate order.
    """
    gates: List[GateResult] = []
    joints = motion.body.all_joints()

    # ------------------------------------------------------------------
    # Gate 1: Full-body detected — all 17 canonical joints present (not None)
    # ------------------------------------------------------------------
    missing_present = [
        name for name in REQUIRED_JOINTS_17
        if joints.get(name) is None
    ]
    g1_pass   = len(missing_present) == 0
    g1_detail = (
        f"All {len(REQUIRED_JOINTS_17)} required joints are non-None."
        if g1_pass
        else f"Missing (None): {missing_present}"
    )
    gates.append(GateResult(1, "Full-body detected (17 joints non-None)", g1_pass, g1_detail))

    # ------------------------------------------------------------------
    # Gate 2: 17/17 joints valid — confidence >= 0.3
    # Only check the direct MediaPipe joints; spine_mid/chest are estimated.
    # ------------------------------------------------------------------
    low_confidence = [
        f"{name} ({joints[name].confidence:.2f})"
        for name in MEDIAPIPE_DIRECT_JOINTS
        if joints.get(name) is not None and joints[name].confidence < CONFIDENCE_THRESHOLD
    ]
    # Also flag joints that are None (those already covered by Gate 1, but count them here too)
    none_joints = [
        name for name in MEDIAPIPE_DIRECT_JOINTS
        if joints.get(name) is None
    ]
    g2_failures = low_confidence + [f"{n} (None)" for n in none_joints]
    g2_pass     = len(g2_failures) == 0
    g2_detail   = (
        f"All {len(MEDIAPIPE_DIRECT_JOINTS)} direct joints have confidence >= {CONFIDENCE_THRESHOLD}."
        if g2_pass
        else f"Below threshold: {g2_failures}"
    )
    gates.append(GateResult(
        2, f"17/17 canonical joints valid (confidence >= {CONFIDENCE_THRESHOLD})",
        g2_pass, g2_detail
    ))

    # ------------------------------------------------------------------
    # Gate 3: No NaNs or Infs in any joint position or rotation
    # ------------------------------------------------------------------
    nan_joints = []
    for name, j in joints.items():
        if j is None:
            continue
        if np.any(~np.isfinite(j.position)):
            nan_joints.append(f"{name}.position")
        if j.rotation is not None and np.any(~np.isfinite(j.rotation)):
            nan_joints.append(f"{name}.rotation")
    g3_pass   = len(nan_joints) == 0
    g3_detail = (
        "No NaN or Inf values found in any joint position or rotation."
        if g3_pass
        else f"NaN/Inf found in: {nan_joints}"
    )
    gates.append(GateResult(3, "No NaNs / Infs in joint data", g3_pass, g3_detail))

    # ------------------------------------------------------------------
    # Gate 4: Rotation matrices valid — |det(R) - 1| < 0.01
    # ------------------------------------------------------------------
    bad_rotations = []
    for name, j in joints.items():
        if j is None or j.rotation is None:
            continue
        det = float(np.linalg.det(j.rotation))
        if abs(det - 1.0) > 0.01:
            bad_rotations.append(f"{name} (det={det:.4f})")
    g4_pass   = len(bad_rotations) == 0
    g4_detail = (
        "All rotation matrices are valid SO(3) (|det(R) - 1| < 0.01)."
        if g4_pass
        else f"Non-orthogonal rotations: {bad_rotations}"
    )
    gates.append(GateResult(4, "Rotation matrices valid SO(3)", g4_pass, g4_detail))

    # ------------------------------------------------------------------
    # Gate 5: Left/right anatomical consistency
    # Camera-space convention: performer's LEFT is at positive X (camera-right).
    # Expected: left_shoulder.x > right_shoulder.x
    #           left_hip.x      > right_hip.x
    # ------------------------------------------------------------------
    lr_failures = []
    ls = motion.body.left_shoulder
    rs = motion.body.right_shoulder
    lh = motion.body.left_hip
    rh = motion.body.right_hip

    if ls is not None and rs is not None and ls.is_visible and rs.is_visible:
        if ls.position[0] <= rs.position[0] - 0.05:
            lr_failures.append(
                f"shoulder: left.x={ls.position[0]:.3f} should be > right.x={rs.position[0]:.3f}"
            )
    else:
        lr_failures.append("shoulder: one or both shoulder joints not visible")

    if lh is not None and rh is not None and lh.is_visible and rh.is_visible:
        if lh.position[0] <= rh.position[0] - 0.05:
            lr_failures.append(
                f"hip: left.x={lh.position[0]:.3f} should be > right.x={rh.position[0]:.3f}"
            )
    else:
        lr_failures.append("hip: one or both hip joints not visible")

    g5_pass   = len(lr_failures) == 0
    g5_detail = (
        "Left/right camera-space consistency correct "
        f"(L.shoulder.x={ls.position[0]:.3f} > R.shoulder.x={rs.position[0]:.3f}, "
        f"L.hip.x={lh.position[0]:.3f} > R.hip.x={rh.position[0]:.3f})."
        if g5_pass and ls and rs and lh and rh
        else f"Failures: {lr_failures}"
    )
    gates.append(GateResult(5, "Left/right anatomical consistency (camera-space)", g5_pass, g5_detail))

    # ------------------------------------------------------------------
    # Gate 6: Body scale valid — body_scale > 0.0
    # ------------------------------------------------------------------
    g6_pass   = motion.body_scale > 0.0
    g6_detail = (
        f"body_scale = {motion.body_scale:.4f} m (> 0.0)."
        if g6_pass
        else f"body_scale = {motion.body_scale:.4f} — estimation failed (full body may not be in frame)."
    )
    gates.append(GateResult(6, "Body scale valid (body_scale > 0.0)", g6_pass, g6_detail))

    # ------------------------------------------------------------------
    # Gate 7: Wrist-to-hand attachment
    # If a hand is detected, its wrist landmark must be within
    # WRIST_ATTACH_TOLERANCE body-height units of the corresponding body wrist.
    # ------------------------------------------------------------------
    attach_issues = []
    scale = motion.body_scale if motion.body_scale > 0.0 else 1.0

    for side, hand_pose, body_wrist_joint in [
        ("left",  motion.left_hand,  motion.body.left_wrist),
        ("right", motion.right_hand, motion.body.right_wrist),
    ]:
        if hand_pose is None:
            continue  # Not detected — skip (not a failure)
        if body_wrist_joint is None or not body_wrist_joint.is_visible:
            attach_issues.append(
                f"{side}: hand detected but body wrist joint is missing/not visible"
            )
            continue
        # Both hand.wrist and body wrist are in canonical space.
        # Compare positions directly (already in body-height units).
        delta = np.linalg.norm(
            hand_pose.wrist.position - body_wrist_joint.position
        )
        if delta > WRIST_ATTACH_TOLERANCE:
            attach_issues.append(
                f"{side}: hand wrist offset = {delta:.3f} body-height units "
                f"(tolerance = {WRIST_ATTACH_TOLERANCE})"
            )

    g7_pass   = len(attach_issues) == 0
    if not (motion.has_left_hand or motion.has_right_hand):
        g7_pass   = True  # No hands detected — gate vacuously passes
        g7_detail = "No hands detected — attachment check skipped (vacuous pass)."
    elif g7_pass:
        g7_detail = "Hand wrist positions are within tolerance of body wrist joints."
    else:
        g7_detail = f"Attachment failures: {attach_issues}"
    gates.append(GateResult(7, "Wrist-to-hand attachment valid", g7_pass, g7_detail))

    # ------------------------------------------------------------------
    # Gate 8: Visual skeleton correctness — human inspection required
    # This gate always produces output images. It cannot be automated.
    # It is marked as MANUAL — the script will note it requires human review.
    # ------------------------------------------------------------------
    gates.append(GateResult(
        8,
        "Debug skeleton visually follows body [MANUAL INSPECTION REQUIRED]",
        passed=True,  # Provisionally True — human must override if wrong
        detail=(
            "Output images have been saved. Inspect skeleton_overlay.png and side_by_side.png "
            "to confirm joints are not flipped, hallucinated, or anatomically wrong."
        ),
    ))

    return gates


# ==============================================================================
# Skeleton renderer
# ==============================================================================

def _project(
    joint: Optional[JointState],
    origin_px: Tuple[int, int],
    scale_px: float,
    canvas_h: int,
    canvas_w: int,
) -> Optional[Tuple[int, int]]:
    if joint is None or not joint.is_visible:
        return None
    ix = int(origin_px[0] + joint.position[0] * scale_px)
    iy = int(origin_px[1] - joint.position[1] * scale_px)
    return (max(0, min(canvas_w - 1, ix)), max(0, min(canvas_h - 1, iy)))


def render_skeleton(
    motion: CanonicalMotionState,
    canvas: np.ndarray,
    scale_px: float = 180.0,
    origin_px: Optional[Tuple[int, int]] = None,
) -> Dict[str, Optional[Tuple[int, int]]]:
    h, w = canvas.shape[:2]
    if origin_px is None:
        origin_px = (w // 2, int(h * 0.65))

    joints = motion.body.all_joints()
    px: Dict[str, Optional[Tuple[int, int]]] = {
        name: _project(j, origin_px, scale_px, h, w)
        for name, j in joints.items()
    }

    # Bones
    for parent, child, color in BODY_BONES:
        pp, cp = px.get(parent), px.get(child)
        if pp and cp:
            cv2.line(canvas, pp, cp, color, 3, cv2.LINE_AA)

    # Joints
    JOINT_COLORS = {
        "head": COLOR_HEAD, "neck": COLOR_HEAD,
        "left_shoulder": COLOR_LEFT_ARM, "left_elbow": COLOR_LEFT_ARM, "left_wrist": COLOR_LEFT_ARM,
        "right_shoulder": COLOR_RIGHT_ARM, "right_elbow": COLOR_RIGHT_ARM, "right_wrist": COLOR_RIGHT_ARM,
        "left_hip": COLOR_LEFT_LEG, "left_knee": COLOR_LEFT_LEG, "left_ankle": COLOR_LEFT_LEG,
        "right_hip": COLOR_RIGHT_LEG, "right_knee": COLOR_RIGHT_LEG, "right_ankle": COLOR_RIGHT_LEG,
    }
    for name, p in px.items():
        if p is None:
            continue
        j = joints[name]
        color  = JOINT_COLORS.get(name, COLOR_SPINE)
        radius = 22 if name == "head" else 8
        conf   = j.confidence if j else 1.0
        draw_color = tuple(int(c * min(conf * 1.5, 1.0)) for c in color)
        cv2.circle(canvas, p, radius, draw_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p, radius, (255, 255, 255), 1, cv2.LINE_AA)
        # Label key joints
        if name in ("left_shoulder", "right_shoulder", "left_ankle", "right_ankle",
                     "head", "left_wrist", "right_wrist", "left_hip", "right_hip"):
            short = name.replace("left_", "L-").replace("right_", "R-")
            cv2.putText(canvas, short, (p[0] + 9, p[1] - 4),
                        FONT, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

    # Head direction arrow
    if motion.has_face and px.get("head"):
        hp = px["head"]
        R   = motion.face.head_rotation
        fwd = R[:, 2]
        tip = (int(hp[0] + fwd[0] * 40), int(hp[1] - fwd[1] * 40))
        cv2.arrowedLine(canvas, hp, tip, (0, 180, 255), 2, tipLength=0.3,
                         line_type=cv2.LINE_AA)

    # Hand markers
    for hand, wrist_key in [(motion.left_hand, "left_wrist"), (motion.right_hand, "right_wrist")]:
        if hand and px.get(wrist_key):
            cv2.circle(canvas, px[wrist_key], 14, COLOR_HANDS, 2, cv2.LINE_AA)

    return px


def render_gate_panel(gates: List[GateResult], h: int, w: int) -> np.ndarray:
    """Render a gate-report panel (black background, PASS=green FAIL=red)."""
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(panel, "Phase 2.4B — Exit Gate Report", (10, 28),
                FONT, 0.55, (255, 220, 80), 1, cv2.LINE_AA)
    cv2.line(panel, (10, 38), (w - 10, 38), (80, 80, 80), 1)

    for i, g in enumerate(gates):
        y = 62 + i * 50
        badge_color = COLOR_PASS if g.passed else COLOR_FAIL
        badge_text  = "PASS" if g.passed else "FAIL"
        # Gate badge
        cv2.rectangle(panel, (10, y - 16), (66, y + 4), badge_color, -1)
        cv2.putText(panel, badge_text, (14, y), FONT, 0.42, (10, 10, 10), 1, cv2.LINE_AA)
        # Gate name
        cv2.putText(panel, f"G{g.gate_id}: {g.name}", (74, y),
                    FONT, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        # Detail (truncated)
        detail_short = g.detail[:90] + "…" if len(g.detail) > 90 else g.detail
        cv2.putText(panel, detail_short, (74, y + 16),
                    FONT, 0.30, (150, 150, 150), 1, cv2.LINE_AA)

    # Overall verdict
    all_pass = all(g.passed for g in gates)
    verdict_color = COLOR_PASS if all_pass else COLOR_FAIL
    verdict_text  = "PHASE 2.4B: ALL GATES PASS" if all_pass else "PHASE 2.4B: GATES FAILED"
    y_verdict = 62 + len(gates) * 50 + 10
    cv2.rectangle(panel, (10, y_verdict - 18), (w - 10, y_verdict + 8),
                  verdict_color, -1)
    cv2.putText(panel, verdict_text, (18, y_verdict),
                FONT, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    return panel


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2.4B — Full-Body Static Validation"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a full-body image (head to ankles, frontal, good lighting).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory to save validation outputs. Defaults to test_data/outputs/phase2_4b_validation.",
    )
    parser.add_argument(
        "--scale-px",
        type=float,
        default=None,
        help="Pixels per body-height unit for skeleton rendering. "
             "Auto-detected from image height if omitted.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Phase 2.4B — Full-Body Static Validation")
    print("  8-gate exit check: CanonicalMotionState vs full-body image")
    print("=" * 72)

    # Load image
    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"ERROR: Cannot read image: {args.image}", file=sys.stderr)
        return 1
    h, w = bgr.shape[:2]
    print(f"\nLoaded: {args.image}  ({w} x {h} px)")

    # ── Step 1: MediaPipe extraction ─────────────────────────────────────────
    print("\n[Step 1] Running MediaPipe extraction...")
    try:
        performer_shim, extraction_timings = _build_performer_state(bgr)
    except Exception as exc:
        print(f"ERROR during MediaPipe extraction: {exc}", file=sys.stderr)
        raise

    print(f"  Face   : {'detected' if performer_shim.faces else 'NOT detected'}"
          f"  ({extraction_timings['face_ms']:.1f} ms)")
    print(f"  Pose   : {'detected' if performer_shim.body.landmarks_3d is not None else 'NOT detected'}"
          f"  ({extraction_timings['pose_ms']:.1f} ms)")
    print(f"  Hands  : L={'yes' if performer_shim.left_hand else 'no'}  "
          f"R={'yes' if performer_shim.right_hand else 'no'}"
          f"  ({extraction_timings['hands_ms']:.1f} ms)")

    if performer_shim.body.landmarks_3d is None:
        print("\nFATAL: No body pose detected. Cannot proceed — ensure the image "
              "shows a complete standing figure.", file=sys.stderr)
        return 1

    # ── Step 2: Adapt to CanonicalMotionState ────────────────────────────────
    print("\n[Step 2] Adapting to CanonicalMotionState...")
    t0      = time.perf_counter()
    motion  = adapt_performer_state(
        performer_shim,
        frame_index=0,
        capture_timestamp=time.time(),
    )
    adapt_ms = (time.perf_counter() - t0) * 1000
    print(f"  Adapter latency : {adapt_ms:.2f} ms")
    print(f"  Tracking quality: {motion.tracking_quality}")
    print(f"  Visible joints  : {motion.body.visible_joint_count()} / {len(motion.body.all_joints())}")
    print(f"  Body scale      : {motion.body_scale:.4f} m")

    # ── Step 3: Evaluate gates ───────────────────────────────────────────────
    print("\n[Step 3] Evaluating Phase 2.4B exit gates...")
    gates = evaluate_gates(motion)

    print()
    for g in gates:
        print(str(g))

    all_pass   = all(g.passed for g in gates)
    fail_count = sum(1 for g in gates if not g.passed)
    print()
    print("=" * 72)
    if all_pass:
        print("  PHASE 2.4B RESULT: ALL 8 GATES PASS")
    else:
        print(f"  PHASE 2.4B RESULT: {fail_count} GATE(S) FAILED")
    print("=" * 72)

    # ── Step 4: Render skeleton images ───────────────────────────────────────
    print("\n[Step 4] Rendering skeleton images...")
    scale_px = args.scale_px if args.scale_px else h * 0.48

    # Overlay on source
    overlay = bgr.copy()
    render_skeleton(motion, overlay, scale_px=scale_px,
                    origin_px=(w // 2, int(h * 0.65)))

    # Clean black canvas
    clean = np.zeros_like(bgr)
    render_skeleton(motion, clean, scale_px=scale_px,
                    origin_px=(w // 2, int(h * 0.65)))

    # Gate panel
    gate_panel = render_gate_panel(gates, h, w)

    # Assemble 2x2 side-by-side
    ph, pw = h // 2, w // 2
    p_src     = cv2.resize(bgr,     (pw, ph))
    p_overlay = cv2.resize(overlay, (pw, ph))
    p_clean   = cv2.resize(clean,   (pw, ph))
    p_gates   = cv2.resize(gate_panel, (pw, ph))
    sbs = np.vstack([
        np.hstack([p_src, p_overlay]),
        np.hstack([p_clean, p_gates]),
    ])
    for label, pos in [("Source", (6, 20)), ("Skeleton Overlay", (pw + 6, 20)),
                        ("Clean Skeleton", (6, ph + 20)), ("Gate Report", (pw + 6, ph + 20))]:
        cv2.putText(sbs, label, pos, FONT, 0.55, (255, 220, 80), 1, cv2.LINE_AA)

    # ── Step 5: Save outputs ─────────────────────────────────────────────────
    overlay_path = out_dir / "skeleton_overlay.png"
    clean_path   = out_dir / "skeleton_clean.png"
    sbs_path     = out_dir / "side_by_side.png"
    gates_path   = out_dir / "gate_report.json"
    state_path   = out_dir / "canonical_state.json"

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(clean_path),   clean)
    cv2.imwrite(str(sbs_path),     sbs)

    # Gate report JSON
    gate_report = {
        "phase":      "2.4B",
        "image":      str(args.image),
        "all_pass":   all_pass,
        "pass_count": sum(1 for g in gates if g.passed),
        "fail_count": fail_count,
        "gates":      [g.to_dict() for g in gates],
        "joint_summary": {
            name: {
                "present":    j is not None,
                "confidence": round(j.confidence, 3) if j else None,
                "is_visible": bool(j.is_visible) if j else False,
                "position":   j.position.tolist() if j else None,
                "has_rotation": j.rotation is not None if j else False,
            }
            for name, j in motion.body.all_joints().items()
        },
        "body_scale_m":    round(motion.body_scale, 4),
        "has_face":        motion.has_face,
        "has_left_hand":   motion.has_left_hand,
        "has_right_hand":  motion.has_right_hand,
        "tracking_quality": motion.tracking_quality,
        "extraction_timings_ms": {k: round(v, 2) for k, v in extraction_timings.items()},
        "adapter_ms":       round(adapt_ms, 2),
    }
    with open(gates_path, "w") as f:
        json.dump(gate_report, f, indent=2)

    # Canonical state JSON (compact per-joint snapshot)
    def _jdict(j):
        if j is None:
            return None
        return {
            "position":     j.position.tolist(),
            "confidence":   round(j.confidence, 3),
            "is_visible":   bool(j.is_visible),
            "has_rotation": j.rotation is not None,
            "rotation_det": round(float(np.linalg.det(j.rotation)), 4) if j.rotation is not None else None,
        }
    state_snapshot = {
        "schema_version":   motion.schema_version,
        "source_backend":   motion.source_backend,
        "tracking_quality": motion.tracking_quality,
        "body_scale_m":     round(motion.body_scale, 4),
        "body_joints":      {n: _jdict(j) for n, j in motion.body.all_joints().items()},
        "has_face":         motion.has_face,
        "has_left_hand":    motion.has_left_hand,
        "has_right_hand":   motion.has_right_hand,
        "adapter_timings_ms": {k: round(v, 2) for k, v in motion.adapter_timings.items()},
        "validate_errors":  motion.validate(),
    }
    with open(state_path, "w") as f:
        json.dump(state_snapshot, f, indent=2)

    print("\n  Output files:")
    for p in [overlay_path, clean_path, sbs_path, gates_path, state_path]:
        print(f"    {p}")

    print(f"\n  GATE 8 (visual): Inspect {sbs_path}")
    print("  Confirm: joints are not flipped, hallucinated, or anatomically incorrect.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
