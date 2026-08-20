"""
Phase 2.4C — Temporal Motion Stability Benchmark

Processes a performer video sequence frame-by-frame:
    Video Frame(t)
         ↓
    MediaPipe Motion Extraction
         ↓
    PerformerState(t)
         ↓
    CanonicalMotionState(t)
         ↓
    Temporal Stability Analysis: State(t-1) ↔ State(t)

Measures frame-to-frame delta metrics across the video:
    1. Δ position (frame-to-frame jump per joint in body-height units)
    2. Δ rotation angle (SO3 geodesic distance / axis-angle jump per joint in degrees)
    3. Joint tracking confidence continuity & drop rate
    4. Left/right anatomical inversion events across frames
    5. NaN / Inf occurrences across frames

Usage
-----
    python services/inference/scripts/validate_temporal_stability.py \\
        --video test_data/inputs/performer/live_test.mp4 \\
        [--output-dir test_data/outputs/YYYY-MM-DD_temporal_stability]

Outputs
-------
    - temporal_stability_report.json  (Summary statistics & gate evaluations)
    - per_frame_metrics.json          (Detailed per-frame, per-joint deltas)
    - skeleton_overlay.mp4            (Performer video with CanonicalMotionState overlay)
"""

from __future__ import annotations

import argparse
import datetime
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

# Gate Thresholds (from PHASE_2_ROADMAP.md § Phase 2.4C)
MAX_POSITION_JUMP_UNITS = 0.10   # max allowed single-frame jump in body-height units
MAX_ROTATION_FLIP_DEG   = 45.0   # max allowed single-frame rotation jump in degrees
MAX_CONFIDENCE_DROP_PCT = 5.0    # max allowed confidence drop rate across sequence (%)

# Color Palette for Skeleton Overlay (BGR)
COLOR_SPINE     = (255, 255, 100)
COLOR_LEFT_ARM  = (80,  200, 255)
COLOR_RIGHT_ARM = (100, 255, 100)
COLOR_LEFT_LEG  = (180, 130, 255)
COLOR_RIGHT_LEG = (255, 160, 80)
COLOR_HEAD      = (255, 255, 255)
COLOR_HANDS     = (50,  220, 220)
COLOR_ALERT     = (0,   0,   255)
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
# Helper math routines
# ==============================================================================

def _rotation_delta_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute geodesic rotation angle difference between two SO(3) matrices in degrees.
    """
    if R1 is None or R2 is None:
        return 0.0
    # R_diff = R1^T * R2
    R_diff = np.dot(R1.T, R2)
    tr = np.trace(R_diff)
    # cos(theta) = (tr - 1) / 2
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.degrees(theta_rad))


# ==============================================================================
# MediaPipe extraction & state construction
# ==============================================================================

class VideoMotionExtractor:
    """Reuses MediaPipe landmarker graph instances across video frames."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.vision = vision

        # Initialize landmarker tasks once
        self.fl = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=tasks.BaseOptions(
                    model_asset_path=str(MODELS_DIR / "face_landmarker.task")
                ),
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self.pl = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=tasks.BaseOptions(
                    model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")
                ),
                num_poses=1,
                output_segmentation_masks=False,
            )
        )
        self.hl = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(
                    model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
                ),
                num_hands=2,
            )
        )

    def process_frame(self, bgr: np.ndarray, frame_idx: int, timestamp: float) -> Tuple[CanonicalMotionState, dict]:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        # Run MediaPipe tasks
        t_face = time.perf_counter()
        face_res = self.fl.detect(mp_img)
        ms_face = (time.perf_counter() - t_face) * 1000

        t_pose = time.perf_counter()
        pose_res = self.pl.detect(mp_img)
        ms_pose = (time.perf_counter() - t_pose) * 1000

        t_hand = time.perf_counter()
        hand_res = self.hl.detect(mp_img)
        ms_hand = (time.perf_counter() - t_hand) * 1000

        # Build PerformerState shim
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

        if pose_res.pose_world_landmarks:
            lms = pose_res.pose_world_landmarks[0]
            state.body.landmarks_3d = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms], dtype=np.float32
            )

        if (face_res.face_landmarks
                and face_res.face_blendshapes
                and face_res.facial_transformation_matrixes):
            bs = {b.category_name: float(b.score) for b in face_res.face_blendshapes[0]}
            mat = np.array(face_res.facial_transformation_matrixes[0], dtype=np.float32)
            state.faces = [_FS(bs, mat, 1.0)]

        if hand_res.hand_landmarks and hand_res.hand_world_landmarks:
            for wlms, handedness_list in zip(
                hand_res.hand_world_landmarks, hand_res.handedness
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

        # Convert to CanonicalMotionState
        t_adapt = time.perf_counter()
        canonical = adapt_performer_state(
            state, frame_index=frame_idx, capture_timestamp=timestamp
        )
        ms_adapt = (time.perf_counter() - t_adapt) * 1000

        timings = {
            "face_ms": ms_face,
            "pose_ms": ms_pose,
            "hand_ms": ms_hand,
            "adapt_ms": ms_adapt,
            "total_ms": (time.perf_counter() - t0) * 1000,
        }
        return canonical, timings


# ==============================================================================
# Visual overlay rendering
# ==============================================================================

def render_overlay_frame(
    bgr: np.ndarray,
    canonical: CanonicalMotionState,
    frame_idx: int,
    total_frames: int,
    fps: float,
    metrics_summary: dict,
) -> np.ndarray:
    canvas = bgr.copy()
    h, w = canvas.shape[:2]
    scale_px = h * 0.48
    origin_px = (w // 2, int(h * 0.65))

    joints = canonical.body.all_joints()

    # Project joints
    px = {}
    for name, j in joints.items():
        if j is not None and j.is_visible:
            ix = int(origin_px[0] + j.position[0] * scale_px)
            iy = int(origin_px[1] - j.position[1] * scale_px)
            px[name] = (max(0, min(w - 1, ix)), max(0, min(h - 1, iy)))
        else:
            px[name] = None

    # Draw bones
    for parent, child, color in BODY_BONES:
        p1, p2 = px.get(parent), px.get(child)
        if p1 and p2:
            cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)

    # Draw joint circles
    for name, p in px.items():
        if p is None:
            continue
        radius = 18 if name == "head" else 6
        cv2.circle(canvas, p, radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, radius, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD Banner
    cv2.rectangle(canvas, (0, 0), (w, 50), (20, 20, 20), -1)
    hud_str = f"Frame {frame_idx}/{total_frames} | Quality: {canonical.tracking_quality} | Scale: {canonical.body_scale:.2f}m"
    cv2.putText(canvas, hud_str, (15, 30), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ==============================================================================
# Main execution & temporal metric evaluation
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2.4C — Temporal Motion Stability Benchmark"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input performer video file (e.g. test_data/inputs/performer/live_test.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save benchmark outputs. Defaults to test_data/outputs/YYYY-MM-DD_temporal_stability.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        return 1

    # Setup run-specific output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        out_dir = PROJECT_ROOT / "test_data" / "outputs" / f"{date_str}_temporal_stability"

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Phase 2.4C — Temporal Motion Stability Benchmark")
    print("=" * 72)
    print(f"  Input Video  : {video_path}")
    print(f"  Output Dir   : {out_dir}\n")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file {video_path}", file=sys.stderr)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = float(cap.get(cv2.CAP_PROP_FPS))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video Stats  : {total_frames} frames @ {fps:.2f} FPS ({width}x{height})")

    # Video Writer for overlay output
    overlay_video_path = out_dir / "skeleton_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(overlay_video_path), fourcc, fps if fps > 0 else 30.0, (width, height))

    extractor = VideoMotionExtractor()

    states: List[CanonicalMotionState] = []
    timings_history: List[dict] = []
    frame_metrics: List[dict] = []

    prev_state: Optional[CanonicalMotionState] = None

    pos_jumps_exceeded = 0
    rot_flips_exceeded = 0
    inversion_events   = 0
    nan_events         = 0
    confidence_drops   = 0

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / (fps if fps > 0 else 30.0)
        canonical, timings = extractor.process_frame(frame, frame_idx, timestamp)
        states.append(canonical)
        timings_history.append(timings)

        # Evaluate frame-to-frame delta vs prev_state
        delta_info = {}
        if prev_state is not None:
            curr_joints = canonical.body.all_joints()
            prev_joints = prev_state.body.all_joints()

            for name in curr_joints.keys():
                cj = curr_joints[name]
                pj = prev_joints[name]

                if cj is not None and pj is not None and cj.is_visible and pj.is_visible:
                    # 1. Position delta (body-height units)
                    pos_delta = float(np.linalg.norm(cj.position - pj.position))
                    if pos_delta > MAX_POSITION_JUMP_UNITS:
                        pos_jumps_exceeded += 1

                    # 2. Rotation delta (degrees)
                    rot_delta = _rotation_delta_deg(pj.rotation, cj.rotation)
                    if rot_delta > MAX_ROTATION_FLIP_DEG:
                        rot_flips_exceeded += 1

                    delta_info[name] = {
                        "pos_delta": round(pos_delta, 4),
                        "rot_delta": round(rot_delta, 2),
                    }
                elif (pj is not None and pj.is_visible) and (cj is None or not cj.is_visible):
                    confidence_drops += 1

            # Left/Right inversion check
            ls = canonical.body.left_shoulder
            rs = canonical.body.right_shoulder
            if ls and rs and ls.is_visible and rs.is_visible:
                if ls.position[0] <= rs.position[0] - 0.05:
                    inversion_events += 1

        # Check NaNs
        val_errors = canonical.validate()
        for err in val_errors:
            if "NaN" in err or "Inf" in err:
                nan_events += 1

        frame_metrics.append({
            "frame": frame_idx,
            "timestamp": round(timestamp, 3),
            "tracking_quality": canonical.tracking_quality,
            "visible_joints": canonical.body.visible_joint_count(),
            "deltas": delta_info,
        })

        # Render overlay frame
        overlay_frame = render_overlay_frame(frame, canonical, frame_idx, total_frames, fps, {})
        writer.write(overlay_frame)

        prev_state = canonical
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    total_sec = time.perf_counter() - t_start

    print(f"\nFinished processing {frame_idx} frames in {total_sec:.2f} s ({frame_idx/total_sec:.1f} FPS).")

    # Evaluate Gate Exit Criteria
    total_joint_frame_checks = max(1, frame_idx * 17)
    confidence_drop_rate_pct = (confidence_drops / total_joint_frame_checks) * 100.0

    gate1_pass = pos_jumps_exceeded == 0
    gate2_pass = rot_flips_exceeded == 0
    gate3_pass = confidence_drop_rate_pct <= MAX_CONFIDENCE_DROP_PCT
    gate4_pass = inversion_events == 0
    gate5_pass = nan_events == 0

    all_gates_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass

    # Save summary report
    report = {
        "phase": "2.4C",
        "video": str(video_path),
        "total_frames": frame_idx,
        "processing_fps": round(frame_idx / total_sec, 2),
        "all_gates_pass": all_gates_pass,
        "gates": [
            {
                "gate": 1,
                "name": "Temporal position smoothness",
                "passed": gate1_pass,
                "detail": f"Position jumps > {MAX_POSITION_JUMP_UNITS} units: {pos_jumps_exceeded} events.",
            },
            {
                "gate": 2,
                "name": "No rotation flips",
                "passed": gate2_pass,
                "detail": f"Rotation flips > {MAX_ROTATION_FLIP_DEG}°: {rot_flips_exceeded} events.",
            },
            {
                "gate": 3,
                "name": "Confidence tracking stability",
                "passed": gate3_pass,
                "detail": f"Confidence drop rate: {confidence_drop_rate_pct:.2f}% (max allowed: {MAX_CONFIDENCE_DROP_PCT}%).",
            },
            {
                "gate": 4,
                "name": "Left/right anatomical consistency",
                "passed": gate4_pass,
                "detail": f"Inversion events: {inversion_events}.",
            },
            {
                "gate": 5,
                "name": "NaN-free sequence",
                "passed": gate5_pass,
                "detail": f"NaN/Inf events: {nan_events}.",
            },
        ],
    }

    report_path = out_dir / "temporal_stability_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    metrics_path = out_dir / "per_frame_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(frame_metrics, f, indent=2)

    print("\n" + "=" * 72)
    print(f"  PHASE 2.4C TEMPORAL STABILITY RESULT: {'ALL GATES PASS' if all_gates_pass else 'GATES FAILED'}")
    print("=" * 72)
    for g in report["gates"]:
        print(f"  Gate {g['gate']} [{ 'PASS' if g['passed'] else 'FAIL' }] {g['name']}")
        print(f"         {g['detail']}")

    print(f"\n  Outputs saved to: {out_dir}")
    print(f"    - {report_path.name}")
    print(f"    - {metrics_path.name}")
    print(f"    - {overlay_video_path.name}")

    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    sys.exit(main())
