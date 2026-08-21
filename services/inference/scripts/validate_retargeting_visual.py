"""
validate_retargeting_visual.py — Phase 2.4D Visual Integration & Pose-Fidelity Validation Benchmark

Processes recorded performer video frame-by-frame through the complete real pipeline:

  Recorded Performer Video (.mp4)
            ↓
  MediaPipe PoseLandmarker (33 3D landmarks)
            ↓
  adapt_performer_state()  →  CanonicalMotionState (t)
            ↓
  TemporalStabilizer  →  StableCanonicalMotionState (t)
            ↓
  AnatomicalFrameBuilder  →  F_world(j) (SO3 anatomical frames with twist)
            ↓
  LocalRotationExtractor  →  R_motion_local(j) (performer motion deltas vs Canonical T-Pose reference)
            ↓
  KinematicRetargeter (DEFAULT, TALL, PETITE actor profiles)
            ↓
  Forward Kinematics  →  RetargetedActorState (actor-proportioned geometry)
            ↓
  Quantitative Verification Checks + 4-Panel Side-by-Side Video & JSON Report

Usage
-----
    python services/inference/scripts/validate_retargeting_visual.py \
        --video "test_data/inputs/performer/WhatsApp Video 2026-08-20 at 10.16.06 PM.mp4" \
        --output_dir "test_data/outputs/retargeting_visual"

Outputs
-------
    side_by_side_retargeting.mp4   — 4-panel visual side-by-side debug video
    retargeting_visual_report.json — Quantitative metrics, pose fidelity, & latency breakdown
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
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

from app.pipeline.result import LandmarkResult
from app.motion.canonical_state import (
    CanonicalMotionState,
    BodyPose,
    JointState,
    CONFIDENCE_THRESHOLD,
)
from app.motion.mediapipe_adapter import adapt_performer_state
from app.motion.temporal_stabilizer import TemporalStabilizer
from app.motion.actor_skeleton import (
    ActorSkeleton, ACTOR_PROFILES, DEFAULT_ACTOR, TALL_ACTOR, PETITE_ACTOR,
    JOINT_HIERARCHY, FK_JOINT_ORDER,
)
from app.motion.anatomical_frame_builder import AnatomicalFrameBuilder
from app.motion.local_rotation_extractor import LocalRotationExtractor
from app.motion.motion_retargeter import KinematicRetargeter, geodesic_angle_deg
from app.motion.retargeted_actor_state import RetargetedActorState

MODELS_DIR = PROJECT_ROOT / "services" / "inference" / "models"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Skeletal Bone Connections
BODY_BONES = [
    ("pelvis",          "left_hip",       (180, 180, 0)),
    ("pelvis",          "right_hip",      (180, 180, 0)),
    ("pelvis",          "spine_mid",      (200, 200, 200)),
    ("spine_mid",       "chest",          (200, 200, 200)),
    ("chest",           "neck",           (200, 200, 200)),
    ("neck",            "head",           (200, 255, 255)),
    ("chest",           "left_shoulder",  (255, 100, 100)),
    ("left_shoulder",   "left_elbow",     (255, 100, 100)),
    ("left_elbow",      "left_wrist",     (255, 100, 100)),
    ("chest",           "right_shoulder", (100, 100, 255)),
    ("right_shoulder",  "right_elbow",    (100, 100, 255)),
    ("right_elbow",     "right_wrist",    (100, 100, 255)),
    ("left_hip",        "left_knee",      (255, 200, 100)),
    ("left_knee",       "left_ankle",     (255, 200, 100)),
    ("right_hip",       "right_knee",     (100, 200, 255)),
    ("right_knee",      "right_ankle",    (100, 200, 255)),
]


# ──────────────────────────────────────────────────────────────────────────────
# Shim to run MediaPipe Tasks
# ──────────────────────────────────────────────────────────────────────────────

def _run_mediapipe_frame(bgr: np.ndarray, fl, pl, hl) -> Tuple[object, dict, object]:
    """Runs MediaPipe landmarkers on frame and builds PerformerState shim."""
    import mediapipe as mp
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timings = {}

    t0 = time.perf_counter()
    face_res = fl.detect(mp_img)
    timings["face_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    pose_res = pl.detect(mp_img)
    timings["pose_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    hand_res = hl.detect(mp_img)
    timings["hands_ms"] = (time.perf_counter() - t0) * 1000.0

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

    shim = _ShimPerformerState()

    # Pose landmarks (world landmarks = 3D array [x,y,z,visibility])
    if pose_res and pose_res.pose_world_landmarks and len(pose_res.pose_world_landmarks) > 0:
        raw_lm = pose_res.pose_world_landmarks[0]
        arr = np.array([[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in raw_lm], dtype=np.float32)
        shim.body.landmarks_3d = arr

    # Face landmarks
    if face_res and face_res.face_blendshapes and face_res.facial_transformation_matrixes:
        bs_dict = {b.category_name: float(b.score) for b in face_res.face_blendshapes[0]}
        mat = np.array(face_res.facial_transformation_matrixes[0], dtype=np.float32)
        shim.faces = [_ShimFaceState(bs_dict, mat, 1.0)]

    return shim, timings, pose_res


# ──────────────────────────────────────────────────────────────────────────────
# 2D Screen Projection Helper
# ──────────────────────────────────────────────────────────────────────────────

def _project_joint(
    pos: Optional[np.ndarray],
    canvas_h: int, canvas_w: int,
    origin_px: Tuple[int, int],
    scale_px: float = 160.0,
) -> Optional[Tuple[int, int]]:
    """Project 3D canonical (+Y up, +X right, pelvis=origin) to 2D image pixels."""
    if pos is None or np.any(np.isnan(pos)):
        return None
    ix = int(origin_px[0] + pos[0] * scale_px)
    iy = int(origin_px[1] - pos[1] * scale_px)
    return (
        max(0, min(canvas_w - 1, ix)),
        max(0, min(canvas_h - 1, iy)),
    )


def _draw_skeleton(
    canvas: np.ndarray,
    positions: Dict[str, Optional[np.ndarray]],
    origin_px: Tuple[int, int],
    scale_px: float,
    title: str,
    accent_color: Tuple[int, int, int] = (0, 255, 0),
):
    """Draw stick figure skeleton onto a panel canvas."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, h), (18, 18, 18), -1)

    # Grid floor reference
    grid_y = origin_px[1]
    cv2.line(canvas, (10, grid_y), (w - 10, grid_y), (40, 40, 40), 1, cv2.LINE_AA)

    # Project joint pixels
    j_px = {}
    for j_name, pos in positions.items():
        j_px[j_name] = _project_joint(pos, h, w, origin_px, scale_px)

    # Draw bones
    for parent_name, child_name, color in BODY_BONES:
        p1 = j_px.get(parent_name)
        p2 = j_px.get(child_name)
        if p1 is not None and p2 is not None:
            cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)

    # Draw joints
    for j_name, p in j_px.items():
        if p is None:
            continue
        rad = 8 if j_name == "head" else 4
        c = (0, 255, 255) if j_name in ("left_wrist", "right_wrist") else accent_color
        cv2.circle(canvas, p, rad, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, p, rad, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel Title
    cv2.putText(canvas, title, (12, 28), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (12, 28), FONT, 0.55, accent_color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main Validation Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.4D End-to-End Retargeting Visual & Quantitative Validation"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4)")
    parser.add_argument("--output_dir", type=str, default="test_data/outputs/retargeting_visual")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to process")
    parser.add_argument("--show", action="store_true", help="Display window while processing")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Input video path not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("PHASE 2.4D VISUAL INTEGRATION & POSE-FIDELITY BENCHMARK")
    print(f"Input Video: {video_path}")
    print(f"Output Dir:  {out_dir}")
    print("================================================================================")

    # Initialize MediaPipe Tasks
    import mediapipe as mp
    from mediapipe.tasks import python as tasks
    from mediapipe.tasks.python import vision

    fl_opts = vision.FaceLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(MODELS_DIR / "face_landmarker.task")),
        num_faces=1, output_facial_transformation_matrixes=True,
    )
    pl_opts = vision.PoseLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")),
        num_poses=1,
    )
    hl_opts = vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
        num_hands=2,
    )

    fl = vision.FaceLandmarker.create_from_options(fl_opts)
    pl = vision.PoseLandmarker.create_from_options(pl_opts)
    hl = vision.HandLandmarker.create_from_options(hl_opts)

    # Initialize Pipeline Modules
    stabilizer = TemporalStabilizer()
    retargeters: Dict[str, KinematicRetargeter] = {
        "DEFAULT": KinematicRetargeter(DEFAULT_ACTOR),
        "TALL":    KinematicRetargeter(TALL_ACTOR),
        "PETITE":  KinematicRetargeter(PETITE_ACTOR),
    }

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup 4-Panel Output Video Writer (4 columns side-by-side)
    panel_w, panel_h = 400, 600
    canvas_w = panel_w * 4
    canvas_h = panel_h

    out_mp4_path = out_dir / "side_by_side_retargeting.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps, (canvas_w, canvas_h))

    # Tracking & Telemetry Containers
    prev_world_rot: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in retargeters}

    # Quantitative Verification Counters
    fk_errors_count = 0
    non_finite_count = 0
    label_swap_count = 0
    bone_length_violation_count = 0
    max_rotation_jump_deg = 0.0

    # Check 6: Pose-Transfer Fidelity Accumulators
    arm_pose_errors: List[float] = []
    leg_pose_errors: List[float] = []
    torso_pose_errors: List[float] = []
    arms_down_test_passes = 0
    arms_down_test_total = 0

    timings_sum: Dict[str, float] = {
        "tracker_ms": 0.0,
        "adapter_ms": 0.0,
        "stabilizer_ms": 0.0,
        "frame_builder_ms": 0.0,
        "extractor_ms": 0.0,
        "retargeter_fk_ms": 0.0,
        "total_ms": 0.0,
    }

    frame_idx = 0
    print("[INFO] Processing frames...")

    try:
        while cap.isOpened() and frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t_start_frame = time.perf_counter()

            # 1. MediaPipe Tracker
            shim, tracker_timings, mp_pose_res = _run_mediapipe_frame(frame, fl, pl, hl)
            t_tracker_done = time.perf_counter()
            tracker_ms = (t_tracker_done - t_start_frame) * 1000.0

            # 2. Canonical Adapter
            raw_state = adapt_performer_state(shim, capture_timestamp=time.time(), frame_index=frame_idx)
            t_adapter_done = time.perf_counter()
            adapter_ms = (t_adapter_done - t_tracker_done) * 1000.0

            # 3. Temporal Stabilizer
            stable_state = stabilizer.process(raw_state)
            t_stab_done = time.perf_counter()
            stabilizer_ms = (t_stab_done - t_adapter_done) * 1000.0

            # 4. Anatomical Frame Builder
            t_fb_start = time.perf_counter()
            retargeter_default = retargeters["DEFAULT"]
            anat_frames = retargeter_default._frame_builder.build_frames(stable_state)
            t_fb_done = time.perf_counter()
            frame_builder_ms = (t_fb_done - t_fb_start) * 1000.0

            # 5. Local Rotation Extractor
            t_ext_start = time.perf_counter()
            motion_deltas = retargeter_default._extractor.extract(anat_frames, stable_state.frame_index)
            t_ext_done = time.perf_counter()
            extractor_ms = (t_ext_done - t_ext_start) * 1000.0

            # 6. Kinematic Retargeting Engine (Composition + FK + Constraints for DEFAULT, TALL, PETITE)
            t_retarget_start = time.perf_counter()
            retargeted_results: Dict[str, RetargetedActorState] = {}
            for name, retargeter in retargeters.items():
                res = retargeter.retarget(stable_state)
                retargeted_results[name] = res

            t_retarget_done = time.perf_counter()
            retarget_fk_ms = (t_retarget_done - t_retarget_start) * 1000.0

            t_end_frame = time.perf_counter()
            total_frame_ms = (t_end_frame - t_start_frame) * 1000.0

            # Accumulate Timings
            timings_sum["tracker_ms"] += tracker_ms
            timings_sum["adapter_ms"] += adapter_ms
            timings_sum["stabilizer_ms"] += stabilizer_ms
            timings_sum["frame_builder_ms"] += frame_builder_ms
            timings_sum["extractor_ms"] += extractor_ms
            timings_sum["retargeter_fk_ms"] += retarget_fk_ms
            timings_sum["total_ms"] += total_frame_ms

            # ──────────────────────────────────────────────────────────────────
            # Quantitative Verification Checks
            # ──────────────────────────────────────────────────────────────────
            def_res = retargeted_results["DEFAULT"]

            # Check 1: FK Consistency
            for j in FK_JOINT_ORDER:
                parent = JOINT_HIERARCHY[j]
                if parent is None:
                    continue
                P_parent = def_res.joints.get(parent)
                R_w_parent = def_res.world_rotations.get(parent)
                P_child = def_res.joints.get(j)
                if P_parent is not None and R_w_parent is not None and P_child is not None:
                    v = def_res.actor_skeleton.v_rest(j)
                    P_expected = P_parent + R_w_parent @ v
                    fk_err = float(np.linalg.norm(P_child - P_expected))
                    if fk_err >= 1e-4:
                        fk_errors_count += 1

            # Check 2: Non-finite values
            for res in retargeted_results.values():
                for pos in res.joints.values():
                    if pos is not None and not np.all(np.isfinite(pos)):
                        non_finite_count += 1
                for R_w in res.world_rotations.values():
                    if R_w is not None and not np.all(np.isfinite(R_w)):
                        non_finite_count += 1

            # Check 3: Anatomical Chirality (L/R swap check)
            P_ls = def_res.joints.get("left_shoulder")
            P_rs = def_res.joints.get("right_shoulder")
            if P_ls is not None and P_rs is not None:
                if P_ls[0] < P_rs[0]:
                    label_swap_count += 1

            # Check 4: Bone Length Preservation
            for profile_name, res in retargeted_results.items():
                actor = res.actor_skeleton
                for j in FK_JOINT_ORDER:
                    parent = JOINT_HIERARCHY[j]
                    if parent is None:
                        continue
                    P_c = res.joints.get(j)
                    P_p = res.joints.get(parent)
                    if P_c is not None and P_p is not None:
                        expected_len = actor.bone_lengths.get(j, 0.0)
                        actual_len = float(np.linalg.norm(P_c - P_p))
                        if abs(actual_len - expected_len) > 1e-3:
                            bone_length_violation_count += 1

            # Check 5: Bounded Rotation Changes (for observed upper body joints)
            src_body = stable_state.body
            for profile_name, res in retargeted_results.items():
                prev_rots = prev_world_rot[profile_name]
                curr_rots = res.world_rotations
                for j, R_curr in curr_rots.items():
                    if R_curr is None:
                        continue
                    if "hip" in j or "knee" in j or "ankle" in j:
                        continue
                    src_j = getattr(src_body, j, None)
                    if src_j is None or not src_j.is_visible or src_j.confidence < CONFIDENCE_THRESHOLD:
                        continue
                    R_prev = prev_rots.get(j)
                    if R_prev is not None:
                        jump_deg = geodesic_angle_deg(R_prev, R_curr)
                        if jump_deg > max_rotation_jump_deg:
                            max_rotation_jump_deg = jump_deg
                            if jump_deg > 20.0:
                                print(f"  [ROT JUMP] Frame {frame_idx:03d} Joint '{j}': jump = {jump_deg:.2f}°")
                    prev_rots[j] = R_curr.copy()

            # Check 6: Pose-Transfer Fidelity (Bone Direction Alignment between source & actor)
            src_body = stable_state.body
            for j in FK_JOINT_ORDER:
                parent = JOINT_HIERARCHY[j]
                if parent is None:
                    continue
                src_p = getattr(src_body, parent, None)
                src_c = getattr(src_body, j, None)
                act_p = def_res.joints.get(parent)
                act_c = def_res.joints.get(j)
                if (src_p is not None and src_p.is_visible and src_c is not None and src_c.is_visible and
                    act_p is not None and act_c is not None):
                    v_src = src_c.position - src_p.position
                    v_act = act_c - act_p
                    n_src, n_act = float(np.linalg.norm(v_src)), float(np.linalg.norm(v_act))
                    if n_src > 1e-6 and n_act > 1e-6:
                        u_src = v_src / n_src
                        u_act = v_act / n_act
                        dot = float(np.clip(np.dot(u_src, u_act), -1.0, 1.0))
                        angle_deg = math.degrees(math.acos(dot))
                        if "shoulder" in j or "elbow" in j:
                            arm_pose_errors.append(angle_deg)
                        elif "hip" in j or "knee" in j:
                            leg_pose_errors.append(angle_deg)
                        else:
                            torso_pose_errors.append(angle_deg)

            # Check 7: Specific Arms-Down Validation (Performer arms down → Actor arms down)
            P_l_sh = getattr(src_body, "left_shoulder", None)
            P_l_el = getattr(src_body, "left_elbow", None)
            P_r_sh = getattr(src_body, "right_shoulder", None)
            P_r_el = getattr(src_body, "right_elbow", None)

            if P_l_sh and P_l_el and P_r_sh and P_r_el and P_l_sh.is_visible and P_l_el.is_visible:
                # Is performer left arm pointing downward? (elbow_y < shoulder_y - 0.15)
                # In canonical space: +Y is UP. So elbow_y < shoulder_y means pointing DOWN.
                l_src_dir_y = (P_l_el.position[1] - P_l_sh.position[1])
                if l_src_dir_y < -0.15:
                    arms_down_test_total += 1
                    act_sh = def_res.joints.get("left_shoulder")
                    act_el = def_res.joints.get("left_elbow")
                    if act_sh is not None and act_el is not None:
                        act_dir = (act_el - act_sh) / (np.linalg.norm(act_el - act_sh) + 1e-9)
                        if act_dir[1] < -0.6:  # Y component must be negative (pointing down)
                            arms_down_test_passes += 1

            # ──────────────────────────────────────────────────────────────────
            # Render 4-Panel Side-by-Side Canvas
            # ──────────────────────────────────────────────────────────────────
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # Panel 1: Original Frame + Overlay
            p1 = cv2.resize(frame, (panel_w, panel_h))
            if mp_pose_res and mp_pose_res.pose_landmarks and len(mp_pose_res.pose_landmarks) > 0:
                lms = mp_pose_res.pose_landmarks[0]
                for lm in lms:
                    px_x = int(lm.x * panel_w)
                    px_y = int(lm.y * panel_h)
                    cv2.circle(p1, (px_x, px_y), 3, (0, 255, 0), -1)
            cv2.putText(p1, "1. Performer Input (MediaPipe)", (12, 28), FONT, 0.5, (0, 255, 0), 2)
            canvas[0:panel_h, 0:panel_w] = p1

            # Panel 2: Stabilized Performer Skeleton
            p2 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            stab_pos = {n: j.position for n, j in stable_state.body.all_joints().items() if j.is_visible}
            _draw_skeleton(p2, stab_pos, (panel_w // 2, int(panel_h * 0.72)), 160.0,
                           "2. Stabilized Performer (Source)", (0, 255, 0))
            canvas[0:panel_h, panel_w:panel_w*2] = p2

            # Panel 3: DEFAULT Actor Skeleton
            p3 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            _draw_skeleton(p3, def_res.joints, (panel_w // 2, int(panel_h * 0.72)), 160.0,
                           "3. Retargeted Actor (DEFAULT)", (255, 255, 0))
            canvas[0:panel_h, panel_w*2:panel_w*3] = p3

            # Panel 4: TALL Actor Skeleton
            p4 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            tall_res = retargeted_results["TALL"]
            _draw_skeleton(p4, tall_res.joints, (panel_w // 2, int(panel_h * 0.72)), 160.0,
                           "4. Retargeted Actor (TALL)", (255, 0, 255))
            canvas[0:panel_h, panel_w*3:panel_w*4] = p4

            # OSD Telemetry Banner across the bottom
            osd = f"Frame: {frame_idx:03d} | FB: {frame_builder_ms:.2f}ms | Ext: {extractor_ms:.2f}ms | FK: {retarget_fk_ms:.2f}ms | Total: {total_frame_ms:.1f}ms | FK errs: {fk_errors_count}"
            cv2.rectangle(canvas, (0, canvas_h - 32), (canvas_w, canvas_h), (10, 10, 10), -1)
            cv2.putText(canvas, osd, (16, canvas_h - 10), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            writer.write(canvas)

            if args.show:
                cv2.imshow("Phase 2.4D Retargeting Visual Validation", canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_vid_frames} frames...")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    n_frames = max(frame_idx, 1)

    # Compute Average Latency
    avg_timings = {k: v / n_frames for k, v in timings_sum.items()}

    # Compute Pose Fidelity Metrics
    mean_arm_err = float(np.mean(arm_pose_errors)) if arm_pose_errors else 0.0
    mean_leg_err = float(np.mean(leg_pose_errors)) if leg_pose_errors else 0.0
    mean_torso_err = float(np.mean(torso_pose_errors)) if torso_pose_errors else 0.0
    arms_down_pass_pct = (arms_down_test_passes / max(arms_down_test_total, 1)) * 100.0

    # Final Verification Gate Statuses
    pass_fk = (fk_errors_count == 0)
    pass_finite = (non_finite_count == 0)
    pass_labels = (label_swap_count == 0)
    pass_bones = (bone_length_violation_count == 0)
    pass_jump = (max_rotation_jump_deg < 65.0)
    pass_arm_pose = (mean_arm_err < 25.0)
    pass_arms_down = (arms_down_pass_pct >= 95.0)

    overall_pass = (pass_fk and pass_finite and pass_labels and pass_bones and
                    pass_jump and pass_arm_pose and pass_arms_down)

    print("\n================================================================================")
    print("PHASE 2.4D VISUAL INTEGRATION & POSE-FIDELITY BENCHMARK RESULTS")
    print("================================================================================")
    print(f"Total Frames Processed: {n_frames}")
    print(f"Output Video Saved:     {out_mp4_path}")
    print("\n--- Quantitative Exit Criteria ---")
    print(f" 1. FK Self-Consistency Errors:    {fk_errors_count:d} [{'PASS' if pass_fk else 'FAIL'}]")
    print(f" 2. Non-Finite Values (NaN/Inf):   {non_finite_count:d} [{'PASS' if pass_finite else 'FAIL'}]")
    print(f" 3. L/R Anatomical Label Swaps:    {label_swap_count:d} [{'PASS' if pass_labels else 'FAIL'}]")
    print(f" 4. Bone Length Preservation:      {bone_length_violation_count:d} violations [{'PASS' if pass_bones else 'FAIL'}]")
    print(f" 5. Max Frame-to-Frame Rot Jump:   {max_rotation_jump_deg:.2f}° [{'PASS' if pass_jump else 'FAIL'}]")
    print(f"\n--- Pose-Transfer Fidelity Gate ---")
    print(f" 6. Arm Direction Error (Mean):    {mean_arm_err:.2f}° (max 25.0°) [{'PASS' if pass_arm_pose else 'FAIL'}]")
    print(f" 7. Leg Direction Error (Mean):    {mean_leg_err:.2f}°")
    print(f" 8. Torso Direction Error (Mean):  {mean_torso_err:.2f}°")
    print(f" 9. Arms-Down Pose Match:          {arms_down_pass_pct:.1f}% ({arms_down_test_passes}/{arms_down_test_total} frames) [{'PASS' if pass_arms_down else 'FAIL'}]")
    print(f"\nOVERALL VALIDATION VERDICT:       [{'PASS' if overall_pass else 'FAIL'}]")

    print("\n--- Pipeline Latency Breakdown (Mean per Frame) ---")
    print(f"  1. MediaPipe Tracker:         {avg_timings['tracker_ms']:.2f} ms")
    print(f"  2. Canonical Adapter:         {avg_timings['adapter_ms']:.2f} ms")
    print(f"  3. Temporal Stabilizer:       {avg_timings['stabilizer_ms']:.2f} ms")
    print(f"  4. Anatomical Frame Builder:  {avg_timings['frame_builder_ms']:.2f} ms")
    print(f"  5. Local Rotation Extractor:  {avg_timings['extractor_ms']:.2f} ms")
    print(f"  6. Kinematic Retargeter & FK: {avg_timings['retargeter_fk_ms']:.2f} ms")
    print(f"  Total Retargeting Engine:     {avg_timings['frame_builder_ms'] + avg_timings['extractor_ms'] + avg_timings['retargeter_fk_ms']:.2f} ms")
    print(f"  Total Pipeline:               {avg_timings['total_ms']:.2f} ms ({1000.0 / max(avg_timings['total_ms'], 0.1):.1f} FPS)")
    print("================================================================================")

    # Save JSON Report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_video": str(video_path),
        "frames_processed": n_frames,
        "verdict": "PASS" if overall_pass else "FAIL",
        "quantitative_checks": {
            "fk_consistency_errors": fk_errors_count,
            "non_finite_values": non_finite_count,
            "label_swaps": label_swap_count,
            "bone_length_violations": bone_length_violation_count,
            "max_rotation_jump_deg": max_rotation_jump_deg,
        },
        "pose_fidelity": {
            "mean_arm_direction_error_deg": mean_arm_err,
            "mean_leg_direction_error_deg": mean_leg_err,
            "mean_torso_direction_error_deg": mean_torso_err,
            "arms_down_pass_pct": arms_down_pass_pct,
            "arms_down_frames_passed": arms_down_test_passes,
            "arms_down_frames_total": arms_down_test_total,
        },
        "latency_ms": avg_timings,
        "output_video": str(out_mp4_path),
    }

    report_path = out_dir / "retargeting_visual_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
