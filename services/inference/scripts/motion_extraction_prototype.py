"""
Phase 2.2 — Performer Motion Extraction Prototype

Extracts a normalized PerformerState from a static test image using the
MediaPipe Tasks API (face landmarks, pose, hands, segmentation).

This is a BENCHMARK and VALIDATION script — not production code.

Usage
-----
    python services/inference/scripts/motion_extraction_prototype.py

Output
------
    test_data/phase2_motion/
        debug_overlay.png       — Annotated frame: face mesh, pose, hands, masks
        performer_state.json    — Per-field summary of extracted PerformerState
        benchmark_report.json   — Per-stage latency (mean, P50, P95, FPS)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SERVICES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SERVICES_DIR.parent.parent
sys.path.insert(0, str(SERVICES_DIR))

# ── model paths ───────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "services" / "inference" / "models"
FACE_LANDMARKER_PATH   = str(MODELS_DIR / "face_landmarker.task")
POSE_LANDMARKER_PATH   = str(MODELS_DIR / "pose_landmarker_lite.task")
HAND_LANDMARKER_PATH   = str(MODELS_DIR / "hand_landmarker.task")
SEGMENTER_MODEL_PATH   = str(MODELS_DIR / "selfie_multiclass_256x256.tflite")
DETECTOR_MODEL_PATH    = str(MODELS_DIR / "blaze_face_short_range.tflite")

# ── test image ────────────────────────────────────────────────────────────────
TEST_IMAGE_PATH = str(PROJECT_ROOT / "test_data" / "2face_validation.png")

# ── output dir ────────────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "test_data" / "phase2_motion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BENCHMARK_ITERS = 30


# ==============================================================================
# Stage wrappers
# ==============================================================================

class FaceLandmarkerBackend:
    def __init__(self):
        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision

        base = tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def run(self, rgb: np.ndarray):
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._lm.detect(mp_img)


class PoseLandmarkerBackend:
    def __init__(self):
        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision

        base = tasks.BaseOptions(model_asset_path=POSE_LANDMARKER_PATH)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._pl = vision.PoseLandmarker.create_from_options(opts)

    def run(self, rgb: np.ndarray):
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._pl.detect(mp_img)


class HandLandmarkerBackend:
    def __init__(self):
        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision

        base = tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hl = vision.HandLandmarker.create_from_options(opts)

    def run(self, rgb: np.ndarray):
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._hl.detect(mp_img)


class SegmenterBackend:
    def __init__(self):
        from app.segmentation.segmenter import MediaPipeSegmenter
        self._seg = MediaPipeSegmenter(model_path=SEGMENTER_MODEL_PATH)

    def run(self, bgr: np.ndarray):
        return self._seg.segment(bgr)


# ==============================================================================
# Drawing helpers
# ==============================================================================

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_pose(canvas: np.ndarray, pose_result, h: int, w: int):
    if not pose_result.pose_landmarks:
        return
    for person_lms in pose_result.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in person_lms]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, pts[a], pts[b], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(canvas, pt, 4, (0, 200, 0), -1)


def draw_hands(canvas: np.ndarray, hand_result, h: int, w: int):
    if not hand_result.hand_landmarks:
        return
    for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        color = (255, 120, 0) if handedness[0].category_name == "Right" else (0, 120, 255)
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(canvas, pt, 4, color, -1)


def draw_face_mesh(canvas: np.ndarray, face_result, h: int, w: int):
    if not face_result.face_landmarks:
        return
    for face_lms in face_result.face_landmarks:
        for lm in face_lms:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(canvas, (px, py), 1, (180, 180, 255), -1)


def draw_segmentation(canvas: np.ndarray, seg_result, alpha: float = 0.35):
    if seg_result is None:
        return
    overlay = canvas.copy()
    if seg_result.hair_mask is not None:
        overlay[seg_result.hair_mask] = (0, 200, 200)      # yellow for hair
    if seg_result.face_mask is not None:
        overlay[seg_result.face_mask] = (80, 80, 255)       # red for face
    if seg_result.skin_mask is not None:
        overlay[seg_result.skin_mask] = (255, 160, 80)      # blue for body skin
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_head_orientation(canvas: np.ndarray, face_result, h: int, w: int):
    if not face_result.face_landmarks or not face_result.facial_transformation_matrixes:
        return
    for i, (face_lms, mat) in enumerate(
        zip(face_result.face_landmarks, face_result.facial_transformation_matrixes)
    ):
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)
        nose_pt = pts[1].astype(int)
        mat_arr = np.array(mat, dtype=np.float32)
        R = mat_arr[:3, :3]
        axes_len = 60.0
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        cx, cy = int(nose_pt[0]), int(nose_pt[1])
        cv2.arrowedLine(canvas, (cx, cy),
            (cx + int(x_axis[0] * axes_len), cy + int(x_axis[1] * axes_len)),
            (0, 0, 255), 2, tipLength=0.2)  # X red
        cv2.arrowedLine(canvas, (cx, cy),
            (cx + int(y_axis[0] * axes_len), cy + int(y_axis[1] * axes_len)),
            (0, 255, 0), 2, tipLength=0.2)  # Y green
        cv2.arrowedLine(canvas, (cx, cy),
            (cx - int(z_axis[0] * axes_len), cy - int(z_axis[1] * axes_len)),
            (255, 0, 0), 2, tipLength=0.2)  # Z blue (into camera)


# ==============================================================================
# Main benchmark and extraction routine
# ==============================================================================

def run_extraction_and_benchmark(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Initializing backends...")
    t0 = time.perf_counter()
    face_backend = FaceLandmarkerBackend()
    pose_backend = PoseLandmarkerBackend()
    hand_backend = HandLandmarkerBackend()
    seg_backend  = SegmenterBackend()
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"  Init time: {init_ms:.1f} ms")

    # ── warm-up ────────────────────────────────────────────────────────────────
    print("Warming up (3 passes)...")
    for _ in range(3):
        face_backend.run(rgb)
        pose_backend.run(rgb)
        hand_backend.run(rgb)
        seg_backend.run(bgr)

    # ── benchmark loop ─────────────────────────────────────────────────────────
    print(f"Benchmarking ({N_BENCHMARK_ITERS} iterations)...")
    face_times, pose_times, hand_times, seg_times, total_times = [], [], [], [], []

    face_result = pose_result = hand_result = seg_result = None

    for _ in range(N_BENCHMARK_ITERS):
        t_frame = time.perf_counter()

        t = time.perf_counter()
        face_result = face_backend.run(rgb)
        face_times.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        pose_result = pose_backend.run(rgb)
        pose_times.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        hand_result = hand_backend.run(rgb)
        hand_times.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        seg_result = seg_backend.run(bgr)
        seg_times.append((time.perf_counter() - t) * 1000)

        total_times.append((time.perf_counter() - t_frame) * 1000)

    def stats(times: list) -> dict:
        a = np.array(times)
        return {
            "mean_ms":   float(np.mean(a)),
            "p50_ms":    float(np.percentile(a, 50)),
            "p95_ms":    float(np.percentile(a, 95)),
            "min_ms":    float(np.min(a)),
            "max_ms":    float(np.max(a)),
        }

    benchmark = {
        "hardware":   "CPU (no GPU acceleration for MediaPipe Tasks API)",
        "image_size": f"{w}x{h}",
        "iterations": N_BENCHMARK_ITERS,
        "stages": {
            "face_landmarker":  stats(face_times),
            "pose_landmarker":  stats(pose_times),
            "hand_landmarker":  stats(hand_times),
            "segmentation":     stats(seg_times),
            "total_pipeline":   stats(total_times),
        },
        "fps_estimate": float(1000.0 / np.mean(total_times)),
    }

    # ── summarize extraction results ───────────────────────────────────────────
    num_faces = len(face_result.face_landmarks) if face_result and face_result.face_landmarks else 0
    num_poses = len(pose_result.pose_landmarks) if pose_result and pose_result.pose_landmarks else 0
    num_hands = len(hand_result.hand_landmarks) if hand_result and hand_result.hand_landmarks else 0

    blendshapes_present = (
        face_result is not None
        and face_result.face_blendshapes is not None
        and len(face_result.face_blendshapes) > 0
    )

    extraction_summary = {
        "frame_shape": [h, w, 3],
        "faces_detected": num_faces,
        "poses_detected": num_poses,
        "hands_detected": num_hands,
        "blendshapes_available": blendshapes_present,
        "blendshapes_per_face": (
            len(face_result.face_blendshapes[0]) if blendshapes_present else 0
        ),
        "segmentation_classes": {
            "face_mask_pixels":     int(seg_result.face_mask.sum()) if seg_result else 0,
            "hair_mask_pixels":     int(seg_result.hair_mask.sum()) if seg_result else 0,
            "body_skin_mask_pixels": int(seg_result.skin_mask.sum()) if seg_result else 0,
        },
    }

    if num_faces > 0 and face_result.facial_transformation_matrixes:
        mat = np.array(face_result.facial_transformation_matrixes[0], dtype=np.float32)
        if mat.shape == (4, 4):
            R = mat[:3, :3]
            r20 = float(np.clip(-R[2, 0], -1, 1))
            yaw = float(np.degrees(np.arcsin(r20)))
            pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            extraction_summary["primary_face_head_rotation"] = {
                "pitch_deg": round(pitch, 2),
                "yaw_deg":   round(yaw, 2),
                "roll_deg":  round(roll, 2),
            }

    # ── draw debug overlay ─────────────────────────────────────────────────────
    canvas = bgr.copy()
    draw_segmentation(canvas, seg_result)
    draw_pose(canvas, pose_result, h, w)
    draw_hands(canvas, hand_result, h, w)
    draw_face_mesh(canvas, face_result, h, w)
    draw_head_orientation(canvas, face_result, h, w)

    # Legend
    labels = [
        ("Face Mesh (478 pts)", (180, 180, 255)),
        ("Head Axes: X=Red, Y=Green, Z=Blue", (220, 220, 220)),
        ("Body Pose (green)", (0, 220, 0)),
        ("Right Hand (orange)", (255, 120, 0)),
        ("Left Hand (blue)", (0, 120, 255)),
        ("Seg: face=red, hair=yellow, skin=blue-ish", (200, 200, 200)),
    ]
    for i, (label, color) in enumerate(labels):
        cv2.putText(canvas, label, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Per-stage latency HUD
    total_mean = np.mean(total_times)
    fps = 1000.0 / total_mean
    hud_lines = [
        f"Face:  {np.mean(face_times):.1f} ms",
        f"Pose:  {np.mean(pose_times):.1f} ms",
        f"Hands: {np.mean(hand_times):.1f} ms",
        f"Seg:   {np.mean(seg_times):.1f} ms",
        f"Total: {total_mean:.1f} ms  ({fps:.1f} FPS)",
    ]
    for i, line in enumerate(hud_lines):
        y = h - 110 + i * 20
        cv2.putText(canvas, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    debug_overlay_path = str(OUT_DIR / "debug_overlay.png")
    cv2.imwrite(debug_overlay_path, canvas)
    print(f"  Saved debug overlay: {debug_overlay_path}")

    return benchmark, extraction_summary


def main():
    print("=" * 72)
    print("  Phase 2.2 — Performer Motion Extraction Prototype")
    print("=" * 72)

    bgr = cv2.imread(TEST_IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE_PATH}")
    print(f"Loaded test image: {TEST_IMAGE_PATH}  ({bgr.shape[1]}x{bgr.shape[0]})")

    benchmark, extraction = run_extraction_and_benchmark(bgr)

    # ── save JSON reports ──────────────────────────────────────────────────────
    bench_path = str(OUT_DIR / "benchmark_report.json")
    state_path = str(OUT_DIR / "performer_state_summary.json")

    with open(bench_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    with open(state_path, "w") as f:
        json.dump(extraction, f, indent=2)

    # ── print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXTRACTION SUMMARY")
    print("=" * 72)
    print(f"  Faces detected:    {extraction['faces_detected']}")
    print(f"  Poses detected:    {extraction['poses_detected']}")
    print(f"  Hands detected:    {extraction['hands_detected']}")
    print(f"  Blendshapes:       {extraction['blendshapes_per_face']} per face")
    if "primary_face_head_rotation" in extraction:
        r = extraction["primary_face_head_rotation"]
        print(f"  Head rotation:     pitch={r['pitch_deg']}  yaw={r['yaw_deg']}  roll={r['roll_deg']}")

    print("\n" + "=" * 72)
    print("  BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  {'Stage':<25} {'Mean ms':>10} {'P50 ms':>10} {'P95 ms':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for stage, s in benchmark["stages"].items():
        print(f"  {stage:<25} {s['mean_ms']:>10.1f} {s['p50_ms']:>10.1f} {s['p95_ms']:>10.1f}")
    print(f"\n  Estimated FPS (pipeline total): {benchmark['fps_estimate']:.1f}")
    print(f"\n  Benchmark JSON:  {bench_path}")
    print(f"  State summary:   {state_path}")
    print(f"  Debug overlay:   {OUT_DIR / 'debug_overlay.png'}")


if __name__ == "__main__":
    main()
