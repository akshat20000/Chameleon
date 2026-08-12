"""
Standalone smoke test for MediaPipeLandmarker.

Requires:
  - services/inference/models/face_landmarker.task  (must be downloaded first)
  - A webcam (default) OR an image path passed as a positional argument.

Usage
-----
  # Webcam mode (press 'q' to quit):
  python services/inference/scripts/test_landmarker.py

  # Single-image mode:
  python services/inference/scripts/test_landmarker.py path/to/image.jpg

  # Override model path:
  python services/inference/scripts/test_landmarker.py --model path/to/face_landmarker.task

Model download (manual step — see repository conventions):
  curl -L -o services/inference/models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Output columns printed per frame:
  image W x H | tracked_faces | landmark_results | track_ids |
  lm_count | first 3 pts_2d | first 3 pts_3d | latency_ms
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from project root or from scripts/ directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.detector import MediaPipeDetector
from app.landmarks.landmarker import MediaPipeLandmarker
from app.tracking.tracker import KalmanFilterTracker


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: MediaPipe face detection + tracking + landmarking."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to an image file.  If omitted, opens webcam 0.",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"),
        help="Path to face_landmarker.task model file.",
    )
    parser.add_argument(
        "--detector-model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"),
        help="Path to blaze_face_short_range.tflite model file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-frame reporting
# ---------------------------------------------------------------------------

def _report_frame(
    frame: np.ndarray,
    tracks,
    lm_results: dict,
    latency_ms: float,
) -> None:
    h, w = frame.shape[:2]
    n_tracks = len(tracks)
    n_results = len(lm_results)
    track_ids = sorted(lm_results.keys())

    print(f"\n{'─' * 60}")
    print(f"  Image         : {w} x {h}")
    print(f"  Tracked faces : {n_tracks}")
    print(f"  Landmark hits : {n_results}")
    print(f"  Track IDs     : {track_ids}")

    for tid in track_ids:
        lr = lm_results[tid]
        print(f"\n  Track {tid}:")
        print(f"    Landmark count (2D): {lr.points_2d.shape[0]}")
        print(f"    Landmark count (3D): {lr.points_3d.shape[0]}")
        print(f"    landmarks_type     : {lr.landmarks_type}")
        print(f"    confidence         : {lr.confidence}")
        print(f"    First 3 pts_2d     :")
        for i in range(min(3, len(lr.points_2d))):
            x, y = lr.points_2d[i]
            print(f"      [{i}] x={x:.1f}  y={y:.1f}")
        print(f"    First 3 pts_3d     :")
        for i in range(min(3, len(lr.points_3d))):
            x, y, z = lr.points_3d[i]
            print(f"      [{i}] x={x:.1f}  y={y:.1f}  z={z:.4f}")

    print(f"\n  Latency       : {latency_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_overlay(frame: np.ndarray, tracks, lm_results: dict, latency_ms: float) -> None:
    for tr in tracks:
        b = tr.bbox
        x1, y1 = int(b.x_min), int(b.y_min)
        x2, y2 = int(b.x_max), int(b.y_max)
        has_lm = tr.track_id in lm_results
        color = (0, 255, 0) if has_lm else (0, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tr.track_id} {'LM' if has_lm else 'no-lm'}"
        cv2.putText(frame, label, (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for lr in lm_results.values():
        pts = lr.points_2d.astype(np.int32)
        for pt in pts[::10]:   # draw every 10th landmark to keep it readable
            cv2.circle(frame, (pt[0], pt[1]), 1, (0, 200, 255), -1)

    cv2.putText(frame, f"{latency_ms:.1f} ms", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ---- Initialize components ----
    print(f"Detector model : {args.detector_model}")
    print(f"Landmarker model: {args.model}")

    detector = MediaPipeDetector(model_path=args.detector_model)
    if detector.detector is None:
        print("ERROR: Face detector model not found. Check --detector-model path.")
        sys.exit(1)

    landmarker = MediaPipeLandmarker(model_path=args.model)
    if not landmarker.is_ready:
        print(
            f"ERROR: face_landmarker.task not found at: {args.model}\n"
            "Download it with:\n"
            "  curl -L -o services/inference/models/face_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        sys.exit(1)

    tracker = KalmanFilterTracker(min_hits=1)

    # ---- Single-image mode ----
    if args.image is not None:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: could not read image: {args.image}")
            sys.exit(1)

        dets = detector.detect(frame)
        tracks = tracker.update(dets)
        t0 = time.perf_counter()
        lm_results = landmarker.detect(frame, tracks)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        _report_frame(frame, tracks, lm_results, latency_ms)
        _draw_overlay(frame, tracks, lm_results, latency_ms)

        cv2.imshow("Landmarker Smoke Test (any key to exit)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ---- Webcam mode ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam 0.")
        sys.exit(1)

    print("Webcam mode — press 'q' to quit.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        tracks = tracker.update(dets)

        t0 = time.perf_counter()
        lm_results = landmarker.detect(frame, tracks)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if frame_idx % 30 == 0:  # print to console every 30 frames
            _report_frame(frame, tracks, lm_results, latency_ms)

        _draw_overlay(frame, tracks, lm_results, latency_ms)
        cv2.imshow("Landmarker Smoke Test (q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
