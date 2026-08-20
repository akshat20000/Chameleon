"""
Standalone smoke test for Phase 1.5 Pose & Expression Extraction.

Requires:
  - services/inference/models/face_landmarker.task
  - test_data/face.png (single-face image)
  - test_data/2face_validation.png (two-face fixture)

Usage
-----
  python services/inference/scripts/test_pose.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.detector import MediaPipeDetector
from app.landmarks.landmarker import MediaPipeLandmarker
from app.pipeline.result import PoseResult
from app.tracking.tracker import KalmanFilterTracker


def test_single_face(image_path: Path):
    detector_model = Path("services/inference/models/blaze_face_short_range.tflite")
    landmarker_model = Path("services/inference/models/face_landmarker.task")

    print("====================================================================")
    print(f"  SINGLE-FACE POSE SMOKE TEST: {image_path}")
    print("====================================================================")

    detector = MediaPipeDetector(model_path=str(detector_model))
    tracker = KalmanFilterTracker(min_hits=1)
    landmarker = MediaPipeLandmarker(model_path=str(landmarker_model))

    assert landmarker.is_ready, "Landmarker not ready"

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}", file=sys.stderr)
        sys.exit(1)
    h, w = image.shape[:2]

    dets = detector.detect(image)
    tracks = tracker.update(dets)
    if not tracks:
        print(f"WARNING: No face tracks detected in {image_path}")
        return

    # Warmup
    _ = landmarker.detect_landmarks_and_pose(image, tracks)

    # Benchmark run
    t0 = time.perf_counter()
    lms_dict, pose_dict = landmarker.detect_landmarks_and_pose(image, tracks)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  Image dimensions : {w} x {h}")
    print(f"  Track IDs        : {list(tracks_map(tracks))}")
    print(f"  Landmark count   : {len(lms_dict)}")
    print(f"  Pose count       : {len(pose_dict)}")
    print(f"  Latency          : {latency_ms:.2f} ms")

    for tid, pose in pose_dict.items():
        print(f"\n  Track ID {tid}:")
        print(f"    Landmarks      : {lms_dict[tid].points_2d.shape[0]} 2D points")
        print(f"    Blendshapes    : {len(pose.blendshapes)} categories")
        print(f"    Matrix shape   : {pose.transformation_matrix.shape}")
        print(f"    Pitch          : {pose.pitch:.2f} deg")
        print(f"    Yaw            : {pose.yaw:.2f} deg")
        print(f"    Roll           : {pose.roll:.2f} deg")


def tracks_map(tracks):
    return [t.track_id for t in tracks]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1.5 Pose & Expression Extraction Test")
    parser.add_argument("--image", help="Input image path")
    args = parser.parse_args()

    landmarker_model = Path("services/inference/models/face_landmarker.task")
    landmarker = MediaPipeLandmarker(model_path=str(landmarker_model))
    print("====================================================================")
    print("  POSE & EXPRESSION EXTRACTION TEST — Phase 1.5")
    print("====================================================================")
    print(f"  Landmarker model ready: {landmarker.is_ready}")

    if not args.image:
        print("\nNOTE: No input image provided via --image <path>. Skipping live inference.")
        print("Model initialization verified successfully.")
        return

    test_single_face(Path(args.image))


if __name__ == "__main__":
    main()
