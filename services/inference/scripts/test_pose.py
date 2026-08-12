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


def test_single_face():
    img_path = Path("test_data/face.png")
    detector_model = Path("services/inference/models/blaze_face_short_range.tflite")
    landmarker_model = Path("services/inference/models/face_landmarker.task")

    print("====================================================================")
    print("  SINGLE-FACE POSE SMOKE TEST: test_data/face.png")
    print("====================================================================")

    detector = MediaPipeDetector(model_path=str(detector_model))
    tracker = KalmanFilterTracker(min_hits=1)
    landmarker = MediaPipeLandmarker(model_path=str(landmarker_model))

    assert landmarker.is_ready, "Landmarker not ready"

    image = cv2.imread(str(img_path))
    assert image is not None, "Failed to load test_data/face.png"
    h, w = image.shape[:2]

    dets = detector.detect(image)
    tracks = tracker.update(dets)
    assert len(tracks) > 0, "No tracks generated"

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
        print(f"    Matrix 4x4:\n{pose.transformation_matrix}")


def test_multi_face():
    img_path = Path("test_data/2face_validation.png")
    detector_model = Path("services/inference/models/blaze_face_short_range.tflite")
    landmarker_model = Path("services/inference/models/face_landmarker.task")

    print("\n====================================================================")
    print("  MULTI-FACE POSE SMOKE TEST: test_data/2face_validation.png")
    print("====================================================================")

    detector = MediaPipeDetector(model_path=str(detector_model))
    tracker = KalmanFilterTracker(min_hits=1)
    landmarker = MediaPipeLandmarker(model_path=str(landmarker_model))

    assert landmarker.is_ready, "Landmarker not ready"

    image = cv2.imread(str(img_path))
    assert image is not None, "Failed to load test_data/2face_validation.png"
    h, w = image.shape[:2]

    dets = detector.detect(image)
    tracks = tracker.update(dets)

    print(f"  Image dimensions : {w} x {h}")
    print(f"  Detector count   : {len(dets)}")
    print(f"  Tracker count    : {len(tracks)}")
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"

    t0 = time.perf_counter()
    lms_dict, pose_dict = landmarker.detect_landmarks_and_pose(image, tracks)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  Landmark results : {len(lms_dict)}")
    print(f"  Pose results     : {len(pose_dict)}")
    print(f"  Latency          : {latency_ms:.2f} ms")

    assert len(lms_dict) == 2, f"Expected 2 landmark results, got {len(lms_dict)}"
    assert len(pose_dict) == 2, f"Expected 2 pose results, got {len(pose_dict)}"

    for tid in sorted(pose_dict.keys()):
        lm = lms_dict[tid]
        pose = pose_dict[tid]
        assert lm.points_2d.shape[0] == 478
        assert lm.points_3d.shape[0] == 478
        assert len(pose.blendshapes) == 52
        assert pose.transformation_matrix.shape == (4, 4)
        assert np.all(np.isfinite(pose.transformation_matrix))
        assert np.all(np.isfinite([pose.pitch, pose.yaw, pose.roll]))
        assert all(np.isfinite(val) for val in pose.blendshapes.values())

        print(f"\n  Track ID {tid}:")
        print(f"    Landmarks      : {lm.points_2d.shape[0]} pts")
        print(f"    Blendshapes    : {len(pose.blendshapes)} categories")
        print(f"    Pitch          : {pose.pitch:.2f} deg")
        print(f"    Yaw            : {pose.yaw:.2f} deg")
        print(f"    Roll           : {pose.roll:.2f} deg")


def tracks_map(tracks):
    return [t.track_id for t in tracks]


def main():
    test_single_face()
    test_multi_face()


if __name__ == "__main__":
    main()
