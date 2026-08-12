"""
Standalone smoke test and performance benchmark for ONNXIdentityEncoder.

Requires:
  - services/inference/models/w600k_mbf.onnx
  - test_data/face.png
  - test_data/2face_validation.png

Usage
-----
  python services/inference/scripts/test_identity.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.detector import MediaPipeDetector
from app.identity.encoder import (
    ONNXIdentityEncoder,
    align_face_5pt,
    extract_5pt_landmarks_from_478,
    fuse_embeddings,
)
from app.landmarks.landmarker import MediaPipeLandmarker
from app.tracking.tracker import KalmanFilterTracker


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def extract_aligned_face(image: np.ndarray, detector, tracker, landmarker):
    dets = detector.detect(image)
    tracks = tracker.update(dets)
    lms_dict, _ = landmarker.detect_landmarks_and_pose(image, tracks)

    chips = {}
    for tid, lm_res in lms_dict.items():
        pts5 = extract_5pt_landmarks_from_478(lm_res.points_2d)
        if pts5 is not None:
            aligned = align_face_5pt(image, pts5, target_size=(112, 112))
            if aligned is not None:
                chips[tid] = aligned
    return chips


def main():
    detector_model = Path("services/inference/models/blaze_face_short_range.tflite")
    landmarker_model = Path("services/inference/models/face_landmarker.task")
    identity_model = Path("services/inference/models/w600k_mbf.onnx")

    print("====================================================================")
    print("  IDENTITY ENCODER SMOKE TEST — Phase 1.6")
    print("====================================================================")
    print(f"  Model path   : {identity_model.resolve()}")
    print(f"  Model exists : {identity_model.exists()}")

    # Measure initialization latency separately
    t_init_0 = time.perf_counter()
    encoder = ONNXIdentityEncoder(model_path=str(identity_model))
    init_latency_ms = (time.perf_counter() - t_init_0) * 1000.0

    print(f"  Encoder ready: {encoder.is_ready}")
    print(f"  Init Latency : {init_latency_ms:.2f} ms")

    if not encoder.is_ready:
        print("ERROR: IdentityEncoder is not ready. Aborting test.")
        sys.exit(1)

    detector = MediaPipeDetector(model_path=str(detector_model))
    tracker = KalmanFilterTracker(min_hits=1)
    landmarker = MediaPipeLandmarker(model_path=str(landmarker_model))

    img_single = cv2.imread("test_data/face.png")
    assert img_single is not None, "Failed to load test_data/face.png"

    chips_single = extract_aligned_face(img_single, detector, tracker, landmarker)
    assert len(chips_single) > 0, "Failed to extract face chip from test_data/face.png"
    chip_single = list(chips_single.values())[0]

    print(f"  Face chip shape : {chip_single.shape} (dtype={chip_single.dtype})")

    # Single-face extraction
    t0 = time.perf_counter()
    emb_single = encoder.extract_embedding(chip_single)
    single_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert emb_single is not None, "Failed to extract embedding"
    print("\n--------------------------------------------------------------------")
    print("  SINGLE-FACE EMBEDDING RESULTS")
    print("--------------------------------------------------------------------")
    print(f"  Embedding dimension : {emb_single.shape[0]}")
    print(f"  Dtype               : {emb_single.dtype}")
    print(f"  L2 Norm             : {np.linalg.norm(emb_single):.6f}")
    print(f"  Values Finite       : {np.all(np.isfinite(emb_single))}")
    print(f"  Latency             : {single_latency_ms:.2f} ms")

    # Multi-face image test (Person A vs Person B)
    img_multi = cv2.imread("test_data/2face_validation.png")
    chips_multi = extract_aligned_face(img_multi, detector, tracker, landmarker)

    embs_multi = {}
    for tid, chip in chips_multi.items():
        embs_multi[tid] = encoder.extract_embedding(chip)

    print("\n--------------------------------------------------------------------")
    print("  MULTI-FACE EMBEDDING & COSINE SIMILARITY TEST")
    print("--------------------------------------------------------------------")
    for tid, emb in embs_multi.items():
        print(f"  Track ID {tid}: shape={emb.shape}, norm={np.linalg.norm(emb):.6f}")

    if len(embs_multi) >= 2:
        tids = list(embs_multi.keys())
        sim_diff = cosine_similarity(embs_multi[tids[0]], embs_multi[tids[1]])
        print(f"\n  Cosine Similarity (Person {tids[0]} vs Person {tids[1]} - DIFFERENT PERSONS): {sim_diff:.4f}")

    # Same person rotated comparison
    h, w = img_single.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
    img_rot = cv2.warpAffine(img_single, M_rot, (w, h))
    chips_rot = extract_aligned_face(img_rot, detector, tracker, landmarker)
    if len(chips_rot) > 0:
        chip_rot = list(chips_rot.values())[0]
        emb_rot = encoder.extract_embedding(chip_rot)
        sim_same = cosine_similarity(emb_single, emb_rot)
        print(f"  Cosine Similarity (Same Person - Baseline vs +10 deg rotated): {sim_same:.4f}")

    # Multi-reference fusion test
    if len(chips_rot) > 0:
        fused = fuse_embeddings([emb_single, emb_rot])
        print(f"\n  Multi-Reference Fusion (2 embeddings): fused shape={fused.shape}, norm={np.linalg.norm(fused):.6f}")

    # Latency Benchmark across 30 iterations
    print("\n--------------------------------------------------------------------")
    print("  STEADY-STATE EMBEDDING LATENCY BENCHMARK (30 iterations)")
    print("--------------------------------------------------------------------")
    latencies = []
    for _ in range(30):
        t_b0 = time.perf_counter()
        _ = encoder.extract_embedding(chip_single)
        latencies.append((time.perf_counter() - t_b0) * 1000.0)

    latencies = np.array(latencies)
    print(f"  Min latency    : {np.min(latencies):.2f} ms")
    print(f"  Mean latency   : {np.mean(latencies):.2f} ms")
    print(f"  Median latency : {np.median(latencies):.2f} ms")
    print(f"  P95 latency    : {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency    : {np.percentile(latencies, 99):.2f} ms")
    print(f"  Max latency    : {np.max(latencies):.2f} ms")

    print("\n  IdentityEncoder smoke test finished successfully!")


if __name__ == "__main__":
    main()
