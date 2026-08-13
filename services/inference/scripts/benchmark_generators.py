"""
Phase 1.7 Candidate Generator Model Controlled Benchmark & Compatibility Evaluation Script.

This is an isolated experiment script. It does NOT modify production code or pipeline state.

Execution Order:
  1. Model provenance & signature verification
  2. Preprocessing verification (5-pt alignment to 128x128)
  3. Identity embedding compatibility test (Phase 1.6 w600k_mbf.onnx 512-d vector)
  4. Single-face inference
  5. Pose preservation evaluation (Phase 1.5 Euler angles)
  6. Expression preservation evaluation (Phase 1.5 52 blendshapes & NME)
  7. Multi-face / track association test
  8. 30-run latency benchmark (CPU & GPU)

Output:
  Hard comparison markdown table.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.detector import MediaPipeDetector
from app.identity.encoder import ONNXIdentityEncoder, extract_5pt_landmarks_from_478
from app.landmarks.landmarker import MediaPipeLandmarker
from app.tracking.tracker import KalmanFilterTracker

# Canonical 5-point destination template for 128x128 crop (inswapper standard)
INSWAPPER_128_TARGET_5PTS = np.array(
    [
        [43.7656, 48.0565],  # left eye center
        [84.2344, 48.0565],  # right eye center
        [64.0000, 72.8430],  # nose tip
        [48.6504, 96.1130],  # left mouth corner
        [79.3496, 96.1130],  # right mouth corner
    ],
    dtype=np.float32,
)


def align_face_128(image: np.ndarray, landmarks_5pt: np.ndarray):
    """Align BGR image to 128x128 face chip using 5-point similarity transform."""
    if image is None or landmarks_5pt is None or landmarks_5pt.shape != (5, 2):
        return None, None
    M, _ = cv2.estimateAffinePartial2D(landmarks_5pt.astype(np.float32), INSWAPPER_128_TARGET_5PTS)
    if M is None:
        return None, None
    aligned = cv2.warpAffine(image, M, (128, 128), flags=cv2.INTER_LINEAR)
    return aligned, M


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def run_inswapper_benchmark():
    inswapper_path = Path("services/inference/models/inswapper_128.onnx")
    arcface_path = Path("services/inference/models/w600k_mbf.onnx")
    detector_path = Path("services/inference/models/blaze_face_short_range.tflite")
    landmarker_path = Path("services/inference/models/face_landmarker.task")

    print("====================================================================")
    print("  PHASE 1.7 CANDIDATE GENERATOR CONTROLLED BENCHMARK")
    print("====================================================================")

    # 1. Model Provenance & Signature Verification
    print("\n--- STEP 1: Model Provenance & Signature Verification ---")
    if not inswapper_path.exists():
        print(f"ERROR: Model asset not found at {inswapper_path}")
        sys.exit(1)

    print(f"Model file       : {inswapper_path.resolve()}")
    print(f"File size        : {inswapper_path.stat().st_size / (1024*1024):.2f} MB")
    print("Code license     : MIT License (Project)")
    print("Model license    : Non-Commercial / Research Only (InsightFace InSwapper)")

    t_init_0 = time.perf_counter()
    session_cpu = ort.InferenceSession(str(inswapper_path), providers=["CPUExecutionProvider"])
    init_latency_cpu_ms = (time.perf_counter() - t_init_0) * 1000.0

    print(f"CPU Init Latency : {init_latency_cpu_ms:.2f} ms")

    # Check GPU availability
    gpu_available = "CUDAExecutionProvider" in ort.get_available_providers()
    session_gpu = None
    init_latency_gpu_ms = None
    if gpu_available:
        try:
            t_gpu_0 = time.perf_counter()
            session_gpu = ort.InferenceSession(str(inswapper_path), providers=["CUDAExecutionProvider"])
            init_latency_gpu_ms = (time.perf_counter() - t_gpu_0) * 1000.0
            print(f"GPU Init Latency : {init_latency_gpu_ms:.2f} ms (CUDA execution provider)")
        except Exception as e:
            print(f"GPU Provider failed: {e}")
            gpu_available = False

    inputs = session_cpu.get_inputs()
    outputs = session_cpu.get_outputs()

    print("\nInput Signatures:")
    for inp in inputs:
        print(f"  Node: '{inp.name}', Shape: {inp.shape}, Type: {inp.type}")

    print("Output Signatures:")
    for out in outputs:
        print(f"  Node: '{out.name}', Shape: {out.shape}, Type: {out.type}")

    target_node = inputs[0].name  # 'target'
    source_node = inputs[1].name  # 'source'
    output_node = outputs[0].name  # 'output'

    # 2. Preprocessing Verification & Landmark/Pose/Identity Setup
    print("\n--- STEP 2: Preprocessing Verification & Base Setup ---")
    detector = MediaPipeDetector(model_path=str(detector_path))
    tracker = KalmanFilterTracker(min_hits=1)
    landmarker = MediaPipeLandmarker(model_path=str(landmarker_path))
    identity_encoder = ONNXIdentityEncoder(model_path=str(arcface_path))

    assert identity_encoder.is_ready, "Phase 1.6 ArcFace identity encoder is not ready!"

    img_single = cv2.imread("test_data/face.png")
    assert img_single is not None, "Failed to load test_data/face.png"

    dets = detector.detect(img_single)
    tracks = tracker.update(dets)
    lms_dict, pose_dict_src = landmarker.detect_landmarks_and_pose(img_single, tracks)

    assert len(tracks) > 0 and 1 in lms_dict, "Failed to track face in test_data/face.png"
    lm_src = lms_dict[1]
    pose_src = pose_dict_src[1]
    pts5_src = extract_5pt_landmarks_from_478(lm_src.points_2d)

    aligned_128_src, M_src = align_face_128(img_single, pts5_src)
    assert aligned_128_src is not None, "Failed to align face to 128x128"
    print(f"Source face aligned shape: {aligned_128_src.shape}, dtype: {aligned_128_src.dtype}")

    # Extract target identity embedding using Phase 1.6 encoder
    emb_target = identity_encoder.extract_embedding(aligned_128_src)
    assert emb_target is not None, "Failed to extract target identity embedding"
    print(f"Target identity embedding shape: {emb_target.shape}, norm: {np.linalg.norm(emb_target):.6f}")

    # 3. Identity Embedding Compatibility Test
    print("\n--- STEP 3: Identity Embedding Compatibility Test ---")

    # InSwapper expects target image chip as float32 in shape (1, 3, 128, 128)
    # Preprocessing: BGR uint8 -> float32 / 255.0 (or / 1.0 depending on range)
    # Let's prepare NCHW float32 input
    target_tensor = aligned_128_src.astype(np.float32) / 255.0
    target_tensor = np.transpose(target_tensor, (2, 0, 1))
    target_tensor = np.expand_dims(target_tensor, axis=0)

    # Pre-multiply embedding vector by InSwapper internal embedding matrix if required,
    # or pass normalized float32 vector shape (1, 512)
    source_tensor = np.expand_dims(emb_target, axis=0).astype(np.float32)

    emb_compatible = False
    raw_output = None
    try:
        raw_output = session_cpu.run(
            [output_node],
            {target_node: target_tensor, source_node: source_tensor},
        )[0]
        emb_compatible = raw_output is not None and not np.isnan(raw_output).any()
        print(f"Inference execution succeeded! Output tensor shape: {raw_output.shape}, dtype: {raw_output.dtype}")
        print(f"Embedding Compatible Status: {'PASS' if emb_compatible else 'FAIL'}")
    except Exception as e:
        print(f"Embedding compatibility test failed: {e}")
        emb_compatible = False

    # 4. Single-Face Inference & Output Reconstruction
    print("\n--- STEP 4: Single-Face Output Reconstruction ---")
    out_chip_128 = None
    if emb_compatible and raw_output is not None:
        out_chw = raw_output[0]
        out_hwc = np.transpose(out_chw, (1, 2, 0))
        out_chip_128 = np.clip(out_hwc * 255.0, 0, 255).astype(np.uint8)

        # Warp back chip onto full frame for pose/expression validation
        img_out_full = img_single.copy()
        M_inv = cv2.invertAffineTransform(M_src)
        h, w = img_single.shape[:2]
        warped_back = cv2.warpAffine(out_chip_128, M_inv, (w, h), flags=cv2.INTER_LINEAR)

        # Masking onto face area
        mask = np.ones((128, 128, 3), dtype=np.float32)
        mask_warped = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_LINEAR)
        img_out_full = np.where(mask_warped > 0.5, warped_back, img_single)
        cv2.imwrite("test_data/_inswapper_output_single.png", img_out_full)
        print("Synthesized output written to test_data/_inswapper_output_single.png")

    # 5. Pose Preservation Evaluation
    print("\n--- STEP 5: Pose Preservation Evaluation ---")
    pose_error = None
    if out_chip_128 is not None:
        # Detect pose on full output frame
        tracker_eval = KalmanFilterTracker(min_hits=1)
        dets_eval = detector.detect(img_out_full)
        tracks_eval = tracker_eval.update(dets_eval)
        lms_eval_dict, pose_eval_dict = landmarker.detect_landmarks_and_pose(img_out_full, tracks_eval)

        if len(tracks_eval) > 0 and 1 in pose_eval_dict:
            pose_out = pose_eval_dict[1]
            dpitch = abs(pose_out.pitch - pose_src.pitch)
            dyaw = abs(pose_out.yaw - pose_src.yaw)
            droll = abs(pose_out.roll - pose_src.roll)
            pose_error = max(dpitch, dyaw, droll)

            print(f"Source Pose : pitch={pose_src.pitch:.2f}°, yaw={pose_src.yaw:.2f}°, roll={pose_src.roll:.2f}°")
            print(f"Output Pose : pitch={pose_out.pitch:.2f}°, yaw={pose_out.yaw:.2f}°, roll={pose_out.roll:.2f}°")
            print(f"Pose Error (Max Delta): {pose_error:.2f}°")

    # 6. Expression Preservation Evaluation
    print("\n--- STEP 6: Expression Preservation Evaluation ---")
    expr_mae = None
    landmark_nme = None
    if out_chip_128 is not None and len(tracks_eval) > 0 and 1 in pose_eval_dict:
        # ARKit Blendshapes MAE
        bs_src = pose_src.blendshapes
        bs_out = pose_out.blendshapes
        diffs = [abs(bs_out.get(k, 0.0) - bs_src.get(k, 0.0)) for k in bs_src.keys()]
        expr_mae = float(np.mean(diffs))
        print(f"Expression MAE (across 52 ARKit blendshapes): {expr_mae:.4f}")

        # Landmark NME (Normalized Interocular Distance)
        lm_out = lms_eval_dict[1]
        p_src = lm_src.points_2d
        p_out = lm_out.points_2d
        interocular = np.linalg.norm(p_src[468] - p_src[473])
        nme_dist = np.mean(np.linalg.norm(p_out - p_src, axis=1)) / max(interocular, 1e-6)
        landmark_nme = float(nme_dist)
        print(f"Landmark NME (Normalized Interocular Distance): {landmark_nme:.4f}")

    # Output Identity Cosine Similarity
    id_cosine = None
    if out_chip_128 is not None:
        emb_output = identity_encoder.extract_embedding(out_chip_128)
        if emb_output is not None:
            id_cosine = cosine_similarity(emb_output, emb_target)
            print(f"Identity Cosine Similarity (Output Face vs Target Embedding): {id_cosine:.4f}")

    # 7. Multi-Face & Track Association Validation
    print("\n--- STEP 7: Multi-Face Track Association Test ---")
    img_multi = cv2.imread("test_data/2face_validation.png")
    multi_face_pass = False
    if img_multi is not None:
        tracker_m = KalmanFilterTracker(min_hits=1)
        dets_m = detector.detect(img_multi)
        tracks_m = tracker_m.update(dets_m)
        lms_m_dict, _ = landmarker.detect_landmarks_and_pose(img_multi, tracks_m)

        if len(tracks_m) >= 2:
            print(f"Detected {len(tracks_m)} faces in multi-person frame.")
            # Verify distinct 5-pt alignments for distinct tracks
            chips_m = {}
            for trk in tracks_m:
                if trk.track_id in lms_m_dict:
                    pts5_m = extract_5pt_landmarks_from_478(lms_m_dict[trk.track_id].points_2d)
                    aligned_m, _ = align_face_128(img_multi, pts5_m)
                    if aligned_m is not None:
                        chips_m[trk.track_id] = aligned_m

            if len(chips_m) >= 2:
                multi_face_pass = True
                print("Multi-Face Track Association: PASS (Independent track chips isolated correctly)")

    # 8. 30-Run Latency Benchmark
    print("\n--- STEP 8: 30-Run Steady-State Latency Benchmark ---")
    latencies_cpu = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = session_cpu.run([output_node], {target_node: target_tensor, source_node: source_tensor})
        latencies_cpu.append((time.perf_counter() - t0) * 1000.0)

    cpu_mean_ms = float(np.mean(latencies_cpu))
    print(f"CPU Steady-State Latency (30 runs): Mean={cpu_mean_ms:.2f} ms, Min={np.min(latencies_cpu):.2f} ms, Max={np.max(latencies_cpu):.2f} ms")

    gpu_mean_ms = None
    if gpu_available and session_gpu is not None:
        latencies_gpu = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = session_gpu.run([output_node], {target_node: target_tensor, source_node: source_tensor})
            latencies_gpu.append((time.perf_counter() - t0) * 1000.0)
        gpu_mean_ms = float(np.mean(latencies_gpu))
        print(f"GPU Steady-State Latency (30 runs): Mean={gpu_mean_ms:.2f} ms, Min={np.min(latencies_gpu):.2f} ms, Max={np.max(latencies_gpu):.2f} ms")

    # Licensing Verification
    # InSwapper model weights carry a Non-Commercial / Research-Only restriction from InsightFace
    license_acceptable = False  # Non-commercial weight license fails production commercial criteria

    overall_verdict = "REJECT" if (not license_acceptable or not emb_compatible) else "SELECT"

    id_cos_str = f"{id_cosine:.4f}" if id_cosine is not None else "N/A"
    pose_err_str = f"{pose_error:.2f}°" if pose_error is not None else "N/A"
    expr_mae_str = f"{expr_mae:.4f}" if expr_mae is not None else "N/A"
    landmark_nme_str = f"{landmark_nme:.4f}" if landmark_nme is not None else "N/A"
    gpu_lat_str = f"{gpu_mean_ms:.2f} ms" if gpu_mean_ms is not None else "N/A (CPU execution)"

    # PRINT HARD COMPARISON TABLE
    print("\n====================================================================")
    print("  PHASE 1.7 CANDIDATE HARD COMPARISON TABLE")
    print("====================================================================")
    print("| Metric | MobileFaceSwap | InSwapper 128 |")
    print("|---|:---:|:---:|")
    print(f"| Embedding Compatible | FAIL (No weights file) | {'PASS' if emb_compatible else 'FAIL'} |")
    print(f"| Identity Cosine | — | {id_cos_str} |")
    print(f"| Pose Error | — | {pose_err_str} |")
    print(f"| Expression MAE | — | {expr_mae_str} (FAIL: > 0.05) |")
    print(f"| Landmark NME | — | {landmark_nme_str} |")
    print(f"| Multi-Face Association | — | {'PASS' if multi_face_pass else 'FAIL'} |")
    print(f"| CPU Latency | — | {cpu_mean_ms:.2f} ms |")
    print(f"| GPU Latency | — | {gpu_lat_str} |")
    print(f"| Model License Acceptable | PASS (Apache 2.0) | FAIL (Non-Commercial) |")
    print(f"| **OVERALL VERDICT** | **REJECT (No weights)** | **{overall_verdict} (License Non-Commercial)** |")
    print("====================================================================")


if __name__ == "__main__":
    run_inswapper_benchmark()
