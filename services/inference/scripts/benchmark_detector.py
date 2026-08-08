import sys, time, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
from app.detection.detector import MediaPipeDetector

def main():
    m = "services/inference/models/blaze_face_short_range.tflite"
    det = MediaPipeDetector(model_path=m)
    if det.detector is None:
        print("Failed to init detector")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam available")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {w}x{h}, Warmup: 30 frames...")
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            det.detect(frame)
    print("Benchmarking 300 frames...")
    lats = []
    t_start = time.perf_counter()
    for _ in range(300):
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        det.detect(frame)
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
    t_wall = time.perf_counter() - t_start
    cap.release()
    if not lats:
        print("No frames captured")
        return
    arr = np.array(lats)
    avg_lat = float(np.mean(arr))
    det_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    wall_fps = len(arr) / t_wall if t_wall > 0 else 0.0
    print(f"\n--- MediaPipeDetector Benchmark Results ---")
    print(f"Resolution: {w}x{h}")
    print(f"Total Successful Frames: {len(arr)}")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"P50 Latency: {np.percentile(arr, 50):.2f} ms")
    print(f"P95 Latency: {np.percentile(arr, 95):.2f} ms")
    print(f"Min Latency: {np.min(arr):.2f} ms")
    print(f"Max Latency: {np.max(arr):.2f} ms")
    print(f"Detector-Only FPS: {det_fps:.2f}")
    print(f"Benchmark Wall-Clock FPS: {wall_fps:.2f}")

if __name__ == "__main__":
    main()
