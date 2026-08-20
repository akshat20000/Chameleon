"""
Standalone smoke test for MediaPipeSegmenter.

Requires:
  - services/inference/models/selfie_multiclass_256x256.tflite
  - An image file path (optional, defaults to test_data/2face_validation.png or synthetic image)

Usage
-----
  python services/inference/scripts/test_segmenter.py
  python services/inference/scripts/test_segmenter.py path/to/image.jpg
  python services/inference/scripts/test_segmenter.py --model path/to/selfie_multiclass_256x256.tflite
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.segmentation.segmenter import MediaPipeSegmenter


LABEL_MAP = {
    0: "background",
    1: "hair",
    2: "body-skin",
    3: "face-skin",
    4: "clothes",
    5: "others",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for MediaPipe face & body segmentation."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to image file. Defaults to test_data/2face_validation.png if present.",
    )
    parser.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parent.parent
            / "models"
            / "selfie_multiclass_256x256.tflite"
        ),
        help="Path to selfie_multiclass_256x256.tflite model file.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    model_path = Path(args.model).resolve()

    print("====================================================================")
    print("  MEDIAPIPE SEGMENTER SMOKE TEST — Phase 1.4")
    print("====================================================================")
    print(f"  Model path   : {model_path}")
    print(f"  Model exists : {model_path.exists()}")

    segmenter = MediaPipeSegmenter(model_path=str(model_path))
    print(f"  Segmenter ready: {segmenter.is_ready}")

    if not segmenter.is_ready:
        print("ERROR: Segmenter is not ready. Aborting test.")
        sys.exit(1)

    # Find or generate image
    img_path = None
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"ERROR: Image path not found: {img_path}", file=sys.stderr)
            sys.exit(1)

    if img_path:
        print(f"  Loading image : {img_path}")
        image = cv2.imread(str(img_path))
    else:
        print("  No image provided via argument. Generating synthetic 640x480 image for smoke test...")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic face/person
        cv2.circle(image, (320, 200), 80, (200, 180, 150), -1) # face/head
        cv2.ellipse(image, (320, 380), (120, 150), 0, 0, 180, (50, 50, 200), -1) # body/shirt

    h, w = image.shape[:2]
    print(f"  Image shape  : {w} x {h} (H={h}, W={w})")

    # Warmup run
    _ = segmenter.segment(image)

    # Benchmark run
    t0 = time.perf_counter()
    result = segmenter.segment(image)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if result is None:
        print("ERROR: Segmentation returned None.")
        sys.exit(1)

    print("\n--------------------------------------------------------------------")
    print("  RESULTS")
    print("--------------------------------------------------------------------")
    print(f"  Latency            : {latency_ms:.2f} ms")
    print(f"  class_mask shape   : {result.class_mask.shape} (dtype={result.class_mask.dtype})")
    print(f"  face_mask shape    : {result.face_mask.shape} (dtype={result.face_mask.dtype})")
    print(f"  hair_mask shape    : {result.hair_mask.shape} (dtype={result.hair_mask.dtype})")
    print(f"  skin_mask shape    : {result.skin_mask.shape} (dtype={result.skin_mask.dtype})")

    total_pixels = h * w
    unique_ids, counts = np.unique(result.class_mask, return_counts=True)
    print("\n  Class Distribution in Image:")
    for class_id, count in zip(unique_ids, counts):
        lbl = LABEL_MAP.get(int(class_id), f"unknown_{class_id}")
        pct = (count / total_pixels) * 100.0
        print(f"    Class {class_id} ({lbl:<10}) : {count:>8} px  ({pct:>6.2f}%)")

    # Save visualization to test_data/outputs/
    out_dir = Path(__file__).resolve().parent.parent.parent / "test_data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segmentation_smoke_out.png"
    
    # Colorize class mask
    color_map = np.array([
        [0, 0, 0],       # 0: background (black)
        [255, 0, 0],     # 1: hair (blue)
        [0, 255, 0],     # 2: body-skin (green)
        [0, 255, 255],   # 3: face-skin (yellow)
        [0, 0, 255],     # 4: clothes (red)
        [255, 0, 255],   # 5: others (magenta)
    ], dtype=np.uint8)
    
    vis_mask = color_map[np.clip(result.class_mask, 0, 5)]
    blended = cv2.addWeighted(image, 0.5, vis_mask, 0.5, 0)
    cv2.imwrite(str(out_path), blended)
    print(f"\n  Saved visualization to: {out_path}")

    segmenter.close()
    print("\n  Smoke test finished successfully!")


if __name__ == "__main__":
    main()
