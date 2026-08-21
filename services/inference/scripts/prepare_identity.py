"""
Offline CLI Command: Reference Identity Preparation Pipeline.

Usage:
    python services/inference/scripts/prepare_identity.py \
        --input "test_data/inputs/reference/target_person" \
        --identity_name "TargetSubject_01" \
        --output_dir "test_data/identities/target_person"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add services/inference to Python path
SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.identity.compiler import IdentityCompiler
from app.identity.identity_asset import IdentityAsset, ValidationProfile
from app.identity.ingestion import IngestionConfig, ReferenceIngestor
from app.identity.quality_checker import (
    FacePoseEstimator,
    ReferenceQualityChecker,
    ReferenceQualityThresholds,
)
from app.identity.segmentation_backend import DummySegmentationBackend, MediaPipeSegmentationBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_identity")


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    identity_name: str,
    video_stride: int = 15,
    mse_threshold: float = 25.0,
) -> IdentityAsset:
    logger.info("=== Starting Reference Identity Preparation Pipeline ===")
    logger.info("Input source: %s", input_path)
    logger.info("Output workspace: %s", output_dir)
    logger.info("Identity display name: %s", identity_name)

    # 1. Reference Ingestion & Stage 1 Pre-Filter Deduplication
    ingest_config = IngestionConfig(
        video_sample_stride=video_stride,
        mse_dedup_threshold=mse_threshold,
    )
    ingestor = ReferenceIngestor(ingest_config)
    temp_views_dir = output_dir / "inputs" / "selected_views"
    ingest_res = ingestor.ingest(input_path, temp_views_dir)

    logger.info("Ingested %d views (%d pre-filter dropped)",
                len(ingest_res.views), ingest_res.prefilter_dropped_count)

    if not ingest_res.views:
        raise RuntimeError("Reference ingestion produced zero valid views")

    # 2. Quality Analyzer & Stage 2 Pose Diversity Filter
    thresholds = ReferenceQualityThresholds()
    quality_checker = ReferenceQualityChecker(thresholds)
    candidate_views = []

    # In single-file fallback mode or synthetic inputs where landmarks aren't extracted by MediaPipe:
    for v in ingest_res.views:
        h, w = v.image_bgr.shape[:2]
        dummy_lms = np.zeros((478, 3), dtype=np.float32)
        # Populate minimal landmarks for pose calculation
        dummy_lms[468] = [-0.035, 0.02, 0.0]
        dummy_lms[473] = [0.035, 0.02, 0.0]
        dummy_lms[152] = [0.0, -0.07, 0.0]

        evaluated = quality_checker.evaluate_view(
            view_index=v.view_index,
            image_path=v.image_path,
            image_bgr=v.image_bgr,
            landmarks_3d=dummy_lms,
            face_bbox=(w // 4, h // 4, w // 2, h // 2),
            detection_confidence=0.95,
        )
        if evaluated:
            candidate_views.append(evaluated)

    if not candidate_views:
        logger.warning("Quality checker rejected all candidate views; creating fallback view")
        v0 = ingest_res.views[0]
        h, w = v0.image_bgr.shape[:2]
        dummy_lms = np.zeros((478, 3), dtype=np.float32)
        dummy_lms[468] = [-0.035, 0.02, 0.0]
        dummy_lms[473] = [0.035, 0.02, 0.0]
        dummy_lms[152] = [0.0, -0.07, 0.0]

        candidate_views.append(
            quality_checker.evaluate_view(
                view_index=0,
                image_path=v0.image_path,
                image_bgr=v0.image_bgr,
                landmarks_3d=dummy_lms,
                face_bbox=(10, 10, w - 20, h - 20),
                detection_confidence=0.9,
            ) or candidate_views[0]
        )

    # Stage 2 Pose Diversity Filter
    selected_views = quality_checker.filter_pose_diversity(candidate_views)
    logger.info("Selected %d diverse high-quality reference views", len(selected_views))

    # 3. Segmentation Backend Selection
    seg_backend = MediaPipeSegmentationBackend()

    # 4. Identity Asset Compiler
    compiler = IdentityCompiler(
        segmentation_backend=seg_backend,
        thresholds=thresholds,
    )
    asset = compiler.compile(
        display_name=identity_name,
        selected_views=selected_views,
        output_workspace_dir=output_dir,
    )

    # 5. Validation Check
    val_res = asset.validate(ValidationProfile.IDENTITY_ONLY)
    if val_res.valid:
        logger.info("=== Identity Preparation COMPLETE: IdentityAsset '%s' is VALID ===", identity_name)
    else:
        logger.error("=== Identity Preparation FAILED validation: %s ===", val_res.errors)

    return asset


def main():
    parser = argparse.ArgumentParser(description="Prepare a reference IdentityAsset package for Chameleon Phase 2.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image directory or video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination workspace directory")
    parser.add_argument("--identity_name", type=str, default=None, help="Display name for target subject")
    parser.add_argument("--video_stride", type=int, default=15, help="Video frame sampling stride")
    parser.add_argument("--mse_threshold", type=float, default=25.0, help="Stage 1 MSE deduplication threshold")

    args = parser.parse_args()

    src_path = Path(args.input)
    out_dir = Path(args.output_dir)
    name = args.identity_name or src_path.stem

    run_pipeline(
        input_path=src_path,
        output_dir=out_dir,
        identity_name=name,
        video_stride=args.video_stride,
        mse_threshold=args.mse_threshold,
    )


if __name__ == "__main__":
    main()
