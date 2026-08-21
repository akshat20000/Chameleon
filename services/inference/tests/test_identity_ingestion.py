"""
Unit test suite for Step 2 — Reference Ingestion & Stage 1 Cheap Pre-Filter Deduplication.
"""

import sys
import cv2
import numpy as np
import pytest
from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.identity.ingestion import (
    IngestionConfig,
    IngestionResult,
    ReferenceIngestor,
)


@pytest.fixture
def sample_image_dir(tmp_path) -> Path:
    img_dir = tmp_path / "raw_reference_images"
    img_dir.mkdir(parents=True)

    # Image 1: Gradient
    img1 = np.zeros((400, 400, 3), dtype=np.uint8)
    img1[:, :] = [100, 150, 200]
    cv2.imwrite(str(img_dir / "img_01.png"), img1)

    # Image 2: Exact duplicate of Image 1
    cv2.imwrite(str(img_dir / "img_02_duplicate.png"), img1)

    # Image 3: Distinct image (noise/checkerboard)
    img3 = np.zeros((400, 400, 3), dtype=np.uint8)
    img3[:, :] = [200, 50, 50]
    cv2.imwrite(str(img_dir / "img_03_distinct.png"), img3)

    return img_dir


def test_ingest_single_image(tmp_path):
    img = np.ones((300, 300, 3), dtype=np.uint8) * 128
    img_path = tmp_path / "single_test.png"
    cv2.imwrite(str(img_path), img)

    out_dir = tmp_path / "out_views"
    ingestor = ReferenceIngestor()
    res = ingestor.ingest(img_path, out_dir)

    assert not res.is_video
    assert res.total_frames_found == 1
    assert len(res.views) == 1
    assert (out_dir / "view_000.png").exists()


def test_ingest_image_directory_with_prefilter_deduplication(sample_image_dir, tmp_path):
    out_dir = tmp_path / "out_views"
    config = IngestionConfig(mse_dedup_threshold=10.0)
    ingestor = ReferenceIngestor(config)

    res = ingestor.ingest(sample_image_dir, out_dir)

    assert not res.is_video
    assert res.total_frames_found == 3
    assert res.prefilter_dropped_count == 1  # img_02_duplicate dropped
    assert len(res.views) == 2                # img_01 and img_03 kept
    assert (out_dir / "view_000.png").exists()
    assert (out_dir / "view_001.png").exists()


def test_ingest_synthetic_video(tmp_path):
    video_path = tmp_path / "synthetic_ref.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (200, 200))

    # Generate 60 frames: first 30 blue, next 30 red
    for i in range(30):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]
        writer.write(frame)

    for i in range(30):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, :] = [0, 0, 255]
        writer.write(frame)

    writer.release()

    out_dir = tmp_path / "video_views"
    config = IngestionConfig(video_sample_stride=5, mse_dedup_threshold=10.0)
    ingestor = ReferenceIngestor(config)

    res = ingestor.ingest(video_path, out_dir)

    assert res.is_video
    assert res.total_frames_found == 60
    assert res.sampled_frames_count == 12  # 60 / 5
    assert res.prefilter_dropped_count >= 8  # consecutive blue and red frames dropped
    assert len(res.views) >= 2                # at least 1 blue and 1 red view stored
