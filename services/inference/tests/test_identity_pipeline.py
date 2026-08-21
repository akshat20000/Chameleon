"""
Comprehensive Integration Test Suite for Phase 2.3 — Reference Identity Preparation Pipeline.

Spec: docs/architecture/IDENTITY_ASSET.md (Revision 3)
"""

import sys
import json
import uuid
import cv2
import numpy as np
import pytest
from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.identity.compiler import IdentityCompiler
from app.identity.encoder import fuse_embeddings, normalize_embedding
from app.identity.identity_asset import (
    IDENTITY_ASSET_SCHEMA_VERSION,
    PIPELINE_VERSION,
    IdentityAsset,
    SegmentedReferenceView,
    SemanticSegmentationResult,
    ValidationProfile,
    ValidationResult,
)
from app.identity.ingestion import IngestionConfig, ReferenceIngestor
from app.identity.quality_checker import (
    FacePoseEstimator,
    ReferenceQualityChecker,
    ReferenceQualityThresholds,
    SelectedReferenceView,
)
from app.identity.segmentation_backend import DummySegmentationBackend, MediaPipeSegmentationBackend
from scripts.prepare_identity import run_pipeline


@pytest.fixture
def real_reference_dir_fixture(tmp_path) -> Path:
    """Fixture producing synthetic reference image files."""
    ref_dir = tmp_path / "reference_photos"
    ref_dir.mkdir(parents=True)

    for i in range(5):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw face oval and texture
        cv2.ellipse(img, (200, 200), (100, 130), 0, 0, 360, (180, 200, 220), -1)
        cv2.circle(img, (160, 170), 15, (50, 50, 50), -1)  # Left eye
        cv2.circle(img, (240, 170), 15, (50, 50, 50), -1)  # Right eye
        img[::10, :] += (i * 20) % 255                    # Subtle variation per frame
        cv2.imwrite(str(ref_dir / f"ref_photo_{i:02d}.png"), img)

    return ref_dir


def test_corrupted_component_fails_checksum(real_reference_dir_fixture, tmp_path):
    out_dir = tmp_path / "compiled_identity"
    asset = run_pipeline(real_reference_dir_fixture, out_dir, "TestSubject")

    ok, errors = asset.verify_checksums(out_dir)
    assert ok

    # Corrupt a view embedding file
    emb_path = out_dir / "components" / "views" / "view_000" / "embedding.npy"
    with open(emb_path, "ab") as f:
        f.write(b"CORRUPTED_BYTES")

    ok_corrupt, errors_corrupt = asset.verify_checksums(out_dir)
    assert not ok_corrupt
    assert any("Checksum mismatch" in err for err in errors_corrupt)


def test_missing_required_component_fails_profile_validation(real_reference_dir_fixture, tmp_path):
    out_dir = tmp_path / "compiled_identity"
    asset = run_pipeline(real_reference_dir_fixture, out_dir, "TestSubject")

    # FULL_REFERENCE profile requires all component classes (face, hair, clothing, body)
    res_full = asset.validate(ValidationProfile.FULL_REFERENCE)
    assert not res_full.valid
    assert any("missing required component categories" in err for err in res_full.errors)


def test_embedding_fusion_is_l2_normalized():
    e1 = np.random.randn(512).astype(np.float32)
    e2 = np.random.randn(512).astype(np.float32)
    fused = fuse_embeddings([e1, e2], weights=[0.8, 0.2])

    assert fused is not None
    assert fused.shape == (512,)
    np.testing.assert_allclose(np.linalg.norm(fused), 1.0, atol=1e-5)


def test_low_quality_views_do_not_dominate_fusion():
    e_high = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e_low = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # High quality weight 0.95 vs low quality weight 0.05
    fused = fuse_embeddings([e_high, e_low], weights=[0.95, 0.05])
    assert fused[0] > 0.95
    assert fused[1] < 0.2


def test_duplicate_video_frames_are_deduplicated(tmp_path):
    video_path = tmp_path / "dup_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (200, 200))

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:, :] = [100, 100, 100]

    for _ in range(40):
        writer.write(frame)
    writer.release()

    out_views = tmp_path / "views"
    ingestor = ReferenceIngestor(IngestionConfig(video_sample_stride=2, mse_dedup_threshold=10.0))
    res = ingestor.ingest(video_path, out_views)

    assert res.prefilter_dropped_count >= 15
    assert len(res.views) == 1  # Only 1 view stored due to cheap deduplication


def test_segmentation_backend_reports_unavailable_classes():
    backend = DummySegmentationBackend(supported_classes={"face", "hair"})
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = backend.segment(img)

    assert res.available_classes == {"face", "hair"}
    assert "clothing" not in res.available_classes
    assert "body" not in res.available_classes


def test_manifest_pipeline_version_is_preserved(real_reference_dir_fixture, tmp_path):
    out_dir = tmp_path / "compiled_identity"
    asset = run_pipeline(real_reference_dir_fixture, out_dir, "VersionTestSubject")

    with open(out_dir / "manifest.json", "r") as f:
        m = json.load(f)

    assert m["pipeline_version"] == PIPELINE_VERSION
    assert m["schema_version"] == IDENTITY_ASSET_SCHEMA_VERSION


def test_asset_load_rejects_schema_version_mismatch(real_reference_dir_fixture, tmp_path):
    out_dir = tmp_path / "compiled_identity"
    run_pipeline(real_reference_dir_fixture, out_dir, "SchemaTestSubject")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        m = json.load(f)

    m["schema_version"] = "0.0.1"
    with open(manifest_path, "w") as f:
        json.dump(m, f)

    with pytest.raises(ValueError, match="Schema version mismatch"):
        IdentityAsset.load(out_dir)


def test_face_pose_estimator_angles():
    lms = np.zeros((478, 3), dtype=np.float32)
    lms[468] = [-0.035, 0.02, 0.0]
    lms[473] = [0.035, 0.02, 0.0]
    lms[152] = [0.0, -0.07, 0.0]

    yaw, pitch, roll = FacePoseEstimator.estimate_pose(lms)
    assert abs(yaw) < 2.0
    assert abs(pitch) < 2.0
    assert abs(roll) < 2.0


def test_end_to_end_real_reference_fixture(real_reference_dir_fixture, tmp_path):
    out_dir = tmp_path / "end_to_end_identity"
    asset = run_pipeline(real_reference_dir_fixture, out_dir, "RealSubject_01")

    assert asset is not None
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "components" / "face" / "fused_identity_embedding.npy").exists()

    val_res = asset.validate(ValidationProfile.IDENTITY_ONLY)
    assert val_res.valid
    assert len(val_res.errors) == 0

    ok_checksums, errors = asset.verify_checksums(out_dir)
    assert ok_checksums
