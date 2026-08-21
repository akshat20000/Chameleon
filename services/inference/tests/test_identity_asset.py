import json
import sys
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.identity.identity_asset import (
    IDENTITY_ASSET_SCHEMA_VERSION,
    PIPELINE_VERSION,
    IdentityAsset,
    SegmentedReferenceView,
    SemanticSegmentationResult,
    ValidationProfile,
    ValidationResult,
)


@pytest.fixture
def sample_identity_asset(tmp_path) -> Tuple[IdentityAsset, Path]:
    """Helper fixture providing a valid populated IdentityAsset."""
    # Synthetic float32 512-D L2-normalized vector
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    mask_face = np.ones((100, 100), dtype=np.uint8) * 255
    mask_hair = np.ones((100, 100), dtype=np.uint8) * 255
    mask_clothing = np.ones((100, 100), dtype=np.uint8) * 255

    seg_res = SemanticSegmentationResult(
        masks={"face": mask_face, "hair": mask_hair, "clothing": mask_clothing},
        available_classes={"face", "hair", "clothing"},
        backend_name="TestSegmentationBackend",
        backend_version="1.0.0",
        confidence_metadata={"face": 0.98, "hair": 0.95},
    )

    view_0 = SegmentedReferenceView(
        view_index=0,
        image_path="inputs/selected_views/view_000.png",
        embedding=vec.copy(),
        segmentation=seg_res,
        quality_score=0.92,
        yaw_deg=5.2,
        pitch_deg=-2.1,
        roll_deg=0.5,
    )

    asset = IdentityAsset(
        schema_version=IDENTITY_ASSET_SCHEMA_VERSION,
        identity_id=str(uuid.uuid4()),
        display_name="TestSubject_01",
        created_at="2026-08-21T20:00:00Z",
        pipeline_version=PIPELINE_VERSION,
        provenance={
            "encoder_model": "w600k_mbf.onnx",
            "encoder_model_version": "1.0",
            "segmentation_backend": "TestSegmentationBackend",
            "segmentation_model_version": "1.0.0",
            "quality_thresholds": {
                "min_face_size_px": 112,
                "min_blur_score": 100.0,
                "min_detection_confidence": 0.8,
                "max_yaw_deg": 30.0,
                "max_pitch_deg": 30.0,
                "min_quality_weight": 0.001,
            },
        },
        fused_identity_embedding=vec.copy(),
        segmented_views=[view_0],
        appearance_policy={"face": "reference", "hair": "reference"},
        body_proportion_hint="DEFAULT",
    )

    return asset, tmp_path / "test_identity_workspace"


def test_identity_asset_serialization_and_deserialization(sample_identity_asset):
    asset, workspace = sample_identity_asset

    # Save to disk
    manifest_path = asset.save(workspace)
    assert manifest_path.exists()
    assert (workspace / "manifest.json").exists()
    assert (workspace / "components" / "face" / "fused_identity_embedding.npy").exists()
    assert (workspace / "components" / "views" / "view_000" / "embedding.npy").exists()
    assert (workspace / "components" / "views" / "view_000" / "masks" / "face.png").exists()

    # Load back from disk
    loaded = IdentityAsset.load(workspace)
    assert loaded.schema_version == IDENTITY_ASSET_SCHEMA_VERSION
    assert loaded.identity_id == asset.identity_id
    assert loaded.display_name == "TestSubject_01"
    assert loaded.pipeline_version == PIPELINE_VERSION
    assert loaded.provenance["encoder_model"] == "w600k_mbf.onnx"
    assert loaded.fused_identity_embedding.shape == (512,)
    assert np.allclose(loaded.fused_identity_embedding, asset.fused_identity_embedding, atol=1e-5)
    assert len(loaded.segmented_views) == 1
    assert loaded.segmented_views[0].quality_score == 0.92
    assert "face" in loaded.segmented_views[0].segmentation.masks


def test_checksum_verification_success_and_failure(sample_identity_asset):
    asset, workspace = sample_identity_asset
    asset.save(workspace)

    # Verify checksums on pristine workspace
    ok, errors = asset.verify_checksums(workspace)
    assert ok, f"Checksum verification failed on clean save: {errors}"
    assert len(errors) == 0

    # Corrupt a component file (fused embedding)
    emb_file = workspace / "components" / "face" / "fused_identity_embedding.npy"
    with open(emb_file, "ab") as f:
        f.write(b"CORRUPTED_BYTES")

    ok_corrupt, errors_corrupt = asset.verify_checksums(workspace)
    assert not ok_corrupt
    assert any("Checksum mismatch" in err for err in errors_corrupt)


def test_schema_version_mismatch_rejected(sample_identity_asset):
    asset, workspace = sample_identity_asset
    asset.save(workspace)

    # Mutate schema_version in manifest.json
    manifest_path = workspace / "manifest.json"
    with open(manifest_path, "r") as f:
        m = json.load(f)
    m["schema_version"] = "9.9.9"
    with open(manifest_path, "w") as f:
        json.dump(m, f)

    with pytest.raises(ValueError, match="Schema version mismatch"):
        IdentityAsset.load(workspace)


def test_profile_validation_identity_only(sample_identity_asset):
    asset, workspace = sample_identity_asset

    # Valid asset under IDENTITY_ONLY
    res = asset.validate(ValidationProfile.IDENTITY_ONLY)
    assert res.valid
    assert len(res.errors) == 0
    assert "face_embedding" in res.available_components
    assert "reference_views" in res.available_components

    # Un-normalized embedding fails validation
    bad_asset = asset
    bad_asset.fused_identity_embedding = np.ones((512,), dtype=np.float32) * 5.0  # norm != 1.0
    res_bad = bad_asset.validate(ValidationProfile.IDENTITY_ONLY)
    assert not res_bad.valid
    assert any("not L2-normalized" in err for err in res_bad.errors)


def test_profile_validation_identity_plus_appearance(sample_identity_asset):
    asset, workspace = sample_identity_asset

    # Asset with masks passes IDENTITY_PLUS_APPEARANCE
    res = asset.validate(ValidationProfile.IDENTITY_PLUS_APPEARANCE)
    assert res.valid
    assert "appearance_masks" in res.available_components

    # Asset with NO masks fails IDENTITY_PLUS_APPEARANCE
    no_mask_view = SegmentedReferenceView(
        view_index=0,
        image_path="inputs/view_0.png",
        embedding=asset.fused_identity_embedding.copy(),
        segmentation=None,
        quality_score=1.0,
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
    )
    asset_no_masks = IdentityAsset(
        schema_version=IDENTITY_ASSET_SCHEMA_VERSION,
        identity_id=str(uuid.uuid4()),
        display_name="NoMaskSubject",
        created_at="2026-08-21T20:00:00Z",
        pipeline_version=PIPELINE_VERSION,
        provenance={},
        fused_identity_embedding=asset.fused_identity_embedding.copy(),
        segmented_views=[no_mask_view],
    )
    res_no_masks = asset_no_masks.validate(ValidationProfile.IDENTITY_PLUS_APPEARANCE)
    assert not res_no_masks.valid
    assert any("requires appearance masks" in err for err in res_no_masks.errors)


def test_profile_validation_full_reference(sample_identity_asset):
    asset, workspace = sample_identity_asset

    # sample_identity_asset has face, hair, clothing (missing body)
    res = asset.validate(ValidationProfile.FULL_REFERENCE)
    assert not res.valid
    assert any("missing required component categories" in err for err in res.errors)
    assert "body" in res.errors[0]

    # Add body mask
    asset.segmented_views[0].segmentation.available_classes.add("body")
    asset.segmented_views[0].segmentation.masks["body"] = np.ones((100, 100), dtype=np.uint8) * 255
    res_full = asset.validate(ValidationProfile.FULL_REFERENCE)
    assert res_full.valid


def test_manifest_provenance_metadata_preserved(sample_identity_asset):
    asset, workspace = sample_identity_asset
    asset.save(workspace)

    with open(workspace / "manifest.json", "r") as f:
        m = json.load(f)

    assert "provenance" in m
    assert m["provenance"]["encoder_model"] == "w600k_mbf.onnx"
    assert m["provenance"]["quality_thresholds"]["min_blur_score"] == 100.0
    assert m["provenance"]["quality_thresholds"]["min_quality_weight"] == 0.001
