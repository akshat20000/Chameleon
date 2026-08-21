"""
IdentityAsset data structures, serialization, manifest generation, and validation engine.

Spec: docs/architecture/IDENTITY_ASSET.md (Version 1.1.0)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IDENTITY_ASSET_SCHEMA_VERSION = "1.1.0"
PIPELINE_VERSION = "2.3.0"


class ValidationProfile(Enum):
    IDENTITY_ONLY = "identity_only"
    IDENTITY_PLUS_APPEARANCE = "identity_plus_appearance"
    FULL_REFERENCE = "full_reference"


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    available_components: Set[str] = field(default_factory=set)


@dataclass
class SemanticSegmentationResult:
    masks: Dict[str, np.ndarray]            # class_name -> uint8 binary mask (0 or 255)
    available_classes: Set[str]
    backend_name: str
    backend_version: str
    confidence_metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentedReferenceView:
    view_index: int
    image_path: str                          # path relative to identity workspace
    embedding: np.ndarray                   # (512,) float32, L2-normalized
    segmentation: Optional[SemanticSegmentationResult]
    quality_score: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class IdentityAsset:
    schema_version: str
    identity_id: str                        # UUID v4
    display_name: str
    created_at: str                         # ISO 8601
    pipeline_version: str
    provenance: Dict                        # Model & threshold provenance

    fused_identity_embedding: np.ndarray    # (512,) float32, L2-normalized
    segmented_views: List[SegmentedReferenceView] = field(default_factory=list)

    appearance_policy: Optional[Dict] = None
    body_proportion_hint: Optional[str] = "DEFAULT"

    def validate(self, profile: ValidationProfile = ValidationProfile.IDENTITY_ONLY) -> ValidationResult:
        """
        Validate runtime integrity against an explicit ValidationProfile.
        """
        errors: List[str] = []
        warnings: List[str] = []
        available: Set[str] = set()

        # Schema version check
        if self.schema_version != IDENTITY_ASSET_SCHEMA_VERSION:
            errors.append(
                f"Schema version mismatch: got '{self.schema_version}', expected '{IDENTITY_ASSET_SCHEMA_VERSION}'"
            )

        # Fused identity embedding check
        if self.fused_identity_embedding is None or not isinstance(self.fused_identity_embedding, np.ndarray):
            errors.append("Missing fused identity embedding array")
        elif self.fused_identity_embedding.shape != (512,):
            errors.append(f"Invalid fused identity embedding shape: got {self.fused_identity_embedding.shape}, expected (512,)")
        elif not np.all(np.isfinite(self.fused_identity_embedding)):
            errors.append("Fused identity embedding contains non-finite values (NaN or Inf)")
        else:
            norm = float(np.linalg.norm(self.fused_identity_embedding))
            if abs(norm - 1.0) > 1e-3:
                errors.append(f"Fused identity embedding is not L2-normalized (norm = {norm:.4f})")
            else:
                available.add("face_embedding")

        # Reference views check
        if not self.segmented_views:
            errors.append("No segmented reference views present in asset")
        else:
            available.add("reference_views")

        # PROFILE: IDENTITY_PLUS_APPEARANCE
        if profile in (ValidationProfile.IDENTITY_PLUS_APPEARANCE, ValidationProfile.FULL_REFERENCE):
            has_masks = any(
                v.segmentation is not None and len(v.segmentation.masks) > 0
                for v in self.segmented_views
            )
            if not has_masks:
                errors.append("Profile requires appearance masks, but none are present in reference views")
            else:
                available.add("appearance_masks")

        # PROFILE: FULL_REFERENCE
        if profile == ValidationProfile.FULL_REFERENCE:
            required_classes = {"face", "hair", "clothing", "body"}
            found_classes: Set[str] = set()
            for v in self.segmented_views:
                if v.segmentation:
                    found_classes.update(v.segmentation.available_classes)
            missing = required_classes - found_classes
            if missing:
                errors.append(f"FULL_REFERENCE profile missing required component categories: {sorted(missing)}")

        return ValidationResult(
            valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            available_components=available,
        )

    def save(self, workspace_dir: Union[str, Path]) -> Path:
        """
        Serialize complete IdentityAsset directory package to disk.
        """
        target_dir = Path(workspace_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        comp_dir = target_dir / "components"
        comp_dir.mkdir(parents=True, exist_ok=True)

        face_dir = comp_dir / "face"
        face_dir.mkdir(parents=True, exist_ok=True)

        views_dir = comp_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        checksums: Dict[str, str] = {}

        # 1. Save fused identity embedding
        fused_path = face_dir / "fused_identity_embedding.npy"
        np.save(fused_path, self.fused_identity_embedding.astype(np.float32))
        checksums["components/face/fused_identity_embedding.npy"] = _compute_sha256(fused_path)

        # 2. Save segmented reference views
        views_meta = []
        for v in self.segmented_views:
            v_dir = views_dir / f"view_{v.view_index:03d}"
            v_dir.mkdir(parents=True, exist_ok=True)

            # View embedding
            emb_path = v_dir / "embedding.npy"
            np.save(emb_path, v.embedding.astype(np.float32))
            rel_emb = f"components/views/view_{v.view_index:03d}/embedding.npy"
            checksums[rel_emb] = _compute_sha256(emb_path)

            # Masks
            masks_dir = v_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_files = {}

            if v.segmentation and v.segmentation.masks:
                for cls_name, mask_arr in v.segmentation.masks.items():
                    m_path = masks_dir / f"{cls_name}.png"
                    cv2.imwrite(str(m_path), mask_arr)
                    rel_m = f"components/views/view_{v.view_index:03d}/masks/{cls_name}.png"
                    checksums[rel_m] = _compute_sha256(m_path)
                    mask_files[cls_name] = f"components/views/view_{v.view_index:03d}/masks/{cls_name}.png"

            v_meta = {
                "view_index": v.view_index,
                "image_path": v.image_path,
                "quality_score": float(v.quality_score),
                "yaw_deg": float(v.yaw_deg),
                "pitch_deg": float(v.pitch_deg),
                "roll_deg": float(v.roll_deg),
                "embedding_path": rel_emb,
                "mask_files": mask_files,
                "segmentation_backend": v.segmentation.backend_name if v.segmentation else None,
                "segmentation_backend_version": v.segmentation.backend_version if v.segmentation else None,
                "available_classes": sorted(v.segmentation.available_classes) if v.segmentation else [],
                "confidence_metadata": v.segmentation.confidence_metadata if v.segmentation else {},
            }

            quality_path = v_dir / "quality_meta.json"
            with open(quality_path, "w") as f:
                json.dump(v_meta, f, indent=2)
            rel_q = f"components/views/view_{v.view_index:03d}/quality_meta.json"
            checksums[rel_q] = _compute_sha256(quality_path)

            views_meta.append(v_meta)

        # 3. Assemble and save manifest.json
        manifest = {
            "schema_version": self.schema_version,
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
            "provenance": self.provenance,
            "body_proportion_hint": self.body_proportion_hint or "DEFAULT",
            "fused_embedding": {
                "embedding_dim": 512,
                "l2_norm": float(np.linalg.norm(self.fused_identity_embedding)),
                "num_contributing_views": len(self.segmented_views),
                "path": "components/face/fused_identity_embedding.npy",
                "checksum_sha256": checksums["components/face/fused_identity_embedding.npy"],
            },
            "segmented_views": views_meta,
            "appearance_policy": self.appearance_policy or {
                "face": "reference",
                "hair": "reference",
                "arms": "reference",
                "hands": "reference",
                "torso": {"preferred": "reference", "fallback": "performer_clothing"},
                "legs": {"preferred": "reference", "fallback": "performer_clothing"},
            },
            "checksums": checksums,
        }

        manifest_path = target_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Save pointer file identity_asset.json
        final_dir = target_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        pointer_path = final_dir / "identity_asset.json"
        with open(pointer_path, "w") as f:
            json.dump({"manifest_path": "../manifest.json", "identity_id": self.identity_id}, f, indent=2)

        logger.info("Saved IdentityAsset '%s' to %s (%d views)", self.identity_id, target_dir, len(self.segmented_views))
        return manifest_path

    @classmethod
    def load(cls, workspace_dir: Union[str, Path]) -> IdentityAsset:
        """
        Load IdentityAsset package from workspace directory and verify manifest integrity.
        """
        target_dir = Path(workspace_dir)
        manifest_path = target_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"IdentityAsset manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        schema_ver = manifest.get("schema_version")
        if schema_ver != IDENTITY_ASSET_SCHEMA_VERSION:
            raise ValueError(f"Schema version mismatch: got '{schema_ver}', expected '{IDENTITY_ASSET_SCHEMA_VERSION}'")

        # Load fused embedding
        fused_rel = manifest.get("fused_embedding", {}).get("path", "components/face/fused_identity_embedding.npy")
        fused_path = target_dir / fused_rel
        if not fused_path.exists():
            raise FileNotFoundError(f"Fused embedding file not found: {fused_path}")
        fused_emb = np.load(fused_path).astype(np.float32)

        # Load segmented views
        views = []
        for v_meta in manifest.get("segmented_views", []):
            idx = v_meta["view_index"]
            emb_rel = v_meta["embedding_path"]
            emb_path = target_dir / emb_rel
            emb = np.load(emb_path).astype(np.float32) if emb_path.exists() else np.zeros((512,), dtype=np.float32)

            masks = {}
            for cls_name, m_rel in v_meta.get("mask_files", {}).items():
                m_path = target_dir / m_rel
                if m_path.exists():
                    img = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        masks[cls_name] = img

            seg_result = None
            if v_meta.get("segmentation_backend"):
                seg_result = SemanticSegmentationResult(
                    masks=masks,
                    available_classes=set(v_meta.get("available_classes", [])),
                    backend_name=v_meta["segmentation_backend"],
                    backend_version=v_meta.get("segmentation_backend_version", "1.0"),
                    confidence_metadata=v_meta.get("confidence_metadata", {}),
                )

            view_obj = SegmentedReferenceView(
                view_index=idx,
                image_path=v_meta.get("image_path", f"inputs/selected_views/view_{idx:03d}.png"),
                embedding=emb,
                segmentation=seg_result,
                quality_score=v_meta.get("quality_score", 1.0),
                yaw_deg=v_meta.get("yaw_deg", 0.0),
                pitch_deg=v_meta.get("pitch_deg", 0.0),
                roll_deg=v_meta.get("roll_deg", 0.0),
            )
            views.append(view_obj)

        asset = cls(
            schema_version=schema_ver,
            identity_id=manifest["identity_id"],
            display_name=manifest.get("display_name", "UnnamedIdentity"),
            created_at=manifest.get("created_at", datetime.datetime.now().isoformat()),
            pipeline_version=manifest.get("pipeline_version", PIPELINE_VERSION),
            provenance=manifest.get("provenance", {}),
            fused_identity_embedding=fused_emb,
            segmented_views=views,
            appearance_policy=manifest.get("appearance_policy"),
            body_proportion_hint=manifest.get("body_proportion_hint", "DEFAULT"),
        )

        return asset

    def verify_checksums(self, workspace_dir: Union[str, Path]) -> Tuple[bool, List[str]]:
        """
        Verify on-disk SHA-256 checksums against manifest.json.
        """
        target_dir = Path(workspace_dir)
        manifest_path = target_dir / "manifest.json"

        if not manifest_path.exists():
            return False, [f"Manifest file missing: {manifest_path}"]

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        checksums = manifest.get("checksums", {})
        errors = []

        for rel_path, expected_hash in checksums.items():
            full_path = target_dir / rel_path
            if not full_path.exists():
                errors.append(f"Component file missing: {rel_path}")
                continue

            actual_hash = _compute_sha256(full_path)
            if actual_hash != expected_hash:
                errors.append(f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

        return (len(errors) == 0), errors


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
