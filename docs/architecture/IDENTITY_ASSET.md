# IdentityAsset Specification

**Version:** 1.1.0  
**Status:** SPECIFICATION (Revision 3)  
**Date:** 2026-08-21

---

## 1. Purpose

`IdentityAsset` is the serializable, versioned package representing a single reference identity for use in Chameleon Phase 2. It is produced once during the offline identity preparation phase (Phase 2.3) and consumed at runtime by the avatar driver and rendering modules (Phases 2.5–2.7).

The `IdentityAsset` is **not** the raw reference images themselves. It is a processed, structured representation derived from selected reference views, containing normalized identity embeddings, per-view segmentation masks, model provenance metadata, and quality telemetry.

---

## 2. Design Principles

1. **Versioned and validatable.** Every asset carries a schema version and model provenance. Outdated assets can be detected and rebuilt.
2. **Reproducible.** Given the same input images, pipeline version, and `ReferenceQualityThresholds`, the asset compilation is 100% reproducible.
3. **Portable.** The asset must be loadable on any machine with the appropriate runtime — no local path dependencies.
4. **Per-View Associated.** Segmentation masks and feature embeddings belong to individual reference views. No inter-view mask blending across different camera angles.
5. **Transparent & Provenance-Tracked.** The manifest records exact model files, model versions, and pipeline thresholds used during extraction.
6. **Multi-Profile Validatable.** Runtime validation evaluates explicit profiles (`IDENTITY_ONLY`, `IDENTITY_PLUS_APPEARANCE`, `FULL_REFERENCE`) returning a structured `ValidationResult`.

---

## 3. Package Structure on Disk

```
identity_workspace/
└── {identity_id}/
    ├── manifest.json                  # Schema version, metadata, model provenance, checksums
    ├── inputs/
    │   ├── raw/                       # Original input images/video (never modified)
    │   └── selected_views/            # Curated views selected for processing (view_000.png, etc.)
    ├── components/
    │   ├── face/
    │   │   ├── fused_identity_embedding.npy # Global normalized fused identity vector (512-D)
    │   │   └── geometry_params.json         # Face geometry parameters from best frontal view
    │   ├── views/
    │   │   ├── view_000/
    │   │   │   ├── embedding.npy      # 512-D ArcFace embedding for this view
    │   │   │   ├── quality_meta.json  # View quality scores & face pose angles (yaw/pitch/roll)
    │   │   │   └── masks/
    │   │   │       ├── face.png       # Binary mask (uint8 0/255)
    │   │   │       ├── hair.png       # Binary mask
    │   │   │       └── clothing.png   # Binary mask
    │   │   └── view_001/
    │   │       └── ...
    │   ├── hair/
    │   │   └── metadata.json          # Hair component metadata (dominant color RGB)
    │   └── clothing/
    │       └── metadata.json          # Clothing component metadata
    └── final/
        └── identity_asset.json        # Assembled asset manifest pointer
```

---

## 4. Manifest Schema (manifest.json)

```json
{
  "schema_version": "1.1.0",
  "identity_id": "string (UUID v4)",
  "display_name": "string",
  "created_at": "ISO 8601 timestamp",
  "pipeline_version": "2.3.0",
  "provenance": {
    "encoder_model": "w600k_mbf.onnx",
    "encoder_model_version": "1.0",
    "segmentation_backend": "MediaPipeImageSegmenter",
    "segmentation_model_version": "1.0",
    "quality_thresholds": {
      "min_face_size_px": 112,
      "min_blur_score": 100.0,
      "min_detection_confidence": 0.8,
      "max_yaw_deg": 30.0,
      "max_pitch_deg": 30.0,
      "min_quality_weight": 0.001
    }
  },
  "input_summary": {
    "num_input_images": 12,
    "input_video_path": null,
    "selected_view_indices": [0, 2, 5]
  },
  "fused_embedding": {
    "embedding_dim": 512,
    "l2_norm": 1.0,
    "num_contributing_views": 3,
    "checksum_sha256": "string"
  },
  "components": {
    "face": { "present": true, "checksum_sha256": "string" },
    "body": { "present": false, "reason": "Insufficient full-body coverage in input" },
    "hair": { "present": true, "checksum_sha256": "string" },
    "hands": { "present": false, "reason": "No clear hand views in input" },
    "clothing": { "present": true, "checksum_sha256": "string" }
  },
  "appearance_policy": {
    "face": "reference",
    "hair": "reference",
    "arms": "reference",
    "hands": "reference",
    "torso": {"preferred": "reference", "fallback": "performer_clothing"},
    "legs": {"preferred": "reference", "fallback": "performer_clothing"}
  },
  "body_proportion_hint": "DEFAULT"
}
```

---

## 5. Validation Profiles & Deterministic Requirements

Runtime validation of an `IdentityAsset` is performed against explicit profiles:

| Profile | Required Components & Invariants |
| :--- | :--- |
| `IDENTITY_ONLY` | Valid manifest, schema version match, SHA-256 checksums valid, $\ge 1$ selected reference view, valid 512-D L2-normalized fused identity embedding ($\|e\| = 1.0$). |
| `IDENTITY_PLUS_APPEARANCE` | Everything in `IDENTITY_ONLY` + required appearance masks supported by configured segmentation backend (`face_mask`, `hair_mask`). |
| `FULL_REFERENCE` | Everything in `IDENTITY_PLUS_APPEARANCE` + all schema component categories present (`face`, `body`, `hair`, `hands`, `clothing`). |

---

## 6. Python Type Definitions

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np

IDENTITY_ASSET_SCHEMA_VERSION = "1.1.0"


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
    masks: Dict[str, np.ndarray]            # class_name -> binary uint8 mask (0 or 255)
    available_classes: Set[str]
    backend_name: str
    backend_version: str
    confidence_metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentedReferenceView:
    view_index: int
    image_path: str
    embedding: np.ndarray                   # (512,) float32, L2-normalized
    segmentation: SemanticSegmentationResult
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
        errors = []
        warnings = []
        available = set()

        if self.schema_version != IDENTITY_ASSET_SCHEMA_VERSION:
            errors.append(f"Schema version mismatch: got '{self.schema_version}', expected '{IDENTITY_ASSET_SCHEMA_VERSION}'")

        if self.fused_identity_embedding is None or self.fused_identity_embedding.shape != (512,):
            errors.append("Invalid or missing 512-D fused identity embedding")
        else:
            norm = float(np.linalg.norm(self.fused_identity_embedding))
            if abs(norm - 1.0) > 1e-3:
                errors.append(f"Fused identity embedding is not L2-normalized (norm = {norm:.4f})")
            else:
                available.add("face_embedding")

        if not self.segmented_views:
            errors.append("No segmented reference views present in asset")
        else:
            available.add("reference_views")

        if profile in (ValidationProfile.IDENTITY_PLUS_APPEARANCE, ValidationProfile.FULL_REFERENCE):
            has_masks = any(v.segmentation and v.segmentation.masks for v in self.segmented_views)
            if not has_masks:
                errors.append("Profile requires appearance masks, but none are present in reference views")
            else:
                available.add("appearance_masks")

        if profile == ValidationProfile.FULL_REFERENCE:
            required_classes = {"face", "hair", "clothing", "body"}
            found_classes = set()
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
```
