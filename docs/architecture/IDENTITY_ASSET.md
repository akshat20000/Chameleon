# IdentityAsset Specification

**Version:** 1.0.0  
**Status:** SPECIFICATION (Not yet implemented)  
**Date:** 2026-08-19

---

## 1. Purpose

`IdentityAsset` is the serializable, versioned package representing a single reference identity for use in Chameleon Phase 2. It is produced once during the offline identity preparation phase and consumed at runtime by the avatar driver and rendering modules.

The `IdentityAsset` is **not** the reference images themselves. It is a processed, structured representation derived from them.

---

## 2. Design Principles

1. **Versioned and validatable.** Every asset carries a schema version. Outdated assets can be detected and rebuilt.
2. **Reproducible.** Given the same input images and pipeline version, the asset must be bit-reproducible.
3. **Portable.** The asset must be loadable on any machine with the appropriate runtime — no path dependencies.
4. **Modular.** Each component of the identity (face, hair, body, hands) is a separate sub-object and can be updated independently.
5. **Transparent.** The asset stores which input frames contributed to each component.
6. **Partial.** An asset can be valid without all components. Missing components fall back to configurable policy.

---

## 3. Package Structure on Disk

```
identity_workspace/
└── {identity_id}/
    ├── manifest.json                  # Schema version, metadata, component checksums
    ├── inputs/
    │   ├── raw/                       # Original input images/video (never modified)
    │   └── selected_views/            # Curated views selected for processing
    ├── intermediate/
    │   ├── segmentation/              # Per-view segmentation masks
    │   ├── face_landmarks/            # Per-view 478-point face landmarks
    │   ├── body_landmarks/            # Per-view body pose estimates
    │   └── depth_estimates/           # Per-view depth maps (if available)
    ├── components/
    │   ├── face/
    │   │   ├── identity_embedding.npy # Global semantic identity vector (512-D)
    │   │   ├── canonical_texture.png  # UV-mapped face texture (optional)
    │   │   └── geometry_params.json   # Face geometry parameters
    │   ├── body/
    │   │   ├── shape_params.json      # Body shape parameters (beta if SMPL-X)
    │   │   └── appearance/            # Appearance representation files
    │   ├── hair/
    │   │   └── appearance/            # Hair appearance representation files
    │   ├── hands/
    │   │   └── appearance/            # Hand appearance representation files
    │   └── clothing/
    │       └── appearance/            # Clothing appearance files
    └── final/
        └── identity_asset.json        # Assembled asset manifest
```

---

## 4. Manifest Schema (manifest.json)

```json
{
  "schema_version": "1.0.0",
  "identity_id": "string (UUID v4)",
  "display_name": "string",
  "created_at": "ISO 8601 timestamp",
  "pipeline_version": "string (semantic version)",
  "input_summary": {
    "num_input_images": "int",
    "input_video_path": "string or null",
    "selected_view_indices": [0, 1, 2]
  },
  "components": {
    "face": {
      "present": true,
      "embedding_dim": 512,
      "embedding_model": "w600k_mbf.onnx",
      "checksum_sha256": "string"
    },
    "body": {
      "present": false,
      "reason": "Insufficient full-body coverage in input"
    },
    "hair": {
      "present": true,
      "checksum_sha256": "string"
    },
    "hands": {
      "present": false,
      "reason": "No clear hand views in input"
    },
    "clothing": {
      "present": true,
      "checksum_sha256": "string"
    }
  },
  "appearance_policy": {
    "face": "reference",
    "hair": "reference",
    "arms": "reference",
    "hands": "reference",
    "torso": {"preferred": "reference", "fallback": "performer_clothing"},
    "legs": {"preferred": "reference", "fallback": "performer_clothing"}
  }
}
```

---

## 5. Component Specifications

### 5.1 Face Component

| Field | Type | Description |
|---|---|---|
| `identity_embedding` | `np.ndarray (512,) float32` | L2-normalized MobileFaceNet embedding. Canonical identity signal. |
| `adaface_embedding` | `np.ndarray (512,) float32` | L2-normalized AdaFace ViT embedding. For generation conditioning. |
| `canonical_texture` | `np.ndarray (512, 512, 3) uint8` | UV-mapped face texture derived from best-view selection. Optional. |
| `geometry_params` | `dict` | Face geometry descriptor. Currently: `{landmarks_3d: np.ndarray (478, 3)}` from best frontal view. |
| `reference_view_images` | `List[str]` | Paths to selected reference view images (relative to workspace). |
| `contributing_frame_indices` | `List[int]` | Input frame indices that contributed to this component. |

### 5.2 Body Component

| Field | Type | Description |
|---|---|---|
| `shape_params` | `dict` | Body shape descriptor. Format TBD pending body model selection. |
| `appearance_representation` | `str` | Type tag: `"texture_map"`, `"gaussian_avatar"`, `"neural_texture"`. |
| `appearance_files` | `List[str]` | Paths to appearance model files (relative to workspace). |

### 5.3 Hair Component

| Field | Type | Description |
|---|---|---|
| `appearance_mask_views` | `List[str]` | Paths to binary hair segmentation masks per view. |
| `appearance_representation` | `str` | Type tag: `"segmentation_only"`, `"neural_texture"`. |
| `dominant_color` | `List[int]` | Approximate dominant hair color `[R, G, B]` for diagnostics. |

### 5.4 Hands Component

| Field | Type | Description |
|---|---|---|
| `landmark_views` | `List[str]` | Paths to hand views with detected landmarks. |
| `appearance_representation` | `str` | Type tag: `"segmentation_only"`, `"neural_texture"`. |

### 5.5 Clothing Component

| Field | Type | Description |
|---|---|---|
| `clothes_mask_views` | `List[str]` | Paths to clothing segmentation masks per view. |
| `appearance_representation` | `str` | Type tag: `"segmentation_only"`, `"texture_map"`. |

---

## 6. Appearance Policy

The `appearance_policy` field defines how each body region is rendered when the reference identity does not provide complete coverage.

```python
class AppearancePolicy(Enum):
    REFERENCE = "reference"                # Use reference identity appearance
    PERFORMER_CLOTHING = "performer_clothing"  # Use performer's clothing/appearance
    INPAINT = "inpaint"                    # Generative inpainting (Phase 2.5+)

@dataclass
class RegionPolicy:
    preferred: AppearancePolicy
    fallback: AppearancePolicy

@dataclass
class IdentityAppearancePolicy:
    face: AppearancePolicy = AppearancePolicy.REFERENCE
    hair: AppearancePolicy = AppearancePolicy.REFERENCE
    arms: AppearancePolicy = AppearancePolicy.REFERENCE
    hands: AppearancePolicy = AppearancePolicy.REFERENCE
    torso: RegionPolicy = RegionPolicy(
        preferred=AppearancePolicy.REFERENCE,
        fallback=AppearancePolicy.PERFORMER_CLOTHING,
    )
    legs: RegionPolicy = RegionPolicy(
        preferred=AppearancePolicy.REFERENCE,
        fallback=AppearancePolicy.PERFORMER_CLOTHING,
    )
```

---

## 7. Python Type Definition

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

IDENTITY_ASSET_SCHEMA_VERSION = "1.0.0"


@dataclass
class FaceIdentityComponent:
    identity_embedding: np.ndarray          # (512,) float32, L2-normalized
    adaface_embedding: Optional[np.ndarray] # (512,) float32, L2-normalized
    reference_view_images: List[str]        # relative paths to workspace
    canonical_texture: Optional[np.ndarray] # (512, 512, 3) uint8, optional


@dataclass
class BodyComponent:
    shape_params: Dict                      # TBD pending body model selection
    appearance_representation: str          # type tag
    appearance_files: List[str]             # relative paths to workspace


@dataclass
class HairComponent:
    appearance_mask_views: List[str]        # relative paths to mask PNGs
    dominant_color_rgb: Optional[List[int]] # [R, G, B]


@dataclass
class HandsComponent:
    landmark_views: List[str]              # relative paths
    appearance_representation: str          # type tag


@dataclass
class ClothingComponent:
    clothes_mask_views: List[str]          # relative paths
    appearance_representation: str          # type tag


@dataclass
class IdentityAsset:
    schema_version: str
    identity_id: str                        # UUID v4
    display_name: str
    created_at: str                         # ISO 8601
    pipeline_version: str

    face: Optional[FaceIdentityComponent] = None
    body: Optional[BodyComponent] = None
    hair: Optional[HairComponent] = None
    hands: Optional[HandsComponent] = None
    clothing: Optional[ClothingComponent] = None

    appearance_policy: Optional[Dict] = None

    @property
    def is_valid(self) -> bool:
        """An asset is minimally valid if it has a face component."""
        return (
            self.face is not None
            and self.face.identity_embedding is not None
            and self.face.identity_embedding.shape == (512,)
        )
```

---

## 8. Validation Rules

An `IdentityAsset` is considered complete (not just minimally valid) if:

1. `schema_version == IDENTITY_ASSET_SCHEMA_VERSION`
2. `face` is present with a valid 512-D L2-normalized identity embedding
3. `face.reference_view_images` contains at least 1 existing path
4. `appearance_policy` is present and all mandatory region keys are defined
5. `manifest.json` checksum for each present component matches the on-disk file

---

## 9. What IdentityAsset Does NOT Contain

- Raw input video frames (only paths to selected views)
- The performer's motion state (that belongs to `PerformerState`)
- Rendered output frames
- Training data for any neural model
