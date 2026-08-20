# PerformerState Specification

**Version:** 1.1.0  
**Status:** SPECIFICATION (Not yet implemented)  
**Date:** 2026-08-19  
**Changelog:**
- v1.1.0 — Corrected segmentation class count (5 → 6) and added authoritative taxonomy table. Added `CanonicalMotionState` reference.

---

## 1. Purpose

`PerformerState` is the normalized, backend-agnostic representation of a performer's complete motion state for a single captured frame. It is the **sole interface** between the motion extraction pipeline and all downstream identity rendering modules.

All downstream components (avatar driver, compositor, evaluation) must consume a `PerformerState`. They must not import tracking-backend-specific types.

---

## 2. Design Principles

1. **Backend-agnostic.** The state is expressed in terms of canonical coordinate systems, not MediaPipe or any other library's internal types.
2. **Self-describing.** Each field documents its coordinate system, units, and validity conditions.
3. **Partially observable.** All fields are optional or have a valid `None` state. The system degrades gracefully when a component (e.g., hands) is unavailable.
4. **Versioned.** The schema version is included so serialized states can be validated against the spec that produced them.
5. **Timestamped.** Every state carries both capture timestamp and frame index for temporal analysis.

---

## 3. Canonical Coordinate Systems

### 3.1 Image Coordinate System

Used for 2D pixel landmarks.

```
Origin: top-left corner of the captured frame
+x: rightward
+y: downward
Units: pixels (float)
```

### 3.2 Normalized Image Coordinate System

Used for proportional values that must be resolution-independent.

```
Origin: top-left corner
Range: [0.0, 1.0] in both axes
+x: rightward
+y: downward
```

### 3.3 Head-Canonical Face Coordinate System

Used for 3D face landmarks returned by MediaPipe FaceLandmarker.

```
Origin: approximately the nose tip
+x: rightward from the performer's perspective
+y: downward
+z: into the face (away from camera)
Units: canonical face units (not metric)
```

### 3.4 Head Rotation Convention

Euler angles, ZYX convention (same as existing `PoseResult`).

```
Pitch: rotation around X-axis  (positive = looking down)
Yaw:   rotation around Y-axis  (positive = looking right)
Roll:  rotation around Z-axis  (positive = tilting right)
Units: degrees
Range: pitch ∈ [-180, 180], yaw ∈ [-90, 90], roll ∈ [-180, 180]
```

### 3.5 Canonical Motion State Coordinate System

See [ADR-004: Canonical Motion Coordinate System & Transformation Contract](file:///e:/My_personal/Projects/ongoing/Chameleon/docs/architecture/ADR/ADR-004-canonical-coordinate-system.md).

```text
Origin: Pelvis (midpoint of hips) = (0.0, 0.0, 0.0)
+X: Camera Right (performer left for front camera)
+Y: UPWARD (away from floor / toward head)
+Z: TOWARD camera (out of chest)
Units: Body height normalized (1.0 = standing height)
```

Vertical Hierarchy Contract: `head.y > neck.y > chest.y > pelvis.y (0.0) > knee.y > ankle.y`

---

## 4. Data Schema (Python Dataclasses)

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

PERFORMER_STATE_SCHEMA_VERSION = "1.0.0"


@dataclass
class HeadRotation:
    """
    Euler angles (ZYX convention) representing head orientation.

    Fields
    ------
    pitch : float
        Rotation around X-axis in degrees. Positive = looking down.
    yaw : float
        Rotation around Y-axis in degrees. Positive = looking right.
    roll : float
        Rotation around Z-axis in degrees. Positive = tilting right.
    transformation_matrix : np.ndarray, shape (4, 4), float32
        Full 4x4 rigid-body transformation matrix from the MediaPipe
        FaceLandmarker output. Rotation submatrix is R = matrix[:3, :3].
    """
    pitch: float
    yaw: float
    roll: float
    transformation_matrix: np.ndarray  # (4, 4), float32


@dataclass
class FaceState:
    """
    Complete face motion state for a single tracked face.

    Fields
    ------
    track_id : int
        Unique, stable per-session identifier for this face.
    landmarks_2d : np.ndarray, shape (478, 2), float32
        Face landmarks in image pixel coordinates (x_pixel, y_pixel).
    landmarks_3d : np.ndarray, shape (478, 3), float32
        Face landmarks in head-canonical space (x, y, z).
        z is MediaPipe canonical depth — NOT metric world-space depth.
    head_rotation : HeadRotation
        Head orientation as Euler angles and 4x4 transformation matrix.
    blendshapes : Dict[str, float]
        ARKit-compatible 52 blendshape coefficients, keyed by name.
        Values are in [0.0, 1.0].
    confidence : float
        Detection/tracking confidence in [0.0, 1.0].
    """
    track_id: int
    landmarks_2d: np.ndarray            # (478, 2) float32, pixels
    landmarks_3d: np.ndarray            # (478, 3) float32, canonical
    head_rotation: HeadRotation
    blendshapes: Dict[str, float]       # 52 ARKit blendshapes
    confidence: float = 1.0


@dataclass
class HandState:
    """
    Hand pose state for one hand.

    Fields
    ------
    landmarks_3d : np.ndarray, shape (21, 3), float32
        Hand landmarks in world-normalized coordinates from MediaPipe.
        Each point is (x, y, z) where x, y are normalized to [0, 1]
        and z is the relative depth with wrist as origin.
    landmarks_2d : np.ndarray, shape (21, 2), float32
        Hand landmarks in image pixel coordinates.
    handedness : str
        'Left' or 'Right' as reported by MediaPipe Hands.
    confidence : float
        Handedness classification confidence in [0.0, 1.0].
    """
    landmarks_3d: np.ndarray            # (21, 3) float32
    landmarks_2d: np.ndarray            # (21, 2) float32
    handedness: str                     # 'Left' or 'Right'
    confidence: float = 1.0


@dataclass
class BodyPoseState:
    """
    Full-body pose state. Currently a structured placeholder for Phase 2.2.
    MediaPipe Pose provides 33 normalized 3D body landmarks.

    Fields
    ------
    landmarks_3d : np.ndarray, shape (33, 4), float32
        Body landmarks in world coordinates from MediaPipe Holistic/Pose.
        Each row is (x, y, z, visibility) where x, y, z are normalized
        to [0, 1] in the frame and visibility is confidence in [0, 1].
    landmarks_2d : np.ndarray, shape (33, 2), float32
        Body landmarks in image pixel coordinates.
    smplx_params : Optional[np.ndarray]
        If a body pose regressor (e.g., 4DHumans) has been run offline,
        this contains the SMPL-X body pose parameters (theta vector).
        None if not estimated.
    """
    landmarks_3d: np.ndarray            # (33, 4) float32, (x, y, z, vis)
    landmarks_2d: np.ndarray            # (33, 2) float32, pixels
    smplx_params: Optional[np.ndarray] = None


@dataclass
class SegmentationState:
    """
    Per-class segmentation masks for the performer.

    Segmentation Taxonomy (selfie_multiclass_256x256.tflite)
    ---------------------------------------------------------
    This is the authoritative class definition. Do not use any other
    mapping. The backend model provides exactly 6 classes (indices 0–5).

        Index 0: background  → not included in any named mask
        Index 1: hair        → hair_mask
        Index 2: body-skin   → body_skin_mask  (exposed skin: neck, arms, hands)
        Index 3: face-skin   → face_mask
        Index 4: clothes     → clothes_mask    (accessible via class_mask==4)
        Index 5: others      → accessible via class_mask==5 only

    person_mask is the union of indices 1+2+3+4+5 (everything non-background).

    All named boolean masks share shape (H, W), dtype bool.
    class_mask has shape (H, W), dtype uint8, values in {0, 1, 2, 3, 4, 5}.

    Fields
    ------
    person_mask : np.ndarray, shape (H, W), bool
        True = pixel belongs to any part of the performer (indices 1–5).
    face_mask : np.ndarray, shape (H, W), bool
        Face skin pixels only (index 3).
    hair_mask : np.ndarray, shape (H, W), bool
        Hair pixels (index 1).
    body_skin_mask : np.ndarray, shape (H, W), bool
        Exposed body skin (index 2): neck, arms, hands.
    clothes_mask : np.ndarray, shape (H, W), bool
        Clothing pixels (index 4).
    class_mask : np.ndarray, shape (H, W), uint8
        Raw 6-class label map. Values: {0=background, 1=hair, 2=body-skin,
        3=face-skin, 4=clothes, 5=others}.
    """
    person_mask: np.ndarray             # (H, W) bool
    face_mask: np.ndarray               # (H, W) bool
    hair_mask: np.ndarray               # (H, W) bool
    body_skin_mask: np.ndarray          # (H, W) bool
    clothes_mask: np.ndarray            # (H, W) bool
    class_mask: np.ndarray              # (H, W) uint8


@dataclass
class PerformerState:
    """
    Normalized, backend-agnostic performer motion state for one frame.

    This is the canonical contract between the motion extraction pipeline
    and all downstream identity rendering and evaluation modules.

    Fields
    ------
    schema_version : str
        Schema version string for forward compatibility validation.
    frame_index : int
        Zero-based monotonically increasing frame counter.
    capture_timestamp : float
        Unix timestamp (seconds) at the moment the frame was captured.
    frame_shape : tuple[int, int, int]
        (height, width, channels) of the original captured frame.
    faces : List[FaceState]
        All detected and tracked faces. Ordered by track_id ascending.
        Empty list when no face is detected.
    left_hand : Optional[HandState]
        Left hand state. None if not detected in this frame.
    right_hand : Optional[HandState]
        Right hand state. None if not detected in this frame.
    body : Optional[BodyPoseState]
        Full-body pose state. None until body pose backend is active.
    segmentation : Optional[SegmentationState]
        Per-class segmentation masks. None if segmenter is not running.
    primary_face_track_id : Optional[int]
        track_id of the face designated as the primary performer face.
        Used when multiple faces are in frame.
        Heuristic: largest face area, unless overridden by application logic.
    backend_timings : Dict[str, float]
        Per-stage latency in milliseconds, keyed by stage name.
        Example: {'detection': 3.2, 'landmarks': 17.0, 'segmentation': 8.5}
    """
    schema_version: str
    frame_index: int
    capture_timestamp: float
    frame_shape: tuple                  # (H, W, C)

    faces: List[FaceState] = field(default_factory=list)
    left_hand: Optional[HandState] = None
    right_hand: Optional[HandState] = None
    body: Optional[BodyPoseState] = None
    segmentation: Optional[SegmentationState] = None

    primary_face_track_id: Optional[int] = None
    backend_timings: Dict[str, float] = field(default_factory=dict)

    @property
    def primary_face(self) -> Optional[FaceState]:
        """Return the primary face state, or None if no face is present."""
        if not self.faces:
            return None
        if self.primary_face_track_id is not None:
            for f in self.faces:
                if f.track_id == self.primary_face_track_id:
                    return f
        return self.faces[0]

    @property
    def has_face(self) -> bool:
        return len(self.faces) > 0

    @property
    def has_hands(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None

    @property
    def has_body(self) -> bool:
        return self.body is not None

    @property
    def total_latency_ms(self) -> float:
        return sum(self.backend_timings.values())
```

---

## 5. Tracking Backend Adapter Contract

Each tracking backend must implement a factory function conforming to:

```python
def extract_performer_state(
    frame_bgr: np.ndarray,
    frame_index: int,
    capture_timestamp: float,
) -> PerformerState:
    """
    Extract a normalized PerformerState from a raw BGR camera frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Raw camera frame, BGR, dtype uint8.
    frame_index : int
        Monotonically increasing frame counter.
    capture_timestamp : float
        Unix timestamp at capture.

    Returns
    -------
    PerformerState
        Fully populated state object. Fields that cannot be extracted
        are set to None / empty list — never raise on partial failure.
    """
    ...
```

---

## 6. Validation Rules

A `PerformerState` is considered valid if:

1. `schema_version == PERFORMER_STATE_SCHEMA_VERSION`
2. `frame_index >= 0`
3. `capture_timestamp > 0`
4. `len(frame_shape) == 3 and frame_shape[2] == 3`
5. For each `FaceState` in `faces`:
   - `landmarks_2d.shape == (478, 2)`
   - `landmarks_3d.shape == (478, 3)`
   - `len(blendshapes) == 52`
6. For each `HandState`:
   - `landmarks_3d.shape == (21, 3)`
   - `landmarks_2d.shape == (21, 2)`
   - `handedness in ('Left', 'Right')`
7. For `BodyPoseState` (when present):
   - `landmarks_3d.shape == (33, 4)`
   - `landmarks_2d.shape == (33, 2)`

---

## 7. What PerformerState Does NOT Contain

- Raw pixel data of the performer frame (only shape is stored)
- Identity embeddings (these belong to the `IdentityAsset`)
- Rendered avatar frames
- Any information about the reference identity
- SMPL-X parameters by default (only populated by optional offline regressor)

---

## 8. Future Extensions (Not in v1.0)

| Field | When to Add |
|---|---|
| `gaze_direction` | When eye tracking is added |
| `body_depth_map` | When metric depth estimation is added |
| `smplx_params` in `BodyPoseState` | When SMPL-X regressor is integrated |
| `temporal_flow` | When optical flow stage is added |
| `hand_shape` | When MANO hand shape estimation is added |
