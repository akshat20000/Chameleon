# CanonicalMotionState Specification

**Version:** 1.0.0  
**Status:** SPECIFICATION  
**Date:** 2026-08-19

---

## 1. Purpose

`CanonicalMotionState` is the backend-independent, normalized joint-level motion representation that sits between the raw tracker output (`PerformerState`) and all downstream avatar/rendering modules.

**Nothing downstream of `CanonicalMotionState` may import or reference MediaPipe APIs, landmark indices, or any tracker-specific type.**

The adapter that converts `PerformerState` → `CanonicalMotionState` is the single, replaceable boundary. Replacing the tracker (MediaPipe → SMPL-X regressor, RTMPose, etc.) requires only replacing this adapter — not the avatar driver or any rendering code.

---

## 2. Design Principles

1. **Physically grounded.** All positions and rotations are expressed in a documented, stable coordinate system — not in normalized tracker-space.
2. **Graceful degradation.** Every joint carries a confidence value. Missing joints are `None`, not fabricated.
3. **No tracker-specific types.** No imports from `mediapipe`, `smplx`, or any other tracker library.
4. **Dual representation.** Joint positions (world 3D) and orientations (rotation matrices) are both stored. Downstream consumers pick whichever representation suits their renderer.
5. **Temporal.** Every state carries a monotonic frame index and wall-clock timestamp for temporal analysis.

---

## 3. Coordinate System

### 3.1 World Coordinate System

See [ADR-004: Canonical Motion Coordinate System & Transformation Contract](file:///e:/My_personal/Projects/ongoing/Chameleon/docs/architecture/ADR/ADR-004-canonical-coordinate-system.md).

```text
Origin:    Pelvis (midpoint of left_hip and right_hip landmarks) = (0.0, 0.0, 0.0)
+Y axis:   Upward (away from the floor / toward sky)
+X axis:   Camera Right (performer's anatomical left in front-facing camera)
+Z axis:   Forward (toward the camera, out of the performer's chest)
Handedness: Right-handed
Units:     Normalized body height
           1.0 unit ≈ full standing body height of the performer.
```

**Anatomical Vertical Hierarchy Contract (+Y Upward):**
$$\text{head.y} > \text{neck.y} > \text{chest.y} > \text{pelvis.y } (0.0) > \text{knee.y} > \text{ankle.y}$$

### 3.2 Joint Rotation Convention

All rotation matrices are 3×3 SO(3) matrices.

**Reference pose (zero rotation):**
- T-pose: arms extended horizontally, palms down, facing +Z (camera direction)
- Each bone's local +Y axis points from proximal joint to distal joint (along the limb)
- Each bone's local +Z axis points forward (toward camera in T-pose)

Joint rotations are expressed relative to the parent joint's coordinate frame. This forms a standard kinematic chain compatible with parametric body models.

For joints derived from position-only trackers (e.g., raw MediaPipe landmarks), rotations are estimated by:
```
bone_direction = normalize(child_position - parent_position)
rotation_matrix = align_vector_to_canonical(bone_direction, canonical_bone_direction)
```

### 3.3 Note on MediaPipe World Coordinate System

MediaPipe Pose Landmarker world landmarks use:
- Origin at hip midpoint (compatible with our origin)
- Y pointing up (same convention)
- Z pointing toward camera (same convention)
- Units: approximately metric (meters), not normalized

The adapter normalizes MediaPipe world coordinates to body-height units before populating `CanonicalMotionState`.

---

## 4. Joint Hierarchy

```
pelvis (root)
├── spine_mid
│   └── chest
│       ├── left_shoulder
│       │   └── left_elbow
│       │       └── left_wrist
│       │           ├── left_thumb     (3 joints: CMC, IP, TIP)
│       │           ├── left_index     (3 joints: MCP, PIP, DIP)
│       │           ├── left_middle    (3 joints: MCP, PIP, DIP)
│       │           ├── left_ring      (3 joints: MCP, PIP, DIP)
│       │           └── left_pinky     (3 joints: MCP, PIP, DIP)
│       ├── right_shoulder
│       │   └── right_elbow
│       │       └── right_wrist
│       │           ├── right_thumb
│       │           ├── right_index
│       │           ├── right_middle
│       │           ├── right_ring
│       │           └── right_pinky
│       └── neck
│           └── head
├── left_hip
│   └── left_knee
│       └── left_ankle
└── right_hip
    └── right_knee
        └── right_ankle
```

---

## 5. Data Schema (Python Dataclasses)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CANONICAL_MOTION_STATE_VERSION = "1.0.0"


@dataclass
class JointState:
    """
    State of a single joint.

    Fields
    ------
    position : np.ndarray, shape (3,), float32
        Joint position in canonical world coordinates (origin = pelvis,
        +Y up, +X right, +Z toward camera, units = normalized body height).
    rotation : np.ndarray or None, shape (3, 3), float32
        3×3 rotation matrix (SO3) relative to parent joint's frame.
        None when rotation cannot be estimated (e.g. leaf joint or
        visibility < confidence_threshold).
    confidence : float
        Landmark visibility / tracking confidence in [0.0, 1.0].
        Values below 0.3 indicate the joint is likely outside the frame
        or occluded. Downstream consumers should treat low-confidence
        joints as unreliable.
    is_visible : bool
        True if confidence >= 0.5. Convenience flag for conditional rendering.
    """
    position: np.ndarray                  # (3,) float32, canonical world coords
    rotation: Optional[np.ndarray] = None # (3,3) float32, SO3
    confidence: float = 1.0
    is_visible: bool = True


@dataclass
class FingerState:
    """
    Pose of a single finger (3 joints: proximal, middle, distal).

    Fields
    ------
    proximal : JointState
    middle : JointState
    distal : JointState
    """
    proximal: JointState
    middle: JointState
    distal: JointState


@dataclass
class HandPose:
    """
    Full 5-finger hand pose in canonical space.

    Fields
    ------
    wrist : JointState
        Wrist position in canonical world coordinates.
    thumb : FingerState    (CMC, IP, TIP)
    index : FingerState    (MCP, PIP, DIP)
    middle : FingerState   (MCP, PIP, DIP)
    ring : FingerState     (MCP, PIP, DIP)
    pinky : FingerState    (MCP, PIP, DIP)
    handedness : str       'Left' or 'Right'
    overall_confidence : float
        Minimum confidence across all finger joints — conservative estimate.
    """
    wrist: JointState
    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState
    handedness: str                        # 'Left' or 'Right'
    overall_confidence: float = 1.0


@dataclass
class FacialExpression:
    """
    Canonical face expression state.

    Fields
    ------
    blendshapes : Dict[str, float]
        52 ARKit blendshape coefficients, keyed by ARKit name.
        Values in [0.0, 1.0]. Empty dict if face not detected.
    head_rotation : np.ndarray, shape (3, 3), float32
        3×3 rotation matrix representing head orientation in canonical world
        coordinates. Identity matrix = neutral forward-facing pose.
    eye_open_left : float
        Left eye openness coefficient from blendshapes in [0.0, 1.0].
    eye_open_right : float
        Right eye openness coefficient in [0.0, 1.0].
    jaw_open : float
        Jaw open coefficient in [0.0, 1.0].
    confidence : float
        Face detection confidence in [0.0, 1.0].
    """
    blendshapes: Dict[str, float]          # 52 ARKit coefficients
    head_rotation: np.ndarray              # (3,3) float32
    eye_open_left: float = 1.0
    eye_open_right: float = 1.0
    jaw_open: float = 0.0
    confidence: float = 1.0


@dataclass
class BodyPose:
    """
    Full-body joint state in canonical world coordinates.

    The root of the kinematic chain is the pelvis.
    All joint positions are in the same canonical world frame
    (pelvis at origin, +Y up, +X right, +Z toward camera).

    Fields marked Optional are None when the joint was not tracked
    with sufficient confidence (< 0.3).
    """
    # Root
    pelvis: JointState

    # Spine chain
    spine_mid: Optional[JointState] = None
    chest: Optional[JointState] = None
    neck: Optional[JointState] = None
    head: Optional[JointState] = None

    # Left arm chain
    left_shoulder: Optional[JointState] = None
    left_elbow: Optional[JointState] = None
    left_wrist: Optional[JointState] = None

    # Right arm chain
    right_shoulder: Optional[JointState] = None
    right_elbow: Optional[JointState] = None
    right_wrist: Optional[JointState] = None

    # Left leg chain
    left_hip: Optional[JointState] = None
    left_knee: Optional[JointState] = None
    left_ankle: Optional[JointState] = None

    # Right leg chain
    right_hip: Optional[JointState] = None
    right_knee: Optional[JointState] = None
    right_ankle: Optional[JointState] = None


@dataclass
class CanonicalMotionState:
    """
    Backend-independent, normalized joint-level motion state for one frame.

    This is the primary contract consumed by all avatar and rendering modules.
    It must not contain any tracker-specific types (MediaPipe, SMPL-X, etc.).

    Fields
    ------
    schema_version : str
        Schema version for forward compatibility validation.
    frame_index : int
        Monotonically increasing frame counter.
    capture_timestamp : float
        Unix timestamp (seconds) at capture.
    source_backend : str
        Identifier of the motion backend that produced this state.
        Example: 'mediapipe_tasks_v1.0.0', 'smplx_4dhumans', 'ground_truth'.
        Used for debugging and pipeline diagnostics only.
    body : BodyPose
        Full-body joint state. pelvis is always present; all other joints
        are Optional and None when not tracked.
    face : Optional[FacialExpression]
        Face expression and head orientation. None when face not detected.
    left_hand : Optional[HandPose]
        Left hand pose. None when not detected.
    right_hand : Optional[HandPose]
        Right hand pose. None when not detected.
    body_scale : float
        Estimated body height in the tracker's native units.
        Used to convert back to metric scale if needed.
        0.0 if estimation failed.
    adapter_timings : Dict[str, float]
        Per-stage adapter latency in milliseconds.
    """
    schema_version: str
    frame_index: int
    capture_timestamp: float
    source_backend: str

    body: BodyPose
    face: Optional[FacialExpression] = None
    left_hand: Optional[HandPose] = None
    right_hand: Optional[HandPose] = None

    body_scale: float = 0.0
    adapter_timings: Dict[str, float] = field(default_factory=dict)

    @property
    def has_face(self) -> bool:
        return self.face is not None and self.face.confidence >= 0.5

    @property
    def has_left_hand(self) -> bool:
        return self.left_hand is not None and self.left_hand.overall_confidence >= 0.3

    @property
    def has_right_hand(self) -> bool:
        return self.right_hand is not None and self.right_hand.overall_confidence >= 0.3

    @property
    def tracking_quality(self) -> str:
        """
        High-level tracking quality assessment.
        Returns: 'full', 'partial', or 'face_only'.
        """
        has_body = (
            self.body.left_shoulder is not None
            and self.body.right_shoulder is not None
            and self.body.left_hip is not None
            and self.body.right_hip is not None
        )
        if has_body and self.has_face:
            return "full"
        if has_body:
            return "partial"
        return "face_only"
```

---

## 6. Adapter Contract

The adapter converting `PerformerState` → `CanonicalMotionState` must:

1. Receive a `PerformerState` (tracker output)
2. Produce a `CanonicalMotionState`
3. Never raise — return a degraded state on partial failure
4. Validate joint consistency (left/right inversion check, NaN check)
5. Propagate confidence from source landmarks to joint states
6. Normalize all coordinates to body-height units

```python
def adapt_performer_state(state: PerformerState) -> CanonicalMotionState:
    """
    Convert a PerformerState to a CanonicalMotionState.

    This is the ONLY function allowed to access MediaPipe-specific field
    structures. All downstream modules receive only CanonicalMotionState.
    """
    ...
```

---

## 7. Validation Rules

A `CanonicalMotionState` is considered valid if:

1. `schema_version == CANONICAL_MOTION_STATE_VERSION`
2. `frame_index >= 0`
3. `capture_timestamp > 0.0`
4. `body.pelvis` is not None (pelvis is always estimated)
5. All joint positions that are not None have shape `(3,)` and no NaN values
6. All joint rotations that are not None have shape `(3, 3)` and are approximately orthogonal (`|det(R) - 1| < 0.01`)
7. No left/right inversion: `body.left_shoulder.position[0] < body.right_shoulder.position[0]` when both are visible
8. All confidence values are in `[0.0, 1.0]`

---

## 8. Missing Joint Policy

When a joint landmark is below the confidence threshold or outside the frame:

- Set the corresponding `JointState` to `None`
- Do **not** extrapolate or hallucinate position from neighboring joints
- Do **not** carry forward the position from the previous frame (temporal smoothing is a separate stage)

The avatar driver is responsible for handling `None` joints according to its own fallback policy (e.g., last known position, bind pose, or hiding the limb).

---

## 9. Relationship to PerformerState

```
PerformerState          CanonicalMotionState
───────────────         ─────────────────────
FaceState            →  FacialExpression
  landmarks_2d           (not stored — 3D only)
  landmarks_3d           (head_rotation extracted from face matrix)
  blendshapes            blendshapes (copied directly)
  head_rotation          head_rotation

BodyPoseState        →  BodyPose
  landmarks_3d[23,24]    pelvis (mean of hips)
  landmarks_3d[11,12]    chest (mean of shoulders)
  landmarks_3d[11]       left_shoulder
  landmarks_3d[12]       right_shoulder
  ...etc...

HandState (left)     →  left_hand: HandPose
HandState (right)    →  right_hand: HandPose

SegmentationState    →  (not included — belongs to compositor, not avatar driver)
```

---

## 10. Future Extensions (Not in v1.0)

| Extension | Trigger |
|---|---|
| `smplx_params` in `BodyPose` | When SMPL-X regressor replaces raw MediaPipe landmarks |
| `velocity` per joint | When temporal smoothing stage is added |
| `contact` flags (foot/floor) | When physics-aware driving is added |
| Finger curl scalar per finger | When simplified hand representation is needed |
