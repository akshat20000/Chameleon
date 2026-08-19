"""
CanonicalMotionState — backend-independent joint-level motion representation.

This module defines the canonical motion contract. Nothing in the avatar or
rendering subsystem should import from mediapipe, smplx, or any other tracker.
All tracker-specific logic lives in the adapter modules (e.g., mediapipe_adapter).

Coordinate system
-----------------
Origin:     pelvis (midpoint of left_hip and right_hip)
+Y:         up (away from floor)
+X:         camera right (NOT performer's anatomical right)
            In a front-facing camera:
              performer's LEFT arm is at +X (camera-right)
              performer's RIGHT arm is at -X (camera-left)
+Z:         toward camera (out of performer's chest)
Handedness: right-handed
Units:      normalized body height
            1.0 unit = full standing height of the performer

NOTE on left/right in camera space:
    MediaPipe world landmarks follow camera-space convention.
    left_shoulder.x > right_shoulder.x (both visible, front-facing camera).
    This is NOT an inversion — it is the expected camera-space orientation.
    Downstream modules that need performer-anatomical coordinates must negate X:
        performer_right_x = -canonical_x

Joint rotation convention
--------------------------
All rotation matrices are 3×3 SO(3) matrices.
Reference T-pose: arms extended horizontally, palms down, facing +Z.
Each bone's local +Y axis points from proximal → distal joint.
Joint rotations are relative to the parent joint's coordinate frame.

Version
-------
CANONICAL_MOTION_STATE_VERSION = "1.0.0"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

CANONICAL_MOTION_STATE_VERSION = "1.0.0"

# Confidence threshold below which a joint is considered unreliable.
CONFIDENCE_THRESHOLD = 0.3


@dataclass
class JointState:
    """
    State of a single skeletal joint in canonical world coordinates.

    Fields
    ------
    position : np.ndarray, shape (3,), float32
        Joint position in canonical world coordinates.
        Origin = pelvis. +Y up. +X right. +Z toward camera.
        Units = normalized body height.
    rotation : np.ndarray or None, shape (3, 3), float32
        3×3 rotation matrix (SO3) relative to parent joint's frame.
        None when rotation cannot be estimated (leaf joint or low confidence).
    confidence : float
        Landmark visibility / tracking confidence in [0.0, 1.0].
        Values below CONFIDENCE_THRESHOLD indicate the joint is likely
        outside the frame or occluded.
    is_visible : bool
        True if confidence >= CONFIDENCE_THRESHOLD. Convenience flag.
    """
    position: np.ndarray                    # (3,) float32
    rotation: Optional[np.ndarray] = None   # (3, 3) float32, SO3 or None
    confidence: float = 1.0
    is_visible: bool = True

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32)
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=np.float32)
        self.is_visible = self.confidence >= CONFIDENCE_THRESHOLD


@dataclass
class FingerState:
    """
    Pose of a single finger (3 joints).

    thumb:  CMC, IP, TIP
    others: MCP, PIP, DIP
    """
    proximal: JointState
    middle: JointState
    distal: JointState


@dataclass
class HandPose:
    """
    Full 5-finger hand pose in canonical world coordinates.

    Fields
    ------
    wrist : JointState
    thumb, index, middle, ring, pinky : FingerState
    handedness : str  ('Left' or 'Right')
    overall_confidence : float
        Minimum confidence across all joints — conservative estimate.
    """
    wrist: JointState
    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState
    handedness: str
    overall_confidence: float = 1.0


@dataclass
class FacialExpression:
    """
    Canonical face expression and head orientation state.

    Fields
    ------
    blendshapes : Dict[str, float]
        52 ARKit blendshape coefficients, keyed by ARKit name.
        Values in [0.0, 1.0]. Empty dict if face not detected.
    head_rotation : np.ndarray, shape (3, 3), float32
        Head orientation in canonical world coordinates.
        Identity = neutral forward-facing T-pose.
    eye_open_left : float
        Left eye openness in [0.0, 1.0].
    eye_open_right : float
        Right eye openness in [0.0, 1.0].
    jaw_open : float
        Jaw open coefficient in [0.0, 1.0].
    confidence : float
        Face detection confidence in [0.0, 1.0].
    """
    blendshapes: Dict[str, float]
    head_rotation: np.ndarray               # (3, 3) float32
    eye_open_left: float = 1.0
    eye_open_right: float = 1.0
    jaw_open: float = 0.0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.head_rotation = np.asarray(self.head_rotation, dtype=np.float32)


@dataclass
class BodyPose:
    """
    Full-body joint state in canonical world coordinates.

    pelvis is the kinematic root and is always populated.
    All other joints are Optional — None when not tracked with
    sufficient confidence (< CONFIDENCE_THRESHOLD).

    Left/Right convention: from the performer's perspective.
    left_shoulder is on the performer's left side (which appears
    on the right side of a front-facing camera view).

    MediaPipe landmark index mapping (for adapter reference only):
        pelvis       = midpoint(lm[23], lm[24])
        spine_mid    = midpoint(pelvis, chest)  [estimated]
        chest        = midpoint(lm[11], lm[12])
        neck         = midpoint(chest, head)    [estimated]
        head         = lm[0] (nose)
        left_shoulder  = lm[11]
        left_elbow     = lm[13]
        left_wrist     = lm[15]
        right_shoulder = lm[12]
        right_elbow    = lm[14]
        right_wrist    = lm[16]
        left_hip       = lm[23]
        left_knee      = lm[25]
        left_ankle     = lm[27]
        right_hip      = lm[24]
        right_knee     = lm[26]
        right_ankle    = lm[28]
    """
    # Root — always populated
    pelvis: JointState

    # Spine chain
    spine_mid: Optional[JointState] = None
    chest: Optional[JointState] = None
    neck: Optional[JointState] = None
    head: Optional[JointState] = None

    # Left arm chain (performer's left)
    left_shoulder: Optional[JointState] = None
    left_elbow: Optional[JointState] = None
    left_wrist: Optional[JointState] = None

    # Right arm chain (performer's right)
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

    def all_joints(self) -> Dict[str, Optional[JointState]]:
        """Return all joints as a name → JointState mapping."""
        return {
            "pelvis":          self.pelvis,
            "spine_mid":       self.spine_mid,
            "chest":           self.chest,
            "neck":            self.neck,
            "head":            self.head,
            "left_shoulder":   self.left_shoulder,
            "left_elbow":      self.left_elbow,
            "left_wrist":      self.left_wrist,
            "right_shoulder":  self.right_shoulder,
            "right_elbow":     self.right_elbow,
            "right_wrist":     self.right_wrist,
            "left_hip":        self.left_hip,
            "left_knee":       self.left_knee,
            "left_ankle":      self.left_ankle,
            "right_hip":       self.right_hip,
            "right_knee":      self.right_knee,
            "right_ankle":     self.right_ankle,
        }

    def visible_joint_count(self) -> int:
        return sum(
            1 for j in self.all_joints().values()
            if j is not None and j.is_visible
        )


@dataclass
class CanonicalMotionState:
    """
    Backend-independent, normalized joint-level motion state for one frame.

    This is the primary contract consumed by all avatar and rendering modules.
    It must not contain any tracker-specific types (MediaPipe, SMPL-X, etc.).

    Fields
    ------
    schema_version : str
        Schema version string.
    frame_index : int
        Zero-based monotonically increasing frame counter.
    capture_timestamp : float
        Unix timestamp (seconds) at the moment the source frame was captured.
    source_backend : str
        Identifier of the motion backend that produced this state.
        For logging and diagnostics only — not a type discriminator.
        Example: 'mediapipe_tasks_v1.0.0'
    body : BodyPose
        Full-body joint state. pelvis is always populated.
    face : Optional[FacialExpression]
        Face expression and head orientation. None if face not detected.
    left_hand : Optional[HandPose]
        Left hand pose. None if not detected or confidence too low.
    right_hand : Optional[HandPose]
        Right hand pose. None if not detected or confidence too low.
    body_scale : float
        Estimated body height in the tracker's native metric units (meters).
        Used to convert canonical units back to metric if needed.
        0.0 if estimation failed (e.g., partial body in frame).
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

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def has_face(self) -> bool:
        return self.face is not None and self.face.confidence >= CONFIDENCE_THRESHOLD

    @property
    def has_left_hand(self) -> bool:
        return (
            self.left_hand is not None
            and self.left_hand.overall_confidence >= CONFIDENCE_THRESHOLD
        )

    @property
    def has_right_hand(self) -> bool:
        return (
            self.right_hand is not None
            and self.right_hand.overall_confidence >= CONFIDENCE_THRESHOLD
        )

    @property
    def tracking_quality(self) -> str:
        """
        High-level tracking quality assessment.
        Returns 'full', 'partial', or 'face_only'.
        'full' = torso + all 4 limbs + face detected at sufficient confidence.
        'partial' = body detected but missing limbs or face.
        'face_only' = only face detected.
        """
        b = self.body
        torso_ok = (
            b.left_shoulder is not None and b.left_shoulder.is_visible
            and b.right_shoulder is not None and b.right_shoulder.is_visible
            and b.left_hip is not None and b.left_hip.is_visible
            and b.right_hip is not None and b.right_hip.is_visible
        )
        if torso_ok and self.has_face:
            return "full"
        if torso_ok:
            return "partial"
        return "face_only"

    def validate(self) -> list[str]:
        """
        Validate the state and return a list of error strings.
        Empty list = valid.
        """
        errors: list[str] = []

        if self.schema_version != CANONICAL_MOTION_STATE_VERSION:
            errors.append(
                f"schema_version mismatch: got {self.schema_version!r}, "
                f"expected {CANONICAL_MOTION_STATE_VERSION!r}"
            )

        if self.frame_index < 0:
            errors.append(f"frame_index must be >= 0, got {self.frame_index}")

        if self.capture_timestamp <= 0:
            errors.append(f"capture_timestamp must be > 0, got {self.capture_timestamp}")

        for name, joint in self.body.all_joints().items():
            if joint is None:
                continue
            if joint.position.shape != (3,):
                errors.append(f"joint {name}: position.shape {joint.position.shape} != (3,)")
            if np.any(np.isnan(joint.position)):
                errors.append(f"joint {name}: position contains NaN")
            if joint.rotation is not None:
                if joint.rotation.shape != (3, 3):
                    errors.append(
                        f"joint {name}: rotation.shape {joint.rotation.shape} != (3, 3)"
                    )
                det = float(np.linalg.det(joint.rotation))
                if abs(det - 1.0) > 0.01:
                    errors.append(f"joint {name}: rotation det={det:.4f}, expected ~1.0")
            if not (0.0 <= joint.confidence <= 1.0):
                errors.append(f"joint {name}: confidence {joint.confidence} not in [0, 1]")

        # Left/right camera-space check
        # In MediaPipe camera-space convention:
        #   performer's LEFT shoulder has POSITIVE X (camera-right)
        #   performer's RIGHT shoulder has NEGATIVE X (camera-left)
        # left_shoulder.x > right_shoulder.x is EXPECTED for a front-facing performer.
        # We only flag genuine inversions where left.x << right.x.
        ls = self.body.left_shoulder
        rs = self.body.right_shoulder
        if ls is not None and rs is not None and ls.is_visible and rs.is_visible:
            # Inversion: right shoulder appears to the camera's right of left shoulder.
            if rs.position[0] > ls.position[0] + 0.05:
                errors.append(
                    "left/right shoulder inversion detected (camera-space): "
                    f"right.x={rs.position[0]:.3f} > left.x={ls.position[0]:.3f}"
                )

        return errors
