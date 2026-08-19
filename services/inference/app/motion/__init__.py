"""app/motion package — canonical motion representation and adapters."""

from app.motion.canonical_state import (
    CANONICAL_MOTION_STATE_VERSION,
    BodyPose,
    CanonicalMotionState,
    CONFIDENCE_THRESHOLD,
    FacialExpression,
    FingerState,
    HandPose,
    JointState,
)

__all__ = [
    "CANONICAL_MOTION_STATE_VERSION",
    "CONFIDENCE_THRESHOLD",
    "BodyPose",
    "CanonicalMotionState",
    "FacialExpression",
    "FingerState",
    "HandPose",
    "JointState",
]
