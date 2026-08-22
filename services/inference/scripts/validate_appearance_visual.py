"""
Visual Debug Benchmark & Animation Generator for Phase 2.5A Appearance Conditioning.

Spec: docs/architecture/ADR/ADR-006-appearance-synthesis-architecture.md
Generates 3-panel side-by-side debug animation video and per-frame inspection panels:
Panel 1: Reference Mannequin Source Regions & Masks
Panel 2: Target Pose & Keypoint Anchors Overlay
Panel 3: Baseline Articulated Synthesizer Output (Transformed Regions + Z-order)
"""

from __future__ import annotations

import logging
import math
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.appearance.conditioning_builder import (
    AppearanceConditioningBuilder,
    AppearanceTemporalState,
)
from app.appearance.pose_renderer import CameraParameters, SkeletalPoseRenderer
from app.appearance.synthesizer import BaselineArticulatedSynthesizer
from app.identity.identity_asset import (
    IDENTITY_ASSET_SCHEMA_VERSION,
    PIPELINE_VERSION,
    IdentityAsset,
    SegmentedReferenceView,
    SemanticSegmentationResult,
)
from app.motion.actor_skeleton import ACTOR_PROFILES, ActorSkeleton
from app.motion.local_rotation_extractor import LocalJointRotations
from app.motion.retargeted_actor_state import RetargetedActorState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_appearance_visual")


def create_multi_region_mannequin_identity() -> Tuple[IdentityAsset, np.ndarray]:
    """Create a synthetic reference mannequin IdentityAsset with distinct region masks."""
    ref_img = np.zeros((512, 512, 3), dtype=np.uint8)
    ref_img[:, :] = [30, 30, 30]  # dark background

    # Masks (512x512)
    m_face = np.zeros((512, 512), dtype=np.uint8)
    m_hair = np.zeros((512, 512), dtype=np.uint8)
    m_torso = np.zeros((512, 512), dtype=np.uint8)
    m_larm = np.zeros((512, 512), dtype=np.uint8)
    m_rarm = np.zeros((512, 512), dtype=np.uint8)
    m_lleg = np.zeros((512, 512), dtype=np.uint8)
    m_rleg = np.zeros((512, 512), dtype=np.uint8)

    # Torso (mid-body blue)
    cv2.rectangle(ref_img, (180, 180), (332, 340), (200, 100, 50), -1)
    cv2.rectangle(m_torso, (180, 180), (332, 340), 255, -1)

    # Left Arm (cyan)
    cv2.rectangle(ref_img, (110, 190), (170, 300), (255, 128, 0), -1)
    cv2.rectangle(m_larm, (110, 190), (170, 300), 255, -1)

    # Right Arm (magenta)
    cv2.rectangle(ref_img, (342, 190), (402, 300), (255, 0, 255), -1)
    cv2.rectangle(m_rarm, (342, 190), (402, 300), 255, -1)

    # Left Leg (dark cyan)
    cv2.rectangle(ref_img, (190, 345), (245, 480), (0, 200, 200), -1)
    cv2.rectangle(m_lleg, (190, 345), (245, 480), 255, -1)

    # Right Leg (yellow)
    cv2.rectangle(ref_img, (267, 345), (322, 480), (200, 200, 0), -1)
    cv2.rectangle(m_rleg, (267, 345), (322, 480), 255, -1)

    # Face (peach)
    cv2.ellipse(ref_img, (256, 120), (50, 65), 0, 0, 360, (180, 200, 230), -1)
    cv2.ellipse(m_face, (256, 120), (50, 65), 0, 0, 360, 255, -1)
    cv2.circle(ref_img, (230, 110), 8, (30, 30, 30), -1)  # Left Eye
    cv2.circle(ref_img, (282, 110), 8, (30, 30, 30), -1)  # Right Eye
    cv2.ellipse(ref_img, (256, 145), (15, 8), 0, 0, 360, (50, 50, 150), -1)  # Mouth

    # Hair (dark red)
    cv2.ellipse(ref_img, (256, 80), (60, 35), 0, 0, 360, (20, 20, 120), -1)
    cv2.ellipse(m_hair, (256, 80), (60, 35), 0, 0, 360, 255, -1)

    m_body = m_face | m_torso | m_larm | m_rarm | m_lleg | m_rleg

    vec = np.ones(512, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    view_0 = SegmentedReferenceView(
        view_index=0,
        image_path="inputs/selected_views/view_000.png",
        embedding=vec.copy(),
        segmentation=SemanticSegmentationResult(
            masks={
                "face": m_face,
                "hair": m_hair,
                "torso": m_torso,
                "clothing": m_torso,
                "left_arm": m_larm,
                "right_arm": m_rarm,
                "left_leg": m_lleg,
                "right_leg": m_rleg,
                "body": m_body,
            },
            available_classes={"face", "hair", "torso", "clothing", "left_arm", "right_arm", "left_leg", "right_leg", "body"},
            backend_name="MultiRegionMannequinFixture",
            backend_version="1.0",
        ),
        quality_score=0.98,
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
    )

    # Attach image_bgr directly to ref_view for visual fixture rendering
    view_0.image_bgr = ref_img

    asset = IdentityAsset(
        schema_version=IDENTITY_ASSET_SCHEMA_VERSION,
        identity_id="MannequinMultiRegionIdentity_01",
        display_name="MultiRegionMannequin",
        created_at="2026-08-21T20:00:00Z",
        pipeline_version=PIPELINE_VERSION,
        provenance={"backend": "MultiRegionMannequinFixture"},
        fused_identity_embedding=vec.copy(),
        segmented_views=[view_0],
    )

    return asset, ref_img


def generate_5_pose_retargeted_sequence(num_frames: int = 60) -> List[RetargetedActorState]:
    """
    Generate RetargetedActorState frames covering 5 distinct pose regimes:
    Pose 0: T-pose (frames 0-11)
    Pose 1: Left arm raised (frames 12-23)
    Pose 2: Right arm raised (frames 24-35)
    Pose 3: Torso motion / rotation (frames 36-47)
    Pose 4: Walking-like pose (frames 48-59)
    """
    skel = ACTOR_PROFILES["DEFAULT"]
    joints = list(skel.rest_positions.keys())
    sequence = []

    for f in range(num_frames):
        t = f * 0.05
        positions = {}
        rotations = {}

        # Pose Regime determination
        regime = f // 12

        for j in joints:
            p = skel.rest_positions[j].copy()

            if regime == 0:
                # Pose 0: T-Pose
                pass
            elif regime == 1:
                # Pose 1: Left arm raised
                if "left_elbow" in j or "left_wrist" in j:
                    p[1] += 0.25
                    p[0] -= 0.10
            elif regime == 2:
                # Pose 2: Right arm raised
                if "right_elbow" in j or "right_wrist" in j:
                    p[1] += 0.25
                    p[0] += 0.10
            elif regime == 3:
                # Pose 3: Torso motion / rotation
                rot_angle = math.sin(t * 4.0) * 0.2
                if j != "pelvis":
                    p[0] += math.sin(rot_angle) * 0.15
            elif regime == 4:
                # Pose 4: Walking-like stride
                stride = math.sin(t * 5.0) * 0.15
                if "left_knee" in j or "left_ankle" in j:
                    p[2] += stride
                elif "right_knee" in j or "right_ankle" in j:
                    p[2] -= stride

            positions[j] = p.astype(np.float32)
            rotations[j] = np.eye(3, dtype=np.float32)

        loc_rot = LocalJointRotations(joints=rotations)
        state = RetargetedActorState(
            frame_index=f,
            capture_timestamp=t,
            actor_name="DEFAULT",
            joints=positions,
            world_rotations=rotations,
            local_rotations=loc_rot,
            motion_deltas=loc_rot,
            actor_skeleton=skel,
            source_frame_index=f,
        )
        sequence.append(state)

    return sequence


def create_keypoint_overlay_panel(
    pose_map: np.ndarray,
    keypoints_2d: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Create keypoint overlay debug panel showing keypoints overlaid on pose map."""
    panel = pose_map.copy()
    for idx, (x, y) in enumerate(keypoints_2d):
        if confidence[idx] >= 0.5:
            pt = (int(round(x)), int(round(y)))
            cv2.circle(panel, pt, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(panel, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, str(idx), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def run_visual_benchmark(output_dir: Path, num_frames: int = 60) -> Path:
    logger.info("=== Starting Phase 2.5A Controlled Multi-Region Visual Debug Benchmark ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraParameters(image_width=512, image_height=512, focal_length_px=600.0, principal_point_x=256.0, principal_point_y=256.0)
    renderer = SkeletalPoseRenderer(camera)
    builder = AppearanceConditioningBuilder(renderer, camera)
    synthesizer = BaselineArticulatedSynthesizer(output_width=512, output_height=512)

    identity_asset, ref_mannequin_img = create_multi_region_mannequin_identity()
    actor_sequence = generate_5_pose_retargeted_sequence(num_frames)

    video_path = output_dir / "debug_animation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (1536, 512))

    temporal_state = AppearanceTemporalState()

    for idx, actor_state in enumerate(actor_sequence):
        cond = builder.build_conditioning(identity_asset, actor_state, frame_index=idx, timestamp_s=actor_state.capture_timestamp)
        synth_res = synthesizer.synthesize_frame(cond, temporal_state)

        temporal_state.previous_frame_bgr = synth_res.frame_bgr
        temporal_state.frame_index = idx

        # Panel 1: Reference Mannequin Source Image & Region Masks
        panel_1 = ref_mannequin_img.copy()
        cv2.putText(panel_1, "Panel 1: Source Regions", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 2: Target Pose & Keypoint Anchors Overlay
        panel_2 = create_keypoint_overlay_panel(cond.pose_map_2d, cond.keypoints_2d, cond.joint_confidence)
        regime_names = ["Pose 0: T-Pose", "Pose 1: Left Arm Raised", "Pose 2: Right Arm Raised", "Pose 3: Torso Motion", "Pose 4: Walking Stride"]
        regime_txt = regime_names[min(idx // 12, 4)]
        cv2.putText(panel_2, f"Panel 2: {regime_txt}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)

        # Panel 3: Baseline Articulated Synthesizer Output (Transformed Regions + Z-order)
        panel_3 = synth_res.frame_bgr.copy()
        cv2.putText(panel_3, "Panel 3: Articulated Render", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel_3, f"Regions: {len(synth_res.metadata.get('transformed_regions', []))}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        composite_frame = np.hstack([panel_1, panel_2, panel_3])
        writer.write(composite_frame)

        # Save individual inspection frame directory for each of the 5 pose regimes
        if idx in (0, 12, 24, 36, 48):
            frame_dir = output_dir / f"frame_{idx:03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_dir / "reference.png"), ref_mannequin_img)
            cv2.imwrite(str(frame_dir / "pose_map.png"), cond.pose_map_2d)
            cv2.imwrite(str(frame_dir / "keypoints_overlay.png"), panel_2)
            cv2.imwrite(str(frame_dir / "synthesized.png"), synth_res.frame_bgr)

    writer.release()
    logger.info("Saved 3-panel multi-region debug video to %s", video_path)
    logger.info("Saved 5 pose regime inspection panels to %s", output_dir)
    logger.info("=== Phase 2.5A Multi-Region Visual Benchmark COMPLETE ===")

    return video_path


if __name__ == "__main__":
    out_dir = SERVICES_DIR.parent.parent / "test_data" / "outputs" / "phase2_5a_debug"
    run_visual_benchmark(out_dir, num_frames=60)
