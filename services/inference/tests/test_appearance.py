"""
Unit test suite for Phase 2.5A — Appearance Conditioning Infrastructure, SkeletalPoseRenderer,
Immutability Contracts, Deterministic Baseline Synthesizer, and Metric Protocols.

Spec: docs/architecture/ADR/ADR-006-appearance-synthesis-architecture.md
"""

import sys
import uuid
import numpy as np
import pytest
from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR))

from app.appearance.conditioning_builder import (
    AppearanceConditioningBuilder,
    AppearanceConditioningState,
)
from app.appearance.pose_renderer import CameraParameters, SkeletalPoseRenderer
from app.appearance.synthesizer import (
    BaselineArticulatedSynthesizer,
    SyntheticFrameResult,
)
from app.identity.identity_asset import (
    IDENTITY_ASSET_SCHEMA_VERSION,
    PIPELINE_VERSION,
    IdentityAsset,
    SegmentedReferenceView,
    SemanticSegmentationResult,
)
from app.motion.actor_skeleton import ACTOR_PROFILES, ActorSkeleton
from app.motion.retargeted_actor_state import RetargetedActorState
from scripts.validate_appearance import (
    MetricStatus,
    compute_arcface_cossim_calibration,
    compute_nke_body,
    compute_warp_lpips_valid,
)


@pytest.fixture
def sample_actor_state() -> RetargetedActorState:
    """Fixture providing a standard T-pose RetargetedActorState."""
    skel = ACTOR_PROFILES["DEFAULT"]
    joints = list(skel.rest_positions.keys())
    positions = {}
    rotations = {}

    for j in joints:
        positions[j] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        rotations[j] = np.eye(3, dtype=np.float32)

    from app.motion.local_rotation_extractor import LocalJointRotations
    loc_rot = LocalJointRotations(joints=rotations)

    return RetargetedActorState(
        frame_index=0,
        capture_timestamp=0.0,
        actor_name="DEFAULT",
        joints=positions,
        world_rotations=rotations,
        local_rotations=loc_rot,
        motion_deltas=loc_rot,
        actor_skeleton=skel,
        source_frame_index=0,
    )


@pytest.fixture
def sample_identity_asset() -> IdentityAsset:
    """Fixture providing a minimal valid IdentityAsset."""
    vec = np.ones(512, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    view_0 = SegmentedReferenceView(
        view_index=0,
        image_path="inputs/selected_views/view_000.png",
        embedding=vec.copy(),
        segmentation=SemanticSegmentationResult(
            masks={"face": np.ones((100, 100), dtype=np.uint8) * 255},
            available_classes={"face"},
            backend_name="TestBackend",
            backend_version="1.0",
        ),
        quality_score=0.9,
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
    )

    return IdentityAsset(
        schema_version=IDENTITY_ASSET_SCHEMA_VERSION,
        identity_id=str(uuid.uuid4()),
        display_name="TestAsset",
        created_at="2026-08-21T20:00:00Z",
        pipeline_version=PIPELINE_VERSION,
        provenance={},
        fused_identity_embedding=vec.copy(),
        segmented_views=[view_0],
    )


def test_retargeted_actor_state_is_not_mutated(sample_actor_state, sample_identity_asset):
    """INVARIANT TEST: RetargetedActorState must remain completely unchanged after pose rendering & conditioning."""
    pos_before = {k: v.copy() for k, v in sample_actor_state.joints.items()}
    rot_before = {k: v.copy() for k, v in sample_actor_state.world_rotations.items()}

    renderer = SkeletalPoseRenderer()
    kpts, conf = renderer.project_joints_to_2d(sample_actor_state)
    _ = renderer.render_pose_map(kpts, conf)

    builder = AppearanceConditioningBuilder(renderer)
    _ = builder.build_conditioning(sample_identity_asset, sample_actor_state)

    # Verify positions and rotations match original arrays exactly
    for j in sample_actor_state.joints:
        np.testing.assert_array_equal(sample_actor_state.joints[j], pos_before[j])
        np.testing.assert_array_equal(sample_actor_state.world_rotations[j], rot_before[j])


def test_pose_renderer_output_shapes(sample_actor_state):
    renderer = SkeletalPoseRenderer(CameraParameters(image_width=256, image_height=256))
    kpts, conf = renderer.project_joints_to_2d(sample_actor_state)

    assert kpts.shape == (len(sample_actor_state.joints), 2)
    assert kpts.dtype == np.float32
    assert conf.shape == (len(sample_actor_state.joints),)

    pose_map = renderer.render_pose_map(kpts, conf)
    assert pose_map.shape == (256, 256, 3)
    assert pose_map.dtype == np.uint8


def test_conditioning_builder_contracts(sample_identity_asset, sample_actor_state):
    builder = AppearanceConditioningBuilder()
    cond = builder.build_conditioning(sample_identity_asset, sample_actor_state, frame_index=12, timestamp_s=0.4)

    assert cond.identity_embedding.shape == (512,)
    assert cond.identity_embedding.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(cond.identity_embedding), 1.0, atol=1e-5)
    assert len(cond.reference_views) == 1
    assert cond.pose_map_2d.shape == (512, 512, 3)
    assert cond.frame_index == 12
    assert cond.timestamp_s == 0.4
    assert "face" in cond.region_guidance


def test_missing_optional_region_guidance_does_not_fail_synthesis(sample_identity_asset, sample_actor_state):
    """
    CONTRACT TEST: Missing optional region masks (e.g. hair, clothing) must produce
    a valid AppearanceConditioningState and SyntheticFrameResult(valid=True, warnings=[...]) without crashing.
    """
    # sample_identity_asset only has 'face' mask (missing hair and clothing)
    builder = AppearanceConditioningBuilder()
    cond = builder.build_conditioning(sample_identity_asset, sample_actor_state)

    synthesizer = BaselineArticulatedSynthesizer()
    res = synthesizer.synthesize_frame(cond)

    assert res.valid
    assert res.frame_bgr.shape == (512, 512, 3)
    assert len(res.warnings) >= 2
    assert any("hair" in w for w in res.warnings)
    assert any("clothing" in w for w in res.warnings)


def test_baseline_synthesizer_deterministic_render(sample_identity_asset, sample_actor_state):
    builder = AppearanceConditioningBuilder()
    cond = builder.build_conditioning(sample_identity_asset, sample_actor_state)

    synth = BaselineArticulatedSynthesizer(output_width=256, output_height=256)
    res1 = synth.synthesize_frame(cond)
    res2 = synth.synthesize_frame(cond)

    assert res1.valid
    assert res1.frame_bgr.shape == (256, 256, 3)
    assert res1.latency_ms >= 0.0
    # Verify deterministic output
    np.testing.assert_array_equal(res1.frame_bgr, res2.frame_bgr)


def test_nke_body_degenerate_scale_handling():
    gen_kpts = np.zeros((17, 2), dtype=np.float32)
    tgt_kpts = np.zeros((17, 2), dtype=np.float32)

    # Degenerate body scale (d_body = 0.0)
    res_deg = compute_nke_body(gen_kpts, tgt_kpts)
    assert res_deg.status == MetricStatus.DEGENERATE_SCALE
    assert res_deg.value is None

    # Valid non-zero body scale
    tgt_kpts[5] = [10.0, 10.0]   # left_shoulder
    tgt_kpts[8] = [30.0, 10.0]   # right_shoulder
    tgt_kpts[11] = [10.0, 50.0]  # left_hip
    tgt_kpts[14] = [30.0, 50.0]  # right_hip

    gen_kpts[5] = [10.0, 12.0]   # 2px offset
    res_valid = compute_nke_body(gen_kpts, tgt_kpts)
    assert res_valid.status == MetricStatus.SUCCESS
    assert res_valid.value is not None
    assert res_valid.value > 0.0


def test_metric_availability_reporting():
    # WarpLPIPS returns UNAVAILABLE when dependencies are not installed or mock handles it
    curr = np.zeros((100, 100, 3), dtype=np.uint8)
    prev = np.zeros((100, 100, 3), dtype=np.uint8)

    res = compute_warp_lpips_valid(curr, prev)
    assert res.status in (MetricStatus.SUCCESS, MetricStatus.UNAVAILABLE)

    # ArcFace CosSim calibration
    e1 = np.ones(512, dtype=np.float32)
    e1 = e1 / np.linalg.norm(e1)
    e2 = e1.copy()

    res_cos = compute_arcface_cossim_calibration(e1, e2)
    assert res_cos.status == MetricStatus.SUCCESS
    assert abs(res_cos.value - 1.0) < 1e-5
