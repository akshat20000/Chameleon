"""
Unit tests for app.landmarks.landmarker.

Design constraints
------------------
- No network access, no model download.
- All tests that exercise MediaPipeLandmarker.detect() mock the internal
  MediaPipe landmarker so that no .task file is required.
- Tests of _match_faces_to_tracks and coordinate conversion are pure and
  require no mocking.
- A missing model file causes is_ready == False; detect() then returns {}.
"""

import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.landmarks.landmarker import BaseLandmarker, MediaPipeLandmarker
from app.pipeline.result import BoundingBox, FaceDetection, LandmarkResult, PoseResult, TrackedFace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackedFace:
    return TrackedFace(
        track_id=track_id,
        bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
        confidence=0.9,
    )


def _make_fake_landmark(x: float, y: float, z: float = 0.0) -> MagicMock:
    """Return a mock that behaves like mediapipe NormalizedLandmark."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_fake_face_landmarks(
    n: int = 478,
    x_norm: float = 0.5,
    y_norm: float = 0.5,
    z: float = 0.0,
) -> List[MagicMock]:
    """Return n fake NormalizedLandmarks, all at (x_norm, y_norm, z)."""
    return [_make_fake_landmark(x_norm, y_norm, z) for _ in range(n)]


def _make_mp_result(face_lm_lists: List[List[MagicMock]]) -> MagicMock:
    """Return a mock FaceLandmarkerResult with the supplied face lists."""
    res = MagicMock()
    res.face_landmarks = face_lm_lists
    return res


_DUMMY_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixture: landmarker backed by a mocked MediaPipe (no real model needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def mocked_landmarker(tmp_path):
    """
    MediaPipeLandmarker whose internal _landmarker attribute is a MagicMock.

    Yields (landmarker, mock_inner) so tests can configure mock_inner.detect
    return values freely.
    """
    model_file = tmp_path / "face_landmarker.task"
    model_file.touch()  # create an empty file so os.path.exists passes

    with patch("mediapipe.tasks.python.vision.FaceLandmarker") as MockFL, \
         patch("mediapipe.tasks.python.vision.FaceLandmarkerOptions"), \
         patch("mediapipe.tasks.python.BaseOptions"):

        mock_inner = MagicMock()
        MockFL.create_from_options.return_value = mock_inner

        lm = MediaPipeLandmarker(model_path=str(model_file))

    # After context exit the patches are removed, but lm._landmarker is already set
    yield lm, mock_inner


# ---------------------------------------------------------------------------
# 1. Missing model
# ---------------------------------------------------------------------------

class TestMissingModel:
    def test_missing_model_does_not_raise(self):
        """Constructor must not raise when model file is absent."""
        lm = MediaPipeLandmarker(model_path="/nonexistent/face_landmarker.task")
        assert lm.is_ready is False

    def test_missing_model_detect_returns_empty(self):
        """detect() must return {} when model is absent, even with valid tracks."""
        lm = MediaPipeLandmarker(model_path="/nonexistent/face_landmarker.task")
        tracks = [_make_track(1, 10, 10, 50, 50)]
        result = lm.detect(_DUMMY_IMAGE, tracks)
        assert result == {}


# ---------------------------------------------------------------------------
# 2. Initialization / is_ready
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_is_ready_false_when_model_absent(self):
        lm = MediaPipeLandmarker(model_path="/does/not/exist.task")
        assert lm.is_ready is False

    def test_is_ready_true_when_model_present(self, mocked_landmarker):
        lm, _ = mocked_landmarker
        assert lm.is_ready is True

    def test_isinstance_of_base(self, mocked_landmarker):
        lm, _ = mocked_landmarker
        assert isinstance(lm, BaseLandmarker)


# ---------------------------------------------------------------------------
# 3. Empty / degenerate inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_empty_tracks_returns_empty(self, mocked_landmarker):
        lm, _ = mocked_landmarker
        result = lm.detect(_DUMMY_IMAGE, [])
        assert result == {}

    def test_none_image_returns_empty(self, mocked_landmarker):
        lm, _ = mocked_landmarker
        tracks = [_make_track(1, 10, 10, 50, 50)]
        result = lm.detect(None, tracks)
        assert result == {}

    def test_empty_image_returns_empty(self, mocked_landmarker):
        lm, _ = mocked_landmarker
        tracks = [_make_track(1, 10, 10, 50, 50)]
        result = lm.detect(np.zeros((0, 0, 3), dtype=np.uint8), tracks)
        assert result == {}

    def test_no_mp_detections_returns_empty(self, mocked_landmarker):
        """When MediaPipe detects no faces, result must be {}."""
        lm, mock_inner = mocked_landmarker
        mock_inner.detect.return_value = _make_mp_result([])

        tracks = [_make_track(1, 10, 10, 50, 50)]
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, tracks)

        assert result == {}


# ---------------------------------------------------------------------------
# 4. Coordinate conversion — 2D
# ---------------------------------------------------------------------------

class TestCoordinateConversion2D:
    def test_2d_pixel_conversion_x_y(self, mocked_landmarker):
        """
        x_pixel = landmark.x * W  and  y_pixel = landmark.y * H.
        """
        lm, mock_inner = mocked_landmarker
        W, H = 640, 480
        image = np.zeros((H, W, 3), dtype=np.uint8)

        x_norm, y_norm = 0.5, 0.25   # expected pixels: x=320, y=120
        face_lms = _make_fake_face_landmarks(n=478, x_norm=x_norm, y_norm=y_norm, z=0.0)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, W, H)  # bbox covers full image → will match
        with patch("mediapipe.Image"):
            result = lm.detect(image, [track])

        assert 1 in result
        pts2d = result[1].points_2d
        np.testing.assert_allclose(pts2d[:, 0], x_norm * W, rtol=1e-5)
        np.testing.assert_allclose(pts2d[:, 1], y_norm * H, rtol=1e-5)

    def test_2d_shape_is_478x2(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].points_2d.shape == (478, 2)

    def test_2d_dtype_is_float32(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].points_2d.dtype == np.float32


# ---------------------------------------------------------------------------
# 5. Coordinate conversion — 3D
# ---------------------------------------------------------------------------

class TestCoordinateConversion3D:
    def test_3d_shape_is_478x3(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].points_3d.shape == (478, 3)

    def test_3d_xy_matches_2d(self, mocked_landmarker):
        """points_3d[:, :2] must equal points_2d exactly."""
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478, x_norm=0.3, y_norm=0.7, z=-0.05)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        np.testing.assert_array_equal(result[1].points_3d[:, :2], result[1].points_2d)

    def test_3d_z_is_mediapipe_depth(self, mocked_landmarker):
        """points_3d[:, 2] must carry the raw MediaPipe z value."""
        lm, mock_inner = mocked_landmarker
        z_val = -0.042
        face_lms = _make_fake_face_landmarks(n=478, x_norm=0.5, y_norm=0.5, z=z_val)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        np.testing.assert_allclose(result[1].points_3d[:, 2], z_val, rtol=1e-5)

    def test_3d_dtype_is_float32(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].points_3d.dtype == np.float32


# ---------------------------------------------------------------------------
# 6. Track association
# ---------------------------------------------------------------------------

class TestTrackAssociation:
    def test_single_face_associates_correct_track_id(self, mocked_landmarker):
        """MP face centroid inside track bbox → assigned to that track's ID."""
        lm, mock_inner = mocked_landmarker
        # Landmark centroid at (160, 120) which is inside the track bbox
        face_lms = _make_fake_face_landmarks(n=478, x_norm=0.25, y_norm=0.25)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(42, 100, 80, 220, 180)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert 42 in result

    def test_two_faces_two_tracks_correct_ids(self, mocked_landmarker):
        """Each MP face should map to the spatially nearest track."""
        lm, mock_inner = mocked_landmarker
        W, H = 640, 480
        image = np.zeros((H, W, 3), dtype=np.uint8)

        # Face 0: landmarks centred at (0.15, 0.15) → pixel (96, 72)
        face0_lms = _make_fake_face_landmarks(n=478, x_norm=0.15, y_norm=0.15)
        # Face 1: landmarks centred at (0.75, 0.75) → pixel (480, 360)
        face1_lms = _make_fake_face_landmarks(n=478, x_norm=0.75, y_norm=0.75)
        mock_inner.detect.return_value = _make_mp_result([face0_lms, face1_lms])

        track_a = _make_track(10, 50, 30, 150, 120)   # left-upper
        track_b = _make_track(20, 400, 300, 550, 430)  # right-lower

        with patch("mediapipe.Image"):
            result = lm.detect(image, [track_a, track_b])

        assert set(result.keys()) == {10, 20}

    def test_result_keyed_by_track_id(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478, x_norm=0.5, y_norm=0.5)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(99, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert 99 in result
        assert isinstance(result[99], LandmarkResult)

    def test_landmarks_type_is_478_pt(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].landmarks_type == "478_pt"

    def test_confidence_is_1_0(self, mocked_landmarker):
        """FaceLandmarker provides no per-face score; confidence must be 1.0."""
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].confidence == 1.0


# ---------------------------------------------------------------------------
# 7. One-to-one matching (pure function tests on _match_faces_to_tracks)
# ---------------------------------------------------------------------------

class TestMatchFacesToTracks:
    def test_returns_empty_when_no_lm_boxes(self):
        result = MediaPipeLandmarker._match_faces_to_tracks([], [], [(0, 0, 50, 50)])
        assert result == {}

    def test_returns_empty_when_no_track_boxes(self):
        result = MediaPipeLandmarker._match_faces_to_tracks(
            [(0, 0, 50, 50)], [(25, 25)], []
        )
        assert result == {}

    def test_one_face_one_track_iou_overlap(self):
        lm_boxes = [(10, 10, 50, 50)]
        lm_centroids = [(30, 30)]
        track_boxes = [(15, 15, 45, 45)]  # overlaps with lm_box
        result = MediaPipeLandmarker._match_faces_to_tracks(lm_boxes, lm_centroids, track_boxes)
        assert result == {0: 0}

    def test_centroid_fallback_when_no_iou_overlap(self):
        # lm_box and track_box do not overlap
        lm_boxes = [(200, 200, 220, 220)]
        lm_centroids = [(210, 210)]
        track_boxes = [(0, 0, 50, 50)]   # no overlap; centroid fallback should match
        result = MediaPipeLandmarker._match_faces_to_tracks(lm_boxes, lm_centroids, track_boxes)
        assert result == {0: 0}

    def test_two_faces_two_tracks_one_to_one(self):
        """Each MP face gets a distinct track; no track is claimed twice."""
        lm_boxes = [(10, 10, 50, 50), (200, 200, 250, 250)]
        lm_centroids = [(30, 30), (225, 225)]
        track_boxes = [(15, 15, 45, 45), (205, 205, 245, 245)]
        result = MediaPipeLandmarker._match_faces_to_tracks(lm_boxes, lm_centroids, track_boxes)

        assert len(result) == 2
        # Values (track indices) must be distinct
        assert len(set(result.values())) == 2

    def test_more_mp_faces_than_tracks_caps_at_track_count(self):
        """If there are more MP faces than tracks, at most len(tracks) get assigned."""
        lm_boxes = [(0, 0, 50, 50), (100, 0, 150, 50), (200, 0, 250, 50)]
        lm_centroids = [(25, 25), (125, 25), (225, 25)]
        track_boxes = [(5, 5, 45, 45)]  # only one track
        result = MediaPipeLandmarker._match_faces_to_tracks(lm_boxes, lm_centroids, track_boxes)

        assert len(result) == 1
        assert len(set(result.values())) == 1  # only one track was available

    def test_more_tracks_than_mp_faces_leaves_tracks_unmatched(self):
        """Surplus tracks have no corresponding MP face and should not appear."""
        lm_boxes = [(10, 10, 50, 50)]
        lm_centroids = [(30, 30)]
        track_boxes = [(15, 15, 45, 45), (300, 300, 400, 400)]
        result = MediaPipeLandmarker._match_faces_to_tracks(lm_boxes, lm_centroids, track_boxes)

        assert len(result) == 1  # only one MP face was available


# ---------------------------------------------------------------------------
# 8. 478-landmark count validation
# ---------------------------------------------------------------------------

class Test478Landmarks:
    def test_points_2d_has_exactly_478_rows(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].points_2d.shape[0] == 478

    def test_points_3d_has_exactly_478_rows(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        mock_inner.detect.return_value = _make_mp_result([face_lms])

        track = _make_track(1, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            result = lm.detect(_DUMMY_IMAGE, [track])

        assert result[1].points_3d.shape[0] == 478


# ---------------------------------------------------------------------------
# 9. Euler angle conversion & Gimbal lock math tests
# ---------------------------------------------------------------------------

class TestEulerAngleConversion:
    def test_identity_matrix_returns_zeros(self):
        R = np.eye(3, dtype=np.float32)
        pitch, yaw, roll = MediaPipeLandmarker.matrix_to_euler_zyx(R)
        np.testing.assert_allclose([pitch, yaw, roll], [0.0, 0.0, 0.0], atol=1e-4)

    def test_pure_pitch_30_degrees(self):
        rad = np.radians(30.0)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c]
        ], dtype=np.float32)
        pitch, yaw, roll = MediaPipeLandmarker.matrix_to_euler_zyx(R)
        np.testing.assert_allclose([pitch, yaw, roll], [30.0, 0.0, 0.0], atol=1e-4)

    def test_pure_yaw_30_degrees(self):
        rad = np.radians(30.0)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ], dtype=np.float32)
        pitch, yaw, roll = MediaPipeLandmarker.matrix_to_euler_zyx(R)
        np.testing.assert_allclose([pitch, yaw, roll], [0.0, 30.0, 0.0], atol=1e-4)

    def test_pure_roll_30_degrees(self):
        rad = np.radians(30.0)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        pitch, yaw, roll = MediaPipeLandmarker.matrix_to_euler_zyx(R)
        np.testing.assert_allclose([pitch, yaw, roll], [0.0, 0.0, 30.0], atol=1e-4)

    def test_gimbal_lock_positive_90_yaw(self):
        # R20 = -1.0 -> yaw = +90 deg
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ], dtype=np.float32)
        pitch, yaw, roll = MediaPipeLandmarker.matrix_to_euler_zyx(R)
        assert np.isfinite(pitch) and np.isfinite(yaw) and np.isfinite(roll)
        np.testing.assert_allclose(yaw, 90.0, atol=1e-4)
        assert roll == 0.0

    def test_gimbal_lock_negative_90_yaw(self):
        # R20 = 1.0 -> yaw = -90 deg
        R = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)
        pitch, yaw, roll = MediaPipeLandmarker.matrix_to_euler_zyx(R)
        assert np.isfinite(pitch) and np.isfinite(yaw) and np.isfinite(roll)
        np.testing.assert_allclose(yaw, -90.0, atol=1e-4)
        assert roll == 0.0


# ---------------------------------------------------------------------------
# 10. Pose extraction & Blendshapes tests
# ---------------------------------------------------------------------------

def _make_fake_blendshapes(n: int = 52) -> List[MagicMock]:
    categories = [
        '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
        'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft',
        'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
        'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
        'eyeSquintRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
        'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
        'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
        'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
        'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
        'noseSneerLeft', 'noseSneerRight', 'tongueOut', 'eyeLookUpLeft_2', 'eyeLookUpRight_2',
        'extra_1', 'extra_2'
    ]
    res = []
    for i in range(n):
        b = MagicMock()
        b.category_name = categories[i] if i < len(categories) else f"cat_{i}"
        b.score = float(0.01 * (i + 1))
        res.append(b)
    return res


def _make_fake_matrix_4x4() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


class TestPoseExtraction:
    def test_detect_landmarks_and_pose_success(self, mocked_landmarker):
        lm, mock_inner = mocked_landmarker
        face_lms = _make_fake_face_landmarks(n=478)
        bs_list = _make_fake_blendshapes(n=52)
        mat_4x4 = _make_fake_matrix_4x4()

        res_mock = MagicMock()
        res_mock.face_landmarks = [face_lms]
        res_mock.face_blendshapes = [bs_list]
        res_mock.facial_transformation_matrixes = [mat_4x4]

        mock_inner.detect.return_value = res_mock

        track = _make_track(7, 0, 0, 640, 480)
        with patch("mediapipe.Image"):
            lms_dict, pose_dict = lm.detect_landmarks_and_pose(_DUMMY_IMAGE, [track])

        assert 7 in lms_dict
        assert 7 in pose_dict

        pose = pose_dict[7]
        assert isinstance(pose, PoseResult)
        assert len(pose.blendshapes) == 52
        assert pose.transformation_matrix.shape == (4, 4)
        assert pose.transformation_matrix.dtype == np.float32

        assert np.isfinite(pose.pitch)
        assert np.isfinite(pose.yaw)
        assert np.isfinite(pose.roll)
        assert all(np.isfinite(val) for val in pose.blendshapes.values())
        assert all(np.isfinite(val) for val in pose.transformation_matrix.flatten())

    def test_detect_landmarks_and_pose_empty_tracks(self, mocked_landmarker):
        lm, _ = mocked_landmarker
        lms_dict, pose_dict = lm.detect_landmarks_and_pose(_DUMMY_IMAGE, [])
        assert lms_dict == {}
        assert pose_dict == {}

    def test_detect_landmarks_and_pose_unready(self):
        lm = MediaPipeLandmarker(model_path="/nonexistent/model.task")
        lms_dict, pose_dict = lm.detect_landmarks_and_pose(_DUMMY_IMAGE, [_make_track(1, 0, 0, 10, 10)])
        assert lms_dict == {}
        assert pose_dict == {}

