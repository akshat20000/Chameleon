"""
Unit tests for app.segmentation.segmenter.

Design constraints
------------------
- No network access, no real model file required during unit tests.
- All tests that exercise MediaPipeSegmenter.segment() mock the internal
  MediaPipe ImageSegmenter so no actual .tflite file is needed.
- Verified model label mapping from metadata:
    Index 0: 'background'
    Index 1: 'hair'
    Index 2: 'body-skin'
    Index 3: 'face-skin'
    Index 4: 'clothes'
    Index 5: 'others'
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.result import SegmentationResult
from app.segmentation.segmenter import BaseSegmenter, MediaPipeSegmenter


def _make_fake_category_mask(h: int = 256, w: int = 256, fill_val: int = 0) -> MagicMock:
    """Return a mock object matching MediaPipe CategoryMask behavior."""
    arr = np.full((h, w), fill_val, dtype=np.uint8)
    mask_mock = MagicMock()
    mask_mock.numpy_view.return_value = arr
    return mask_mock


def _make_mp_segmentation_result(category_mask_mock: MagicMock) -> MagicMock:
    """Return a mock ImageSegmenter result containing category_mask."""
    res = MagicMock()
    res.category_mask = category_mask_mock
    return res


_DUMMY_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mocked_segmenter(tmp_path):
    """
    MediaPipeSegmenter whose internal _segmenter attribute is a MagicMock.
    """
    model_file = tmp_path / "selfie_multiclass_256x256.tflite"
    model_file.touch()

    with patch("mediapipe.tasks.python.vision.ImageSegmenter") as MockIS, \
         patch("mediapipe.tasks.python.vision.ImageSegmenterOptions"), \
         patch("mediapipe.tasks.python.BaseOptions"):

        mock_inner = MagicMock()
        MockIS.create_from_options.return_value = mock_inner

        segmenter = MediaPipeSegmenter(model_path=str(model_file))

    yield segmenter, mock_inner


class TestMissingAndInvalidModel:
    def test_missing_model_does_not_raise(self):
        segmenter = MediaPipeSegmenter(model_path="/nonexistent/model.tflite")
        assert segmenter.is_ready is False

    def test_missing_model_segment_returns_none(self):
        segmenter = MediaPipeSegmenter(model_path="/nonexistent/model.tflite")
        result = segmenter.segment(_DUMMY_IMAGE)
        assert result is None

    def test_invalid_model_init_failure_handled(self, tmp_path):
        model_file = tmp_path / "corrupt_model.tflite"
        model_file.touch()

        with patch("mediapipe.tasks.python.vision.ImageSegmenter.create_from_options", side_effect=RuntimeError("Corrupt TFLite")):
            segmenter = MediaPipeSegmenter(model_path=str(model_file))
            assert segmenter.is_ready is False
            assert segmenter.segment(_DUMMY_IMAGE) is None


class TestInitialization:
    def test_is_ready_true_when_model_present(self, mocked_segmenter):
        segmenter, _ = mocked_segmenter
        assert segmenter.is_ready is True

    def test_isinstance_of_base_segmenter(self, mocked_segmenter):
        segmenter, _ = mocked_segmenter
        assert isinstance(segmenter, BaseSegmenter)


class TestEmptyAndInvalidInputs:
    def test_none_image_returns_none(self, mocked_segmenter):
        segmenter, _ = mocked_segmenter
        assert segmenter.segment(None) is None

    def test_empty_image_returns_none(self, mocked_segmenter):
        segmenter, _ = mocked_segmenter
        assert segmenter.segment(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_2d_image_returns_none(self, mocked_segmenter):
        segmenter, _ = mocked_segmenter
        assert segmenter.segment(np.zeros((480, 640), dtype=np.uint8)) is None

    def test_no_mp_result_returns_none(self, mocked_segmenter):
        segmenter, mock_inner = mocked_segmenter
        mock_inner.segment.return_value = None
        with patch("mediapipe.Image"):
            assert segmenter.segment(_DUMMY_IMAGE) is None

    def test_no_category_mask_returns_none(self, mocked_segmenter):
        segmenter, mock_inner = mocked_segmenter
        res = MagicMock()
        res.category_mask = None
        mock_inner.segment.return_value = res
        with patch("mediapipe.Image"):
            assert segmenter.segment(_DUMMY_IMAGE) is None


class TestCategoryMaskAndDerivedSubmasks:
    def test_derived_submasks_and_dtypes(self, mocked_segmenter):
        segmenter, mock_inner = mocked_segmenter
        H, W = 480, 640
        image = np.zeros((H, W, 3), dtype=np.uint8)

        # Create category mask with specific regions
        # 0: background, 1: hair, 2: body-skin, 3: face-skin
        cat_arr = np.zeros((H, W), dtype=np.uint8)
        cat_arr[10:50, 10:50] = 1   # Hair
        cat_arr[50:100, 10:50] = 2  # Body skin
        cat_arr[100:150, 10:50] = 3 # Face skin

        mask_mock = MagicMock()
        mask_mock.numpy_view.return_value = cat_arr
        mock_inner.segment.return_value = _make_mp_segmentation_result(mask_mock)

        with patch("mediapipe.Image"):
            res = segmenter.segment(image)

        assert isinstance(res, SegmentationResult)
        assert res.class_mask.dtype == np.uint8
        assert res.class_mask.shape == (H, W)

        assert res.hair_mask.dtype == bool
        assert res.skin_mask.dtype == bool
        assert res.face_mask.dtype == bool

        assert np.all(res.hair_mask[10:50, 10:50])
        assert not np.any(res.hair_mask[50:100, 10:50])

        assert np.all(res.skin_mask[50:100, 10:50])
        assert not np.any(res.skin_mask[10:50, 10:50])

        assert np.all(res.face_mask[100:150, 10:50])
        assert not np.any(res.face_mask[10:50, 10:50])

    def test_resize_behavior_nearest_neighbor(self, mocked_segmenter):
        """When MP returns a 256x256 mask for a 480x640 image, resize to 480x640."""
        segmenter, mock_inner = mocked_segmenter
        H_orig, W_orig = 480, 640
        image = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)

        # 256x256 mask filled with face-skin (3)
        small_mask = np.full((256, 256), 3, dtype=np.uint8)
        mask_mock = MagicMock()
        mask_mock.numpy_view.return_value = small_mask
        mock_inner.segment.return_value = _make_mp_segmentation_result(mask_mock)

        with patch("mediapipe.Image"):
            res = segmenter.segment(image)

        assert res.class_mask.shape == (H_orig, W_orig)
        assert res.face_mask.shape == (H_orig, W_orig)
        assert np.all(res.face_mask)
