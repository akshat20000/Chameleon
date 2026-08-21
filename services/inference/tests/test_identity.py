"""
Unit tests for app.identity.encoder.

Design constraints
------------------
- No network access, no real model file required for unit tests.
- All tests that exercise ONNXIdentityEncoder mock onnxruntime.InferenceSession.
- Pure functions (normalize_embedding, fuse_embeddings, align_face_5pt, extract_5pt_landmarks_from_478)
  are tested directly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.identity.encoder import (
    MEDIAPIPE_5PT_INDICES,
    BaseIdentityEncoder,
    ONNXIdentityEncoder,
    align_face_5pt,
    extract_5pt_landmarks_from_478,
    fuse_embeddings,
    normalize_embedding,
)


class TestPureNormalization:
    def test_valid_vector_normalized_to_unit_length(self):
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        norm_vec = normalize_embedding(vec)
        assert norm_vec is not None
        assert norm_vec.dtype == np.float32
        assert norm_vec.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(norm_vec), 1.0, atol=1e-6)
        np.testing.assert_allclose(norm_vec, [0.6, 0.8, 0.0], atol=1e-6)

    def test_zero_vector_returns_none(self):
        zero_vec = np.zeros(512, dtype=np.float32)
        res = normalize_embedding(zero_vec)
        assert res is None

    def test_nan_inf_returns_none(self):
        nan_vec = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        inf_vec = np.array([1.0, np.inf, 2.0], dtype=np.float32)
        assert normalize_embedding(nan_vec) is None
        assert normalize_embedding(inf_vec) is None

    def test_none_or_empty_returns_none(self):
        assert normalize_embedding(None) is None
        assert normalize_embedding(np.array([], dtype=np.float32)) is None


class TestPureFusion:
    def test_fusion_single_embedding(self):
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fused = fuse_embeddings([v1])
        expected = normalize_embedding(v1)
        assert fused is not None
        np.testing.assert_allclose(fused, expected, atol=1e-6)

    def test_fusion_multiple_embeddings(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        fused = fuse_embeddings([v1, v2])
        assert fused is not None
        np.testing.assert_allclose(np.linalg.norm(fused), 1.0, atol=1e-6)
        np.testing.assert_allclose(fused, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], atol=1e-6)

    def test_fusion_identical_embeddings(self):
        v1 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        v2 = np.array([6.0, 8.0, 0.0], dtype=np.float32)
        fused = fuse_embeddings([v1, v2])
        np.testing.assert_allclose(fused, [0.6, 0.8, 0.0], atol=1e-6)

    def test_fusion_dimension_mismatch_returns_none(self):
        v1 = np.ones(512, dtype=np.float32)
        v2 = np.ones(128, dtype=np.float32)
        assert fuse_embeddings([v1, v2]) is None

    def test_fusion_empty_or_invalid_returns_none(self):
        assert fuse_embeddings([]) is None
        assert fuse_embeddings([np.zeros(512)]) is None

    def test_fusion_with_quality_weights(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # v1 weight 0.9, v2 weight 0.1
        fused = fuse_embeddings([v1, v2], weights=[0.9, 0.1])
        assert fused is not None
        np.testing.assert_allclose(np.linalg.norm(fused), 1.0, atol=1e-6)
        # Fused vector should be predominantly pointing along +X
        assert fused[0] > 0.95
        assert fused[1] < 0.2

    def test_fusion_omits_sub_epsilon_weights(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # v2 weight 0.0001 < min_quality_weight (1e-3)
        fused = fuse_embeddings([v1, v2], weights=[0.9, 0.0001], min_quality_weight=1e-3)
        assert fused is not None
        np.testing.assert_allclose(fused, [1.0, 0.0, 0.0], atol=1e-6)


class TestPureAlignment:
    def test_extract_5pt_landmarks_indices(self):
        pts2d = np.zeros((478, 2), dtype=np.float32)
        pts2d[468] = [10.0, 20.0]
        pts2d[473] = [30.0, 40.0]
        pts2d[1] = [50.0, 60.0]
        pts2d[61] = [70.0, 80.0]
        pts2d[291] = [90.0, 100.0]

        res5 = extract_5pt_landmarks_from_478(pts2d)
        assert res5 is not None
        assert res5.shape == (5, 2)
        np.testing.assert_allclose(res5[0], [10.0, 20.0])
        np.testing.assert_allclose(res5[1], [30.0, 40.0])
        np.testing.assert_allclose(res5[2], [50.0, 60.0])
        np.testing.assert_allclose(res5[3], [70.0, 80.0])
        np.testing.assert_allclose(res5[4], [90.0, 100.0])

    def test_align_face_5pt_shape_and_dtype(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks_5pt = np.array([
            [200.0, 200.0],
            [400.0, 200.0],
            [300.0, 300.0],
            [250.0, 400.0],
            [350.0, 400.0],
        ], dtype=np.float32)

        aligned = align_face_5pt(image, landmarks_5pt, target_size=(112, 112))
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)
        assert aligned.dtype == np.uint8

    def test_align_face_invalid_inputs_return_none(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        assert align_face_5pt(None, np.zeros((5, 2))) is None
        assert align_face_5pt(image, np.zeros((4, 2))) is None
        assert align_face_5pt(image, np.full((5, 2), np.nan)) is None


class TestONNXIdentityEncoderMocked:
    @pytest.fixture
    def mocked_encoder(self, tmp_path):
        model_file = tmp_path / "w600k_mbf.onnx"
        model_file.touch()

        with patch("onnxruntime.InferenceSession") as MockSession:
            session_instance = MagicMock()
            inp_mock = MagicMock()
            inp_mock.name = "input.1"
            out_mock = MagicMock()
            out_mock.name = "516"
            out_mock.shape = [1, 512]

            session_instance.get_inputs.return_value = [inp_mock]
            session_instance.get_outputs.return_value = [out_mock]

            # Raw mock output shape (1, 512)
            session_instance.run.return_value = [np.ones((1, 512), dtype=np.float32)]

            MockSession.return_value = session_instance

            encoder = ONNXIdentityEncoder(model_path=str(model_file))

        yield encoder, session_instance

    def test_missing_model_does_not_raise(self):
        enc = ONNXIdentityEncoder(model_path="/nonexistent/model.onnx")
        assert enc.is_ready is False
        assert enc.extract_embedding(np.zeros((112, 112, 3), dtype=np.uint8)) is None

    def test_invalid_model_init_failure_handled(self, tmp_path):
        model_file = tmp_path / "corrupt.onnx"
        model_file.touch()

        with patch("onnxruntime.InferenceSession", side_effect=RuntimeError("Corrupt ONNX")):
            enc = ONNXIdentityEncoder(model_path=str(model_file))
            assert enc.is_ready is False

    def test_readiness_and_properties(self, mocked_encoder):
        enc, _ = mocked_encoder
        assert enc.is_ready is True
        assert enc.embedding_dim == 512
        assert isinstance(enc, BaseIdentityEncoder)

    def test_extract_embedding_shape_dtype_norm(self, mocked_encoder):
        enc, _ = mocked_encoder
        face_chip = np.zeros((112, 112, 3), dtype=np.uint8)

        emb = enc.extract_embedding(face_chip)
        assert emb is not None
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-6)

    def test_invalid_face_chip_returns_none(self, mocked_encoder):
        enc, _ = mocked_encoder
        assert enc.extract_embedding(None) is None
        assert enc.extract_embedding(np.zeros((0, 0, 3), dtype=np.uint8)) is None
