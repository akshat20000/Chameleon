import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.result import BoundingBox, FaceDetection
from app.tracking.tracker import KalmanFilterTracker

def test_first_detection_tentative():
    tr = KalmanFilterTracker(min_hits=3)
    d = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    assert tr.update([d]) == []
    assert tr.update([d]) == []

def test_track_confirmed_at_min_hits():
    tr = KalmanFilterTracker(min_hits=3)
    d = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    tr.update([d])
    tr.update([d])
    res = tr.update([d])
    assert len(res) == 1 and res[0].track_id == 1 and res[0].hits == 3

def test_track_id_persists():
    tr = KalmanFilterTracker(min_hits=1)
    d = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    for i in range(1, 6):
        res = tr.update([d])
        assert len(res) == 1 and res[0].track_id == 1 and res[0].hits == i

def test_two_faces_distinct_ids():
    tr = KalmanFilterTracker(min_hits=1)
    d1 = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    d2 = FaceDetection(BoundingBox(150.0, 150.0, 200.0, 200.0), 0.85)
    res = tr.update([d1, d2])
    assert len(res) == 2 and {res[0].track_id, res[1].track_id} == {1, 2}

def test_face_movement_preserves_identity():
    tr = KalmanFilterTracker(min_hits=1)
    for i in range(5):
        d = FaceDetection(BoundingBox(10.0 + i * 2, 10.0, 50.0 + i * 2, 50.0), 0.9)
        res = tr.update([d])
        assert len(res) == 1 and res[0].track_id == 1

def test_low_iou_creates_new_track():
    tr = KalmanFilterTracker(min_hits=1)
    d1 = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    tr.update([d1])
    d2 = FaceDetection(BoundingBox(300.0, 300.0, 350.0, 350.0), 0.9)
    res = tr.update([d2])
    ids = {t.track_id for t in res}
    assert 2 in ids

def test_missing_detection_ages_track():
    tr = KalmanFilterTracker(min_hits=1)
    d = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    tr.update([d])
    res = tr.update([])
    assert len(res) == 1
    assert res[0].time_since_update == 1
    assert res[0].is_observed is False

def test_max_age_removes_track():
    tr = KalmanFilterTracker(min_hits=1, max_age=2)
    d = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    tr.update([d])
    assert len(tr.update([])) == 1
    assert len(tr.update([])) == 1
    assert len(tr.update([])) == 0

def test_monotonic_track_ids():
    tr = KalmanFilterTracker(min_hits=1, max_age=1)
    d1 = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    tr.update([d1])
    d2 = FaceDetection(BoundingBox(200.0, 200.0, 250.0, 250.0), 0.9)
    tr.update([d2])
    tr.update([])
    tr.update([])
    d3 = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    res = tr.update([d3])
    assert res[0].track_id > 2

def test_reset():
    tr = KalmanFilterTracker(min_hits=1)
    d = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    tr.update([d])
    tr.reset()
    assert len(tr.tracks) == 0
    res = tr.update([d])
    assert len(res) == 1 and res[0].track_id == 1

def test_invalid_detection_does_not_crash():
    tr = KalmanFilterTracker(min_hits=1)
    bad_dets = [
        FaceDetection(BoundingBox(float('nan'), 10.0, 50.0, 50.0), 0.9),
        FaceDetection(BoundingBox(10.0, 10.0, 10.0, 50.0), 0.9),
        FaceDetection(BoundingBox(50.0, 10.0, 10.0, 50.0), 0.9),
    ]
    res = tr.update(bad_dets)
    assert res == []

def test_empty_detection_sequence():
    tr = KalmanFilterTracker(min_hits=1)
    for _ in range(10):
        assert tr.update([]) == []

def test_long_sequence_numerical_stability():
    tr = KalmanFilterTracker(min_hits=1)
    d1 = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    d2 = FaceDetection(BoundingBox(150.0, 150.0, 200.0, 200.0), 0.85)
    for i in range(120):
        res = tr.update([d1, d2])
        for t in res:
            b = t.bbox
            assert np.isfinite([b.x_min, b.y_min, b.x_max, b.y_max]).all()
            assert np.isfinite(t.confidence)
