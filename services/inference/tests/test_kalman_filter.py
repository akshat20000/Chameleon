import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.tracking.kalman_filter import KalmanFilter

def test_bbox_round_trip():
    kf = KalmanFilter()
    bbox = (10.0, 20.0, 110.0, 220.0)
    z = kf.bbox_to_z(bbox)
    np.testing.assert_allclose(bbox, kf.z_to_bbox(z), rtol=1e-6, atol=1e-6)

def test_invalid_bbox_validation():
    kf = KalmanFilter()
    boxes = [
        (100.0, 100.0, 100.0, 200.0),
        (100.0, 100.0, 50.0, 200.0),
        (100.0, 100.0, 200.0, 100.0),
        (float('nan'), 100.0, 200.0, 200.0),
        (100.0, float('inf'), 200.0, 200.0),
    ]
    for b in boxes:
        with pytest.raises(ValueError):
            kf.bbox_to_z(b)

def test_F_matrix():
    kf = KalmanFilter()
    F1 = kf.get_F(dt=1.0)
    F05 = kf.get_F(dt=0.5)
    assert F1[0, 4] == 1.0 and F1[1, 5] == 1.0
    assert F05[0, 4] == 0.5 and F05[1, 5] == 0.5

def test_Q_and_R_matrices():
    kf = KalmanFilter()
    Q1 = kf.get_Q(height=100.0, aspect_ratio=1.0)
    Q2 = kf.get_Q(height=200.0, aspect_ratio=1.0)
    assert Q2[0, 0] == 4.0 * Q1[0, 0]
    R1 = kf.get_R(height=100.0, aspect_ratio=1.0)
    R2 = kf.get_R(height=200.0, aspect_ratio=1.0)
    assert R2[0, 0] == 4.0 * R1[0, 0]

def test_initiate():
    kf = KalmanFilter()
    bbox = (10.0, 20.0, 110.0, 220.0)
    m, c = kf.initiate(bbox)
    assert m.shape == (8,) and c.shape == (8, 8)
    assert np.all(m[4:] == 0.0)
    assert np.all(np.diag(c) > 0.0)

def test_predict():
    kf = KalmanFilter()
    bbox = (10.0, 20.0, 110.0, 220.0)
    m, c = kf.initiate(bbox)
    pm, pc = kf.predict(m, c, dt=1.0)
    assert pm.shape == (8,) and pc.shape == (8, 8)
    assert np.all(np.isfinite(pm)) and np.all(np.isfinite(pc))

def test_update():
    kf = KalmanFilter()
    bbox1 = (10.0, 20.0, 110.0, 220.0)
    m, c = kf.initiate(bbox1)
    pm, pc = kf.predict(m, c, dt=1.0)
    bbox2 = (12.0, 22.0, 112.0, 222.0)
    um, uc = kf.update(pm, pc, bbox2)
    assert um.shape == (8,) and uc.shape == (8, 8)
    assert np.all(np.isfinite(um)) and np.all(np.isfinite(uc))

def test_repeated_predict_update_stability():
    kf = KalmanFilter()
    bbox = (100.0, 100.0, 200.0, 200.0)
    m, c = kf.initiate(bbox)
    for i in range(100):
        m, c = kf.predict(m, c, dt=1.0)
        meas = (100.0 + i, 100.0 + i, 200.0 + i, 200.0 + i)
        m, c = kf.update(m, c, meas)
        assert np.all(np.isfinite(m)), f"Non-finite mean at step {i}"
        assert np.all(np.isfinite(c)), f"Non-finite cov at step {i}"
        assert np.all(np.diag(c) > 0), f"Non-positive variance at step {i}"
