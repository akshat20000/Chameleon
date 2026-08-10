import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.tracking.association import bbox_iou, build_iou_cost_matrix, hungarian_match
from app.pipeline.result import BoundingBox

def test_bbox_iou():
    b1 = (0.0, 0.0, 10.0, 10.0)
    assert bbox_iou(b1, b1) == 1.0
    assert bbox_iou(b1, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert bbox_iou(b1, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1.0 / 3.0)
    assert bbox_iou(b1, (0.0, 0.0, 0.0, 10.0)) == 0.0
    bb1 = BoundingBox(0.0, 0.0, 10.0, 10.0)
    bb2 = BoundingBox(5.0, 0.0, 15.0, 10.0)
    assert bbox_iou(bb1, bb2) == pytest.approx(1.0 / 3.0)

def test_build_iou_cost_matrix():
    tr = [(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 20.0, 20.0)]
    det = [(0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0), (0.0, 0.0, 20.0, 20.0)]
    cm = build_iou_cost_matrix(tr, det)
    assert cm.shape == (2, 3)
    assert cm[0, 0] == pytest.approx(0.0)
    assert cm[0, 1] == pytest.approx(1.0)
    assert build_iou_cost_matrix([], det).shape == (0, 3)
    assert build_iou_cost_matrix(tr, []).shape == (2, 0)

def test_hungarian_match_basic():
    cm = np.array([[0.1]])
    m, ut, ud = hungarian_match(cm, 0.2)
    assert m == [(0, 0)] and ut == [] and ud == []
    
    cm_multi = np.array([[0.1, 0.9], [0.9, 0.1]])
    m, ut, ud = hungarian_match(cm_multi, 0.2)
    assert m == [(0, 0), (1, 1)] and ut == [] and ud == []
    
    cm_low = np.array([[0.9]])
    m, ut, ud = hungarian_match(cm_low, 0.2)
    assert m == [] and ut == [0] and ud == [0]
    
    cm_empty = np.empty((0, 2))
    m, ut, ud = hungarian_match(cm_empty, 0.2)
    assert m == [] and ut == [] and ud == [0, 1]

def test_hungarian_global_optima():
    cm = np.array([
        [0.1, 0.2],
        [0.15, 0.9]
    ])
    m, ut, ud = hungarian_match(cm, 0.0)
    assert m == [(0, 1), (1, 0)]

def test_hungarian_threshold_boundary():
    cm_exact = np.array([[0.8]])
    m, ut, ud = hungarian_match(cm_exact, 0.2)
    assert m == [(0, 0)]
    
    cm_below = np.array([[0.8001]])
    m, ut, ud = hungarian_match(cm_below, 0.2)
    assert m == [] and ut == [0] and ud == [0]
