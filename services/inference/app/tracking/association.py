import numpy as np
from typing import List, Tuple, Union
from scipy.optimize import linear_sum_assignment
from app.pipeline.result import BoundingBox

BoxLike = Union[BoundingBox, Tuple[float, float, float, float]]

def _to_coords(box: BoxLike) -> Tuple[float, float, float, float]:
    if isinstance(box, BoundingBox):
        return (float(box.x_min), float(box.y_min), float(box.x_max), float(box.y_max))
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

def bbox_iou(box_a: BoxLike, box_b: BoxLike) -> float:
    ax1, ay1, ax2, ay2 = _to_coords(box_a)
    bx1, by1, bx2, by2 = _to_coords(box_b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    ia = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - ia
    return float(ia / union) if union > 0.0 else 0.0

def build_iou_cost_matrix(
    track_boxes: List[BoxLike],
    detection_boxes: List[BoxLike]
) -> np.ndarray:
    n_tr = len(track_boxes)
    n_det = len(detection_boxes)
    if n_tr == 0 or n_det == 0:
        return np.empty((n_tr, n_det), dtype=np.float64)
    cost_matrix = np.zeros((n_tr, n_det), dtype=np.float64)
    for i, tr_box in enumerate(track_boxes):
        for j, det_box in enumerate(detection_boxes):
            cost_matrix[i, j] = 1.0 - bbox_iou(tr_box, det_box)
    return cost_matrix

def hungarian_match(
    cost_matrix: np.ndarray,
    iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    n_tr, n_det = cost_matrix.shape
    if cost_matrix.size == 0:
        return [], list(range(n_tr)), list(range(n_det))
    max_cost = 1.0 - float(iou_threshold)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, matched_r, matched_c = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_r.add(int(r))
            matched_c.add(int(c))
    unmatched_tr = [r for r in range(n_tr) if r not in matched_r]
    unmatched_det = [c for c in range(n_det) if c not in matched_c]
    return matches, unmatched_tr, unmatched_det
