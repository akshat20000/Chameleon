import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from app.pipeline.result import BoundingBox, FaceDetection, TrackedFace
from app.tracking.kalman_filter import KalmanFilter
from app.tracking.association import build_iou_cost_matrix, hungarian_match

@dataclass
class _SingleTrack:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_confidence: float = 0.0

    @property
    def is_observed(self) -> bool:
        return self.time_since_update == 0

class KalmanFilterTracker:
    def __init__(
        self,
        min_iou_threshold: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.min_iou_threshold = float(min_iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.kf = KalmanFilter()
        self._next_id: int = 1
        self.tracks: List[_SingleTrack] = []

    def reset(self) -> None:
        self._next_id = 1
        self.tracks.clear()

    @staticmethod
    def _to_bbox_tuple(bbox: BoundingBox) -> Optional[Tuple[float, float, float, float]]:
        b = (float(bbox.x_min), float(bbox.y_min), float(bbox.x_max), float(bbox.y_max))
        if not (np.isfinite(b[0]) and np.isfinite(b[1]) and np.isfinite(b[2]) and np.isfinite(b[3])):
            return None
        if b[2] <= b[0] or b[3] <= b[1]:
            return None
        return b

    def _predict_tracks(self, dt: float) -> None:
        for t in self.tracks:
            if t.hits < self.min_hits:
                continue
            t.mean, t.covariance = self.kf.predict(t.mean, t.covariance, dt=dt)
            t.last_bbox = self.kf.z_to_bbox(t.mean)

    def _update_matched_tracks(
        self,
        matches: List[Tuple[int, int]],
        detections: List[FaceDetection],
        det_boxes: List[Tuple[float, float, float, float]]
    ) -> None:
        for tr_idx, det_idx in matches:
            t = self.tracks[tr_idx]
            det = detections[det_idx]
            box_tuple = det_boxes[det_idx]
            t.mean, t.covariance = self.kf.update(t.mean, t.covariance, box_tuple)
            t.last_bbox = box_tuple
            t.last_confidence = float(det.confidence)
            t.hits += 1
            t.time_since_update = 0

    def _age_unmatched_tracks(self, unmatched_tr_indices: List[int]) -> None:
        for tr_idx in unmatched_tr_indices:
            t = self.tracks[tr_idx]
            t.time_since_update += 1

    def _create_tracks(
        self,
        unmatched_det_indices: List[int],
        detections: List[FaceDetection],
        det_boxes: List[Tuple[float, float, float, float]]
    ) -> None:
        for det_idx in unmatched_det_indices:
            box = det_boxes[det_idx]
            if box is None:
                continue
            m, c = self.kf.initiate(box)
            t = _SingleTrack(
                track_id=self._next_id,
                mean=m, covariance=c, age=1, hits=1,
                time_since_update=0, last_bbox=box,
                last_confidence=float(detections[det_idx].confidence)
            )
            self._next_id += 1
            self.tracks.append(t)

    def _remove_dead_tracks(self) -> None:
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def _to_tracked_faces(self) -> List[TrackedFace]:
        res = []
        for t in self.tracks:
            if t.hits < self.min_hits:
                continue
            b = t.last_bbox
            res.append(TrackedFace(
                track_id=t.track_id,
                bbox=BoundingBox(b[0], b[1], b[2], b[3]),
                confidence=t.last_confidence,
                age=t.age, hits=t.hits,
                time_since_update=t.time_since_update,
                is_observed=t.is_observed,
            ))
        return res

    def update(
        self,
        detections: List[FaceDetection],
        dt: float = 1.0
    ) -> List[TrackedFace]:
        for t in self.tracks:
            if t.hits < self.min_hits:
                continue
            t.age += 1
        self._predict_tracks(dt)

        v_indices, v_boxes, box_map = [], [], {}
        for i, det in enumerate(detections):
            b = self._to_bbox_tuple(det.bbox)
            if b is not None:
                v_indices.append(i)
                v_boxes.append(b)
                box_map[i] = b

        tr_boxes = [t.last_bbox for t in self.tracks]
        cost_matrix = build_iou_cost_matrix(tr_boxes, v_boxes)
        matches, un_tr, un_det_v = hungarian_match(cost_matrix, self.min_iou_threshold)

        real_matches = [(tr_i, v_indices[vi]) for tr_i, vi in matches]
        real_un_det = [v_indices[vi] for vi in un_det_v]

        self._update_matched_tracks(real_matches, detections, box_map)
        self._age_unmatched_tracks(un_tr)
        self._create_tracks(real_un_det, detections, box_map)
        self._remove_dead_tracks()
        return self._to_tracked_faces()

