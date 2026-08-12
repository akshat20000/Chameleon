"""
Face landmark detection using MediaPipe Tasks API (mediapipe 1.0.0).

Coordinate semantics
--------------------
points_2d : np.ndarray, shape (478, 2), dtype float32
    [x_pixel, y_pixel] in image pixel coordinates.
    Conversion: x_pixel = landmark.x * image_width
                y_pixel = landmark.y * image_height

points_3d : np.ndarray, shape (478, 3), dtype float32
    [x_pixel, y_pixel, z]
    x_pixel and y_pixel are the same as points_2d.
    z is MediaPipe's per-landmark depth value in canonical face space —
    NOT metric world-space depth.  z = 0 is roughly the nose tip;
    negative z means closer to the camera than the nose tip.
    This value is returned directly from NormalizedLandmark.z without
    further transformation.

Confidence
----------
MediaPipe FaceLandmarker does not expose a per-face detection confidence
in its result object.  LandmarkResult.confidence is always set to 1.0
as a documented, intentional fallback consistent with the existing
LandmarkResult dataclass default.

Track association
-----------------
FaceLandmarker returns faces in an ordered list with no track_id.
Association is performed in two stages:
  1. IoU matching (Hungarian algorithm, threshold > 0).
     The bounding box of each MediaPipe face is derived from the min/max
     pixel coordinates of its 478 landmarks.
  2. Centroid-distance fallback for any MediaPipe face that had zero IoU
     with every remaining unmatched track.
The result is always one-to-one (a track claims at most one MP face).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config.settings import get_settings
from app.pipeline.result import BoundingBox, LandmarkResult, TrackedFace
from app.tracking.association import build_iou_cost_matrix, hungarian_match


class BaseLandmarker(ABC):
    """Abstract base class for face landmark detectors."""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        tracks: List[TrackedFace],
    ) -> Dict[int, LandmarkResult]:
        """
        Run face landmark detection on the full image and associate results
        with the provided tracked faces.

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.
        tracks : List[TrackedFace]
            Confirmed tracked faces from the tracker.  The returned dict
            is keyed by TrackedFace.track_id.

        Returns
        -------
        Dict[int, LandmarkResult]
            Mapping track_id -> LandmarkResult.
            Returns an empty dict when no landmarks can be detected,
            no tracks are provided, or the component is not ready.
        """


class MediaPipeLandmarker(BaseLandmarker):
    """
    Face landmarker backed by MediaPipe Tasks API (mediapipe 1.0.0).

    Runs a single FaceLandmarker call on the full frame with
    num_faces = settings.max_faces.  Results are then associated with
    caller-supplied track IDs via IoU + centroid-distance matching.

    Parameters
    ----------
    model_path : str, optional
        Filesystem path to the face_landmarker.task model file.
        Defaults to settings.landmark_model_path.
    num_faces : int, optional
        Maximum number of faces to detect.  Defaults to settings.max_faces.
    min_detection_confidence : float, optional
        Defaults to settings.landmark_min_detection_confidence.
    min_presence_confidence : float, optional
        Defaults to settings.landmark_min_presence_confidence.
    min_tracking_confidence : float, optional
        Defaults to settings.landmark_min_tracking_confidence.

    Notes
    -----
    If the model file is absent the instance is created in a degraded state
    (is_ready == False).  All detect() calls will return {} rather than
    raising.  This mirrors the existing repository convention used by
    MediaPipeDetector and YuNetDetector.
    """

    _IOU_MATCH_THRESHOLD: float = 0.01  # any positive overlap qualifies

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
        min_presence_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
    ) -> None:
        import os

        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision

        settings = get_settings()
        resolved_path = str(model_path if model_path is not None else settings.landmark_model_path)
        self._num_faces = num_faces if num_faces is not None else settings.max_faces
        self._min_detection = (
            min_detection_confidence
            if min_detection_confidence is not None
            else settings.landmark_min_detection_confidence
        )
        self._min_presence = (
            min_presence_confidence
            if min_presence_confidence is not None
            else settings.landmark_min_presence_confidence
        )
        self._min_tracking = (
            min_tracking_confidence
            if min_tracking_confidence is not None
            else settings.landmark_min_tracking_confidence
        )

        self._landmarker = None

        if not os.path.exists(resolved_path):
            # Graceful degradation: model absent → is_ready == False.
            # Consistent with MediaPipeDetector / YuNetDetector convention.
            return

        base_options = tasks.BaseOptions(model_asset_path=resolved_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=self._num_faces,
            min_face_detection_confidence=self._min_detection,
            min_face_presence_confidence=self._min_presence,
            min_tracking_confidence=self._min_tracking,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    @property
    def is_ready(self) -> bool:
        """True when the model loaded successfully and detect() will run inference."""
        return self._landmarker is not None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        tracks: List[TrackedFace],
    ) -> Dict[int, LandmarkResult]:
        """
        Run FaceLandmarker on the full frame and return per-track results.

        Returns {} when:
        - is_ready is False (model absent)
        - image is None or empty
        - no tracks are provided
        - MediaPipe detects no faces in the frame

        Unexpected runtime exceptions from MediaPipe are NOT swallowed.
        """
        if self._landmarker is None or image is None or image.size == 0 or not tracks:
            return {}

        import mediapipe as mp

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {}

        # ---- Convert each MP face to pixel coordinates ----
        face_data: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float], Tuple[float, float]]] = []
        for face_lms in result.face_landmarks:
            pts2d = np.array(
                [[lm.x * w, lm.y * h] for lm in face_lms],
                dtype=np.float32,
            )  # (478, 2)
            pts3d = np.array(
                [[lm.x * w, lm.y * h, lm.z] for lm in face_lms],
                dtype=np.float32,
            )  # (478, 3) — z is MediaPipe canonical depth, not world-space
            x_min = float(np.min(pts2d[:, 0]))
            y_min = float(np.min(pts2d[:, 1]))
            x_max = float(np.max(pts2d[:, 0]))
            y_max = float(np.max(pts2d[:, 1]))
            centroid = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
            face_data.append((pts2d, pts3d, (x_min, y_min, x_max, y_max), centroid))

        # ---- Associate MP faces with caller-supplied track IDs ----
        track_boxes: List[Tuple[float, float, float, float]] = [
            (t.bbox.x_min, t.bbox.y_min, t.bbox.x_max, t.bbox.y_max) for t in tracks
        ]
        lm_boxes = [fd[2] for fd in face_data]
        lm_centroids = [fd[3] for fd in face_data]

        mp_to_track_idx = self._match_faces_to_tracks(lm_boxes, lm_centroids, track_boxes)

        out: Dict[int, LandmarkResult] = {}
        for mp_idx, track_idx in mp_to_track_idx.items():
            pts2d, pts3d, _, _ = face_data[mp_idx]
            track_id = tracks[track_idx].track_id
            out[track_id] = LandmarkResult(
                points_2d=pts2d,
                points_3d=pts3d,
                confidence=1.0,   # FaceLandmarker returns no per-face score; see module docstring
                landmarks_type="478_pt",
            )

        return out

    # ------------------------------------------------------------------
    # Internal matching logic
    # ------------------------------------------------------------------

    @staticmethod
    def _match_faces_to_tracks(
        lm_boxes: List[Tuple[float, float, float, float]],
        lm_centroids: List[Tuple[float, float]],
        track_boxes: List[Tuple[float, float, float, float]],
    ) -> Dict[int, int]:
        """
        Match MediaPipe face indices to track indices (one-to-one).

        Stage 1 — IoU Hungarian matching.
            Pairs with IoU == 0 are excluded (threshold = 0.01).
            Uses the same build_iou_cost_matrix / hungarian_match
            utilities already in app.tracking.association.

        Stage 2 — Centroid-distance fallback.
            Any MediaPipe face not matched in stage 1 is greedily
            assigned to the nearest unmatched track by Euclidean
            centroid distance.  Greedy order follows the order that
            MediaPipe returns faces.

        Parameters
        ----------
        lm_boxes : List[Tuple]
            Bounding boxes derived from each MP face's landmark pixels.
            Format: (x_min, y_min, x_max, y_max).
        lm_centroids : List[Tuple]
            (cx, cy) centroid for each MP face.
        track_boxes : List[Tuple]
            Bounding boxes of confirmed tracked faces, in track order.

        Returns
        -------
        Dict[mp_face_index, track_index]
            One-to-one mapping; at most min(len(lm_boxes), len(track_boxes))
            entries.
        """
        if not lm_boxes or not track_boxes:
            return {}

        # cost_matrix shape: (n_tracks, n_mp_faces)
        iou_cost = build_iou_cost_matrix(track_boxes, lm_boxes)
        matches_iou, unmatched_tr, unmatched_mp = hungarian_match(
            iou_cost, iou_threshold=MediaPipeLandmarker._IOU_MATCH_THRESHOLD
        )

        result: Dict[int, int] = {}
        for tr_idx, mp_idx in matches_iou:
            result[mp_idx] = tr_idx

        # Stage 2: centroid fallback for MP faces that had no IoU overlap
        remaining_tracks = list(unmatched_tr)
        for mp_idx in unmatched_mp:
            if not remaining_tracks:
                break
            cx, cy = lm_centroids[mp_idx]
            best_tr_idx: Optional[int] = None
            best_dist = float("inf")
            for tr_idx in remaining_tracks:
                tx_min, ty_min, tx_max, ty_max = track_boxes[tr_idx]
                tcx = (tx_min + tx_max) / 2.0
                tcy = (ty_min + ty_max) / 2.0
                dist = (cx - tcx) ** 2 + (cy - tcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_tr_idx = tr_idx
            if best_tr_idx is not None:
                result[mp_idx] = best_tr_idx
                remaining_tracks.remove(best_tr_idx)

        return result
