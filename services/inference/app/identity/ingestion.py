"""
Reference Ingestion & Stage 1 Cheap Image Deduplication Pre-Filter.

Ingests reference photos or video files, extracts sampled views, applies cheap
MSE image similarity filtering to drop near-duplicate frames before running landmark
or embedding inference, and writes deterministically named images to workspace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class IngestionConfig:
    video_sample_stride: int = 15           # sample 1 frame every N video frames
    max_views_to_ingest: int = 60           # upper limit on raw ingested views
    mse_dedup_threshold: float = 25.0        # MSE pixel difference threshold (below this = duplicate)
    target_max_dimension: int = 1920        # resize oversized images to max dimension


@dataclass
class IngestedView:
    view_index: int
    source_identifier: str
    original_frame_index: Optional[int]
    image_path: Path                        # path to saved image in workspace
    image_bgr: np.ndarray


@dataclass
class IngestionResult:
    source_path: Path
    is_video: bool
    total_frames_found: int
    sampled_frames_count: int
    prefilter_dropped_count: int
    views: List[IngestedView]


class ReferenceIngestor:
    """
    Ingests reference images or video and applies Stage 1 cheap pre-filtering.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    def ingest(
        self,
        source_path: Union[str, Path],
        workspace_selected_dir: Union[str, Path],
    ) -> IngestionResult:
        """
        Ingest reference material from a file/directory path and write selected views to workspace.

        Parameters
        ----------
        source_path : Path or str
            Directory of images or single video file.
        workspace_selected_dir : Path or str
            Destination directory (e.g. workspace / inputs / selected_views)

        Returns
        -------
        IngestionResult containing metadata and list of IngestedView objects.
        """
        src = Path(source_path)
        out_dir = Path(workspace_selected_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise FileNotFoundError(f"Reference source path does not exist: {src}")

        if src.is_file() and src.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            return self._ingest_video(src, out_dir)
        elif src.is_dir():
            return self._ingest_image_directory(src, out_dir)
        elif src.is_file() and src.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            return self._ingest_single_image(src, out_dir)
        else:
            raise ValueError(f"Unsupported reference source format: {src}")

    def _ingest_video(self, video_path: Path, out_dir: Path) -> IngestionResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        sampled_count = 0
        dropped_count = 0

        views: List[IngestedView] = []
        prev_gray: Optional[np.ndarray] = None

        try:
            while cap.isOpened() and len(views) < self.config.max_views_to_ingest:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.config.video_sample_stride == 0:
                    sampled_count += 1
                    frame_resized = self._resize_if_needed(frame)
                    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                    # Stage 1 Cheap Pre-Filter: MSE similarity check against previous saved frame
                    if prev_gray is not None:
                        mse = float(np.mean((gray.astype(np.float32) - prev_gray.astype(np.float32)) ** 2))
                        if mse < self.config.mse_dedup_threshold:
                            dropped_count += 1
                            frame_idx += 1
                            continue

                    v_idx = len(views)
                    v_filename = f"view_{v_idx:03d}.png"
                    v_path = out_dir / v_filename
                    cv2.imwrite(str(v_path), frame_resized)

                    views.append(
                        IngestedView(
                            view_index=v_idx,
                            source_identifier=f"{video_path.name}:frame_{frame_idx:06d}",
                            original_frame_index=frame_idx,
                            image_path=v_path,
                            image_bgr=frame_resized,
                        )
                    )
                    prev_gray = gray.copy()

                frame_idx += 1
        finally:
            cap.release()

        logger.info(
            "Ingested video %s: %d total frames, %d sampled, %d pre-filter dropped, %d stored",
            video_path.name, total_frames, sampled_count, dropped_count, len(views)
        )

        return IngestionResult(
            source_path=video_path,
            is_video=True,
            total_frames_found=total_frames,
            sampled_frames_count=sampled_count,
            prefilter_dropped_count=dropped_count,
            views=views,
        )

    def _ingest_image_directory(self, dir_path: Path, out_dir: Path) -> IngestionResult:
        image_files = sorted(
            [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        )

        total_files = len(image_files)
        sampled_count = 0
        dropped_count = 0
        views: List[IngestedView] = []
        prev_gray: Optional[np.ndarray] = None

        for f_path in image_files:
            if len(views) >= self.config.max_views_to_ingest:
                break

            frame = cv2.imread(str(f_path))
            if frame is None:
                continue

            sampled_count += 1
            frame_resized = self._resize_if_needed(frame)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and prev_gray.shape == gray.shape:
                mse = float(np.mean((gray.astype(np.float32) - prev_gray.astype(np.float32)) ** 2))
                if mse < self.config.mse_dedup_threshold:
                    dropped_count += 1
                    continue

            v_idx = len(views)
            v_filename = f"view_{v_idx:03d}.png"
            v_path = out_dir / v_filename
            cv2.imwrite(str(v_path), frame_resized)

            views.append(
                IngestedView(
                    view_index=v_idx,
                    source_identifier=f_path.name,
                    original_frame_index=None,
                    image_path=v_path,
                    image_bgr=frame_resized,
                )
            )
            prev_gray = gray.copy()

        logger.info(
            "Ingested directory %s: %d images found, %d pre-filter dropped, %d stored",
            dir_path.name, total_files, dropped_count, len(views)
        )

        return IngestionResult(
            source_path=dir_path,
            is_video=False,
            total_frames_found=total_files,
            sampled_frames_count=sampled_count,
            prefilter_dropped_count=dropped_count,
            views=views,
        )

    def _ingest_single_image(self, img_path: Path, out_dir: Path) -> IngestionResult:
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Failed to read image file: {img_path}")

        frame_resized = self._resize_if_needed(frame)
        v_path = out_dir / "view_000.png"
        cv2.imwrite(str(v_path), frame_resized)

        view = IngestedView(
            view_index=0,
            source_identifier=img_path.name,
            original_frame_index=None,
            image_path=v_path,
            image_bgr=frame_resized,
        )

        return IngestionResult(
            source_path=img_path,
            is_video=False,
            total_frames_found=1,
            sampled_frames_count=1,
            prefilter_dropped_count=0,
            views=[view],
        )

    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.config.target_max_dimension:
            return img

        scale = self.config.target_max_dimension / float(max_dim)
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
