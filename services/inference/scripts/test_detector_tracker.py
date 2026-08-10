import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detection.detector import MediaPipeDetector
from app.tracking.tracker import KalmanFilterTracker

def run_integration_smoke():
    model_path = Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"
    detector = MediaPipeDetector(model_path=str(model_path))
    tracker = KalmanFilterTracker(min_hits=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam 0")
        return

    total_frames, total_detections, total_tracks = 0, 0, 0
    frames_with_tracks, unobserved_frames = 0, 0
    new_track_ids, track_switches = set(), 0
    single_face_frames, prev_single_id = 0, None
    current_streak, max_streak = 0, 0

    multi_person_frames = 0
    concurrent_track_ids = set()
    multi_person_track_set_changes = 0
    max_concurrent_tracks = 0
    prev_multi_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        dets = detector.detect(frame)
        total_detections += len(dets)
        tracks = tracker.update(dets)
        total_tracks += len(tracks)

        max_concurrent_tracks = max(max_concurrent_tracks, len(tracks))
        current_ids = {tr.track_id for tr in tracks}

        if len(tracks) > 0:
            frames_with_tracks += 1
        for tr in tracks:
            new_track_ids.add(tr.track_id)
            if not tr.is_observed:
                unobserved_frames += 1

        if len(tracks) == 1:
            single_face_frames += 1
            cid = tracks[0].track_id
            if prev_single_id is not None and cid != prev_single_id:
                track_switches += 1
                current_streak = 1
            else:
                current_streak += 1
            prev_single_id = cid
            max_streak = max(max_streak, current_streak)
        elif len(tracks) == 0:
            prev_single_id = None
            current_streak = 0

        if len(tracks) >= 2:
            multi_person_frames += 1
            concurrent_track_ids.update(current_ids)
            if len(prev_multi_ids) >= 2 and current_ids != prev_multi_ids:
                multi_person_track_set_changes += 1
            prev_multi_ids = current_ids
        else:
            prev_multi_ids = set()

        for tr in tracks:
            b = tr.bbox
            x1, y1 = int(b.x_min), int(b.y_min)
            x2, y2 = int(b.x_max), int(b.y_max)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lbl = f"ID:{tr.track_id} H:{tr.hits} T:{tr.time_since_update} Obs:{tr.is_observed}"
            cv2.putText(frame, lbl, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Detector + Tracker Continuity (q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total Frames: {total_frames}")
    print(f"Frames with Tracks: {frames_with_tracks}")
    print(f"Single-Face Frames: {single_face_frames}")
    print(f"Unique Track IDs: {len(new_track_ids)}")
    print(f"Track Switches (Single-Face): {track_switches}")
    print(f"Max Consecutive Single-Face ID Streak: {max_streak}")
    print(f"Unobserved Track Instances: {unobserved_frames}")
    print(f"Multi-Person Frames (>=2): {multi_person_frames}")
    print(f"Max Concurrent Tracks: {max_concurrent_tracks}")
    print(f"Unique Track IDs in Multi-Person Frames: {len(concurrent_track_ids)}")
    print(f"Multi-Person Track Set Changes: {multi_person_track_set_changes}")

if __name__ == "__main__":
    run_integration_smoke()
