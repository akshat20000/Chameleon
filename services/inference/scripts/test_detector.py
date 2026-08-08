import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import cv2
from app.detection.detector import MediaPipeDetector

def main():
    m = "services/inference/models/blaze_face_short_range.tflite"
    det = MediaPipeDetector(model_path=m)
    if det.detector is None:
        print("Failed to init detector")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam")
        return
    print("Press q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for d in det.detect(frame):
            b = d.bbox
            p1 = (int(b.x_min), int(b.y_min))
            p2 = (int(b.x_max), int(b.y_max))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, f"{d.confidence:.2f}", (p1[0], max(0, p1[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
