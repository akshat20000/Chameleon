import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.result import BoundingBox, FaceDetection
from app.tracking.tracker import KalmanFilterTracker

def run_smoke_tests():
    tr = KalmanFilterTracker(min_hits=3)
    
    assert tr.update([]) == []
    print("1. Empty update passed")
    
    d1 = FaceDetection(BoundingBox(10.0, 10.0, 50.0, 50.0), 0.9)
    res1 = tr.update([d1])
    res2 = tr.update([d1])
    assert len(res1) == 0 and len(res2) == 0
    print("2. Tentative tracks (hits 1, 2) hidden passed")
    
    res3 = tr.update([d1])
    assert len(res3) == 1 and res3[0].track_id == 1 and res3[0].hits == 3
    print("3. Confirmed track at hits=3 passed")

    d2 = FaceDetection(BoundingBox(150.0, 150.0, 200.0, 200.0), 0.85)
    tr.update([d1, d2])
    res4 = tr.update([d1, d2])
    ids = {t.track_id for t in res4}
    assert len(res4) == 2 and ids == {1, 2}
    print("4. Two separated detections confirmed distinct IDs passed")
    
    res5 = tr.update([])
    assert len(res5) == 2 and res5[0].time_since_update == 1
    print("5. Temporary disappearance aging passed")
    
    for _ in range(10):
        for t in tr.update([d1, d2]):
            b = t.bbox
            assert np.isfinite([b.x_min, b.y_min, b.x_max, b.y_max]).all()
            assert np.isfinite(t.confidence)
    print("6. 10 cycles finite values passed")

if __name__ == "__main__":
    run_smoke_tests()
