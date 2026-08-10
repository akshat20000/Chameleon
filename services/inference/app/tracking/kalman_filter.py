import numpy as np
from typing import Tuple

class KalmanFilter:
    """
    8D Constant Velocity Kalman Filter for Bounding Box Tracking.
    State vector x: [cx, cy, a, h, vx, vy, va, vh]^T
    Measurement vector z: [cx, cy, a, h]^T
    """
    def __init__(
        self,
        std_weight_position: float = 0.05,
        std_weight_velocity: float = 0.00625,
        std_weight_aspect: float = 0.01,
        std_weight_aspect_vel: float = 0.001,
    ):
        self.std_weight_position = std_weight_position
        self.std_weight_velocity = std_weight_velocity
        self.std_weight_aspect = std_weight_aspect
        self.std_weight_aspect_vel = std_weight_aspect_vel
        self.H = np.eye(4, 8, dtype=np.float64)

    @staticmethod
    def bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
        x_min, y_min, x_max, y_max = bbox
        if not (np.isfinite(x_min) and np.isfinite(y_min) and np.isfinite(x_max) and np.isfinite(y_max)):
            raise ValueError(f"Bounding box contains non-finite values: {bbox}")
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid dimensions (w={w}, h={h}): {bbox}")
        return np.array([x_min + w / 2.0, y_min + h / 2.0, w / h, h], dtype=np.float64)

    @staticmethod
    def z_to_bbox(z: np.ndarray) -> Tuple[float, float, float, float]:
        cx, cy, a, h = z[:4]
        cx, cy = float(cx), float(cy)
        a, h = max(1e-4, float(a)), max(1e-4, float(h))
        w = a * h
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def get_F(self, dt: float = 1.0) -> np.ndarray:
        F = np.eye(8, dtype=np.float64)
        dt_val = float(dt)
        F[0, 4] = dt_val
        F[1, 5] = dt_val
        F[2, 6] = dt_val
        F[3, 7] = dt_val
        return F

    def get_Q(self, height: float, aspect_ratio: float) -> np.ndarray:
        h = max(1e-4, float(height))
        a = max(1e-4, float(aspect_ratio))
        sp, pa = self.std_weight_position * h, self.std_weight_aspect * a
        sv, va = self.std_weight_velocity * h, self.std_weight_aspect_vel * a
        return np.diag([sp**2, sp**2, pa**2, sp**2, sv**2, sv**2, va**2, sv**2])

    def get_R(self, height: float, aspect_ratio: float) -> np.ndarray:
        h = max(1e-4, float(height))
        a = max(1e-4, float(aspect_ratio))
        sm, ma = self.std_weight_position * h, self.std_weight_aspect * a
        return np.diag([sm**2, sm**2, ma**2, sm**2])

    def initiate(self, bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        z = self.bbox_to_z(bbox)
        cx, cy, a, h = z
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = z
        sp0, pa0 = 2.0 * h, 2.0 * a
        sv0, va0 = 10.0 * h, 10.0 * a
        cov_diag = np.array([sp0**2, sp0**2, pa0**2, sp0**2, sv0**2, sv0**2, va0**2, sv0**2], dtype=np.float64)
        cov = np.diag(cov_diag)
        if not np.all(np.isfinite(cov)) or np.any(cov_diag <= 0):
            raise ValueError(f"Invalid covariance initialized: {cov_diag}")
        return mean, cov

    def predict(self, mean: np.ndarray, covariance: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        F = self.get_F(dt)
        new_mean = F @ mean
        Q = self.get_Q(new_mean[3], new_mean[2])
        new_cov = F @ covariance @ F.T + Q
        if not (np.all(np.isfinite(new_mean)) and np.all(np.isfinite(new_cov))):
            raise ValueError("Predict produced non-finite values")
        return new_mean, new_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        z = self.bbox_to_z(bbox)
        R = self.get_R(mean[3], mean[2])
        y = z - (self.H @ mean)
        S = self.H @ covariance @ self.H.T + R
        K = np.linalg.solve(S.T, (self.H @ covariance)).T
        I_KH = np.eye(8, dtype=np.float64) - (K @ self.H)
        new_cov = I_KH @ covariance @ I_KH.T + K @ R @ K.T
        new_mean = mean + (K @ y)
        if not (np.all(np.isfinite(new_mean)) and np.all(np.isfinite(new_cov))):
            raise ValueError("Update produced non-finite values")
        return new_mean, new_cov
