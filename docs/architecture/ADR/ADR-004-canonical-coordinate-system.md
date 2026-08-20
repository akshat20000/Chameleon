# ADR-004: Canonical Motion Coordinate System & Transformation Contract

**Status:** ACCEPTED  
**Date:** 2026-08-20  
**Deciders:** Chameleon Engineering  

---

## 1. Context

During Phase 2.4C temporal testing, visual skeleton overlay inspection revealed that the skeleton was rendered vertically inverted (ankle joints appearing above pelvis, head appearing below pelvis).

An audit of the coordinate transformation pipeline traced the root cause to an un-negated Y-axis during MediaPipe Pose World Landmark adaptation:
- MediaPipe Pose World Landmarks define `+y` as **DOWNWARD** (pointing toward feet / ground).
- `CanonicalMotionState` specification defines `+Y` as **UPWARD** (pointing toward head / sky).

`mediapipe_adapter.py` normalized `(pos_metric - pelvis)` by body scale and negated `Z`, but failed to negate `Y`. As a result, `CanonicalMotionState` carried negative Y values for head joints and positive Y values for ankle joints.

---

## 2. Decision & Transformation Pipeline

We establish **one explicit canonical coordinate system** and document the precise transformation pipeline across all boundaries.

### Complete Transformation Chain

```
MediaPipe Pose World Landmarks (meters)
   Origin: hip midpoint
   +x: camera right
   +y: DOWNWARD (towards ground)
   +z: AWAY from camera (into screen)
        │
        ▼  (mediapipe_adapter.py)
        │  X_canon = (X_metric - pelvis_x) / body_scale
        │  Y_canon = -(Y_metric - pelvis_y) / body_scale   [Y NEGATED]
        │  Z_canon = -(Z_metric - pelvis_z) / body_scale   [Z NEGATED]
        ▼
CanonicalMotionState (units: body height, 1.0 = height)
   Origin: Pelvis (0.0, 0.0, 0.0)
   +X: camera right (performer left for front-facing camera)
   +Y: UPWARD (toward head)
   +Z: TOWARD camera (out of chest)
        │
        ▼  (renderers / visualization boundary)
        │  screen_x = origin_x + X_canon * scale_px
        │  screen_y = origin_y - Y_canon * scale_px   [screen +y is DOWN]
        ▼
Image Pixel Space (pixels)
   Origin: top-left (0, 0)
   +x: rightward
   +y: DOWNWARD
```

---

## 3. Mathematical Contracts

### 3.1 3D Canonical Space Contract (+Y is UPWARD)

For a standard upright standing performer:
$$\text{head.y} > \text{neck.y} > \text{chest.y} > \text{pelvis.y } (0.0)$$
$$\text{pelvis.y } (0.0) > \text{knee.y} > \text{ankle.y}$$

### 3.2 2D Image Pixel Space Contract (+y is DOWNWARD)

At the visualization boundary:
$$\text{pixel\_y(head)} < \text{pixel\_y(neck)} < \text{pixel\_y(chest)} < \text{pixel\_y(pelvis)}$$
$$\text{pixel\_y(pelvis)} < \text{pixel\_y(knee)} < \text{pixel\_y(ankle)}$$

---

## 4. Automated Validation & Testing

1. **`CanonicalMotionState.validate()`**: Automatically rejects any state exhibiting vertical anatomical hierarchy inversions (e.g. `head.y <= pelvis.y` or `knee.y >= pelvis.y`).
2. **`test_coordinate_sanity.py`**: Automated test suite asserting both 3D canonical space ordering and 2D image pixel projection hierarchy.

---

## 5. Consequences

- `CanonicalMotionState` is now mathematically guaranteed to be upright (+Y upward).
- Renderers do appropriate conversion (`screen_y = origin_y - Y_canon * scale_px`) only at the visualization boundary.
- Any tracker adapter (MediaPipe, SMPL-X, RTMPose) must adhere to this exact canonical coordinate convention.
