# 🦎 Chameleon

### Real-Time High-Fidelity Identity & Appearance Transfer System

Chameleon is a research-oriented real-time computer-vision system designed to transform the visible appearance of a consenting subject into the appearance of a selected target person while preserving the source subject's motion, pose, expressions, gaze, and temporal continuity.

The long-term goal is a **high-fidelity, low-latency real-time identity/appearance transfer pipeline** suitable for webcam and video input.

> **Completed:** Face detection, 8D Kalman filter face tracking,
> multi-person track continuity validation,
> dense facial landmarks & 3D pose (478-pt MediaPipe FaceLandmarker)
> **Next:** Face & body segmentation

---

## Project Goals

The final Chameleon system is intended to preserve the source subject's:

- Head pose
- Facial expression
- Eye gaze
- Mouth movement
- Body pose
- Temporal motion
- Scene/background
- Lighting as much as possible

while transferring the selected target person's visible identity/appearance.

The system is being developed incrementally rather than starting with a monolithic generative model.

### Core Design Principle

> **Build and benchmark every stage independently before combining them into the complete transformation pipeline.**

The system will therefore consist of explicit processing stages:

```text
Camera / Video
      │
      ▼
Frame Capture
      │
      ▼
Face Detection
      │
      ▼
Face Tracking
      │
      ▼
Facial Landmarks
      │
      ▼
Face / Body Segmentation
      │
      ▼
Pose & Expression Representation
      │
      ▼
Target Identity Representation
      │
      ▼
Appearance / Identity Transfer
      │
      ▼
Temporal Stabilization
      │
      ▼
Compositing / Rendering
      │
      ▼
Output Video
```

---

# Current Progress

## Phase 0 — Architecture & Project Foundation

**Status: ✅ Completed**

The initial repository architecture and engineering guidelines have been established.

### Completed

- Root `CLAUDE.md`
- System architecture documentation
- AI pipeline documentation
- Hardware roadmap
- Latency benchmark plan
- Development roadmap
- Risk register
- Security and consent model
- API design
- Model research tracking
- Initial monorepo structure
- Subsystem-specific `CLAUDE.md` files
- Environment configuration
- `.gitignore`
- `.env.example`

The project is intentionally structured as a modular system rather than a single ML script.

---

# Phase 1 — Computer Vision Baseline

**Status: 🟢 In Progress**

The first objective is establishing a reliable real-time CV foundation before implementing identity transfer.

## 1. Environment

**Status: ✅ Complete**

Verified environment:

| Component | Version / Status |
|---|---|
| Python | Available |
| OpenCV | 5.0.0 |
| PyTorch | 2.12.1+cpu |
| CUDA | Not available |
| MediaPipe | 1.0.0 |
| ONNX Runtime | Installed |
| NumPy | Installed |
| SciPy | Installed |
| scikit-learn | Installed |
| FastAPI | Installed |
| Pydantic Settings | 2.14.1 |
| Pytest | Installed |

Current development machine is running the inference baseline on **CPU**.

---

# Face Detection

**Status: ✅ Implemented & Validated**

Chameleon currently uses the **MediaPipe Tasks Face Detector** with the BlazeFace short-range model.

### Model

```text
blaze_face_short_range.tflite
```

Location:

```text
services/inference/models/blaze_face_short_range.tflite
```

Model size:

```text
229,746 bytes
```

The model was verified against the installed MediaPipe Tasks API:

```text
mediapipe.tasks.python.vision.FaceDetector
```

The legacy:

```python
mp.solutions.face_detection
```

API is **not used**, because MediaPipe 1.0.0 uses the newer Tasks API.

---

# Detector Architecture

The detector subsystem exposes an abstract interface:

```text
BaseDetector
     │
     ├── YuNetDetector
     │
     └── MediaPipeDetector
```

### BaseDetector

Provides the common detector interface:

```text
detect(image) → List[FaceDetection]
```

### MediaPipeDetector

Implemented using:

```text
MediaPipe Tasks
        +
BlazeFace Short Range
        +
TensorFlow Lite XNNPACK
```

The detector:

1. Receives an OpenCV BGR frame.
2. Converts BGR → RGB.
3. Creates a MediaPipe `Image`.
4. Runs face detection.
5. Extracts confidence.
6. Converts MediaPipe bounding boxes into Chameleon's `BoundingBox`.
7. Clamps coordinates to image boundaries.
8. Returns structured `FaceDetection` objects.

The detector model is initialized once and reused across frames.

---

# Detector Validation

## Webcam Smoke Test

**Status: ✅ Passed**

The detector was tested against a live webcam.

Validated:

- Webcam initialization
- Frame capture
- MediaPipe initialization
- Real-time inference
- Bounding-box rendering
- Confidence rendering
- Clean shutdown

The detector successfully tracked the visible face region during live webcam operation.

Smoke-test script:

```text
services/inference/scripts/test_detector.py
```

---

# Detector Performance Benchmark

**Status: ✅ Completed**

Benchmark configuration:

```text
Resolution: 640 × 480
Warmup: 30 frames
Measured frames: 300
Execution: CPU / TFLite XNNPACK
```

### Results

| Metric | Result |
|---|---:|
| Average latency | **2.33 ms** |
| P50 latency | **2.28 ms** |
| P95 latency | **2.78 ms** |
| Minimum latency | **1.96 ms** |
| Maximum latency | **4.34 ms** |
| Detector-only FPS | **428.64 FPS** |
| Webcam throughput | **29.95 FPS** |

The webcam throughput is approximately 30 FPS because the camera capture rate is the limiting factor, not the detector.

### Interpretation

At 30 FPS, one frame has approximately:

```text
33.33 ms
```

of total frame budget.

The current detector consumes approximately:

```text
2.33 ms
```

per frame on average.

Therefore, the detector currently represents a relatively small portion of the eventual real-time processing budget.

> These results describe the current hardware and configuration only. They are not treated as universal performance guarantees.

---

# Data Structures

**Status: ✅ Implemented**

The inference pipeline currently defines structured result objects including:

```text
BoundingBox
FaceDetection
TrackedFace
LandmarkResult
SegmentationResult
PipelineResult
```

These provide explicit contracts between future CV pipeline stages.

Current location:

```text
services/inference/app/pipeline/result.py
```

The module has passed Python import validation.

---

# Current Repository Structure

```text
Chameleon/
│
├── CLAUDE.md
├── README.md
├── .gitignore
├── .env.example
│
├── apps/
│   └── web/
│
├── services/
│   ├── api/
│   │
│   └── inference/
│       ├── app/
│       │   ├── __init__.py
│       │   │
│       │   ├── config/
│       │   │   ├── __init__.py
│       │   │   └── settings.py
│       │   │
│       │   ├── pipeline/
│       │   │   ├── __init__.py
│       │   │   └── result.py
│       │   │
│       │   ├── detection/
│       │   │   ├── __init__.py
│       │   │   └── detector.py
│       │   │
│       │   ├── tracking/
│       │   │
│       │   ├── landmarks/
│       │   │
│       │   ├── segmentation/
│       │   │
│       │   ├── visualization/
│       │   │
│       │   └── metrics/
│       │
│       ├── models/
│       │   └── blaze_face_short_range.tflite
│       │
│       ├── scripts/
│       │   ├── test_detector.py
│       │   └── benchmark_detector.py
│       │
│       ├── tests/
│       │
│       ├── configs/
│       │
│       └── README.md
│
├── packages/
│
├── docs/
│   ├── architecture/
│   ├── research/
│   └── benchmarks/
│
├── datasets/
│
├── experiments/
│
├── infrastructure/
│
└── scripts/
```

---

# Next Task

## Phase 1.4 — Face & Body Segmentation

**Status: ⏳ Next**

The next subsystem will add semantic segmentation of face and body regions
on top of the existing detection, tracking, and landmark pipeline.

Target capabilities:

- Face mask
- Skin mask
- Hair mask
- Eyes
- Mouth
- Neck
- Background
- Clothing where supported

The segmentation stage will provide masks required for accurate compositing
and identity transfer.

Candidate technology:

```text
MediaPipe Image Segmenter
```

---

### Segmentation

Target capabilities:

- Face mask
- Skin mask
- Hair mask
- Eyes
- Mouth
- Neck
- Background
- Clothing where supported

The segmentation stage will provide masks required for accurate compositing and identity transfer.

---

### Visualization

The development pipeline will expose:

- Detection boxes
- Track IDs
- Facial landmarks
- Segmentation masks
- FPS
- Per-stage latency
- Total pipeline latency
- Hardware information

---

# Future Identity Transfer Pipeline

After the CV baseline is reliable, Chameleon will move toward the actual identity-transfer problem.

The intended architecture is modular:

```text
Source Frame
     │
     ▼
Detection
     │
     ▼
Tracking
     │
     ▼
Landmarks / Pose / Expression
     │
     ├───────────────┐
     │               │
     ▼               ▼
Source Geometry   Target Identity
     │               │
     └───────┬───────┘
             ▼
      Identity Transfer
             │
             ▼
       Temporal Model
             │
             ▼
        Compositing
             │
             ▼
        Output Frame
```

The exact generative architecture will be selected through controlled experimentation and benchmarking rather than assumed in advance.

---

# Model Strategy

Chameleon deliberately avoids beginning with a large diffusion-based video model.

A monolithic generative approach introduces substantial challenges:

- High GPU/VRAM requirements
- High inference latency
- Temporal flickering
- Difficult debugging
- Difficult failure isolation
- Large engineering complexity
- Difficult real-time optimization

Instead, the project follows a staged architecture where each component can be benchmarked independently.

Potential future model families will be evaluated based on:

- Identity similarity
- Pose preservation
- Expression preservation
- Temporal consistency
- Occlusion handling
- Visual quality
- Latency
- VRAM consumption
- Stability
- Licensing

---

# Performance Philosophy

Real-time performance is treated as a measurable engineering constraint.

For 30 FPS:

```text
Frame budget ≈ 33.33 ms
```

For 60 FPS:

```text
Frame budget ≈ 16.67 ms
```

Every major pipeline stage will eventually have:

- Average latency
- P50
- P95
- P99 where useful
- Maximum observed latency
- FPS
- Memory usage

The project will optimize based on measured bottlenecks rather than assumptions.

---

# Safety & Consent

Chameleon is intended for **consenting subjects and authorized target identities**.

The system should not be designed or used to impersonate people without their permission, create deceptive identity-based content, or bypass consent requirements.

Future releases should include explicit consent and provenance controls appropriate to the deployment environment.

---

# Development Rules

The project follows several engineering rules.

### 1. Validate before optimizing

Do not optimize a component before establishing correctness and measuring its baseline.

### 2. One subsystem at a time

Do not implement detection, tracking, landmarks, segmentation, and generation simultaneously.

### 3. Test every boundary

Each subsystem must have:

- Unit tests where applicable
- Integration tests
- Real-data validation
- Performance measurements

### 4. No silent failures

Missing models, invalid configuration, or unavailable hardware should produce explicit errors rather than silently returning incorrect results.

### 5. Preserve modularity

Each major CV stage must have an explicit interface so implementations can be replaced without rewriting the entire pipeline.

---

# Current Status Summary

```text
┌──────────────────────────────────────┬────────────┐
│ Component                            │ Status     │
├──────────────────────────────────────┼────────────┤
│ Project architecture                 │ ✅ Done    │
│ Documentation foundation             │ ✅ Done    │
│ Environment setup                    │ ✅ Done    │
│ Configuration system                 │ ✅ Done    │
│ Pipeline result structures           │ ✅ Done    │
│ MediaPipe Tasks integration          │ ✅ Done    │
│ BlazeFace model                      │ ✅ Done    │
│ Face detector                        │ ✅ Done    │
│ Webcam detector smoke test           │ ✅ Passed  │
│ Detector benchmark                   │ ✅ Passed  │
│ Face tracking (8D Kalman + Hungarian)│ ✅ Passed  │
│ Dense landmarks (478-pt, 3D)         │ ✅ Done    │
│ Segmentation                         │ ⏳ Next    │
│ Pose / expression representation     │ ⏳ Planned │
│ Identity representation              │ ⏳ Planned │
│ Appearance transfer                  │ ⏳ Planned │
│ Temporal stabilization               │ ⏳ Planned │
│ Final real-time pipeline             │ ⏳ Planned │
└──────────────────────────────────────┴────────────┘
```

---
## Phase 1.2 — Face Tracking

**Status: ✅ PASS**

Implemented a multi-frame face tracking pipeline using an 8D Kalman filter
and Hungarian assignment.

### Tracking Architecture

- **State Vector:** 8D `[cx, cy, a, h, vx, vy, va, vh]`
- **Measurement Vector:** 4D `[cx, cy, a, h]`
- **Association:** Hungarian algorithm
- **Cost:** `1.0 - IoU`
- **IoU Gating Threshold:** `0.2`
- **Minimum Hits:** `3`
- **Maximum Track Age:** `30` missed frames
- **Re-identification:** Not implemented in Phase 1.2

### Validation

Unit and integration tests completed successfully:

- Kalman filter tests: **8/8 passed**
- Association tests: **5/5 passed**
- Tracker tests: **13/13 passed**
- Tracker integration smoke tests: **6/6 passed**
- Full inference test suite: **26/26 passed**

### Multi-Person Webcam Validation

A controlled live webcam test was performed with multiple people visible.

| Metric | Result |
|---|---:|
| Total frames | 2172 |
| Frames with tracks | 1900 |
| Single-face frames | 1243 |
| Multi-person frames | 657 |
| Maximum concurrent tracks | 3 |
| Unique track IDs | 23 |
| Track switches during single-face tracking | 11 |
| Multi-person track-set changes | 2 |
| Maximum same-ID streak | 309 frames |
| Unobserved track instances | 827 |

No identity swaps were observed during the controlled multi-person
crossing/occlusion test.

### Known Limitations

The tracker currently relies on motion and spatial continuity. Track
fragmentation can occur if detections are unavailable for longer than the
configured maximum track age or if a face undergoes an extreme sudden
spatial displacement.

Appearance-based re-identification / feature embeddings are intentionally
deferred to a later phase.

---

## Phase 1.3 — Dense Facial Landmarks & 3D Pose

**Status: ✅ PASS**

Implemented dense facial landmark detection using the MediaPipe FaceLandmarker
(Tasks API) with IoU + Hungarian algorithm association to tracked faces.

### Landmark Architecture

- **Model:** MediaPipe FaceLandmarker (`face_landmarker.task`, float16, 478-pt)
- **Output — 2D:** `points_2d` shape `(478, 2)` — pixel coordinates `[x, y]`
- **Output — 3D:** `points_3d` shape `(478, 3)` — `[x_pixel, y_pixel, z]` where `z` is MediaPipe canonical depth
- **Association — Stage 1:** IoU cost matrix + Hungarian algorithm (threshold > 0)
- **Association — Stage 2:** Centroid-distance greedy fallback for zero-IoU cases
- **Confidence:** Fixed at `1.0` (FaceLandmarker exposes no per-face score)
- **Graceful degradation:** `is_ready = False` when model absent; `detect()` returns `{}`

### Validation

Unit tests completed successfully:

- Missing model tests: **2/2 passed**
- Initialization tests: **3/3 passed**
- Empty/degenerate input tests: **4/4 passed**
- 2D coordinate conversion tests: **3/3 passed**
- 3D coordinate conversion tests: **4/4 passed**
- Track association tests: **5/5 passed**
- Matching function (pure) tests: **7/7 passed**
- 478-landmark count tests: **2/2 passed**

### Real-World Validation

Single-face inference (real MediaPipe model):

| Metric | Result |
|---|---:|
| Landmark count (2D) | **478** |
| Landmark count (3D) | **478** |
| Latency | **15.4 ms** |
| NaN / Inf values | None |
| IoU (LM bbox vs track bbox) | **0.838** |

Multi-face spatial association (controlled two-face fixture, 820×400):

| Metric | Result |
|---|---:|
| Detector face count | **2** |
| Tracker face count | **2** |
| Landmark result count | **2** |
| IoU Track 1 (LM vs track bbox) | **0.8203** |
| IoU Track 2 (LM vs track bbox) | **0.7795** |
| Off-diagonal IoU (cross-assignment) | **0.0000** |
| Duplicate assignments | **0** |
| Association errors | **0** |

Temporal stability (10-frame synthetic sequence — converging, close, diverging):

| Metric | Result |
|---|---:|
| Frames tested | **10** |
| Track IDs stable across all frames | **✅ [1, 2] constant** |
| Association errors (wrong track) | **0** |
| Duplicate assignments | **0** |
| IoU min / mean / max | **0.61 / 0.72 / 0.84** |
| Per-frame landmark latency | **15–21 ms** |

### Known Limitations

- FaceLandmarker's internal detector requires faces to be sufficiently large
  in the frame; very small or distant faces may not produce landmark results
  even when BlazeFace detects them.
- Landmark association is purely spatial (IoU + centroid distance).
  Appearance-based identity disambiguation is deferred to a later phase.
- `z` depth is MediaPipe canonical space, not metric world-space depth.

---

The next phase will add semantic face and body segmentation on top of the
existing detection, tracking, and landmark pipeline.

# Immediate Next Step

The next development session should begin with:

```text
Phase 1.4 — Face & Body Segmentation
```

# Project Philosophy

Chameleon is being built as an **engineering and research system**, not a single demo script.

The objective is not merely:

> "Make a face swap work."

The objective is:

> **Build a measurable, modular, temporally stable, real-time identity-transfer pipeline whose individual components can be understood, tested, benchmarked, replaced, and optimized independently.**

## License

Chameleon is licensed under the MIT License.

Copyright © 2026 Akshat Prashar.

See [LICENSE](LICENSE) for the full license text.

### Third-Party Components

Chameleon uses third-party libraries, frameworks, and model assets whose licenses remain separate from this project's license.

Current major dependencies include:

- MediaPipe
- BlazeFace / MediaPipe Face Detector model
- OpenCV
- PyTorch
- ONNX Runtime
- NumPy
- SciPy
- FastAPI

Before redistributing Chameleon or deploying it commercially, review the applicable licenses and terms for each dependency and model asset.

---