# 🦎 Chameleon

### Real-Time High-Fidelity Identity & Appearance Transfer System

Chameleon is a research-oriented real-time computer-vision system designed to transform the visible appearance of a consenting subject into the appearance of a selected target person while preserving the source subject's motion, pose, expressions, gaze, and temporal continuity.

The long-term goal is a **high-fidelity, low-latency real-time identity/appearance transfer pipeline** suitable for webcam and video input.

> **Current status:** Phase 1 — Computer Vision Baseline  
> **Completed:** Face detection pipeline and real-time webcam validation  
> **Next:** Multi-frame face tracking

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

## Phase 1.2 — Face Tracking

**Status: ⏳ Next**

The next subsystem will introduce temporal identity persistence.

Initial baseline:

```text
Face Detection
      │
      ▼
Kalman Filter Prediction
      │
      ▼
IoU Cost Matrix
      │
      ▼
Hungarian Assignment
      │
      ▼
Track Management
      │
      ▼
Persistent Track IDs
```

Planned Kalman state:

```text
[cx, cy, aspect_ratio, height,
 vx, vy, va, vh]
```

The tracker will support:

- Track creation
- Track prediction
- Detection-to-track association
- Track updates
- Persistent integer IDs
- Lost-track handling
- `max_age`
- `min_hits`
- Configurable matching thresholds

### Important correctness requirement

A person's track identity must remain stable across frames:

```text
Frame 1 → Person A → Track #1
Frame 2 → Person A → Track #1
Frame 3 → Person A → Track #1
...
```

This is especially important for the eventual identity-transfer system.

If track identity changes or swaps between people, Chameleon could potentially apply the wrong target appearance to the wrong subject.

---

# Planned Phase 1 Components

After tracking:

### Facial Landmarks

Target capabilities:

- Dense facial landmarks
- Eye landmarks
- Iris/gaze landmarks
- Mouth landmarks
- Face contour
- 3D landmark representation where available

Candidate technology:

```text
MediaPipe Face Landmarker
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
│ Pipeline result structures            │ ✅ Done    │
│ MediaPipe Tasks integration           │ ✅ Done    │
│ BlazeFace model                      │ ✅ Done    │
│ Face detector                        │ ✅ Done    │
│ Webcam detector smoke test           │ ✅ Passed  │
│ Detector benchmark                   │ ✅ Passed  │
│ Face tracking                        │ ⏳ Next    │
│ Dense landmarks                      │ ⏳ Planned │
│ Segmentation                         │ ⏳ Planned │
│ Pose / expression representation     │ ⏳ Planned │
│ Identity representation              │ ⏳ Planned │
│ Appearance transfer                  │ ⏳ Planned │
│ Temporal stabilization               │ ⏳ Planned │
│ Final real-time pipeline             │ ⏳ Planned │
└──────────────────────────────────────┴────────────┘
```

---

# Immediate Next Step

The next development session should begin with:

```text
Phase 1.2 — Face Tracking
```

Before implementing anything, inspect the existing tracking subsystem and establish its current repository state.

The first target is a deterministic:

```text
Detection → Kalman Prediction → IoU Matching → Persistent Track ID
```

baseline.

Only after that baseline is validated should more sophisticated tracking approaches such as ByteTrack or appearance-assisted ReID be evaluated.

---

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