# 🦎Chameleon

> **A modular real-time digital human pipeline that separates identity, motion, geometry, and appearance.**

Chameleon is an engineering and research project focused on building a controllable digital human pipeline.

The core idea is to separate **who a person looks like** from **how they move**.

A performer provides motion, pose, and expression. A target identity provides facial identity and appearance. These signals are processed independently and will eventually be combined into a temporally stable generated output.

The long-term goal is not simply face swapping.

The goal is to build a system capable of transferring:

- body motion,
- pose,
- facial expression,
- and temporal behavior,

while preserving the identity and appearance of a target person.

---

# The Core Idea

A video contains multiple independent types of information.

```text
                         VIDEO
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      MOTION / GEOMETRY           IDENTITY / APPEARANCE
             │                           │
             ▼                           ▼
      Pose Representation         Identity Representation
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  Controlled Generation
                           │
                           ▼
                      OUTPUT VIDEO

Chameleon treats these as separate engineering problems rather than relying on a single monolithic model.

This allows individual components to be:

tested independently,
benchmarked,
debugged,
replaced,
and optimized without redesigning the entire system.
Why This Architecture?

A naive approach would be:

Performer Video
      │
      ▼
Generative Model
      │
      ▼
Final Video

This makes failures difficult to understand.

If the output flickers, it is unclear whether the problem comes from:

pose tracking,
identity representation,
body geometry,
temporal instability,
the generative model,
compositing,
or rendering.

Chameleon instead creates explicit representation boundaries:

Identity
   ≠
Motion
   ≠
Geometry
   ≠
Appearance

Each boundary has a defined representation and validation strategy.

System Architecture
                         PERFORMER PIPELINE
                         =================


Recorded Video / Camera
          │
          ▼
   Face + Body Detection
          │
          ▼
   Tracking & Association
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
Facial Geometry                               Body Motion
          │                                          │
          ▼                                          ▼
Face Landmarks                            MediaPipe Pose
Expression / Head Pose                           │
          │                                      ▼
          │                            CanonicalMotionState
          │                                      │
          │                                      ▼
          │                            Temporal Stabilization
          │                                      │
          │                                      ▼
          │                         Anatomical SO(3) Frames
          │                                      │
          │                                      ▼
          │                           Local Motion Deltas
          │                                      │
          │                                      ▼
          │                         Kinematic Retargeting
          │                                      │
          │                                      ▼
          │                         Actor Skeleton / FK
          │                                      │
          └──────────────────────┬───────────────────┘
                                 │
                                 ▼


                         FUTURE PIPELINE
                         ===============


                         Target Identity
                                │
                                ▼
                    Identity Representation
                                │
                                ▼
                    Appearance / Identity Model
                                │
                                ▼
                      Generated Appearance
                                │
                                ▼
                     Composition / Rendering
                                │
                                ▼
                            Output Video
Motion Pipeline

The current major focus of Chameleon has been building a reliable motion pipeline before connecting it to appearance generation.

1. Canonical Motion Representation

Raw tracker landmarks are converted into a normalized coordinate system.

CanonicalMotionState


Origin: Pelvis


+X → Camera Right
+Y → Upward
+Z → Toward Camera

Positions are normalized relative to body scale, allowing motion to be represented independently of the performer's absolute height.

Coordinate-system contracts are explicitly validated to prevent:

vertical inversion,
axis inconsistencies,
renderer-specific coordinate assumptions,
invalid anatomical hierarchy.
2. Temporal Stabilization

Raw pose tracking is not sufficiently stable for direct avatar control.

The temporal stabilization pipeline addresses three separate failure modes:

Raw CanonicalMotionState
          │
          ▼
Temporal Association Policy
Left / Right consistency
          │
          ▼
Adaptive Position Filtering
1€ Filter
          │
          ▼
SO(3) Rotation Filtering
Quaternion Continuity + SLERP
          │
          ▼
StableCanonicalMotionState

The stabilizer is responsible for reducing:

sudden position jumps,
high-frequency jitter,
rotation flips,
left/right association errors.

Raw motion remains observable so the quality of the tracker and the effectiveness of stabilization can be measured independently.

3. Anatomical Frame Reconstruction

A bone direction alone does not fully define a 3D rotation.

Bone Direction
      │
      ▼
Defines only part of orientation


Rotation around the bone axis
remains ambiguous

Chameleon reconstructs complete anatomical SO(3) frames using multiple geometric directions.

Primary Bone Direction
          +
Secondary Anatomical Direction
          │
          ▼
Complete Orthonormal Frame
          │
          ▼
        SO(3)

This resolves twist ambiguity before motion is transferred to another skeleton.

4. Kinematic Motion Retargeting

The performer and target actor may have completely different body proportions.

Directly copying joint positions would force the target actor to inherit the performer's geometry.

Instead, Chameleon transfers motion while preserving the target skeleton's intrinsic proportions.

Stable Performer Motion
          │
          ▼
Anatomical SO(3) Frames
          │
          ▼
Local Motion Deltas
          │
          ▼
Target Actor Rest Skeleton
          │
          ▼
Forward Kinematics
          │
          ▼
Retargeted Actor Pose

The core invariant is:

Same Motion
+
Different Body Proportions
=
Different Geometry,
Consistent Articulation

The current retargeting system supports multiple actor proportion profiles, including:

default proportions,
tall actors,
petite actors,
long-arm variants,
short-arm variants.

The current milestone focuses on pose retargeting. Global root trajectory is intentionally deferred to a future stage.

Current Progress
Phase 0 — Architecture Foundation

Status: Complete

Established:

modular repository structure,
architecture documentation,
development roadmap,
benchmark methodology,
subsystem boundaries,
architectural decision records.
Phase 1 — Computer Vision Foundation
Face Detection

Status: Complete

The pipeline detects faces and produces structured face detections and bounding boxes.

The detection interface is designed so implementations can be replaced without affecting downstream systems.

Face Tracking

Status: Complete

Tracking infrastructure maintains persistent face identities across frames using spatial and temporal continuity.

Appearance-based re-identification remains a future improvement.

Dense Facial Landmarks and Pose

Status: Complete

The facial pipeline extracts:

dense 2D landmarks,
canonical 3D landmark geometry,
facial transformation information,
expression-related signals.
Face and Body Segmentation

Status: Complete

Segmentation infrastructure provides masks required for future:

face synthesis,
compositing,
skin preservation,
hair handling,
background separation.
Target Identity Representation

Status: Complete

Target identity is represented independently from performer motion using normalized identity embeddings.

The architecture supports multiple reference images through embedding fusion.

Phase 2 — Motion Representation
Phase 2.4A — Motion Representation Boundary

Status: Complete

Defined the canonical representation that isolates tracker-specific coordinates from downstream motion processing.

Phase 2.4B — Full-Body Static Validation

Status: Complete

Validated:

full-body pose representation,
canonical coordinate conversion,
anatomical hierarchy,
coordinate-system consistency.
Phase 2.4C — Temporal Motion Stability

Status: Complete

Implemented a dedicated temporal stabilization stage.

Raw Motion
    │
    ▼
Stable Motion

The architecture deliberately keeps raw and stabilized motion separate so tracker quality and stabilization quality can be independently observed and benchmarked.

The stabilization stage addresses:

positional jitter,
sudden position jumps,
rotation discontinuities,
left/right association instability.
Phase 2.4D — Kinematic Motion Retargeting

Status: Complete

The current system can:

reconstruct complete anatomical SO(3) frames,
resolve bone-axis twist ambiguity,
extract performer motion deltas,
transfer motion to different actor proportions,
reconstruct actor poses using forward kinematics,
preserve target bone lengths,
preserve left/right anatomical labels,
validate FK consistency,
visualize performer and retargeted actor motion side-by-side.

The retargeting system does not treat legacy joint rotations as authoritative.

Instead, complete anatomical frames are reconstructed from stabilized landmark geometry.

Validation Philosophy

Chameleon does not treat a visually plausible demo as sufficient evidence that a system is correct.

Each subsystem is expected to have measurable invariants.

Coordinate System
    └── Axis and anatomical hierarchy validation


Temporal Stability
    ├── Position jump detection
    ├── Rotation continuity
    ├── Left / Right consistency
    └── NaN / Inf detection


Retargeting
    ├── Bone-length preservation
    ├── Forward-kinematics consistency
    ├── Motion preservation
    ├── Anatomical label preservation
    └── Temporal continuity

Validation combines:

unit tests,
mathematical invariants,
synthetic adversarial cases,
recorded-video benchmarks,
visual debug outputs,
latency measurements.

The objective is to understand why a system works, not simply observe that it sometimes produces a convincing result.

What's Next?

With the motion representation and retargeting pipeline established, the next major challenge is:

Identity and Appearance Transfer

The next stages will focus on combining target identity with performer-driven motion.

The system must eventually preserve:

target facial identity,
performer pose,
facial expression,
body motion,
temporal continuity,
visual realism.

The future pipeline will look broadly like this:

Target Reference Images
          │
          ▼
Identity Representation
          │
          ▼
Appearance / Identity Model
          │
          ▼
Generated Target Appearance
          │
          ├──────────────────────────┐
          │                          │
          ▼                          │
     Performer Motion                │
          │                          │
          └────────────┬─────────────┘
                       ▼
              Composition / Rendering
                       │
                       ▼
                  Output Video

The exact appearance-generation architecture is intentionally not fixed yet.

Candidate approaches will be evaluated based on:

identity similarity,
pose preservation,
expression preservation,
temporal consistency,
visual quality,
latency,
VRAM requirements,
licensing,
deployment complexity.
Project Principles
1. Separate Representations from Models

A model should not define the entire architecture.

External models should be replaceable as long as they satisfy the project's data contracts.

2. Measure Before Optimizing

Performance claims should be benchmarked.

Correctness claims should be tested.

Visual output should be inspected.

3. Preserve Debuggability

The system intentionally exposes intermediate representations.

Raw Motion
    │
    ▼
Stable Motion
    │
    ▼
Anatomical Frames
    │
    ▼
Local Motion Deltas
    │
    ▼
Retargeted Skeleton
    │
    ▼
Generated Appearance
    │
    ▼
Final Composite

This makes failures traceable instead of hiding them inside a single end-to-end model.

4. Avoid Monolithic Magic

The goal is not:

Feed a video into a large model and hope the output looks correct.

The goal is to understand where identity, motion, geometry, and appearance enter the system and validate every transformation between them.

Repository Structure
Chameleon/
│
├── services/
│   └── inference/
│       ├── app/
│       │   ├── detection/
│       │   ├── tracking/
│       │   ├── landmarks/
│       │   ├── segmentation/
│       │   ├── identity/
│       │   └── motion/
│       │
│       ├── scripts/
│       └── tests/
│
├── docs/
│   ├── architecture/
│   │   └── ADR/
│   ├── development/
│   ├── research/
│   └── benchmarks/
│
├── test_data/
│   ├── inputs/
│   └── outputs/
│
├── experiments/
├── infrastructure/
└── scripts/
Project Status
FOUNDATION
████████████████████  Complete


COMPUTER VISION
████████████████████  Core pipeline established


MOTION REPRESENTATION
████████████████████  Complete


TEMPORAL STABILIZATION
████████████████████  Complete


KINEMATIC RETARGETING
████████████████████  Complete


IDENTITY / APPEARANCE GENERATION
████░░░░░░░░░░░░░░░░  Next Major Stage


FINAL TEMPORAL RENDERING
░░░░░░░░░░░░░░░░░░░░  Future
Development Philosophy

Chameleon is not intended to be a single demo script.

It is an engineering and research system designed so its components can be:

Understood, tested, benchmarked, replaced, and optimized independently.

The long-term objective is a controllable pipeline where identity and motion are independently represented, validated, and ultimately recombined into a temporally stable digital human output.
