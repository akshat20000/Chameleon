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
