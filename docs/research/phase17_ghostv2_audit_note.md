# Phase 1.7 Candidate Evaluation Note — GhostV2

**Date:** 2026-08-19  
**Candidate Model:** GhostV2 (`dimitribarbot/ghostv2`)  
**Status:** **REJECTED (Visual Quality Gate Failure)**  

---

## Summary of Findings

1. **Evaluation Scope:**
   GhostV2 was evaluated through embedding compatibility tests, AdaFace vs MobileFaceNet identity conditioning experiments, 3×3 permutation audits, zero-vector baseline audits, CPU profiling, GTX 1650 GPU benchmarking, FP16 vs FP32 precision testing, 5-point alignment parity audits, and official repository end-to-end reproduction.

2. **Technical Parity Achieved:**
   Technical executability (Gates 1–5) and 100% parameter key loading parity with the official reference implementation were successfully achieved.

3. **Visual Quality Gate Failure:**
   The official end-to-end GhostV2 inference pipeline itself produced visually unacceptable identity blending, facial geometry distortion, and eye/nose structural artifacts on the project's evaluation inputs.

4. **Production Decision:**
   GhostV2 is **REJECTED** for production integration. Zero files under `services/inference/app/` were modified or integrated. The project is pivoting toward researching a custom identity-preserving full-body generation system.
