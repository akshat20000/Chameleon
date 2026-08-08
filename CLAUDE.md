# BodySwap - Real-Time High-Fidelity Identity/Appearance Transfer System

## Project Purpose

Production-quality real-time appearance/identity-transfer application that transforms the visible appearance of a consenting source person toward a selected target person's appearance while preserving:
- Head pose
- Facial expression
- Eye gaze
- Mouth movement
- Body pose
- Body motion
- Camera perspective
- Lighting consistency
- Temporal continuity

## Core Principles

1. **Consent-first**: All features require explicit authorization for target identities
2. **Separation of concerns**: Identity information from target, motion/expression/pose from source
3. **Temporal consistency as first-class requirement**: No flickering, identity drift, or frame-to-frame instability
4. **Measurable quality**: Quantitative evaluation, not just "looks good"
5. **Incremental development**: Build correct baseline before optimizing
6. **Honest status**: Mark unimplemented features explicitly, never fake functionality

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js/React)                     │
│  Camera Preview │ Target Selection │ Quality Controls │ Metrics     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ WebSocket/WebRTC
┌────────────────────────────────┴────────────────────────────────────┐
│                        API SERVER (Node.js/TypeScript)              │
│  Auth │ Sessions │ Target Management │ Orchestration │ Audit        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP/gRPC
┌────────────────────────────────┴────────────────────────────────────┐
│                     INFERENCE SERVICE (Python/PyTorch)              │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │Detection │ → │Tracking  │ → │Pose/Expr │ → │Identity  │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
│        │              │              │              │               │
│        └──────────────┴──────────────┴──────────────┘               │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │Transform │ → │Temporal  │ → │Composite │ → │Output    │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────────┐
│                        DATA LAYER                                   │
│  PostgreSQL (users, sessions, targets, audit) │ Redis (state)       │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
BodySwap/
├── apps/
│   ├── web/                    # Next.js frontend
│   │   ├── app/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── lib/
│   │   └── types/
│   │
│   └── api/                    # Node.js backend
│       ├── src/
│       │   ├── controllers/
│       │   ├── services/
│       │   ├── middleware/
│       │   ├── routes/
│       │   ├── repositories/
│       │   └── types/
│       └── tests/
│
├── services/
│   └── inference/              # Python ML service
│       ├── app/
│       │   ├── api/
│       │   ├── pipeline/
│       │   ├── detection/
│       │   ├── tracking/
│       │   ├── identity/
│       │   ├── pose/
│       │   ├── segmentation/
│       │   ├── generation/
│       │   ├── temporal/
│       │   ├── compositing/
│       │   ├── preprocessing/
│       │   └── postprocessing/
│       ├── models/
│       ├── configs/
│       ├── benchmarks/
│       ├── tests/
│       └── scripts/
│
├── packages/
│   ├── shared-types/
│   ├── config/
│   └── observability/
│
├── infrastructure/
│   ├── docker/
│   ├── compose/
│   ├── database/
│   └── deployment/
│
├── experiments/
│   ├── notebooks/
│   ├── ablations/
│   ├── benchmarks/
│   └── results/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── manifests/
│
├── docs/
│   ├── architecture/
│   ├── research/
│   ├── api/
│   ├── setup/
│   └── decisions/
│
├── scripts/
│
├── .env.example
├── docker-compose.yml
├── README.md
└── CLAUDE.md
```

## Development Commands

### Setup
```bash
# Install all dependencies
pnpm install

# Setup Python environment
cd services/inference
python -m venv venv
source venv/bin/activate  # or `.\venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Development
```bash
# Start all services
pnpm dev

# Start specific service
pnpm --filter @bodyswap/web dev
pnpm --filter @bodyswap/api dev
pnpm --filter @bodyswap/inference dev

# Run database migrations
pnpm --filter @bodyswap/api db:migrate
```

### Testing
```bash
# Run all tests
pnpm test

# Run specific package tests
pnpm --filter @bodyswap/inference test

# Run Python tests with coverage
cd services/inference
pytest --cov=app tests/
```

### Linting/Type Checking
```bash
# Lint all
pnpm lint

# Type check all
pnpm typecheck

# Python linting
cd services/inference
ruff check .
mypy app/
```

### Docker
```bash
# Build all images
docker-compose build

# Start all services
docker-compose up

# Start with GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Model Management Rules

1. **Never commit model weights to Git**
   - Use Git LFS for small models (< 100MB)
   - Use external storage (S3, HuggingFace Hub) for large models
   - Document download instructions in `models/README.md`

2. **Model versioning**
   - Each model has a version tag in config
   - Store model checksums for verification
   - Document model licenses explicitly

3. **Model downloading**
   - Use `scripts/download_models.py` for automated downloads
   - Verify checksums after download
   - Fail fast if models missing

## Dataset Rules

1. **Never commit datasets to Git**
   - Use `.gitignore` for all dataset directories
   - Document dataset sources in `datasets/README.md`

2. **Consent requirements**
   - Only use properly licensed datasets
   - Document consent/license for each dataset
   - Prefer synthetic data where possible

3. **Dataset organization**
   - `datasets/raw/` - Original data (never processed in place)
   - `datasets/processed/` - Preprocessed data
   - `datasets/manifests/` - Metadata and splits

## Security Rules

1. **No secrets in Git**
   - Use `.env` files (never commit)
   - Provide `.env.example` templates
   - Use secret management in production

2. **Consent verification**
   - Every target identity requires explicit consent record
   - Consent status checked before any transformation
   - Audit log for all operations

3. **Output provenance**
   - All generated media includes watermarking
   - Metadata includes session, target, and timestamp
   - Provenance data is tamper-evident

## Performance Requirements

### Latency Budget (Target)
| Stage | Target (ms) | Notes |
|-------|-------------|-------|
| Capture | 5 | Camera frame acquisition |
| Detection | 15 | Face/person detection |
| Tracking | 5 | Identity association |
| Pose/Expression | 10 | Motion estimation |
| Identity Encoding | 10 | Target identity (cached) |
| Transformation | 50-100 | Primary generation |
| Temporal | 10-20 | Consistency processing |
| Compositing | 5 | Blending/output |
| **Total** | **110-170** | ~6-9 FPS minimum |

### Quality Metrics
- Identity similarity: > 0.7 (ArcFace cosine similarity)
- Expression preservation: < 5° angular error
- Pose preservation: < 3° angular error
- Temporal stability: < 0.01 frame-to-frame LPIPS
- No visible flicker in 95% of frames

## GPU Requirements

### Development
- Minimum: NVIDIA GTX 1080 Ti (11GB VRAM)
- Recommended: NVIDIA RTX 3080 (10GB VRAM)

### Offline Inference
- Minimum: NVIDIA RTX 3080 (10GB VRAM)
- Recommended: NVIDIA RTX 4080 (16GB VRAM)

### Real-Time Inference
- Minimum: NVIDIA RTX 3090 (24GB VRAM)
- Recommended: NVIDIA RTX 4090 (24GB VRAM)

## Prohibited Shortcuts

1. **No fake real-time**: Don't claim real-time if batching/preprocessing hides latency
2. **No identity mixing**: Target identity must come from authorized reference only
3. **No blind pasting**: Must preserve source pose/expression, not paste static face
4. **No ignoring temporal**: Flickering output is a bug, not a limitation
5. **No hiding failures**: Document known failure modes explicitly
6. **No placeholder AI**: Mark unimplemented AI features as NOT IMPLEMENTED

## Definition of Done

A feature is complete only when:
1. Implementation is correct and tested
2. Unit and integration tests pass
3. Documentation exists
4. Failure cases are considered and documented
5. Performance is measured (where relevant)
6. Integration with architecture is verified
7. No secrets are committed
8. Dependencies are documented with justification
9. Implementation is reproducible
10. Known limitations are documented

## Current Status

**Phase**: 0 - Architecture
**Started**: 2026-08-08
**Status**: In Progress

See `docs/architecture/ROADMAP.md` for detailed milestone tracking.

## Key Documents

- Architecture: `docs/architecture/SYSTEM_ARCHITECTURE.md`
- AI Pipeline: `docs/architecture/AI_PIPELINE.md`
- Model Research: `docs/research/MODEL_EVALUATION.md`
- Hardware Plan: `docs/setup/HARDWARE_REQUIREMENTS.md`
- Roadmap: `docs/architecture/ROADMAP.md`
- Risk Register: `docs/architecture/RISK_REGISTER.md`
- Benchmark Plan: `docs/research/BENCHMARK_PLAN.md`
- Security Model: `docs/architecture/SECURITY_MODEL.md`
