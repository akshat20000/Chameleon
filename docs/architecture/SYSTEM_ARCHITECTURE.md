# System Architecture

## Overview

BodySwap is a real-time identity/appearance transfer system that transforms a source person's visible appearance to match a target person's identity while preserving the source's pose, expression, and motion.

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL CLIENTS                                │
│                    (Browsers via WebSocket/WebRTC)                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOAD BALANCER (nginx/haproxy)                       │
│                    (SSL termination, routing, rate limiting)               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         │                                               │
         ▼                                               ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│     WEB FRONTEND        │                 │       API SERVER        │
│    (Next.js/React)      │                 │   (Node.js/TypeScript)  │
│   Port: 3000 (dev)      │                 │   Port: 3001 (dev)      │
│                         │                 │                         │
│  - Camera preview       │                 │  - Authentication       │
│  - Target selection     │◄───────────────►│  - Session management  │
│  - Output preview       │    REST/WSS     │  - Target management   │
│  - Quality controls     │                 │  - Metrics collection  │
│  - Performance stats    │                 │  - Audit logging       │
└───────────┬─────────────┘                 └───────────┬─────────────┘
            │                                             │
            │ WebSocket                                   │ HTTP/gRPC
            │                                             ▼
            │                              ┌─────────────────────────────┐
            │                              │    INFERENCE SERVICE        │
            │                              │    (Python/PyTorch)         │
            │                              │    Port: 8000 (FastAPI)     │
            │                              │                             │
            │                              │  ┌───────────────────────┐  │
            │                              │  │   GPU Inference Pool  │  │
            │                              │  │   (1-4 instances)     │  │
            │                              │  └───────────────────────┘  │
            │                              │                             │
            │                              │  - Frame preprocessing     │
            └─────────────────────────────►│  - Detection               │
                              (frames)     │  - Tracking                │
                                           │  - Identity encoding       │
                                           │  - Generation              │
                                           │  - Temporal processing     │
                                           │  - Compositing             │
                                           └─────────────────────────────┘
                                                        │
                                                        ▼
                                           ┌─────────────────────────────┐
                                           │      DATA LAYER             │
                                           │                             │
                                           │  PostgreSQL (port 5432)     │
                                           │  - Users                    │
                                           │  - Authorized targets       │
                                           │  - Sessions                 │
                                           │  - Audit logs               │
                                           │                             │
                                           │  Redis (port 6379)          │
                                           │  - Session state            │
                                           │  - Frame queues             │
                                           │  - Metrics cache            │
                                           └─────────────────────────────┘
```

## Service Responsibilities

### 1. Web Frontend (Next.js)

**Technology**: Next.js 14+, React 18, TypeScript, TailwindCSS

**Purpose**: User interface for camera capture, target selection, and output preview

**Responsibilities**:
- Camera access and frame capture (getUserMedia API)
- WebSocket connection management for real-time streaming
- Target image upload and selection UI
- Quality control sliders and presets
- Performance metrics display (FPS, latency, GPU usage)
- Session state management
- Consent acknowledgment UI

**Ports**:
- Development: 3000
- Production: 80/443 (via nginx)

### 2. API Server (Node.js)

**Technology**: Node.js 20+, Express/Fastify, TypeScript, Prisma

**Purpose**: Backend orchestration, authentication, and session management

**Responsibilities**:
- User authentication (JWT + OAuth)
- Target identity management (CRUD + consent status)
- Session lifecycle management
- Inference job orchestration
- Metrics aggregation
- Audit logging
- Rate limiting and access control
- Provenance metadata generation

**API Endpoints**:
```
POST   /api/auth/login
POST   /api/auth/register
GET    /api/auth/me

POST   /api/targets              # Register new target
GET    /api/targets              # List authorized targets
GET    /api/ttargets/:id         # Get target details
DELETE /api/targets/:id          # Remove target (with audit)

POST   /api/sessions             # Create inference session
GET    /api/sessions             # List sessions
GET    /api/sessions/:id         # Get session details
DELETE /api/sessions/:id         # End session

POST   /api/inference/start      # Start inference
POST   /api/inference/stop       # Stop inference
GET    /api/inference/:id/status # Get inference status

GET    /api/metrics              # System metrics
GET    /api/health               # Health check
```

**Ports**:
- Development: 3001
- Production: 3001

### 3. Inference Service (Python)

**Technology**: Python 3.11+, PyTorch 2.x, FastAPI, CUDA

**Purpose**: Core ML inference pipeline for identity transformation

**Responsibilities**:
- Frame preprocessing and normalization
- Face/person detection
- Face tracking across frames
- Pose and expression estimation
- Target identity encoding
- Identity-conditioned generation
- Temporal consistency processing
- Segmentation and compositing
- Output encoding

**Architecture**:
```
Input Frame
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING MODULE                         │
│  - Color space conversion (BGR → RGB)                          │
│  - Resolution normalization                                     │
│  - Face detection preprocessing                                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DETECTION MODULE                            │
│  - Person detection (YOLOv8/RetinaFace)                         │
│  - Face detection                                               │
│  - Face landmark detection (498 points)                         │
│  - Face parsing                                                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRACKING MODULE                             │
│  - Face identity tracking                                       │
│  - Pose tracking                                                │
│  - Occlusion detection                                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   SOURCE MOTION         │     │   TARGET IDENTITY       │
│                         │     │                         │
│  - Head pose (Euler)    │     │  - Identity encoder     │
│  - Expression weights   │     │  - Feature extraction   │
│  - Eye gaze vector      │     │  - Reference fusion     │
│  - Mouth shape          │     │                         │
│  - Body pose keypoints  │     │                         │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  GENERATION MODULE                               │
│  - Identity-conditioned synthesis                               │
│  - Expression injection                                         │
│  - Pose adaptation                                              │
│  - Lighting estimation and transfer                             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TEMPORAL MODULE                                 │
│  - Feature propagation                                          │
│  - Latent smoothing                                             │
│  - Keyframe management                                          │
│  - Occlusion handling                                           │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COMPOSITING MODULE                              │
│  - Face mask generation                                         │
│  - Edge refinement                                              │
│  - Color matching                                               │
│  - Poisson blending                                             │
│  - Background preservation                                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                          Output Frame
```

**Ports**:
- FastAPI: 8000
- gRPC: 8001

### 4. Data Layer

**PostgreSQL Schema**:
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Authorized targets
CREATE TABLE authorized_targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    reference_images TEXT[] NOT NULL,
    consent_status VARCHAR(50) DEFAULT 'pending',
    consent_timestamp TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    target_id UUID REFERENCES authorized_targets(id),
    status VARCHAR(50) DEFAULT 'created',
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    settings JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Inference metadata
CREATE TABLE inference_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    metrics JSONB,
    status VARCHAR(50)
);

-- Audit log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    target_id UUID REFERENCES authorized_targets(id),
    session_id UUID REFERENCES sessions(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Redis Usage**:
```redis
# Session state
SET session:{session_id}:state "running"
SET session:{session_id}:frame_count 0

# Frame queues (for async processing)
RPUSH frame_queue:{inference_id} frame_data

# Metrics cache
SET metrics:gpu:current "{'utilization': 85, 'memory': 16000}"
SET metrics:inference:{session_id} "{'fps': 24.5, 'latency_p50': 120}"

# Target identity cache
SET target:{target_id}:embedding "base64_encoded_tensor"
```

## Data Flow

### 1. Session Initialization

```
User (Browser)                    API Server                    Inference Service
     │                                │                               │
     │──POST /sessions──────────────►│                               │
     │                                │──Create session─────────────►│
     │                                │                               │
     │◄─session_id───────────────────│                               │
     │                                │                               │
     │──WebSocket connect────────────│                               │
     │                                │                               │
```

### 2. Frame Processing

```
Browser                           API Server                    Inference Service
     │                                │                               │
     │──Frame (via WebSocket)───────►│                               │
     │                                │──Forward frame──────────────►│
     │                                │                               │
     │                                │◄─Transformed frame───────────│
     │◄─Transformed frame────────────│                               │
     │                                │                               │
     │──Metrics update──────────────►│                               │
     │                                │                               │
```

### 3. Target Registration

```
Browser                           API Server                    Storage
     │                                │                               │
     │──Upload reference images─────►│                               │
     │                                │──Save to S3/本地─────────────►│
     │                                │                               │
     │──POST /targets────────────────│                               │
     │                                │──Create DB record───────────►│ PostgreSQL
     │                                │                               │
     │◄─target_id + consent UI───────│                               │
     │                                │                               │
     │──User acknowledges consent────│                               │
     │                                │──Update consent_status──────►│
     │                                │                               │
```

## Communication Protocols

### 1. Browser → API (HTTP/REST)
- Authentication: JWT Bearer tokens
- Content-Type: application/json
- File upload: multipart/form-data

### 2. Browser ↔ Inference (WebSocket)
- Protocol: ws:// or wss://
- Message format: JSON + binary (frames)
- Binary frame encoding: JPEG/PNG
- Latency target: < 50ms

### 3. API ↔ Inference (HTTP/gRPC)
- REST for job submission/status
- gRPC for streaming frame data (optional)
- Protocol buffers for structured data

### 4. Inter-Service (Internal)
- REST within same network
- gRPC for high-throughput paths
- Message queue (Redis) for async tasks

## Failure Modes and Recovery

### Detection Failures
- **No face detected**: Skip frame, use last known state
- **Multiple faces**: Prioritize largest/most centered
- **Low confidence**: Increase detection threshold, log warning

### Tracking Failures
- **Identity switch**: Reset tracker, log event
- **Occlusion**: Maintain state with confidence score
- **Lost track**: Expand search window, fallback to detection

### Generation Failures
- **Model error**: Return original frame with error flag
- **Timeout**: Return timeout frame, retry next frame
- **OOM**: Reduce batch size, fall back to CPU

### Network Failures
- **WebSocket disconnect**: Auto-reconnect with backoff
- **Frame drop**: Skip frame, don't block pipeline
- **API unreachable**: Queue operations, retry with idempotency

## Security Model

### Authentication
- JWT tokens with 15-minute expiry
- Refresh tokens with 7-day expiry
- OAuth2 support (optional)

### Authorization
- Role-based access control (RBAC)
- Target ownership verification
- Session-level consent check

### Data Protection
- Reference images encrypted at rest
- Output frames include watermark
- Audit logs tamper-evident

### Network
- TLS 1.3 for all external traffic
- mTLS for service-to-service
- Rate limiting per user/IP

## Observability

### Logging
- Structured JSON logging
- Request IDs propagated through pipeline
- Log levels: DEBUG, INFO, WARN, ERROR

### Metrics
- **System**: CPU, GPU, memory, network
- **Application**: FPS, latency (P50/P95/P99), errors
- **Business**: Sessions, active users, transformations

### Tracing
- OpenTelemetry for distributed tracing
- Span per pipeline stage
- Trace ID in all logs

### Alerts
- High latency (>200ms sustained)
- Low FPS (<20 sustained)
- GPU memory >90%
- Error rate >1%

## Deployment

### Development
- Single-node docker-compose
- GPU passthrough via nvidia-docker
- Hot reload for all services

### Production
- Kubernetes with GPU operators
- Horizontal pod autoscaling
- Multi-AZ deployment
- CDN for static assets

### Scaling Strategy
- Inference: Horizontal scaling (max 4 replicas per GPU)
- API: Horizontal scaling with load balancer
- Database: Read replicas + connection pooling
- Redis: Cluster mode for high throughput