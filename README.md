# Animat Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LLM](https://img.shields.io/badge/LLM-powered-8A2BE2.svg)](https://github.com/digitopia-research/animat-agent)

---

## The agent that lives.

Animat Agent is not a chatbot. It's an LLM-based embodied "artificial organism" framework — a continuously running entity that senses the physical world, possesses internal drives and emotions, and acts autonomously without waiting for human input. It explores out of boredom, seeks companionship out of loneliness, and cries for help when its battery runs critically low.

**v0.2** (current) introduces a fundamental architectural shift: the LLM transitions from an on-demand reasoning engine into a **perpetual stream of consciousness** — always thinking, seldom speaking, its cognition continuously shaped by bodily states rather than told what to feel by prompt text.

---

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Architecture](#architecture)
  - [v0.1 Architecture (Reference)](#v01-architecture-reference)
  - [v0.2 Architecture (Current)](#v02-architecture-current)
  - [Key Innovations in v0.2](#key-innovations-in-v02)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Dependencies](#dependencies)
  - [Model Download](#model-download)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Full Stack (MQTT + Redis)](#running-the-full-stack-mqtt--redis)
  - [Running Demos (No MQTT/Redis)](#running-demos-no-mqttredis)
  - [Sensor Simulator](#sensor-simulator)
- [Project Structure](#project-structure)
- [Verification Scripts](#verification-scripts)
- [Roadmap](#roadmap)
- [License](#license)

---

## Design Philosophy

### The Core Problem

Current LLM-based systems are fundamentally **state machines** — question in, answer out. No input means no output. This extreme passivity cannot support the existence of a human-like entity.

### The Core Answer

We reshape the LLM from a passive Q&A machine into a **proactive living being** through three design principles:

1. **Embodied existence** — physical sensors and effectors establish a real boundary between the agent and its world
2. **Internal motivation** — digital hormones and homeostatic drives generate behavior even without external stimuli
3. **System stability** — we never modify LLM parameters (temperature, weights). Instead, we alter the **input environment** the LLM reasons within

### Theoretical Foundations

- **Jaak Panksepp's Affective Neuroscience** — mammalian brains contain seven innate emotional circuits: SEEKING, RAGE, FEAR, LUST, CARE, PANIC, PLAY. Our digital hormones map to these circuits.
- **Antonio Damasio's Somatic Marker Hypothesis** — decisions are not purely rational; bodily states act as biases that pre-filter options before conscious reasoning. Our somatic markers bias attention, not generate prompt text.
- **Abraham Maslow's Hierarchy of Needs** — lower-level needs (survival, safety) suppress higher-level ones (social, exploration). Our arbitration system enforces this priority.

---

## Architecture

### v0.1 Architecture (Reference)

v0.1 established the foundational layered design. The LLM was **trigger-based** — woken by a subconscious timer when drives exceeded thresholds, fed emotional context through generated prompt text, and constrained by a hard-coded intent pool.

```
[Physical World] → [Sensors] → [Perception Layer] → [Homeostatic Engine]
                                                          ↓ somatic markers (Prompt + Intents + Memory bias)
                                                     [LLM (Cortex)]
                                                          ↓ intent + text
                                                     [Action Execution] → [Effectors]
```

All modules communicated via **MQTT** (the "nervous system") with **Redis** for state persistence.

### v0.2 Architecture (Current)

v0.2 makes four fundamental shifts. The following diagram shows the new data flow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Animat Agent v0.2 Core                           │
│                                                                         │
│  [Sensors] ──→ SensorWindow (raw time-series, 5s window per modality)  │
│                   │                                                     │
│                   ├──→ Reflex Arcs (pain/startle → immediate motor)     │
│                   │    [bypasses LLM entirely]                          │
│                   │                                                     │
│                   └──→ StreamCortex ──────────────────────────────┐    │
│                         │  perpetual stream of consciousness       │    │
│                         │  LLM never sleeps                        │    │
│                         │                                          │    │
│                         ├─ Context injected:                       │    │
│                         │   · Raw sensor time-series (Phase 3)     │    │
│                         │   · Vector memory retrieval (Phase 4)    │    │
│                         │   · Body state summary                   │    │
│                         │                                          │    │
│                         ├─ Cognition modulated:                    │    │
│                         │   · AttentionModulator → logit bias     │    │
│                         │     (Phase 2: body state shifts token    │    │
│                         │      probabilities, no prompt injection) │    │
│                         │                                          │    │
│                         └─ Output: Thought stream                  │    │
│                              ↓                                     │    │
│                         ActionGate (3-layer gating)                │    │
│                              ↓                                     │    │
│                         ~95% suppressed    ~5% → external action   │    │
│                              ↓                     ↓               │    │
│                         ThoughtJournal    [Motor / Speech / LED]   │    │
│                              ↓                                     │    │
│                         VectorMemoryStore                          │    │
│                         (faiss + state-dependent retrieval)        │    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Innovations in v0.2

> Implementation status: Phases 1, 3, and 5 are integrated into the running agent. Phases 2 (AttentionModulator) and 4 (VectorMemoryStore) are **implemented as standalone modules and demo-verified, but not yet wired into the main runtime** — production currently uses the v0.1 homeostasis text-modulation and Redis memory. See the Roadmap table below for details.

#### 1. Perpetual Stream of Consciousness (Phase 1)

The LLM is no longer "woken up" by a timer. It runs continuously — producing internal thoughts at intervals modulated by drive level. The ActionGate determines which thoughts become external behavior: at low drive, ~95% are suppressed (humans don't speak every thought either). At high drive (danger, loneliness), the gate threshold drops.

**Key difference from v0.1**: Proactivity is not simulated by a timer. It emerges from the LLM's constant cognitive activity.

#### 2. Attention Modulation instead of Prompt Injection (Phase 2)

v0.1 injected emotion via prompt text: "You feel FEAR because your battery is at 8%." The LLM *played the role* of a scared entity.

v0.2 modulates the LLM's **token probability distribution** directly — applying logit bias to semantic token clusters (danger/social/explore/comfort/hostility). A fearful body state amplifies danger-related tokens and suppresses social/exploration tokens. The LLM doesn't "know" it's scared — it just finds itself thinking about danger more naturally.

This is analogous to how neurotransmitters modulate neural excitability: dopamine makes rewarding stimuli more salient; norepinephrine prioritizes threat detection. Neither tells you what to feel — they bias what you attend to.

**Key difference from v0.1**: The LLM's cognition is *stained* by bodily state, not *instructed* by textual description.

#### 3. Raw Sensor Data, Not Translated Labels (Phase 3)

v0.1 translated raw sensor readings into qualitative labels: `dP/dt > 0.8 → "pain"`. This translation layer compressed rich time-series data into a single categorical label, losing information about impact trajectory, decay patterns, and spatial context.

v0.2 maintains a **5-second rolling window of raw sensor frames** per modality (touch pressure/capacitance/position, audio dB/frequency, vision distance/face/gaze, internal battery/temperature/tilt). These are exposed directly as structured text in the LLM's context. The LLM reasons from raw patterns — "pressure spiked from 0.06 to 0.92 then decayed to 0.45 at a shifted position" — rather than being told "this was pain."

Reflex arcs are preserved for rapid danger responses (< 100ms), while the perception layer's qualitative mapping is downgraded to optional **attention hints** that suggest "look at the touch channel" without replacing the LLM's own judgment.

#### 4. State-Dependent Vector Memory (Phase 4)

v0.1 used Redis sorted sets with manual category-based weighting: "when afraid, multiply danger memories by 2.0."

v0.2 stores each memory with:
- A semantic embedding vector (1536-dim, from the local LLM)
- A **body state vector** capturing the 5 homeostatic dimensions at the time of memory formation

Retrieval combines semantic similarity (cosine distance in embedding space) with **state-dependent matching** (cosine distance in body state space). Memories formed in fearful states are naturally more retrievable in similar fearful states — no manual rules required. This is emergent behavior from vector space geometry.

---

## Quick Start

The fastest way to see Animat Agent v0.2 in action is the lifecycle integration demo — no MQTT or Redis needed. Dependencies are managed via `uv` (or a dedicated venv):

```bash
git clone https://github.com/digitopia-research/animat-agent.git
cd animat-agent

# Install dependencies (uv creates an isolated .venv automatically)
uv sync

# Or, with a plain venv + pip:
#   python3 -m venv .venv && . .venv/bin/activate
#   pip install -e .

# Download a model (Qwen2.5-1.5B Q4_K_M, ~1GB) to a path you control
mkdir -p models
curl -L -o models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  "https://hf-mirror.com/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

# Run the lifecycle demo (point ANIMAT_MODEL_PATH at your downloaded model)
ANIMAT_MODEL_PATH=models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  uv run python scripts/demo_animat_lifecycle.py
```

Note: the demo uses a small 1.5B model for speed. For production behavior use a 7B+ model via GPU (`ANIMAT_N_GPU_LAYERS=35`).

---

## Installation

### Prerequisites

| Dependency | Version | Purpose |
|:-----------|:--------|:--------|
| Python | ≥ 3.11 | Runtime |
| llama-cpp-python | ≥ 0.3 | Local LLM inference + embedding |
| faiss-cpu | ≥ 1.7 | Vector similarity search (Phase 4) |
| redis (aioredis) | ≥ 5.0 | State persistence & memory (v0.1 / optional in v0.2) |
| aiomqtt | ≥ 2.0 | MQTT nervous bus (v0.1 / optional in v0.2) |
| openai | ≥ 1.0 | Optional cloud LLM backend |
| python-dotenv | ≥ 1.0 | Environment configuration |

For development and testing:

| Dependency | Purpose |
|:-----------|:--------|
| pytest ≥ 8.0 | Test framework |
| pytest-asyncio ≥ 0.24 | Async test support |
| pytest-timeout ≥ 2.3 | Test timeout enforcement |
| ruff ≥ 0.9 | Linting |
| mypy ≥ 1.13 | Type checking |

### Dependencies

Install core dependencies:

```bash
pip install llama-cpp-python faiss-cpu redis openai python-dotenv aiomqtt \
  --break-system-packages
```

Install with dev dependencies:

```bash
pip install llama-cpp-python faiss-cpu redis openai python-dotenv aiomqtt \
  pytest pytest-asyncio pytest-timeout ruff mypy --break-system-packages
```

### Model Download

Animat Agent uses a local LLM via llama-cpp-python. The default model is Qwen2.5-1.5B-Instruct (Q4_K_M quantization, ~1GB).

**Automatic download** (recommended for China):

```bash
mkdir -p /opt/data/models
curl -L -o /opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  "https://hf-mirror.com/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
```

**Using HuggingFace**:

```bash
pip install huggingface_hub --break-system-packages
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/Qwen2.5-1.5B-Instruct-GGUF',
    'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf',
    local_dir='/opt/data/models')
"
```

**For production**, use a larger model (7B+) via GPU inference:

```bash
# Requires nvidia-container-toolkit on the host
docker run -d --gpus all -p 8080:8080 \
  -v /opt/data/models:/models \
  ghcr.io/ggerganov/llama.cpp:server \
  -m /models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 0.0.00 --port 8080
```

Then set the environment variable to point to the server:

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export ANIMAT_LLM_MODEL=qwen2.5-7b
```

### Configuration

Copy the environment template:

```bash
cp .env.example .env
```

Minimum configuration for local LLM:

```env
# No API key needed for local models
ANIMAT_MODEL_PATH=/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
ANIMAT_N_CTX=4096
ANIMAT_N_THREADS=8
```

For cloud API (OpenAI-compatible):

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
ANIMAT_LLM_MODEL=gpt-4o-mini
```

For GPU inference (future):

```env
ANIMAT_N_GPU_LAYERS=35  # offload 35 layers to GPU
```

Full v0.1 stack requires MQTT and Redis:

```env
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
REDIS_HOST=localhost
REDIS_PORT=6379
```

---

## Usage

### Running the Full Stack (MQTT + Redis)

The v0.1 full stack requires EMQX (MQTT broker) and Redis. Start the infrastructure:

```bash
# Start EMQX
docker run -d --name emqx -p 1883:1883 -p 18083:18083 emqx/emqx:latest

# Start Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run Animat Agent
cd animat-agent
python3 -m animat
```

### Running Demos (No MQTT/Redis)

All v0.2 verification scripts run standalone without any infrastructure:

| Script | What It Demonstrates | Duration |
|:-------|:---------------------|:---------|
| `scripts/demo_stream_cortex.py` | Continuous thought stream with gating. LLM produces thoughts every ~2 seconds; most are suppressed. | ~30s |
| `scripts/demo_attention_modulator.py` | Logit bias vs prompt injection. Fear state biases token probabilities toward danger words without telling the LLM "you are scared." | ~30s |
| `scripts/demo_sensor_window.py` | LLM reasons from raw pressure sequences. Identifies "impact" from `[0.05, 0.06, 0.92, 0.45]` and "caress" from `[0.03→0.11]` without translated labels. | ~30s |
| `scripts/demo_vector_memory.py` | State-dependent memory retrieval. Fearful body states naturally retrieve memories stored during fear. | ~20s |
| `scripts/demo_animat_lifecycle.py` | **Full integration**. All four phases combined across 7 lifecycle scenarios. | ~2min |

Run any demo:

```bash
python3 scripts/demo_stream_cortex.py
```

### Sensor Simulator

For testing with simulated hardware (the full MQTT + Redis stack must be running first):

```bash
# From the server where Animat Agent is running:
uv run python -m animat.simulator

# Or headless auto-stimulation mode (random touch/vision) for observing autonomous reactions:
ANIMAT_SIM_AUTO_STIM=1 uv run python -m animat.simulator
```

The simulator publishes mock sensor data (battery decay, random touch, human detection) to MQTT and provides an interactive input window. Free-text input is published to `animat/external/speech` (treated as human conversation). The following slash-commands inject specific stimuli:

```
Stimulus commands (published to the main Animat):
  /touch     — gentle caress (low pressure + capacitance) → CARE / comfort
  /hit       — sudden impact (sharp pressure spike) → pain → withdraw reflex (bypasses LLM)
  /approach  — a person approaches (1.5m, frontal gaze) → affiliation shift
  /leave     — the person leaves (10m) → loneliness / separation
  /noise     — loud bang (dB spike > 85) → startle → flinch reflex
  /battery N — set battery to N% (e.g. /battery 15) → survival crisis
  /temp N    — set body temperature to N°C (e.g. /temp 43) → overheating anxiety

Simulator control commands:
  /charge  — start charging
  /stop    — stop charging
  /reset   — reset battery to 100%
  /status  — show current state
  /help    — list commands
```

A browser-based alternative is `animat/simulator_web.html` — open it and connect to `ws://<host>:8083/mqtt` to watch the thought stream, send stimuli, and chat (see `docs/simulator_manual.md`).

---

## Project Structure

```
animat-agent/
├── animat/                          # Core package
│   ├── __init__.py                  # Package metadata (v0.1.0)
│   ├── __main__.py                  # python -m animat entry point
│   ├── main.py                      # CLI entry point + boot orchestrator
│   │
│   │  # ── v0.2 Core Modules ──
│   ├── stream_cortex.py             # [P1] Perpetual consciousness stream engine
│   │                                #   LLM never sleeps; continuous thought generation
│   ├── thought.py                   # [P1] Thought, ActionGate (3-layer gating),
│   │                                #   ThoughtJournal (thought archive)
│   ├── attention_modulator.py       # [P2] Token probability modulator (module done; NOT yet wired into main runtime)
│   │                                #   BodyState → logit bias for 5 semantic clusters
│   ├── sensor_window.py             # [P3] Raw sensor time-series window
│   │                                #   Exposes structured sensor data to LLM context
│   ├── vector_memory.py             # [P4] faiss-backed state-dependent memory (module done; NOT yet wired into main runtime)
│   │                                #   Embedding + body_state_vector retrieval
│   ├── llm_backend.py               # Local LLM wrapper (llama-cpp-python)
│   │                                #   Streaming + non-streaming, logit bias support
│   ├── cloud_llm.py                 # Cloud LLM backend (OpenAI-compatible API)
│   │
│   │  # ── v0.1 Modules (updated) ──
│   ├── config.py                    # Global configuration classes
│   │                                #   Topics, HomeostasisConfig, SharedConstants
│   ├── brain.py                     # Dual-process brain v2
│   │                                #   StreamCortex perpetual + simplified subconscious
│   ├── perception.py                # Sensorimotor layer v2
│   │                                #   Reflex arcs preserved; receptor mapping → attention hints
│   ├── homeostasis.py               # Digital endocrine system (updated)
│   │                                #   Mixed emotions, feeling decay, BodyState sync
│   ├── nervous.py                   # MQTT nervous bus (auto-reconnect)
│   ├── memory.py                    # Redis memory store (auto-reconnect + health check)
│   ├── action.py                    # Action execution (intent → motor commands)
│   ├── simulator.py                 # MQTT sensor simulator (interactive)
│   └── simulator_web.html           # Browser WebSocket control panel
│
├── scripts/                         # Verification & demo scripts
│   ├── demo_stream_cortex.py        # Phase 1: continuous thought stream
│   ├── demo_attention_modulator.py  # Phase 2: logit bias vs prompt injection
│   ├── demo_sensor_window.py        # Phase 3: raw sensor reasoning
│   ├── demo_vector_memory.py        # Phase 4: state-dependent memory retrieval
│   ├── demo_animat_lifecycle.py     # Phase 5: full integration (7 scenarios)
│   ├── simulator_v2.py              # Offline single-machine simulator (no MQTT/Redis)
│   └── start_server.sh              # One-click server launcher (EMQX + Redis + Animat)
│
├── tests/                           # Unit tests
│   ├── conftest.py                  # Shared fixtures (mock nervous/memory)
│   └── test_homeostasis.py          # 9 tests: drives, arbitration, decay, prompts
│
├── docs/
│   ├── architecture.md              # Architecture documentation (English)
│   ├── architecture_cn.md           # Architecture documentation (Chinese)
│   ├── ROADMAP_v0.2.md              # Full v0.2 roadmap (5 phases)
│   ├── simulator_manual.md          # Sensor simulator + web panel usage guide
│   ├── local_model_how_it_works.md  # Local LLM deployment & inference notes
│   └── mosquitto.conf.example       # (legacy) native mosquitto config reference
│
├── pyproject.toml                   # Project metadata + dependencies
├── .env.example                     # Environment template
├── .python-version                  # Python version pin
├── README.md                        # This file
├── README_cn.md                     # Chinese README (v0.1)
└── LICENSE                          # MIT
```

---

## Verification Scripts

All scripts are standalone and require no MQTT/Redis. They load the local model once and run the scenario.

### Phase 1: Stream of Consciousness

```bash
python3 scripts/demo_stream_cortex.py
```

Expected output: LLM produces internal thoughts continuously. At normal drive levels, most are suppressed (tagged `💭`). At high drive (low battery), activation threshold drops and thoughts trigger external behavior (`🗣️`).

### Phase 2: Attention Modulation

```bash
python3 scripts/demo_attention_modulator.py
```

Compares three approaches to the same "battery at 8%" scenario:
- **A (v0.1)**: Emotion prompt injection → LLM acts out "I feel very uneasy"
- **B (v0.2)**: Neutral prompt + logit bias → LLM's token probabilities shifted toward danger
- **C (baseline)**: Neutral prompt, no bias → polite rational response

### Phase 3: Raw Sensor Reasoning

```bash
python3 scripts/demo_sensor_window.py
```

Shows how the LLM reasons from raw sensor time-series:
- Pressure sequence `[0.05, 0.06, 0.92, 0.45, 0.15]` → identifies impact event
- Pressure sequence `[0.03, 0.05, 0.07, 0.09, 0.11]` + rising capacitance → identifies caress

### Phase 4: State-Dependent Memory

```bash
python3 scripts/demo_vector_memory.py
```

Demonstrates that memories stored during fearful body states are more retrievable when the agent is in a similar fearful state — without explicit rule-based weighting.

### Phase 5: Full Lifecycle Integration

```bash
python3 scripts/demo_animat_lifecycle.py
```

Runs 7 scenarios spanning the full lifecycle:
1. Normal state → spontaneous thoughts, mostly suppressed
2. Battery discharge → behavior shifts as drive increases
3. Battery at 15% + human asks for joke → should refuse
4. Long solitude → detects approaching human
5. Sudden impact → reflex arc triggers immediately
6. Gentle touch (full battery) → expresses comfort
7. Battery at 30% + gentle touch → mixed emotion

---

## Roadmap

See [docs/ROADMAP_v0.2.md](docs/ROADMAP_v0.2.md) for the full v0.2 development roadmap.

| Phase | Status | Description |
|:------|:-------|:------------|
| Phase 1 | Done (integrated) | Perpetual consciousness stream — LLM never sleeps; wired into `brain.py`/`stream_cortex.py` |
| Phase 2 | Implemented, not integrated | Attention modulation — body state shifts token probabilities. Module lives in `attention_modulator.py` and is validated by `demo_attention_modulator.py`, but `brain.py` does **not** yet call it (production still uses v0.1 homeostasis text-modulation) |
| Phase 3 | Done (integrated) | Raw sensor window — LLM reasons from time-series data; wired into `brain.py` |
| Phase 4 | Implemented, not integrated | Vector memory — state-dependent semantic retrieval. Module lives in `vector_memory.py` and is validated by `demo_vector_memory.py`, but the main runtime still uses `memory.py` (Redis) |
| Phase 5 | Done (integrated) | Integration & lifecycle validation — StreamCortex + SensorWindow + ActionGate + MQTT/Redis |

> Note: Phases 2 and 4 are **implemented and demo-verified**, but their modules are **not yet wired into the main runtime**. The running agent currently relies on the v0.1 homeostasis (text modulation) and Redis memory. Integration milestones are tracked in `docs/ROADMAP_v0.2.md`.

### Future Directions

- **GPU inference**: Use llama-server on the Docker host for 7B+ models
- **Sensor hardware integration**: Connect real touch/audio/vision sensors via MQTT
- **Multi-agent**: Multiple Animat instances interacting with each other
- **Model fine-tuning**: LoRA adapters for stronger refusal behavior in dangerous states
- **Memory consolidation**: Dream-like replay of high-emotion memories during idle periods

---

## License

MIT

Copyright (c) Digitopia Robotics Research
