# Animat Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LLM](https://img.shields.io/badge/LLM-powered-8A2BE2.svg)](https://github.com/your-repo)

---

## 🧬 *The agent that lives.*

An LLM-based embodied "artificial organism" framework, from passive Q&A to proactive existence.

## What It Is

Animat Agent is not a chatbot — it's a continuously running "living being." It senses the physical world through sensors, possesses drives and emotions, and can act autonomously even without human interaction — exploring out of boredom, seeking companionship out of loneliness, or crying for help when battery runs low.

Core design principles:

- **Dual-Process Brain**: A subconscious process running at high frequency (homeostatic decay, reflex responses) and a conscious process triggered to invoke the LLM
- **Digital Endocrine System**: Maslow-hierarchy-based homeostatic vitals that drive emotion synthesis and Maslow arbitration
- **Somatic Markers**: Safely influences LLM reasoning through three pathways — dynamic prompts, intent constraints, and memory biases — without modifying model parameters

Theoretical foundations: Panksepp's affective neuroscience, Damasio's somatic marker hypothesis, Maslow's hierarchy of needs.

## Architecture Overview

```
[Physical World]
   │  ▲
[Sensors / Effectors]
   │  │
   ▼  │
┌──────────────────────────────────────────────────┐
│  Sensorimotor Layer (perception.py)              │
│  · Receptor mapping: raw data → feelings        │
│  · Reflex arcs: danger → fast response bypassing │
│    the brain                                     │
└──────────────────────────────────────────────────┘
   │ Feeling signals                 ▲ Motor commands
   ▼                                │
┌──────────────────────────────────────────────────┐
│  Homeostatic Engine (homeostasis.py) — Limbic    │
│  · Five vital signs → drive computation          │
│  · Emotion synthesis → Maslow arbitration        │
│  · Somatic marker generation (Prompt / Intent    │
│    space / Memory bias)                           │
└──────────────────────────────────────────────────┘
   │ Somatic markers
   ▼
┌──────────────────────────────────────────────────┐
│  Conscious Reasoning (cortex.py) — Neocortex     │
│  · Reasoning within constrained context, intent  │
│    pool, and memory                              │
│  · Output: intent + dialogue text                │
└──────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────┐
│  Action Execution (action.py)                    │
│  · Intent → action tree → effector commands      │
└──────────────────────────────────────────────────┘
```

**MQTT** serves as the nervous system connecting all layers, **Redis** as memory and state storage.

## Quick Start

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- MQTT Broker (e.g., [EMQX](https://www.emqx.io/))
- Redis

### Install EMQX (MQTT Broker)

EMQX is a high-performance MQTT Broker with a built-in Dashboard for debugging.

```bash
# Docker one-click start (recommended)
docker run -d --name emqx \
  -p 1883:1883 \
  -p 8083:8083 \
  -p 8084:8084 \
  -p 8883:8883 \
  -p 18083:18083 \
  emqx/emqx:latest

# Dashboard: http://localhost:18083
# Default credentials: admin / public
```

<details>
<summary>Alternative installation methods</summary>

**macOS (Homebrew):**
```bash
brew install emqx
emqx start
```

**Windows (Chocolatey):**
```bash
choco install emqx
emqx start
```

**Linux (APT):**
```bash
# See https://www.emqx.io/downloads for distribution-specific instructions
```
</details>

### Install Redis

```bash
# Docker one-click start (recommended)
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Verify connection
docker exec redis redis-cli ping
# → PONG
```

<details>
<summary>Alternative installation methods</summary>

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Windows:**
1. Download installer from https://github.com/microsoftarchive/redis/releases
2. Or use the Linux version via WSL

**Linux (APT):**
```bash
sudo apt install redis-server
sudo systemctl enable --now redis-server
```
</details>

### Install Project

```bash
# Clone the repository
git clone https://github.com/digitopia-research/animat-agent.git
cd animat-agent

# Install dependencies
uv sync
```

### Configuration

```bash
# Copy the environment template and edit
cp .env.example .env
```

Edit `.env` — at minimum, configure `OPENAI_API_KEY`. To use a compatible API endpoint, also set `OPENAI_BASE_URL` and `ANIMAT_LLM_MODEL`.

### Launch

```bash
# Start Animat Agent
uv run animat

# Or run as a module
uv run python -m animat
```

### Sensor Simulator

When no physical hardware is available, use the built-in simulator for development and testing:

```bash
uv run python -m animat.simulator
```

The simulator publishes mock sensor data to MQTT (battery decay, random touch events, human detection, etc.) and provides an interactive input window to converse with the Animat.

## Project Structure

```
animat-agent/
├── animat/
│   ├── config.py          # Global config (MQTT topics, homeostatic thresholds, emotion priorities)
│   ├── nervous.py         # MQTT nervous bus
│   ├── perception.py      # Sensorimotor layer (receptor mapping + reflex arcs)
│   ├── homeostasis.py     # Homeostatic engine (drives → emotion synthesis → Maslow arbitration → somatic markers)
│   ├── cortex.py          # Conscious reasoning (LLM)
│   ├── memory.py          # Redis memory system (episodic / semantic memory)
│   ├── action.py          # Action execution (intent → action tree)
│   ├── brain.py           # Dual-process brain (subconscious loop + conscious triggers + heartbeat)
│   ├── main.py            # Entry point
│   └── simulator.py       # Sensor simulator
├── pyproject.toml
├── .python-version
├── .env.example
└── 设计框架.md             # Full theoretical design document (Chinese)
```

## MQTT Topic Pathways

| Topic | Direction | Description |
|:------|:----------|:------------|
| `animat/sensor/touch` | Sensor → Perception | Raw touch data |
| `animat/sensor/audio` | Sensor → Perception | Raw audio data |
| `animat/sensor/vision` | Sensor → Perception | Raw vision data |
| `animat/sensor/interoception` | Sensor → Perception | Raw interoception data |
| `animat/feeling` | Perception → Limbic | Feeling events |
| `animat/reflex` | Perception → Effector | Reflex arcs (bypassing LLM) |
| `animat/emotion` | Within Limbic System | Emotion circuit activation |
| `animat/cortex/prompt` | Limbic → Neocortex | Dynamic System Prompt |
| `animat/cortex/intents` | Limbic → Neocortex | Intent option space constraints |
| `animat/cortex/memory_bias` | Limbic → Neocortex | Memory retrieval bias |
| `animat/cortex/input` | → Neocortex | Wake LLM |
| `animat/cortex/output` | Neocortex → | LLM reasoning results |
| `animat/motor` | Action → Effector | Low-level motor commands |
| `animat/external/speech` | External → | Human speech input |

## License

MIT
