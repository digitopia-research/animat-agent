# Animat Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LLM](https://img.shields.io/badge/LLM-powered-8A2BE2.svg)](https://github.com/your-repo)

---

## 🧬 *The agent that lives.*

基于大语言模型(LLM)的具身"人工系统"框架，从被动问答到主动存在的理论构建。

## 它是什么

Animat Agent 不是聊天机器人，而是一个持续运行的"生命体"。它通过传感器感知物理世界，拥有内驱力和情感，能在无人交互时主动行动——因为无聊而探索，因为孤独而寻求陪伴，因为电量低而焦虑求救。

核心设计原则：

- **双进程脑**：潜意识进程高频运行（稳态衰减、反射响应），意识进程触发式唤醒 LLM
- **数字内分泌系统**：基于马斯洛层次的内稳态指标，驱动情感合成与马斯洛仲裁
- **躯体标记**：不修改 LLM 参数，而是通过动态 Prompt、意图约束、记忆偏向三条通路安全影响推理

理论基石：潘克塞普情感神经科学、达马西奥躯体标记假说、马斯洛需求层次理论。

## 架构概览

```
[物理世界]
   │  ▲
[传感器/效应器]
   │  │
   ▼  │
┌──────────────────────────────────────────────────┐
│  感知运动层 (perception.py)                       │
│  · 感受器映射: 原始数据 → 心理感受               │
│  · 反射弧: 危险 → 绕过大脑的快速响应              │
└──────────────────────────────────────────────────┘
   │ 感受信号                        ▲ 动作指令
   ▼                                │
┌──────────────────────────────────────────────────┐
│  内稳态引擎 (homeostasis.py) — 边缘系统          │
│  · 五大稳态指标 → 驱力计算                       │
│  · 情感合成 → 马斯洛仲裁                         │
│  · 躯体标记生成 (Prompt / 意图空间 / 记忆偏向)    │
└──────────────────────────────────────────────────┘
   │ 躯体标记
   ▼
┌──────────────────────────────────────────────────┐
│  意识推理层 (stream_cortex.py) — 新皮层 (LLM)     │
│  · StreamCortex 持续意识流 (Phase 1, 已集成):     │
│    LLM 永不休眠, 持续生成思维与意图               │
│  · 在约束的上下文、意图池、记忆中推理             │
│  · 输出: 意图 + 对话文本                         │
└──────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────┐
│  行为执行层 (action.py)                           │
│  · 意图 → 动作树 → 效应器指令                    │
└──────────────────────────────────────────────────┘
```

**MQTT** 作为神经系统连接各层，**Redis** 作为记忆与状态存储。

## 快速开始

### 前置条件

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器
- MQTT Broker（如 [EMQX](https://www.emqx.io/)）
- Redis

### 安装 EMQX (MQTT Broker)

EMQX 是高性能 MQTT Broker，提供 Dashboard 便于调试。

```bash
# Docker 一键启动 (推荐)
docker run -d --name emqx \
  -p 1883:1883 \
  -p 8083:8083 \
  -p 8084:8084 \
  -p 8883:8883 \
  -p 18083:18083 \
  emqx/emqx:latest

# 访问 Dashboard: http://localhost:18083
# 默认账号: admin / public
```

<details>
<summary>其他安装方式</summary>

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
# 参见 https://www.emqx.io/downloads 安装对应发行版
```
</details>

### 安装 Redis

```bash
# Docker 一键启动 (推荐)
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# 验证连接
docker exec redis redis-cli ping
# → PONG
```

<details>
<summary>其他安装方式</summary>

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Windows:**
1. 从 https://github.com/microsoftarchive/redis/releases 下载安装包
2. 或通过 WSL 使用 Linux 版本

**Linux (APT):**
```bash
sudo apt install redis-server
sudo systemctl enable --now redis-server
```
</details>

### 安装项目

```bash
# 克隆项目
git clone https://github.com/digitopia-research/animat-agent.git
cd animat-agent

# 安装依赖
uv sync
```

### 配置

```bash
# 复制环境变量模板并编辑
cp .env.example .env
```

编辑 `.env`，至少需要配置 `OPENAI_API_KEY`。如使用其他兼容接口，修改 `OPENAI_BASE_URL` 和 `ANIMAT_LLM_MODEL`。

### 启动

```bash
# 启动 Animat Agent
uv run animat

# 或通过模块方式
uv run python -m animat
```

### 传感器模拟器

无实体硬件时，可使用内置模拟器进行开发测试（需先启动 EMQX + Redis）：

```bash
# 交互式模拟器
uv run python -m animat.simulator

# 后台自动刺激模式 (随机触觉/视觉, 用于观察自主反应)
ANIMAT_SIM_AUTO_STIM=1 uv run python -m animat.simulator
```

模拟器会向 MQTT 发布模拟传感器数据（电量衰减、随机触碰、人体检测等），并提供交互式输入窗口与 Animat 对话。自由文本作为人类语音（`animat/external/speech`）输入；下列斜杠命令注入特定刺激：

```
刺激命令 (发布给主 Animat):
  /touch     — 轻抚 (低压+电容) → CARE/安抚
  /hit       — 撞击 (压力骤升) → 疼痛 → 退缩反射 (绕过 LLM)
  /approach  — 有人靠近 (1.5m) → 归属偏移
  /leave     — 人离开 (10m) → 孤独/分离
  /noise     — 巨响 (dB>85) → 惊吓 → 闪避反射
  /battery N — 设置电量 N% (如 /battery 15) → 生存危机
  /temp N    — 设置体温 N°C (如 /temp 43) → 过热焦虑

模拟器控制命令:
  /charge  — 开始充电
  /stop    — 停止充电
  /reset   — 电量重置为 100%
  /status  — 显示当前状态
  /help    — 列出命令
```

浏览器版控制面板见 `animat/simulator_web.html`（连接 `ws://<host>:8083/mqtt`，详见 `docs/simulator_manual.md`）。

## 项目结构

```
animat-agent/
├── animat/
│   ├── config.py          # 全局配置 (MQTT Topics、稳态阈值、情感优先级)
│   ├── nervous.py         # MQTT 神经总线 (自动重连)
│   ├── perception.py      # 感知运动层 (感受器映射 + 反射弧)
│   ├── homeostasis.py     # 内稳态引擎 (驱力 → 情感合成 → 马斯洛仲裁 → 躯体标记)
│   ├── brain.py           # 双进程脑 (StreamCortex 持续层 + 潜意识循环 + 心跳)
│   ├── stream_cortex.py   # [P1] 恒常意识流引擎 (LLM 永不休眠, 已集成)
│   ├── thought.py         # [P1] Thought、ActionGate 三层门控、ThoughtJournal (已集成)
│   ├── attention_modulator.py # [P2] 注意力调制器 (模块已完成, 未接入主运行时)
│   ├── sensor_window.py   # [P3] 原始传感时序窗口 (已集成)
│   ├── vector_memory.py   # [P4] faiss 状态依赖记忆 (模块已完成, 未接入主运行时)
│   ├── llm_backend.py     # 本地 LLM 封装 (llama-cpp-python)
│   ├── cloud_llm.py       # 云端 LLM backend (OpenAI 兼容)
│   ├── memory.py          # Redis 记忆系统 (情景/语义记忆)
│   ├── action.py          # 行为执行层 (意图 → 动作树)
│   ├── main.py            # 启动入口
│   ├── simulator.py       # MQTT 传感器模拟器 (交互式)
│   └── simulator_web.html # 浏览器 WebSocket 控制面板
├── scripts/               # demo 与工具脚本 (含 simulator_v2.py, start_server.sh)
├── tests/                 # 单元测试 (conftest.py, test_homeostasis.py)
├── docs/                  # 设计/部署文档 (architecture, ROADMAP_v0.2, simulator_manual…)
├── pyproject.toml
├── .python-version
├── .env.example
├── README.md              # 英文文档 (含 Roadmap 与各 Phase 集成状态)
├── README_cn.md           # 本文件
└── LICENSE
```

> 实现状态：Phase 1 / 3 / 5 已集成到运行中的 agent；Phase 2（AttentionModulator）与
> Phase 4（VectorMemoryStore）已完成独立模块并经 demo 验证，但主运行时暂未接线（当前生产
> 仍使用 v0.1 稳态文本调制 + Redis 记忆）。详见 `README.md` 的 Roadmap 表与 `docs/ROADMAP_v0.2.md`。

## MQTT Topic 通路

| Topic | 方向 | 说明 |
|:------|:-----|:-----|
| `animat/sensor/touch` | 传感器 → 感知层 | 触觉原始数据 |
| `animat/sensor/audio` | 传感器 → 感知层 | 听觉原始数据 |
| `animat/sensor/vision` | 传感器 → 感知层 | 视觉原始数据 |
| `animat/sensor/interoception` | 传感器 → 感知层 | 内感原始数据 |
| `animat/feeling` | 感知层 → 边缘系统 | 感受事件 |
| `animat/reflex` | 感知层 → 效应器 | 反射弧（绕过 LLM） |
| `animat/emotion` | 边缘系统内部 | 情感回路激活 |
| `animat/cortex/prompt` | 边缘系统 → 新皮层 | 动态 System Prompt |
| `animat/cortex/intents` | 边缘系统 → 新皮层 | 意图选项空间约束 |
| `animat/cortex/memory_bias` | 边缘系统 → 新皮层 | 记忆检索偏向 |
| `animat/cortex/input` | → 新皮层 | 唤醒 LLM |
| `animat/cortex/output` | 新皮层 → | LLM 推理结果 |
| `animat/motor` | 执行层 → 效应器 | 底层动作指令 |
| `animat/external/speech` | 外部 → | 人类语音输入 |

## License

MIT
