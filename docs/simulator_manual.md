# Animat 传感器模拟器使用手册

> 适用模块：`animat/simulator.py`
> 适用版本：Animat Agent v2（MQTT + Redis 双进程架构）

---

## 1. 这是什么

`animat.simulator` 是一个**无实体硬件时的开发测试工具**。它本身**不加载 LLM**，而是：

- 作为独立的 MQTT 客户端，向 `animat/sensor/*` 主题发布**传感器数据**（模拟触觉、视觉、听觉、内感）；
- 订阅 `animat/motor`、`animat/cortex/output`、`animat/heartbeat`，把 Animat 的动作指令与思维流**打印到终端**，让你"看见"它在做什么；
- 提供一个交互式终端，让你用自然语言对话，或用手动命令**主动刺激** Animat，观察它的反应。

它必须和主进程 `uv run animat` 配合使用——模拟器是 Animat 的"身体与外界"，主进程是 Animat 的"大脑"。

---

## 2. 启动方式

模拟器依赖 `aiomqtt`、`redis` 等包，**必须用 `uv run` 启动**（它会自动使用项目虚拟环境，否则会出现 `ModuleNotFoundError: aiomqtt`）：

```bash
# 终端 1：启动主 agent（大脑）
uv run animat

# 终端 2：启动模拟器（身体与外界）
cd ~/Workspace/animat-agent
uv run python -m animat.simulator
```

启动后终端 2 会显示：

```
Sensor simulator started (Ctrl+C to stop)
Auto stimulus: OFF (touch/vision only via /touch /hit /approach /leave). Interoception (battery) still auto-published.
Type /help for commands. Special sim commands: /charge, /stop, /reset, /status
You>
```

> ⚠️ **不要**用 `python -m animat.simulator`（系统/conda 的 python 通常没装依赖）。
> ⚠️ 如果 broker 不在本机，用环境变量指定：`MQTT_BROKER_HOST=192.168.x.x uv run python -m animat.simulator`

---

## 3. 两种刺激模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **手动模式（默认）** | 触觉/视觉**不会**自动随机发送，只由你手动命令发起 | 调试、观察单次刺激→反应 |
| **自动模式** | 触觉/视觉按随机节奏自动发送（旧行为） | 长期无人值守跑、压力测试 |

开启自动模式：

```bash
ANIMAT_SIM_AUTO_STIM=1 uv run python -m animat.simulator
```

无论哪种模式，**内感（电量/体温）始终自动上报**——这是必须的，否则主进程的电量指标会卡死。

---

## 4. 交互命令一览

在 `You>` 提示符下输入。命令分三类：

### 4.1 对话（直接输入文本）

不与 `/` 开头的文本会被当作对话，发布到 `animat/external/speech`，主进程以 `score=0.85` 豁免门控直接回复：

```
You> 你好，你在想什么？
```

### 4.2 模拟器自身控制命令

| 命令 | 作用 |
|------|------|
| `/charge` | 开始充电（模拟器内感循环改为电量上升，主进程收到后 Animat 进入充电行为） |
| `/stop` | 停止充电 |
| `/reset` | 电量重置为满（100%） |
| `/status` | 显示当前模拟器电量与充电状态 |
| `/help` | 显示所有命令帮助 |

### 4.3 环境刺激命令（发布到主进程 Animat）

这些命令向对应的 `animat/sensor/*` 主题发送数据，用来主动刺激 Animat：

| 命令 | 发送内容 | 主进程预期反应 |
|------|----------|----------------|
| `/touch` | 低压 + 电容（轻柔接触） | 触发 `caress` 感受，表达"被抚摸"的温暖/好奇 |
| `/hit` | 双帧压力突变（低→高） | 触发 `pain` 感受 + `withdraw` 反射（缩回），LED 变红闪烁，表达被撞的不适 |
| `/approach` | 有人靠近 1.5m、正面凝视 3s | Animat 感知"有人靠近并注视"，倾向社交/警惕 |
| `/leave` | 人离开（距离 10m） | 社交线索消失，可能转入独处/探索 |
| `/noise` | 巨响（dB 突变 >85） | 触发 `startle` 感受 + `flinch` 反射（惊吓） |
| `/battery N` | 设置电量到 N%（如 `/battery 15`） | 下一内感周期生效，低电量会触发恐惧/寻找充电 |
| `/temp N` | 设置体温到 N°C（如 `/temp 43`） | 下一内感周期生效，高温会触发过热焦虑/失衡 |

> **关键机制提示**：
> - `/hit` 必须发"双帧压力突变"才能让主进程感知到痛——因为痛觉依赖压力**变化率（dP/dt）**，单发一帧高压不够。模拟器已内置该双帧逻辑，直接打 `/hit` 即可。
> - 表现反应（"被抚摸""被撞"）由主进程的 LLM 从 SensorWindow 原始序列自行判断，因此措辞会自然变化，不保证逐字一致。

---

## 5. 内感（电量/体温）行为说明

模拟器的内感循环每 2 秒向 `animat/sensor/interoception` 发布一次：

- **电量**：
  - 未充电时缓慢放电（默认 `0.05%/周期`，约 66 分钟从满电到 0）；
  - 充电时按 `3.0%/周期` 上升，满 100% 自动停止；
  - 可用 `/battery N`、`/charge`、`/reset` 干预。
- **体温**：
  - 默认恒定 **37°C**，每周期缓慢回归正常值，**不会随机乱跳**（旧版本会随机跳到 42°C 导致无端"过热焦虑"，已修复）；
  - 充电时极缓上升（模拟发热），停充后回落；
  - 想测试过热场景用 `/temp 43` 手动设置。
- **倾角（imu_tilt_deg）**：每周期围绕 0° 高斯抖动（±2°），一般不会触发失衡反射。

### 可调环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ANIMAT_SIM_BATTERY` | `100` | 初始电量（%） |
| `ANIMAT_SIM_DISCHARGE` | `0.05` | 放电速率（%/周期），设更小（如 `0.02`）更耐用 |
| `ANIMAT_SIM_AUTO_STIM` | `0` | 设为 `1` 开启自动随机触觉/视觉 |
| `MQTT_BROKER_HOST` | `localhost` | MQTT broker 地址 |
| `MQTT_BROKER_PORT` | `1883` | MQTT broker 端口 |

示例：

```bash
# 大电池 + 极慢放电 + 手动刺激
ANIMAT_SIM_BATTERY=100 ANIMAT_SIM_DISCHARGE=0.02 uv run python -m animat.simulator
```

---

## 6. 输出解读

终端 2 会打印三类信息：

```
[TOUCH] Caress detected              ← 模拟器发出/收到触觉事件
[VISION] Human detected at 1.27m     ← 视觉事件（仅自动模式出现）
[INTERO] ⚡ CHARGING battery=85.3% temp=37.2°C  ← 内感周期（debug 级）

🗣️  Animat says: <文本>             ← 主进程 emit 的 speak 意图（Agent 说的话）
🔧 Motor: led {'color': 'red', 'pattern': 'flash'}  ← 主进程 emit 的动作指令
🧠 Intent: move | Text: <文本>      ← 主进程 cortex output（思考后的意图+文本）
```

当 Animat 说话或执行动作时，说明对应思维**通过了 ActionGate 门控**并被输出；被门控拦截的思维不会出现在 `🗣️`/`🧠` 行。

---

## 7. 典型调试流程

1. 启动主进程 `uv run animat`；
2. 启动模拟器 `uv run python -m animat.simulator`（默认手动模式）；
3. 想测社交：连续 `/approach` + `/touch`，观察 Animat 是否表达被关注/温暖；
4. 想测痛觉反射：打 `/hit`，应看到 `🔧 Motor: move back` + `expression pain` + LED 红闪；
5. 想测低电量行为：先 `/charge` 充满，再 `/battery 15`，观察是否转入"寻找充电站"叙事；
6. 想测过热：`/temp 43`，观察是否出现"体温过高/失衡"焦虑；
7. 想测对话：`You> 你叫什么名字` —— 主进程以外部语音豁免门控直接回复。

---

## 8. 常见问题

**Q：启动报错 `ModuleNotFoundError: No module named 'aiomqtt'`**
A：用了系统 python。改用 `uv run python -m animat.simulator`。

**Q：输入中文时崩溃 `UnicodeDecodeError`**
A：终端 stdin 出现了非法 UTF-8 字节（如输入法混合序列）。模拟器已改为二进制读取 + 容错解码，升级到最新代码后不会再因此退出。同时确保终端 locale 为 UTF-8（`export LANG=en_US.UTF-8`）。

**Q：模拟器启动了，但 Animat 没反应**
A：检查主进程是否连上同一个 MQTT broker；确认模拟器日志出现 `Sensor simulator started`，且主进程日志出现 `StreamCortex consciousness stream started`。跨机时需用 `MQTT_BROKER_HOST` 指定 broker 地址。

**Q：为什么 `/touch` 后 Animat 说"有人轻柔碰了我一下"，但 `/hit` 后没立刻说话？**
A：`/hit` 触发的是**反射弧**（绕过 LLM 的 `withdraw` 动作），LLM 的"被撞"表达会出现在下一轮自发思维中，而非即时对话。这是设计上的"反射快于意识"。

**Q：重启主进程后状态还在？**
A：正常。主进程把稳态指标持久化到 Redis，重启时 `State restored` 读回，Animat 会延续上次叙事。想干净重启：`redis-cli flushall` 后重启主进程。

**Q：如何清空 Animat 在 Redis 里的记忆（稳态/名字/对话历史等）？**
A：按彻底程度从弱到强：
- **只清核心状态（推荐）**：`uv run animat --reset`。会清空稳态指标、情感、对话历史、语义记忆（含记住的"名字"、对话历史），但保留情景记忆与感受时间线。
- **全清 Redis**：`redis-cli flushall`。清空 Redis 全部库所有 key，Animat 所有记忆一次性归零，重启即"全新出生"。
- **只清 Animat 命名空间**：`redis-cli --scan --pattern 'animat:*' | xargs -r redis-cli del`（PowerShell 用 `redis-cli --scan --pattern 'animat:*' | ForEach-Object { redis-cli del $_ }`），精准删除 `animat:` 前缀的 key，不动其他服务的数据。

> 注意：模拟器本地维护的电量/体温是浏览器/进程内存，**不在 Redis**，重启模拟器本身就归零，无需清 Redis。

**Q：我和 Animat 的对话，会影响它"没人在说话时"的内心独白吗？**
>A：会影响，但分两条路径，行为不同：
>- **对话回复**：你每次发话，主进程都会把这条对话与 Animat 的回复写入"最近对话历史"（最多保留 6 条，持久化在 Redis 的语义记忆 `conversation_history`）。
>- **自发思维（内心独白）**：当没有外部刺激、Animat 进入自发思考时，`_think_spontaneous` 会把最近的对话历史作为 `conversation_context` 注入 LLM 提示词（"## 最近和对方的对话"），因此它**会引用你们刚才聊过的内容**（比如你说过"我是李先生""我在实验室"）。
>- **身份记忆（记住名字/地点）**：对话中出现的"对方是谁、在哪儿"这类信息，会被 LLM 抽取成结构化事实（JSON），存进语义记忆的 `self_info` 并持久化到 Redis。下次对话或自发独白时，`_recall_self_info` 会把它拼回系统提示词，所以它能"记住李先生""在实验室"，而不是每次都问。
>
>如果你发现独白没引用对话或没记住名字：先确认对话历史已写入（见上方"如何清空记忆"，`animat:*` 内应有 `CORTEX_HISTORY`/`MEMORY_SEMANTIC`）；若仍不生效，多为本地 LLM 抽取 JSON 不稳定导致静默跳过，属已知边界，可重新聊一次让信息再次被抽取。

---

## 8.1 网页调试模拟客户端（MQTT over WebSocket）

除终端模拟器外，还提供了一个**纯网页版**调试客户端 `animat/simulator_web.html`，功能与终端版对等，适合无 ssh/无终端的场景，或直接用浏览器交互。

### 启用步骤

1. **让 MQTT broker 开启 WebSocket 监听（端口 8083，路径 `/mqtt`）**。
   终端模拟器走 TCP 1883，网页版必须走 WebSocket。有三种方式：

   ```bash
   # 方式 A：本地 mosquitto（仓库提供示例配置 docs/mosquitto.conf.example）
   mosquitto -c docs/mosquitto.conf.example -v

   # 方式 B：Docker（eclipse-mosquitto 镜像默认即开启 1883/8083）
   docker run -it --rm -p 1883:1883 -p 8083:8083 eclipse-mosquitto

   # 方式 C：EMQX（本项目实际部署方式，推荐）
   # EMQX 默认已开启 8083 WebSocket 监听（路径 /mqtt），无需额外配置：
   docker run -d --name emqx -p 1883:1883 -p 18083:18083 -p 8083:8083 -p 8084:8084 --shm-size=1g emqx/emqx:latest
   ```

   > 当前服务器采用**方式 C（EMQX）**：`1883`（MQTT）、`18083`（管理面板）、`8083`（WebSocket）、`8084`（WSS）均已映射。浏览器面板连 `ws://<server-ip>:8083/mqtt` 即可。

2. **用浏览器打开 `animat/simulator_web.html`**（双击即可，无需起服务器）。
   页面顶部默认 broker 地址为 `ws://localhost:8083/mqtt`，与 `.env` 的 `MQTT_TRANSPORT=websockets` / `MQTT_WS_PATH=/mqtt` 对应。跨机时改成 `ws://<broker-ip>:8083/mqtt`。

3. 点「连接」。状态点变绿即成功。

### 网页版功能对照

| 区域 | 功能 | 等效命令 |
|------|------|----------|
| 左侧「Animat 内部状态 (实时)」 | 订阅主进程 `animat/state` 广播，显示稳态/情绪/驱力/激活率/记住的对方/最近对话 | 实时 |
| 左侧「我发送的内感 (本地模拟)」 | 浏览器本地维护的电量/体温/充电 | `/status` |
| 「环境刺激」5 个按钮 | 抚摸/撞击/靠近/离开/巨响 | `/touch /hit /approach /leave /noise` |
| 电量/体温滑块 + 应用 | 设置内感初值，下周期生效 | `/battery N` `/temp N` |
| 「模拟器控制」4 按钮 | 充/停/重置/状态 | `/charge /stop /reset /status` |
| 主区左「与 Animat 对话」 | 你与 Animat 的**对白**（你说 / 它说出口的话） | 普通文本输入 |
| 主区右「Animat 内心思考」 | 订阅 `animat/cortex/thought`，展示**每一次内部思维流**（大多不说出口，仅"说出口"的标绿） | 实时 |
| 底部输入框 | 直接对话 | 普通文本输入 |
| 主日志区 | 打印 Animat 的 🗣️说话 / 🔧Motor / 🧠Intent | 终端日志 |

> **内部状态实时面板**：主进程 `brain.py` 的 heartbeat 循环每秒向 `animat/state` 广播一份完整快照（见下方 §8.2 字段表）。网页端订阅后，左侧「Animat 内部状态」面板会即时刷新——这是**主进程真实内部状态**，区别于下方「我发送的内感（本地模拟）」（那是浏览器自己算的、即将发布给主进程的电量/体温）。
> 网页版的内感（电量/体温）循环在**浏览器本地**维护并自动发布到 `animat/sensor/interoception`，与终端模拟器行为一致。拼音乱码拦截同样内置。
> 网页版不依赖 `uv`、不加载 LLM，只是一个 MQTT WebSocket 客户端，需配合主进程 `uv run animat` 使用。

### 内部状态广播字段（`animat/state`）

主进程每秒广播一次，调试端订阅即可监控。字段：

| 字段 | 含义 | 来源 |
|------|------|------|
| `tick` | 潜意识循环计数 | brain |
| `viability` | 当前生存度（电量，%） | homeostasis |
| `vitals` | 五大稳态指标：`viability/integrity/affiliation/competence/exploration`（各 0–100） | homeostasis |
| `emotion.dominant` | 主导情绪（SEEKING/FEAR/CARE…） | 情感仲裁 |
| `emotion.intensity` | 主导情绪强度（0–1） | 情感仲裁 |
| `emotion.cause` | 触发原因描述 | 情感仲裁 |
| `emotion.all` | 全部激活情绪（含次导）列表 | 马斯洛仲裁 |
| `drive_level` | 当前最大驱力（0–1，越高越焦虑/越想行动） | homeostasis |
| `thoughts` / `activations` | 累计思维数 / 被门控放行的行为数 | stream_cortex |
| `activation_rate_pct` | 激活率（放行/总思维） | stream_cortex |
| `thoughts_per_min` | 思维速率（思维/分钟） | stream_cortex |
| `conversation_history` | 最近对话历史（末尾 6 条，`{role, content}`） | stream_cortex |
| `self_info` | 记住的对方信息（如 `{name:"李先生"}`） | 语义记忆 |

> 终端模拟器也会订阅 `animat/heartbeat`（轻量版：tick/viability/emotion/drive_level/thoughts/activations），但完整快照请以 `animat/state` 为准。

---

## 9. 命令速查表

```
对话          : 直接输入文本
/help         : 显示帮助
/charge       : 开始充电
/stop         : 停止充电
/reset        : 电量重置为 100%
/status       : 查看模拟器电量/充电状态
/touch        : 抚摸（caress）
/hit          : 撞击（pain + withdraw 反射）
/approach     : 有人靠近 1.5m
/leave        : 人离开
/noise        : 巨响（startle + flinch 反射）
/battery N    : 设电量 N%
/temp N       : 设体温 N°C
```
