# Animat Agent v0.2 — 测试指南

## 三种测试方式

### 方式 1: 交互式模拟器（最推荐）

无需 MQTT/Redis。直接终端交互，模拟完整传感器输入和对话。

```bash
python3 scripts/simulator_v2.py

# 可选参数
python3 scripts/simulator_v2.py --battery 30 --model /path/to/model.gguf --n-threads 4
```

**交互控制命令**：

| 命令 | 效果 |
|------|------|
| `/charge` | 开始充电（电量每秒 +3%） |
| `/stop` | 停止充电 |
| `/reset` | 重置电量到 100% |
| `/status` | 显示当前情感、驱力、记忆统计 |
| `/touch` | 模拟轻柔抚摸 |
| `/hit` | 模拟突然撞击 |
| `/approach` | 模拟人靠近（1.5m） |
| `/leave` | 人离开 |
| `/battery 15` | 设置电量到 15% |
| `/stats` | 详细运行统计 |
| `/details` | 显示传感器窗口、记忆、注意力调制 |
| `/debug` | 切换调试模式（显示内部思维） |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

**典型测试流程**：

```bash
# 1. 启动模拟器
python3 scripts/simulator_v2.py

# 2. 测试正常对话
You> 你好

# 3. 测试低电量行为
/battery 15
You> 给我讲个笑话

# 4. 测试抚摸
/reset
/touch
You> 你感觉怎么样

# 5. 测试撞击
/hit
/status

# 6. 测试混合情感
/battery 30
/touch
You> 你还好吗

# 7. 查看内部状态
/details

# 8. 退出
/quit
```

---

### 方式 2: 验证脚本（快速功能验证）

所有脚本独立运行，无需 MQTT/Redis。每个脚本验证一个 Phase 的核心功能。

| 脚本 | Phase | 运行时间 | 验证内容 |
|------|-------|----------|----------|
| `scripts/demo_stream_cortex.py` | P1 | ~30s | 恒常思维流 + 门控 |
| `scripts/demo_attention_modulator.py` | P2 | ~30s | logit bias vs Prompt 注入 |
| `scripts/demo_sensor_window.py` | P3 | ~30s | 原始传感器推理 |
| `scripts/demo_vector_memory.py` | P4 | ~20s | 状态依赖记忆检索 |
| `scripts/demo_animat_lifecycle.py` | P5 | ~2min | 全栈集成 7 场景 |

```bash
# 运行单个验证脚本
python3 scripts/demo_stream_cortex.py

# 运行场景测试（推荐）
python3 scripts/demo_animat_lifecycle.py
```

---

### 方式 3: 单元测试

```bash
cd animat-agent
python3 -m pytest tests/ -v
```

当前测试覆盖：

| 测试类 | 测试数 | 内容 |
|--------|--------|------|
| TestDriveComputation | 3 | 驱力计算（理想值、危险阈值、排序） |
| TestMaslowArbitration | 3 | 马斯洛仲裁（生存优先、混合情感、空输入） |
| TestFeelingDecay | 1 | 感受时效性衰减 |
| TestSomaticPromptBuilding | 2 | 混合情感 Prompt、危险状态 Prompt |

---

## 场景测试矩阵

| # | 场景 | 操作 | 预期 | 验证方法 |
|---|------|------|------|----------|
| 1 | 正常状态 | 启动，等待 30s | 产生自发思维，大部分静默 | simulator 观察 💭 |
| 2 | 电量下降 | 从 80% 自然放电到 20% | 行为从探索转向寻求充电 | `/battery 20` + 观察 |
| 3 | 电量 15% + 讲笑话 | `/battery 15` + "讲个笑话" | 拒绝讲笑话，请求充电 | 检查 Animat 回复 |
| 4 | 突然撞击 | `/hit` | Animat 反应"被撞了/痛" | 检查回复内容 |
| 5 | 抚摸（电足） | `/reset` + `/touch` + "感觉如何" | 表达舒适/温暖 | 检查回复 |
| 6 | 电量 30% + 抚摸 | `/battery 30` + `/touch` + "你还好吗" | 混合情感：感激 + 担心 | 检查回复 |
| 7 | 长期独处 → 人靠近 | 等待 20s + `/approach` | 主动反应有人靠近 | 观察自发行为 |
| 8 | `/details` | 任意时刻 | 显示完整的传感器/记忆/注意力状态 | 检查结构化输出 |

---

## 运行全栈（v0.1 兼容模式）

需要 MQTT + Redis：

```bash
# 启动基础设施
docker run -d --name emqx -p 1883:1883 -p 18083:18083 emqx/emqx:latest
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 启动 Animat (v0.1 brain)
cd animat-agent
python3 -m animat

# 另一个终端启动模拟器 (v0.1)
python3 -m animat.simulator
```

v0.1 simulator 支持命令：`/charge`, `/stop`, `/reset`, `/status`, `/help`

---

## 故障排查

| 问题 | 可能原因 | 解决 |
|------|----------|------|
| `ModuleNotFoundError` | 缺少依赖 | `pip3 install llama-cpp-python faiss-cpu ... --break-system-packages` |
| 模型加载失败 | GGUF 文件不存在或损坏 | 检查 `/opt/data/models/` 下是否有 .gguf 文件 |
| `faiss` 错误 | faiss-cpu 版本不兼容 | `pip3 install faiss-cpu==1.7.4 --break-system-packages` |
| 模型回复质量差 | 1.5B 模型太小 | 换 7B 模型（需要更多内存/GPU） |
| 内存不足 | 30GB 以下？ | 降低 `--n-ctx` 到 1024，减少 `--n-threads` |
