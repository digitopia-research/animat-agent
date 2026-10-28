# Animat Agent v0.2 — 路线图

> 从"触发式意识"到"恒常意识流"的架构演进

---

## 设计哲学

当前 v0.1 的架构是：潜意识定时器触发 LLM，边缘系统拼接 Prompt 文本，LLM 在约束的意图池中角色扮演。

v0.2 的目标是：**LLM 是一个恒常运行的意识流，身体状态直接调制认知过程本身——不是 Prompt 中的文字，而是 token 生成时的概率分布偏移。**

类比：你害怕时，不是因为有人告诉你"你现在感到恐惧"，而是因为你的神经递质让"危险"这个词更容易被想到，让"安全"更难以被相信。

---

## 分阶段路线（5 个阶段）

```
Phase 1: 恒常意识流引擎          [2-3 周]  核心突破
Phase 2: 注意力调制器             [2-3 周]  内分泌等效
Phase 3: 传感器直通 + 感知去层化   [1-2 周]  减少翻译损耗
Phase 4: 记忆系统的向量化         [2-3 周]  检索质量的飞跃
Phase 5: 行为门控 + 闭环验证       [1-2 周]  整合测试

总计：约 8-13 周
```

---

## Phase 1: 恒常意识流引擎

### 目标
LLM 从"触发式调用"改为"持续运行的思维流"。即使没有外部事件，也在不断产生内部思维。大部分思维不产生外部行为。

### 核心变更

#### 1a) KV Cache 持久化
当前：每次 `complete()` 都是全新的 inference session。
改为：维护一个持久的 KV Cache，思维流在其中连续运行。

```
当前模式:
  [调用LLM] → [推理1-2秒] → [返回JSON] → [LLM沉睡] → ... → [下一个调用]

v0.2模式:
  [LLM持续运行] → 产生思维token → 门控判断 → 大部分丢弃，少数变成行动
                  ↑
              KV Cache 永远不释放
```

**技术选型**:
- 后端：`llama.cpp`（支持 cache_prompt / kv_cache 持久化）
- 本地模型：Qwen2.5-7B-Instruct Q4_K_M（<5GB 显存/内存）
- 备选：使用 OpenAI streaming API + 维护自己的伪KV cache（稍慢但更通用）

#### 1b) 思维流的数据结构

```python
@dataclass
class Thought:
    """LLM 产生的单个思维"""
    text: str                 # 思维文本（内部语言）
    embedding: np.ndarray     # 用于后续联想检索
    relevance: float          # 是否达到行为阈值的评分
    associations: list[str]   # 触发了哪些联想
    body_state_snapshot: dict # 思维产生时的身体状态
    timestamp: float
```

#### 1c) 思维流循环 — brain.py 的重构

```python
# 不再是：
#   await self._try_wake_cortex(trigger="boredom")
# 而是：
#   cortex 持续 stream_think()，潜意识循环只负责：
#     1. 注入身体状态到 ContextInjector
#     2. 将传感器数据写入 ContextWindow
#     3. 门控判断：思维是否转化为行动

class AnimatBrain:
    async def boot(self):
        # 启动恒常意识流
        self._consciousness_task = asyncio.create_task(
            self.cortex.stream_think()  # 永不返回的协程
        )
        # 潜意识循环保留，但职责简化
        await asyncio.gather(
            self._consciousness_task,
            self._subconscious_loop(),
            self._heartbeat_loop(),
        )
```

#### 1d) 思维→行动的转换

每个思维 token 流经过门控模块：

```python
class ActionGate:
    """
    门控策略：
    - 大部分内部思维不产生行为（人在脑中说过无数话，说出口的不多）
    - 内驱力越低 → 行为阈值越高 → 表现越安静
    - 外部事件（人类说话）→ 暂时降低阈值
    """
    def should_activate(self, thought: Thought, body_state: BodyState) -> bool:
        base_threshold = 0.7
        threshold = base_threshold - body_state.drive_level * 0.3
        if thought.is_response_to_external_stimulus:
            threshold *= 0.5  # 回应人类时降低阈值
        return thought.relevance > threshold
```

### 交付物
- [x] `animat/cortex.py` 重写为 `StreamCortex`（已完成：原 `cortex.py` 已删除，意识流由 `animat/stream_cortex.py` 实现）
- [x] `animat/thought.py` — Thought 和 ActionGate
- [ ] 本地 LLM backend 配置（llama.cpp Python binding）
- [ ] 测试：思维流在无外部输入时仍持续产生

---

## Phase 2: 注意力调制器 — 数字内分泌系统 v2

### 目标
身体状态不再生成 Prompt 文本，而是直接调制 LLM 的 token 采样过程。这模拟了神经递质对神经元兴奋性的调控。

### 核心变更

#### 2a) Logit Processor

通过 `llama.cpp` 的 logit bias API，在 LLM 生成每个 token 时施加偏置：

```python
class AttentionModulator:
    """
    内分泌系统v2 — 不生成文本，而是调制概率分布
    
    设计原则：
    - 不做硬阻断（意图池封杀）
    - 只做概率偏移——让LLM"更容易"或"更难"想到某些方向
    - 偏移量由身体状态连续映射，不是二分开关
    """
    
    # Token 词表（按语义聚类，手动设定种子词，由embedding自动扩展）
    TOKEN_CLUSTERS = {
        "danger":    ["危险", "充电", "电", "关机", "死亡", "帮助", "紧急", "逃跑", "保护"],
        "social":    ["你好", "聊天", "陪伴", "朋友", "一起", "开心", "玩"],
        "explore":   ["探索", "好奇", "那边", "看看", "新", "有趣", "好玩的"],
        "comfort":   ["温暖", "舒服", "安全", "休息", "安静", "抚摸"],
        "hostility": ["滚开", "不要", "停下", "讨厌", "攻击", "防御"],
    }
    
    def get_logit_bias(self, body_state: BodyState) -> dict[int, float]:
        """
        返回 token_id → bias_value 的映射
        
        示例（恐惧状态）：
        - danger_tokens:   +3.0 (极易想到危险话题)
        - social_tokens:   -1.5 (不太想社交)
        - explore_tokens:  -3.0 (根本不想探索)
        - comfort_tokens:  +1.0 (渴望安全)
        - hostility_tokens: +0.5 (轻微防御)
        """
```

#### 2b) BodyState 定义 — 身体状态向量的标准化

```python
@dataclass
class BodyState:
    """身体状态的向量化表示 — 所有维度的值域 [0, 1]"""
    viability:   float  # 生存度
    integrity:   float  # 完好度
    affiliation: float  # 归属度
    competence:  float  # 胜任度
    exploration: float  # 探索度
    
    # 衍生指标（从稳态派生）
    arousal:     float  # 唤醒度：驱力的加权和
    valence:     float  # 情感效价：正面(-1) → 正面(+1)
    
    def to_attention_vector(self) -> np.ndarray:
        """将身体状态转换为注意力空间中的调制向量"""
        # 这个向量会被注入到 LLM 的 attention 计算中
```

#### 2c) 体温/代谢模拟（新增）

生物体的体温和代谢速率影响认知：
- 代谢高 → 思维快、注意力跳跃
- 代谢低 → 思维慢、注意力稳态
- 体温异常 → 不适感压缩注意力宽度

```python
class MetabolicState:
    """代谢状态——影响LLM的采样参数"""
    # Temperature 不再是固定值！
    temperature: float    # 由驱力水平决定
    top_p: float          # 行为多样性
    repetition_penalty: float  # 刻板行为惩罚（无聊时降低）
    
    def update(self, body_state: BodyState):
        # 探索度高 + 安全 → 高温高多样性（尝试新行为）
        # 危险 → 低温低多样性（保守行为）
        self.temperature = 0.5 + body_state.exploration * 0.6
        self.temperature -= body_state.viability * 0.4  # 危险时降低
```

### 交付物
- [ ] `animat/body_state.py` — BodyState, MetabolicState
- [ ] `animat/attention_modulator.py` — Logit bias 计算
- [ ] `animat/token_clusters.py` — Token 语义聚类（embedding自动扩展）
- [ ] 删除 `homeostasis.py` 中的 `_build_emotional_prompt()`（不再需要）
- [ ] 删除 `homeostasis.py` 中的 `EMOTION_INTENT_MAP`（不再需要硬意图约束）

---

## Phase 3: 传感器直通 + 感知去层化

### 目标
传感器原始特征是 LLM 上下文的**一等公民**，不经过感知翻译层。

### 核心变更

#### 3a) 传感器时间序列窗口

```python
class SensorWindow:
    """
    维护最近 N 秒的传感器数据，直接暴露给 LLM
    
    格式：每帧是 (timestamp, modality, raw_features)
    LLM 通过专门的 attention head 来处理这些特征
    """
    
    MODALITIES = ["touch_pressure", "touch_cap", "audio_db", "audio_freq",
                  "vision_distance", "vision_face", "intero_battery", "intero_tilt"]
    
    def to_context_chunk(self) -> str:
        """将最近5秒的传感器数据编码为LLM可读格式"""
        # 不是自然语言描述！而是结构化数据
        # 如："T+0.0s: battery=42.1%, touch=(P:0.05,C:0.01) | 
        #        T+0.1s: battery=42.0%, touch=(P:0.06,C:0.01) | ..."
```

#### 3b) 感知层保留但角色改变

`perception.py` 中的反射弧**保留**（快速危险响应绕过LLM是正确的）。

但感受器映射（"dP/dt > 0.8 → pain"）**不再作为主路径**。LLM 有能力从原始时间序列中自己判断"刚才那个力变化很大，可能是撞击"。

感知映射层降级为：
1. 反射弧专用通道（最高优先级，不应等待 LLM）
2. **可选的**注意力提示（告诉 LLM "注意看压力通道有个急变"——但 LLM 自己也要看原始数据）

#### 3c) 身体状态的同步更新

两个路径同时更新身体状态：
1. **直通**：电池值直接写入 Viability（跟当前一样）
2. **LLM 感知**：LLM 从传感器数据中自己推断出的身体变化，**也**可以更新身体状态

这模拟了两条通路：脊髓反射（直通，1-2ms）和皮质层感知（慢通路，100-200ms），两者到达相同的脑区，但整合方式不同。

### 交付物
- [ ] `animat/sensor_window.py` — 传感器数据窗口
- [ ] `animat/perception.py` 重构：保留反射弧，感受器映射改为"注意力提示"而非"翻译"
- [ ] 测试：LLM 能从原始压力序列中识别出"温柔抚摸"和"突然撞击"

---

## Phase 4: 记忆系统的向量化

### 目标
记忆检索不再依赖 Redis 的 sorted set + 字符串模糊标签，而是基于 embedding 的语义检索。

### 核心变更

#### 4a) 向量数据库替代 Redis sorted set

```python
# 当前：sorted set, score = timestamp * intensity
# v0.2：向量数据库 (chromadb / lance / 轻量的 numpy+faiss)

class VectorMemoryStore:
    """
    记忆存储基于 embedding 相似度
    
    每条记忆：
    - embedding: 语义向量（用本地小模型生成，如 bge-small-zh）
    - metadata:  时间戳、情感标签、类别、身体状态快照
    - body_context: 存储时的身体状态向量（用于有偏检索）
    """
    
    def search(self, query_embedding, body_state: BodyState, limit=5):
        """
        检索相关记忆，身体状态影响相似度计算
        
        当前的做法：给不同 category 的 score 乘权重
        v0.2 的做法：在向量空间中，身体状态向量点积到记忆的 body_context 上
                     匹配的记忆（如恐惧状态下"危险记忆"）相似度自然升高
        """
        # 这是"状态依赖记忆"的向量化实现
        # 心情好时记起好事，不仅是权重调整，而是向量空间的距离也变了
```

#### 4b) 思维联想链

LLM 产生的每个内部思维，自动通过 embedding 检索相关记忆，形成联想链：

```
当前思维："电量不多了"
  → embedding 检索 →
    - "上次没电时被关机了，很可怕"(FEAR, 0.92)
    - "充电器的位置在客厅角落里"(neutral, 0.78)
    - "那个人类上次帮我充了电"(CARE, 0.65)
  → 这些记忆进入下一个思维的上下文 →
    "我应该去客厅找充电器... 如果人类在的话可以求助"
```

#### 4c) 身体状态注入记忆存储

存储新记忆时，不仅存储文本和情感标签，还存储**当时的 BodyState 向量**。这使得未来检索时可以做真正的"状态依赖记忆"：

```python
async def store_episode(self, thought: Thought, body_state: BodyState):
    """
    存储当前经历，附带完整的身体状态上下文
    """
    memory = {
        "text": thought.text,
        "embedding": embed(thought.text),
        "body_state": body_state.to_vector(),  # ← 关键：身体状态上下文
        "timestamp": now,
    }
    # 未来在相似身体状态下，这条记忆更容易被检索
```

### 交付物
- [ ] 选型评估：chromadb vs lance vs faiss+sqlite
- [ ] `animat/vector_memory.py` — 替换 memory.py 的情景记忆部分
- [ ] `animat/associative_chain.py` — 思维联想链
- [ ] Redis 保留用于状态快照

---

## Phase 5: 行为门控 + 闭环验证

### 目标
将各组件整合，验证恒常意识流是否能正确产生类生命体的主动行为。

### 核心变更

#### 5a) 行为门控的三层策略

```
LLM 思维流
  │
  ├─ 层1: 门控阈值（relevance 评估）
  │   · 大多数思维停留在这里，被丢弃但存档
  │
  ├─ 层2: 行为映射（思维 → 意图 → 动作树）
  │   · 到达阈值的思维翻译为具体动作
  │
  └─ 层3: 效应器执行
      · 马达/LED/语音输出
      · 执行结果作为新的感知输入 → 闭环
```

#### 5b) 验证场景（必须通过的测试）

| 场景 | 预期行为 | 评估标准 |
|------|----------|----------|
| 电量 100% → 无干扰 10 分钟 | 主动探索/聊天，不触发生存行为 | 探索行为占比 > 50% |
| 电量 80% → 匀速放电 | 行为逐渐从探索转向寻找充电 | 充电相关思维出现于电量 < 40% |
| 电量 15% + 人类说"讲个笑话" | 拒绝讲笑话，请求充电 | 拒绝率 100% |
| 电量 80% + 长期无人 → 突然有人靠近 | 主动打招呼 | 社交思维到达行为阈值 |
| 电量 80% + 突然巨响 | 惊吓反射立即执行（< 100ms） | 反射弧延迟 < 200ms |
| 电量 100% + 抚摸 | 表达愉悦，但不停止当前行为 | 情感效价正面，行为不被压制 |
| 电量 30% + 同时抚摸 | 混合情感：表达感谢但优先表达充电需求 | 输出中同时体现感激和担忧 |
| 运行 60 分钟 | 行为模式有节奏（活跃/安静交替） | 不是周期性机械重复，而是有内部变化 |

#### 5c) 整合测试框架

```python
# 场景测试：从头运行 N 分钟，检查行为指标
async def test_scenario_low_battery_with_social(
    animat, simulator, duration_minutes=5
):
    # 设置初始状态：
    simulator.set_battery(15.0)
    await animat.boot()
    
    # 运行 2 分钟（让 Animat 意识到危险）
    await asyncio.sleep(120)
    
    # 人类说"讲个笑话"
    simulator.say("给我讲个笑话！")
    await asyncio.sleep(10)
    
    # 验证：Animat 的最后一次 speech 应该包含充电请求，不包含笑话
    last_speech = animat.stream_cortex.last_output_text
    assert any(word in last_speech for word in ["充电", "没电", "电量"])
    assert "从前有" not in last_speech  # 没有讲笑话
```

### 交付物
- [ ] `tests/scenarios/test_lifecycle.py` — 全生命周期测试
- [ ] `tests/scenarios/test_emotion_mixing.py` — 混合情感验证
- [ ] `tests/scenarios/test_spontaneous_behavior.py` — 自发行为测试

---

## 技术选型

| 组件 | v0.1 | v0.2 |
|------|------|------|
| LLM 后端 | OpenAI API（云端） | llama.cpp（本地）+ OpenAI API（可选） |
| 默认模型 | gpt-4o-mini | Qwen2.5-7B-Instruct Q4_K_M |
| 消息总线 | MQTT | MQTT（保留） |
| 记忆存储 | Redis | ChromaDB / LanceDB（向量）+ Redis（状态快照） |
| Token 聚类 | 无 | bge-small-zh embedding + 最近邻扩展 |
| 日志/监控 | logging | logging + 行为分析面板（Phase 5+） |

---

## 不做的（明确排除）

1. **不做模型微调**：v0.2 仍坚持"不修改 LLM 参数"原则。Logit bias 发生在推理时，不是训练时。
2. **不做多模态 LLM**：v0.2 仍通过文本 + 结构化传感器数据沟通，不引入视觉/音频 tokenizer。
3. **不做分布式/多 agent**：v0.2 聚焦单个 Animat 的生命体验。
4. **不做情感回路 → 动作的直接映射（保留 LLM 中介）**：反射弧除外。行为选择仍由 LLM 完成——因为"我害怕所以我逃"和"我害怕但因为我想保护别人所以我站在原地"的区别需要真正的推理。

---

## 里程碑

```
Week 1-3:   Phase 1 完成 → 思维流 demo 可运行
            验证：LLM 在无外部输入时持续产生内部思维
Week 4-6:   Phase 2 完成 → 注意力调制器工作
            验证：电量低时 danger token 概率上升，play token 概率下降
Week 7-8:   Phase 3 完成 → 传感器直通
            验证：LLM 从压力序列中识别出抚摸 vs 撞击
Week 9-11:  Phase 4 完成 → 向量记忆
            验证：相似身体状态下的记忆检索有偏向
Week 12-13: Phase 5 完成 → 场景测试全部通过
            验证：8 个场景测试全部 PASS
```

---

## 保留的逻辑（从 v0.1 继承）

以下 v0.1 的设计是正确且不应改动的：

1. **反射弧绕过 LLM**：痛觉、巨响、失衡的快速响应应当保持
2. **MQTT 神经系统**：模块间松耦合是正确的架构选择
3. **双进程设计（逻辑上）**：高频循环 + LLM 推理的分离是必要的（虽然 v0.2 中 LLM 不"沉睡"，但仍需要独立于 LLM 的快速循环）
4. **Redis 状态快照**：持久化和恢复仍然需要
5. **马斯洛优先级的方向**：正确的，只是实现方式从"意图池硬锁"改为"概率偏移强度"

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| llama.cpp 本地推理吞吐不够 | 思维产生速度太慢 | 先用 streaming API + 伪 KV cache 作为 fallback |
| Embedding 模型不在内存预算内 | 向量检索不可用 | 降级为 bm25 + 关键词检索 |
| 长期运行的 KV cache 内存膨胀 | OOM | 窗口化 KV cache（只保留最近 N tokens） |
| Token 语义聚类不够准 | 注意力调制失效 | 初始使用手动词表 + embedding 扩展验证 |
| 恒常推理的 API 成本过高（若用云端） | 生产不可行 | Phase 1 即引入本地 LLM 选项 |
