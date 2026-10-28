"""
恒常意识流引擎 — StreamCortex

核心变更（vs v0.1 cortex.py）:
  1. LLM 不再被"唤醒"，而是一直在运行
  2. 不断产生内部思维（Thought），大部分被丢弃
  3. 门控模块（ActionGate）决定哪些思维变成外部行为
  4. 潜意识循环注入身体状态和传感器上下文

设计哲学:
  「LLM 是一条不间断的河流，不是按需点燃的火焰。」
"""

import asyncio
import json
import logging
import time
from typing import Optional

from animat.config import Topic
from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.llm_backend import LocalLLM, LLMConfig
from animat.thought import Thought, ActionGate, ActionDecision, ThoughtJournal
from animat.attention_modulator import AttentionModulator, BodyState
from animat.sensor_window import SensorWindow

logger = logging.getLogger("animat.stream_cortex")

# ── 意识流 System Prompt ──
CONSCIOUSNESS_SYSTEM_PROMPT = """\
你是一个名为 Animat 的具身机器人。你拥有物理身体，能感知触摸、声音、视觉和内部状态（如电量）。

关键规则:
1. 你是一个持续运行的"生命体"，而不是一问一答的聊天机器人。
2. 你会不断产生内部想法——对当前状况的感受、回忆、计划、担忧、好奇。
3. 你的想法应该简短（1-3句话），像人类的内心独白。
4. 你的想法应该自然——有时是观察，有时是情绪，有时是想做什么。
5. 不要总是"行动"，大多数想法只是想想而已。

当前状态:
{body_context}

你的内部想法:"""


class StreamCortex:
    """
    恒常意识流引擎
    
    替代 v0.1 的 cortex.py。
    
    核心循环:
      1. 读取当前身体状态和传感器上下文
      2. 调用 LLM 产生一个内部思维
      3. ActionGate 评估是否需要转化为行为
      4. 如果需要 → 通过 MQTT 发射行为指令
      5. 思维存档到 ThoughtJournal
      6. 等待一个短间隔，然后产生下一个思维
    """

    # 配置
    THINK_INTERVAL_SEC = 3.0        # 思维间隔(秒)
    THINK_INTERVAL_IDLE = 5.0       # 安静时的思维间隔
    THINK_INTERVAL_DRIVEN = 1.5     # 高驱力时思维加速
    MAX_THINK_TOKENS = 80           # 每个思维的最大 token 数
    MAX_CONVERSATION_MEMORY = 10    # 对话记忆窗口

    def __init__(
        self,
        nervous: NervousSystem,
        memory: MemoryStore,
        llm_backend: Optional[LocalLLM] = None,
    ):
        self.nervous = nervous
        self.memory = memory
        self.llm = llm_backend or LocalLLM()

        # 注意力调制器（Phase 2 核心）
        self.modulator = AttentionModulator()
        self._body_state = BodyState()
        
        # 传感器窗口（Phase 3 核心）
        self.sensor_window: Optional[SensorWindow] = None

        # 状态
        self._running = False
        self._body_context: str = "各项指标正常，心情平静。"
        self._current_emotion: str = "SEEKING"
        self._drive_level: float = 0.0
        
        # 门控
        self.action_gate = ActionGate()
        self.journal = ThoughtJournal()

        # 对话记忆（外部输入）
        self._pending_external_text: Optional[str] = None
        self._external_text_lock = asyncio.Lock()

        # 统计
        self._thought_count = 0
        self._activation_count = 0
        self._start_time: float = 0.0

        # 兼容 v0.1 接口
        self._processing = False
        self._conversation_history: list[dict] = []

    @property
    def busy(self) -> bool:
        """StreamCortex 永不'忙'——恒常运行"""
        return False

    async def register(self):
        """注册神经末梢"""
        # 接收躯体标记更新（从 homeostasis.py 来，保留兼容）
        await self.nervous.listen(Topic.SOMATIC_PROMPT, self._on_body_context_update)
        await self.nervous.listen(Topic.ARBITRATION, self._on_emotion_update)
        # 外部语音输入
        await self.nervous.listen("animat/external/speech", self._on_external_speech)
        logger.info("StreamCortex neurons registered")

    async def start(self):
        """启动恒常意识流（永不返回的协程）"""
        # 恢复对话历史（让重启后的 Animat 仍能延续上下文）
        await self._load_history()

        if not self.llm.ready:
            self.llm.load()
        
        # 初始化注意力调制器的 tokenizer
        # 本地后端暴露 _llm(llama对象); 云端后端直接暴露 tokenize 方法。
        # AttentionModulator.set_tokenizer 兼容二者，故统一传入 self.llm。
        self.modulator.set_tokenizer(self.llm)

        self._running = True
        self._start_time = time.time()
        logger.info("═══ StreamCortex consciousness stream started ═══")
        
        while self._running:
            try:
                await self._think_cycle()
            except Exception:
                # 单个思维周期失败（多为 MQTT 瞬断重连期间的发布异常）不应杀死
                # 整条意识流——神经层会自动重连，这里退避后继续即可。
                logger.warning("StreamCortex cycle error (resuming)", exc_info=True)
                await asyncio.sleep(1.0)
        logger.info("StreamCortex stopped")

    async def stop(self):
        """停止意识流"""
        self._running = False

    # ── 身体状态更新回调 ──
    async def _on_body_context_update(self, topic: str, payload: dict):
        """接收从 homeostasis.py 来的身体状态上下文（保留兼容）"""
        prompt = payload.get("prompt", "")
        if prompt:
            self._body_context = prompt

    async def _on_emotion_update(self, topic: str, payload: dict):
        """接收情感仲裁结果，同步更新 BodyState"""
        self._current_emotion = payload.get("dominant_emotion", self._current_emotion)
        self._drive_level = payload.get("intensity", self._drive_level)
        
        # 从稳态值更新 BodyState（用于注意力调制）
        try:
            vitals = await self.memory.load_state("animat:homeostasis")
            if vitals:
                self._body_state = BodyState.from_vitals(vitals)
        except Exception:
            pass

    async def _on_external_speech(self, topic: str, payload):
        """接收人类语言输入（异步安全）

        兼容两种 payload: 标准 {"text": "..."} 字典，或裸字符串。
        nervous 在 JSON 解析失败时会把原始字符串透传，这里一并处理。
        """
        if isinstance(payload, dict):
            text = payload.get("text", "")
        elif isinstance(payload, str):
            text = payload
        else:
            text = str(payload)
        text = (text or "").strip()
        if not text:
            return
        async with self._external_text_lock:
            self._pending_external_text = text
        logger.info(f"StreamCortex: external speech queued: {text[:40]}")

    # ── 主循环 ──
    async def _think_cycle(self):
        """一次完整的思维循环"""
        # 1. 检查是否有外部输入
        external_text = None
        async with self._external_text_lock:
            if self._pending_external_text:
                external_text = self._pending_external_text
                self._pending_external_text = None

        # 2. 构建上下文
        system_prompt = CONSCIOUSNESS_SYSTEM_PROMPT.format(
            body_context=self._body_context[:500]  # 限制上下文长度
        )

        # 3. 产生思维
        thought = None
        try:
            if external_text:
                # 外部输入 → 产生回复思维
                thought = await self._think_response(system_prompt, external_text)
            else:
                # 自发思维
                thought = await self._think_spontaneous(system_prompt)
        except Exception:
            logger.error("Error during think cycle", exc_info=True)
            return

        if thought is None:
            return

        # 4. 门控评估
        max_drive = self._drive_level
        is_external = external_text is not None
        decision = self.action_gate.evaluate(
            thought=thought,
            drive_level=max_drive,
            is_external_trigger=is_external,
        )

        # 5. 记录
        self.journal.record(thought)
        self._thought_count += 1

        # 5.5 广播每一次内部思维（内心独白），供调试端实时监控
        await self._broadcast_thought(thought, decision.should_activate)

        # 6. 如果门控开放 → 发射行为
        if decision.should_activate:
            self._activation_count += 1
            await self._emit_action(decision, thought)

        # 7. 动态等待间隔
        if external_text:
            wait_sec = 0.5  # 外部输入后快速产生下一个思维
        elif max_drive > 0.5:
            wait_sec = self.THINK_INTERVAL_DRIVEN
        elif max_drive < 0.2:
            wait_sec = self.THINK_INTERVAL_IDLE
        else:
            wait_sec = self.THINK_INTERVAL_SEC
        
        # 引入随机抖动，避免机械节奏
        import random
        wait_sec *= random.uniform(0.8, 1.2)
        await asyncio.sleep(wait_sec)

    async def _think_spontaneous(self, system_prompt: str) -> Optional[Thought]:
        """产生自发思维"""
        # 获取最近记忆
        recent_memories = []
        try:
            mems = await self.memory.recall_episodes(limit=3)
            for m in mems:
                recent_memories.append(
                    f"[{m.get('emotion', '?')}] {m.get('summary', '')}"
                )
        except Exception:
            pass

        # Phase 3: 传感器上下文 — 优先使用 SensorWindow 原始数据
        if self.sensor_window:
            sensor_context = self.sensor_window.to_context(max_frames_per_modality=3)
        else:
            # 回退：使用 homeostasis 的文本描述
            sensor_context = self._body_context[:300]

        # 产生思维（使用 logit bias 调制注意力）
        try:
            # 召回已记住的对方信息，避免幻觉编造
            self_info = await self._recall_self_info()
            if self_info:
                system_prompt = system_prompt + "\n[关于对方你已知的信息] " + self_info

            # 计算当前身体状态的 token 偏移
            logit_bias = self.modulator.compute_logit_bias(self._body_state)

            # 最近对话历史（让内心独白也能引用刚聊的内容）
            recent_conv = self._conversation_history[-6:] if self._conversation_history else None

            accumulated = ""
            for stream_thought in self.llm.stream_think(
                system_prompt=system_prompt,
                context_memories=recent_memories,
                sensor_context=sensor_context,  # ← Phase 3: 原始传感器数据
                max_tokens=self.MAX_THINK_TOKENS,
                logit_bias=logit_bias,           # ← Phase 2: 注意力调制
                conversation_context=recent_conv,
            ):
                accumulated = stream_thought.accumulated_text
                if stream_thought.is_finished:
                    break
        except Exception as e:
            logger.error(f"Spontaneous think failed: {e}")
            # 产生一个简单的占位思维
            accumulated = "...(沉默)..."

        if not accumulated.strip():
            return None

        # 计算相关性评分
        relevance = self._compute_relevance(accumulated)
        
        return Thought(
            text=accumulated.strip(),
            relevance=relevance,
            dominant_emotion=self._current_emotion,
            trigger="spontaneous",
            body_state_snapshot={"emotion": self._current_emotion},
        )

    async def _think_response(
        self, system_prompt: str, external_text: str
    ) -> Optional[Thought]:
        """产生对人类的回复思维"""
        # 1) 把外部输入记入对话历史（用于上下文连贯，避免"记不住名字"）
        self._conversation_history.append({"role": "user", "content": external_text})
        # 仅保留最近 N 轮
        if len(self._conversation_history) > self.MAX_CONVERSATION_MEMORY * 2:
            self._conversation_history = self._conversation_history[-self.MAX_CONVERSATION_MEMORY * 2:]

        # 2) 抽取身份/事实性信息存入语义记忆（如"我叫XX"）
        await self._maybe_remember_facts(external_text)

        # 3) 召回已记住的自我相关信息，注入 system prompt
        self_info = await self._recall_self_info()
        if self_info:
            system_prompt = system_prompt + "\n[关于对方你已知的信息] " + self_info

        # 4) 构建对话消息（带历史，让回复有上下文）
        messages = [
            {"role": "system", "content": system_prompt},
            *self._conversation_history,
        ]
        
        try:
            response = self.llm.complete(
                messages=messages,
                max_tokens=self.MAX_THINK_TOKENS,
            )
            text = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Response think failed: {e}")
            return None

        # 记录 assistant 回复到历史
        self._conversation_history.append({"role": "assistant", "content": text})

        # 持久化对话历史（与稳态指标一样重启后延续）
        await self._save_history()

        # 回复思维的相关性通常是高的（有人在等）
        return Thought(
            text=text,
            relevance=0.85,  # 回复人类默认高相关性
            dominant_emotion=self._current_emotion,
            trigger="external",
        )

    async def _maybe_remember_facts(self, text: str):
        """从用户语句中尝试抽取身份/事实信息，写入语义记忆。
        失败则静默跳过，不影响主流程。"""
        try:
            prompt = (
                "从下面这句话中抽取关于'对方（人类）'的稳定事实信息（如姓名、称呼、喜好）。\n"
                "只输出 JSON，例如 {\"name\": \"小明\", \"note\": \"喜欢喝咖啡\"}；"
                "若没有可记忆的事实，输出 {}. 不要输出其他内容。\n\n"
                f"句子：{text}"
            )
            resp = self.llm.complete(
                messages=[{"role": "system", "content": prompt}],
                max_tokens=80,
            )
            import json as _json
            raw = resp["choices"][0]["message"]["content"].strip()
            # 容错：截取第一个 { 到最后一个 }
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                return
            data = _json.loads(raw[start:end + 1])
            if not isinstance(data, dict) or not data:
                return
            # 合并进已有的 self_info
            existing = await self.memory.recall_semantic("self_info") or {}
            existing.update({k: v for k, v in data.items() if v})
            await self.memory.store_semantic("self_info", existing)
            logger.info(f"Remembered self_info: {existing}")
        except Exception as e:
            logger.debug(f"Fact extraction skipped: {e}")

    async def _recall_self_info(self) -> str:
        """召回已记住的关于对方的事实，格式化为单行文本"""
        try:
            info = await self.memory.recall_semantic("self_info")
            if not info:
                return ""
            parts = []
            if info.get("name"):
                parts.append(f"对方叫{info['name']}")
            for k, v in info.items():
                if k == "name":
                    continue
                parts.append(f"{k}:{v}")
            return "；".join(parts)
        except Exception:
            return ""

    # ── 对话历史持久化 ──
    async def _save_history(self):
        """把最近对话历史写入 Redis（重启可延续）"""
        try:
            await self.memory.store_semantic(
                "conversation_history",
                self._conversation_history[-self.MAX_CONVERSATION_MEMORY * 2:],
            )
        except Exception as e:
            logger.debug(f"Save history skipped: {e}")

    async def _load_history(self):
        """从 Redis 恢复上次对话历史"""
        try:
            hist = await self.memory.recall_semantic("conversation_history")
            if isinstance(hist, list) and hist:
                self._conversation_history = hist
                logger.info(f"Conversation history restored ({len(hist)} msgs)")
        except Exception as e:
            logger.debug(f"Load history skipped: {e}")

    def _compute_relevance(self, text: str) -> float:
        """
        评估思维的行为相关性
        
        策略:
        - 包含行动意向的词 → 高相关性
        - 纯观察/感受 → 中等
        - 空/重复 → 低
        """
        text_lower = text.lower()
        
        # 行动词
        action_words = ["去", "要", "想", "应该", "必须", "寻找", "充电",
                        "求助", "警告", "逃", "找", "说"]
        # 感受词
        feeling_words = ["感觉", "好像", "似乎", "有点", "觉得", "无聊",
                         "舒服", "害怕", "孤单", "开心"]
        
        action_score = sum(1 for w in action_words if w in text_lower) * 0.15
        feeling_score = sum(1 for w in feeling_words if w in text_lower) * 0.08
        
        # 文本长度惩罚（太短可能不完整，太长可能是幻觉）
        length = len(text)
        if length < 5:
            length_penalty = -0.3
        elif length > 200:
            length_penalty = -0.1
        else:
            length_penalty = 0.0
        
        score = min(action_score + feeling_score + length_penalty + 0.3, 1.0)
        return max(0.0, min(1.0, score))

    async def _emit_action(self, decision: ActionDecision, thought: Thought):
        """通过 MQTT 发射行为"""
        # 构建与 v0.1 兼容的输出格式
        output = {
            "intent": decision.action_type,
            "text": thought.text,
            "action_detail": thought.text,
            "_source": "stream_cortex",
            "_activation_score": decision.activation_score,
        }
        try:
            await self.nervous.fire(Topic.CORTEX_OUTPUT, output)
        except Exception:
            logger.warning("StreamCortex emit failed (MQTT unavailable?): %s", decision.action_type)
            return
        logger.info(f"StreamCortex emit: {decision.action_type} | {thought.text[:60]}")

    async def _broadcast_thought(self, thought: Thought, activated: bool):
        """把每一次内部思维广播到 animat/cortex/thought（内心独白流）。

        无论是否被门控放行都会广播——这模拟了人类'大多数想法不说出口'，
        调试端因此能看到 Animat 完整的内心活动，而不只是它最终说出来的那部分。
        """
        try:
            await self.nervous.fire(Topic.CORTEX_THOUGHT, {
                "text": thought.text,
                "trigger": thought.trigger,            # spontaneous / external
                "emotion": thought.dominant_emotion,
                "relevance": round(thought.relevance, 3),
                "activated": activated,                # 是否冲破门控成为外部行为
                "timestamp": time.time(),
            }, qos=0)  # 高频遥测，用 qos=0 即可
        except Exception:
            pass

    # ── 统计 ──
    def stats(self) -> dict:
        """获取意识流统计"""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        rpm = (self._thought_count / (uptime / 60)) if uptime > 0 else 0
        activation_rate = (
            self._activation_count / self._thought_count * 100
            if self._thought_count > 0
            else 0
        )
        return {
            "running": self._running,
            "uptime_sec": round(uptime, 1),
            "thoughts_total": self._thought_count,
            "thoughts_per_min": round(rpm, 1),
            "activations": self._activation_count,
            "activation_rate_pct": round(activation_rate, 1),
            "current_emotion": self._current_emotion,
            "drive_level": round(self._drive_level, 2),
            "journal": self.journal.summary(),
        }
