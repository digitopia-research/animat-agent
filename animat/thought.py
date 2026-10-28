"""
思维模块 — 恒常意识流的核心数据结构

定义:
  1. Thought：LLM 产生的单个内部思维
  2. ActionGate：门控模块，决定思维是否转化为外部行为
  3. ThoughtJournal：思维日志（所有思维存档，用于长期记忆）
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("animat.thought")


@dataclass
class Thought:
    """LLM 产生的单个内部思维"""
    text: str                        # 思维文本
    relevance: float = 0.0           # 行为相关性评分 [0, 1]
    timestamp: float = field(default_factory=time.time)
    
    # 上下文快照
    dominant_emotion: str = "SEEKING"
    body_state_snapshot: Optional[dict] = field(default_factory=dict)
    trigger: str = "spontaneous"     # 触发源: spontaneous / drive / external / survival
    
    # 产生方式
    generation_method: str = "stream"  # stream / complete
    
    def __repr__(self):
        text_preview = self.text[:60].replace("\n", " ")
        return (
            f"Thought(relevance={self.relevance:.2f}, "
            f"emotion={self.dominant_emotion}, "
            f"trigger={self.trigger}, "
            f"text=\"{text_preview}...\")"
        )


@dataclass
class ActionDecision:
    """门控模块对思维的处理决定"""
    thought: Thought
    should_activate: bool            # 是否转化为行为
    action_type: str = "none"        # speak / move / gesture / none
    action_params: dict = field(default_factory=dict)
    activation_score: float = 0.0    # 激活评分（用于调试）
    threshold: float = 0.7           # 当时的阈值
    reason: str = ""                 # 决策原因（用于调试）
    _proactive: bool = False         # 是否由主动开口机制触发


class ActionGate:
    """
    行为门控 — 决定哪些内部思维变成外部行为
    
    设计原则:
    - 默认所有思维都不产生行为（人的大脑大多数想法不会说出口）
    - 驱力越高 → 阈值越低 → 更多行为
    - 外部刺激（人类说话）→ 暂时降低阈值
    - 同一类型的思维连续多次 → 累积激活（念念不忘必有回响）
    
    五层门控:
      1. 相关性评分（relevance）
      2. 动态阈值（根据身体状态）
      3. 累积激活（重复出现的思维更容易通过）
      4. 不应期抑制（刚说过的话短时间内不再重复说出口）
      5. 速率限制（无论说什么，两次输出间要有最小间隔）

    关于 3 与 4/5 的关系:
      累积激活让"反复挂念的事"更容易冲破阈值，这是有意为之；
      但冲破之后若不加抑制，同一句话会被反复输出（复读机效应）。
      不应期只作用于**已激活的输出**，类比神经元放电后的不应期：
      念头仍可持续累积，但嘴巴需要冷却。
      被 4/5 挡下的思维不计入 3 的累积，否则复读会自我强化。

    为什么需要 5:
      4 依赖字符相似度，只能抓住逐字复读。同义改写（"想感受地面纹理"
      vs "想象踏在表面上"）在中文里 bigram 相似度可以是 0.0，字面重合
      与语义相关几乎脱钩，靠调阈值只会在漏判和误伤之间摇摆。
      在接入 embedding 之前，5 提供与内容无关的兜底：不管说什么，
      话密就削。这是权宜之计，不是终局方案。
    """

    # 配置
    BASE_THRESHOLD = 0.7              # 基础阈值
    DRIVE_THRESHOLD_REDUCTION = 0.3   # 最大驱力时减少的量
    EXTERNAL_THRESHOLD_REDUCTION = 0.3 # 外部刺激时减少的量
    ACCUMULATION_WINDOW = 30.0         # 累积窗口(秒)
    ACCUMULATION_BOOST = 0.1           # 每次重复的累积提升

    # 不应期（复读抑制）
    REFRACTORY_WINDOW = 25.0           # 不应期时长(秒)
    REFRACTORY_SIMILARITY = 0.72       # 判定为"逐字复读"的相似度阈值
    REFRACTORY_DECAY = True            # 不应期内相似度容忍度随时间线性放宽

    # 输出速率限制（主题打转的兜底）
    #   同义改写的复读无法靠字符相似度识别: 实测"想感受地面纹理"与
    #   "想象踏在表面上"语义等价但 bigram 相似度为 0.0——字面重合度
    #   和语义相关性在中文里几乎脱钩，调阈值只会在漏判和误伤间摇摆。
    #   在接入 embedding 之前，用速率限制兜底: 不判断"说了什么"，
    #   只限制"多久说一次"。复读再多也漏不出去，代价是话密时会被削。
    MIN_OUTPUT_INTERVAL = 6.0          # 两次输出的最小间隔(秒)

    def __init__(self):
        self._recent_thoughts: list[Thought] = []  # 用于累积激活计算
        # 想说出口的话 [(timestamp, action_type, text)]，含被限流挡下的，用于不应期判定
        self._recent_activations: list[tuple[float, str, str]] = []
        # 上一次真正输出的时刻，用于速率限制
        self._last_output_at: float = 0.0

    def evaluate(
        self,
        thought: Thought,
        drive_level: float = 0.0,
        is_external_trigger: bool = False,
    ) -> ActionDecision:
        """
        评估一个思维是否应该转化为行为
        
        Args:
            thought: 待评估的思维
            drive_level: 当前最大驱力水平 [0, 1]
            is_external_trigger: 是否由外部刺激（人类说话）触发
        
        Returns:
            ActionDecision 包含是否激活及原因
        """
        # 1. 计算动态阈值
        threshold = self.BASE_THRESHOLD
        threshold -= drive_level * self.DRIVE_THRESHOLD_REDUCTION
        
        if is_external_trigger:
            threshold -= self.EXTERNAL_THRESHOLD_REDUCTION
        
        threshold = max(0.2, min(0.95, threshold))
        
        # 2. 累积激活：检查最近是否有类似思维
        accumulation_boost = self._compute_accumulation(thought)
        
        # 3. 最终激活分数
        activation_score = thought.relevance + accumulation_boost
        should_activate = activation_score >= threshold
        
        # 4. 决定动作类型
        action_type = self._classify_action(thought) if should_activate else "none"

        # 5. 不应期抑制：刚说过的话不重复说
        #    外部触发（有人跟它说话）时豁免——被问到就该回应，哪怕答案相似
        refractory_hit = None
        rate_limited = None
        if should_activate and not is_external_trigger:
            refractory_hit = self._check_refractory(thought.text)
            if refractory_hit is not None:
                should_activate = False
                action_type = "none"
            else:
                rate_limited = self._check_rate_limit()
                if rate_limited is not None:
                    should_activate = False
                    action_type = "none"

        # 6. 记录到近期思维（用于累积计算）
        #    被不应期挡下的思维不计入累积：否则复读会自我强化——
        #    同主题思维互相把 accumulation 推高，反而更容易冲破阈值，
        #    与抑制目标直接冲突。念头可以继续想，但不该给它加分。
        if refractory_hit is None and rate_limited is None:
            self._recent_thoughts.append(thought)
        self._trim_recent_thoughts()

        # 7. 登记历史
        #    不应期记录**所有评估过的思维**，而不只是输出成功的。
        #    否则会漏: 某条思维因分数不够被挡(未留痕)，下一轮同样的话
        #    靠 accumulation 加成刚好过线，不应期查不到前一条，逐字复读放行。
        #    速率限制则只看真正说出口的，见 _last_output_at。
        self._recent_activations.append((time.time(), action_type, thought.text))
        if should_activate:
            self._last_output_at = time.time()
        self._trim_recent_activations()

        reason = self._build_reason(
            thought, activation_score, threshold,
            drive_level, is_external_trigger, accumulation_boost,
        )
        if refractory_hit is not None:
            reason += f", refractory(sim={refractory_hit:.2f})"
        if rate_limited is not None:
            reason += f", rate_limited(gap={rate_limited:.1f}s)"

        decision = ActionDecision(
            thought=thought,
            should_activate=should_activate,
            action_type=action_type,
            activation_score=activation_score,
            threshold=threshold,
            reason=reason,
        )

        if should_activate:
            logger.info(
                f"ActionGate: ACTIVATE {action_type} "
                f"(score={activation_score:.2f} >= threshold={threshold:.2f}, "
                f"drive={drive_level:.2f}, accumulation={accumulation_boost:.2f})"
            )
        elif refractory_hit is not None:
            logger.info(
                f"ActionGate: REFRACTORY suppress "
                f"(score={activation_score:.2f} passed, but similarity={refractory_hit:.2f} "
                f"to a recent utterance) | {thought.text[:40]}"
            )
        elif rate_limited is not None:
            logger.info(
                f"ActionGate: RATE-LIMIT suppress "
                f"(score={activation_score:.2f} passed, but only {rate_limited:.1f}s "
                f"since last output, min={self.MIN_OUTPUT_INTERVAL:.0f}s) "
                f"| {thought.text[:40]}"
            )
        else:
            logger.debug(
                f"ActionGate: suppress (score={activation_score:.2f} < threshold={threshold:.2f})"
            )

        return decision

    def _check_refractory(self, text: str) -> Optional[float]:
        """
        检查是否落在不应期内。

        Returns:
            命中时返回触发抑制的相似度；未命中返回 None。
        """
        now = time.time()

        for ts, _past_action, past_text in self._recent_activations:
            age = now - ts
            if age > self.REFRACTORY_WINDOW:
                continue

            # 不按 action_type 分桶：关键字分类很脆弱，
            # "我需要去户外呼吸新鲜空气"(move) 和 "我需要呼吸新鲜空气"(speak)
            # 只因一个"去"字被判成不同动作，但对观察者而言就是复读。
            # 复读与否取决于内容，不取决于内部动作标签。
            sim = self._similarity_heuristic(past_text, text)

            # 轨道一：逐字/近逐字复读 —— 单条命中即抑制
            # 不应期前半段全强度抑制，后半段容忍度线性放宽到 1.0
            if self.REFRACTORY_DECAY:
                progress = age / self.REFRACTORY_WINDOW
                relax = max(0.0, (progress - 0.5) * 2.0)
                effective_threshold = (
                    self.REFRACTORY_SIMILARITY
                    + (1.0 - self.REFRACTORY_SIMILARITY) * relax
                )
            else:
                effective_threshold = self.REFRACTORY_SIMILARITY

            if sim >= effective_threshold:
                return sim

        return None

    def _check_rate_limit(self) -> Optional[float]:
        """
        输出速率限制。

        Returns:
            距上次输出的间隔（被限流时）；未触发返回 None。
        """
        if self._last_output_at <= 0.0:
            return None
        gap = time.time() - self._last_output_at
        return gap if gap < self.MIN_OUTPUT_INTERVAL else None

    def _compute_accumulation(self, thought: Thought) -> float:
        """计算同类思维的累积激活"""
        now = time.time()
        count = 0
        for past in self._recent_thoughts:
            if now - past.timestamp > self.ACCUMULATION_WINDOW:
                continue
            # 简单策略：相同触发源 + 相似文本长度视为同类
            if past.trigger == thought.trigger:
                # 更精确的同类判断可以用 embedding 相似度，此处先用简单策略
                if self._similarity_heuristic(past.text, thought.text) > 0.3:
                    count += 1
        return min(count * self.ACCUMULATION_BOOST, 0.5)  # 上限 0.5

    @staticmethod
    def _similarity_heuristic(a: str, b: str) -> float:
        """
        文本相似度启发式（后续可替换为 embedding）。

        中文没有空格分词，按空白切词会让"我需要呼吸新鲜空气"和
        "我需要去户外呼吸新鲜空气"相似度为 0。这里改用字符二元组(bigram)
        的 Jaccard 相似度，中英文都能工作。
        """
        def _normalize(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        na, nb = _normalize(a), _normalize(b)
        if not na or not nb:
            return 0.0
        if na == nb:
            return 1.0

        def _bigrams(s: str) -> set[str]:
            if len(s) < 2:
                return {s}
            return {s[i:i + 2] for i in range(len(s) - 1)}

        ga, gb = _bigrams(na), _bigrams(nb)
        if not ga or not gb:
            return 0.0

        overlap = len(ga & gb)
        # 用重叠系数（除以较短者）而非 Jaccard：
        # "我需要呼吸新鲜空气" vs "我需要去户外呼吸新鲜空气" 只是插入了修饰成分，
        # 语义上是同一句抱怨，Jaccard 会因长度差被稀释到 0.43 而漏判。
        return overlap / min(len(ga), len(gb))

    def _classify_action(self, thought: Thought) -> str:
        """将思维分类为动作类型"""
        text = thought.text.lower()
        # 简单的关键字分类（后续用 LLM 输出自带意图替代）
        if any(w in text for w in ["说", "讲", "告诉", "你好", "谢谢", "再见", "回答"]):
            return "speak"
        elif any(w in text for w in ["走", "去", "移动", "前进", "后退", "找", "探索"]):
            return "move"
        elif any(w in text for w in ["表情", "笑", "哭", "害怕", "开心"]):
            return "expression"
        else:
            return "speak"  # 默认说话

    def _build_reason(
        self, thought: Thought, score: float, threshold: float,
        drive: float, external: bool, accumulation: float,
    ) -> str:
        parts = []
        parts.append(f"score={score:.2f} vs threshold={threshold:.2f}")
        parts.append(f"drive_level={drive:.2f}")
        if external:
            parts.append("external_trigger")
        if accumulation > 0:
            parts.append(f"accumulation={accumulation:.2f}")
        return ", ".join(parts)

    def _trim_recent_thoughts(self):
        """清理超出累积窗口的旧思维"""
        now = time.time()
        self._recent_thoughts = [
            t for t in self._recent_thoughts
            if now - t.timestamp <= self.ACCUMULATION_WINDOW
        ]

    def _trim_recent_activations(self):
        """清理超出不应期窗口的旧输出记录"""
        now = time.time()
        self._recent_activations = [
            rec for rec in self._recent_activations
            if now - rec[0] <= self.REFRACTORY_WINDOW
        ]


class ThoughtJournal:
    """思维日志 — 将所有思维存档，用于长期记忆和反思"""

    def __init__(self, max_entries: int = 1000):
        self.thoughts: list[Thought] = []
        self.max_entries = max_entries

    def record(self, thought: Thought):
        self.thoughts.append(thought)
        if len(self.thoughts) > self.max_entries:
            self.thoughts = self.thoughts[-self.max_entries:]

    def recent(self, n: int = 10) -> list[Thought]:
        return self.thoughts[-n:]

    def by_emotion(self, emotion: str, limit: int = 10) -> list[Thought]:
        return [t for t in self.thoughts if t.dominant_emotion == emotion][-limit:]

    def activated_count(self, window_seconds: float = 60) -> int:
        """最近 N 秒内转化为行为的思维数量"""
        now = time.time()
        return sum(
            1 for t in self.thoughts
            if now - t.timestamp <= window_seconds and t.relevance >= 0.5
        )

    def summary(self) -> dict:
        """思维统计摘要"""
        if not self.thoughts:
            return {"total": 0}
        emotions = {}
        for t in self.thoughts:
            emotions[t.dominant_emotion] = emotions.get(t.dominant_emotion, 0) + 1
        return {
            "total": len(self.thoughts),
            "recent_activated": self.activated_count(),
            "emotions": emotions,
            "avg_relevance": sum(t.relevance for t in self.thoughts) / len(self.thoughts),
        }
