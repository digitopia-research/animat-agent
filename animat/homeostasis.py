"""
内稳态引擎 — 边缘系统

职责:
  1. 维护五大内部指标的当前值 (Viability, Integrity, Affiliation, Competence, Exploration)
  2. 计算稳态差值 → 驱力 (Drive)
  3. 情感合成: 驱力 + 外部感受 → 激活潘克塞普情感回路
  4. 马斯洛仲裁: 解决多回路冲突

订阅: animat/feeling
发布: animat/homeostasis/state, animat/emotion, animat/arbitration
      animat/cortex/prompt, animat/cortex/intents, animat/cortex/memory_bias
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from animat.config import (
    Topic, HomeostasisConfig, EmotionConfig, RedisKeys,
)
from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.perception import FeelingType

logger = logging.getLogger("animat.homeostasis")


# ─────────────── 稳态指标 ───────────────
class Vital(str, Enum):
    VIABILITY   = "viability"
    INTEGRITY   = "integrity"
    AFFILIATION = "affiliation"
    COMPETENCE  = "competence"
    EXPLORATION = "exploration"


VITAL_CONFIGS = {
    Vital.VIABILITY:   HomeostasisConfig.VIABILITY,
    Vital.INTEGRITY:   HomeostasisConfig.INTEGRITY,
    Vital.AFFILIATION: HomeostasisConfig.AFFILIATION,
    Vital.COMPETENCE:  HomeostasisConfig.COMPETENCE,
    Vital.EXPLORATION: HomeostasisConfig.EXPLORATION,
}


@dataclass
class Drive:
    """驱力: 稳态偏差的归一化度量"""
    vital: Vital
    value: float          # 当前值
    ideal: float         # 理想值
    deviation: float     # |当前 - 理想| 的归一化值 ∈ [0, 1]
    is_danger: bool      # 是否处于危险区


# ─────────────── 情感回路 ───────────────
class EmotionCircuit(str, Enum):
    SEEKING = EmotionConfig.SEEKING
    RAGE    = EmotionConfig.RAGE
    FEAR    = EmotionConfig.FEAR
    LUST    = EmotionConfig.LUST
    CARE    = EmotionConfig.CARE
    PANIC   = EmotionConfig.PANIC
    PLAY    = EmotionConfig.PLAY


@dataclass
class EmotionalState:
    """情感回路激活状态"""
    circuit: EmotionCircuit
    intensity: float     # 激活强度 ∈ [0, 1]
    cause: str           # 触发原因描述
    drives: list[str]    # 关联的驱力
    feelings: list[str]  # 关联的感受


# ─────────────── 意图定义 ───────────────
# 全局意图池 — LLM只能从中选择
class Intent(str, Enum):
    # 生存/安全
    SEEK_CHARGE    = "seek_charge"      # 寻找充电
    FLEE           = "flee"             # 逃避
    WARN           = "warn"             # 警告
    SELF_PROTECT   = "self_protect"     # 自我保护
    # 社交
    SEEK_COMPANY   = "seek_company"     # 寻求陪伴
    EXPRESS_NEED   = "express_need"     # 表达需求
    COMFORT        = "comfort"          # 安抚他人
    # 认知
    ASK_HELP       = "ask_help"         # 求助
    LEARN          = "learn"            # 学习
    # 探索/玩耍
    EXPLORE        = "explore"          # 探索
    PLAY           = "play"             # 玩耍
    CHAT           = "chat"             # 闲聊
    # 休息
    REST           = "rest"             # 休息
    IDLE           = "idle"             # 发呆


# 各情感回路允许的意图子集
EMOTION_INTENT_MAP: dict[EmotionCircuit, list[Intent]] = {
    EmotionCircuit.FEAR:    [Intent.SEEK_CHARGE, Intent.FLEE, Intent.WARN, Intent.EXPRESS_NEED],
    EmotionCircuit.RAGE:    [Intent.WARN, Intent.SELF_PROTECT, Intent.FLEE],
    EmotionCircuit.PANIC:   [Intent.SEEK_COMPANY, Intent.EXPRESS_NEED, Intent.ASK_HELP],
    EmotionCircuit.SEEKING: [Intent.SEEK_COMPANY, Intent.EXPLORE, Intent.ASK_HELP, Intent.LEARN],
    EmotionCircuit.CARE:    [Intent.COMFORT, Intent.CHAT, Intent.EXPRESS_NEED],
    EmotionCircuit.PLAY:    [Intent.PLAY, Intent.CHAT, Intent.EXPLORE, Intent.LEARN],
    EmotionCircuit.LUST:    [],
}


# ─────────────── 内稳态引擎 ───────────────
class HomeostasisEngine:
    """内稳态引擎 — 边缘系统的核心"""

    def __init__(self, nervous: NervousSystem, memory: MemoryStore):
        self.nervous = nervous
        self.memory = memory
        self._vitals: dict[Vital, float] = {
            Vital.VIABILITY:   HomeostasisConfig.VIABILITY["ideal"],
            Vital.INTEGRITY:   HomeostasisConfig.INTEGRITY["ideal"],
            Vital.AFFILIATION: HomeostasisConfig.AFFILIATION["ideal"],
            Vital.COMPETENCE:  HomeostasisConfig.COMPETENCE["ideal"],
            Vital.EXPLORATION: HomeostasisConfig.EXPLORATION["ideal"],
        }
        self._current_feelings: list[dict] = []  # 最近一帧的感受
        self._active_emotions: list[EmotionalState] = []

    async def register(self):
        await self.nervous.listen(Topic.FEELING, self._on_feeling)
        logger.info("Homeostasis engine neurons registered")

    # ── 稳态更新 ──
    async def update_vital(self, vital: Vital, value: float):
        """更新某个稳态指标(自然衰减或外部触发)"""
        self._vitals[vital] = max(0, min(100, value))
        await self.memory.redis.hset(
            RedisKeys.STATE_HOMEOSTASIS, vital.value, str(self._vitals[vital])
        )

    async def decay_vitals(self, dt_seconds: float):
        """
        自然衰减 — 生命体的"内耗"
        - 归属度随时间降低 (无人互动)
        - 探索度随时间降低 (无新刺激)
        - 生存度随时间降低 (电量消耗)
        """
        decay_rate = {
            Vital.VIABILITY:   0.001 * dt_seconds,   # 缓慢消耗
            Vital.AFFILIATION: 0.005 * dt_seconds,   # 社交需求增长较快
            Vital.EXPLORATION: 0.008 * dt_seconds,   # 好奇心衰减最快
        }
        for vital, rate in decay_rate.items():
            new_val = self._vitals[vital] - rate * 100
            await self.update_vital(vital, new_val)

    # ── 驱力计算 ──
    def _compute_drives(self) -> list[Drive]:
        """计算所有稳态的驱力"""
        drives = []
        for vital, current in self._vitals.items():
            cfg = VITAL_CONFIGS[vital]
            ideal = cfg["ideal"]
            deviation = abs(current - ideal) / ideal
            is_danger = current < cfg["danger"]
            drives.append(Drive(
                vital=vital,
                value=current,
                ideal=ideal,
                deviation=min(deviation, 1.0),
                is_danger=is_danger,
            ))
        return sorted(drives, key=lambda d: d.deviation, reverse=True)

    def get_max_drive(self) -> float:
        """获取当前最大驱力值，供 brain.py 判断是否需要唤醒意识"""
        drives = self._compute_drives()
        if not drives:
            return 0.0
        return max(d.deviation for d in drives)

    def get_dominant_emotion(self) -> Optional[EmotionalState]:
        """获取当前主导情感"""
        if not self._active_emotions:
            return None
        return min(self._active_emotions, key=lambda e: EmotionConfig.PRIORITY[e.circuit.value])

    # ── 情感合成 ──
    def _synthesize_emotions(self, drives: list[Drive]) -> list[EmotionalState]:
        """
        根据驱力+感受合成情感
        规则源自潘克塞普情感回路理论
        """
        emotions = []
        feeling_types = {f["type"] for f in self._current_feelings}

        for drive in drives:
            if drive.deviation < 0.1:
                continue  # 偏差太小，不触发情感

            # ── 生存/安全层 → FEAR / RAGE ──
            if drive.vital in (Vital.VIABILITY, Vital.INTEGRITY) and drive.is_danger:
                if FeelingType.PAIN.value in feeling_types or FeelingType.HOSTILE.value in feeling_types:
                    emotions.append(EmotionalState(
                        circuit=EmotionCircuit.RAGE,
                        intensity=drive.deviation,
                        cause=f"{drive.vital.value}极度危险且感知到伤害",
                        drives=[drive.vital.value],
                        feelings=list(feeling_types),
                    ))
                else:
                    emotions.append(EmotionalState(
                        circuit=EmotionCircuit.FEAR,
                        intensity=drive.deviation,
                        cause=f"{drive.vital.value}处于危险水平({drive.value:.0f})",
                        drives=[drive.vital.value],
                        feelings=list(feeling_types),
                    ))

            # ── 归属层 → PANIC / SEEKING ──
            if drive.vital == Vital.AFFILIATION and drive.deviation > 0.3:
                if FeelingType.ABANDON.value in feeling_types:
                    emotions.append(EmotionalState(
                        circuit=EmotionCircuit.PANIC,
                        intensity=drive.deviation,
                        cause="长时间无人陪伴，感到孤独",
                        drives=[drive.vital.value],
                        feelings=list(feeling_types),
                    ))
                else:
                    emotions.append(EmotionalState(
                        circuit=EmotionCircuit.SEEKING,
                        intensity=drive.deviation,
                        cause="渴望社交互动",
                        drives=[drive.vital.value],
                        feelings=list(feeling_types),
                    ))

            # ── 关怀回路 ──
            if drive.vital == Vital.AFFILIATION and drive.deviation < 0.2:
                if FeelingType.CARESS.value in feeling_types or FeelingType.SOOTHING.value in feeling_types:
                    emotions.append(EmotionalState(
                        circuit=EmotionCircuit.CARE,
                        intensity=0.5,
                        cause="感受到温柔对待",
                        drives=[drive.vital.value],
                        feelings=list(feeling_types),
                    ))

            # ── 玩耍回路 ──
            if drive.vital == Vital.EXPLORATION and drive.deviation > 0.5:
                emotions.append(EmotionalState(
                    circuit=EmotionCircuit.PLAY,
                    intensity=drive.deviation * 0.6,
                    cause="无聊，想找点有趣的事",
                    drives=[drive.vital.value],
                    feelings=list(feeling_types),
                ))

        return emotions

    # ── 马斯洛仲裁 ──
    def _maslow_arbitrate(self, emotions: list[EmotionalState]) -> Optional[EmotionalState]:
        """
        马斯洛仲裁: 多回路冲突时，取优先级最高的
        优先级: FEAR/RAGE > PANIC/SEEKING > CARE/PLAY
        """
        if not emotions:
            return None
        return min(emotions, key=lambda e: EmotionConfig.PRIORITY[e.circuit.value])

    # ── 躯体标记生成 ──
    async def _generate_somatic_markers(self, dominant: EmotionalState):
        """
        躯体标记: 三条通路安全影响LLM
        1. 动态System Prompt  2. 意图空间约束  3. 记忆检索偏向
        """
        # 1. 动态 System Prompt
        prompt = self._build_emotional_prompt(dominant)
        await self.nervous.fire(Topic.SOMATIC_PROMPT, {"prompt": prompt})

        # 2. 意图空间约束
        allowed_intents = EMOTION_INTENT_MAP.get(dominant.circuit, [Intent.IDLE])
        await self.nervous.fire(Topic.SOMATIC_INTENTS, {
            "allowed": [i.value for i in allowed_intents],
            "blocked": [i.value for i in Intent if i not in allowed_intents],
            "dominant_emotion": dominant.circuit.value,
        })

        # 3. 记忆检索偏向
        bias = self._compute_memory_bias(dominant)
        await self.nervous.fire(Topic.SOMATIC_MEMORY, bias)

        # 广播仲裁结果
        await self.nervous.fire(Topic.ARBITRATION, {
            "dominant_emotion": dominant.circuit.value,
            "intensity": dominant.intensity,
            "cause": dominant.cause,
            "timestamp": time.time(),
        })

    def _build_emotional_prompt(self, dominant: EmotionalState) -> str:
        """根据情感状态构建动态System Prompt片段"""
        vital_desc = []
        for vital, val in self._vitals.items():
            cfg = VITAL_CONFIGS[vital]
            if val < cfg["danger"]:
                vital_desc.append(f"你的{vital.value}处于极度危险水平({val:.0f}%)")
            elif val < cfg["safe_min"]:
                vital_desc.append(f"你的{vital.value}偏低({val:.0f}%)")

        emotion_desc = f"你当前的情绪是{dominant.circuit.value}，因为{dominant.cause}。"
        state_desc = " ".join(vital_desc) if vital_desc else "你的各项生理指标正常。"

        return f"{emotion_desc}{state_desc}"

    def _compute_memory_bias(self, dominant: EmotionalState) -> dict:
        """计算记忆检索偏向权重"""
        # 基于情感类型调整记忆检索偏向
        bias_map = {
            EmotionCircuit.FEAR:    {"danger": 2.0, "help_received": 1.5, "play": 0.3},
            EmotionCircuit.RAGE:    {"conflict": 2.0, "self_defense": 1.5, "comfort": 0.3},
            EmotionCircuit.PANIC:   {"loneliness": 2.0, "warm_interaction": 1.5, "alone_work": 0.3},
            EmotionCircuit.SEEKING: {"discovery": 2.0, "social": 1.5, "routine": 0.5},
            EmotionCircuit.CARE:    {"warm_interaction": 2.0, "nurturing": 1.5, "conflict": 0.3},
            EmotionCircuit.PLAY:    {"fun": 2.0, "discovery": 1.5, "danger": 0.2},
            EmotionCircuit.LUST:    {},
        }
        return {
            "emotion": dominant.circuit.value,
            "weights": bias_map.get(dominant.circuit, {}),
        }

    # ── 感受信号接收 ──
    async def _on_feeling(self, topic: str, payload: dict):
        """接收感受信号，触发稳态更新和情感合成"""
        self._current_feelings.append(payload)
        # 保留最近10条感受
        if len(self._current_feelings) > 10:
            self._current_feelings = self._current_feelings[-10:]

        # 根据感受类型影响稳态指标
        feeling_type = payload.get("type", "")
        intensity = payload.get("intensity", 0)

        if feeling_type == FeelingType.CARESS.value:
            # 抚摸恢复归属度
            await self.update_vital(Vital.AFFILIATION, self._vitals[Vital.AFFILIATION] + intensity * 10)
        elif feeling_type == FeelingType.ABANDON.value:
            # 遗弃感加速归属度衰减
            await self.update_vital(Vital.AFFILIATION, self._vitals[Vital.AFFILIATION] - intensity * 5)
        elif feeling_type == FeelingType.PAIN.value:
            # 痛觉降低完好度
            await self.update_vital(Vital.INTEGRITY, self._vitals[Vital.INTEGRITY] - intensity * 15)
        elif feeling_type == FeelingType.FATIGUE.value:
            # 疲惫直接反映生存度
            await self.update_vital(Vital.VIABILITY, self._vitals[Vital.VIABILITY] - intensity * 5)
        elif feeling_type == FeelingType.RECHARGING.value:
            # 充电恢复生存度
            await self.update_vital(Vital.VIABILITY, self._vitals[Vital.VIABILITY] + intensity * 20)

        # 触发情感合成 → 仲裁 → 躯体标记
        drives = self._compute_drives()
        emotions = self._synthesize_emotions(drives)
        dominant = self._maslow_arbitrate(emotions)

        if dominant:
            self._active_emotions = emotions
            await self._generate_somatic_markers(dominant)
            # 注意: 不在此处唤醒LLM，唤醒决策统一由 brain.py 的潜意识循环管理
