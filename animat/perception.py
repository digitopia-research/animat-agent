"""
感知运动层 v2 — 外周神经系统

核心变更（vs v0.1）:
  1. 保留反射弧：突发危险 → 绕过 LLM 的快速响应（不变，这是对的）
  2. 感受器映射降级为"注意力提示"：不再做 pain/caress 的定性判断，
     而是告诉 LLM "注意看某通道有个急变"，让 LLM 自己从原始序列判断
  3. 传感器原始数据写入 SensorWindow → 直接暴露给 LLM
  4. 导数计算保留（供反射弧和注意力提示使用）

与 v0.1 的区别:
  v0.1:   压力值 → dP/dt > 0.8 → 发射 "pain" 事件 → 边缘系统处理
  v0.2:   压力值 → 写入 SensorWindow → 同时
             ├→ 反射弧检查（保留）：dP/dt > 0.8 → 直接缩回
             └→ 注意力提示：发送 AttentionHint 到 LLM上下文
           LLM 自己从 [T+0.0, touch: P=0.05], [T+0.1, touch: P=0.88], ...
           读取原始序列，自己判断"这是个撞击"

订阅: animat/sensor/*
发布: animat/reflex (保留), animat/feeling (保留)
新增: sensor_window 写入
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from animat.config import Topic, PerceptionConfig, RedisKeys
from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.sensor_window import SensorWindow, AttentionHint

logger = logging.getLogger("animat.perception")


# ─────────────── 感受质定义（保留，用于反射弧和注意力提示）───────────────
class FeelingType(str, Enum):
    PAIN       = "pain"
    CARESS     = "caress"
    PRESSURE   = "pressure"
    STARTLE    = "startle"
    HOSTILE    = "hostile"
    SOOTHING   = "soothing"
    APPROACH   = "approach"
    GAZE       = "gaze"
    ABANDON    = "abandon"
    FATIGUE    = "fatigue"
    RECHARGING = "recharging"
    IMBALANCE  = "imbalance"


@dataclass
class Feeling:
    """一个感受事件（保留，用于边缘系统兼容）"""
    type: FeelingType
    intensity: float
    source: str
    raw_data: dict
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "intensity": self.intensity,
            "source": self.source,
            "raw_data": self.raw_data,
            "timestamp": self.timestamp,
        }


class PerceptionLayer:
    """
    感知运动层 v2
    
    职责:
      1. 写入传感器原始数据到 SensorWindow
      2. 生成注意力提示（供 LLM 参考，不替代 LLM 的判断）
      3. 反射弧（突发危险快速响应，不变）
      4. 向边缘系统发射感受事件（保留，homeostasis 仍需要）
    """

    def __init__(
        self,
        nervous: NervousSystem,
        memory: MemoryStore,
        sensor_window: Optional[SensorWindow] = None,
    ):
        self.nervous = nervous
        self.memory = memory
        self.sensor_window = sensor_window or SensorWindow()
        self._prev_sensor_data: dict[str, dict] = {}
        self._last_human_detected: Optional[float] = None
        self._prev_battery: float = 100.0

    async def register(self):
        """注册所有感觉神经末梢"""
        await self.nervous.listen(Topic.SENSOR_TOUCH, self._on_touch)
        await self.nervous.listen(Topic.SENSOR_AUDIO, self._on_audio)
        await self.nervous.listen(Topic.SENSOR_VISION, self._on_vision)
        await self.nervous.listen(Topic.SENSOR_INTERO, self._on_interoception)
        logger.info("Perception layer v2 neurons registered")

    # ── 导数计算（保留）──
    async def _compute_derivative(self, key: str, current: float) -> float:
        prev_data = await self.memory.redis.hget(RedisKeys.SENSOR_DERIVATIVES, key)
        prev_val = float(prev_data) if prev_data else current
        derivative = current - prev_val
        await self.memory.redis.hset(RedisKeys.SENSOR_DERIVATIVES, key, str(current))
        return derivative

    # ── 发布感受（保留，边缘系统仍需要）──
    async def _emit_feeling(self, feeling: Feeling):
        logger.info(f"Feeling: {feeling.type.value} @ {feeling.intensity:.2f} from {feeling.source}")
        await self.nervous.fire(Topic.FEELING, feeling.to_dict())

    # ── 触觉处理 ──
    async def _on_touch(self, topic: str, payload: dict):
        """触觉：写入 SensorWindow + 注意力提示 + 反射弧"""
        pressure = payload.get("pressure", 0)
        capacitance = payload.get("capacitance", 0)
        now = time.time()

        # v0.2: 原始数据直接写入 SensorWindow
        self.sensor_window.add_frame("touch", payload)

        dp_dt = await self._compute_derivative("touch_pressure", pressure)

        # ── 反射弧（保留）──
        if dp_dt > PerceptionConfig.TOUCH_PAIN_THRESHOLD:
            # 注意力提示（供 LLM 参考，不替代 LLM 判断）
            self.sensor_window.add_attention_hint(AttentionHint(
                modality="touch",
                hint_type="pain",
                confidence=min(dp_dt / 2.0, 1.0),
                description="触觉急变（可能撞击）",
            ))
            # 仍发射感受事件（边缘系统需要）
            await self._emit_feeling(Feeling(
                type=FeelingType.PAIN,
                intensity=min(dp_dt, 1.0),
                source="touch",
                raw_data=payload,
                timestamp=now,
            ))
            # 反射弧：绕过 LLM 缩回
            try:
                await self.nervous.reflex_fire(Topic.REFLEX, {
                    "action": "withdraw",
                    "intensity": min(dp_dt, 1.0),
                    "timestamp": now,
                })
            except Exception:
                logger.error("Reflex 'withdraw' failed", exc_info=True)
            return

        # 低压 + 电容 → 可能抚摸（注意力提示）
        if pressure < PerceptionConfig.TOUCH_CARE_THRESHOLD and capacitance > 0.1:
            self.sensor_window.add_attention_hint(AttentionHint(
                modality="touch",
                hint_type="caress",
                confidence=capacitance,
                description="轻柔接触（可能抚摸）",
            ))
            await self._emit_feeling(Feeling(
                type=FeelingType.CARESS,
                intensity=capacitance,
                source="touch",
                raw_data=payload,
                timestamp=now,
            ))
            return

        # 持续高压
        if pressure > 0.7:
            await self._emit_feeling(Feeling(
                type=FeelingType.PRESSURE,
                intensity=pressure,
                source="touch",
                raw_data=payload,
                timestamp=now,
            ))

    # ── 听觉处理 ──
    async def _on_audio(self, topic: str, payload: dict):
        """听觉：写入 SensorWindow + 注意力提示 + 反射弧"""
        db = payload.get("volume_db", 0)
        freq = payload.get("fundamental_freq", 0)
        now = time.time()

        # v0.2: 原始数据直写
        self.sensor_window.add_frame("audio", payload)

        d_db = await self._compute_derivative("audio_db", db)

        # 反射弧：巨响 → 防御
        if d_db > PerceptionConfig.AUDIO_STARTLE_DB:
            self.sensor_window.add_attention_hint(AttentionHint(
                modality="audio",
                hint_type="startle",
                confidence=min(d_db / 100, 1.0),
                description="音量突变（可能巨响）",
            ))
            await self._emit_feeling(Feeling(
                type=FeelingType.STARTLE,
                intensity=min(d_db / 100, 1.0),
                source="audio",
                raw_data=payload,
                timestamp=now,
            ))
            try:
                await self.nervous.reflex_fire(Topic.REFLEX, {
                    "action": "flinch",
                    "intensity": min(d_db / 100, 1.0),
                    "timestamp": now,
                })
            except Exception:
                logger.error("Reflex 'flinch' failed", exc_info=True)
            return

        # 高频尖叫
        if freq > PerceptionConfig.AUDIO_HOSTILE_FREQ and db > 70:
            await self._emit_feeling(Feeling(
                type=FeelingType.HOSTILE,
                intensity=min((freq / 1000 + db / 100) / 2, 1.0),
                source="audio",
                raw_data=payload,
                timestamp=now,
            ))
            return

        # 低频安抚
        if db < PerceptionConfig.AUDIO_SOOTHING_DB and freq < 300:
            await self._emit_feeling(Feeling(
                type=FeelingType.SOOTHING,
                intensity=0.3,
                source="audio",
                raw_data=payload,
                timestamp=now,
            ))

    # ── 视觉处理 ──
    async def _on_vision(self, topic: str, payload: dict):
        """视觉：写入 SensorWindow + 注意力提示"""
        distance = payload.get("distance_m", 10)
        face = payload.get("face_detected", False)
        gaze = payload.get("gaze_seconds", 0)
        now = time.time()

        self.sensor_window.add_frame("vision", payload)

        d_dist = await self._compute_derivative("vision_distance", distance)

        # 逼近
        if d_dist < -PerceptionConfig.VISION_APPROACH_SPEED:
            self.sensor_window.add_attention_hint(AttentionHint(
                modality="vision",
                hint_type="approach",
                confidence=min(abs(d_dist) / 2, 1.0),
                description="物体快速逼近",
            ))
            await self._emit_feeling(Feeling(
                type=FeelingType.APPROACH,
                intensity=min(abs(d_dist) / 2, 1.0),
                source="vision",
                raw_data=payload,
                timestamp=now,
            ))

        # 凝视
        if face and gaze > PerceptionConfig.VISION_GAZE_DURATION:
            await self._emit_feeling(Feeling(
                type=FeelingType.GAZE,
                intensity=min(gaze / 5, 1.0),
                source="vision",
                raw_data=payload,
                timestamp=now,
            ))

        # 遗弃
        if face:
            self._last_human_detected = now
        elif self._last_human_detected:
            elapsed = now - self._last_human_detected
            if elapsed > PerceptionConfig.VISION_ABANDON_TIME:
                self.sensor_window.add_attention_hint(AttentionHint(
                    modality="vision",
                    hint_type="abandon",
                    confidence=min(elapsed / 600, 1.0),
                    description=f"长时间({elapsed:.0f}s)无人",
                ))
                await self._emit_feeling(Feeling(
                    type=FeelingType.ABANDON,
                    intensity=min(elapsed / 600, 1.0),
                    source="vision",
                    raw_data=payload,
                    timestamp=now,
                ))

    # ── 内感处理 ──
    async def _on_interoception(self, topic: str, payload: dict):
        """内感：写入 SensorWindow + 直通稳态"""
        battery = payload.get("battery_pct", 100)
        tilt = payload.get("imu_tilt_deg", 0)
        now = time.time()

        self.sensor_window.add_frame("intero", payload)

        # 直通同步：电量 → 生存度（不变）
        from animat.homeostasis import Vital
        await self.memory.redis.hset(
            RedisKeys.STATE_HOMEOSTASIS, Vital.VIABILITY.value, str(battery)
        )

        # 充电检测
        if battery > self._prev_battery + 0.5:
            delta = battery - self._prev_battery
            await self._emit_feeling(Feeling(
                type=FeelingType.RECHARGING,
                intensity=min(delta / 5.0, 1.0),
                source="interoception",
                raw_data=payload,
                timestamp=now,
            ))
        elif battery < PerceptionConfig.INTERO_LOW_BATTERY:
            await self._emit_feeling(Feeling(
                type=FeelingType.FATIGUE,
                intensity=1.0 - (battery / PerceptionConfig.INTERO_LOW_BATTERY),
                source="interoception",
                raw_data=payload,
                timestamp=now,
            ))

        self._prev_battery = battery

        # 失衡
        if tilt > PerceptionConfig.INTERO_IMU_TILT:
            self.sensor_window.add_attention_hint(AttentionHint(
                modality="intero",
                hint_type="imbalance",
                confidence=min(tilt / 90, 1.0),
                description="身体失衡",
            ))
            await self._emit_feeling(Feeling(
                type=FeelingType.IMBALANCE,
                intensity=min(tilt / 90, 1.0),
                source="interoception",
                raw_data=payload,
                timestamp=now,
            ))
            try:
                await self.nervous.reflex_fire(Topic.REFLEX, {
                    "action": "stabilize",
                    "intensity": min(tilt / 90, 1.0),
                    "timestamp": now,
                })
            except Exception:
                logger.error("Reflex 'stabilize' failed", exc_info=True)
