"""
感知运动层 — 外周神经系统

职责:
  1. 感受器映射: 原始传感器数据 → 心理感受 (Qualia)
  2. 反射弧: 突发危险 → 绕过大脑的快速响应
  3. 导数计算: 追踪变化率，存入Redis供后续使用

订阅: animat/sensor/*
发布: animat/feeling, animat/reflex
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from animat.config import Topic, PerceptionConfig, RedisKeys
from animat.nervous import NervousSystem
from animat.memory import MemoryStore

logger = logging.getLogger("animat.perception")


# ─────────────── 感受质 (Qualia) 定义 ───────────────
class FeelingType(str, Enum):
    """所有可能的感受类型"""
    # 触觉
    PAIN       = "pain"         # 痛觉
    CARESS     = "caress"       # 抚摸
    PRESSURE   = "pressure"     # 压迫
    # 听觉
    STARTLE    = "startle"      # 惊吓
    HOSTILE    = "hostile"      # 敌意
    SOOTHING   = "soothing"     # 安抚
    # 视觉
    APPROACH   = "approach"     # 逼近
    GAZE       = "gaze"         # 凝视
    ABANDON    = "abandon"      # 遗弃
    # 内感
    FATIGUE    = "fatigue"      # 疲惫
    RECHARGING = "recharging"   # 充电中（生存度恢复）
    IMBALANCE  = "imbalance"    # 失衡


@dataclass
class Feeling:
    """一个感受事件"""
    type: FeelingType
    intensity: float        # 强度 ∈ [0, 1]
    source: str             # 来源传感器ID
    raw_data: dict          # 原始数据快照
    timestamp: float        # 时间戳

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "intensity": self.intensity,
            "source": self.source,
            "raw_data": self.raw_data,
            "timestamp": self.timestamp,
        }


# ─────────────── 感受器映射器 ───────────────
class PerceptionLayer:
    """感知运动层 — 将原始数据映射为感受"""

    def __init__(self, nervous: NervousSystem, memory: MemoryStore):
        self.nervous = nervous
        self.memory = memory
        self._prev_sensor_data: dict[str, dict] = {}  # 上一帧数据(用于导数计算)
        self._last_human_detected: Optional[float] = None  # 上次检测到人类的时间
        self._prev_battery: float = 100.0  # 上一帧电量(用于检测充电状态)

    async def register(self):
        """注册所有感觉神经末梢"""
        await self.nervous.listen(Topic.SENSOR_TOUCH, self._on_touch)
        await self.nervous.listen(Topic.SENSOR_AUDIO, self._on_audio)
        await self.nervous.listen(Topic.SENSOR_VISION, self._on_vision)
        await self.nervous.listen(Topic.SENSOR_INTERO, self._on_interoception)
        logger.info("Perception layer neurons registered")

    # ── 导数计算 ──
    async def _compute_derivative(self, key: str, current: float) -> float:
        """计算某个传感器值的导数(变化率)，并存储到Redis"""
        prev_data = await self.memory.redis.hget(RedisKeys.SENSOR_DERIVATIVES, key)
        prev_val = float(prev_data) if prev_data else current
        derivative = current - prev_val
        await self.memory.redis.hset(RedisKeys.SENSOR_DERIVATIVES, key, str(current))
        return derivative

    # ── 发布感受 ──
    async def _emit_feeling(self, feeling: Feeling):
        """将感受信号发射到神经总线"""
        logger.info(f"Feeling: {feeling.type.value} @ {feeling.intensity:.2f} from {feeling.source}")
        await self.nervous.fire(Topic.FEELING, feeling.to_dict())

    # ── 触觉处理 ──
    async def _on_touch(self, topic: str, payload: dict):
        """
        触觉感受器映射
        输入: {"pressure": float, "capacitance": float, "x": float, "y": float}
        """
        pressure = payload.get("pressure", 0)
        capacitance = payload.get("capacitance", 0)
        now = time.time()

        dp_dt = await self._compute_derivative("touch_pressure", pressure)

        # 痛觉: 瞬间大力
        if dp_dt > PerceptionConfig.TOUCH_PAIN_THRESHOLD:
            await self._emit_feeling(Feeling(
                type=FeelingType.PAIN,
                intensity=min(dp_dt, 1.0),
                source="touch",
                raw_data=payload,
                timestamp=now,
            ))
            # 反射弧: 痛觉触发缩回(绕过大脑)
            await self.nervous.reflex_fire(Topic.REFLEX, {
                "action": "withdraw",
                "intensity": min(dp_dt, 1.0),
                "timestamp": now,
            })
            return

        # 抚摸: 低压+电容变化(滑动轨迹)
        if (pressure < PerceptionConfig.TOUCH_CARE_THRESHOLD
                and capacitance > 0.1):
            await self._emit_feeling(Feeling(
                type=FeelingType.CARESS,
                intensity=capacitance,
                source="touch",
                raw_data=payload,
                timestamp=now,
            ))
            return

        # 压迫: 持续高压
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
        """
        听觉感受器映射
        输入: {"volume_db": float, "fundamental_freq": float, "text": str}
        """
        db = payload.get("volume_db", 0)
        freq = payload.get("fundamental_freq", 0)
        now = time.time()

        d_db = await self._compute_derivative("audio_db", db)

        # 惊吓: 巨响突变
        if d_db > PerceptionConfig.AUDIO_STARTLE_DB:
            await self._emit_feeling(Feeling(
                type=FeelingType.STARTLE,
                intensity=min(d_db / 100, 1.0),
                source="audio",
                raw_data=payload,
                timestamp=now,
            ))
            # 反射弧: 惊吓触发防御姿态
            await self.nervous.reflex_fire(Topic.REFLEX, {
                "action": "flinch",
                "intensity": min(d_db / 100, 1.0),
                "timestamp": now,
            })
            return

        # 敌意: 高频尖叫
        if freq > PerceptionConfig.AUDIO_HOSTILE_FREQ and db > 70:
            await self._emit_feeling(Feeling(
                type=FeelingType.HOSTILE,
                intensity=min((freq / 1000 + db / 100) / 2, 1.0),
                source="audio",
                raw_data=payload,
                timestamp=now,
            ))
            return

        # 安抚: 低频平稳
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
        """
        视觉感受器映射
        输入: {"distance_m": float, "face_detected": bool, "gaze_seconds": float}
        """
        distance = payload.get("distance_m", 10)
        face = payload.get("face_detected", False)
        gaze = payload.get("gaze_seconds", 0)
        now = time.time()

        d_dist = await self._compute_derivative("vision_distance", distance)

        # 逼近: 距离快速缩小
        if d_dist < -PerceptionConfig.VISION_APPROACH_SPEED:
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

        # 遗弃: 长时间无人
        if face:
            self._last_human_detected = now
        elif self._last_human_detected:
            elapsed = now - self._last_human_detected
            if elapsed > PerceptionConfig.VISION_ABANDON_TIME:
                await self._emit_feeling(Feeling(
                    type=FeelingType.ABANDON,
                    intensity=min(elapsed / 600, 1.0),
                    source="vision",
                    raw_data=payload,
                    timestamp=now,
                ))

    # ── 内感处理 ──
    async def _on_interoception(self, topic: str, payload: dict):
        """
        内感感受器映射
        输入: {"battery_pct": float, "temperature": float, "imu_tilt_deg": float}
        """
        battery = payload.get("battery_pct", 100)
        tilt = payload.get("imu_tilt_deg", 0)
        now = time.time()

        # 直通同步: 电量直接映射到生存度 (viability)
        # 这是内感的核心——身体状态直接决定稳态，而非仅通过感受间接影响
        from animat.homeostasis import Vital
        from animat.config import RedisKeys
        await self.memory.redis.hset(
            RedisKeys.STATE_HOMEOSTASIS, Vital.VIABILITY.value, str(battery)
        )

        # 充电检测: 电量正在上升 → RECHARGING 感受
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
            # 疲惫/饥饿: 电量低
            await self._emit_feeling(Feeling(
                type=FeelingType.FATIGUE,
                intensity=1.0 - (battery / PerceptionConfig.INTERO_LOW_BATTERY),
                source="interoception",
                raw_data=payload,
                timestamp=now,
            ))

        self._prev_battery = battery

        # 失衡: 姿态偏差
        if tilt > PerceptionConfig.INTERO_IMU_TILT:
            await self._emit_feeling(Feeling(
                type=FeelingType.IMBALANCE,
                intensity=min(tilt / 90, 1.0),
                source="interoception",
                raw_data=payload,
                timestamp=now,
            ))
            # 反射弧: 失衡触发平衡修正
            await self.nervous.reflex_fire(Topic.REFLEX, {
                "action": "stabilize",
                "intensity": min(tilt / 90, 1.0),
                "timestamp": now,
            })
