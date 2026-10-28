"""
双进程脑 v2 — 适配恒常意识流

核心变更（vs v0.1 brain.py）:
  1. StreamCortex 恒常运行（不唤醒/沉睡）
  2. 潜意识循环职责简化：只有稳态衰减 + 传感器同步
  3. 心跳循环保留：广播身体状态
  4. 无聊/驱力检测不再唤醒 LLM，而是调整 _drive_level
     （StreamCortex 自己根据驱力水平决定思维频率和内容）
"""

import asyncio
import logging
import time

from animat.config import BrainConfig, Topic, RedisKeys
from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.perception import PerceptionLayer
from animat.homeostasis import HomeostasisEngine, Vital
from animat.stream_cortex import StreamCortex
from animat.action import ActionExecutor
from animat.llm_backend import LocalLLM, create_llm
from animat.sensor_window import SensorWindow

logger = logging.getLogger("animat.brain")


class AnimatBrain:
    """
    Animat 双进程脑 v2 — 核心控制器

    ┌─────────────────────────────────────────┐
    │  恒常意识流 (StreamCortex)               │
    │  · LLM 永不停止                          │
    │  · 持续产生思维                          │
    │  · 门控决定哪些变成行为                   │
    ├─────────────────────────────────────────┤
    │  潜意识循环 (100ms tick)                 │
    │  · 稳态自然衰减                          │
    │  · 传感器 → 身体状态 同步                │
    │  · 驱力水平同步到 StreamCortex           │
    └─────────────────────────────────────────┘
    """

    def __init__(
        self,
        nervous: NervousSystem,
        memory: MemoryStore,
        llm_backend: LocalLLM = None,
    ):
        self.nervous = nervous
        self.memory = memory

        # 初始化各层
        self.llm = llm_backend or create_llm()
        self.sensor_window = SensorWindow()
        self.perception = PerceptionLayer(nervous, memory, self.sensor_window)
        self.homeostasis = HomeostasisEngine(nervous, memory)
        self.stream_cortex = StreamCortex(nervous, memory, self.llm)
        self.stream_cortex.sensor_window = self.sensor_window  # Phase 3: 传感器直通
        self.action = ActionExecutor(nervous, memory)

        self._running = False
        self._tick_count = 0

    async def boot(self):
        """启动 Animat 大脑 v2"""
        logger.info("═══ Animat Agent v2 Booting ═══")

        # 连接神经系统
        logger.info(f"Connecting to MQTT broker ({self.nervous.host}:{self.nervous.port})...")
        try:
            await self.nervous.start()
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            raise
        logger.info("MQTT broker connected ✓")

        # 连接记忆系统
        logger.info(f"Connecting to Redis ({self.memory.host}:{self.memory.port})...")
        try:
            await self.memory.connect()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
        logger.info("Redis connected ✓")

        # 加载 LLM
        if not self.llm.ready:
            self.llm.load()

        # 从 Redis 恢复状态
        await self._restore_state()

        # 注册各层神经末梢
        await self.perception.register()
        await self.homeostasis.register()
        await self.stream_cortex.register()
        await self.action.register()

        self._running = True
        logger.info("═══ Animat Agent v2 Online ═══")

        # 启动三进程
        await asyncio.gather(
            self.stream_cortex.start(),      # 恒常意识流（主进程）
            self._subconscious_loop(),       # 高频循环
            self._heartbeat_loop(),          # 心跳广播
        )

    async def shutdown(self):
        """关闭 Animat 大脑"""
        self._running = False
        await self.stream_cortex.stop()
        await self._save_state()
        await self.nervous.stop()
        await self.memory.close()
        logger.info("═══ Animat Agent v2 Offline ═══")

    # ── 潜意识循环 ──
    async def _subconscious_loop(self):
        """
        潜意识进程 — 高频运行 (100ms)
        职责简化：只做稳态衰减和传感器同步
        
        StreamCortex 自己管理思维产生节奏，
        潜意识不再负责"决定是否唤醒"。
        """
        logger.info("Subconscious loop v2 started")
        while self._running:
            tick_start = time.time()
            self._tick_count += 1

            # 1. 同步传感器直通稳态值
            await self._sync_sensor_driven_vitals()

            # 2. 稳态自然衰减
            dt = BrainConfig.SUBCONSCIOUS_TICK_MS / 1000.0
            await self.homeostasis.decay_vitals(dt)

            # 3. 同步驱力水平到 StreamCortex（用于思维频率调节）
            max_drive = self.homeostasis.get_max_drive()
            self.stream_cortex._drive_level = max_drive

            # 4. 周期性状态保存
            if self._tick_count % 100 == 0:
                await self._save_state()

            # 控制循环频率
            elapsed = (time.time() - tick_start) * 1000
            sleep_ms = max(0, BrainConfig.SUBCONSCIOUS_TICK_MS - elapsed)
            await asyncio.sleep(sleep_ms / 1000.0)

    # ── 心跳循环 ──
    async def _heartbeat_loop(self):
        """心跳 — 广播完整内部状态（稳态 / 情感 / 记忆 / 意识流统计）

        一次性把所有可观测的内部状态打包到 animat/state，
        调试端（网页/模拟器）订阅后就能实时监控 Animat 的"身心"。
        """
        logger.info("Heartbeat loop v2 started")
        while self._running:
            # ── 稳态五大指标 ──
            vitals = {
                v.value: round(self.homeostasis._vitals[v], 1)
                for v in Vital
            }

            # ── 情感（主导 + 次导 + 强度 + 原因）──
            dominant = self.homeostasis.get_dominant_emotion()
            all_emotions = getattr(self.homeostasis, "_active_emotions", []) or []
            emotion_payload = {
                "dominant": dominant.circuit.value if dominant else "SEEKING",
                "intensity": round(dominant.intensity, 2) if dominant else 0.0,
                "cause": dominant.cause if dominant else "",
                "all": [
                    {"circuit": e.circuit.value, "intensity": round(e.intensity, 2),
                     "cause": e.cause}
                    for e in all_emotions
                ],
            }

            # ── 意识流统计 ──
            stats = self.stream_cortex.stats()

            # ── 记忆摘要 ──
            try:
                self_info = await self.memory.recall_semantic("self_info") or {}
            except Exception:
                self_info = {}
            recent_conv = self.stream_cortex._conversation_history[-6:]

            # ── 打包广播 ──
            await self.nervous.fire("animat/state", {
                "tick": self._tick_count,
                "viability": vitals[Vital.VIABILITY.value],
                "vitals": vitals,
                "emotion": emotion_payload,
                "drive_level": self.stream_cortex._drive_level,
                "thoughts": stats["thoughts_total"],
                "activations": stats["activations"],
                "activation_rate_pct": stats["activation_rate_pct"],
                "thoughts_per_min": stats["thoughts_per_min"],
                "conversation_history": recent_conv,
                "self_info": self_info,
                "timestamp": time.time(),
            })

            await asyncio.sleep(1.0)

    # ── 传感器同步 ──
    async def _sync_sensor_driven_vitals(self):
        """从 Redis 同步传感器直通稳态值"""
        if self.memory.healthy and self.memory.redis:
            raw = await self.memory._safe_execute(
                "sync_vitals",
                self.memory.redis.hget(RedisKeys.STATE_HOMEOSTASIS, Vital.VIABILITY.value)
            )
            if raw is not None:
                sensor_value = float(raw)
                self.homeostasis._vitals[Vital.VIABILITY] = sensor_value

    async def _save_state(self):
        """保存当前状态到 Redis"""
        vitals = {v.value: val for v, val in self.homeostasis._vitals.items()}
        await self.memory.save_state(RedisKeys.STATE_HOMEOSTASIS, vitals)

    async def _restore_state(self):
        """从 Redis 恢复状态"""
        state = await self.memory.load_state(RedisKeys.STATE_HOMEOSTASIS)
        if state:
            for vital_name, value in state.items():
                try:
                    vital = Vital(vital_name)
                    self.homeostasis._vitals[vital] = float(value)
                except (ValueError, TypeError):
                    pass
            logger.info(f"State restored: {self.homeostasis._vitals}")
