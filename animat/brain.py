"""
双进程脑 — Animat Agent 的主控循环

职责:
  1. 潜意识进程: 高频循环(100ms)，驱动稳态衰减 + 驱力检测 + 无聊检测
  2. 意识唤醒: 统一由潜意识循环决策，所有唤醒请求必须经过分级冷却检查
  3. 心跳: 维持"活着"的感觉 — 即使无外部输入也在持续运行

关键设计:
  - homeostasis 只做情感合成和躯体标记，不直接唤醒 LLM
  - brain.py 是唯一可以发射 CORTEX_INPUT 的模组
  - 冷却从 LLM 输出完成后开始计算（而非唤醒时），防止 LLM 耗时导致冷却失效
  - 不同触发源有不同冷却时长（生存10秒、驱力30秒、无聊60秒、外部5秒）
  - 所有触发源统一走冷却检查，生存危机也不例外 — 紧急不等于高频
"""

import asyncio
import logging
import time

from animat.config import BrainConfig, Topic, RedisKeys
from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.perception import PerceptionLayer
from animat.homeostasis import HomeostasisEngine, Vital
from animat.cortex import Cortex
from animat.action import ActionExecutor

logger = logging.getLogger("animat.brain")

# 触发源 → 冷却时长(秒)的映射
# 所有触发源统一走冷却检查，不再有 force 绕过
TRIGGER_COOLDOWN = {
    "survival_critical":  BrainConfig.COOLDOWN_SURVIVAL,
    "drive":              BrainConfig.COOLDOWN_DRIVE,
    "boredom":            BrainConfig.COOLDOWN_BOREDOM,
    "external_speech":    BrainConfig.COOLDOWN_EXTERNAL,
}


class AnimatBrain:
    """
    Animat 双进程脑 — 核心控制器

    ┌─────────────────────────────────────────┐
    │  潜意识循环 (100ms tick)                │
    │  ├─ 稳态自然衰减                        │
    │  ├─ 驱力检测 → 唤醒意识 (统一冷却)      │
    │  ├─ 无聊检测 → 主动探索                 │
    │  └─ 反射弧响应                          │
    ├─────────────────────────────────────────┤
    │  意识进程 (触发式)                       │
    │  ├─ 外部语言输入 → 唤醒LLM             │
    │  └─ 生存危机 → 唤醒LLM                 │
    └─────────────────────────────────────────┘
    """

    def __init__(
        self,
        nervous: NervousSystem,
        memory: MemoryStore,
    ):
        self.nervous = nervous
        self.memory = memory

        # 初始化各层
        self.perception = PerceptionLayer(nervous, memory)
        self.homeostasis = HomeostasisEngine(nervous, memory)
        self.cortex = Cortex(nervous, memory)
        self.action = ActionExecutor(nervous, memory)

        self._running = False
        self._tick_count = 0

        # 冷却管理: 记录上次LLM输出完成的时间
        # 冷却从 LLM 完成时开始算，而非从唤醒时算
        self._last_cortex_output_time = 0.0
        self._last_trigger_type = ""

        # 驱力唤醒节流
        self._last_drive_wake_tick = 0

    async def boot(self):
        """启动Animat大脑"""
        logger.info("═══ Animat Agent Booting ═══")

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

        # 从Redis恢复状态
        await self._restore_state()

        # 注册各层神经末梢
        await self.perception.register()
        await self.homeostasis.register()
        await self.cortex.register()
        await self.action.register()

        # 注册外部语言输入通道
        await self.nervous.listen("animat/external/speech", self._on_external_speech)

        # 监听LLM输出 — 用于在LLM完成时更新冷却计时
        await self.nervous.listen(Topic.CORTEX_OUTPUT, self._on_cortex_output)

        self._running = True
        logger.info("═══ Animat Agent Online ═══")

        # 启动双进程
        await asyncio.gather(
            self._subconscious_loop(),
            self._heartbeat_loop(),
        )

    async def shutdown(self):
        """关闭Animat大脑"""
        self._running = False
        await self._save_state()
        await self.nervous.stop()
        await self.memory.close()
        logger.info("═══ Animat Agent Offline ═══")

    # ── LLM 输出回调 — 冷却从此刻开始计算 ──
    async def _on_cortex_output(self, topic: str, payload: dict):
        """
        LLM完成推理并输出结果时，更新冷却计时起点
        这是防止高频调用的关键：冷却从"行为完成"算，而非"行为开始"算
        """
        self._last_cortex_output_time = time.time()
        logger.debug(f"Cortex output received, cooldown reset for {self._last_trigger_type}")

    # ── 统一的意识唤醒（所有路径都必须经过此方法）──
    async def _try_wake_cortex(self, trigger: str, text: str = ""):
        """
        统一的意识唤醒入口

        冷却逻辑:
          - Cortex 忙碌时直接拒绝（避免无效 MQTT 消息）
          - 冷却从上次LLM输出完成时开始计算
          - 不同触发源有不同的冷却时长
          - 所有触发源（包括生存危机）统一走冷却，不再有 force 绕过
        """
        # 1. Cortex 正在处理 LLM 请求 → 直接拒绝
        if self.cortex.busy:
            logger.debug(f"Cortex busy, '{trigger}' deferred")
            return False

        # 2. 冷却检查
        now = time.time()
        cooldown_s = TRIGGER_COOLDOWN.get(trigger, BrainConfig.COOLDOWN_DRIVE)
        elapsed_since_output = now - self._last_cortex_output_time

        if elapsed_since_output < cooldown_s:
            logger.debug(
                f"Cortex in cooldown ({elapsed_since_output:.0f}s < {cooldown_s}s), "
                f"'{trigger}' deferred"
            )
            return False

        self._last_trigger_type = trigger

        dominant = self.homeostasis.get_dominant_emotion()
        max_drive = self.homeostasis.get_max_drive()

        logger.info(f"Waking cortex: trigger={trigger}, emotion={dominant.circuit.value if dominant else 'SEEKING'}")
        await self.nervous.fire(Topic.CORTEX_INPUT, {
            "trigger": trigger,
            "text": text,
            "dominant_emotion": dominant.circuit.value if dominant else "SEEKING",
            "max_drive": max_drive,
            "timestamp": now,
        })
        return True

    # ── 潜意识循环 ──
    async def _subconscious_loop(self):
        """
        潜意识进程 — 高频运行
        即使无外部输入也在持续运行，这是"活着"的基础
        """
        logger.info("Subconscious loop started")
        while self._running:
            tick_start = time.time()
            self._tick_count += 1

            # 1. 稳态自然衰减
            dt = BrainConfig.SUBCONSCIOUS_TICK_MS / 1000.0
            await self.homeostasis.decay_vitals(dt)

            # 2. 驱力检测 — 按配置间隔检查
            if self._tick_count - self._last_drive_wake_tick >= BrainConfig.DRIVE_CHECK_INTERVAL:
                await self._check_drive_wake()

            # 3. 无聊检测 — 每300个tick(≈30秒)检查一次
            if self._tick_count % 300 == 0:
                await self._check_boredom()

            # 4. 周期性状态保存
            if self._tick_count % 100 == 0:
                await self._save_state()

            # 控制循环频率
            elapsed = (time.time() - tick_start) * 1000
            sleep_ms = max(0, BrainConfig.SUBCONSCIOUS_TICK_MS - elapsed)
            await asyncio.sleep(sleep_ms / 1000.0)

    # ── 驱力唤醒检测 ──
    async def _check_drive_wake(self):
        """
        检测驱力是否超过唤醒阈值
        这是Animat主动行为的核心来源之一
        """
        max_drive = self.homeostasis.get_max_drive()
        if max_drive > BrainConfig.WAKE_DRIVE_THRESHOLD:
            if await self._try_wake_cortex(trigger="drive"):
                self._last_drive_wake_tick = self._tick_count

    # ── 心跳循环 ──
    async def _heartbeat_loop(self):
        """
        心跳 — 向外广播"我还活着"
        同时监测生存度，极度危险时主动求救
        """
        logger.info("Heartbeat loop started")
        while self._running:
            viability = self.homeostasis._vitals[Vital.VIABILITY]

            # 生存度极低时唤醒（统一走冷却检查）
            if viability < 15:
                await self._try_wake_cortex(
                    trigger="survival_critical",
                    text=f"电量仅剩{viability:.0f}%，需要立刻充电！",
                )

            # 广播心跳
            await self.nervous.fire("animat/heartbeat", {
                "tick": self._tick_count,
                "viability": viability,
                "emotion": self.homeostasis._active_emotions[0].circuit.value if self.homeostasis._active_emotions else "SEEKING",
                "timestamp": time.time(),
            })

            await asyncio.sleep(1.0)

    # ── 无聊检测 ──
    async def _check_boredom(self):
        """
        检测无聊状态 — 长期无新刺激时主动探索
        这是Animat从被动转为主动的关键机制
        """
        exploration = self.homeostasis._vitals[Vital.EXPLORATION]
        affiliation = self.homeostasis._vitals[Vital.AFFILIATION]

        # 探索度低 + 归属度低 → 无聊，主动发起行为
        if exploration < 20 and affiliation < 30:
            await self._try_wake_cortex(trigger="boredom")

    # ── 外部语言输入 ──
    async def _on_external_speech(self, topic: str, payload: dict):
        """
        接收外部语言(人类对Animat说的话)
        这不是"被动响应"，而是感知层的一种输入
        """
        text = payload.get("text", "")
        if not text:
            return

        logger.info(f"External speech received: {text[:50]}")
        await self._try_wake_cortex(trigger="external_speech", text=text)

    # ── 状态持久化 ──
    async def _save_state(self):
        """保存当前状态到Redis"""
        vitals = {v.value: val for v, val in self.homeostasis._vitals.items()}
        await self.memory.save_state(RedisKeys.STATE_HOMEOSTASIS, vitals)

    async def _restore_state(self):
        """从Redis恢复状态"""
        state = await self.memory.load_state(RedisKeys.STATE_HOMEOSTASIS)
        if state:
            for vital_name, value in state.items():
                try:
                    vital = Vital(vital_name)
                    self.homeostasis._vitals[vital] = float(value)
                except (ValueError, TypeError):
                    pass
            logger.info(f"State restored: {self.homeostasis._vitals}")
