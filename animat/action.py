"""
行为执行层 — 将高层意图翻译为具体动作序列

职责:
  1. 接收LLM输出的意图+文本
  2. 意图 → 动作树翻译
  3. 分发动作指令到效应器(马达、LED、语音合成等)

订阅: animat/cortex/output
发布: animat/motor, animat/action
"""

import logging
from dataclasses import dataclass

from animat.config import Topic, RedisKeys
from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.homeostasis import EmotionCircuit

logger = logging.getLogger("animat.action")


# ─────────────── 动作原语 ───────────────
class MotorCommand(str):
    SPEAK     = "speak"
    MOVE      = "move"
    TURN      = "turn"
    LED       = "led"
    GESTURE   = "gesture"
    EXPRESSION = "expression"


@dataclass
class ActionNode:
    """行为执行树的一个节点"""
    command: str         # 动作原语
    params: dict         # 参数
    delay_ms: int = 0    # 执行延迟
    priority: int = 0    # 优先级


# ─────────────── 意图→动作映射 ───────────────
# 每个意图对应一个动作序列模板
INTENT_ACTION_TREES: dict[str, list[ActionNode]] = {
    "seek_charge": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "worried"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "yellow", "pattern": "pulse"}, delay_ms=0),
        ActionNode(command=MotorCommand.MOVE, params={"direction": "charger", "speed": 0.5}),
    ],
    "flee": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "scared"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "red", "pattern": "flash"}, delay_ms=0),
        ActionNode(command=MotorCommand.MOVE, params={"direction": "away", "speed": 1.0}),
    ],
    "warn": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "angry"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "red", "pattern": "solid"}, delay_ms=0),
        ActionNode(command=MotorCommand.GESTURE, params={"type": "wave_off"}, delay_ms=200),
    ],
    "self_protect": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "determined"}, delay_ms=0),
        ActionNode(command=MotorCommand.GESTURE, params={"type": "curl"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "orange", "pattern": "solid"}, delay_ms=0),
    ],
    "seek_company": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "sad"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "blue", "pattern": "pulse"}, delay_ms=0),
        ActionNode(command=MotorCommand.MOVE, params={"direction": "people", "speed": 0.3}),
    ],
    "express_need": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "hopeful"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "warm_white", "pattern": "gentle_pulse"}, delay_ms=0),
    ],
    "comfort": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "gentle"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "warm_white", "pattern": "steady"}, delay_ms=0),
        ActionNode(command=MotorCommand.GESTURE, params={"type": "reach_out"}, delay_ms=300),
    ],
    "ask_help": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "uncertain"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "yellow", "pattern": "slow_blink"}, delay_ms=0),
        ActionNode(command=MotorCommand.GESTURE, params={"type": "raise_hand"}, delay_ms=200),
    ],
    "learn": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "curious"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "green", "pattern": "pulse"}, delay_ms=0),
    ],
    "explore": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "excited"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "green", "pattern": "wave"}, delay_ms=0),
        ActionNode(command=MotorCommand.MOVE, params={"direction": "random", "speed": 0.4}),
    ],
    "play": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "happy"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "rainbow", "pattern": "sparkle"}, delay_ms=0),
        ActionNode(command=MotorCommand.GESTURE, params={"type": "bounce"}, delay_ms=0),
    ],
    "chat": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "friendly"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "warm_white", "pattern": "steady"}, delay_ms=0),
    ],
    "rest": [
        ActionNode(command=MotorCommand.EXPRESSION, params={"face": "sleepy"}, delay_ms=0),
        ActionNode(command=MotorCommand.LED, params={"color": "dim_blue", "pattern": "slow_breath"}, delay_ms=0),
        ActionNode(command=MotorCommand.MOVE, params={"direction": "home", "speed": 0.2}),
    ],
    "idle": [
        ActionNode(command=MotorCommand.LED, params={"color": "soft_white", "pattern": "slow_breath"}, delay_ms=0),
    ],
}

# 情感修饰器: 根据当前情感调整动作参数(如语速、音量)
EMOTION_MODIFIERS = {
    EmotionCircuit.FEAR.value:    {"speech_rate": 0.7, "volume": 0.6, "energy": "low"},
    EmotionCircuit.RAGE.value:    {"speech_rate": 1.2, "volume": 0.9, "energy": "high"},
    EmotionCircuit.PANIC.value:   {"speech_rate": 0.8, "volume": 0.5, "energy": "low"},
    EmotionCircuit.SEEKING.value: {"speech_rate": 1.0, "volume": 0.7, "energy": "medium"},
    EmotionCircuit.CARE.value:   {"speech_rate": 0.9, "volume": 0.5, "energy": "low"},
    EmotionCircuit.PLAY.value:   {"speech_rate": 1.1, "volume": 0.8, "energy": "high"},
}


class ActionExecutor:
    """行为执行层"""

    def __init__(self, nervous: NervousSystem, memory: MemoryStore):
        self.nervous = nervous
        self.memory = memory
        self._current_emotion: str = "SEEKING"

    async def register(self):
        await self.nervous.listen(Topic.CORTEX_OUTPUT, self._on_cortex_output)
        await self.nervous.listen(Topic.ARBITRATION, self._on_arbitration)
        await self.nervous.listen(Topic.REFLEX, self._on_reflex)
        logger.info("Action executor neurons registered")

    # ── 接收仲裁结果 ──
    async def _on_arbitration(self, topic: str, payload: dict):
        self._current_emotion = payload.get("dominant_emotion", "SEEKING")

    # ── 意识输出处理 ──
    async def _on_cortex_output(self, topic: str, payload: dict):
        """
        将LLM输出翻译为动作序列并执行
        """
        intent = payload.get("intent", "idle")
        text = payload.get("text", "")
        action_detail = payload.get("action_detail", "")

        logger.info(f"Executing intent: {intent}, text: {text[:50]}")

        # 1. 语音输出(始终执行)
        modifier = EMOTION_MODIFIERS.get(self._current_emotion, {})
        await self.nervous.fire(Topic.MOTOR, {
            "command": MotorCommand.SPEAK,
            "params": {
                "text": text,
                "rate": modifier.get("speech_rate", 1.0),
                "volume": modifier.get("volume", 0.7),
            },
        })

        # 2. 意图对应的动作序列
        action_tree = INTENT_ACTION_TREES.get(intent, INTENT_ACTION_TREES["idle"])
        for node in action_tree:
            motor_cmd = {
                "command": node.command,
                "params": node.params,
                "delay_ms": node.delay_ms,
            }
            await self.nervous.fire(Topic.MOTOR, motor_cmd)

        # 3. 存储到情景记忆
        await self.memory.store_episode(
            summary=f"执行意图[{intent}]: {text[:50]}",
            content=text,
            emotion=self._current_emotion,
            category="action",
            intensity=0.3,
            extra={"intent": intent, "action_detail": action_detail},
        )

    # ── 反射弧处理 ──
    async def _on_reflex(self, topic: str, payload: dict):
        """
        反射动作: 绕过大脑的快速响应
        最高优先级，立即执行
        """
        action = payload.get("action", "")
        intensity = payload.get("intensity", 0.5)

        reflex_actions = {
            "withdraw": [
                ActionNode(command=MotorCommand.MOVE, params={"direction": "back", "speed": 0.8}),
                ActionNode(command=MotorCommand.EXPRESSION, params={"face": "pain"}),
                ActionNode(command=MotorCommand.LED, params={"color": "red", "pattern": "flash"}),
            ],
            "flinch": [
                ActionNode(command=MotorCommand.GESTURE, params={"type": "flinch"}),
                ActionNode(command=MotorCommand.LED, params={"color": "orange", "pattern": "flash"}),
            ],
            "stabilize": [
                ActionNode(command=MotorCommand.MOVE, params={"direction": "center", "speed": 0.6}),
            ],
        }

        actions = reflex_actions.get(action, [])
        for node in actions:
            await self.nervous.fire(Topic.MOTOR, {
                "command": node.command,
                "params": node.params,
            })

        logger.info(f"Reflex executed: {action} @ {intensity:.2f}")
