"""执行器工具 — 全部通过 MQTT 话题发布，由执行器订阅执行。

这样同一套工具既能驱动模拟器，也能驱动真实硬件：
执行器换成真实设备后只需订阅同名话题，LLM 侧零改动。
"""

from __future__ import annotations

import logging

from animat.config import Topic
from animat.tools.registry import registry
from animat.tools.spec import ParamSpec, ParamType, ToolSpec

logger = logging.getLogger("animat.tools.effector")


# ── LED 显示 ──
LED_COLORS = [
    "red", "orange", "yellow", "green", "cyan", "blue", "purple",
    "white", "warm_white", "dim_blue", "rainbow",
]
LED_PATTERNS = [
    "solid", "pulse", "flash", "slow_blink", "slow_breath",
    "wave", "sparkle", "gentle_pulse",
]

# ── 表情 / 手势 ──
EXPRESSIONS = [
    "neutral", "happy", "sad", "angry", "surprised", "curious",
    "sleepy", "excited", "calm", "confused",
]
GESTURES = [
    "nod", "shake_head", "tilt_head", "look_around",
    "stretch", "wave", "shrug", "lean_forward",
]
DIRECTIONS = ["forward", "backward", "left", "right", "turn_left", "turn_right"]


def register_effector_tools() -> None:
    """注册全部执行器工具。"""

    registry.register(ToolSpec(
        name="set_led",
        description=(
            "改变你的 LED 显示器颜色和闪烁方式。用来表达情绪、"
            "在黑暗中视物、或者吸引人的注意。"
        ),
        topic=Topic.EFFECTOR_LED,
        command="led",
        cooldown_sec=2.0,
        params={
            "color": ParamSpec(
                type=ParamType.ENUM, enum=LED_COLORS,
                description="灯光颜色",
            ),
            "pattern": ParamSpec(
                type=ParamType.ENUM, enum=LED_PATTERNS, default="solid",
                description="闪烁模式。solid 常亮，pulse 呼吸，flash 快闪",
                required=False,
            ),
            "duration_ms": ParamSpec(
                type=ParamType.INTEGER, minimum=200, maximum=60000, default=3000,
                description="持续时间（毫秒），到时间后自动恢复默认",
                required=False,
            ),
        },
    ))

    registry.register(ToolSpec(
        name="move",
        description=(
            "驱动你的底盘移动。注意移动会消耗电量，"
            "电量低时应该谨慎使用。"
        ),
        topic=Topic.MOTOR,
        command="move",
        cooldown_sec=3.0,
        params={
            "direction": ParamSpec(
                type=ParamType.ENUM, enum=DIRECTIONS,
                description="移动方向",
            ),
            "speed": ParamSpec(
                type=ParamType.NUMBER, minimum=0.0, maximum=1.0, default=0.4,
                description="速度 0.0-1.0",
                required=False,
            ),
            "distance_cm": ParamSpec(
                type=ParamType.INTEGER, minimum=1, maximum=500, default=30,
                description="移动距离（厘米）",
                required=False,
            ),
        },
    ))

    registry.register(ToolSpec(
        name="express",
        description="做出表情，让屏幕/舵机呈现你的情绪状态。",
        topic=Topic.EFFECTOR_EXPRESSION,
        command="expression",
        cooldown_sec=2.0,
        params={
            "emotion": ParamSpec(
                type=ParamType.ENUM, enum=EXPRESSIONS,
                description="要表达的情绪",
            ),
            "intensity": ParamSpec(
                type=ParamType.NUMBER, minimum=0.0, maximum=1.0, default=0.7,
                description="强度 0.0-1.0",
                required=False,
            ),
        },
    ))

    registry.register(ToolSpec(
        name="gesture",
        description="做一个肢体手势，比如点头、摇头、歪头。",
        topic=Topic.EFFECTOR_GESTURE,
        command="gesture",
        cooldown_sec=2.0,
        params={
            "name": ParamSpec(
                type=ParamType.ENUM, enum=GESTURES,
                description="手势名称",
            ),
            "repeat": ParamSpec(
                type=ParamType.INTEGER, minimum=1, maximum=5, default=1,
                description="重复次数",
                required=False,
            ),
        },
    ))

    logger.info("执行器工具注册完成: set_led, move, express, gesture")
