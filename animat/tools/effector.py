"""执行器工具 — 全部通过 MQTT 话题发布，由执行器订阅执行。

这样同一套工具既能驱动模拟器，也能驱动真实硬件：
执行器换成真实设备后只需订阅同名话题，LLM 侧零改动。

设计上只有一条规则：**凡是能改变世界或对外可见的动作，都是一个工具。**
说话、做表情、走过去、扭头看，全都走 tool call。
没有第二套"意图标签 → 动作树"的翻译路径——两套语法并存时，模型永远会挑
语法更简单的那一套（实测它挑了 <intent>，于是 move_to 七小时零调用），
而真正能改变世界的那一套就被绕开了。
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

# ── 说话方式 ──
# 这不是"意图标签"，而是 speak 工具的一个参数：同一句话可以喊出去，也可以
# 只对自己念叨。区别关乎"外面的人该不该听到"，所以交给模型显式声明，
# 而不是留给门控按情感强度去猜。
SPEECH_MODES = ["mutter", "aloud", "call"]
_SPEECH_MODE_DESC = (
    "mutter=自言自语，只有自己知道；"
    "aloud=普通说话，旁边有人时会听到；"
    "call=大声呼唤，希望有人能回应"
)


def register_effector_tools() -> None:
    """注册全部执行器工具。"""

    # ── 说话 ──
    # 说话是一个动作，不是"思维附带的副产物"。此前它是唯一没有工具形态的
    # 输出，模型只能靠 <intent> 标签间接表达，门控再用相似度/速率去拦——
    # 结果 96.6% 的思维挤在 expression 标签上，抑制次数超过输出次数。
    # 改成工具后，"说不说"由模型决定，"说什么"就在参数里，不再需要猜测。
    registry.register(ToolSpec(
        name="speak",
        # display 不能置真：display 的用途是"这一轮做了动作但没说出口，
        # 拿正文补一句旁白"。speak 已经把原话放在参数里说完了，
        # 再走一次旁白通道就是同一句话发两遍（实测对话窗出现重复气泡）。
        display=False,
        description=(
            "开口说话。想说点什么、想喊人、想自言自语时都用这个。"
            "文字必须是你要说的原话，不要写动作描写。"
        ),
        # 说话不走 motor，走 CORTEX_OUTPUT。
        #
        # 这不是绕远路：action.py 订阅 CORTEX_OUTPUT 后会按当前情感
        # 注入语速与音量再发 motor/speak。若工具自己也直接发 motor，
        # 同一句话会被投递两次——实测对话窗里出现过重复气泡。
        # 现在 motor/speak 只有一个发起点（action.py），
        # 工具这里只声明"我要说这句、说给谁听"。
        topic=Topic.CORTEX_OUTPUT,
        command=None,
        cooldown_sec=0.0,
        params={
            "text": ParamSpec(
                type=ParamType.STRING,
                description="你要说的原话，1-3 句，口语，不要加引号或旁白",
            ),
            "mode": ParamSpec(
                type=ParamType.ENUM, enum=SPEECH_MODES, default="mutter",
                description=_SPEECH_MODE_DESC,
                required=False,
            ),
        },
    ))

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

    # ── 移动 ──
    # 移动能力只有这一条通路：带目的地、经空间模型更新真实坐标。
    # 早期还有一个"按方向走一米"的 move，两者并存时模型总选省事的前者，
    # 走完位置对人类没有意义，3D 里就是原地抖动。已删除。
    #
    # 可选地点直接取自空间模型，不在这里另抄一份：
    # 两份清单迟早会不同步，而且枚举能挡住模型编造"花园""地下室"
    # 这种走到一半才发现不存在的地方。
    from animat.spatial import World
    places = [o.name for o in World().objects]

    registry.register(ToolSpec(
        name="move_to",
        display=True,
        description=(
            "朝房间里的某件东西走过去。走到之后你会看到它附近的样子。"
            "感到无聊、好奇、想换个地方时就用它，而不是只在心里想。"
        ),
        topic=Topic.MOTOR,
        command="move_to",
        cooldown_sec=2.0,
        params={
            "place": ParamSpec(
                type=ParamType.ENUM, enum=places,
                description="要去的地方",
            ),
            "speed": ParamSpec(
                type=ParamType.NUMBER, minimum=0.0, maximum=1.0, default=0.4,
                description="速度 0.0-1.0",
                required=False,
            ),
        },
    ))

    registry.register(ToolSpec(
        name="look_around",
        description=(
            "转个身重新打量房间，会得到一份新的空间信息（方位与距离）。"
            "不确定自己在哪、或想确认某件东西还在不在时用。"
        ),
        topic=Topic.MOTOR,
        command="look_around",
        cooldown_sec=3.0,
        params={
            "direction": ParamSpec(
                type=ParamType.ENUM,
                enum=["left", "right", "around"],
                default="around",
                description="往哪边看：left 左转，right 右转，around 环视一圈",
                required=False,
            ),
        },
    ))

    registry.register(ToolSpec(
        name="express",
        display=True,
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
        display=True,
        description="做一个肢体手势，比如点头、摇头、歪头、挥手。",
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

    # 从注册表取实际名字再打印。硬写字符串的话，以后增删工具时
    # 这行日志会一直说谎（上一版就写着已经不存在的 move）。
    logger.info("执行器工具注册完成: %s", ", ".join(
        n for n in registry.names()
        if n not in ("web_search",)
    ))
