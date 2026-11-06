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

from animat.config import Topic
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


# INTENT_ACTION_TREES 已经整张删除。
#
# 它把十几种"意图"映射到写死的动作序列（intent=explore → 随机方向走一米）。
# 那张表真正在做的事，是把模型的表达能力压缩到十来棵预设的树里：
# 它永远无法表达"我想去书架"，只能选一棵最接近的树，然后由系统
# 替它播放一套固定动作。配上 <intent> 标签后问题更明显——
# 模型只学会了填标签，动作本身和它的想法是脱钩的。
#
# 现在动作全部由工具调用直接表达，参数就是模型的原话：
#   move_to(place="书架") / express(emotion="sad") / speak(text="...")
# 不需要中间层翻译，也就不需要这张表。

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
        """处理一次"说出去了"的通知。

        这里曾经是 LLM 输出的主要翻译层：拿 <intent> 标签去
        INTENT_ACTION_TREES 里查一棵写死的动作树，然后逐条发 motor 指令。
        那套设计有两个问题，现在都已不存在：

          1. 动作树是写死的（intent=explore → 随机方向走一米），
             模型无法表达"走向书架"，只能挑一棵最接近的树。
          2. 系统替模型发声：门控放行 = 说出去，模型自己从没表达过
             "这句话我说不说"。于是 96.6% 的思维挤在同一个标签上，
             剩下全世界都在做抑制。

        现在动作由模型的工具调用直接下发到 effector/* 与 motor 话题，
        这里只保留两件系统该做的事：语音合成（注入情感语速音量）
        和情景记忆。它不再决定说什么，也不再决定做什么。
        """
        text = payload.get("text", "")
        mode = payload.get("mode") or "aloud"
        utterance_type = payload.get("utterance_type") or "proactive"

        logger.info(f"Cortex output ({mode}): {text[:50]}")

        # 1. 语音合成。mutter 也要发——自言自语是人听得见的小声嘀咕，
        #    只是不进对话窗口（分流在 webui 里做）。
        #
        # 这里是**唯一**把"文字"变成 motor/speak 的地方：无论文字来自
        # 模型的 speak 工具，还是对话轮的正文，都汇到这一层再发声。
        # 两个发起点会让同一句话出现两次（实测确有重复气泡），
        # 而发声需要情感修饰，只有这一层看得到当前情感。
        if text:
            modifier = EMOTION_MODIFIERS.get(self._current_emotion, {})
            await self.nervous.fire(Topic.MOTOR, {
                "command": MotorCommand.SPEAK,
                "params": {
                    "text": text,
                    "mode": mode,
                    "rate": modifier.get("speech_rate", 1.0),
                    "volume": modifier.get("volume", 0.7),
                    "utterance_type": utterance_type,
                },
            })

        # 2. 存储到情景记忆
        await self.memory.store_episode(
            summary=f"[{mode}] {text[:50]}",
            content=text,
            emotion=self._current_emotion,
            category="action",
            intensity=0.3,
            extra={"mode": mode},
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
