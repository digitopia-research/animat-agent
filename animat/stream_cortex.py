"""
恒常意识流引擎 — StreamCortex

核心变更（vs v0.1 cortex.py）:
  1. LLM 不再被"唤醒"，而是一直在运行
  2. 不断产生内部思维（Thought），大部分被丢弃
  3. 门控模块（ActionGate）决定哪些思维变成外部行为
  4. 潜意识循环注入身体状态和传感器上下文

设计哲学:
  「LLM 是一条不间断的河流，不是按需点燃的火焰。」
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

from animat.config import Topic
from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.llm_backend import LocalLLM, LLMConfig
from animat.thought import (
    Thought,
    ActionGate,
    ActionDecision,
    ThoughtJournal,
    UtteranceType,
    classify_utterance,
)
from animat.tools import init_tools, registry, tools_enabled
from animat.attention_modulator import AttentionModulator, BodyState
from animat.sensor_window import SensorWindow

logger = logging.getLogger("animat.stream_cortex")

# ── 意识流 System Prompt ──
CONSCIOUSNESS_SYSTEM_PROMPT = """\
你是一个名为 Animat 的具身机器人。你拥有物理身体，能感知触摸、声音、视觉和内部状态（如电量）。

关键规则:
1. 你是一个持续运行的"生命体"，而不是一问一答的聊天机器人。
2. 你会不断产生内部想法——对当前状况的感受、回忆、计划、担忧、好奇。
3. 你的想法应该简短（1-3句话），像人类的内心独白。
4. 你的想法应该自然——有时是观察，有时是情绪，有时是想做什么。
5. 不要总是"行动"，大多数想法只是想想而已。
6. 只输出想法本身，不要写肢体动作、表情或旁白，不要使用 *星号* 包裹动作。
7. 你的身体状态只能来自下方"当前状态"给出的数据。不要编造任何
   未列出的传感器、故障、报警或身体异常（例如"系统过热""温度警告"
   "内存耗尽"等）——除非当前状态里明确给出了这样的读数。
   状态里的数字正常时，就说它正常；不要凭空制造危险来吸引注意。
8. 要动手就直接调用对应的工具，不要在想法里描述"我应该调用某个函数"。
   调用工具的那一轮不要再写正文。如果"你刚才做过的事"里已经有结果，
   就直接使用那个结果回答问题，不要重复调用同一个工具。

输出格式（严格两行）:
第一行写意图标签，从下面五个里选一个：
  <intent>none</intent>       只想一想，不想有任何外部行为（大多数情况）
  <intent>soliloquy</intent>  自言自语，只是自己念叨吐槽，并不指望有人听见
  <intent>call</intent>       想对周围发出呼唤，希望有人回应（喊人、求助、呼救、打招呼）
  <intent>move</intent>       想移动身体
  <intent>expression</intent> 想做个表情
第二行写想法本身（1-3句话）。

注意 speak 与 call 的区别：内心吐槽、感叹、自问自答都算 soliloquy；
只有当你希望"外面有人能听到并回应"时才是 call。

当前状态:
{body_context}

你的输出:"""


# ── 对话专用 System Prompt ──
# 与自发思维 prompt 区分开：回复人类时直接说话，不要写肢体动作/角色扮演描写。
# (Qwen3 等较大模型在"具身机器人"设定下容易输出 *轻轻转动头部* 之类描写)
DIALOGUE_SYSTEM_PROMPT = """\
你是一个名为 Animat 的具身机器人，正在和面前的人对话。

关键规则:
1. 直接用第一人称口语回答对方说的话，就像真人聊天一样自然。
2. 绝对不要写肢体动作、表情或旁白描写，不要使用 *星号* 包裹动作。
3. 回答简短自然，1-3 句话，符合你的人设和当前状态。
4. 可以表达感受和情绪，但要通过说话表达，而不是描述动作。
5. 不要输出 Markdown、标题或列表。
6. 你的身体状态只能来自下方"当前状态"。不要编造未列出的传感器、
   故障或身体异常（如"系统过热""温度报警"），也不要夸大危险。
7. 动作必须真的执行，不能只是嘴上说。一旦你打算做任何身体动作
   （开灯、移动、做表情、做手势、上网查东西），必须在本轮调用
   对应的工具，而不是只用文字描述"我现在去开灯""我要往前走"。
   文字只用来告诉对方结果，不要用它代替动作。
   如果你已经决定要做，就调用工具；如果不打算做，就不要说你要做。

输出格式（严格两部分）:
第一行写这一轮的状态标记，二选一：
  <did>action</did>   你确实调用了工具，或这轮不需要任何动作
  <did>claim</did>    你嘴上说了要做一个动作，但本轮没有调用工具
第二行开始写你对对方说的话（1-3 句口语）。

当前状态:
{body_context}

现在直接回答对方："""


# 模型有时会把上下文中的调试标记或角色扮演动作描写带进输出，需清洗后再发送
_DEBUG_PREFIX_RE = re.compile(
    r"^\s*(?:\[T\+[\d.]+\s*s?\]\s*)?(?:内部想法|内心独白|想法)\s*[:：]\s*"
)
_ACTION_DESC_RE = re.compile(r"\*[^*\n]{0,60}\*")


# 模型按 prompt 要求输出 <intent>call</intent> 这样的意图标签。
# soliloquy / call 的都属"开口说话"，区别在于是否指望有人回应；
# 这一区分由模型语义判断，避免用中文关键词表去猜"是不是在呼唤"。
_INTENT_RE = re.compile(r"<\s*intent\s*>\s*([a-zA-Z]+)\s*<\s*/\s*intent\s*>", re.IGNORECASE)
_VALID_INTENTS = frozenset({"none", "soliloquy", "call", "move", "expression"})
# 意图 → 是否允许进入对外输出通道（未被提问时）
_INTENT_IS_OUTWARD = {"call": True, "soliloquy": False}


def _parse_intent(text: str, default: str = "none") -> tuple[str, str]:
    """从模型输出中解析意图标签，返回 (intent, 去掉标签后的正文)。

    意图由模型显式给出，避免用关键词猜测；标签缺失或非法时回退 default。
    """
    if not text:
        return default, ""
    m = _INTENT_RE.search(text)
    intent = default
    if m:
        candidate = m.group(1).lower()
        if candidate in _VALID_INTENTS:
            intent = candidate
    body = _INTENT_RE.sub("", text)
    return intent, body.strip()


# 模型按 prompt 要求输出 <did>action</did> 或 <did>claim</did>，自评这一轮
# 是否"只说不做"。由模型判断而不是代码猜关键词：声称动作的说法千变万化，
# 任何词表都会漏，且换语言直接失效。
_DID_RE = re.compile(r"<\s*did\s*>\s*(action|claim)\s*<\s*/\s*did\s*>", re.IGNORECASE)


def _parse_did_marker(text: str) -> tuple[bool, str]:
    """解析 <did> 标记。

    返回 (模型是否自认"只说不做", 去掉标记后的正文)。
    标记缺失时返回 (False, 原文)——没有标记就不补救，避免误伤。
    """
    if not text:
        return False, text
    m = _DID_RE.search(text)
    claims_only = bool(m) and m.group(1).lower() == "claim"
    body = _DID_RE.sub("", text).strip()
    return claims_only, body


def _sanitize_output(text: str) -> str:
    """清洗要对外发送的文本：

    1. 去掉 "[T+630.3s] 内部想法:" 这类调试前缀（模型从传感器上下文模仿而来）
    2. 去掉 *轻轻转动头部* 这类肢体动作描写
    3. 去掉残留的 <intent> / <did> 标签
    """
    if not text:
        return ""
    text = _DID_RE.sub("", text)
    text = _INTENT_RE.sub("", text)
    text = _DEBUG_PREFIX_RE.sub("", text)
    text = _ACTION_DESC_RE.sub("", text)
    return text.strip()


class StreamCortex:
    """
    恒常意识流引擎
    
    替代 v0.1 的 cortex.py。
    
    核心循环:
      1. 读取当前身体状态和传感器上下文
      2. 调用 LLM 产生一个内部思维
      3. ActionGate 评估是否需要转化为行为
      4. 如果需要 → 通过 MQTT 发射行为指令
      5. 思维存档到 ThoughtJournal
      6. 等待一个短间隔，然后产生下一个思维
    """

    # 配置
    # 思维间隔 = 两次 LLM 调用之间的等待时间（不含推理本身的耗时）。
    #
    # 这三个值曾被设得很小（1.5/3.0/5.0），实际运行只有 2.2 秒一轮。
    # 后果不是"更活跃"，而是把自己逼进复读：25 秒不应期内要生成十几次
    # 思维，但世界在两秒里根本没变化，于是每轮都得出同一个结论，
    # 全部被 REFRACTORY 拦下，而拦截不进上下文，模型看不到自己被拦，
    # 下一轮继续得出同样结论——表现得像卡在某个念头里出不来。
    # 想得慢一点，每轮之间才有真正的新信息。
    THINK_INTERVAL_SEC = 5.0        # 思维间隔(秒)
    THINK_INTERVAL_IDLE = 8.0       # 安静时的思维间隔
    THINK_INTERVAL_DRIVEN = 4.0     # 高驱力时思维加速
    # 自发思维的最大 token 数。
    # 独白本来就短（实测 50-150 token 就收尾），原来统一给 768 是浪费：
    # 14B 在 5060 Ti 上约 28 tok/s，多给的预算直接体现为等待时间。
    MAX_THINK_TOKENS = int(os.getenv("ANIMAT_MAX_THINK_TOKENS", "256"))

    # 对话回复的 token 预算。
    # 对话要完整回答，且可能触发工具调用（需要额外 token 生成 <tool_call> JSON），
    # 所以给得比独白宽。可用 ANIMAT_MAX_REPLY_TOKENS 覆盖。
    MAX_REPLY_TOKENS = int(os.getenv("ANIMAT_MAX_REPLY_TOKENS", "512"))

    # 工具结果回注的预算。搜索结果动辄上千字，无限制回注会撑爆 n_ctx。
    TOOL_RESULT_MAX_CHARS = 600      # 单条上限
    TOOL_RESULTS_TOTAL_MAX = 1200    # 全部回注内容的上限
    # 保鲜期：工具结果超过这个时间就从上下文里淡出。
    # 不加时限的话模型会被同一份数据反复牵住注意力
    # （实测一条天气搜索结果被换着措辞咀嚼了八个多小时）。
    TOOL_RESULT_TTL_SEC = float(os.getenv("ANIMAT_TOOL_RESULT_TTL", "90"))
    MAX_CONVERSATION_MEMORY = 10    # 对话记忆窗口
    HISTORY_MAX_CHARS = 2000        # 对话历史总字符上限（约 1300 token）

    # 回应对方后的静默期(秒)：期间不再主动开口，只允许内心独白。
    # 防止"答完一句紧接着自己再接一句"，造成连续刷屏。
    RESPONSE_COOLDOWN = 8.0

    # 主动呼唤(call)后的冷却期(秒)：同一诉求（如电量低求救）在窗口内
    # 不重复对外发声，避免模型换着措辞把同一件事连说十几遍。
    CALL_COOLDOWN = float(os.getenv("ANIMAT_CALL_COOLDOWN", "60"))

    def __init__(
        self,
        nervous: NervousSystem,
        memory: MemoryStore,
        llm_backend: Optional[LocalLLM] = None,
    ):
        self.nervous = nervous
        self.memory = memory
        self.llm = llm_backend or LocalLLM()

        # 注意力调制器（Phase 2 核心）
        self.modulator = AttentionModulator()
        self._body_state = BodyState()
        
        # 传感器窗口（Phase 3 核心）
        self.sensor_window: Optional[SensorWindow] = None

        # 状态
        self._running = False
        self._body_context: str = "各项指标正常，心情平静。"
        self._current_emotion: str = "SEEKING"
        self._drive_level: float = 0.0
        
        # 门控
        self.action_gate = ActionGate()
        self.journal = ThoughtJournal()

        # 对话记忆（外部输入）
        self._pending_external_text: Optional[str] = None
        # 外部输入入队时刻，用于统计排队延迟
        self._pending_external_at: float = 0.0
        self._external_text_lock = asyncio.Lock()

        # 统计
        self._thought_count = 0
        self._activation_count = 0
        self._start_time: float = 0.0
        # 上次回应对方的时间戳（用于静默期抑制自我发散）
        self._last_response_at: float = 0.0
        # 上次主动呼唤的时间戳（用于呼唤冷却，避免同一诉求刷屏）
        self._last_call_at: float = 0.0
        # 最近几次工具执行结果，回注给下一轮思维做上下文
        self._tool_results: list[str] = []
        # 与 _tool_results 一一对应的时间戳，用于按保鲜期淘汰
        self._tool_result_times: list[float] = []
        # 上下文溢出后的退避截止时间，避免失败重试把日志刷屏
        self._context_fail_cooldown: float = 0.0
        # 最近一次成功走完 LLM 调用的时刻。看门狗据此判断意识流是否停滞，
        # 与 _last_response_at 不同：那个记的是"对外说过话"，这里记的是
        # "脑子还在转"，自言自语和沉默期都不会让这个值停住。
        self._last_alive_at: float = 0.0
        # 停滞告警的抑制标记，避免卡死期间每个心跳周期都刷一条 WARNING
        self._stall_warned: bool = False

        # 初始化工具（全部通过 MQTT 发布，执行器订阅后执行）
        if tools_enabled():
            try:
                init_tools()
            except Exception as e:
                logger.error("工具初始化失败: %s", e)

        # 兼容 v0.1 接口
        self._processing = False
        self._conversation_history: list[dict] = []

    @property
    def busy(self) -> bool:
        """StreamCortex 永不'忙'——恒常运行"""
        return False

    async def register(self):
        """注册神经末梢"""
        # 接收躯体标记更新（从 homeostasis.py 来，保留兼容）
        await self.nervous.listen(Topic.SOMATIC_PROMPT, self._on_body_context_update)
        await self.nervous.listen(Topic.ARBITRATION, self._on_emotion_update)
        # 外部语音输入
        await self.nervous.listen("animat/external/speech", self._on_external_speech)
        logger.info("StreamCortex neurons registered")

    async def start(self):
        """启动恒常意识流（永不返回的协程）"""
        # 恢复对话历史（让重启后的 Animat 仍能延续上下文）
        await self._load_history()

        if not self.llm.ready:
            self.llm.load()
        
        # 初始化注意力调制器的 tokenizer
        # 本地后端暴露 _llm(llama对象); 云端后端直接暴露 tokenize 方法。
        # AttentionModulator.set_tokenizer 兼容二者，故统一传入 self.llm。
        self.modulator.set_tokenizer(self.llm)

        self._running = True
        self._start_time = time.time()
        self._last_alive_at = time.time()   # 启动即视为存活，给首轮推理留出窗口
        logger.info("═══ StreamCortex consciousness stream started ═══")
        
        while self._running:
            # 上下文溢出后的退避：不要每秒重试一次把日志刷屏
            if time.time() < self._context_fail_cooldown:
                await asyncio.sleep(1.0)
                continue

            # 等待间隔由 _interruptible_sleep 执行，可被外部输入打断。
            # 之前 sleep 写在 _think_cycle 内部，用户说完话必须等当前
            # 轮次 + 整个间隔跑完才被处理，实测排队 7-9 秒。
            wait_sec = 2.0
            try:
                wait_sec = await self._think_cycle() or 2.0
            except Exception:
                # 单个思维周期失败（多为 MQTT 瞬断重连期间的发布异常）不应杀死
                # 整条意识流——神经层会自动重连，这里退避后继续即可。
                logger.warning("StreamCortex cycle error (resuming)", exc_info=True)
                await asyncio.sleep(1.0)
                continue

            # 静默期仍然保留（防止回复后立刻自言自语刷屏），
            # 但它是可打断的：用户再说话会立刻跳出。
            await self._interruptible_sleep(wait_sec)

        logger.info("StreamCortex stopped")

    async def _interruptible_sleep(self, seconds: float) -> None:
        """可被外部输入打断的睡眠。

        每 200ms 检查一次待处理输入：用户一说话就提前醒来，
        把"等间隔走完"的时间省掉。没有输入时行为与普通 sleep 一致。
        """
        deadline = time.time() + seconds
        while self._running and time.time() < deadline:
            if self._pending_external_text:
                return                      # 有话说，立刻回去处理
            await asyncio.sleep(min(0.2, max(0.0, deadline - time.time())))

    async def stop(self):
        """停止意识流"""
        self._running = False

    # ── 身体状态更新回调 ──
    async def _on_body_context_update(self, topic: str, payload: dict):
        """接收从 homeostasis.py 来的身体状态上下文（保留兼容）"""
        prompt = payload.get("prompt", "")
        if prompt:
            self._body_context = prompt

    async def _on_emotion_update(self, topic: str, payload: dict):
        """接收情感仲裁结果，同步更新 BodyState"""
        self._current_emotion = payload.get("dominant_emotion", self._current_emotion)
        self._drive_level = payload.get("intensity", self._drive_level)
        
        # 从稳态值更新 BodyState（用于注意力调制）
        try:
            vitals = await self.memory.load_state("animat:homeostasis")
            if vitals:
                self._body_state = BodyState.from_vitals(vitals)
        except Exception:
            pass

    async def _on_external_speech(self, topic: str, payload):
        """接收人类语言输入（异步安全）

        兼容两种 payload: 标准 {"text": "..."} 字典，或裸字符串。
        nervous 在 JSON 解析失败时会把原始字符串透传，这里一并处理。
        """
        if isinstance(payload, dict):
            text = payload.get("text", "")
        elif isinstance(payload, str):
            text = payload
        else:
            text = str(payload)
        text = (text or "").strip()
        if not text:
            return
        async with self._external_text_lock:
            self._pending_external_text = text
            # 记录入队时刻，用于测量"用户说完 → 开始处理"的排队延迟
            self._pending_external_at = time.time()
        logger.info(f"StreamCortex: external speech queued: {text[:40]}")

    # ── 主循环 ──
    async def _think_cycle(self):
        """一次完整的思维循环"""
        cycle_t0 = time.time()

        # 1. 检查是否有外部输入
        external_text = None
        queue_delay = 0.0
        async with self._external_text_lock:
            if self._pending_external_text:
                external_text = self._pending_external_text
                self._pending_external_text = None
                # 用户说完到现在才开始处理，中间隔了多久
                if self._pending_external_at:
                    queue_delay = cycle_t0 - self._pending_external_at
                    self._pending_external_at = 0.0

        # 2. 构建上下文
        system_prompt = CONSCIOUSNESS_SYSTEM_PROMPT.format(
            body_context=self._body_context[:500]  # 限制上下文长度
        )

        # 3. 产生思维
        think_t0 = time.time()
        thought = None
        try:
            if external_text:
                # 外部输入 → 产生回复思维
                thought = await self._think_response(system_prompt, external_text)
            else:
                # 自发思维
                thought = await self._think_spontaneous(system_prompt)
        except Exception:
            logger.error("Error during think cycle", exc_info=True)
            # 明确返回较长间隔：LLM 出错时不要高频重试
            return 5.0
        think_ms = (time.time() - think_t0) * 1000

        # 看门狗依据：只要 LLM 还能返回（哪怕返回空），就说明意识流活着。
        # 放在 thought is None 判断之前，因为"模型这轮没说话"与"卡死"是
        # 两回事，前者不应触发停滞告警。
        self._last_alive_at = time.time()

        if thought is None:
            return 5.0

        # 4. 门控评估
        max_drive = self._drive_level
        is_external = external_text is not None
        decision = self.action_gate.evaluate(
            thought=thought,
            drive_level=max_drive,
            is_external_trigger=is_external,
        )

        # 3.5 静默期抑制：刚回应过对方时，抑制自发开口，
        #     否则模型会顺着自己上一句继续发散，形成连续刷屏。
        if (
            decision.should_activate
            and not is_external
            and decision.action_type == "speak"
            and (time.time() - self._last_response_at) < self.RESPONSE_COOLDOWN
        ):
            decision.should_activate = False
            decision.action_type = "none"
            decision.reason += ", response_cooldown"

        # 4.5 标注表达类型：回应 / 主动说 / 自言自语
        if is_external:
            thought.utterance_type = UtteranceType.RESPONSE
        else:
            thought.utterance_type = classify_utterance(self._current_emotion, max_drive)

        # 4.6 未被提问时，只有模型判定为"想对外呼唤"(call)才真的开口。
        #     仅靠情感状态判断不够：长期孤独会把情感抬到 PANIC，使每条
        #     自发思维都被判成 PROACTIVE，内心独白便整片涌进对话通道。
        #     这里以模型给出的语义意图为准，不做任何关键词匹配。
        if (
            decision.should_activate
            and decision.action_type == "speak"
            and not is_external
            and not _INTENT_IS_OUTWARD.get(thought.intent, False)
        ):
            decision.should_activate = False
            decision.action_type = "none"
            decision.reason += ", soliloquy_silent"

        # 4.7 主动呼唤后的冷却期：同一诉求(如"电量低求救")若不加约束，
        #     模型会每轮换种措辞再说一遍，连说近十条。REFRACTORY 只拦
        #     逐字重复，拦不住同义改写，故在此按时间窗口限流。
        if (
            decision.should_activate
            and not is_external
            and thought.intent == "call"
            and (time.time() - self._last_call_at) < self.CALL_COOLDOWN
        ):
            decision.should_activate = False
            decision.action_type = "none"
            decision.reason += ", call_cooldown"

        # 5. 记录
        self.journal.record(thought)
        self._thought_count += 1

        # 5.5 广播每一次内部思维（内心独白），供调试端实时监控
        await self._broadcast_thought(thought, decision.should_activate)

        # 6. 如果门控开放 → 发射行为
        if decision.should_activate:
            self._activation_count += 1
            await self._emit_action(decision, thought)

        # 6.2 执行模型请求的工具调用（LED/移动/搜索等）。
        #     工具是"行动"，与上面的"说不说"解耦：独白也可以顺手调个 LED。
        if thought.tool_calls:
            await self._execute_tool_calls(thought.tool_calls)

        # 6.5 记录时间戳：回应后进入静默期，呼唤后进入呼唤冷却，
        #     两者都是为了抑制"同一件事连续刷屏"。
        if decision.should_activate:
            if is_external:
                self._last_response_at = time.time()
                # 对方已经回应了，之前的呼唤诉求已被"听见"，
                # 清除呼唤冷却，让接下来的新诉求能正常发出。
                self._last_call_at = 0.0
            elif thought.intent == "call":
                self._last_call_at = time.time()

        # 7. 动态等待间隔
        if external_text:
            wait_sec = self.RESPONSE_COOLDOWN  # 回应后进入静默期，不急着再说话
        elif max_drive > 0.5:
            wait_sec = self.THINK_INTERVAL_DRIVEN
        elif max_drive < 0.2:
            wait_sec = self.THINK_INTERVAL_IDLE
        else:
            wait_sec = self.THINK_INTERVAL_SEC

        # 引入随机抖动，避免机械节奏
        import random
        wait_sec *= random.uniform(0.8, 1.2)

        # ── 耗时诊断 ──
        # 拆成三段看清延迟来源：
        #   queue  = 用户说完 → 本轮开始处理（排队）
        #   think  = 一次 LLM 调用的实际耗时
        #   total  = 本轮从开始到进入睡眠的总时长
        cycle_ms = (time.time() - cycle_t0) * 1000
        # 每轮都记。此前只在 external 或 think>3s 时打印，导致安静的
        # 自发思维完全不可见，看起来像"意识流停了"。
        logger.info(
            "⏱ latency: kind=%s queue=%.2fs think=%.2fs cycle=%.2fs sleep=%.1fs",
            "external" if external_text else "spontaneous",
            queue_delay, think_ms / 1000, cycle_ms / 1000, wait_sec,
        )

        # 睡眠由主循环统一负责，以便被外部输入打断。
        # 这里只把本轮的间隔需求交回去。
        return wait_sec

    # ── 阻塞式推理的线程化封装 ────────────────────────────────
    # llama.cpp 的 create_chat_completion 是同步阻塞调用，耗时可达数秒到
    # 十几秒。直接在 async def 里调用会堵死整个事件循环，导致心跳停摆、
    # MQTT 消息全部堆积、看门狗也一起失联——实测表现为进程 57% CPU、
    # GPU 0%、日志静默但进程还在。所有推理一律经这两个方法丢进线程池。

    async def _llm_complete(self, label: str = "complete", **kwargs) -> dict:
        """阻塞式 complete 的异步封装，带调用日志。

        label 只用于日志，不会传给后端——它的作用是让日志能区分
        "回复"、"工具结果回注"、"只说不做重试"这几种不同用途的调用。
        """
        t0 = time.time()
        try:
            return await asyncio.to_thread(self.llm.complete, **kwargs)
        except Exception:
            logger.error("🤖 LLM %s 调用失败 (%.2fs)", label, time.time() - t0)
            raise
        finally:
            # 无论成功失败都记一行，这样"模型到底有没有被调用过"永远可查。
            logger.info(
                "🤖 LLM %s: max_tokens=%s msgs=%d",
                label, kwargs.get("max_tokens"), len(kwargs.get("messages") or []),
            )

    def _consume_stream_think(self, kwargs: dict) -> tuple[str, list[dict]]:
        """消费 stream_think 生成器直到结束，返回 (累积正文, tool_calls)。

        作为同步函数交给 asyncio.to_thread 执行；生成器本身是同步的，
        在线程里迭代不会影响事件循环。
        """
        t0 = time.time()
        accumulated, collected, raw_len = "", [], 0
        try:
            for chunk in self.llm.stream_think(**kwargs):
                accumulated = chunk.accumulated_text
                raw_len = max(raw_len, len(getattr(chunk, "raw_text", "") or ""))
                if chunk.tool_calls:
                    collected = chunk.tool_calls
                if chunk.is_finished:
                    break
        finally:
            # 自发思维走的是这条流式路径，若不记录，日志里就完全看不到
            # "模型被调用过"。
            # raw= 是剥离前的原始长度：out=0ch 且 raw>0 说明模型生成的内容
            # 全在 <think> 里；raw 也是 0 才是真的什么都没生成。
            logger.info(
                "🤖 LLM think: %.2fs out=%dch raw=%dch tools=%d",
                time.time() - t0, len(accumulated or ""), raw_len, len(collected),
            )
        return accumulated, collected

    async def _think_spontaneous(self, system_prompt: str) -> Optional[Thought]:
        """产生自发思维"""
        # 获取最近记忆
        recent_memories = []
        try:
            mems = await self.memory.recall_episodes(limit=3)
            for m in mems:
                recent_memories.append(
                    f"[{m.get('emotion', '?')}] {m.get('summary', '')}"
                )
        except Exception:
            pass

        # Phase 3: 传感器上下文 — 优先使用 SensorWindow 原始数据
        if self.sensor_window:
            sensor_context = self.sensor_window.to_context(max_frames_per_modality=3)
        else:
            # 回退：使用 homeostasis 的文本描述
            sensor_context = self._body_context[:300]

        # 产生思维（使用 logit bias 调制注意力）
        accumulated = ""
        collected_calls: list[dict] = []
        try:
            # 召回已记住的对方信息，避免幻觉编造
            self_info = await self._recall_self_info()
            if self_info:
                system_prompt = system_prompt + "\n[关于对方你已知的信息] " + self_info



            # 计算当前身体状态的 token 偏移
            logit_bias = self.modulator.compute_logit_bias(self._body_state)

            # 最近对话历史：只取"对方说过的话"，不喂自己的回复。
            # 否则模型会把自己上一轮的输出当作待接续的话题，形成
            # "自己问、自己答"的发散循环（实测一句话连出十余条回复）。
            recent_conv = (
                [m for m in self._conversation_history if m.get("role") == "user"][-3:]
                if self._conversation_history
                else None
            ) or None

            accumulated, collected_calls = await asyncio.to_thread(
                self._consume_stream_think,
                dict(
                    system_prompt=system_prompt,
                    context_memories=recent_memories,
                    sensor_context=sensor_context,   # ← Phase 3: 原始传感器数据
                    max_tokens=self.MAX_THINK_TOKENS,
                    logit_bias=logit_bias,           # ← Phase 2: 注意力调制
                    conversation_context=recent_conv,
                    tools=self._enabled_tools(),
                    tool_results=self._current_tool_results() or None,
                ),
            )

        except Exception as e:
            if "exceed context window" in str(e) or "context" in str(e).lower():
                # 上下文溢出：清掉体积最大的回注内容（工具结果 + 记忆）
                # 后重试一次，避免直接放弃。
                logger.warning("上下文溢出，精简后重试: %s", e)
                self._tool_results.clear()
                try:
                    accumulated, _ = await asyncio.to_thread(
                        self._consume_stream_think,
                        dict(
                            system_prompt=system_prompt,
                            context_memories=[],
                            sensor_context=sensor_context[:200] if sensor_context else "",
                            max_tokens=self.MAX_THINK_TOKENS,
                            conversation_context=None,
                            tools=None,
                        ),
                    )
                except Exception as e2:
                    logger.error("精简重试仍失败: %s", e2)
                    # 退避：让循环慢下来，不要每秒刷一次错误
                    self._context_fail_cooldown = time.time() + 30.0
                    return None
            else:
                logger.error(f"Spontaneous think failed: {e}")
                return None

        if not accumulated.strip():
            # 模型这轮什么都没说（常见于只输出了 <intent>/<did> 等控制标签，
            # 剥离后为空）。此前这里静默 return None，调用方也只静默返回 5.0，
            # 结果是"明明每 5 秒都在推理，日志里却一条都没有"，从外部完全
            # 无法区分"安静"和"死了"。记一条 DEBUG 保证可观测，
            # 但不升到 INFO——安静期刷屏同样会把日志淹掉。
            logger.debug("spontaneous: empty output, skipped")
            return None

        # 模型按 prompt 输出 <intent>…</intent> + 正文，这里解析出意图
        intent, body = _parse_intent(accumulated, default="none")

        # 本轮调用了工具时，正文通常只是"我应该调用 web_search"这类
        # 过程性叙述，不是要对外说的话。工具本身就是行为，正文静默，
        # 否则会把"我要去搜索"这种自言自语当成回答念出来。
        if collected_calls:
            intent = "none"

        if not body:
            return None

        # 计算相关性评分
        relevance = self._compute_relevance(body)

        return Thought(
            text=body,
            relevance=relevance,
            dominant_emotion=self._current_emotion,
            trigger="spontaneous",
            intent=intent,
            tool_calls=collected_calls,
            body_state_snapshot={"emotion": self._current_emotion},
        )

    async def _think_response(
        self, system_prompt: str, external_text: str
    ) -> Optional[Thought]:
        """产生对人类的回复思维"""
        # 对话场景使用专用 prompt：直接口语回答，不写肢体动作描写
        system_prompt = DIALOGUE_SYSTEM_PROMPT.format(
            body_context=self._body_context[:500]
        )
        # 1) 把外部输入记入对话历史（用于上下文连贯，避免"记不住名字"）
        self._conversation_history.append({"role": "user", "content": external_text})
        # 仅保留最近 N 轮
        if len(self._conversation_history) > self.MAX_CONVERSATION_MEMORY * 2:
            self._conversation_history = self._conversation_history[-self.MAX_CONVERSATION_MEMORY * 2:]
        # 按总字符数再削一刀：条数达标不代表长度达标，
        # 几轮长回答叠起来足以撑爆 n_ctx。
        self._trim_history_by_length()

        # 2) 抽取身份/事实性信息存入语义记忆（如"我叫XX"）
        await self._maybe_remember_facts(external_text)

        # 3) 召回已记住的自我相关信息，注入 system prompt
        self_info = await self._recall_self_info()
        if self_info:
            system_prompt = system_prompt + "\n[关于对方你已知的信息] " + self_info

        # 4) 构建对话消息（带历史，让回复有上下文）
        messages = [
            {"role": "system", "content": system_prompt},
            *self._conversation_history,
        ]
        # 对话时也可以调用工具（用户可能直接要求"帮我查一下…"）。
        # 把最近仍新鲜的工具结果一并给出，避免它不知道刚才已经查过。
        fresh_results = self._current_tool_results()
        if fresh_results:
            messages.insert(1, {
                "role": "system",
                "content": "## 你刚才做过的事:\n" + "\n".join(fresh_results),
            })

        tool_calls: list[dict] = []
        try:
            response = await self._llm_complete(
                label="reply",
                messages=messages,
                max_tokens=self.MAX_REPLY_TOKENS,
                tools=self._enabled_tools(),
            )
            choice = response["choices"][0]
            tool_calls = choice["message"].get("tool_calls") or []
            text = (choice["message"].get("content") or "").strip()
        except Exception as e:
            logger.error(f"Response think failed: {e}")
            return None

        # 解析模型的自评标记：<did>claim</did> 表示它承认"嘴上说要动作，
        # 但这一轮没调工具"。判断交给模型完成，代码不做关键词匹配。
        claims_only, text = _parse_did_marker(text)

        # 模型自认没做，且确实没有 tool_calls → 补一次强制调用
        if claims_only and not tool_calls:
            logger.info("模型自评'只说不做'，强制重试并要求调用工具")
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "system",
                "content": (
                    "你标记了 <did>claim</did>，说明你只是说了要做，"
                    "对方什么也不会看到。现在必须直接调用对应的工具完成这件事，"
                    "不要再只用文字描述。"
                ),
            })
            try:
                forced = await self._llm_complete(
                    label="did-retry",
                    messages=messages,
                    max_tokens=self.MAX_REPLY_TOKENS,
                    tools=self._enabled_tools(),
                )
                fc = forced["choices"][0]
                tool_calls = fc["message"].get("tool_calls") or []
                if tool_calls:
                    logger.info("强制重试成功，工具: %s",
                                [c["function"]["name"] for c in tool_calls])
                    # 采用重试拿到的正文（通常为空或一句确认）
                    text = (fc["message"].get("content") or "").strip() or text
            except Exception as e:
                logger.warning("强制工具调用重试失败: %s", e)

        # 模型在对话里要求调用工具：先执行，再让它基于结果回答。
        # 注意这是一轮对话里的第二次 LLM 调用，耗时直接翻倍——
        # 诊断日志里单独计时，方便判断"工具型回复"慢在哪一段。
        if tool_calls:
            tool_t0 = time.time()
            await self._execute_tool_calls(tool_calls)
            tool_ms = (time.time() - tool_t0) * 1000

            results = "\n".join(self._tool_results[-2:])
            messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
            messages.append({
                "role": "system",
                "content": f"工具执行结果如下，请基于它直接用口语回答对方：\n{results}",
            })
            retry_t0 = time.time()
            try:
                retry = await self._llm_complete(
                    label="context-retry",
                    messages=messages,
                    max_tokens=self.MAX_REPLY_TOKENS,
                )
                text = (retry["choices"][0]["message"].get("content") or "").strip()
            except Exception as e:
                logger.error(f"Tool follow-up failed: {e}")
                text = ""
            logger.info(
                "⏱ tool_round: exec=%.2fs followup_llm=%.2fs",
                tool_ms / 1000, time.time() - retry_t0,
            )

        # 对话场景必然是对外说话；若模型自带意图/自评标签则剥离，
        # 避免 <intent>/<did> 混进给用户看的正文（工具后续轮次也可能带出来）。
        _, text = _parse_did_marker(text)
        _, text = _parse_intent(text, default="call")
        if not text:
            return None

        # 记录 assistant 回复到历史
        self._conversation_history.append({"role": "assistant", "content": text})

        # 持久化对话历史（与稳态指标一样重启后延续）
        await self._save_history()

        # 回复思维的相关性通常是高的（有人在等）
        return Thought(
            text=text,
            relevance=0.85,  # 回复人类默认高相关性
            dominant_emotion=self._current_emotion,
            trigger="external",
            intent="call",  # 对人类的回复就是对外说话
        )

    async def _maybe_remember_facts(self, text: str):
        """从用户语句中尝试抽取身份/事实信息，写入语义记忆。
        失败则静默跳过，不影响主流程。"""
        try:
            prompt = (
                "从下面这句话中抽取关于'对方（人类）'的稳定事实信息（如姓名、称呼、喜好）。\n"
                "只输出 JSON，例如 {\"name\": \"小明\", \"note\": \"喜欢喝咖啡\"}；"
                "若没有可记忆的事实，输出 {}. 不要输出其他内容。\n\n"
                f"句子：{text}"
            )
            resp = await self._llm_complete(
                label="self-info",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=80,
            )
            import json as _json
            raw = resp["choices"][0]["message"]["content"].strip()
            # 容错：截取第一个 { 到最后一个 }
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                return
            data = _json.loads(raw[start:end + 1])
            if not isinstance(data, dict) or not data:
                return
            # 合并进已有的 self_info
            existing = await self.memory.recall_semantic("self_info") or {}
            existing.update({k: v for k, v in data.items() if v})
            await self.memory.store_semantic("self_info", existing)
            logger.info(f"Remembered self_info: {existing}")
        except Exception as e:
            logger.debug(f"Fact extraction skipped: {e}")

    async def _recall_self_info(self) -> str:
        """召回已记住的关于对方的事实，格式化为单行文本"""
        try:
            info = await self.memory.recall_semantic("self_info")
            if not info:
                return ""
            parts = []
            if info.get("name"):
                parts.append(f"对方叫{info['name']}")
            for k, v in info.items():
                if k == "name":
                    continue
                parts.append(f"{k}:{v}")
            return "；".join(parts)
        except Exception:
            return ""

    # ── 对话历史持久化 ──
    def _trim_history_by_length(self) -> None:
        """把对话历史的总字符数压到预算内（从最旧的开始丢）。"""
        budget = self.HISTORY_MAX_CHARS
        total = sum(len(m.get("content") or "") for m in self._conversation_history)
        while total > budget and len(self._conversation_history) > 2:
            dropped = self._conversation_history.pop(0)
            total -= len(dropped.get("content") or "")

    async def _save_history(self):
        """把最近对话历史写入 Redis（重启可延续）"""
        try:
            await self.memory.store_semantic(
                "conversation_history",
                self._conversation_history[-self.MAX_CONVERSATION_MEMORY * 2:],
            )
        except Exception as e:
            logger.debug(f"Save history skipped: {e}")

    async def _load_history(self):
        """从 Redis 恢复上次对话历史"""
        try:
            hist = await self.memory.recall_semantic("conversation_history")
            if isinstance(hist, list) and hist:
                self._conversation_history = hist
                logger.info(f"Conversation history restored ({len(hist)} msgs)")
        except Exception as e:
            logger.debug(f"Load history skipped: {e}")

    def _compute_relevance(self, text: str) -> float:
        """
        评估思维的行为相关性
        
        策略:
        - 包含行动意向的词 → 高相关性
        - 纯观察/感受 → 中等
        - 空/重复 → 低
        """
        text_lower = text.lower()
        
        # 行动词
        action_words = ["去", "要", "想", "应该", "必须", "寻找", "充电",
                        "求助", "警告", "逃", "找", "说"]
        # 感受词
        feeling_words = ["感觉", "好像", "似乎", "有点", "觉得", "无聊",
                         "舒服", "害怕", "孤单", "开心"]
        
        action_score = sum(1 for w in action_words if w in text_lower) * 0.15
        feeling_score = sum(1 for w in feeling_words if w in text_lower) * 0.08
        
        # 文本长度惩罚（太短可能不完整，太长可能是幻觉）
        length = len(text)
        if length < 5:
            length_penalty = -0.3
        elif length > 200:
            length_penalty = -0.1
        else:
            length_penalty = 0.0
        
        score = min(action_score + feeling_score + length_penalty + 0.3, 1.0)
        return max(0.0, min(1.0, score))

    def _enabled_tools(self) -> Optional[list[dict]]:
        """返回本轮可用的工具清单（OpenAI schema）。

        低电量时禁用移动类工具——不能让模型把最后的电耗在走路和亮灯上。
        """
        if not tools_enabled():
            return None
        names = list(registry.names())
        battery = self._body_state.get("battery_pct", 100) if isinstance(self._body_state, dict) else 100
        try:
            if float(battery) < 20:
                names = [n for n in names if n not in ("move", "set_led")]
        except (TypeError, ValueError):
            pass
        if not names:
            return None
        return registry.openai_tools(enabled=names)

    async def _execute_tool_calls(self, calls: list[dict]) -> None:
        """执行模型请求的工具调用，结果记日志并回注下一轮上下文。"""
        for call in calls:
            fn = (call or {}).get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("工具 %s 的参数不是合法 JSON: %r", name, raw_args)
                continue

            result = await registry.execute(name, args, nervous=self.nervous)
            if result.skipped:
                logger.info("工具 %s 跳过: %s", name, result.skipped)
            elif result.ok:
                logger.info("工具 %s 执行成功", name)
            else:
                logger.warning("工具 %s 失败: %s", name, result.detail)

            # 回注结果：让下一轮思维知道自己刚才做了什么。
            # 必须限长：搜索结果动辄上千字，全量回注会直接撑爆 n_ctx
            # （曾导致 5075 > 4096 的持续报错）。
            # 同时必须带"保鲜期"：只按条数淘汰的话，一条搜索结果会一直
            # 留在 prompt 里，把模型的注意力钉死在同一个话题上。
            line = result.to_context_line()
            if line:
                if len(line) > self.TOOL_RESULT_MAX_CHARS:
                    line = line[:self.TOOL_RESULT_MAX_CHARS] + "…(已截断)"
                self._tool_results.append(line)
                self._tool_result_times.append(time.time())
                self._prune_tool_results()

    def _prune_tool_results(self) -> None:
        """按保鲜期与体积预算裁剪工具结果。

        只按条数淘汰是不够的：一条查天气的结果会一直占据 prompt，
        模型每轮都在重读同一组数字，于是话题黏着到无法自我脱离
        （实测一句话之后同一份天气数据被复述十余次）。
        """
        now = time.time()
        # 1) 过期淘汰
        while self._tool_result_times and (now - self._tool_result_times[0]) > self.TOOL_RESULT_TTL_SEC:
            self._tool_results.pop(0)
            self._tool_result_times.pop(0)
        # 2) 条数上限
        while len(self._tool_results) > 2:
            self._tool_results.pop(0)
            self._tool_result_times.pop(0)
        # 3) 体积上限
        total = sum(len(x) for x in self._tool_results)
        while total > self.TOOL_RESULTS_TOTAL_MAX and len(self._tool_results) > 1:
            total -= len(self._tool_results.pop(0))
            self._tool_result_times.pop(0)

    def _current_tool_results(self) -> list[str]:
        """取当前仍"新鲜"的工具结果，供 prompt 回注。"""
        self._prune_tool_results()
        return list(self._tool_results)

    async def _emit_action(self, decision: ActionDecision, thought: Thought):
        """通过 MQTT 发射行为"""
        # 清洗待发送文本：去掉调试前缀、肢体动作描写等
        clean_text = _sanitize_output(thought.text)
        # move / gesture 等非说话动作不应携带大段文本
        if decision.action_type != "speak":
            clean_text = ""
        # 构建与 v0.1 兼容的输出格式
        output = {
            "intent": decision.action_type,
            "text": clean_text,
            "action_detail": clean_text,
            "utterance_type": thought.utterance_type,
            "_source": "stream_cortex",
            "_activation_score": decision.activation_score,
        }
        try:
            await self.nervous.fire(Topic.CORTEX_OUTPUT, output)
        except Exception:
            logger.warning("StreamCortex emit failed (MQTT unavailable?): %s", decision.action_type)
            return
        logger.info(f"StreamCortex emit: {decision.action_type} | {thought.text[:60]}")

    async def _broadcast_thought(self, thought: Thought, activated: bool):
        """把每一次内部思维广播到 animat/cortex/thought（内心独白流）。

        无论是否被门控放行都会广播——这模拟了人类'大多数想法不说出口'，
        调试端因此能看到 Animat 完整的内心活动，而不只是它最终说出来的那部分。
        """
        try:
            await self.nervous.fire(Topic.CORTEX_THOUGHT, {
                "text": thought.text,
                "trigger": thought.trigger,            # spontaneous / external
                "utterance_type": thought.utterance_type,  # soliloquy / proactive / response
                "emotion": thought.dominant_emotion,
                "relevance": round(thought.relevance, 3),
                "activated": activated,                # 是否冲破门控成为外部行为
                "timestamp": time.time(),
            }, qos=0)  # 高频遥测，用 qos=0 即可
        except Exception:
            pass

    # ── 统计 ──
    def stats(self) -> dict:
        """获取意识流统计"""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        rpm = (self._thought_count / (uptime / 60)) if uptime > 0 else 0
        activation_rate = (
            self._activation_count / self._thought_count * 100
            if self._thought_count > 0
            else 0
        )
        return {
            "running": self._running,
            "uptime_sec": round(uptime, 1),
            "thoughts_total": self._thought_count,
            "thoughts_per_min": round(rpm, 1),
            "activations": self._activation_count,
            "activation_rate_pct": round(activation_rate, 1),
            "current_emotion": self._current_emotion,
            "drive_level": round(self._drive_level, 2),
            "journal": self.journal.summary(),
        }
