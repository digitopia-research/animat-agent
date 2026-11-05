"""
本地 LLM 后端 — 基于 llama-cpp-python 的推理服务

提供:
  1. 加载/卸载本地 GGUF 模型
  2. 流式/非流式对话推理
  3. KV Cache 持久化（为恒常意识流准备）
  4. Logit bias 支持（为注意力调制器准备）

未来可以无缝切换到 llama-server HTTP API（宿主机 GPU 推理）
"""

import os
import re
import json
import shutil
import logging
from typing import Optional, AsyncIterator
from dataclasses import dataclass, field

logger = logging.getLogger("animat.llm_backend")

# Qwen3 等模型默认开启"思考"(thinking)模式，会在输出里包裹 <think>...</think>
# 推理块。本项目需要模型直接产出简短动作/想法，因此：
#   1) 通过 chat_template_kwargs 关闭 thinking (Qwen3 官方模板支持 enable_thinking)
#   2) 作为兜底，再从输出中剥离任何残留的 <think> 块
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

# Qwen 系列在部分 chat 模板下不返回结构化 tool_calls，而是把调用写成
# <tool_call>{"name": "...", "arguments": {...}}</tool_call> 塞进 content。
# llama-cpp-python 的 chat_template.default 不会解析它，这里做兜底提取。
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """移除输出中的 <think>...</think> 思考块（含未闭合的尾部 <think>）。"""
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def _strip_tool_call_partial(text: str) -> str:
    """流式场景下剥离 <tool_call> 块，含尚未闭合的半截标签。

    流式时标签往往被切成多片到达，如果只匹配完整块，
    中途的 "<tool_call>{"name":…" 会被当成正文发出去。
    """
    if not text or "<tool_call>" not in text:
        return text
    # 先去掉完整块
    text = _TOOL_CALL_RE.sub("", text)
    # 再去掉未闭合的尾部（有开标签但还没等到闭标签）
    start = text.find("<tool_call>")
    if start != -1:
        text = text[:start]
    return text.strip()


def _repair_truncated_json(payload: str) -> Optional[dict]:
    """尝试修复被 max_tokens 截断的工具调用 JSON。

    模型生成长参数时可能被截断，典型形态:
        {"name": "move", "arguments": {"direction": "forward", "distance_cm": 100
    只要单次调用本身信息完整（name + 若干成对参数），就值得尽量救回来，
    而不是整条丢弃。
    """
    text = payload.strip()
    if not text:
        return None

    # 先去掉末尾残缺的键值对（如 `, "speed": 0.` 或 `, "args`）
    for cut in range(len(text), 0, -1):
        candidate = text[:cut].rstrip()
        candidate = candidate.rstrip(",")
        # 逐级补全括号
        opens = candidate.count("{") - candidate.count("}")
        if opens < 0:
            continue
        fixed = candidate + "}" * opens
        try:
            obj = json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and (obj.get("name") or obj.get("function")):
            return obj
    return None


def _extract_tool_calls(text: str) -> tuple[list[dict], str]:
    """从 content 文本中提取 <tool_call> 标签内的调用。

    返回 (tool_calls, 去掉标签后的正文)。tool_calls 采用 OpenAI 结构，
    与原生返回格式统一，上层无需区分来源。
    """
    if not text or "<tool_call>" not in text:
        return [], text

    # 完整块 + 末尾可能存在的未闭合块（被 max_tokens 截断的情形）
    raw_payloads = _TOOL_CALL_RE.findall(text)
    tail = text.rsplit("<tool_call>", 1)
    if len(tail) == 2 and "</tool_call>" not in tail[1]:
        raw_payloads = list(raw_payloads) + [tail[1]]

    calls: list[dict] = []
    for i, raw in enumerate(raw_payloads):
        payload = raw.strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            obj = _repair_truncated_json(payload)
            if obj is not None:
                logger.info("工具调用 JSON 被截断，已修复后继续: %s", obj.get("name"))
        if not isinstance(obj, dict):
            logger.debug("无法解析 tool_call 负载: %r", payload[:120])
            continue
        name = obj.get("name") or (obj.get("function") or {}).get("name")
        args = obj.get("arguments")
        if args is None:
            args = (obj.get("function") or {}).get("arguments", {})
        if not name:
            continue
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({
            "id": obj.get("id") or f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })

    cleaned = _TOOL_CALL_RE.sub("", text).strip()
    return calls, cleaned


def _detect_nvidia_gpu() -> bool:
    """探测系统是否可用 NVIDIA GPU (通过 nvidia-smi)。

    仅用于决定默认是否卸载到 GPU，不依赖具体驱动/CUDA 版本。
    若 llama-cpp-python 未带 CUDA 后端，即便探测到 GPU 也会在加载时回退 CPU。
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        ret = __import__("subprocess").run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return ret.returncode == 0 and bool(ret.stdout.strip())
    except Exception:
        return False


# 用户未显式设置 ANIMAT_N_GPU_LAYERS 时:
#   - 探测到 NVIDIA GPU → 默认全卸载到 GPU (99 层, llama.cpp 会按模型实际层数截断)
#   - 未探测到         → 纯 CPU (0)
_GPU_AUTO = 99 if _detect_nvidia_gpu() else 0
_N_GPU_LAYERS_ENV = os.getenv("ANIMAT_N_GPU_LAYERS")


@dataclass
class LLMConfig:
    model_path: str = os.getenv(
        "ANIMAT_MODEL_PATH",
        "/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    )
    n_ctx: int = int(os.getenv("ANIMAT_N_CTX", "4096"))
    n_threads: int = int(os.getenv("ANIMAT_N_THREADS", "8"))
    # 显式设置 ANIMAT_N_GPU_LAYERS 则优先；否则自动探测 GPU 决定是否卸载
    n_gpu_layers: int = (
        int(_N_GPU_LAYERS_ENV) if _N_GPU_LAYERS_ENV is not None else _GPU_AUTO
    )
    temperature: float = 0.7
    top_p: float = 0.9
    # 流式推理配置
    stream_interval: float = 0.05  # 流式输出的最低间隔(秒)
    # KV Cache 配置
    cache_prompt: bool = True      # 启用 KV cache 持久化
    # llama.cpp 详细日志 (设为 1 可在启动时看到 ggml_backend_cuda 初始化，
    # 用于确认 GPU 是否生效；默认关闭保持安静)
    verbose: bool = bool(int(os.getenv("ANIMAT_LLM_VERBOSE", "0")))


@dataclass
class StreamThought:
    """流式推理中产生的一个思维片段"""
    token_id: int
    token_text: str
    accumulated_text: str
    is_finished: bool = False
    finish_reason: Optional[str] = None
    # 未经剥离的原始累加输出。诊断用：当 accumulated_text 为空时，
    # 可以据此区分"模型真的没生成"和"生成了但全在 <think> 里被剥离"。
    raw_text: str = ""
    # 仅在流结束时填充：模型请求的工具调用（OpenAI tool_calls 格式）
    tool_calls: list = field(default_factory=list)


class LocalLLM:
    """
    本地 LLM 后端封装
    
    设计原则:
    - 接口与 OpenAI 兼容（方便未来切换到云端 API）
    - 支持流式输出（为恒常意识流提供 token 级别的控制）
    - KV Cache 持久化（减少重复推理）
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._llm = None
        self._loaded = False

    def load(self):
        """加载模型到内存"""
        if self._loaded:
            return
        import llama_cpp
        t0 = __import__("time").time()
        mode = "GPU" if self.config.n_gpu_layers > 0 else "CPU"
        logger.info(
            f"Loading local LLM: {self.config.model_path} "
            f"[{mode} mode, n_gpu_layers={self.config.n_gpu_layers}]"
        )
        self._llm = llama_cpp.Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose,
        )
        self._loaded = True
        logger.info(
            f"Local LLM loaded in {__import__('time').time() - t0:.1f}s "
            f"[{mode} mode, n_ctx={self.config.n_ctx}, threads={self.config.n_threads}]"
        )

    def unload(self):
        self._llm = None
        self._loaded = False
        logger.info("Local LLM unloaded")

    @property
    def ready(self) -> bool:
        return self._loaded and self._llm is not None

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        """委托给底层 llama 对象的 tokenize，供 AttentionModulator 使用。

        签名与 llama-cpp-python / CloudLLM 对齐：接受 bytes，返回 token_id 列表。
        """
        self._ensure_loaded()
        return self._llm.tokenize(text, add_bos=add_bos)

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
    ) -> dict:
        """
        非流式推理（兼容 OpenAI 返回格式）

        Args:
            messages: chat messages
            max_tokens: 最大生成 token 数
            temperature: 温度（None=使用默认）
            logit_bias: token_id → bias_value 映射（Phase 2 注意力调制器用）
            tools: OpenAI tools schema 列表；None 表示本轮不做工具调用
            tool_choice: "auto" 让模型自行决定是否调用
        """
        self._ensure_loaded()
        kwargs: dict = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            logit_bias=logit_bias,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        response = self._create_chat_completion(**kwargs)
        choice = (response.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            content = _strip_thinking(msg["content"])
            # 部分模板把工具调用写在 content 里，这里解析成结构化 tool_calls，
            # 使上层只需处理一种格式。
            if tools and "<tool_call>" in content:
                calls, content = _extract_tool_calls(content)
                if calls:
                    msg["tool_calls"] = calls
                    choice["finish_reason"] = "tool_calls"
            msg["content"] = content
        return response

    def _create_chat_completion(self, **kwargs):
        """封装 llama_cpp 的 create_chat_completion，并关闭 Qwen3 思考模式。

        关于 enable_thinking 的一段历史：
        早先这里用 chat_template_kwargs={"enable_thinking": False} 注入，
        但 llama-cpp-python 0.3.x 的 create_chat_completion() 根本不接受
        这个参数，每次都会抛 TypeError，然后被 except 吞掉、剔除参数重试。
        实际结果是"看起来关了思考，其实一直开着"——模型每轮生成四五百
        字符全在 <think> 里，被 _strip_thinking 剥离后正文为空，
        表现为日志里 out=0ch 而 raw=471ch，意识流"静悄悄"。

        Qwen3 模板真正认的是消息里的 /no_think 软开关，且必须放在
        最后一条 user 消息末尾（放 system 里无效，实测仍会思考）。
        这里统一追加，比依赖模板变量更可靠。
        """
        kwargs["messages"] = self._apply_no_think(kwargs.get("messages") or [])

        try:
            return self._llm.create_chat_completion(**kwargs)
        except Exception as e:
            # 只有"模板不认识 tools 参数"才降级；上下文溢出等错误
            # 必须原样抛出，否则会被误判成模板不支持，掩盖真正的问题。
            if "tools" in kwargs and "context" not in str(e).lower():
                logger.warning("当前模型的 chat 模板不支持 tools，本轮退化为纯文本输出")
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                return self._llm.create_chat_completion(**kwargs)
            raise

    @staticmethod
    def _apply_no_think(messages: list[dict]) -> list[dict]:
        """在最后一条 user 消息末尾追加 /no_think。

        - 已经在末尾的不重复添加
        - 没有 user 消息时不动（例如只有 system 的一次性调用）
        - 不改原列表，避免同一份 messages 被多次调用时叠加出一串 /no_think
        """
        if not messages:
            return messages
        out = [dict(m) for m in messages]
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") != "user":
                continue
            content = out[i].get("content") or ""
            if "/no_think" in content:
                return out
            out[i]["content"] = content.rstrip() + "\n/no_think"
            return out
        return out

    def stream_complete(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
        tools: Optional[list[dict]] = None,
    ):
        """
        流式推理 — 返回生成器，每个 token 实时产出

        这是恒常意识流的核心接口。
        LLM 不断产生 token，调用方可以：
        - 实时读取每个 token
        - 根据中间结果决定是否提前终止
        - 在 token 间注入新的上下文（未实现，需要底层支持）
        """
        self._ensure_loaded()
        kwargs: dict = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            logit_bias=logit_bias,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        stream = self._create_chat_completion(**kwargs)
        accumulated = ""
        last_stripped = ""
        # 是否已经向调用方发出过任何片段。用于流结束时判断要不要补发终止片段
        # （模型整轮只输出 <think> 时，中间所有 yield 都会被 new_part 为空跳过）。
        yielded_any = False
        # 流式 tool_calls 是分片到达的：按 index 累积 name 与 arguments 片段
        pending_calls: dict[int, dict] = {}
        for chunk in stream:
            if "choices" not in chunk or not chunk["choices"]:
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}) or {}

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = pending_calls.setdefault(
                    idx, {"id": tc.get("id", f"call_{idx}"), "type": "function",
                          "function": {"name": "", "arguments": ""}}
                )
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

            content = delta.get("content", "")
            if not content:
                continue
            accumulated += content
            # 增量剥离 <think> 与 <tool_call> 块：基于完整累加文本计算，
            # 避免流式中途的半截标签被当作正式内容发出
            # （曾导致 speak text 变成 "<think>" 或 "<tool_call>{"name":…"）。
            stripped = _strip_tool_call_partial(_strip_thinking(accumulated))
            new_part = stripped[len(last_stripped):]
            last_stripped = stripped
            if not new_part:
                continue
            finish_reason = choice.get("finish_reason")
            yielded_any = True
            yield StreamThought(
                token_text=new_part,
                token_id=-1,  # llama-cpp-python stream 不提供 token_id
                accumulated_text=stripped,
                is_finished=finish_reason is not None,
                finish_reason=finish_reason,
            )

        # 流结束：把累积的工具调用随最后一个片段带出。
        # 两种来源都要收：原生的分片 tool_calls，以及写在 content 里的
        # <tool_call> 标签（Qwen 部分模板的行为）。
        if not pending_calls and "<tool_call>" in accumulated:
            text_calls, _ = _extract_tool_calls(accumulated)
            for i, c in enumerate(text_calls):
                pending_calls[i] = c

        if pending_calls:
            final_text = _extract_tool_calls(_strip_thinking(accumulated))[1]
            yield StreamThought(
                token_text="",
                token_id=-1,
                accumulated_text=final_text,
                is_finished=True,
                finish_reason="tool_calls",
                tool_calls=[pending_calls[i] for i in sorted(pending_calls)],
            )
            return

        # 兜底：流已经结束，但一个片段都没发出去过。
        #
        # 触发条件很常见——模型这一轮整段输出都在 <think> 里，或者只输出了
        # 被剥掉的控制标签。上面每个分支都靠 new_part 非空才 yield，
        # 这种情况下生成器静默返回空迭代，调用方拿到的 accumulated 是初始
        # 值 ""，于是"模型明明跑了 6 秒"和"什么都没生成"在外观上完全一致
        # （实测日志表现为 out=0ch）。
        #
        # 这里补发一个终止片段，把真实的生成结果交出去：raw 是未剥离的
        # 原始累加，上层据此可以判断"确实有输出但都是内部思考"。
        if not yielded_any:
            yield StreamThought(
                token_text="",
                token_id=-1,
                accumulated_text=_strip_tool_call_partial(_strip_thinking(accumulated)),
                is_finished=True,
                finish_reason="stop",
                raw_text=accumulated,
            )

    def stream_think(
        self,
        system_prompt: str,
        context_memories: list[str],
        sensor_context: str,
        max_tokens: int = 100,
        logit_bias: Optional[dict[int, float]] = None,
        conversation_context: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        tool_results: Optional[list[str]] = None,
    ):
        """
        恒常意识流的核心方法: 产生一个"思维"

        与 stream_complete 的区别：
        - 输入不是对话，而是系统 prompt + 上下文
        - 输出是"内部思维"而非"对话回复"
        - 使用较高的 temperature 鼓励多样性

        conversation_context: 最近的人机对话历史(user/assistant)，
          让自发思维也能"记得刚聊了什么"，否则内心独白与对话割裂。
        tools: OpenAI 格式工具清单；传入后模型可请求调用执行器。
        tool_results: 最近几次工具执行结果。必须回注，否则模型不知道
          自己刚刚已经搜过/已经开过灯，会原地反复请求同一个工具。
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # 构建上下文
        context_parts = []
        if context_memories:
            context_parts.append("## 最近的记忆:\n" + "\n".join(f"- {m}" for m in context_memories))
        if conversation_context:
            conv_lines = "\n".join(
                f"- {('对方' if m['role'] == 'user' else '我')}说：{m['content']}"
                for m in conversation_context
            )
            context_parts.append("## 最近和对方的对话:\n" + conv_lines)
        if sensor_context:
            context_parts.append("## 身体感觉:\n" + sensor_context)
        if tool_results:
            # 回注工具结果：模型据此判断"这件事已经做过了"，
            # 不再无谓地重复请求同一个工具（曾出现同一搜索连发四次）。
            context_parts.append(
                "## 你刚才做过的事:\n" + "\n".join(tool_results)
            )
        
        if context_parts:
            messages.append({
                "role": "user",
                "content": "\n\n".join(context_parts) + "\n\n按上面要求的格式产生一个简短的内部想法（先 <intent> 标签，再写想法）："
            })
        else:
            messages.append({
                "role": "user",
                "content": "按上面要求的格式产生一个简短的内部想法（先 <intent> 标签，再写想法）："
            })

        return self.stream_complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.config.temperature * 1.2,  # 思维阶段温度略高，鼓励多样性
            logit_bias=logit_bias,
            tools=tools,
        )

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()


def create_llm() -> "LocalLLM":
    """
    根据环境变量创建 LLM 后端。

    设置 ANIMAT_LLM_BACKEND=cloud 使用 OpenAI 兼容云端 API（DeepSeek / 百炼等）；
    默认或 local 使用本地 GGUF 模型。

    Cloud 模式所需环境变量:
      OPENAI_API_KEY    - API 密钥
      OPENAI_BASE_URL   - API 地址（如 https://api.deepseek.com/v1）
      ANIMAT_LLM_MODEL  - 模型名（如 deepseek-chat）
    """
    backend = os.getenv("ANIMAT_LLM_BACKEND", "local").lower()
    if backend == "cloud":
        try:
            from animat.cloud_llm import CloudLLM, CloudLLMConfig
        except ImportError:
            raise ImportError(
                "使用云端后端需要安装: pip install openai>=1.0.0 tiktoken"
            )
        logger.info("Using Cloud LLM backend (OpenAI-compatible API)")
        return CloudLLM(CloudLLMConfig())
    logger.info("Using Local LLM backend (llama-cpp-python)")
    return LocalLLM()
