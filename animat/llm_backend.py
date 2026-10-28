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
import shutil
import logging
from typing import Optional, AsyncIterator
from dataclasses import dataclass, field

logger = logging.getLogger("animat.llm_backend")


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
    ) -> dict:
        """
        非流式推理（兼容 OpenAI 返回格式）
        
        Args:
            messages: chat messages
            max_tokens: 最大生成 token 数
            temperature: 温度（None=使用默认）
            logit_bias: token_id → bias_value 映射（Phase 2 注意力调制器用）
        """
        self._ensure_loaded()
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            logit_bias=logit_bias,
        )
        return response

    def stream_complete(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
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
        stream = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            logit_bias=logit_bias,
            stream=True,
        )
        accumulated = ""
        for chunk in stream:
            if "choices" not in chunk or not chunk["choices"]:
                continue
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if not content:
                continue
            accumulated += content
            finish_reason = chunk["choices"][0].get("finish_reason")
            yield StreamThought(
                token_text=content,
                token_id=-1,  # llama-cpp-python stream 不提供 token_id
                accumulated_text=accumulated,
                is_finished=finish_reason is not None,
                finish_reason=finish_reason,
            )

    def stream_think(
        self,
        system_prompt: str,
        context_memories: list[str],
        sensor_context: str,
        max_tokens: int = 100,
        logit_bias: Optional[dict[int, float]] = None,
        conversation_context: Optional[list[dict]] = None,
    ):
        """
        恒常意识流的核心方法: 产生一个"思维"
        
        与 stream_complete 的区别：
        - 输入不是对话，而是系统 prompt + 上下文
        - 输出是"内部思维"而非"对话回复"
        - 使用较高的 temperature 鼓励多样性

        conversation_context: 最近的人机对话历史(user/assistant)，
          让自发思维也能"记得刚聊了什么"，否则内心独白与对话割裂。
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
        
        if context_parts:
            messages.append({
                "role": "user",
                "content": "\n\n".join(context_parts) + "\n\n产生一个简短的内部想法："
            })
        else:
            messages.append({
                "role": "user",
                "content": "产生一个简短的内部想法："
            })

        return self.stream_complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.config.temperature * 1.2,  # 思维阶段温度略高，鼓励多样性
            logit_bias=logit_bias,
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
