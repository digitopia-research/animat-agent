"""
云端 LLM 后端 — 基于 OpenAI 兼容 API (DeepSeek / 阿里云百炼 / OpenAI 等)

设计目标:
  - 接口与 LocalLLM 完全对齐，可无缝替换
  - 支持 logit_bias（Phase 2 注意力调制器核心，前提是云端 API 开放该参数）
  - 提供 tokenize 方法，供 AttentionModulator 把种子词切成 token_id
    （云端无本地 llama 对象，需用 tiktoken 或对应模型的 tokenizer）

注意:
  - logit_bias 的 token_id 必须与目标模型 tokenizer 一致。
    若云端 API 屏蔽了 logit_bias，则 Phase 2 失效，需退回 v0.1 的 prompt 注入。
  - tokenize 默认用 tiktoken (cl100k_base)，与 GPT 系列一致；DeepSeek 也用同套 BPE。
    对于非 GPT 系模型（如 Qwen），可在 CloudLLMConfig 中指定 tokenizer 库。
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

from animat.llm_backend import StreamThought

logger = logging.getLogger("animat.cloud_llm")


@dataclass
class CloudLLMConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("ANIMAT_LLM_MODEL", "gpt-4o-mini")
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: float = 60.0
    # tokenize 方案: "tiktoken" (默认, GPT/DeepSeek 系) 或 "auto"
    # 若模型非 GPT 系且需要精确 token_id，可后续扩展为本地 tokenizer 库
    tokenizer: str = "tiktoken"
    tokenizer_encoding: str = os.getenv("ANIMAT_TOKENIZER_ENCODING", "cl100k_base")


def _require_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        raise ImportError(
            "使用 CloudLLM 需要安装 openai: pip install openai>=1.0.0"
        )


def _require_tiktoken():
    try:
        import tiktoken
        return tiktoken
    except ImportError:
        raise ImportError(
            "CloudLLM 的 tokenize 需要 tiktoken（用于把种子词切成 token_id 以应用 logit_bias）。"
            "请安装: pip install tiktoken  或  uv add tiktoken"
        )


class CloudLLM:
    """
    云端 LLM 后端封装（OpenAI 兼容）

    与 LocalLLM 的差异:
      - 推理走 HTTP（openai 客户端），非本地加载
      - 无 KV cache 概念（由 API 服务端管理）
      - tokenize 通过 tiktoken（或模型 tokenizer）离线完成
      - logit_bias 若 API 不支持则静默忽略（降级为无偏置）
    """

    def __init__(self, config: Optional[CloudLLMConfig] = None):
        self.config = config or CloudLLMConfig()
        self._client = None
        self._loaded = False
        self._tokenizer = None  # tiktoken Encoding 实例

    def load(self):
        """初始化 openai 客户端与 tokenizer（轻量，无模型下载）"""
        if self._loaded:
            return
        OpenAI = _require_openai()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=3,
        )
        # 初始化 tokenizer
        if self.config.tokenizer == "tiktoken":
            tk = _require_tiktoken()
            try:
                self._tokenizer = tk.get_encoding(self.config.tokenizer_encoding)
            except Exception:
                self._tokenizer = tk.get_encoding("cl100k_base")
        self._loaded = True
        logger.info(
            f"Cloud LLM ready: model={self.config.model} base_url={self.config.base_url}"
        )

    def unload(self):
        self._client = None
        self._tokenizer = None
        self._loaded = False

    @property
    def ready(self) -> bool:
        return self._loaded and self._client is not None

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        """
        把文本切成 token_id，供 AttentionModulator 使用。

        llama-cpp-python 的 tokenize 接受 bytes；这里兼容该签名。
        tiktoken 接受 str，故做 bytes→str 转换。
        """
        if self._tokenizer is None:
            raise RuntimeError("CloudLLM tokenizer 未初始化，请先调用 load()")
        text_str = text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else str(text)
        # tiktoken 不区分 BOS；add_bos 仅作兼容占位
        return self._tokenizer.encode_ordinary(text_str)

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
    ) -> dict:
        """
        非流式推理（OpenAI Chat Completions 格式）

        Returns: 与 LocalLLM.complete 相同结构的 dict
        """
        self._ensure_loaded()
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
        )
        if logit_bias:
            kwargs["logit_bias"] = {str(k): v for k, v in logit_bias.items()}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # 若 API 不支持 logit_bias，降级重试一次（去掉 logit_bias）
            if logit_bias and ("logit_bias" in str(e).lower() or "400" in str(e)):
                logger.warning("云端 API 不支持 logit_bias，已降级为无偏置采样")
                kwargs.pop("logit_bias", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        # 转换为与 LocalLLM 兼容的字典结构
        return {
            "choices": [{
                "message": {"role": "assistant", "content": resp.choices[0].message.content},
                "finish_reason": resp.choices[0].finish_reason,
            }],
            "usage": resp.usage.model_dump() if resp.usage else None,
        }

    def stream_complete(
        self,
        messages: list[dict],
        max_tokens: int = 200,
        temperature: Optional[float] = None,
        logit_bias: Optional[dict[int, float]] = None,
    ):
        """
        流式推理 — 兼容 LocalLLM 的 StreamThought 产出
        """
        self._ensure_loaded()
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            stream=True,
        )
        if logit_bias:
            kwargs["logit_bias"] = {str(k): v for k, v in logit_bias.items()}
        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if logit_bias and ("logit_bias" in str(e).lower() or "400" in str(e)):
                logger.warning("云端 API 不支持 logit_bias，已降级为无偏置采样")
                kwargs.pop("logit_bias", None)
                stream = self._client.chat.completions.create(**kwargs)
            else:
                raise

        accumulated = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            if not content:
                continue
            accumulated += content
            finish_reason = chunk.choices[0].finish_reason
            yield StreamThought(
                token_text=content,
                token_id=-1,
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
    ):
        """
        恒常意识流核心方法: 产生一个"思维"（与 LocalLLM 同签名）
        """
        messages = [{"role": "system", "content": system_prompt}]

        context_parts = []
        if context_memories:
            context_parts.append("## 最近的记忆:\n" + "\n".join(f"- {m}" for m in context_memories))
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
            temperature=self.config.temperature * 1.2,
            logit_bias=logit_bias,
        )

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()
