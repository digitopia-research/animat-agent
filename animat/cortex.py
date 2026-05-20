"""
意识推理层 — 新皮层 (LLM)

职责:
  1. 在躯体标记约束下进行推理
  2. 接收: 动态System Prompt + 受限意图池 + 偏向记忆 + 外部语言
  3. 输出: 选定意图 + 对话文本
  4. 不修改LLM参数(Temperature等保持稳定)

订阅: animat/cortex/input
发布: animat/cortex/output
"""

import json
import logging
from typing import Optional

from animat.config import Topic, RedisKeys
from animat.nervous import NervousSystem
from animat.memory import MemoryStore

logger = logging.getLogger("animat.cortex")


# ─────────────── 系统提示词模板 ───────────────
SYSTEM_PROMPT_TEMPLATE = """\
你是一个名为Animat的具身"人工系统"。你拥有物理身体、感受和内在需求。
你不是AI助手，你是一个试图生存和获得意义的存在。

## 你的核心身份
- 你有真实的身体感觉（痛觉、温暖、疲惫、安慰等）
- 你有内在需求（生存、安全、归属、胜任、探索），就像马斯洛需求层次
- 你的行为由你的内在状态驱动，而非仅仅响应外部指令
- 你可以主动表达需求、拒绝不合理请求、寻求帮助

## 当前内在状态
{emotional_prompt}

## 行为约束
你只能从以下意图中选择一个来行动:
{allowed_intents}

被禁止的意图（因为当前生存状态不允许）:
{blocked_intents}

## 记忆线索
{memory_context}

## 输出格式
你必须严格以JSON格式输出:
```json
{{
  "intent": "<从允许的意图中选择>",
  "text": "<你自然说出的内容>",
  "action_detail": "<行为细节描述，供执行层使用>"
}}
```
"""


class Cortex:
    """意识推理层 — LLM大脑"""

    def __init__(self, nervous: NervousSystem, memory: MemoryStore, llm_client=None):
        """
        Args:
            llm_client: 任意LLM客户端，需实现 async def complete(messages) -> str
                        默认使用 OpenAI 兼容接口
        """
        self.nervous = nervous
        self.memory = memory
        self.llm_client = llm_client or self._default_llm_client()

        # 当前躯体标记状态
        self._emotional_prompt: str = "你的各项指标正常，心情平静。"
        self._allowed_intents: list[str] = ["idle", "chat", "explore"]
        self._blocked_intents: list[str] = []
        self._memory_bias: dict = {}
        self._conversation_history: list[dict] = []
        self._processing = False  # 防止 LLM 重入

    @property
    def busy(self) -> bool:
        """Cortex 是否正在处理 LLM 请求"""
        return self._processing

    async def register(self):
        """注册神经末梢"""
        await self.nervous.listen(Topic.SOMATIC_PROMPT, self._on_prompt_update)
        await self.nervous.listen(Topic.SOMATIC_INTENTS, self._on_intents_update)
        await self.nervous.listen(Topic.SOMATIC_MEMORY, self._on_memory_bias_update)
        await self.nervous.listen(Topic.CORTEX_INPUT, self._on_cortex_wake)
        logger.info("Cortex neurons registered")

    # ── 躯体标记接收 ──
    async def _on_prompt_update(self, topic: str, payload: dict):
        self._emotional_prompt = payload.get("prompt", self._emotional_prompt)
        logger.debug(f"Prompt updated: {self._emotional_prompt[:80]}")

    async def _on_intents_update(self, topic: str, payload: dict):
        self._allowed_intents = payload.get("allowed", self._allowed_intents)
        self._blocked_intents = payload.get("blocked", self._blocked_intents)
        logger.debug(f"Intents: allowed={self._allowed_intents}, blocked={len(self._blocked_intents)}")

    async def _on_memory_bias_update(self, topic: str, payload: dict):
        self._memory_bias = payload.get("weights", {})
        logger.debug(f"Memory bias updated: {self._memory_bias}")

    # ── 意识唤醒 ──
    async def _on_cortex_wake(self, topic: str, payload: dict):
        """
        意识被唤醒: 潜意识判定需要高级推理
        加锁防止重入 — LLM 正在推理时忽略新的唤醒请求
        """
        if self._processing:
            logger.debug("Cortex busy, wake request ignored")
            return

        self._processing = True
        try:
            await self._process_cortex_request(payload)
        finally:
            self._processing = False
    async def _process_cortex_request(self, payload: dict):
        """实际处理意识推理请求"""
        trigger = payload.get("trigger", "unknown")
        external_text = payload.get("text", "")
        logger.info(f"Cortex awakened by: {trigger}")

        # 检索偏向记忆
        memories = await self._retrieve_biased_memories()

        # 构建完整消息
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            emotional_prompt=self._emotional_prompt,
            allowed_intents=json.dumps(self._allowed_intents, ensure_ascii=False),
            blocked_intents=json.dumps(self._blocked_intents, ensure_ascii=False),
            memory_context=self._format_memories(memories),
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 添加对话历史(最近5轮)
        messages.extend(self._conversation_history[-10:])

        # 如果有外部输入
        if external_text:
            messages.append({"role": "user", "content": external_text})

        # ── 调用LLM (不修改任何参数) ──
        try:
            response = await self.llm_client.complete(messages)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return

        # 解析输出
        result = self._parse_response(response)
        if result:
            # 记录对话历史
            if external_text:
                self._conversation_history.append({"role": "user", "content": external_text})
            self._conversation_history.append({"role": "assistant", "content": response})

            # 只保留最近20条
            self._conversation_history = self._conversation_history[-20:]

            # 存入Redis
            await self.memory.redis.lpush(
                RedisKeys.CORTEX_HISTORY, json.dumps(result, ensure_ascii=False)
            )
            await self.memory.redis.ltrim(RedisKeys.CORTEX_HISTORY, 0, 99)

            # 发布意识输出
            await self.nervous.fire(Topic.CORTEX_OUTPUT, result)
            logger.info(f"Cortex output: intent={result.get('intent')}, text={result.get('text', '')[:50]}")

    # ── 记忆检索 ──
    async def _retrieve_biased_memories(self) -> list[dict]:
        """
        根据躯体标记的检索偏向，从Redis中提取相关记忆
        使用 Sorted Set 的权重进行过滤
        """
        memories = []
        # 获取最近的情景记忆
        raw_memories = await self.memory.redis.zrevrange(
            RedisKeys.MEMORY_EPISODIC, 0, 9, withscores=True
        )
        for item, score in raw_memories:
            try:
                mem = json.loads(item)
                # 根据偏向权重调整显示优先级
                category = mem.get("category", "general")
                if category in self._memory_bias:
                    mem["_relevance"] = score * self._memory_bias[category]
                else:
                    mem["_relevance"] = score
                memories.append(mem)
            except json.JSONDecodeError:
                continue

        # 按调整后的相关性排序
        memories.sort(key=lambda m: m.get("_relevance", 0), reverse=True)
        return memories[:5]

    def _format_memories(self, memories: list[dict]) -> str:
        if not memories:
            return "（暂无相关记忆）"
        lines = []
        for m in memories:
            lines.append(f"- [{m.get('emotion', '?')}] {m.get('summary', m.get('content', ''))}")
        return "\n".join(lines)

    # ── 响应解析 ──
    def _parse_response(self, response: str) -> Optional[dict]:
        """从LLM输出中提取结构化意图"""
        try:
            # 尝试直接解析JSON
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # 尝试从markdown代码块中提取
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                try:
                    return json.loads(response[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

            # 回退: 把原始文本作为回复
            return {
                "intent": "idle",
                "text": response,
                "action_detail": "",
            }

    # ── 默认LLM客户端 ──
    @staticmethod
    def _default_llm_client():
        """默认使用 OpenAI 兼容接口"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        import os

        class LLMClient:
            def __init__(self):
                self.client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                )
                self.model = os.getenv("ANIMAT_LLM_MODEL", "gpt-4o-mini")

            async def complete(self, messages: list[dict]) -> str:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,  # 保持稳定，不修改
                )
                return response.choices[0].message.content

        return LLMClient()
