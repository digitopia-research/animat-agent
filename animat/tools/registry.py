"""工具注册表 — 白名单、校验、频控、分发。

设计要点:
  - 只执行注册过的工具（不认识的名字直接丢弃，防止模型幻觉调用）
  - 参数先按 ParamSpec 校验/裁剪，再执行
  - 同一工具在 cooldown_sec 内不重复执行
  - 执行结果统一返回 ToolResult，供上游记录或回注上下文
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from animat.tools.spec import ToolSpec

logger = logging.getLogger("animat.tools.registry")


@dataclass
class ToolResult:
    """一次工具执行的结果。"""

    name: str
    ok: bool
    detail: str = ""
    data: Any = None
    skipped: Optional[str] = None  # 非错误性跳过原因（冷却/未启用）

    def to_context_line(self) -> str:
        """生成可回注给 LLM 的一行描述。"""
        if self.skipped:
            return f"[工具 {self.name} 未执行: {self.skipped}]"
        if not self.ok:
            return f"[工具 {self.name} 失败: {self.detail}]"
        if self.data:
            return f"[工具 {self.name} 结果]\n{self.data}"
        return f"[工具 {self.name} 已执行: {self.detail}]"


class ToolRegistry:
    """工具注册与执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._last_call: dict[str, float] = {}

    # ── 注册 ──
    def register(self, spec: ToolSpec) -> None:
        if spec.topic is None and spec.handler is None:
            raise ValueError(f"工具 {spec.name} 必须指定 topic 或 handler")
        self._tools[spec.name] = spec
        logger.info("工具已注册: %s -> %s", spec.name, spec.topic or "handler")

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    # ── 给 LLM 的 schema ──
    def openai_tools(self, enabled: Optional[list[str]] = None) -> list[dict]:
        specs = [t for n, t in self._tools.items() if enabled is None or n in enabled]
        return [s.to_openai_schema() for s in specs]

    # ── 执行 ──
    async def execute(
        self,
        name: str,
        raw_args: dict,
        nervous=None,
    ) -> ToolResult:
        """执行一次工具调用。nervous 用于 MQTT 发布。"""
        spec = self._tools.get(name)
        if spec is None:
            logger.warning("模型调用了未注册的工具: %s（已忽略）", name)
            return ToolResult(name=name, ok=False, detail="未注册的工具")

        # 频控
        now = time.time()
        last = self._last_call.get(name, 0.0)
        if now - last < spec.cooldown_sec:
            remain = spec.cooldown_sec - (now - last)
            return ToolResult(
                name=name, ok=True, skipped=f"冷却中，还需 {remain:.1f}s"
            )

        args, err = spec.validate_args(raw_args or {})
        if err is not None:
            logger.warning("工具 %s 参数非法: %s", name, err)
            return ToolResult(name=name, ok=False, detail=err)

        try:
            if spec.topic is not None:
                if nervous is None:
                    return ToolResult(name=name, ok=False, detail="MQTT 总线不可用")
                payload: dict[str, Any] = dict(args)
                if spec.command:
                    payload = {"command": spec.command, **payload}
                else:
                    # 没有 command 的工具，用工具名标识这次投递。
                    # 订阅方（action.py）按 intent 字段识别，
                    # 缺了它整条消息会被当成未知意图丢掉。
                    payload = {"intent": spec.name, "command": spec.name, **payload}
                # 异步 fire，MQTT 拥堵不应阻塞思维循环
                await nervous.fire(spec.topic, payload)
                self._last_call[name] = now
                logger.info("工具 %s 已发布 -> %s %s", name, spec.topic, args)
                return ToolResult(
                    name=name, ok=True, detail=f"已发送 {json.dumps(args, ensure_ascii=False)}"
                )

            # handler 形式（可能是同步函数，也可能是协程）
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(spec.handler):
                data = await spec.handler(**args)
            else:
                data = await asyncio.to_thread(spec.handler, **args)

            self._last_call[name] = now
            logger.info("工具 %s 执行完成: %s", name, args)
            return ToolResult(name=name, ok=True, detail="执行完成", data=data)

        except Exception as e:  # 工具失败不能拖垮思维循环
            logger.error("工具 %s 执行异常: %s", name, e)
            return ToolResult(name=name, ok=False, detail=str(e))

    def reset_cooldowns(self) -> None:
        self._last_call.clear()


# 全局单例
registry = ToolRegistry()
