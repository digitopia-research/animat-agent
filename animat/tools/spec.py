"""工具规格定义。

一个工具 = 名字 + 说明 + 参数规格 + 执行方式。
执行方式有两种:
  - topic:  发布到 MQTT 话题，由执行器订阅（LED/电机/表情等）
  - handler: 直接回调（联网搜索等非执行器能力）

参数规格同时用于两件事:
  1. 生成给 LLM 的 JSON Schema（不手写第二份，避免文档漂移）
  2. 执行前校验/裁剪模型给的参数（防止越界、枚举幻觉）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("animat.tools.spec")


class ParamType:
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"


@dataclass
class ParamSpec:
    """单个参数的规格。"""

    type: str
    description: str = ""
    enum: Optional[list[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Any = None
    required: bool = True

    def to_json_schema(self) -> dict:
        """转成 JSON Schema 片段（OpenAI tools 格式）。"""
        schema: dict[str, Any] = {"description": self.description}
        if self.type == ParamType.ENUM:
            schema["type"] = "string"
            schema["enum"] = list(self.enum or [])
        elif self.type == ParamType.INTEGER:
            schema["type"] = "integer"
        elif self.type == ParamType.NUMBER:
            schema["type"] = "number"
        elif self.type == ParamType.BOOLEAN:
            schema["type"] = "boolean"
        else:
            schema["type"] = "string"

        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.default is not None:
            schema["default"] = self.default
        return schema

    def coerce(self, raw: Any) -> tuple[Any, Optional[str]]:
        """把模型给的值规整到合法范围。

        返回 (值, 错误说明)。错误说明为 None 表示通过。
        枚举值不在白名单时返回错误（拒绝执行），
        数值越界则裁剪（比整体拒绝更友好，避免模型一个小错就完全失效）。
        """
        if raw is None:
            if self.required and self.default is None:
                return None, "缺少必填参数"
            return self.default, None

        if self.type == ParamType.ENUM:
            value = str(raw).strip()
            if self.enum and value not in self.enum:
                return None, f"取值 {value!r} 不在允许列表 {self.enum}"
            return value, None

        if self.type == ParamType.INTEGER:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                return None, f"无法转为整数: {raw!r}"
            if self.minimum is not None and value < self.minimum:
                value = int(self.minimum)
            if self.maximum is not None and value > self.maximum:
                value = int(self.maximum)
            return value, None

        if self.type == ParamType.NUMBER:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None, f"无法转为数字: {raw!r}"
            if self.minimum is not None and value < self.minimum:
                value = float(self.minimum)
            if self.maximum is not None and value > self.maximum:
                value = float(self.maximum)
            return value, None

        if self.type == ParamType.BOOLEAN:
            if isinstance(raw, bool):
                return raw, None
            return str(raw).strip().lower() in ("1", "true", "yes", "on"), None

        return str(raw), None


@dataclass
class ToolSpec:
    """一个可被 LLM 调用的工具。"""

    name: str
    description: str
    params: dict[str, ParamSpec] = field(default_factory=dict)

    # 执行方式一：发布到 MQTT 话题（执行器订阅后执行）
    topic: Optional[str] = None
    # 该话题 payload 里表示"动作名"的字段，例如 {"command": "led", ...}
    command: Optional[str] = None
    # 执行方式二：直接回调（topic 为空时使用）
    handler: Optional[Callable[..., Any]] = None

    # 频控：同一工具两次调用之间的最小间隔（秒）
    cooldown_sec: float = 3.0

    def to_openai_schema(self) -> dict:
        """转成 OpenAI tools 格式，直接喂给 llama-cpp-python。"""
        properties = {k: v.to_json_schema() for k, v in self.params.items()}
        required = [k for k, v in self.params.items() if v.required and v.default is None]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_args(self, raw_args: dict) -> tuple[Optional[dict], Optional[str]]:
        """校验并规整参数。返回 (clean_args, error)。"""
        if not isinstance(raw_args, dict):
            return None, "参数必须是对象"

        clean: dict[str, Any] = {}
        # 未知参数直接丢弃（模型常幻觉出多余字段），但记一笔
        unknown = set(raw_args) - set(self.params)
        if unknown:
            logger.debug("工具 %s 丢弃未知参数: %s", self.name, sorted(unknown))

        for name, spec in self.params.items():
            value, err = spec.coerce(raw_args.get(name))
            if err:
                return None, f"参数 {name}: {err}"
            if value is not None:
                clean[name] = value
        return clean, None
