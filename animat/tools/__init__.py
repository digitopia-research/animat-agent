"""LLM 可调用工具集。

所有执行器类工具都通过 MQTT 话题发布，执行器订阅后执行
—— 模拟器与真实硬件可互换，LLM 侧不需要任何改动。
"""

from __future__ import annotations

import logging
import os

from animat.tools.registry import ToolRegistry, ToolResult, registry
from animat.tools.spec import ParamSpec, ParamType, ToolSpec

logger = logging.getLogger("animat.tools")

_initialized = False


def init_tools() -> ToolRegistry:
    """注册全部工具（幂等）。"""
    global _initialized
    if _initialized:
        return registry

    from animat.tools.effector import register_effector_tools
    from animat.tools.knowledge import register_knowledge_tools

    register_effector_tools()
    register_knowledge_tools()

    _initialized = True
    logger.info("工具初始化完成，共 %d 个: %s", len(registry.names()), ", ".join(registry.names()))
    return registry


def tools_enabled() -> bool:
    """工具调用总开关，可用 ANIMAT_TOOLS=0 关闭。"""
    return os.getenv("ANIMAT_TOOLS", "1") not in ("0", "false", "False")


__all__ = [
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ParamSpec",
    "ParamType",
    "registry",
    "init_tools",
    "tools_enabled",
]
