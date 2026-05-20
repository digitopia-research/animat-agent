"""
MQTT 神经总线 — Animat Agent 的"神经系统"

设计理念:
  - 模拟生物神经系统的信息传递机制
  - 每个模组既是发布者(传出神经)又是订阅者(传入神经)
  - Topic 即"神经通路"，Payload 即"神经脉冲"
  - QoS1 保证关键信号(如痛觉、恐惧)不丢失
"""

import os
import json
import logging
import asyncio
from typing import Any, Callable, Optional
import aiomqtt

logger = logging.getLogger("animat.nervous")


class NervousSystem:
    """MQTT 神经总线 — 全局共享的消息中枢"""

    def __init__(
        self,
        client_id: str = "animat-brain",
        host: str = os.getenv("MQTT_BROKER_HOST", "localhost"),
        port: int = int(os.getenv("MQTT_BROKER_PORT", "1883")),
        alive: int = int(os.getenv("MQTT_KEEPALIVE", "60")),
    ):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.alive = alive
        self._client: Optional[aiomqtt.Client] = None
        self._subscribers: dict[str, list[Callable]] = {}
        self._running = False

    async def start(self, timeout: float = 10.0):
        """启动神经总线连接"""
        self._client = aiomqtt.Client(
            hostname=self.host,
            port=self.port,
            identifier=self.client_id,
            keepalive=self.alive,
        )
        try:
            await asyncio.wait_for(self._client.__aenter__(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"MQTT broker connection timed out ({timeout}s): "
                f"{self.host}:{self.port} — is EMQX running?"
            )
        except ConnectionRefusedError:
            raise ConnectionError(
                f"MQTT broker refused connection: "
                f"{self.host}:{self.port} — is EMQX running?"
            )
        except Exception as e:
            raise ConnectionError(
                f"MQTT broker connection failed: {self.host}:{self.port} — {e}"
            )
        self._running = True
        logger.info(f"Nervous system connected: {self.host}:{self.port}")
        asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        """关闭神经总线"""
        self._running = False
        if self._client:
            await self._client.__aexit__(None, None, None)
        logger.info("Nervous system disconnected")

    # ── 神经脉冲发射（发布） ──
    async def fire(self, topic: str, payload: Any, qos: int = 1):
        """
        发射神经脉冲
        - topic: 神经通路
        - payload: 脉冲内容 (自动序列化为JSON)
        - qos: 1=至少一次(关键信号), 0=至多一次(高频遥测)
        """
        if not self._client:
            logger.warning("Nervous system not connected, pulse lost")
            return
        data = json.dumps(payload, ensure_ascii=False, default=str)
        await self._client.publish(topic, data.encode("utf-8"), qos=qos)
        logger.debug(f"Fire [{topic}]: {data[:100]}")

    # ── 神经末梢注册（订阅） ──
    async def listen(self, topic: str, handler: Callable, qos: int = 1):
        """
        注册神经末梢(回调)
        - handler: async def handler(topic, payload) -> None
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
            await self._client.subscribe(topic, qos=qos)
        self._subscribers[topic].append(handler)
        logger.info(f"Neuron registered: {topic} -> {handler.__name__}")

    # ── 内部分发循环 ──
    async def _dispatch_loop(self):
        """持续接收脉冲并分发给对应的神经末梢"""
        async for message in self._client.messages:
            topic = str(message.topic)
            try:
                payload = json.loads(message.payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = message.payload.decode("utf-8", errors="replace")

            handlers = self._subscribers.get(topic, [])
            for handler in handlers:
                try:
                    await handler(topic, payload)
                except Exception as e:
                    logger.error(f"Handler error [{topic}]: {e}")

    # ── 便捷: 一次性发布+忘记 ──
    async def reflex_fire(self, topic: str, payload: Any):
        """反射弧专用: 最高优先级、最低延迟的脉冲发射"""
        await self.fire(topic, payload, qos=1)
