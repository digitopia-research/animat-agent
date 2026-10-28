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
from enum import Enum
from typing import Any, Callable, Optional
import aiomqtt

logger = logging.getLogger("animat.nervous")

# ── 重连配置 ──
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY  = 30.0
RECONNECT_BACKOFF    = 2.0
HEALTH_CHECK_INTERVAL = 5.0


class NervousHealth(str, Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    DEAD      = "dead"


class NervousSystem:
    """MQTT 神经总线 — 全局共享的消息中枢 (带自动重连)"""

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
        self._health = NervousHealth.DEAD
        self._reconnecting = False
        self._max_consecutive_failures = 5

    @property
    def healthy(self) -> bool:
        return self._health != NervousHealth.DEAD

    async def start(self, timeout: float = 10.0):
        """启动神经总线连接"""
        await self._do_connect(timeout)
        self._running = True
        # 后台健康检查
        asyncio.create_task(self._health_check_loop())

    async def _do_connect(self, timeout: float = 10.0):
        self._client = aiomqtt.Client(
            hostname=self.host,
            port=self.port,
            identifier=self.client_id,
            keepalive=self.alive,
        )
        try:
            await asyncio.wait_for(self._client.__aenter__(), timeout=timeout)
        except asyncio.TimeoutError:
            self._health = NervousHealth.DEAD
            raise ConnectionError(
                f"MQTT broker connection timed out ({timeout}s): "
                f"{self.host}:{self.port} — is EMQX running?"
            )
        except ConnectionRefusedError:
            self._health = NervousHealth.DEAD
            raise ConnectionError(
                f"MQTT broker refused connection: "
                f"{self.host}:{self.port} — is EMQX running?"
            )
        except Exception as e:
            self._health = NervousHealth.DEAD
            raise ConnectionError(
                f"MQTT broker connection failed: {self.host}:{self.port} — {e}"
            )
        self._health = NervousHealth.HEALTHY
        logger.info(f"Nervous system connected: {self.host}:{self.port}")
        asyncio.create_task(self._dispatch_loop())
        # 重订阅所有已注册的 handler
        await self._resubscribe_all()

    async def _reconnect_loop(self):
        if self._reconnecting:
            return
        self._reconnecting = True
        delay = RECONNECT_BASE_DELAY
        while self._health == NervousHealth.DEAD:
            try:
                logger.info(f"Attempting MQTT reconnect (delay={delay:.0f}s)...")
                await self._do_connect()
                logger.info("MQTT reconnected successfully")
                break
            except Exception as e:
                logger.warning(f"MQTT reconnect failed: {e}, next attempt in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * RECONNECT_BACKOFF, RECONNECT_MAX_DELAY)
        self._reconnecting = False

    async def _health_check_loop(self):
        """后台定期检查 MQTT 连接健康"""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if self._client is None:
                continue
            # MQTT 协议自带 keepalive ping，检查 client 是否还存活
            try:
                # 通过尝试一次发布来探测连接状态
                await asyncio.wait_for(
                    self._client.publish("animat/health", b"ping", qos=0),
                    timeout=3.0,
                )
                if self._health == NervousHealth.DEAD:
                    self._health = NervousHealth.DEGRADED
            except Exception:
                if self._health != NervousHealth.DEAD:
                    logger.error("MQTT health check failed — marking DEAD")
                self._health = NervousHealth.DEAD
                self._running = False
                asyncio.create_task(self._reconnect_loop())

    async def _resubscribe_all(self):
        """重连后重新订阅所有 topic"""
        for topic in self._subscribers:
            await self._client.subscribe(topic, qos=1)
            logger.info(f"Resubscribed: {topic}")

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
        try:
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
                    except Exception:
                        logger.error(f"Handler error [{topic}]: {handler.__name__}", exc_info=True)
        except asyncio.CancelledError:
            raise
        except aiomqtt.MqttError as e:
            if self._running:
                logger.warning(f"Dispatch loop interrupted: {e}")
                self._health = NervousHealth.DEGRADED
            else:
                logger.debug("Dispatch loop closed during shutdown")

    # ── 便捷: 一次性发布+忘记 ──
    async def reflex_fire(self, topic: str, payload: Any):
        """反射弧专用: 最高优先级、最低延迟的脉冲发射"""
        await self.fire(topic, payload, qos=1)
