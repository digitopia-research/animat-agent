"""
记忆系统 — 基于Redis的记忆存储与检索

设计:
  - 情景记忆 (Episodic): Sorted Set, score = 时间戳*情感权重
  - 语义记忆 (Semantic): Hash, 结构化知识
  - 短期状态: Hash, 当前稳态/情感/感受
  - 对话历史: List, LLM交互记录

记忆编码规则:
  - 每条情景记忆带有: 时间、情感标签、摘要、原始内容、类别
  - 写入时根据情感强度加权score，高情感事件更容易被检索
"""

import os
import json
import time
import logging
import asyncio
from enum import Enum
from typing import Optional

import redis.asyncio as aioredis

from animat.config import RedisKeys

logger = logging.getLogger("animat.memory")

# ── 重连配置 ──
RECONNECT_BASE_DELAY = 1.0    # 初始重连延迟(秒)
RECONNECT_MAX_DELAY  = 30.0   # 最大重连延迟(秒)
RECONNECT_BACKOFF    = 2.0    # 退避因子
HEALTH_CHECK_INTERVAL = 5.0   # 健康检查间隔(秒)


class MemoryHealth(str, Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"     # 最近一次操作失败，但尚未判定死亡
    DEAD      = "dead"         # 连续失败，不可用


class MemoryStore:
    """Redis 记忆存储 — 带自动重连和健康检查"""

    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", "6379")),
        db: int = int(os.getenv("REDIS_DB", "0")),
    ):
        self.host = host
        self.port = port
        self.db = db
        self.redis: Optional[aioredis.Redis] = None
        self._connected = False
        self._reconnecting = False
        self._health = MemoryHealth.DEAD
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    @property
    def healthy(self) -> bool:
        return self._health != MemoryHealth.DEAD

    async def connect(self, timeout: float = 10.0):
        """建立连接并启动后台健康检查"""
        await self._do_connect(timeout)
        # 后台健康检查
        asyncio.create_task(self._health_check_loop())

    async def _do_connect(self, timeout: float = 10.0):
        try:
            self.redis = aioredis.Redis(
                host=self.host, port=self.port, db=self.db, decode_responses=True
            )
            await asyncio.wait_for(self.redis.ping(), timeout=timeout)
            self._connected = True
            self._health = MemoryHealth.HEALTHY
            self._consecutive_failures = 0
            logger.info(f"Memory store connected: {self.host}:{self.port}")
        except asyncio.TimeoutError:
            self._health = MemoryHealth.DEAD
            raise ConnectionError(
                f"Redis connection timed out ({timeout}s): "
                f"{self.host}:{self.port} — is Redis running?"
            )
        except Exception as e:
            self._health = MemoryHealth.DEAD
            raise ConnectionError(
                f"Redis connection failed: {self.host}:{self.port} — {e}"
            )

    async def _reconnect_loop(self):
        """指数退避自动重连"""
        if self._reconnecting:
            return
        self._reconnecting = True
        delay = RECONNECT_BASE_DELAY
        while self._health == MemoryHealth.DEAD:
            try:
                logger.info(f"Attempting Redis reconnect (delay={delay:.0f}s)...")
                await self._do_connect()
                logger.info("Redis reconnected successfully")
                break
            except Exception:
                logger.warning(f"Redis reconnect failed, next attempt in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * RECONNECT_BACKOFF, RECONNECT_MAX_DELAY)
        self._reconnecting = False

    async def _health_check_loop(self):
        """后台定期 ping 检测连接健康"""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if self.redis is None:
                continue
            try:
                await asyncio.wait_for(self.redis.ping(), timeout=3.0)
                if self._health == MemoryHealth.DEAD:
                    self._health = MemoryHealth.DEGRADED
                self._consecutive_failures = 0
            except Exception:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    if self._health != MemoryHealth.DEAD:
                        logger.error("Redis health check failed repeatedly — marking DEAD")
                    self._health = MemoryHealth.DEAD
                    self._connected = False
                    asyncio.create_task(self._reconnect_loop())
                else:
                    self._health = MemoryHealth.DEGRADED

    async def _safe_execute(self, operation_name: str, coro):
        """安全执行 Redis 操作，失败时自动降级"""
        if self._health == MemoryHealth.DEAD and not self._reconnecting:
            asyncio.create_task(self._reconnect_loop())
            return None
        try:
            return await coro
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(f"Redis operation '{operation_name}' failed: {e}")
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._health = MemoryHealth.DEAD
                self._connected = False
                asyncio.create_task(self._reconnect_loop())
            return None

    async def close(self):
        if self.redis:
            await self.redis.close()
        self._connected = False
        self._health = MemoryHealth.DEAD

    # ─────────────── 情景记忆 ───────────────
    async def store_episode(
        self,
        summary: str,
        content: str,
        emotion: str,
        category: str = "general",
        intensity: float = 0.5,
        extra: dict = None,
    ):
        """
        存储一段情景记忆
        - score = timestamp * (1 + intensity) — 高情感事件更容易被回忆
        """
        episode = {
            "summary": summary,
            "content": content,
            "emotion": emotion,
            "category": category,
            "intensity": intensity,
            "timestamp": time.time(),
            **(extra or {}),
        }
        score = time.time() * (1 + intensity)
        await self._safe_execute("store_episode", self.redis.zadd(
            RedisKeys.MEMORY_EPISODIC,
            {json.dumps(episode, ensure_ascii=False): score},
        ))
        # 保持最近1000条
        count = await self._safe_execute("store_episode_trim", self.redis.zcard(RedisKeys.MEMORY_EPISODIC))
        if count is not None and count > 1000:
            await self._safe_execute("store_episode_trim", self.redis.zremrangebyrank(
                RedisKeys.MEMORY_EPISODIC, 0, count - 1001
            ))
        logger.debug(f"Episode stored: [{emotion}] {summary[:40]}")

    async def recall_episodes(
        self,
        limit: int = 5,
        emotion_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """按时间倒序检索情景记忆，可选情感/类别过滤"""
        raw = await self._safe_execute(
            "recall_episodes",
            self.redis.zrevrange(RedisKeys.MEMORY_EPISODIC, 0, limit * 3 - 1, withscores=True)
        )
        if raw is None:
            return []
        results = []
        for item, score in raw:
            try:
                ep = json.loads(item)
            except json.JSONDecodeError:
                continue
            if emotion_filter and ep.get("emotion") != emotion_filter:
                continue
            if category_filter and ep.get("category") != category_filter:
                continue
            results.append(ep)
            if len(results) >= limit:
                break
        return results

    # ─────────────── 语义记忆 ───────────────
    async def store_semantic(self, key: str, value: dict):
        """存储语义知识"""
        await self._safe_execute("store_semantic",
            self.redis.hset(RedisKeys.MEMORY_SEMANTIC, key, json.dumps(value, ensure_ascii=False)))

    async def recall_semantic(self, key: str) -> Optional[dict]:
        """检索语义知识"""
        raw = await self._safe_execute("recall_semantic",
            self.redis.hget(RedisKeys.MEMORY_SEMANTIC, key))
        if raw:
            return json.loads(raw)
        return None

    # ─────────────── 状态存取 ───────────────
    async def save_state(self, key: str, state: dict):
        """保存当前状态快照"""
        await self._safe_execute("save_state",
            self.redis.hset(key, mapping={k: json.dumps(v, ensure_ascii=False) for k, v in state.items()}))

    async def load_state(self, key: str) -> dict:
        """加载状态快照"""
        raw = await self._safe_execute("load_state", self.redis.hgetall(key))
        if raw is None:
            return {}
        result = {}
        for k, v in raw.items():
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v
        return result

    # ─────────────── 感受时间线 ───────────────
    async def append_feeling(self, feeling: dict):
        """追加感受到时间线"""
        await self._safe_execute("append_feeling",
            self.redis.lpush(RedisKeys.STATE_FEELINGS, json.dumps(feeling, ensure_ascii=False)))
        await self._safe_execute("append_feeling_trim",
            self.redis.ltrim(RedisKeys.STATE_FEELINGS, 0, 99))  # 保留最近100条

    async def get_recent_feelings(self, limit: int = 10) -> list[dict]:
        """获取最近的感受"""
        raw = await self._safe_execute("get_recent_feelings",
            self.redis.lrange(RedisKeys.STATE_FEELINGS, 0, limit - 1))
        if raw is None:
            return []
        return [json.loads(item) for item in raw]
