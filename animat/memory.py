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
from typing import Optional

import redis.asyncio as aioredis

from animat.config import RedisKeys

logger = logging.getLogger("animat.memory")


class MemoryStore:
    """Redis 记忆存储"""

    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", "6379")),
        db: int = int(os.getenv("REDIS_DB", "0")),
    ):
        self.host = host
        self.port = port
        self.db = db
        self.redis = aioredis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._connected = False

    async def connect(self, timeout: float = 10.0):
        try:
            await asyncio.wait_for(self.redis.ping(), timeout=timeout)
            self._connected = True
            logger.info(f"Memory store connected: {self.host}:{self.port}")
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Redis connection timed out ({timeout}s): "
                f"{self.host}:{self.port} — is Redis running?"
            )
        except Exception as e:
            raise ConnectionError(
                f"Redis connection failed: {self.host}:{self.port} — {e}"
            )

    async def close(self):
        await self.redis.close()
        self._connected = False

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
        await self.redis.zadd(
            RedisKeys.MEMORY_EPISODIC,
            {json.dumps(episode, ensure_ascii=False): score},
        )
        # 保持最近1000条
        count = await self.redis.zcard(RedisKeys.MEMORY_EPISODIC)
        if count > 1000:
            await self.redis.zremrangebyrank(RedisKeys.MEMORY_EPISODIC, 0, count - 1001)
        logger.debug(f"Episode stored: [{emotion}] {summary[:40]}")

    async def recall_episodes(
        self,
        limit: int = 5,
        emotion_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """按时间倒序检索情景记忆，可选情感/类别过滤"""
        raw = await self.redis.zrevrange(RedisKeys.MEMORY_EPISODIC, 0, limit * 3 - 1, withscores=True)
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
        await self.redis.hset(RedisKeys.MEMORY_SEMANTIC, key, json.dumps(value, ensure_ascii=False))

    async def recall_semantic(self, key: str) -> Optional[dict]:
        """检索语义知识"""
        raw = await self.redis.hget(RedisKeys.MEMORY_SEMANTIC, key)
        if raw:
            return json.loads(raw)
        return None

    # ─────────────── 状态存取 ───────────────
    async def save_state(self, key: str, state: dict):
        """保存当前状态快照"""
        await self.redis.hset(key, mapping={k: json.dumps(v, ensure_ascii=False) for k, v in state.items()})

    async def load_state(self, key: str) -> dict:
        """加载状态快照"""
        raw = await self.redis.hgetall(key)
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
        await self.redis.lpush(RedisKeys.STATE_FEELINGS, json.dumps(feeling, ensure_ascii=False))
        await self.redis.ltrim(RedisKeys.STATE_FEELINGS, 0, 99)  # 保留最近100条

    async def get_recent_feelings(self, limit: int = 10) -> list[dict]:
        """获取最近的感受"""
        raw = await self.redis.lrange(RedisKeys.STATE_FEELINGS, 0, limit - 1)
        return [json.loads(item) for item in raw]
