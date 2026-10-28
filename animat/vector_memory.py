"""
向量记忆存储 — VectorMemoryStore

替换 v0.1 的 Redis sorted set 情景记忆。

核心思想:
  记忆检索基于 embedding 语义相似度，而非字符串标签 + score 加权。
  
  身体状态不是记忆存储时的分类标签，而是作为检索时的"状态上下文"——
  相似身体状态下形成的记忆，在相似状态下更容易被检索到。
  （就像你焦虑时容易想起所有焦虑的回忆，不需要显式说明"这是焦虑记忆"）

技术选型:
  - faiss IndexFlatIP（内积 = 余弦相似度，向量已归一化时）
  - 内存存储（元数据存 Python list，embedding 存 faiss index）
  - 轻量，零依赖（除了 faiss-cpu + numpy）
"""

import time
import json
import math
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("animat.vector_memory")


@dataclass
class VectorMemory:
    """一条向量记忆"""
    text: str                       # 原始文本
    summary: str                    # 摘要
    embedding: np.ndarray           # 语义向量
    body_state_vector: np.ndarray   # 存储时的身体状态向量（用于状态依赖检索）
    emotion: str = "neutral"        # 情感标签
    category: str = "general"       # 类别
    intensity: float = 0.5          # 情感强度
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "text": self.text,
            "emotion": self.emotion,
            "category": self.category,
            "intensity": self.intensity,
            "timestamp": self.timestamp,
        }


class VectorMemoryStore:
    """
    向量记忆存储
    
    设计:
    - faiss IndexFlatIP 做 embedding 检索（内积，向量 L2 归一化后 = 余弦相似度）
    - body_state_vector 作为额外的检索偏置维度
    - 状态依赖检索: query_body_state 点积 memory_body_state → 相似状态下的记忆加分
    """

    def __init__(self, embedding_dim: int = 1536, body_state_dim: int = 5, max_entries: int = 1000):
        self.embedding_dim = embedding_dim
        self.body_state_dim = body_state_dim
        self.max_entries = max_entries
        
        # faiss index（内积检索）
        import faiss
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # 元数据
        self.memories: list[VectorMemory] = []
        
        # body_state 向量缓存（单独存储，在检索时做状态依赖加分）
        self._body_vectors: list[np.ndarray] = []
    
    def store(
        self,
        text: str,
        summary: str,
        embedding: np.ndarray,
        body_state_vector: np.ndarray,
        emotion: str = "neutral",
        category: str = "general",
        intensity: float = 0.5,
    ):
        """存储一条记忆"""
        # L2 归一化（用于内积 = 余弦相似度）
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        body_state_vector = body_state_vector.astype(np.float32)
        
        mem = VectorMemory(
            text=text,
            summary=summary,
            embedding=embedding,
            body_state_vector=body_state_vector,
            emotion=emotion,
            category=category,
            intensity=intensity,
        )
        
        self.index.add(embedding.reshape(1, -1))
        self.memories.append(mem)
        self._body_vectors.append(body_state_vector)
        
        # 保持上限
        if len(self.memories) > self.max_entries:
            self._prune()
        
        logger.debug(f"Memory stored: [{emotion}] {summary[:40]} (total={len(self.memories)})")

    def search(
        self,
        query_embedding: np.ndarray,
        body_state_vector: Optional[np.ndarray] = None,
        limit: int = 5,
        state_bias_weight: float = 0.3,  # 状态依赖的权重（0=纯语义, 1=纯状态依赖）
    ) -> list[dict]:
        """
        检索相关记忆
        
        Args:
            query_embedding: 查询向量
            body_state_vector: 当前身体状态向量（用于状态依赖检索）
            limit: 返回数量
            state_bias_weight: 状态依赖在总分中的权重
        
        Returns:
            记忆列表，每项带 _score（综合评分）和元数据
        
        综合评分公式:
          score = (1 - state_bias_weight) * semantic_similarity
                + state_bias_weight * body_state_match
        
        其中:
          semantic_similarity = dot(query_embed, memory_embed)  # 余弦相似度
          body_state_match = dot(query_body, memory_body)       # 状态相似度
            (body_state 向量也需归一化)
        """
        if len(self.memories) == 0:
            return []
        
        query_embedding = query_embedding.astype(np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # faiss 语义检索
        k = min(limit * 3, len(self.memories))  # 多取一些做重排序
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.memories):
                continue
            mem = self.memories[idx]
            semantic_score = float(dist)  # 内积 ≈ 余弦相似度
            
            # 状态依赖匹配
            if body_state_vector is not None and idx < len(self._body_vectors):
                mem_body = self._body_vectors[idx]
                # 归一化
                query_body_norm = np.linalg.norm(body_state_vector)
                mem_body_norm = np.linalg.norm(mem_body)
                if query_body_norm > 0 and mem_body_norm > 0:
                    body_match = np.dot(
                        body_state_vector / query_body_norm,
                        mem_body / mem_body_norm,
                    )
                else:
                    body_match = 0.0
            else:
                body_match = 0.0
            
            # 综合评分
            combined_score = (
                (1 - state_bias_weight) * semantic_score
                + state_bias_weight * body_match
            )
            
            # 情感强度加权（高情感记忆更突出）
            combined_score += mem.intensity * 0.1
            
            results.append({
                **mem.to_dict(),
                "_score": round(combined_score, 4),
                "_semantic_score": round(semantic_score, 4),
                "_body_match": round(body_match, 4),
            })
        
        # 按综合评分排序
        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:limit]

    def _prune(self):
        """删除最旧的记忆"""
        # faiss 不支持删除单条，只能重建
        # 简单策略：保留最近的 max_entries 条
        excess = len(self.memories) - self.max_entries
        if excess > 0:
            self.memories = self.memories[excess:]
            self._body_vectors = self._body_vectors[excess:]
            self._rebuild_index()

    def _rebuild_index(self):
        """重建 faiss index"""
        import faiss
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if self.memories:
            embeddings = np.vstack([m.embedding.reshape(1, -1) for m in self.memories])
            self.index.add(embeddings)

    def recent(self, n: int = 5) -> list[dict]:
        """获取最近的 n 条记忆（忽略语义，纯时间序）"""
        recent = self.memories[-n:]
        return [m.to_dict() for m in reversed(recent)]

    def count(self) -> int:
        return len(self.memories)

    def stats(self) -> dict:
        if not self.memories:
            return {"total": 0}
        emotions = {}
        for m in self.memories:
            emotions[m.emotion] = emotions.get(m.emotion, 0) + 1
        return {
            "total": len(self.memories),
            "emotions": emotions,
            "avg_intensity": sum(m.intensity for m in self.memories) / len(self.memories),
            "oldest_ts": self.memories[0].timestamp,
            "newest_ts": self.memories[-1].timestamp,
        }


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2 归一化"""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v
