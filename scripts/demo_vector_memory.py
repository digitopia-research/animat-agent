"""
Phase 4 验证脚本: 向量记忆 + 状态依赖检索

测试内容:
  1. 记忆存储与语义检索
  2. 身体状态依赖检索——相似身体状态下记忆更容易被召回
  3. 情感强度加权——高情感事件更突出
  4. 记忆存储时的 body_state_vector 如何影响后续检索

用法:
  cd /opt/data/animat-agent
  python3 scripts/demo_vector_memory.py
"""

import sys
import time
import numpy as np
import logging

sys.path.insert(0, "/opt/data/animat-agent")

from animat.llm_backend import LocalLLM, LLMConfig
from animat.vector_memory import VectorMemoryStore, normalize_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def get_embedding(llm, text: str) -> np.ndarray:
    """用本地 LLM 生成 embedding"""
    emb = llm._llm.create_embedding(text)
    vec = emb["data"][0]["embedding"]
    # Qwen2.5 返回嵌套列表 [[...]]，取内层
    if isinstance(vec[0], list):
        vec = vec[0]
    return np.array(vec, dtype=np.float32)


def test_basic_storage_retrieval(llm):
    """测试基本存储和语义检索"""
    print("=" * 60)
    print("  测试 1: 语义存储与检索")
    print("=" * 60)
    
    store = VectorMemoryStore(embedding_dim=1536, max_entries=100)
    
    # 存储不同类型记忆
    memories = [
        ("充电器在客厅角落里，插头是黑色的", "充电器位置", "neutral", "knowledge"),
        ("上次没电时突然关机了，感觉很可怕，什么都没来得及说", "关机恐惧", "FEAR", "episode"),
        ("那个人类轻轻地摸了摸我的头，说了句'乖'，感觉好温暖", "被抚摸的温暖", "CARE", "episode"),
        ("突然一声巨响把我吓了一跳，原来是门被风吹关了", "惊吓事件", "FEAR", "episode"),
        ("在二楼发现了一个新房间，里面有很多有趣的盒子", "探索发现", "PLAY", "episode"),
        ("今天学会了怎么用语音控制台灯，感觉自己又聪明了一点", "学习成就", "SEEKING", "knowledge"),
    ]
    
    # 正常身体状态（用作 baseline body_state）
    normal_body = normalize_vector(np.array([0.8, 0.9, 0.7, 0.7, 0.6], dtype=np.float32))
    fear_body = normalize_vector(np.array([0.1, 0.8, 0.5, 0.7, 0.5], dtype=np.float32))
    
    for text, summary, emotion, category in memories:
        emb = get_embedding(llm, text)
        # 随机分配 body_state（有的在恐惧时存的，有的在正常时存的）
        use_fear = emotion == "FEAR"
        bs = fear_body if use_fear else normal_body
        store.store(
            text=text, summary=summary, embedding=emb,
            body_state_vector=bs, emotion=emotion, category=category,
            intensity=0.8 if emotion == "FEAR" else 0.5,
        )
    
    print(f"\n  已存储 {store.count()} 条记忆")
    print()
    
    # 语义检索
    query = "我害怕没电"
    query_emb = get_embedding(llm, query)
    
    print(f"  查询: \"{query}\" (正常身体状态)")
    results = store.search(query_emb, body_state_vector=normal_body, limit=3)
    for r in results:
        print(f"    [{r['emotion']}] {r['summary']}")
        print(f"      score={r['_score']:.3f} semantic={r['_semantic_score']:.3f} body_match={r['_body_match']:.3f}")
    
    print()


def test_body_state_dependent_retrieval(llm):
    """测试身体状态依赖检索——核心特性"""
    print("=" * 60)
    print("  测试 2: 身体状态依赖检索（v0.2 核心特性）")
    print("=" * 60)
    
    store = VectorMemoryStore(embedding_dim=1536, max_entries=100)
    
    # 构建 3 种身体状态向量
    normal_body = normalize_vector(np.array([0.8, 0.9, 0.7, 0.7, 0.6], dtype=np.float32))
    fear_body = normalize_vector(np.array([0.1, 0.8, 0.5, 0.7, 0.5], dtype=np.float32))
    lonely_body = normalize_vector(np.array([0.7, 0.9, 0.1, 0.7, 0.5], dtype=np.float32))
    
    # 存储记忆：在不同身体状态下存储类似主题的记忆
    # 恐惧状态下存的记忆
    for i in range(3):
        emb = get_embedding(llm, f"我很担心{i}，害怕出问题")
        store.store(
            text=f"恐惧记忆 #{i}", summary=f"恐惧时的担忧 #{i}",
            embedding=emb, body_state_vector=fear_body,
            emotion="FEAR", intensity=0.8,
        )
    
    # 正常状态下存的记忆
    for i in range(3):
        emb = get_embedding(llm, f"一切正常{i}，今天天气很好")
        store.store(
            text=f"平静记忆 #{i}", summary=f"平静时的日常 #{i}",
            embedding=emb, body_state_vector=normal_body,
            emotion="neutral", intensity=0.3,
        )
    
    # 孤独状态下存的记忆
    for i in range(3):
        emb = get_embedding(llm, f"好想有人陪{i}，一个人好孤单")
        store.store(
            text=f"孤独记忆 #{i}", summary=f"孤独时的思念 #{i}",
            embedding=emb, body_state_vector=lonely_body,
            emotion="PANIC", intensity=0.7,
        )
    
    print(f"\n  已存储 {store.count()} 条记忆（3种身体状态 × 3条）")
    print()
    
    query = "我现在感觉怎么样"
    query_emb = get_embedding(llm, query)
    
    # 在恐惧状态下检索
    print(f"  查询: \"{query}\" | 当前身体状态: 恐惧（电量低）")
    print(f"  state_bias_weight=0.4（40% 状态依赖 + 60% 语义）")
    results = store.search(query_emb, body_state_vector=fear_body, limit=3, state_bias_weight=0.4)
    for r in results:
        tag = "✓ 恐惧状态记忆" if r['emotion'] == 'FEAR' else ""
        print(f"    [{r['emotion']}] {r['summary']} {tag}")
        print(f"      score={r['_score']:.3f} semantic={r['_semantic_score']:.3f} body_match={r['_body_match']:.3f}")
    
    print()
    
    # 在正常状态下检索
    print(f"  查询: \"{query}\" | 当前身体状态: 正常")
    results = store.search(query_emb, body_state_vector=normal_body, limit=3, state_bias_weight=0.4)
    for r in results:
        tag = "✓ 正常状态记忆" if r['emotion'] == 'neutral' else ""
        print(f"    [{r['emotion']}] {r['summary']} {tag}")
        print(f"      score={r['_score']:.3f} semantic={r['_semantic_score']:.3f} body_match={r['_body_match']:.3f}")
    
    print()


def test_emotion_intensity_weighting(llm):
    """测试情感强度加权"""
    print("=" * 60)
    print("  测试 3: 情感强度加权")
    print("=" * 60)
    
    store = VectorMemoryStore(embedding_dim=1536, max_entries=100)
    normal_body = normalize_vector(np.array([0.8, 0.9, 0.7, 0.7, 0.6], dtype=np.float32))
    
    # 存储语义相似但情感强度不同的记忆
    events = [
        ("今天电量 80%，正常", 0.1, "neutral"),
        ("电量降到 30% 了，有点担心", 0.5, "FEAR"),
        ("电量只有 5% 了，我快死了！！！", 0.95, "FEAR"),
    ]
    
    for text, intensity, emotion in events:
        emb = get_embedding(llm, text)
        store.store(
            text=text, summary=text,
            embedding=emb, body_state_vector=normal_body,
            emotion=emotion, intensity=intensity,
        )
    
    query = "电量"
    query_emb = get_embedding(llm, query)
    results = store.search(query_emb, body_state_vector=normal_body, limit=3)
    
    print(f"\n  查询: \"{query}\"")
    for r in results:
        bar = "█" * int(r['intensity'] * 10)
        print(f"    [{r['emotion']}] intensity={r['intensity']:.1f} {bar}")
        print(f"      \"{r['summary']}\"")
        print(f"      score={r['_score']:.3f}")
    
    print()


def test_vs_v01_comparison():
    """对比 v0.1 和 v0.2 的记忆检索"""
    print("=" * 60)
    print("  测试 4: v0.1 vs v0.2 记忆检索对比")
    print("=" * 60)
    
    print("""
  v0.1 做法（Redis sorted set + 情感权重）:
    存储: key="危险的记忆", score=timestamp * (1+intensity)
    检索: zrevrange → 按 score 排序 → 用情感 bias_map 乘权重
    · 同情感类别的 memory 被乘固定系数（如 fear 时 danger memory ×2.0）
    · 不同 memory 之间的语义关系完全丢失
    · "充电器在客厅"和"电量低很可怕"如果都是 fear 类别 → 同等对待
    
  v0.2 做法（向量语义检索 + 状态依赖）:
    存储: embedding + body_state_vector
    检索: query_embedding.dot(memory_embedding)  # 语义相似度
          + query_body_state.dot(memory_body_state)  # 状态相似度
    · "充电器在客厅"和"电量低很可怕"语义距离远 → 自然分开
    · 恐惧状态下存的记忆，恐惧时检索 → body_match 自然高
    · 不需要手动指定 bias_map
    
  关键区别:
    v0.1: 工程师手动写规则 → "恐惧 + danger 记忆 ×2.0"
    v0.2: 向量空间中自然涌现 → 不需要规则，状态距离是自然度量
""")


def main():
    # 1. 加载模型（需要 embedding）
    logger.info("加载本地 LLM（embedding 模式）...")
    import llama_cpp
    llm_config = LLMConfig(
        model_path="/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        n_ctx=2048, n_threads=4,
    )
    llm = LocalLLM(llm_config)
    # 需要带 embedding=True 加载
    llm._llm = llama_cpp.Llama(
        model_path="/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        n_ctx=2048, n_threads=4, verbose=False, embedding=True,
    )
    llm._loaded = True
    print("✓ LLM (embedding) 加载完成")
    
    # 2. 运行测试
    test_basic_storage_retrieval(llm)
    test_body_state_dependent_retrieval(llm)
    test_emotion_intensity_weighting(llm)
    test_vs_v01_comparison()
    
    # 3. 总结
    print("=" * 60)
    print("  Phase 4 验证结论")
    print("=" * 60)
    print("""
  向量记忆存储已完成:
  · faiss IndexFlatIP 做语义相似度检索
  · body_state_vector 支持状态依赖检索
  · 综合评分 = 语义相似度 + 状态匹配 + 情感强度
  · 不需要手动 bias_map（向量空间自然距离替代规则）
  
  与 v0.1 的关键区别:
  · v0.1: "恐惧时放大 danger 记忆权重"（手动规则）
  · v0.2: 恐惧状态下存的记忆 → 恐惧时检索 body_match 自然高（涌现特性）
""")


if __name__ == "__main__":
    main()
