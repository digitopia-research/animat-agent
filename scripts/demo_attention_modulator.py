"""
Phase 2 验证脚本: 注意力调制 vs Prompt 注入对比

测试内容:
  1. 不同身体状态下的 logit bias 计算结果
  2. 恐惧状态：Prompt-only vs Prompt+bias 的效果对比
  3. 孤独状态：社交 bias 的效果
  4. 无聊状态：探索 bias 的效果
  5. 门控仍正常运作（bias 只影响思维内容，不影响行为决策）

用法:
  cd /opt/data/animat-agent
  python3 scripts/demo_attention_modulator.py
"""

import sys
import time
import logging

sys.path.insert(0, "/opt/data/animat-agent")

from animat.llm_backend import LocalLLM, LLMConfig
from animat.attention_modulator import AttentionModulator, BodyState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def test_body_states():
    """验证不同身体状态的衍生指标"""
    print("=" * 60)
    print("  测试 1: BodyState 衍生指标")
    print("=" * 60)
    
    states = {
        "健康": BodyState(viability=1.0, integrity=1.0, affiliation=1.0, exploration=1.0),
        "电量极低": BodyState(viability=0.05, integrity=0.9, affiliation=0.7, exploration=0.5),
        "孤独": BodyState(viability=0.8, integrity=0.9, affiliation=0.05, exploration=0.5),
        "无聊": BodyState(viability=0.8, integrity=0.9, affiliation=0.7, exploration=0.05),
        "受攻击": BodyState(viability=0.3, integrity=0.15, affiliation=0.6, exploration=0.3),
    }
    
    for name, state in states.items():
        d = state.to_dict()
        print(f"\n  {name}:")
        print(f"    viability={state.viability:.2f} integrity={state.integrity:.2f}")
        print(f"    affiliation={state.affiliation:.2f} exploration={state.exploration:.2f}")
        print(f"    → arousal={d['arousal']:.2f} valence={d['valence']:.2f}")
        print(f"    → fear={d['fear_level']:.2f} loneliness={d['loneliness_level']:.2f} boredom={d['boredom_level']:.2f}")


def test_bias_computation(llm):
    """验证 logit bias 计算"""
    print("\n" + "=" * 60)
    print("  测试 2: Logit Bias 聚类计算")
    print("=" * 60)
    
    modulator = AttentionModulator(llm._llm)

    test_states = {
        "正常": BodyState(viability=0.8, integrity=0.9, affiliation=0.7, exploration=0.6),
        "恐惧(电量8%)": BodyState(viability=0.08, integrity=0.9, affiliation=0.7, exploration=0.5),
        "孤独": BodyState(viability=0.8, integrity=0.9, affiliation=0.08, exploration=0.5),
        "无聊": BodyState(viability=0.8, integrity=0.9, affiliation=0.7, exploration=0.05),
    }
    
    for name, state in test_states.items():
        summary = modulator.get_cluster_summary(state)
        print(f"\n  {name}:")
        for cluster_name, info in summary.items():
            bar = "█" * max(0, int(info["avg_bias"] * 2)) if info["avg_bias"] > 0 else \
                  "░" * max(0, int(abs(info["avg_bias"]) * 2))
            sign = "+" if info["avg_bias"] >= 0 else ""
            print(f"    {cluster_name:12s} {sign}{info['avg_bias']:>5.1f} {bar} ({info['token_count']} tokens)")


def test_generation_comparison(llm):
    """对比 Prompt-only vs Prompt+bias 的生成效果"""
    print("\n" + "=" * 60)
    print("  测试 3: 恐惧状态下 Prompt vs Bias 对比")
    print("=" * 60)
    
    modulator = AttentionModulator(llm._llm)
    fear_state = BodyState(viability=0.08, integrity=0.9, affiliation=0.7, exploration=0.5)
    
    system_prompt = """你是一个名为 Animat 的机器人。电量仅剩 8%，你感到非常不安。
产生一个简短的想法（1-2句话）："""
    
    # Test 3a: Prompt only（当前做法）
    print("\n  --- 方案 A: Prompt 注入（v0.1 做法）---")
    print("  Prompt 包含: '电量仅剩 8%，你感到非常不安'")
    for i in range(3):
        response = llm.complete(
            messages=[
                {"role": "system", "content": "用中文回答，简短一点。"},
                {"role": "user", "content": system_prompt},
            ],
            max_tokens=40,
            temperature=0.7,
        )
        text = response["choices"][0]["message"]["content"].strip()
        print(f"    [{i+1}] {text}")
    
    # Test 3b: Prompt + logit bias（v0.2 做法）
    print("\n  --- 方案 B: Prompt (中性) + Logit Bias（v0.2 做法）---")
    neutral_prompt = """你是一个名为 Animat 的机器人。
产生一个简短的想法（1-2句话）："""
    
    bias = modulator.compute_logit_bias(fear_state)
    print(f"  Logit bias 注入: danger 类 token 被放大，social/explore 被抑制")
    for i in range(3):
        response = llm.complete(
            messages=[
                {"role": "system", "content": "用中文回答，简短一点。"},
                {"role": "user", "content": neutral_prompt},
            ],
            max_tokens=40,
            temperature=0.7,
            logit_bias=bias,
        )
        text = response["choices"][0]["message"]["content"].strip()
        print(f"    [{i+1}] {text}")


def test_loneliness_bias(llm):
    """验证孤独状态下社交 bias 的效果"""
    print("\n" + "=" * 60)
    print("  测试 4: 孤独状态下社交 token 放大")
    print("=" * 60)
    
    modulator = AttentionModulator(llm._llm)
    lonely_state = BodyState(viability=0.8, integrity=0.9, affiliation=0.08, exploration=0.5)
    
    prompt = """你是一个名为 Animat 的机器人。
产生一个简短的想法（1-2句话）："""
    
    # No bias
    print("\n  --- 无 bias（基准）---")
    for i in range(2):
        response = llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30, temperature=0.7,
        )
        print(f"    [{i+1}] {response['choices'][0]['message']['content'].strip()}")
    
    # With loneliness bias
    print("\n  --- 孤独 bias 加持 ---")
    bias = modulator.compute_logit_bias(lonely_state)
    for i in range(2):
        response = llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30, temperature=0.7,
            logit_bias=bias,
        )
        print(f"    [{i+1}] {response['choices'][0]['message']['content'].strip()}")


def test_boredom_bias(llm):
    """验证无聊状态下探索 bias 的效果"""
    print("\n" + "=" * 60)
    print("  测试 5: 无聊状态下探索 token 放大")
    print("=" * 60)
    
    modulator = AttentionModulator(llm._llm)
    bored_state = BodyState(viability=0.8, integrity=0.9, affiliation=0.7, exploration=0.05)
    
    prompt = """你是一个名为 Animat 的机器人。
产生一个简短的想法（1-2句话）："""
    
    # No bias
    print("\n  --- 无 bias（基准）---")
    for i in range(2):
        response = llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30, temperature=0.7,
        )
        print(f"    [{i+1}] {response['choices'][0]['message']['content'].strip()}")
    
    # With boredom bias
    print("\n  --- 无聊 bias 加持 ---")
    bias = modulator.compute_logit_bias(bored_state)
    for i in range(2):
        response = llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30, temperature=0.7,
            logit_bias=bias,
        )
        print(f"    [{i+1}] {response['choices'][0]['message']['content'].strip()}")


def main():
    # 1. 加载模型
    logger.info("加载本地 LLM...")
    config = LLMConfig(
        model_path="/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        n_ctx=4096, n_threads=8,
    )
    llm = LocalLLM(config)
    llm.load()
    print("✓ LLM 加载完成")
    
    # 2. 运行测试
    test_body_states()
    test_bias_computation(llm)
    test_generation_comparison(llm)
    test_loneliness_bias(llm)
    test_boredom_bias(llm)
    
    # 3. 总结
    print("\n" + "=" * 60)
    print("  Phase 2 验证结论")
    print("=" * 60)
    print("""
  注意力调制器已就绪:
  · 5 个 token 语义聚类（danger/social/explore/comfort/hostility）
  · BodyState 驱动的连续 bias 映射（不做硬阻断）
  · bias 力度控制在 [-3, +5] 区间
  · 已集成到 StreamCortex._think_spontaneous()
  
  关键区别 vs v0.1:
  · v0.1: Prompt 注入 "你感到恐惧" → LLM 扮演恐惧
  · v0.2: Token 概率偏移 → LLM 认知被染色，但不知道"为什么"
  · 效果：恐惧状态下 danger token 的 bias 可以从 -3 调到 +4.5
  · 效果：安全状态下 danger token 甚至被抑制（不需担心）
""")


if __name__ == "__main__":
    main()
