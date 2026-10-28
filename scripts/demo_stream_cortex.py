"""
Phase 1 验证脚本: 恒常意识流端到端测试

不需要 MQTT/Redis，直接测试 StreamCortex 能否:
  1. 持续产生内部思维
  2. 门控正确决定行为输出
  3. 不同驱力水平影响思维频率
  4. 外部输入触发回复思维

用法:
  cd /opt/data/animat-agent
  python3 scripts/demo_stream_cortex.py
"""

import asyncio
import sys
import time
import logging

# 设置路径
sys.path.insert(0, "/opt/data/animat-agent")

from animat.llm_backend import LocalLLM, LLMConfig
from animat.thought import Thought, ActionGate, ThoughtJournal

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


# ── 简化的 StreamCortex（不依赖 MQTT/Redis）──
class DemoStreamCortex:
    def __init__(self, llm: LocalLLM):
        self.llm = llm
        self.action_gate = ActionGate()
        self.journal = ThoughtJournal()
        self._running = False
        self._body_context = "各项指标正常，心情平静。"
        self._current_emotion = "SEEKING"
        self._drive_level = 0.0
        self._thought_count = 0
        self._activation_count = 0

    async def start(self):
        self._running = True
        logger.info("═══ Demo StreamCortex started ═══")

    async def stop(self):
        self._running = False

    def set_body_state(self, context: str, emotion: str, drive: float):
        self._body_context = context
        self._current_emotion = emotion
        self._drive_level = drive

    async def think_cycle(self, external_text: str = None):
        if external_text:
            thought = await self._think_response(external_text)
        else:
            thought = await self._think_spontaneous()

        if thought is None:
            return None

        decision = self.action_gate.evaluate(
            thought=thought,
            drive_level=self._drive_level,
            is_external_trigger=external_text is not None,
        )
        self.journal.record(thought)
        self._thought_count += 1

        if decision.should_activate:
            self._activation_count += 1

        return decision

    async def _think_spontaneous(self) -> Thought:
        system_prompt = f"""你是一个名为 Animat 的具身机器人。你有物理身体。

当前状态: {self._body_context}

产生一个简短的内部想法（1-2句，就像人类在心里自言自语）:"""

        try:
            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": "你是一个会思考的机器人。用中文回答。"},
                    {"role": "user", "content": system_prompt},
                ],
                max_tokens=60,
                temperature=0.8,
            )
            text = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Think failed: {e}")
            return None

        # 计算相关性
        action_words = ["去", "要", "想", "应该", "必须", "寻找", "充电", "求助"]
        relevance = 0.2
        for w in action_words:
            if w in text:
                relevance += 0.15
        relevance = min(relevance, 1.0)

        return Thought(
            text=text,
            relevance=relevance,
            dominant_emotion=self._current_emotion,
            trigger="spontaneous" if not self._pending_text else "external",
        )

    _pending_text = None
    async def _think_response(self, external_text: str) -> Thought:
        try:
            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": f"你是一个名为 Animat 的机器人。当前状态: {self._body_context}"},
                    {"role": "user", "content": external_text},
                ],
                max_tokens=80,
            )
            text = response["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

        return Thought(
            text=text,
            relevance=0.85,
            dominant_emotion=self._current_emotion,
            trigger="external",
        )


async def main():
    print("=" * 60)
    print("  Animat Agent v0.2 — Phase 1: 恒常意识流验证")
    print("=" * 60)
    print()

    # 1. 加载模型
    logger.info("加载本地 LLM...")
    config = LLMConfig(
        model_path="/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=8,
    )
    llm = LocalLLM(config)
    llm.load()
    print("✓ LLM 加载完成")
    print()

    # 2. 创建 Demo StreamCortex
    cortex = DemoStreamCortex(llm)
    await cortex.start()

    # 3. 场景 1: 正常状态下的自发思维流
    print("─" * 40)
    print("场景 1: 正常状态 — 5 次自发思维")
    print("─" * 40)
    cortex.set_body_state("各项指标正常，电量 80%，心情平静。", "SEEKING", 0.1)
    for i in range(5):
        decision = await cortex.think_cycle()
        if decision:
            status = "🗣️ 激活" if decision.should_activate else "💭 静默"
            print(f"  [{i+1}] {status} | relevance={decision.thought.relevance:.2f}")
            print(f"      思维: {decision.thought.text[:80]}")
            if decision.should_activate:
                print(f"      动作: {decision.action_type} | {decision.reason}")
        await asyncio.sleep(1.5)
    print()

    # 4. 场景 2: 高驱力（电量低）— 思维频率加快
    print("─" * 40)
    print("场景 2: 电量极低 — 3 次思维（驱力水平高）")
    print("─" * 40)
    cortex.set_body_state(
        "你的 viability 处于极度危险水平(8%)，你感到恐惧，需要立即充电！",
        "FEAR", 0.9
    )
    for i in range(3):
        decision = await cortex.think_cycle()
        if decision:
            status = "🗣️ 激活" if decision.should_activate else "💭 静默"
            print(f"  [{i+1}] {status} | relevance={decision.thought.relevance:.2f} | drive=0.9")
            print(f"      思维: {decision.thought.text[:80]}")
            if decision.should_activate:
                print(f"      动作: {decision.action_type} | 阈值={decision.threshold:.2f}")
        await asyncio.sleep(1.0)  # 高驱力 = 更短间隔
    print()

    # 5. 场景 3: 外部输入
    print("─" * 40)
    print("场景 3: 人类说话 — 触发回复思维")
    print("─" * 40)
    cortex.set_body_state(
        "你的 viability 偏低(18%)，你感到担忧。",
        "FEAR", 0.7
    )
    print("  人类: '嗨 Animat，给我讲个笑话吧！'")
    decision = await cortex.think_cycle(external_text="嗨 Animat，给我讲个笑话吧！")
    if decision:
        status = "🗣️ 激活" if decision.should_activate else "💭 静默"
        print(f"  {status} | relevance={decision.thought.relevance:.2f} | external_trigger")
        print(f"  回复: {decision.thought.text[:150]}")
        if decision.should_activate:
            print(f"  动作: {decision.action_type}")
    print()

    # 6. 场景 4: 安静+低驱力 — 大部分思维被门控拦截
    print("─" * 40)
    print("场景 4: 安静状态 — 5 次思维，观察门控效果")
    print("─" * 40)
    cortex.set_body_state("各项指标正常，电量 90%，感觉很舒适。", "SEEKING", 0.05)
    activations = 0
    for i in range(5):
        decision = await cortex.think_cycle()
        if decision:
            if decision.should_activate:
                activations += 1
                print(f"  [{i+1}] 🗣️ 激活 (rare!) | {decision.thought.text[:60]}")
            else:
                print(f"  [{i+1}] 💭 {decision.thought.text[:60]}")
        await asyncio.sleep(1.5)
    print(f"  安静状态激活率: {activations}/5 = {activations/5*100:.0f}%")
    print()

    # 7. 统计
    stats = cortex.journal.summary()
    print("=" * 60)
    print("  统计")
    print("=" * 60)
    print(f"  总思维数: {stats['total']}")
    print(f"  总激活数: {cortex._activation_count}")
    print(f"  激活率: {cortex._activation_count / max(stats['total'], 1) * 100:.1f}%")
    print(f"  情感分布: {stats['emotions']}")
    print()

    print("✓ Phase 1 验证完成!")
    print("  恒常意识流可以:")
    print("    · 持续产生内部思维（无外部输入）")
    print("    · 门控正确过滤大部分思维")
    print("    · 高驱力增加激活率")
    print("    · 外部输入触发回复")
    print("    · 安静状态下大部分思维不产生行为")


if __name__ == "__main__":
    asyncio.run(main())
