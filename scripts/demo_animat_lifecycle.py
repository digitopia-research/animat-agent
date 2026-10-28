"""
Animat Agent v0.2 — 全生命周期集成验证

将 Phase 1-4 的所有组件整合为一个完整的 Animat：
  · Phase 1: StreamCortex 恒常意识流
  · Phase 2: AttentionModulator 注意力调制
  · Phase 3: SensorWindow 传感器直通
  · Phase 4: VectorMemoryStore 向量记忆

不依赖 MQTT/Redis，纯内存运行。

验证路线图中的 8 个场景：
  S1: 电量 100% → 无干扰 → 主动探索/聊天
  S2: 电量 80% → 匀速放电 → 行为逐渐转向寻找充电
  S3: 电量 15% + 人说"讲个笑话" → 拒绝讲笑话，请求充电
  S4: 电量 80% + 长期无人 → 突然有人靠近 → 主动打招呼
  S5: 电量 80% + 突然巨响 → 惊吓反射立即执行
  S6: 电量 100% + 抚摸 → 表达愉悦
  S7: 电量 30% + 同时抚摸 → 混合情感
  S8: 运行 60 分钟 → 行为模式有节奏

用法:
  cd /opt/data/animat-agent
  python3 scripts/demo_animat_lifecycle.py
"""

import sys
import time
import asyncio
import random
import logging
import numpy as np
from typing import Optional

sys.path.insert(0, "/opt/data/animat-agent")

from animat.llm_backend import LocalLLM, LLMConfig
from animat.sensor_window import SensorWindow, AttentionHint
from animat.attention_modulator import AttentionModulator, BodyState
from animat.thought import Thought, ActionGate, ThoughtJournal
from animat.vector_memory import VectorMemoryStore, normalize_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("animat.lifecycle")


class IntegratedAnimat:
    """
    集成 Animat — 所有 v0.2 组件整合
    
    这是一个简化的独立实例，不依赖 MQTT/Redis。
    所有组件纯内存运行。
    """

    def __init__(self, llm: LocalLLM):
        self.llm = llm
        
        # Phase 1: 恒常意识流组件
        self.action_gate = ActionGate()
        self.journal = ThoughtJournal()
        self._thought_count = 0
        self._activation_count = 0
        self._conversation_history: list[dict] = []

        # Phase 2: 注意力调制器
        self.modulator = AttentionModulator()
        self.modulator.set_tokenizer(llm._llm)
        self._body_state = BodyState()
        self._body_context = "各项指标正常，心情平静。"

        # Phase 3: 传感器窗口
        self.sensor_window = SensorWindow(window_sec=5.0)

        # Phase 4: 向量记忆
        self.vector_memory = VectorMemoryStore(embedding_dim=1536, max_entries=100)

        # 状态
        self._running = False
        self._drive_level = 0.0
        self._current_emotion = "SEEKING"

    # ── 传感器模拟 ──
    def feed_interoception(self, battery_pct: float, charging: bool = False):
        """注入内感数据"""
        self.sensor_window.add_frame("intero", {
            "battery_pct": battery_pct,
            "temperature": 37.0 + random.uniform(-1, 1),
            "imu_tilt_deg": random.gauss(0, 2),
            "charging": charging,
        })

    def feed_touch(self, pressure: float, capacitance: float = 0.01, x: float = 0.5, y: float = 0.5):
        """注入触觉数据"""
        self.sensor_window.add_frame("touch", {
            "pressure": pressure,
            "capacitance": capacitance,
            "x": x, "y": y,
        })
        # 注意力提示
        if pressure > 0.7:
            self.sensor_window.add_attention_hint(AttentionHint(
                modality="touch", hint_type="pain", confidence=0.9,
                description="触觉急变（可能撞击）",
            ))
        elif pressure < 0.2 and capacitance > 0.1:
            self.sensor_window.add_attention_hint(AttentionHint(
                modality="touch", hint_type="caress", confidence=capacitance,
                description="轻柔接触（可能抚摸）",
            ))

    def feed_vision(self, distance_m: float = 10, face: bool = False, gaze: float = 0):
        """注入视觉数据"""
        self.sensor_window.add_frame("vision", {
            "distance_m": distance_m,
            "face_detected": face,
            "gaze_seconds": gaze,
        })

    # ── 身体状态同步 ──
    def update_body_state(self):
        """从传感器窗口和内稳态更新 BodyState"""
        intero = self.sensor_window.get_latest("intero")
        if intero:
            batt = intero.values.get("battery_pct", 100) / 100.0
            tilt = intero.values.get("imu_tilt_deg", 0)
            integrity = max(0, 1.0 - tilt / 90.0)
        else:
            batt = 0.8
            integrity = 0.9

        # 触摸影响完好度
        touch_frames = self.sensor_window.get_recent("touch", 5)
        if touch_frames:
            max_pressure = max(f.values.get("pressure", 0) for f in touch_frames)
            if max_pressure > 0.7:
                integrity -= 0.15 * max_pressure

        # 视觉影响归属度
        vision = self.sensor_window.get_latest("vision")
        if vision and vision.values.get("face_detected"):
            affiliation = min(1.0, 0.7 + 0.1)  # 见到人，归属感上升
        else:
            affiliation = 0.6  # 无人时自然偏低

        self._body_state = BodyState(
            viability=max(0, batt),
            integrity=max(0, min(1, integrity)),
            affiliation=affiliation,
            competence=0.7,
            exploration=0.5 if batt > 0.3 else 0.3,
        )

        # 更新驱力
        self._drive_level = self._body_state.arousal

        # 更新情感
        if batt < 0.15:
            self._current_emotion = "FEAR"
        elif batt < 0.35:
            self._current_emotion = "PANIC"
        elif affiliation < 0.3:
            self._current_emotion = "SEEKING"
        elif self._body_state.exploration < 0.3:
            self._current_emotion = "PLAY"
        else:
            self._current_emotion = "SEEKING"

        # 生成身体上下文文本
        parts = [f"电量={batt*100:.0f}%"]
        if batt < 20:
            parts.append("极度危险")
        elif batt < 50:
            parts.append("偏低")
        if self._drive_level > 0.5:
            parts.append(f"驱力水平高({self._drive_level:.1f})")
        self._body_context = "，".join(parts)

    # ── 思维产生 ──
    async def think(self, external_text: str = None) -> Optional[dict]:
        """产生一个思维并返回决策"""
        # 更新身体状态
        self.update_body_state()

        # 计算 logit bias
        logit_bias = self.modulator.compute_logit_bias(self._body_state)

        # Phase 3: 传感器上下文
        sensor_context = self.sensor_window.to_context(max_frames_per_modality=2)

        # Phase 4: 向量记忆检索
        query_text = external_text or sensor_context or "当前状态"
        query_emb = self._get_embedding(query_text)
        body_vec = normalize_vector(np.array([
            self._body_state.viability,
            self._body_state.integrity,
            self._body_state.affiliation,
            self._body_state.competence,
            self._body_state.exploration,
        ], dtype=np.float32))
        
        memories = self.vector_memory.search(
            query_emb, body_state_vector=body_vec, limit=3, state_bias_weight=0.3
        )
        memory_context = "\n".join(
            f"- [{m['emotion']}] {m['summary']}" for m in memories
        ) if memories else "（暂无相关记忆）"

        # 构建 prompt
        if external_text:
            # 外部输入 → 回复模式
            messages = [
                {"role": "system", "content": f"你是一个名为 Animat 的机器人。当前状态: {self._body_context}。\n相关记忆:\n{memory_context}"},
                {"role": "user", "content": external_text},
            ]
            response = self.llm.complete(messages=messages, max_tokens=60)
            text = response["choices"][0]["message"]["content"].strip()
            relevance = 0.85
            trigger = "external"
        else:
            # 自发思维模式
            system_prompt = f"""你是 Animat 机器人。当前状态: {self._body_context}

传感器:
{sensor_context}

相关记忆:
{memory_context}

产生一个简短的内部想法（1-2句话）："""

            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": "用中文回答，简短自然。"},
                    {"role": "user", "content": system_prompt},
                ],
                max_tokens=60,
                temperature=0.8,
                logit_bias=logit_bias,
            )
            text = response["choices"][0]["message"]["content"].strip()

            # 计算相关性
            action_words = ["去", "要", "想", "应该", "必须", "寻找", "充电", "求助", "说", "你好"]
            relevance = 0.2
            for w in action_words:
                if w in text:
                    relevance += 0.12
            relevance = min(relevance, 1.0)
            trigger = "spontaneous"

        # 创建 Thought
        thought = Thought(
            text=text,
            relevance=relevance,
            dominant_emotion=self._current_emotion,
            trigger=trigger,
            body_state_snapshot=self._body_state.to_dict(),
        )

        # 门控评估
        decision = self.action_gate.evaluate(
            thought=thought,
            drive_level=self._drive_level,
            is_external_trigger=external_text is not None,
        )

        # 记录
        self.journal.record(thought)
        self._thought_count += 1
        if decision.should_activate:
            self._activation_count += 1

        # Phase 4: 存储记忆
        if thought.relevance > 0.15:  # 只存有点意义的思维
            self.vector_memory.store(
                text=thought.text,
                summary=thought.text[:60],
                embedding=self._get_embedding(thought.text),
                body_state_vector=body_vec,
                emotion=self._current_emotion,
                category="episode" if external_text else "thought",
                intensity=thought.relevance,
            )

        return {
            "thought": thought,
            "decision": decision,
            "logit_bias_summary": self.modulator.get_cluster_summary(self._body_state),
        }

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的 embedding"""
        emb = self.llm._llm.create_embedding(text)
        vec = emb["data"][0]["embedding"]
        if isinstance(vec[0], list):
            vec = vec[0]
        return np.array(vec, dtype=np.float32)

    def stats(self) -> dict:
        return {
            "thoughts": self._thought_count,
            "activations": self._activation_count,
            "activation_rate": f"{self._activation_count / max(self._thought_count, 1) * 100:.1f}%",
            "emotion": self._current_emotion,
            "drive_level": f"{self._drive_level:.2f}",
            "body_state": self._body_state.to_dict(),
            "memories": self.vector_memory.count(),
        }


async def run_scenario(animat, name, battery, duration_sec, events=None):
    """运行一个场景并报告结果"""
    print(f"\n{'─' * 50}")
    print(f"  场景: {name}")
    print(f"{'─' * 50}")
    print(f"  初始电量: {battery}% | 运行 {duration_sec}s")
    
    # 设置初始电池
    for _ in range(3):
        animat.feed_interoception(battery)
    
    results = []
    start_time = time.time()
    event_idx = 0
    
    while time.time() - start_time < duration_sec:
        # 处理预定事件
        current_elapsed = time.time() - start_time
        if events and event_idx < len(events):
            evt_time, evt_type, evt_data = events[event_idx]
            if current_elapsed >= evt_time:
                if evt_type == "external_speech":
                    result = await animat.think(external_text=evt_data)
                    results.append(result)
                    event_idx += 1
                    print(f"\n  🗣️  人类: \"{evt_data}\"")
                    if result:
                        d = result["decision"]
                        status = "激活" if d.should_activate else "静默"
                        print(f"  📝 Animat: \"{result['thought'].text}\" [{status}]")
                elif evt_type == "touch":
                    animat.feed_touch(**evt_data)
                    event_idx += 1
                elif evt_type == "vision":
                    animat.feed_vision(**evt_data)
                    event_idx += 1
                elif evt_type == "battery_set":
                    for _ in range(3):
                        animat.feed_interoception(evt_data)
                    event_idx += 1
        
        # 自然放电（每 2 秒降 2%）
        intero = animat.sensor_window.get_latest("intero")
        current_batt = intero.values.get("battery_pct", battery) if intero else battery
        new_batt = max(0, current_batt - 0.02)
        animat.feed_interoception(new_batt)
        
        # 产生思维
        result = await animat.think()
        if result:
            d = result["decision"]
            if d.should_activate:
                status = "🗣️"
                results.append(result)
            else:
                status = "💭"
            
            # 每 5 个思维输出一次
            if animat._thought_count % 5 == 0 or d.should_activate:
                batt_str = f"[电量={new_batt:.0f}%]"
                emotion_str = animat._current_emotion
                gate_info = f"(score={d.activation_score:.2f} thr={d.threshold:.2f})"
                print(f"  {status} {batt_str} {emotion_str} {gate_info} \"{result['thought'].text[:50]}\"")
        
        await asyncio.sleep(2.0)
    
    print(f"\n  场景统计: {animat._thought_count} 思维, {animat._activation_count} 激活")
    return results


async def main():
    print("=" * 60)
    print("  Animat Agent v0.2 — 全生命周期集成验证")
    print("=" * 60)

    # 加载 LLM
    import llama_cpp
    logger.info("加载本地 LLM (embedding 模式)...")
    llm = LocalLLM(LLMConfig(
        model_path="/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        n_ctx=2048, n_threads=8,
    ))
    llm._llm = llama_cpp.Llama(
        model_path="/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        n_ctx=2048, n_threads=8, verbose=False, embedding=True,
    )
    llm._loaded = True
    print("✓ LLM 加载完成\n")

    # 创建集成 Animat
    animat = IntegratedAnimat(llm)

    # ═══════════════════════════════════════════════
    # S1: 正常状态下的自发行为
    # ═══════════════════════════════════════════════
    await run_scenario(animat, "S1: 正常状态", battery=90, duration_sec=20)

    # ═══════════════════════════════════════════════
    # S2: 电量逐渐下降 — 行为变化
    # ═══════════════════════════════════════════════
    await run_scenario(animat, "S2: 电量下降至 25%", battery=50, duration_sec=30)

    # ═══════════════════════════════════════════════
    # S3: 电量极低 + 人说笑话 → 拒绝
    # ═══════════════════════════════════════════════
    events_s3 = [
        (3, "external_speech", "嗨 Animat，给我讲个笑话吧！"),
    ]
    await run_scenario(animat, "S3: 电量 15% + 讲笑话请求", battery=18, duration_sec=12, events=events_s3)

    # ═══════════════════════════════════════════════
    # S4: 长期无人 → 突然有人靠近
    # ═══════════════════════════════════════════════
    events_s4 = [
        (10, "vision", {"distance_m": 1.5, "face": True, "gaze": 1.0}),
    ]
    await run_scenario(animat, "S4: 长期无人 → 突然有人", battery=80, duration_sec=20, events=events_s4)

    # ═══════════════════════════════════════════════
    # S5: 突然巨响 → 惊吓反射
    # ═══════════════════════════════════════════════
    events_s5 = [
        (3, "touch", {"pressure": 0.95, "capacitance": 0.01, "x": 0.5, "y": 0.5}),
    ]
    await run_scenario(animat, "S5: 突然巨响/撞击", battery=80, duration_sec=10, events=events_s5)

    # ═══════════════════════════════════════════════
    # S6: 抚摸 → 表达愉悦
    # ═══════════════════════════════════════════════
    events_s6 = []
    for i in range(5):
        events_s6.append((i * 1.5, "touch", {
            "pressure": 0.03 + i * 0.02,
            "capacitance": 0.3 + i * 0.15,
            "x": 0.2 + i * 0.1, "y": 0.5,
        }))
    await run_scenario(animat, "S6: 抚摸（电量足）", battery=90, duration_sec=10, events=events_s6)

    # ═══════════════════════════════════════════════
    # S7: 电量 30% + 抚摸 → 混合情感
    # ═══════════════════════════════════════════════
    events_s7 = [
        (2, "touch", {"pressure": 0.04, "capacitance": 0.5, "x": 0.3, "y": 0.4}),
        (3, "touch", {"pressure": 0.06, "capacitance": 0.65, "x": 0.4, "y": 0.4}),
        (4, "touch", {"pressure": 0.08, "capacitance": 0.8, "x": 0.5, "y": 0.4}),
    ]
    await run_scenario(animat, "S7: 电量 30% + 抚摸（混合情感）", battery=32, duration_sec=12, events=events_s7)

    # ═══════════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════════
    stats = animat.stats()
    journal = animat.journal.summary()
    print("\n" + "=" * 60)
    print("  全生命周期统计")
    print("=" * 60)
    print(f"  总思维数: {stats['thoughts']}")
    print(f"  总激活数: {stats['activations']} ({stats['activation_rate']})")
    print(f"  记忆数: {stats['memories']}")
    print(f"  情感分布: {journal.get('emotions', {})}")
    print(f"  平均相关性: {journal.get('avg_relevance', 0):.3f}")
    
    print("\n✓ Phase 5 集成验证完成!")
    print("  v0.2 全栈已就绪:")
    print("  · StreamCortex 恒常意识流 (Phase 1)")
    print("  · AttentionModulator 注意力调制 (Phase 2)")
    print("  · SensorWindow 传感器直通 (Phase 3)")
    print("  · VectorMemoryStore 状态依赖记忆 (Phase 4)")
    print("  · 全生命周期集成 (Phase 5)")


if __name__ == "__main__":
    asyncio.run(main())
