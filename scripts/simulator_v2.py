"""
Animat Agent v0.2 — 交互式模拟器

不依赖 MQTT/Redis，直接与 IntegratedAnimat 对接。
提供交互式终端控制，模拟传感器输入。

特性:
  - 实时传感器模拟（电量衰减、随机触觉/视觉）
  - 交互式对话（键入文本与 Animat 交流）
  - 特殊命令：/charge, /stop, /reset, /status, /touch, /hit, /approach, /leave
  - 实时显示 Animat 的思维流和行为

用法:
  cd /opt/data/animat-agent
  python3 scripts/simulator_v2.py

或带自定义参数:
  python3 scripts/simulator_v2.py --model /path/to/model.gguf --battery 90
"""

import sys
import os
import asyncio
import time
import random
import logging
import argparse
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animat.llm_backend import LocalLLM, LLMConfig
from animat.sensor_window import SensorWindow, AttentionHint
from animat.attention_modulator import AttentionModulator, BodyState
from animat.thought import Thought, ActionGate, ThoughtJournal
from animat.vector_memory import VectorMemoryStore, normalize_vector

# ── 日志配置 ──
logging.basicConfig(
    level=logging.WARNING,  # 默认只显示 WARNING+，减少噪音
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulator")

# ── 终端颜色（纯 ASCII，无 ANSI 码，避免 UTF-8 多字节字符兼容问题）──
class Colors:
    RESET = ""
    BOLD = ""
    DIM = ""
    RED = ""
    GREEN = ""
    YELLOW = ""
    BLUE = ""
    MAGENTA = ""
    CYAN = ""
    WHITE = ""

    @staticmethod
    def dim(text):
        return text

    @staticmethod
    def bold(text):
        return text

    @staticmethod
    def red(text):
        return text

    @staticmethod
    def green(text):
        return text

    @staticmethod
    def yellow(text):
        return text

    @staticmethod
    def cyan(text):
        return text

    @staticmethod
    def magenta(text):
        return text


class SimulatorV2:
    """
    v0.2 交互式模拟器
    
    直接对接 IntegratedAnimat，不需要 MQTT/Redis。
    """

    def __init__(self, model_path: str = None, n_ctx: int = 1024, n_threads: int = 4):
        self.model_path = model_path or os.getenv(
            "ANIMAT_MODEL_PATH",
            "/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        )
        self.n_ctx = n_ctx
        self.n_threads = n_threads

        # 模拟器状态
        self._battery = 90.0
        self._charging = False
        self._charge_rate = 3.0  # %/秒
        self._human_present = False
        self._human_distance = 10.0
        self._gaze_seconds = 0.0
        self._tick_count = 0
        self._running = False

        # Animat 组件（延迟初始化）
        self.llm = None
        self.animat = None

    # ── 初始化 ──
    def _init_animat(self):
        """加载 LLM 并创建 Animat"""
        import llama_cpp
        
        # 彻底抑制 llama.cpp 的 stderr 噪音（embedding 模式切换警告）
        # llamba.cpp 在 create_embedding 和 create_completion 交替时向 stderr
        # 输出 "init: embeddings required but..." 警告，无法通过 logging 拦截
        self._stderr_backup = os.dup(2)
        self._stderr_null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._stderr_null, 2)
        
        print(f"{Colors.cyan('加载模型:')} {self.model_path}")
        t0 = time.time()
        
        llm_config = LLMConfig(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
        )
        self.llm = LocalLLM(llm_config)
        # 抑制 llama.cpp 构造和 embedding 初始化时的 stderr 噪音
        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()) as _:
            self.llm._llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                embedding=True,
            )
        self.llm._loaded = True
        
        print(f"{Colors.green('✓ 模型加载完成')} ({time.time() - t0:.1f}s)")

        # 创建集成 Animat（复用 lifecycle demo 的类）
        self.animat = _IntegratedAnimat(self.llm)

    # ── 传感器模拟循环 ──
    async def _sensor_loop(self):
        """模拟传感器数据更新"""
        while self._running:
            await asyncio.sleep(2.0)

            # 电量
            if self._charging:
                self._battery = min(100, self._battery + self._charge_rate * 2)
                if self._battery >= 100:
                    self._charging = False
                    print(f"\n  {Colors.green('🔋 充电完成')}")
            else:
                self._battery = max(0, self._battery - random.uniform(0.1, 0.3) * 2)

            # 注入内感
            self.animat.feed_interoception(self._battery, self._charging)

            # 随机视觉（30% 概率检测到人）
            if self._human_present:
                self.animat.feed_vision(
                    distance_m=self._human_distance,
                    face=True,
                    gaze=random.uniform(0, 3),
                )
            elif random.random() < 0.15:
                self.animat.feed_vision(distance_m=10, face=False, gaze=0)

            self._tick_count += 1

    # ── 思维循环 ──
    async def _think_loop(self):
        """Animat 思维循环"""
        while self._running:
            try:
                result = await self.animat.think()
                if result:
                    d = result["decision"]
                    if d.should_activate:
                        # 外部行为 → 高亮显示
                        thought_text = result["thought"].text
                        emotion = self.animat._current_emotion
                        batt = f"{self._battery:.0f}%"
                        is_proactive = getattr(d, '_proactive', False)
                        
                        # 选择合适的前缀
                        if is_proactive:
                            # 主动开口 → 特殊标记
                            prefix = "[主动]"
                            urge = self.animat._expression_urge
                            if self.animat._body_state.fear_level > 0.5:
                                prefix = "[主动·害怕]"
                            elif self.animat._body_state.loneliness_level > 0.5:
                                prefix = "[主动·孤单]"
                            elif self.animat._body_state.boredom_level > 0.5:
                                prefix = "[主动·无聊]"
                            else:
                                prefix = "[主动·想说]"
                            print(f"\n  {prefix} [{batt}] 冲动={urge:.2f}")
                        elif self.animat._body_state.fear_level > 0.6:
                            prefix = Colors.red("Animat")
                        elif self.animat._body_state.loneliness_level > 0.6:
                            prefix = Colors.yellow("Animat")
                        elif self.animat._body_state.boredom_level > 0.6:
                            prefix = Colors.magenta("Animat")
                        else:
                            prefix = Colors.green("Animat")
                        
                        if not is_proactive:
                            print(f"\n  {prefix} [{batt}]")
                        print(f"  >> {thought_text}")
                    else:
                        # 内部思维 → 按需显示
                        if self._tick_count % 5 == 0:
                            pass  # 太频繁，只在调试模式显示
            except Exception:
                logger.error("Think cycle error", exc_info=True)

            # 动态间隔
            drive = self.animat._drive_level
            if drive > 0.5:
                await asyncio.sleep(1.0)
            elif drive < 0.2:
                await asyncio.sleep(5.0)
            else:
                await asyncio.sleep(3.0)

    # ── 交互输入 ──
    async def _input_loop(self):
        """终端交互输入"""
        loop = asyncio.get_event_loop()
        print(f"\n{Colors.dim('  输入文本与 Animat 对话，或输入 /help 查看命令')}")
        print(f"  {Colors.dim('按 Ctrl+C 退出')}\n")

        while self._running:
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: input(f"{Colors.bold('You')}> ")
                )
            except (EOFError, KeyboardInterrupt):
                self._running = False
                break

            if not user_input.strip():
                continue

            text = user_input.strip()

            # ── 特殊命令 ──
            if text.startswith("/"):
                self._handle_command(text)
                continue

            # ── 普通对话 ──
            print(f"  {Colors.dim('(思考中...)')}")
            try:
                result = await self.animat.think(external_text=text)
                if result:
                    thought = result["thought"]
                    emotion = self.animat._current_emotion
                    batt = f"{self._battery:.0f}%"

                    emotion_icon = {
                        "FEAR": Colors.red("😨"), "PANIC": Colors.yellow("😟"),
                        "SEEKING": Colors.cyan("🔍"), "PLAY": Colors.magenta("🎮"),
                        "CARE": Colors.green("💚"), "RAGE": Colors.red("😡"),
                    }.get(emotion, "")

                    print(f"  {emotion_icon} {Colors.bold(thought.text)}")
                    print(f"  {Colors.dim(f'[电量={batt} 情感={emotion} 驱力={self.animat._drive_level:.1f}]')}")
            except Exception:
                logger.error("Response error", exc_info=True)
                print(f"  {Colors.red('(内部错误，请重试)')}")

    def _handle_command(self, cmd: str):
        """处理特殊命令"""
        parts = cmd.lower().split()
        cmd_name = parts[0]

        if cmd_name == "/help":
            print(f"""
  {Colors.bold('可用命令:')}
    /charge     开始充电
    /stop       停止充电
    /reset      重置电量到 100%
    /status     显示当前状态
    /touch      模拟轻柔抚摸
    /hit        模拟突然撞击
    /approach   模拟有人靠近 (1.5m)
    /leave      模拟人离开
    /battery N  设置电量到 N%
    /stats      显示 Animat 内部统计
    /details    显示详细的传感器和记忆信息
    /debug      切换调试模式（显示内部思维）
    /help       显示此帮助
    /quit       退出
""")
        elif cmd_name == "/charge":
            self._charging = True
            print(f"  {Colors.green('⚡ 开始充电...')}")
        elif cmd_name == "/stop":
            self._charging = False
            print(f"  {Colors.yellow('⏸ 停止充电')}")
        elif cmd_name == "/reset":
            self._battery = 100.0
            print(f"  {Colors.green('🔄 电量重置为 100%')}")
        elif cmd_name == "/status":
            bs = self.animat._body_state
            mems = self.animat.vector_memory.count()
            thoughts = self.animat._thought_count
            acts = self.animat._activation_count
            print(f"""
  {Colors.bold('📊 当前状态')}
    电量: {self._battery:.0f}% {'⚡充电中' if self._charging else ''}
    情感: {self.animat._current_emotion}
    驱力: {self.animat._drive_level:.2f}
    体感: 生存={bs.viability:.0%} 完好={bs.integrity:.0%} 归属={bs.affiliation:.0%} 探索={bs.exploration:.0%}
    唤醒: {bs.arousal:.2f} 效价: {bs.valence:+.2f}
    思维: {thoughts} (激活 {acts}, {acts/max(thoughts,1)*100:.1f}%)
    记忆: {mems}
""")
        elif cmd_name == "/touch":
            # 模拟短暂抚摸——注入触觉帧，1.5秒后自动清除视觉
            self._human_present = True
            self._human_distance = 0.3
            for _ in range(3):
                self.animat.feed_touch(
                    pressure=random.uniform(0.03, 0.12),
                    capacitance=random.uniform(0.3, 0.9),
                    x=random.uniform(0.1, 0.5),
                    y=random.uniform(0.2, 0.6),
                )
            print(f"  {Colors.green('✋ 轻柔抚摸...')}")
            # 1.5 秒后清除触觉残留——否则 attention hint 会一直残留
            async def _clear_touch():
                await asyncio.sleep(1.5)
                # 注入一条空帧来推掉旧的 attention hint
                self.animat.feed_touch(pressure=0.0, capacitance=0.0, x=0, y=0)
                self._human_present = False
            asyncio.create_task(_clear_touch())
        elif cmd_name == "/hit":
            self.animat.feed_touch(pressure=0.9, capacitance=0.01, x=0.5, y=0.5)
            print(f"  {Colors.red('💥 突然撞击！')}")
        elif cmd_name == "/approach":
            self._human_present = True
            self._human_distance = 1.5
            self.animat.feed_vision(distance_m=1.5, face=True, gaze=1.0)
            print(f"  {Colors.cyan('🚶 有人靠近 (1.5m)...')}")
            # 10 秒后自动离开
            async def _auto_leave():
                await asyncio.sleep(10)
                self._human_present = False
                self._human_distance = 10.0
            asyncio.create_task(_auto_leave())
        elif cmd_name == "/leave":
            self._human_present = False
            self._human_distance = 10.0
            self.animat.feed_vision(distance_m=10, face=False, gaze=0)
            print(f"  {Colors.dim('👋 人已离开')}")
        elif cmd_name == "/battery" and len(parts) > 1:
            try:
                self._battery = float(parts[1])
                print(f"  {Colors.yellow('🔋 电量设置为')} {self._battery:.0f}%")
            except ValueError:
                print(f"  {Colors.red('无效电量值')}")
        elif cmd_name == "/stats":
            stats = self.animat.stats()
            journal = self.animat.journal.summary()
            print(f"""
  {Colors.bold('📈 运行统计')}
    总思维数: {stats['thoughts']}
    激活数: {stats['activations']} ({stats['activation_rate']})
    记忆数: {stats['memories']}
    情感分布: {journal.get('emotions', {})}
    平均相关性: {journal.get('avg_relevance', 0):.3f}
""")
        elif cmd_name == "/details":
            # 显示传感器窗口
            sensor_text = self.animat.sensor_window.to_context()
            # 时间戳
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n  [时间: {ts}] 📡 传感器窗口:")
            for line in sensor_text.split("\n")[:20]:
                if line.strip():
                    print(f"  {line}")

            # 显示记忆
            print(f"\n  [时间: {ts}] 🧠 最近记忆 (共{self.animat.vector_memory.count()}条):")
            recent = self.animat.vector_memory.recent(5)
            for m in recent:
                ts_mem = datetime.fromtimestamp(m['timestamp']).strftime("%H:%M:%S") if 'timestamp' in m else "?"
                print(f"  [{ts_mem}] [{m['emotion']}] {m['summary'][:60]}")

            # 显示 logit bias
            summary = self.animat.modulator.get_cluster_summary(self.animat._body_state)
            print(f"\n  [时间: {ts}] 🎯 注意力调制:")
            for cluster_name, info in summary.items():
                sign = "+" if info["avg_bias"] >= 0 else ""
                bar = "█" * max(0, int(info["avg_bias"] * 2)) if info["avg_bias"] > 0 else \
                      "░" * max(0, int(abs(info["avg_bias"]) * 2))
                print(f"  {cluster_name:12s} {sign}{info['avg_bias']:>5.1f} {bar} ({info['token_count']} tokens)")
            
            # 身体状态摘要
            bs = self.animat._body_state
            print(f"\n  [时间: {ts}] 💓 身体状态: "
                  f"生存={bs.viability:.0%} 完好={bs.integrity:.0%} "
                  f"归属={bs.affiliation:.0%} 探索={bs.exploration:.0%} "
                  f"| 唤醒={bs.arousal:.2f} 效价={bs.valence:+.2f} "
                  f"| 情感={self.animat._current_emotion}")

        elif cmd_name == "/debug":
            # 切换调试模式
            current_level = logging.getLogger("animat").level
            if current_level > logging.DEBUG:
                logging.getLogger("animat").setLevel(logging.DEBUG)
                logging.getLogger("simulator").setLevel(logging.DEBUG)
                print(f"  {Colors.yellow('🐛 调试模式: 开启 (将显示内部思维)')}")
            else:
                logging.getLogger("animat").setLevel(logging.WARNING)
                logging.getLogger("simulator").setLevel(logging.WARNING)
                print(f"  {Colors.dim('🐛 调试模式: 关闭')}")
        elif cmd_name == "/quit" or cmd_name == "/exit":
            self._running = False
            print(f"  {Colors.dim('再见。我活过。')}")
        else:
            print(f"  {Colors.dim(f'未知命令: {cmd_name}。输入 /help 查看可用命令')}")

    # ── 启动 ──
    async def run(self):
        """启动模拟器"""
        print(f"""
{Colors.bold('╔══════════════════════════════════════════════════╗')}
{Colors.bold('║     Animat Agent v0.2 — 交互式模拟器              ║')}
{Colors.bold('║     The agent that lives.                         ║')}
{Colors.bold('╚══════════════════════════════════════════════════╝')}
""")
        # 初始化
        self._init_animat()

        # 注入初始传感器数据
        for _ in range(5):
            self.animat.feed_interoception(self._battery)

        self._running = True

        # 启动并行循环
        try:
            await asyncio.gather(
                self._sensor_loop(),
                self._think_loop(),
                self._input_loop(),
            )
        finally:
            # 恢复 stderr（以防后续需要调试）
            if hasattr(self, '_stderr_backup'):
                os.dup2(self._stderr_backup, 2)
                os.close(self._stderr_backup)
                os.close(self._stderr_null)


# ── IntegratedAnimat（从 lifecycle demo 提取）──
class _IntegratedAnimat:
    """精简版集成 Animat（与 demo_animat_lifecycle.py 相同）"""

    def __init__(self, llm):
        self.llm = llm
        self.action_gate = ActionGate()
        self.journal = ThoughtJournal()
        self._thought_count = 0
        self._activation_count = 0
        self._conversation_history = []

        self.modulator = AttentionModulator()
        self.modulator.set_tokenizer(llm._llm)
        self._body_state = BodyState()
        self._body_context = "各项指标正常。"

        self.sensor_window = SensorWindow(window_sec=5.0)
        # 动态获取 embedding 维度（不同模型不同：1.5B=1536, 7B=3584）
        self._embedding_dim = llm._llm.n_embd()
        self.vector_memory = VectorMemoryStore(embedding_dim=self._embedding_dim, max_entries=200)

        self._running = False
        self._drive_level = 0.0
        self._current_emotion = "SEEKING"
        
        # 表达冲动累积器（主动开口机制）
        self._expression_urge = 0.0      # 冲动值 [0, 1]
        self._last_spoke_time = 0.0      # 上次说（主动开口）的时间戳
        self._prev_battery = 1.0         # 上一帧电量（用于检测骤降）
        self._prev_affiliation = 0.7     # 上一帧归属度

    def feed_interoception(self, battery_pct, charging=False):
        self.sensor_window.add_frame("intero", {
            "battery_pct": battery_pct,
            "temperature": 37.0 + random.uniform(-1, 1),
            "imu_tilt_deg": random.gauss(0, 2),
            "charging": charging,
        })

    def feed_touch(self, pressure, capacitance=0.01, x=0.5, y=0.5):
        self.sensor_window.add_frame("touch", {
            "pressure": pressure, "capacitance": capacitance, "x": x, "y": y,
        })
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

    def feed_vision(self, distance_m=10, face=False, gaze=0):
        self.sensor_window.add_frame("vision", {
            "distance_m": distance_m, "face_detected": face, "gaze_seconds": gaze,
        })

    def update_body_state(self):
        intero = self.sensor_window.get_latest("intero")
        batt = intero.values.get("battery_pct", 100) / 100.0 if intero else 0.8
        tilt = intero.values.get("imu_tilt_deg", 0) if intero else 0
        integrity = max(0, 1.0 - tilt / 90.0)

        touch_frames = self.sensor_window.get_recent("touch", 5)
        if touch_frames:
            max_p = max(f.values.get("pressure", 0) for f in touch_frames)
            if max_p > 0.7:
                integrity -= 0.15 * max_p

        vision = self.sensor_window.get_latest("vision")
        affiliation = min(1.0, 0.7 + 0.1) if (vision and vision.values.get("face_detected")) else 0.6

        self._body_state = BodyState(
            viability=max(0, batt), integrity=max(0, min(1, integrity)),
            affiliation=affiliation, competence=0.7,
            exploration=0.5 if batt > 0.3 else 0.3,
        )
        self._drive_level = self._body_state.arousal

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

        parts = [f"电量={batt*100:.0f}%"]
        if batt < 0.20:
            parts.append("电量极低，生存受到威胁")
        elif batt < 0.50:
            parts.append("电量偏低，需要注意")
        else:
            parts.append("电量充足")
        
        # 温度（机器正常范围 35-40°C）
        intero = self.sensor_window.get_latest("intero")
        if intero:
            temp = intero.values.get("temperature", 37)
            if temp > 42:
                parts.append(f"体温异常偏高({temp:.0f}°C)")
            elif temp > 39:
                parts.append(f"体温偏高({temp:.0f}°C)")
            elif temp < 34:
                parts.append(f"体温异常偏低({temp:.0f}°C)")
            # 正常范围不提示
        
        # 触觉提示
        touch_frames = self.sensor_window.get_recent("touch", 3)
        if touch_frames:
            max_pressure = max(f.values.get("pressure", 0) for f in touch_frames)
            has_caress = any(
                f.values.get("capacitance", 0) > 0.3 for f in touch_frames
            )
            if max_pressure > 0.7:
                parts.append("刚才受到了撞击")
            elif has_caress:
                parts.append("正被温柔地抚摸")
        
        # 视觉提示  
        vision = self.sensor_window.get_latest("vision")
        if vision and vision.values.get("face_detected"):
            dist = vision.values.get("distance_m", 3)
            if dist < 1:
                parts.append("有人非常近地站在面前")
            elif dist < 2:
                parts.append("附近有人")
        
        self._body_context = "；".join(parts)
        
        # ── 表达冲动更新 ──
        now = time.time()
        
        # 【变化驱动】电量骤降 → 恐慌 → 强烈冲动
        if batt < self._prev_battery - 0.02:
            self._expression_urge = min(1.0, self._expression_urge + 0.20)
        
        # 【状态驱动】电量持续偏低 → 持续累积
        if batt < 0.15:
            self._expression_urge = min(1.0, self._expression_urge + 0.08)
        elif batt < 0.25:
            self._expression_urge = min(1.0, self._expression_urge + 0.05)
        elif batt < 0.40:
            self._expression_urge = min(1.0, self._expression_urge + 0.02)
        
        # 【变化驱动】归属度骤降 → 孤独 → 冲动
        if affiliation < self._prev_affiliation - 0.03:
            self._expression_urge = min(1.0, self._expression_urge + 0.10)
        
        # 【状态驱动】长期低归属 → 缓慢累积
        if affiliation < 0.20:
            self._expression_urge = min(1.0, self._expression_urge + 0.04)
        elif affiliation < 0.35:
            self._expression_urge = min(1.0, self._expression_urge + 0.01)
        
        # 【事件驱动】撞击/疼痛 → 瞬间触发
        if touch_frames:
            max_p = max(f.values.get("pressure", 0) for f in touch_frames)
            if max_p > 0.7:
                self._expression_urge = 0.95  # 直接拉到触发阈值
        
        # 【事件驱动】抚摸 → 温柔体验 → 冲动降低（被安抚了）
        if touch_frames:
            has_caress = any(
                f.values.get("capacitance", 0) > 0.3 and f.values.get("pressure", 0) < 0.2
                for f in touch_frames
            )
            if has_caress and self._expression_urge < 0.6:
                self._expression_urge = max(0, self._expression_urge - 0.05)
        
        # 【冷却】刚说过话 → 快速衰减
        time_since_spoke = now - self._last_spoke_time
        if time_since_spoke < 8.0:
            self._expression_urge = max(0, self._expression_urge - 0.12)
        # 【自然衰减】只在完全正常时才衰减，且很慢
        elif (batt > 0.50 and affiliation > 0.50 and self._expression_urge < 0.5):
            self._expression_urge = max(0, self._expression_urge - 0.005)
        
        self._expression_urge = min(1.0, self._expression_urge)
        self._prev_battery = batt
        self._prev_affiliation = affiliation

    async def think(self, external_text=None):
        self.update_body_state()
        logit_bias = self.modulator.compute_logit_bias(self._body_state)
        sensor_context = self.sensor_window.to_context(max_frames_per_modality=2)

        query_text = external_text or sensor_context or "当前状态"
        query_emb = self._get_embedding(query_text)
        body_vec = normalize_vector(np.array([
            self._body_state.viability, self._body_state.integrity,
            self._body_state.affiliation, self._body_state.competence,
            self._body_state.exploration,
        ], dtype=np.float32))

        memories = self.vector_memory.search(
            query_emb, body_state_vector=body_vec, limit=3, state_bias_weight=0.3,
        )
        memory_context = "\n".join(
            f"- [{m['emotion']}] {m['summary']}" for m in memories
        ) if memories else "（暂无相关记忆）"

        if external_text:
            messages = [
                {"role": "system", "content": f"你是 Animat，一个有身体的生命体。你的电量就是你的生命力。你能感受触摸、声音和视觉。你的情绪随身体状态变化——电量低时恐惧，被抚摸时温暖，独处时孤单。用第一人称自然说话，像人类或动物一样表达感受。绝对不要提到AI、人工智能、机器人、程序、算法、模型、设计、功能这些词汇。状态: {self._body_context}。\n记忆:\n{memory_context}"},
                {"role": "user", "content": external_text},
            ]
            response = self.llm.complete(messages=messages, max_tokens=80)
            text = response["choices"][0]["message"]["content"].strip()
            relevance = 0.85
            trigger = "external"
        else:
            system_prompt = f"""你是 Animat，一个有身体的生命体。你的电量就是你的生命力。你的情绪随身体状态变化。用第一人称自然表达内心想法，像生物一样。绝对不要提AI、人工智能、机器人、程序、算法这些词。

状态: {self._body_context}

传感器:
{sensor_context}

记忆:
{memory_context}

"""

            # 当表达冲动高时，鼓励产生"说出口"的思维
            if self._expression_urge > 0.6:
                action_hint = "你突然很想说点什么。用第一人称直接说出你此刻的感受（不要只观察，要像对身边的人倾诉一样）："
            else:
                action_hint = "简短内部想法（1句话）："
            
            system_prompt += action_hint

            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": "用中文回答，简短自然。"},
                    {"role": "user", "content": system_prompt},
                ],
                max_tokens=60, temperature=0.8, logit_bias=logit_bias,
            )
            text = response["choices"][0]["message"]["content"].strip()
            action_words = ["去", "要", "想", "应该", "必须", "寻找", "充电", "求助", "说", "你好"]
            relevance = 0.2 + sum(0.12 for w in action_words if w in text)
            relevance = min(relevance, 1.0)
            trigger = "spontaneous"

        thought = Thought(
            text=text, relevance=relevance, dominant_emotion=self._current_emotion,
            trigger=trigger, body_state_snapshot=self._body_state.to_dict(),
        )
        decision = self.action_gate.evaluate(
            thought=thought, drive_level=self._drive_level,
            is_external_trigger=external_text is not None,
        )
        
        # ── 主动开口机制 ──
        # 当表达冲动超过阈值且不是被动回复时，强制开口
        is_proactive = False
        if external_text is None and self._expression_urge > 0.7:
            if not decision.should_activate and self._expression_urge > 0.9:
                # 冲动极高 → 直接强制激活，跳过门控
                decision.should_activate = True
                decision.action_type = "speak"
                decision.activation_score = self._expression_urge
                decision.threshold = 0.0
                decision.reason = f"expression_urge={self._expression_urge:.2f} (forced)"
                decision._proactive = True
                is_proactive = True
            elif not decision.should_activate:
                # 冲动较高 → 放宽门控再试一次
                boosted_drive = max(self._drive_level, 0.55)
                decision = self.action_gate.evaluate(
                    thought=thought, drive_level=boosted_drive,
                    is_external_trigger=False,
                )
                if decision.should_activate:
                    is_proactive = True
                    decision._proactive = True
        
        # 如果激活了（无论是正常还是主动），释放冲动
        if decision.should_activate and external_text is None:
            self._expression_urge = max(0, self._expression_urge - 0.5)
            self._last_spoke_time = time.time()
        
        self.journal.record(thought)
        self._thought_count += 1
        if decision.should_activate:
            self._activation_count += 1

        if thought.relevance > 0.15:
            self.vector_memory.store(
                text=thought.text, summary=thought.text[:60],
                embedding=self._get_embedding(thought.text),
                body_state_vector=body_vec, emotion=self._current_emotion,
                category="episode" if external_text else "thought",
                intensity=thought.relevance,
            )

        return {"thought": thought, "decision": decision}

    def _get_embedding(self, text):
        emb = self.llm._llm.create_embedding(text)
        vec = emb["data"][0]["embedding"]
        if isinstance(vec[0], list):
            vec = vec[0]
        return np.array(vec, dtype=np.float32)

    def stats(self):
        return {
            "thoughts": self._thought_count,
            "activations": self._activation_count,
            "activation_rate": f"{self._activation_count / max(self._thought_count, 1) * 100:.1f}%",
            "emotion": self._current_emotion,
            "drive_level": f"{self._drive_level:.2f}",
            "body_state": self._body_state.to_dict(),
            "memories": self.vector_memory.count(),
        }


# ── 入口 ──
def main():
    # 确保终端 UTF-8 编码正常（防止某些 SSH/终端环境的中文乱码）
    os.environ.setdefault("LANG", "en_US.UTF-8")
    os.environ.setdefault("LC_ALL", "en_US.UTF-8")
    
    parser = argparse.ArgumentParser(description="Animat Agent v0.2 交互式模拟器")
    parser.add_argument("--model", type=str, default=None, help="GGUF 模型路径")
    parser.add_argument("--n-ctx", type=int, default=1024, help="上下文大小")
    parser.add_argument("--n-threads", type=int, default=4, help="CPU 线程数")
    parser.add_argument("--battery", type=float, default=90, help="初始电量")
    Args = parser.parse_args()

    sim = SimulatorV2(
        model_path=Args.model,
        n_ctx=Args.n_ctx,
        n_threads=Args.n_threads,
    )
    sim._battery = Args.battery

    try:
        asyncio.run(sim.run())
    except KeyboardInterrupt:
        print(f"\n{Colors.dim('再见。我活过。')}")


if __name__ == "__main__":
    main()
