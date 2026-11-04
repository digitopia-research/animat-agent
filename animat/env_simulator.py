"""
Animat 环境世界模拟器 — 持续向 Animat 注入有语义的外界刺激。

与 simulator.py 的分工:
  simulator.py      调试工具：手动刺激（/touch /hit）、电量控制、动作回显
  env_simulator.py  世界本身：昼夜节律 + 随机事件，让 Animat"活在有变化的世界里"

为什么需要它:
  只有单一输入、且外部事件停止时，意识流会固着在同一件事上
  （实测问过一次天气后，同一份数据被咀嚼了 8 小时）。
  真实生物靠外界不断变化来打断固着——这个进程就是在提供那种变化。

设计要点:
  - 复用 animat/sensor/* 四个既有话题，perception 侧零改动
  - 事件带语义字段（event/ambient_light 等），模型能直接读懂"发生了什么"
  - 内感有因果：活动 → 耗电/升温，而不是无意义的随机抖动
  - 随机噪声会诱发幻觉（曾把 37°C 编成"系统过热"），故只发有语义的事件

用法:
  uv run python -m animat.env_simulator
  ANIMAT_ENV_SEED=42 uv run python -m animat.env_simulator   # 固定随机种子（可复现）

交互命令:
  /help     显示帮助
  /status   当前环境状态
  /person   有人经过
  /touch    被轻轻碰了一下
  /light    光线突变
  /noise    环境异响
  /quiet    全部事件暂停
  /lively   恢复事件
  /scene N  切换场景密度: quiet / normal / lively
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import aiomqtt

from animat.config import Topic

logger = logging.getLogger("animat.env_simulator")


# ─────────────────── 场景密度 ───────────────────
# 事件间隔的基准（秒）。真实世界的刺激是稀疏的：
# 太密会变成噪声（模型转而解释噪声，产生幻觉），太疏则无法打断固着。
SCENES: dict[str, dict] = {
    "quiet":  {"event_gap": (420, 900), "label": "安静（仓库）"},
    "normal": {"event_gap": (150, 300), "label": "日常（室内）"},
    "lively": {"event_gap": (40, 90),   "label": "热闹（集市）"},
}


# ─────────────────── 昼夜节律 ───────────────────
@dataclass
class CircadianState:
    """当前时刻的外部世界该是什么样子。"""

    hour: int
    ambient_light: float      # 0.0 全黑 ~ 1.0 明亮
    ambient_db: float         # 环境底噪
    activity: float           # 人类活动强度 0.0~1.0
    phase: str                # 深夜 / 清晨 / 白天 / 傍晚

    def describe(self) -> str:
        return f"{self.phase}(光照{self.ambient_light:.1f}, 底噪{self.ambient_db:.0f}dB, 活跃{self.activity:.1f})"


def circadian_now(now: Optional[datetime] = None) -> CircadianState:
    """根据真实钟点推算环境状态。

    用连续函数而非分段常数：光照在日出日落期间平滑变化，
    这样 Animat 能看到"天在慢慢变亮"，而不是某一刻突然跳变。
    """
    now = now or datetime.now()
    h = now.hour + now.minute / 60.0

    # 光照：6 点日出、18 点日落，用余弦平滑
    if 6 <= h < 18:
        light = 0.5 - 0.5 * math.cos(math.pi * (h - 6) / 12)
        light = 0.15 + 0.85 * light
    else:
        # 夜间：深夜最低，靠近清晨/傍晚略高
        dist = min(abs(h - 18), abs(h - 6 + 24) if h >= 18 else abs(h + 24 - 6))
        light = max(0.02, 0.15 - 0.02 * dist)

    # 人类活动强度：白天高，深夜接近零
    if 7 <= h < 9:
        activity = 0.4 + 0.3 * (h - 7) / 2
    elif 9 <= h < 18:
        activity = 0.7 + 0.15 * math.sin((h - 9) / 9 * math.pi)
    elif 18 <= h < 22:
        activity = 0.6 - 0.25 * (h - 18) / 4
    else:
        activity = 0.05

    # 底噪跟着活动走
    ambient_db = 28.0 + 22.0 * activity

    if h < 6:
        phase = "深夜"
    elif h < 9:
        phase = "清晨"
    elif h < 18:
        phase = "白天"
    elif h < 22:
        phase = "傍晚"
    else:
        phase = "深夜"

    return CircadianState(
        hour=now.hour,
        ambient_light=round(light, 2),
        ambient_db=round(ambient_db, 1),
        activity=round(activity, 2),
        phase=phase,
    )


# ─────────────────── 事件定义 ───────────────────
@dataclass
class EnvEvent:
    """一次外界事件。"""

    name: str
    topic: str
    payload: dict
    description: str
    # 事件持续时间（秒），用于倾角等需要回正的量
    duration: float = 0.0


class EnvironmentSimulator:
    """环境世界：昼夜节律 + 随机事件注入。"""

    def __init__(self) -> None:
        self.host = os.getenv("MQTT_BROKER_HOST", "localhost")
        self.port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
        self.alive = int(os.getenv("MQTT_KEEPALIVE", "60"))

        seed = os.getenv("ANIMAT_ENV_SEED")
        self._rng = random.Random(int(seed)) if seed else random.Random()

        scene = os.getenv("ANIMAT_ENV_SCENE", "normal")
        self._scene = scene if scene in SCENES else "normal"
        self._events_enabled = os.getenv("ANIMAT_ENV_EVENTS", "1") != "0"

        self._running = False
        self._client: Optional[aiomqtt.Client] = None

        # 内感状态（带因果）
        self._battery = float(os.getenv("ANIMAT_ENV_BATTERY", "100"))
        self._temperature = 37.0
        self._tilt = 0.0
        self._charging = False
        self._load = 0.0          # 最近的"认知负载"，驱动耗电与升温

        # 当前环境快照
        self._circadian = circadian_now()
        self._last_event = "（无）"
        self._event_count = 0

    # ── 事件工厂 ──
    def _make_person_approach(self) -> EnvEvent:
        dist = round(self._rng.uniform(0.8, 3.0), 2)
        return EnvEvent(
            name="person_approach",
            topic=Topic.SENSOR_VISION,
            payload={
                "distance_m": dist,
                "face_detected": True,
                "gaze_seconds": round(self._rng.uniform(1.0, 4.0), 1),
                "ambient_light": self._circadian.ambient_light,
                "event": "person_approach",
                "event_text": f"有人靠近到 {dist} 米并看向你",
            },
            description=f"有人靠近 {dist}m",
        )

    def _make_person_leave(self) -> EnvEvent:
        return EnvEvent(
            name="person_leave",
            topic=Topic.SENSOR_VISION,
            payload={
                "distance_m": 10.0,
                "face_detected": False,
                "gaze_seconds": 0,
                "ambient_light": self._circadian.ambient_light,
                "event": "person_leave",
                "event_text": "刚才那个人走开了",
            },
            description="人离开了",
        )

    def _make_gentle_touch(self) -> EnvEvent:
        return EnvEvent(
            name="gentle_touch",
            topic=Topic.SENSOR_TOUCH,
            payload={
                "pressure": round(self._rng.uniform(0.05, 0.15), 3),
                "capacitance": round(self._rng.uniform(0.3, 0.8), 3),
                "x": round(self._rng.uniform(0.1, 0.9), 2),
                "y": round(self._rng.uniform(0.1, 0.9), 2),
                "event": "gentle_touch",
                "event_text": "有人轻轻碰了碰你",
            },
            description="被轻轻触碰",
        )

    def _make_light_shift(self) -> EnvEvent:
        delta = self._rng.choice([-0.35, -0.2, 0.25, 0.4])
        new_light = max(0.02, min(1.0, self._circadian.ambient_light + delta))
        word = "变亮" if delta > 0 else "变暗"
        return EnvEvent(
            name="light_shift",
            topic=Topic.SENSOR_VISION,
            payload={
                "distance_m": 10.0,
                "face_detected": False,
                "gaze_seconds": 0,
                "ambient_light": round(new_light, 2),
                "event": "light_shift",
                "event_text": f"周围光线突然{word}",
            },
            description=f"光线{word}",
        )

    def _make_ambient_noise(self) -> EnvEvent:
        db = round(self._rng.uniform(45, 70), 1)
        freq = round(self._rng.uniform(150, 900), 0)
        return EnvEvent(
            name="ambient_noise",
            topic=Topic.SENSOR_AUDIO,
            payload={
                "volume_db": db,
                "fundamental_freq": freq,
                "event": "ambient_noise",
                "event_text": f"远处传来一阵声响（{db:.0f}dB）",
            },
            description=f"环境异响 {db:.0f}dB",
        )

    def _make_gentle_bump(self) -> EnvEvent:
        return EnvEvent(
            name="gentle_bump",
            topic=Topic.SENSOR_INTERO,
            payload={
                "battery_pct": round(self._battery, 1),
                "temperature": round(self._temperature, 1),
                "imu_tilt_deg": round(self._rng.choice([-1, 1]) * self._rng.uniform(2.0, 4.0), 1),
                "charging": self._charging,
                "event": "gentle_bump",
                "event_text": "身体被轻碰了一下，晃了晃",
            },
            description="身体被碰",
            duration=2.0,
        )

    # ── 事件调度 ──
    def _event_pool(self) -> list[tuple[float, Callable[[], EnvEvent]]]:
        """按当前情境给出候选事件及权重。

        深夜人少，靠近事件的权重要降低——世界是自洽的，
        不是均匀随机的，否则凌晨三点老是有人凑过来很出戏。
        """
        act = self._circadian.activity
        pool: list[tuple[float, Callable[[], EnvEvent]]] = [
            (0.35, self._make_ambient_noise),
            (0.20, self._make_light_shift),
            (0.15, self._make_gentle_bump),
        ]
        # 人类相关事件随时段缩放
        pool.append((0.30 * act + 0.03, self._make_person_approach))
        pool.append((0.12 * act + 0.01, self._make_gentle_touch))
        return pool

    def _pick_event(self) -> EnvEvent:
        pool = self._event_pool()
        total = sum(w for w, _ in pool)
        r = self._rng.uniform(0, total)
        acc = 0.0
        for weight, factory in pool:
            acc += weight
            if r <= acc:
                return factory()
        return pool[0][1]()

    async def _publish(self, topic: str, payload: dict, qos: int = 1) -> None:
        if not self._client:
            return
        try:
            await self._client.publish(topic, json.dumps(payload).encode(), qos=qos)
        except Exception as e:
            logger.warning("发布失败 %s: %s", topic, e)

    async def _emit_event(self, event: EnvEvent) -> None:
        await self._publish(event.topic, event.payload, qos=1)
        self._event_count += 1
        self._last_event = f"{event.description} @ {datetime.now():%H:%M:%S}"
        logger.info("🌍 事件: %s -> %s", event.description, event.topic)

        # 碰触类事件：短暂改变倾角，然后回正
        if event.duration > 0:
            tilt = event.payload.get("imu_tilt_deg", 0)
            self._tilt = tilt
            asyncio.create_task(self._restore_tilt_after(event.duration))

    async def _restore_tilt_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self._tilt = 0.0

    # ── 主循环 ──
    async def _interoception_loop(self) -> None:
        """内感循环：电量/体温/倾角，全部带因果。"""
        while self._running:
            circ = circadian_now()
            self._circadian = circ

            if self._charging:
                self._battery = min(100.0, self._battery + 3.0)
                self._temperature = min(45.0, self._temperature + 0.05)
                if self._battery >= 100:
                    self._charging = False
                    logger.info("🔋 电量已满，停止充电")
            else:
                # 基础放电 + 认知负载额外消耗：思考越活跃，掉电越快
                drain = 0.05 + 0.15 * self._load
                self._battery = max(0.0, self._battery - drain)
                # 体温：负载升高，空闲回落。范围控制在 36.8~38.0，
                # 超出正常区间会让模型编造"过热"叙事。
                target = 37.0 + 0.8 * self._load
                self._temperature += (target - self._temperature) * 0.25

            # 负载自然衰减（没有新思维时会回落）
            self._load = max(0.0, self._load - 0.08)

            await self._publish(Topic.SENSOR_INTERO, {
                "battery_pct": round(self._battery, 1),
                "temperature": round(self._temperature, 1),
                "imu_tilt_deg": round(self._tilt, 1),
                "charging": self._charging,
                "ambient_light": circ.ambient_light,
                "ambient_db": circ.ambient_db,
                "event_text": circ.describe(),
            }, qos=0)

            await asyncio.sleep(2)

    async def _event_loop(self) -> None:
        """事件循环：按场景密度随机注入。"""
        while self._running:
            lo, hi = SCENES[self._scene]["event_gap"]
            # 深夜事件更稀疏
            scale = 1.0 if self._circadian.activity > 0.3 else 2.2
            gap = self._rng.uniform(lo, hi) * scale
            await asyncio.sleep(gap)
            if not self._running or not self._events_enabled:
                continue
            try:
                await self._emit_event(self._pick_event())
            except Exception as e:
                logger.warning("事件注入失败: %s", e)

    async def _listen_motor(self, client: aiomqtt.Client) -> None:
        """监听 Animat 的动作，用来驱动"认知负载"。

        这是内感因果链的关键：它说话/移动时电量掉得快、体温微升，
        于是"我很累/我有点热"就不是凭空编的，而是有其物理前因。
        """
        async for message in client.messages:
            topic = str(message.topic)
            if topic == Topic.MOTOR:
                try:
                    payload = json.loads(message.payload.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                cmd = payload.get("command")
                if cmd == "speak":
                    self._load = min(1.0, self._load + 0.35)
                elif cmd == "move":
                    self._load = min(1.0, self._load + 0.5)
                if cmd == "move":
                    direction = (payload.get("params") or {}).get("direction", "")
                    if direction == "charger":
                        self._charging = True
            elif topic == Topic.CORTEX_THOUGHT:
                # 每产生一次思维就计入负载
                self._load = min(1.0, self._load + 0.12)

    async def _command_loop(self) -> None:
        """交互命令。"""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, input, "env> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not await self._handle_command(line.strip()):
                break

    async def _handle_command(self, line: str) -> bool:
        if not line:
            return True
        cmd, _, arg = line.partition(" ")

        if cmd in ("/quit", "/exit", "/q"):
            self._running = False
            return False
        if cmd == "/help":
            print(__doc__)
        elif cmd == "/status":
            print(f"  场景    : {SCENES[self._scene]['label']}")
            print(f"  事件开关: {'开' if self._events_enabled else '关'}")
            print(f"  环境    : {self._circadian.describe()}")
            print(f"  内感    : 电量{self._battery:.1f}% 体温{self._temperature:.1f}°C "
                  f"倾角{self._tilt:.1f}° 负载{self._load:.2f}")
            print(f"  已发事件: {self._event_count} 次，最近: {self._last_event}")
        elif cmd == "/person":
            await self._emit_event(self._make_person_approach())
        elif cmd == "/leave":
            await self._emit_event(self._make_person_leave())
        elif cmd == "/touch":
            await self._emit_event(self._make_gentle_touch())
        elif cmd == "/light":
            await self._emit_event(self._make_light_shift())
        elif cmd == "/noise":
            await self._emit_event(self._make_ambient_noise())
        elif cmd == "/bump":
            await self._emit_event(self._make_gentle_bump())
        elif cmd == "/quiet":
            self._events_enabled = False
            print("  事件已暂停（内感仍在上报）")
        elif cmd == "/lively":
            self._events_enabled = True
            print("  事件已恢复")
        elif cmd == "/scene":
            if arg in SCENES:
                self._scene = arg
                print(f"  场景 -> {SCENES[arg]['label']}")
            else:
                print(f"  可用场景: {', '.join(SCENES)}")
        else:
            print(f"  未知命令: {cmd}（/help 查看）")
        return True

    async def run(self) -> None:
        async with aiomqtt.Client(
            hostname=self.host, port=self.port, identifier="animat-env-sim",
        ) as client:
            self._client = client
            self._running = True
            logger.info("🌍 环境世界启动 — %s", SCENES[self._scene]["label"])
            logger.info("   当前环境: %s", self._circadian.describe())
            logger.info("   输入 /help 查看命令，/quiet 暂停事件")

            await client.subscribe(Topic.MOTOR, qos=1)
            await client.subscribe(Topic.CORTEX_THOUGHT, qos=0)

            await asyncio.gather(
                self._interoception_loop(),
                self._event_loop(),
                self._listen_motor(client),
                self._command_loop(),
            )


def main() -> None:
    logging.basicConfig(
        level=os.getenv("ANIMAT_ENV_LOGLEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    try:
        asyncio.run(EnvironmentSimulator().run())
    except KeyboardInterrupt:
        logger.info("环境世界已停止")


if __name__ == "__main__":
    main()
