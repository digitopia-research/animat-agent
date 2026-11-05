"""
Animat 环境世界模拟器 — 持续向 Animat 注入有语义的外界刺激。

与 bot_simulator 的分工:
  bot_simulator  身体：电量、体温、充电、倾角（body.html 控制）
  env_simulator  世界：昼夜节律 + 随机事件，让 Animat"活在有变化的世界里"

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
  uv run python -m animat.simulators.env_simulator
  ANIMAT_ENV_SEED=42 uv run python -m animat.simulators.env_simulator   # 固定种子（可复现）

控制方式（仅 MQTT，无终端交互）:
  本进程是后端服务，不读 stdin、不提供命令行。
  手动干预一律走 MQTT，浏览器打开 webui/world.html 即可：
    发布 animat/env/control  {"cmd": "person"}
    发布 animat/env/control  {"cmd": "scene", "arg": "lively"}
    发布 animat/env/control  {"cmd": "status"}
  可用 cmd: status / person / leave / touch / bump / light / noise
            / quiet / lively / scene(带 arg) / help
  执行结果与当前状态发布在 animat/env/status。
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

        # 是否有人在场，以及在场截止时刻与当时的距离。
        # person_approach 会打开这个状态，person_leave 或到期后关闭。
        self._person_until: float = 0.0
        self._person_distance: float = 1.5
        self._person_gaze: float = 1.0

        # 内感状态：只保留本模块拥有的部分。
        # 电量/体温/充电判定全部交给 bot_simulator（body.html），
        # 这里再存一份只会产生两个互相矛盾的"真相"。
        self._tilt = 0.0
        self._load = 0.0          # 认知负载，反映在放电叙事的强度上

        # 当前环境快照
        self._circadian = circadian_now()
        self._last_event = "（无）"
        self._event_count = 0

    # 本进程只做后端服务：不读 stdin、不提供终端命令行。
    # 所有手动干预走 MQTT（animat/env/control），见模块 docstring。

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
                # 只发倾角（这是"世界碰了身体"造成的物理扰动），
                # 不带 battery/temperature —— 那两个字段归 bot_simulator。
                "imu_tilt_deg": round(self._rng.choice([-1, 1]) * self._rng.uniform(2.0, 4.0), 1),
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

        # 人在场状态：靠近则开启、离开则关闭。持续期间由
        # _keep_person_present 每 2 秒补一帧，让"有人陪着"成为状态而非瞬间。
        if event.name == "person_approach":
            self._person_distance = event.payload.get("distance_m", 1.5)
            self._person_gaze = event.payload.get("gaze_seconds", 1.0)
            # 停留 20-40 秒，比"有人走了"事件的平均间隔短，
            # 这样既足够让归属度真正回升，又不会一直挂着不离开。
            self._person_until = time.time() + self._rng.uniform(20.0, 40.0)
        elif event.name == "person_leave":
            self._person_until = 0.0

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
        """环境侧的内感补充：只发环境量，不发身体数值。

        电量与体温的唯一权威来源是 bot_simulator（body.html 控制的那个）。
        早先这里也算电量，结果两个进程每 2 秒各写一次
        animat/sensor/interoception，数值在两种模型之间来回跳，
        在网页上完全无法判断某次掉电是认知负载算的还是人为设的。
        现在这里只发布昼夜相关的环境量（光照、底噪），
        让 Animat 在同一份内感里既知道"外面天黑了"也知道"身体还有电"。

        认知负载仍然保留：它影响的是放电速度的叙事，不再直接改数值。
        """
        while self._running:
            circ = circadian_now()
            self._circadian = circ

            # 负载自然衰减（没有新思维时会回落）
            self._load = max(0.0, self._load - 0.08)

            await self._publish(Topic.SENSOR_INTERO, {
                "ambient_light": circ.ambient_light,
                "ambient_db": circ.ambient_db,
                "event_text": circ.describe(),
                "load": round(self._load, 2),
            }, qos=0)

            await self._keep_person_present()

            await asyncio.sleep(2)

    async def _keep_person_present(self) -> None:
        """人在场期间持续刷新视觉帧。

        person_approach 只发一帧，之后视野就空了。而感知层的归属度恢复
        依赖感受事件持续出现，衰减却是每时每刻的——实测一次 approach
        (intensity 0.28) 只能补 1.12 点，2.2 秒就被 0.5 点/秒的衰减吃光，
        于是"有人来了"这件事在生理上几乎不留痕迹，模型依旧困在孤独里。

        现实里人走过来会待一会儿。这里用 `until` 时间戳表达这个状态，
        期间每 2 秒重复一帧 vision，让"有人在"成为一个持续事实而不是
        一次抖动。凝视秒数随在场时长增长，越待越有"被看见"的感觉。
        """
        if not self._person_until or time.time() > self._person_until:
            self._person_until = 0.0
            return

        await self._publish(Topic.SENSOR_VISION, {
            "distance_m": round(self._person_distance, 2),
            "face_detected": True,
            "gaze_seconds": round(min(6.0, self._person_gaze + 2.0), 1),
            "ambient_light": self._circadian.ambient_light,
            "event": "person_present",
            # 不写 event_text：这不是"发生了某事"，而是"状态仍在持续"，
            # 写成事件会让模型误以为每隔两秒又来了一个人。
        }, qos=0)

        self._person_gaze = min(6.0, self._person_gaze + 2.0)

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
                # 不再因为"去找充电器"就自行改充电状态：
                # 充电由 bot_simulator 根据同一条 motor 指令处理，
                # 这里插一脚会让两边状态不一致。
            elif topic == Topic.CORTEX_THOUGHT:
                # 每产生一次思维就计入负载
                self._load = min(1.0, self._load + 0.12)

    async def _handle_command(self, line: str, reply: Optional[Callable[[str], None]] = None) -> bool:
        """执行一条命令，把结果交给 reply 回调（默认打印）。

        命令逻辑本身与传输方式无关：MQTT 入口传一个 list.append 收集回执文字。
        """
        emit = reply or print

        if not line:
            return True
        cmd, _, arg = line.partition(" ")
        cmd = cmd.lstrip("/").lower()
        arg = arg.strip()

        if cmd in ("quit", "exit", "q"):
            self._running = False
            return False
        if cmd == "help":
            # 进程没有终端，help 只是给 MQTT 客户端的一份能力清单，
            # 因此不要再把整段模块 docstring 当作"帮助"发出去。
            emit("可用命令: " + " / ".join([
                "status", "person", "leave", "touch", "bump",
                "light", "noise", "quiet", "lively", "scene <quiet|normal|lively>",
            ]))
        elif cmd == "status":
            emit(self._status_text())
        elif cmd == "person":
            await self._emit_event(self._make_person_approach())
            emit(f"已注入：{self._last_event}")
        elif cmd == "leave":
            await self._emit_event(self._make_person_leave())
            emit(f"已注入：{self._last_event}")
        elif cmd == "touch":
            await self._emit_event(self._make_gentle_touch())
            emit(f"已注入：{self._last_event}")
        elif cmd == "light":
            await self._emit_event(self._make_light_shift())
            emit(f"已注入：{self._last_event}")
        elif cmd == "noise":
            await self._emit_event(self._make_ambient_noise())
            emit(f"已注入：{self._last_event}")
        elif cmd == "bump":
            await self._emit_event(self._make_gentle_bump())
            emit(f"已注入：{self._last_event}")
        elif cmd == "quiet":
            self._events_enabled = False
            emit("事件已暂停（内感仍在上报）")
        elif cmd == "lively":
            self._events_enabled = True
            emit("事件已恢复")
        elif cmd == "scene":
            if arg in SCENES:
                self._scene = arg
                emit(f"场景 -> {SCENES[arg]['label']}")
            else:
                emit(f"可用场景: {', '.join(SCENES)}")
        else:
            emit(f"未知命令: {cmd}（help 查看）")
        return True

    # ── 状态快照 ──
    def _status_dict(self) -> dict:
        """供 MQTT / 网页读取的完整状态。"""
        lo, hi = SCENES[self._scene]["event_gap"]
        return {
            "type": "status",
            "scene": self._scene,
            "scene_label": SCENES[self._scene]["label"],
            "scenes": {k: v["label"] for k, v in SCENES.items()},
            "event_gap": [lo, hi],
            "events_enabled": self._events_enabled,
            "phase": self._circadian.phase,
            "ambient_light": self._circadian.ambient_light,
            "ambient_db": self._circadian.ambient_db,
            "activity": self._circadian.activity,
            # 电量/体温/充电不在这里报：它们归 bot_simulator 管，
            # 本页只提供环境量与认知负载。
            "imu_tilt_deg": round(self._tilt, 1),
            "load": round(self._load, 2),
            "event_count": self._event_count,
            "last_event": self._last_event,
            "ts": time.time(),
        }

    def _status_text(self) -> str:
        lo, hi = SCENES[self._scene]["event_gap"]
        return (
            f"  场景    : {SCENES[self._scene]['label']}（间隔 {lo}-{hi}s）\n"
            f"  事件开关: {'开' if self._events_enabled else '关'}\n"
            f"  环境    : {self._circadian.describe()}\n"
            f"  倾角{self._tilt:.1f}° 认知负载{self._load:.2f}"
            f"（电量与体温见 body.html）\n"
            f"  已发事件: {self._event_count} 次，最近: {self._last_event}"
        )

    async def _publish_status(self) -> None:
        await self._publish(Topic.ENV_STATUS, self._status_dict(), qos=0)

    # ── MQTT 控制通道 ──
    async def _listen_control(self, client: aiomqtt.Client) -> None:
        """接收远程命令并回执。

        与终端命令行共用 _handle_command：只有输出通路不同。
        回执里带 ok/msg 两个字段，网页可以据此给出即时反馈，
        不必等下一次状态轮询。
        """
        async for message in client.messages:
            topic = str(message.topic)
            if topic != Topic.ENV_CONTROL:
                continue
            # 无条件记一行：排查"按钮没反应"时，第一个要确认的就是
            # 命令究竟有没有到达模拟器。放在解析之前，连载荷损坏也能看见。
            raw = message.payload.decode("utf-8", errors="replace")
            logger.info("📥 收到控制命令: %s", raw[:120])
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("忽略非法控制载荷（不是合法 JSON）: %r", raw[:80])
                continue
            if not isinstance(payload, dict):
                logger.warning("忽略非法控制载荷（应为对象）: %r", raw[:80])
                continue

            cmd = str(payload.get("cmd", "")).strip()
            arg = payload.get("arg", "")
            if not cmd:
                logger.warning("控制载荷缺少 cmd 字段: %r", raw[:80])
                continue

            lines: list[str] = []
            try:
                keep = await self._handle_command(
                    f"{cmd} {arg}".strip(), reply=lines.append
                )
            except Exception as e:
                logger.warning("命令执行失败 %s: %s", cmd, e)
                await self._publish(Topic.ENV_STATUS, {
                    "type": "ack", "cmd": cmd, "ok": False,
                    "msg": f"执行失败: {e}", "ts": time.time(),
                }, qos=0)
                continue

            logger.info("🌍 远程命令已执行: %s -> %s", cmd, " / ".join(lines) or "ok")
            await self._publish(Topic.ENV_STATUS, {
                "type": "ack",
                "cmd": cmd,
                "ok": True,
                "msg": " / ".join(lines) if lines else "ok",
                "ts": time.time(),
            }, qos=0)
            await self._publish_status()

            if not keep:            # quit / exit
                self._running = False
                return

    async def _status_loop(self) -> None:
        """周期性广播世界状态，供网页实时刷新。

        3 秒一次：足够让按钮反馈看起来即时，又不会给 broker 添负担。
        """
        while self._running:
            try:
                await self._publish_status()
            except Exception as e:
                logger.debug("状态广播失败: %s", e)
            await asyncio.sleep(3)

    async def run(self) -> None:
        async with aiomqtt.Client(
            hostname=self.host, port=self.port, identifier="animat-env-sim",
        ) as client:
            self._client = client
            self._running = True
            logger.info("🌍 环境世界启动 — %s", SCENES[self._scene]["label"])
            logger.info("   当前环境: %s", self._circadian.describe())
            logger.info("   控制通道: %s  （手动注入请用 webui/world.html）",
                        Topic.ENV_CONTROL)

            await client.subscribe(Topic.MOTOR, qos=1)
            await client.subscribe(Topic.CORTEX_THOUGHT, qos=0)
            await client.subscribe(Topic.ENV_CONTROL, qos=1)

            # 启动即广播一次，网页一连上就能看到状态，不必等 3 秒
            await self._publish_status()

            tasks = [
                asyncio.create_task(self._interoception_loop()),
                asyncio.create_task(self._event_loop()),
                asyncio.create_task(self._status_loop()),
                asyncio.create_task(self._listen_control(client)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                for t in tasks:
                    t.cancel()


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
