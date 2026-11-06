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
from animat.spatial import SpatialState

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

        # 机器人在房间里的真实位置。与 3D 视图共用同一套坐标，
        # 前端按这份数据摆放物体和机器人，模型按这份数据算距离，
        # 所以"它说走向书架"和"画面上真的走过去"是同一件事。
        self._space = SpatialState()
        # 行走序号：每次 move_to 自增，旧的行走循环看到序号变了就退出，
        # 于是"半路改主意去别处"能立刻生效而不是两条路径同时改坐标。
        self._walk_seq = 0

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

    async def _on_motor(self, message) -> None:
        """处理一条 motor 指令。

        aiomqtt 的 client.messages 是单一异步流，同一 client 上开两个
        `async for` 会互相抢消息——后起的那个会一直收不到东西。
        因此这里不再自己开循环，而是由 _listen_control 那个唯一消费者
        按话题转发进来。

        历史教训：这个函数原来是 `_listen_motor(client)`，自带
        `async for`，但 run() 的 tasks 列表里漏了它——于是它从上线起
        就没跑过。表现是"模型在调 move_to，3D 里人一动不动"，
        而日志里连一条错误都没有。
        """
        topic = str(message.topic)
        if topic == Topic.MOTOR:
            try:
                payload = json.loads(message.payload.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            cmd = payload.get("command")
            # 每条指令单独兜异常：_apply_* 抛一次不该让整个 MQTT
            # 消费循环静默结束（那样外部只会看到"服务还活着、
            # 状态还在发，但从此不响应任何动作指令"）。
            try:
                if cmd == "speak":
                    self._load = min(1.0, self._load + 0.35)
                elif cmd in ("move", "move_to"):
                    self._load = min(1.0, self._load + 0.5)
                    # 机器人真的移动了：更新空间坐标，并把"走到了哪、
                    # 现在离什么更近了"作为一帧视觉反馈发回去。
                    await self._apply_move(payload)
                elif cmd == "look_around":
                    self._load = min(1.0, self._load + 0.1)
                    await self._apply_look(payload)
            except Exception:
                logger.exception("处理 motor 指令失败: %s", cmd)
            # 不再因为"去找充电器"就自行改充电状态：
            # 充电由 bot_simulator 根据同一条 motor 指令处理，
            # 这里插一脚会让两边状态不一致。
        elif topic == Topic.CORTEX_THOUGHT:
            # 每产生一次思维就计入负载
            self._load = min(1.0, self._load + 0.12)

    async def _apply_move(self, payload: dict) -> None:
        """处理一次真实移动。

        `place` 由模型通过 move_to 工具声明。工具把它直接写在 payload 顶层
        （registry 会把参数平铺进 payload），而旧的 motor 指令把它放在
        params 里——两种都要认，否则"模型说去书架"与"机器人动了"
        会各走各的。
        """
        params = payload.get("params") or {}
        place = payload.get("place") or params.get("place")

        if place:
            # 走路必须是一段过程，不能是一次坐标跳跃。
            #
            # 这里曾经是"按 speed 算一个 1~4 米的步长，直接改坐标"——
            # 一次调用就把人从房间这头挪到那头，画面上看是整体搬移，
            # 而不是走过去。当时的动机是"1 米/轮太慢，模型以为没动"，
            # 但那是**每轮只走一步**造成的，不是步长太小造成的。
            # 正确的修法是让单次 move_to 覆盖整段路程，但把它拆成
            # 许多帧逐步推进；而不是把一步调大，一次跨完。
            await self._walk_to(place, float(payload.get("speed") or params.get("speed") or 0.4))
            return

        # 无目的地：沿当前朝向前进 1 米。
        # 这条分支只服务内部的旧调用方，模型走的都是带 place 的那条。
        self._space.x += math.sin(math.radians(self._space.heading))
        self._space.z -= math.cos(math.radians(self._space.heading))
        r = math.hypot(self._space.x, self._space.z)
        if r > self._space.world.radius:
            self._space.x *= self._space.world.radius / r
            self._space.z *= self._space.world.radius / r

        here = self._space.at_object()
        await self._publish_space_frame("arrived" if here else "walking", None)

    async def _walk_to(self, place: str, speed: float) -> None:
        """走向某个地方，途中持续广播位置。

        这是"走路"和"瞬移"的分界线：位置不是一次改完，而是按约 20Hz
        小步推进，每一帧都发布一次空间状态，前端据此插值播放。

        参数 speed 的语义从"一步跨多远"变成"走多快"：
          0.0 → 0.6 m/s（散步），1.0 → 3.0 m/s（快走）。
        单次 move_to 会一路走到底，所以对模型来说"走过去"
        仍然是一次调用就完成，它不需要连喊十几次。

        途中被打断（新的移动指令）就停下，由新指令接管。
        """
        obj = next((o for o in self._space.world.objects if o.name == place), None)
        if obj is None:
            logger.warning("移动目标不存在: %s", place)
            return

        # 用序号标记这次行走，新指令到来时旧循环自行退出。
        self._walk_seq += 1
        seq = self._walk_seq
        self._space.target = place
        self._space.heading = math.degrees(
            math.atan2(obj.x - self._space.x, -(obj.z - self._space.z))
        ) % 360

        speed = max(0.0, min(1.0, speed))
        mps = 0.6 + speed * 2.4          # 米/秒
        tick = 0.05                      # 20Hz

        logger.info("🚶 出发 -> %s（%.1f 米，%.1f m/s）",
                    place, self._space.distance_to(obj), mps)

        while self._walk_seq == seq:
            remaining = self._space.distance_to(obj)
            if remaining <= obj.reach:
                self._space.target = None
                await self._publish_space_frame("arrived", obj.name)
                logger.info("🚶 到达 %s (%.1f, %.1f)",
                            obj.name, self._space.x, self._space.z)
                return

            step = min(mps * tick, remaining)
            dx, dz = obj.x - self._space.x, obj.z - self._space.z
            self._space.x += dx / remaining * step
            self._space.z += dz / remaining * step

            # 出界拉回，与前端 BOUND 行为一致
            r = math.hypot(self._space.x, self._space.z)
            if r > self._space.world.radius:
                self._space.x *= self._space.world.radius / r
                self._space.z *= self._space.world.radius / r

            await self._publish_space_frame("walking", place)
            await asyncio.sleep(tick)

    async def _publish_space_frame(self, event: str, target: Optional[str]) -> None:
        """广播一帧位置。

        两条通道各有分工，缺一不可：

          1. animat/env/status —— 前端画小人用的。网页只在这个话题上
             调用 renderSpace 更新 robot 坐标；如果只发 SENSOR_VISION，
             画面上的人是不会动的（SENSOR_VISION 走的是事件日志渲染）。
          2. animat/sensor/vision —— 模型读的"视野里发生了什么"。
             只在到达时给 event_text，途中给空字符串：否则每 50ms
             一句"还有 x 米"会把感知窗口刷满同一句话。

        走路期间状态话题从 3 秒一次临时提到 20Hz，是这一处刻意的例外：
        位置必须连续，否则看起来就是瞬移。整段路通常只有几秒。
        """
        here = self._space.at_object()
        if event == "arrived" and here:
            text = f"你走到了{here.name}旁。{here.desc}。"
        else:
            tgt = next((o for o in self._space.world.objects
                        if o.name == (target or "")), None)
            left = f"，还有 {self._space.distance_to(tgt):.1f} 米" if tgt else ""
            text = f"你正朝{target or '前方'}移动{left}。"

        # 前端绘制用：整份状态（含物件），每帧一次
        await self._publish_status()

        await self._publish(Topic.SENSOR_VISION, {
            "event": "moved" if event == "arrived" else "walking",
            # 途中不写 event_text：每帧一句"还有 x 米"会刷满感知窗口
            "event_text": text if event == "arrived" else "",
            "place": here.name if here else None,
            "position": {"x": round(self._space.x, 2), "z": round(self._space.z, 2)},
            "heading": round(self._space.heading, 1),
            "ambient_light": self._circadian.ambient_light,
        }, qos=0)

    async def _apply_look(self, payload: dict) -> None:
        """处理一次 look_around：原地转身，回一帧"现在看到什么"。

        这是最廉价的动作：不走动、不改坐标，只换一个视角。
        它的意义在于给"我想确认一下"一个可执行的形态——
        没有它的话，模型对空间的好奇只能二选一：要么什么都不做，
        要么真的走一趟，中间没有便宜的选择。
        """
        # direction 有两种来源，必须都认：
        #   工具调用（registry 平铺参数）  {"command":"look_around","direction":"left"}
        #   内部路径（嵌套）              {"command":"...","params":{"direction":...}}
        direction = (payload.get("direction")
                     or (payload.get("params") or {}).get("direction")
                     or "around")
        facing = self._space.turn(direction)
        text = self._space.describe()
        logger.info("👀 环视 -> 面向%s", facing)

        await self._publish(Topic.SENSOR_VISION, {
            "event": "looked",
            "event_text": text,
            "facing": facing,
            "position": {"x": round(self._space.x, 2), "z": round(self._space.z, 2)},
            "heading": round(self._space.heading, 1),
            "ambient_light": self._circadian.ambient_light,
        }, qos=0)
        # 位置没变但朝向变了，_on_space_update 的指纹包含 heading，
        # 会触发一次空间文本重算，模型下一轮读到的是新视角。
        await self._publish_status()

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
            # 空间状态：位置、朝向、可见物体及真实距离。
            # 前端据此绘制，模型据此决定去哪——同一份数据，不是两套叙事。
            "space": self._space.look(),
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
        logger.info("👂 控制通道已就绪，开始消费消息")
        async for message in client.messages:
            topic = str(message.topic)
            # 这里是本进程**唯一**的 client.messages 消费者。
            # motor / cortex.thought 不分发出去的话，它们就永远没人处理
            # （aiomqtt 的 messages 是单一异步流，不能被两个 async for 共享）。
            if topic != Topic.ENV_CONTROL:
                await self._on_motor(message)
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
