"""
Bot 模拟器 —— Animat 的身体

用于无实体硬件时，模拟这台机器人自身的生理状态：电量、体温、充电。
不模拟外部世界——昼夜、光照、路人、触碰、噪声一律由 env_simulator 负责。

职责边界（重要）:
  bot_simulator  身体  → 电量 / 体温 / 充电 / 倾角，写 animat/sensor/interoception
  env_simulator  世界  → 昼夜节律 / 光照 / 路人 / 轻微触碰，写其余 sensor 话题

两者都会写 SENSOR_INTERO，所以设计上分工明确：
  - 本模块是电量与体温的唯一权威来源
  - env_simulator 的认知负载只影响"放电速度"，不直接写数值
  同时启动时以本模块为准，避免两套生理模型互相打架。

控制方式（仅 MQTT，无终端交互）:
  发布 animat/body/control  {"cmd": "charge"}
  可用 cmd:
    charge        开始充电
    stop          停止充电
    reset         电量重置到理想值
    status        当前状态（回执）
    scale         设定电量，需 arg，如 {"cmd":"scale","arg":15}
    temp          设定体温 °C，需 arg，如 {"cmd":"temp","arg":43}
    discharge     设定放电速率 %/周期，需 arg
  执行结果与当前状态发布在 animat/body/status。
  浏览器打开 webui/body.html 即可操作。

环境变量:
  ANIMAT_SIM_BATTERY=100     初始电量(%)
  ANIMAT_SIM_DISCHARGE=0.05  放电速率(%/周期)

用法:
  uv run python -m animat.simulators.bot_simulator
"""

import os
import asyncio
import json
import logging
import random
import time
from typing import Callable, Optional

import aiomqtt

from animat.config import Topic, HomeostasisConfig

logger = logging.getLogger("animat.bot_simulator")


class BodySimulator:
    """模拟 Animat 自身的生理状态。

    只做三件事：电量怎么变、体温怎么变、在没在充电。
    任何与"世界"有关的输入都不在这里产生。
    """

    # 体温在充电时的上限。真实电池包长时间满功率充电会热，
    # 但不会烧到 45°C，这里给一个物理上说得通的天花板。
    TEMP_CHARGE_CAP = 41.0
    TEMP_NORMAL = 37.0
    CHARGE_RATE = 3.0          # %/秒
    INTERO_INTERVAL = 2.0      # 内感上报周期(秒)

    def __init__(self):
        self._battery = float(os.getenv("ANIMAT_SIM_BATTERY", "100"))
        self._temperature = self.TEMP_NORMAL
        self._charging = False
        self._running = False
        self._discharge_rate = float(os.getenv("ANIMAT_SIM_DISCHARGE", "0.05"))

        # 倾角：静止时轻微抖动，给感知层一个"传感器有噪声"的真实感
        self._tilt = 0.0

        self.host = os.getenv("MQTT_BROKER_HOST", "localhost")
        self.port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
        self.alive = int(os.getenv("MQTT_KEEPALIVE", "60"))

        # 连接建立后由 run() 赋值；_publish 在未连接时静默跳过，
        # 这样任何提前触发的广播都不会抛 AttributeError。
        self._client: Optional[aiomqtt.Client] = None

    @property
    def charging(self) -> bool:
        return self._charging

    @charging.setter
    def charging(self, value: bool):
        if value and not self._charging:
            logger.info("⚡ 开始充电")
        elif not value and self._charging:
            logger.info("🔋 停止充电")
        self._charging = value

    def reset_battery(self):
        self._battery = float(HomeostasisConfig.VIABILITY["ideal"])
        logger.info("🔄 电量重置为 %.0f%%", self._battery)

    # ── 状态快照 ──
    def _status_dict(self) -> dict:
        return {
            "type": "status",
            "battery_pct": round(self._battery, 1),
            "temperature": round(self._temperature, 1),
            "charging": self._charging,
            "discharge_rate": round(self._discharge_rate, 3),
            "imu_tilt_deg": round(self._tilt, 1),
            "temp_normal": self.TEMP_NORMAL,
            "temp_charge_cap": self.TEMP_CHARGE_CAP,
            "ideal_battery": HomeostasisConfig.VIABILITY["ideal"],
            "danger_battery": HomeostasisConfig.VIABILITY["danger"],
            "ts": time.time(),
        }

    async def _publish(self, topic: str, payload: dict, qos: int = 0):
        if self._client is not None:
            await self._client.publish(
                topic, json.dumps(payload, ensure_ascii=False).encode(), qos=qos
            )

    async def _publish_status(self):
        await self._publish(Topic.BODY_STATUS, self._status_dict(), qos=0)

    # ── 内感上报 ──
    async def _interoception_loop(self):
        """电量和体温的演化与上报。

        这是本模块唯一写 SENSOR_INTERO 的地方。每一步都保持物理自洽：
        充电 → 电量涨 + 体温升；放电 → 电量降 + 体温回落。
        """
        while self._running:
            if self._charging:
                self._battery = min(100.0, self._battery + self.CHARGE_RATE)
                # 充电升温，但封顶
                self._temperature = min(
                    self.TEMP_CHARGE_CAP, self._temperature + 0.05
                )
                logger.info(
                    "[INTERO] ⚡ 充电中 battery=%.1f%% temp=%.1f°C",
                    self._battery, self._temperature,
                )
            else:
                self._battery = max(0.0, self._battery - self._discharge_rate)
                # 向正常体温缓慢回归
                self._temperature += (self.TEMP_NORMAL - self._temperature) * 0.1
                logger.debug(
                    "[INTERO] battery=%.1f%% temp=%.1f°C",
                    self._battery, self._temperature,
                )

            # 静止时的小幅抖动，模拟 IMU 噪声
            self._tilt = round(random.gauss(0, 1.5), 1)

            await self._publish(Topic.SENSOR_INTERO, {
                "battery_pct": round(self._battery, 1),
                "temperature": round(self._temperature, 1),
                "imu_tilt_deg": self._tilt,
                "charging": self._charging,
            }, qos=0)

            if self._charging and self._battery >= 100.0:
                self.charging = False
                logger.info("🔋 电量已满，自动停止充电")

            await asyncio.sleep(self.INTERO_INTERVAL)

    async def _status_loop(self):
        while self._running:
            try:
                await self._publish_status()
            except Exception as e:
                logger.debug("状态广播失败: %s", e)
            await asyncio.sleep(2)

    async def _on_motor(self, payload: dict):
        """处理 Animat 的动作，维护"走到充电位就有电"这条物理因果。

        这条逻辑属于身体而非世界：机器人自己移动到充电位，电量就该上涨。
        放在这里可保证电池状态只有一个写入方。
        """
        if payload.get("command") != "move":
            return
        direction = (payload.get("params") or {}).get("direction", "")
        if direction == "charger":
            logger.info("🔌 Animat 移动到充电位，开始充电")
            self.charging = True
        elif self._charging:
            # 离开充电位 → 断电。这是物理常识，不是额外策略。
            logger.info("🔌 Animat 离开充电位，停止充电")
            self.charging = False

    # ── 命令处理 ──
    async def _handle_command(
        self, cmd: str, arg: str = "", reply: Optional[Callable[[str], None]] = None
    ) -> bool:
        """执行一条身体命令，返回 False 表示要退出进程。

        与传输方式无关：MQTT 入口传一个 list.append 收集回执文字。
        """
        emit = reply or (lambda s: logger.info("%s", s))
        cmd = cmd.strip().lstrip("/").lower()
        arg = (arg or "").strip()

        if cmd in ("quit", "exit", "q"):
            self._running = False
            return False

        if cmd == "status":
            s = self._status_dict()
            emit(
                f"电量 {s['battery_pct']:.1f}% / 体温 {s['temperature']:.1f}°C / "
                f"{'充电中' if s['charging'] else '放电中'}"
            )

        elif cmd == "charge":
            self.charging = True
            emit("已开始充电")

        elif cmd == "stop":
            self.charging = False
            emit("已停止充电")

        elif cmd == "reset":
            self.reset_battery()
            emit(f"电量已重置为 {self._battery:.0f}%")

        elif cmd == "scale":
            try:
                val = max(0.0, min(100.0, float(arg)))
            except (TypeError, ValueError):
                emit(f"无效电量值: {arg!r}（需要 0-100 的数字）")
                return True
            self._battery = val
            # 满电还开着充电没有意义，顺手关掉
            if val >= 99.5:
                self.charging = False
            emit(f"电量已设为 {val:.0f}%")

        elif cmd == "temp":
            try:
                val = float(arg)
            except (TypeError, ValueError):
                emit(f"无效体温值: {arg!r}（需要数字）")
                return True
            # 给一个物理上合理的钳制，避免前端误传导致荒谬状态
            val = max(20.0, min(60.0, val))
            self._temperature = val
            emit(f"体温已设为 {val:.1f}°C")

        elif cmd == "discharge":
            try:
                val = max(0.0, min(10.0, float(arg)))
            except (TypeError, ValueError):
                emit(f"无效放电速率: {arg!r}")
                return True
            self._discharge_rate = val
            emit(f"放电速率已设为 {val:.2f}%/周期")

        elif cmd == "help":
            emit("可用命令: charge / stop / reset / status / "
                 "scale<N> / temp<N> / discharge<N>")
        else:
            emit(f"未知命令: {cmd}")

        return True

    async def _message_loop(self, client: aiomqtt.Client):
        """统一的消息消费循环。

        aiomqtt 的 client.messages 是单一异步流，多个 `async for` 会互相
        抢消息，所以这里用一次循环、按话题分发，而不是每类各开一个消费者。
        """
        async for message in client.messages:
            topic = str(message.topic)

            if topic == Topic.MOTOR:
                try:
                    payload = json.loads(message.payload.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                try:
                    await self._on_motor(payload)
                except Exception as e:
                    logger.warning("motor 处理失败: %s", e)
                continue

            if topic != Topic.BODY_CONTROL:
                continue

            raw = message.payload.decode("utf-8", errors="replace")
            logger.info("📥 收到身体命令: %s", raw[:120])

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
                keep = await self._handle_command(cmd, arg, reply=lines.append)
            except Exception as e:
                logger.warning("命令执行失败 %s: %s", cmd, e)
                await self._publish(Topic.BODY_STATUS, {
                    "type": "ack", "cmd": cmd, "ok": False,
                    "msg": f"执行失败: {e}", "ts": time.time(),
                })
                continue

            msg = " / ".join(lines) if lines else "ok"
            logger.info("🧍 身体命令已执行: %s -> %s", cmd, msg)
            await self._publish(Topic.BODY_STATUS, {
                "type": "ack", "cmd": cmd, "ok": True, "msg": msg, "ts": time.time(),
            })
            await self._publish_status()

            if not keep:
                self._running = False
                return

    async def run(self):
        async with aiomqtt.Client(
            hostname=self.host,
            port=self.port,
            identifier="animat-body-sim",
        ) as client:
            self._client = client
            self._running = True
            logger.info("🧍 身体模拟器启动 — 电量 %.0f%% / 体温 %.1f°C",
                        self._battery, self._temperature)
            logger.info("   控制通道: %s  （操作请用 webui/body.html）",
                        Topic.BODY_CONTROL)

            # 订阅自己的控制通道，以及 animat/motor 用于"到达充电位即充电"
            await client.subscribe(Topic.BODY_CONTROL, qos=1)
            await client.subscribe(Topic.MOTOR, qos=1)
            await self._publish_status()

            tasks = [
                asyncio.create_task(self._interoception_loop()),
                asyncio.create_task(self._status_loop()),
                asyncio.create_task(self._message_loop(client)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for t in tasks:
                    t.cancel()
                self._running = False


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sim = BodySimulator()
    try:
        await sim.run()
    except KeyboardInterrupt:
        sim._running = False


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
