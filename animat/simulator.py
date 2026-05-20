"""
Animat Agent 传感器模拟器

用于无实体硬件时的开发测试。模拟传感器数据发布到MQTT，
模拟效应器接收动作指令并打印。

关键特性:
  - 动作反馈: 收到 move charger 指令时自动开始充电
  - 交互控制: /charge, /reset, /status 命令

用法:
  python -m animat.simulator
"""

import os
import asyncio
import json
import random
import time
import logging

import aiomqtt

from animat.config import (
    Topic, HomeostasisConfig,
)

logger = logging.getLogger("animat.simulator")


class SensorSimulator:
    """模拟传感器数据输入 + 动作反馈"""

    def __init__(self):
        self._battery = HomeostasisConfig.VIABILITY["ideal"]
        self._running = False
        self._charging = False          # 是否正在充电
        self._charge_rate = 3.0         # 充电速率: %/秒
        self.host = os.getenv("MQTT_BROKER_HOST", "localhost"),
        self.port = os.getenv("MQTT_BROKER_PORT", "1883"),
        self.alive= os.getenv("MQTT_KEEPALIVE", "60"),

    @property
    def charging(self) -> bool:
        return self._charging

    @charging.setter
    def charging(self, value: bool):
        if value and not self._charging:
            logger.info("⚡ Charging started!")
        elif not value and self._charging:
            logger.info("🔋 Charging stopped")
        self._charging = value

    def reset_battery(self):
        """重置电量为满"""
        self._battery = HomeostasisConfig.VIABILITY["ideal"]
        logger.info(f"🔄 Battery reset to {self._battery:.0f}%")

    async def run(self):
        async with aiomqtt.Client(
            hostname=self.host,
            port=self.port,
            identifier="animat-sensor-sim",
        ) as client:
            self._running = True
            logger.info("Sensor simulator started (Ctrl+C to stop)")
            logger.info("Special commands: /charge, /stop, /reset, /status")

            # 订阅动作指令(观察Animat的输出)
            await client.subscribe(Topic.MOTOR, qos=1)
            await client.subscribe(Topic.CORTEX_OUTPUT, qos=1)
            await client.subscribe("animat/heartbeat", qos=0)

            tasks = [
                asyncio.create_task(self._simulate_interoception(client)),
                asyncio.create_task(self._simulate_touch(client)),
                asyncio.create_task(self._simulate_vision(client)),
                asyncio.create_task(self._listen_motor(client)),
            ]

            await asyncio.gather(*tasks)

    async def _simulate_interoception(self, client: aiomqtt.Client):
        """模拟内感: 电量根据充放电状态变化"""
        while self._running:
            if self._charging:
                # 充电: 电量上升
                self._battery = min(100, self._battery + self._charge_rate)
                logger.info(f"[INTERO] ⚡ CHARGING battery={self._battery:.1f}%")
            else:
                # 放电: 电量缓慢下降
                self._battery = max(0, self._battery - random.uniform(0.1, 0.3))
                logger.debug(f"[INTERO] battery={self._battery:.1f}%")

            payload = {
                "battery_pct": round(self._battery, 1),
                "temperature": round(random.uniform(35, 42), 1),
                "imu_tilt_deg": round(random.gauss(0, 2), 1),
                "charging": self._charging,
            }
            await client.publish(
                Topic.SENSOR_INTERO,
                json.dumps(payload).encode(),
                qos=0,
            )

            # 充满自动停止
            if self._charging and self._battery >= 100:
                self.charging = False
                logger.info("🔋 Battery full, charging stopped")

            await asyncio.sleep(2)

    async def _simulate_touch(self, client: aiomqtt.Client):
        """模拟触觉: 偶尔被抚摸或碰撞"""
        while self._running:
            await asyncio.sleep(random.uniform(5, 20))
            if random.random() < 0.7:
                payload = {
                    "pressure": round(random.uniform(0.05, 0.15), 3),
                    "capacitance": round(random.uniform(0.3, 0.8), 3),
                    "x": round(random.uniform(0, 1), 2),
                    "y": round(random.uniform(0, 1), 2),
                }
                logger.info("[TOUCH] Caress detected")
            else:
                payload = {
                    "pressure": round(random.uniform(0.6, 1.0), 3),
                    "capacitance": 0.01,
                    "x": 0.5,
                    "y": 0.5,
                }
                logger.info("[TOUCH] Impact detected!")
            await client.publish(
                Topic.SENSOR_TOUCH,
                json.dumps(payload).encode(),
                qos=1,
            )

    async def _simulate_vision(self, client: aiomqtt.Client):
        """模拟视觉: 偶尔检测到人"""
        while self._running:
            await asyncio.sleep(random.uniform(3, 10))
            human_present = random.random() < 0.5
            payload = {
                "distance_m": round(random.uniform(0.5, 3.0), 2) if human_present else 10.0,
                "face_detected": human_present,
                "gaze_seconds": round(random.uniform(0.5, 3.0), 1) if human_present else 0,
            }
            if human_present:
                logger.info(f"[VISION] Human detected at {payload['distance_m']}m")
            await client.publish(
                Topic.SENSOR_VISION,
                json.dumps(payload).encode(),
                qos=0,
            )

    async def _listen_motor(self, client: aiomqtt.Client):
        """监听Animat的动作输出，并做出环境反馈"""
        async for message in client.messages:
            topic = str(message.topic)
            try:
                payload = json.loads(message.payload.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = message.payload.decode(errors="replace")

            if topic == Topic.MOTOR:
                cmd = payload.get("command", "?")
                params = payload.get("params", {})
                if cmd == "speak":
                    logger.info(f"🗣️  Animat says: {params.get('text', '')}")
                elif cmd == "move":
                    direction = params.get("direction", "")
                    if direction == "charger":
                        # Animat 在找充电器 → 模拟环境响应: 开始充电
                        self.charging = True
                    else:
                        # 离开充电器方向 → 停止充电
                        if self._charging:
                            self.charging = False
                    logger.info(f"🔧 Motor: {cmd} {params}")
                else:
                    logger.info(f"🔧 Motor: {cmd} {params}")
            elif topic == Topic.CORTEX_OUTPUT:
                logger.info(f"🧠 Intent: {payload.get('intent')} | Text: {payload.get('text', '')[:80]}")
            elif topic == "animat/heartbeat":
                logger.debug(
                    f"💓 tick={payload.get('tick')} "
                    f"battery={payload.get('viability', 0):.0f}% "
                    f"emotion={payload.get('emotion')}"
                )


# ─── 交互式命令 ───
SIM_COMMANDS = {
    "/charge": "开始充电",
    "/stop":   "停止充电",
    "/reset":  "重置电量到100%",
    "/status": "显示当前状态",
    "/help":   "显示帮助",
}


def _handle_sim_command(text: str, sim: SensorSimulator) -> bool:
    """处理模拟器特殊命令，返回 True 表示已处理"""
    cmd = text.strip().lower()
    if cmd == "/charge":
        sim.charging = True
        return True
    elif cmd == "/stop":
        sim.charging = False
        return True
    elif cmd == "/reset":
        sim.reset_battery()
        return True
    elif cmd == "/status":
        logger.info(f"📊 Battery: {sim._battery:.1f}% | Charging: {sim._charging}")
        return True
    elif cmd == "/help":
        for k, v in SIM_COMMANDS.items():
            logger.info(f"  {k:10s} {v}")
        return True
    elif cmd.startswith("/"):
        logger.info(f"Unknown command: {cmd}. Type /help for available commands.")
        return True
    return False


async def interactive_input(sim: SensorSimulator):
    """交互式输入: 模拟人类对Animat说话 + 模拟器控制命令"""
    async with aiomqtt.Client(
        hostname=os.getenv("MQTT_BROKER_HOST", "localhost"),
        port=os.getenv("MQTT_BROKER_PORT", "1883"),
        identifier="animat-human-input",
    ) as client:
        loop = asyncio.get_event_loop()

        while True:
            text = await loop.run_in_executor(None, lambda: input("You> "))
            if not text.strip():
                continue

            # 检查是否是模拟器命令
            if _handle_sim_command(text, sim):
                continue

            # 普通语言输入 → 发送给 Animat
            await client.publish(
                "animat/external/speech",
                json.dumps({"text": text.strip(), "timestamp": time.time()}).encode(),
                qos=1,
            )


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sim = SensorSimulator()
    sim_task = asyncio.create_task(sim.run())
    input_task = asyncio.create_task(interactive_input(sim))

    try:
        await asyncio.gather(sim_task, input_task)
    except KeyboardInterrupt:
        sim._running = False
        sim_task.cancel()
        input_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
