"""
Animat Agent 传感器模拟器

用于无实体硬件时的开发测试。模拟传感器数据发布到MQTT，
模拟效应器接收动作指令并打印。

关键特性:
  - 动作反馈: 收到 move charger 指令时自动开始充电
  - 内感(电量)始终自动上报，避免主进程电量卡死
  - 触觉/视觉默认不自动随机发送，改由用户手动命令发起(便于调试):
    /touch 抚摸, /hit 撞击, /approach 靠近, /leave 离开, /noise 巨响
  - 交互控制: /charge, /stop, /reset, /status
  - 设环境变量 ANIMAT_SIM_AUTO_STIM=1 可恢复自动随机触觉/视觉刺激
  - 内感(电量/体温)默认稳定: 电量满电慢放, 体温恒定37°C不随机跳变
  - 环境变量:
      ANIMAT_SIM_BATTERY=100   初始电量(%)
      ANIMAT_SIM_DISCHARGE=0.05 放电速率(%/周期, 越小越耐用)
      ANIMAT_SIM_AUTO_STIM=1    开启自动随机触觉/视觉

用法:
  uv run python -m animat.simulator        # 默认手动刺激模式
  ANIMAT_SIM_AUTO_STIM=1 uv run python -m animat.simulator   # 自动随机刺激
"""

import os
import sys
import asyncio
import json
import random
import time
import logging

import aiomqtt

from animat.config import (
    Topic, HomeostasisConfig, SharedConstants,
)

logger = logging.getLogger("animat.simulator")


class SensorSimulator:
    """模拟传感器数据输入 + 动作反馈"""

    def __init__(self, auto_stim: bool = False):
        self._battery = float(os.getenv("ANIMAT_SIM_BATTERY", "100"))
        self._temperature = 37.0        # 当前体温(°C)，默认正常，不再随机乱跳
        self._running = False
        self._charging = False          # 是否正在充电
        self._charge_rate = 3.0         # 充电速率: %/秒
        # 放电速率: 大电池，降低放电让交流更持久 (默认 0.03~0.08 %/2s)
        self._discharge_rate = float(os.getenv("ANIMAT_SIM_DISCHARGE", "0.05"))
        # 自动随机刺激(触觉/视觉)开关。默认关闭，由用户手动命令发起，
        # 避免自动噪声干扰调试。设 ANIMAT_SIM_AUTO_STIM=1 可重新开启。
        self.auto_stim = auto_stim or os.getenv("ANIMAT_SIM_AUTO_STIM", "0") == "1"
        self.host = os.getenv("MQTT_BROKER_HOST", "localhost")
        self.port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
        self.alive = int(os.getenv("MQTT_KEEPALIVE", "60"))

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
            if self.auto_stim:
                logger.info("Auto stimulus: ON (random touch/vision). "
                            "Set ANIMAT_SIM_AUTO_STIM=0 to disable.")
            else:
                logger.info("Auto stimulus: OFF (touch/vision only via /touch /hit "
                            "/approach /leave). Interoception (battery) still auto-published.")
            logger.info("Type /help for commands. Special sim commands: "
                        "/charge, /stop, /reset, /status")

            # 订阅动作指令(观察Animat的输出)
            await client.subscribe(Topic.MOTOR, qos=1)
            await client.subscribe(Topic.CORTEX_OUTPUT, qos=1)
            await client.subscribe("animat/heartbeat", qos=0)

            # 内感(电量)始终自动上报，否则主进程电量卡死。
            # 触觉/视觉仅在 auto_stim 开启时自动循环，否则由用户手动命令发起。
            tasks = [
                asyncio.create_task(self._simulate_interoception(client)),
                asyncio.create_task(self._listen_motor(client)),
            ]
            if self.auto_stim:
                tasks.append(asyncio.create_task(self._simulate_touch(client)))
                tasks.append(asyncio.create_task(self._simulate_vision(client)))

            await asyncio.gather(*tasks)

    async def _simulate_interoception(self, client: aiomqtt.Client):
        """模拟内感: 电量根据充放电状态变化，体温保持稳定"""
        while self._running:
            if self._charging:
                # 充电: 电量上升，体温随充电轻微升高(模拟发热)
                self._battery = min(100, self._battery + self._charge_rate)
                self._temperature = min(45.0, self._temperature + 0.05)
                logger.info(f"[INTERO] ⚡ CHARGING battery={self._battery:.1f}% "
                            f"temp={self._temperature:.1f}°C")
            else:
                # 放电: 电量缓慢下降(大电池，默认很慢)
                self._battery = max(0, self._battery - self._discharge_rate)
                # 体温缓慢回归正常 37°C，避免随机噪声触发热焦虑
                self._temperature += (37.0 - self._temperature) * 0.1
                logger.debug(f"[INTERO] battery={self._battery:.1f}% temp={self._temperature:.1f}°C")

            payload = {
                "battery_pct": round(self._battery, 1),
                "temperature": round(self._temperature, 1),
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
                    if direction == SharedConstants.DIRECTION_CHARGER:
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
# 模拟器自身控制命令
SIM_COMMANDS = {
    "/charge": "开始充电",
    "/stop":   "停止充电",
    "/reset":  "重置电量到100%",
    "/status": "显示当前状态",
    "/help":   "显示帮助",
}

# 环境刺激命令（发布到 MQTT 传感器主题 → 刺激主进程 Animat）
STIM_COMMANDS = {
    "/touch":    "模拟轻柔抚摸 (低压+电容)",
    "/hit":      "模拟突然撞击 (高压突变→痛觉)",
    "/approach": "模拟有人靠近 (1.5m，正面凝视)",
    "/leave":    "模拟人离开 (距离10m)",
    "/noise":    "模拟巨响惊吓 (听觉突变)",
    "/battery":  "设置电量, 如 /battery 15",
    "/temp":     "设置体温(°C), 如 /temp 43 测试过热",
}


def _handle_sim_command(text: str, sim: SensorSimulator) -> bool:
    """处理模拟器自身命令（charge/stop/reset/status），返回 True 表示已处理"""
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
        logger.info("Simulator commands:")
        for k, v in SIM_COMMANDS.items():
            logger.info(f"  {k:10s} {v}")
        logger.info("Stimulation commands (publish to main process):")
        for k, v in STIM_COMMANDS.items():
            logger.info(f"  {k:10s} {v}")
        return True
    elif cmd.startswith("/"):
        # 其余 / 开头命令交给刺激处理器
        return False
    return False


async def _publish_stim(client: aiomqtt.Client, topic: str, payload: dict):
    """发布一条刺激到 MQTT（QoS=1 确保主进程收到）"""
    await client.publish(topic, json.dumps(payload).encode(), qos=1)


async def _handle_stim_command(text: str, sim: SensorSimulator, client: aiomqtt.Client) -> bool:
    """
    处理环境刺激命令，返回 True 表示已处理。
    这些命令向主进程 Animat 发布传感器数据，绕过模拟器自身的随机循环，
    让用户可以主动制造触觉/视觉/听觉/内感变化来观察 agent 反应。
    """
    parts = text.strip().lower().split()
    cmd = parts[0]

    if cmd == "/touch":
        # 抚摸: 低压 + 电容(滑动感)。主进程 perception 判定 pressure<0.15 & capacitance>0.1 → caress
        await _publish_stim(client, Topic.SENSOR_TOUCH, {
            "pressure": round(random.uniform(0.03, 0.12), 3),
            "capacitance": round(random.uniform(0.3, 0.9), 3),
            "x": round(random.uniform(0.1, 0.5), 2),
            "y": round(random.uniform(0.2, 0.6), 2),
        })
        logger.info("[STIM] ✋ 轻柔抚摸已发送 → animat/sensor/touch")
        return True

    if cmd == "/hit":
        # 撞击: 主进程痛觉依赖 dP/dt (压力突变)。先发一帧低压，再发高压，制造导数跳变。
        await _publish_stim(client, Topic.SENSOR_TOUCH, {
            "pressure": 0.05, "capacitance": 0.01, "x": 0.5, "y": 0.5,
        })
        await _publish_stim(client, Topic.SENSOR_TOUCH, {
            "pressure": round(random.uniform(0.8, 1.0), 3), "capacitance": 0.01,
            "x": 0.5, "y": 0.5,
        })
        logger.info("[STIM] 💥 撞击已发送 (双帧高压突变→主进程触发 pain/withdraw 反射)")
        return True

    if cmd == "/approach":
        # 靠近: 发一帧近距离+人脸+凝视，主进程 LLM 看到"有人靠近并注视"
        await _publish_stim(client, Topic.SENSOR_VISION, {
            "distance_m": 1.5, "face_detected": True, "gaze_seconds": 3.0,
        })
        logger.info("[STIM] 🚶 有人靠近 (1.5m, 正面凝视) → animat/sensor/vision")
        return True

    if cmd == "/leave":
        await _publish_stim(client, Topic.SENSOR_VISION, {
            "distance_m": 10.0, "face_detected": False, "gaze_seconds": 0,
        })
        logger.info("[STIM] 👋 人已离开 → animat/sensor/vision")
        return True

    if cmd == "/noise":
        # 巨响: 主进程 auditory startle 依赖 d_db > 85。先静音再爆发。
        await _publish_stim(client, Topic.SENSOR_AUDIO, {
            "volume_db": 30, "fundamental_freq": 200,
        })
        await _publish_stim(client, Topic.SENSOR_AUDIO, {
            "volume_db": 110, "fundamental_freq": 600,
        })
        logger.info("[STIM] 🔊 巨响已发送 (dB突变>85 → 主进程 flinch 反射)")
        return True

    if cmd == "/battery" and len(parts) > 1:
        try:
            val = float(parts[1])
            # 设置电量: 直接改写模拟器内感循环将发布的电量
            sim._battery = val
            if val >= 95:
                sim.charging = False
            logger.info(f"[STIM] 🔋 电量将设为 {val:.0f}% (下个内感周期生效)")
        except ValueError:
            logger.info("[STIM] 无效电量值")
        return True

    if cmd == "/temp" and len(parts) > 1:
        try:
            val = float(parts[1])
            sim._temperature = val
            logger.info(f"[STIM] 🌡️ 体温将设为 {val:.1f}°C (下个内感周期生效)")
        except ValueError:
            logger.info("[STIM] 无效温度值")
        return True

    if cmd == "/help":
        # 已在 _handle_sim_command 中处理
        return True

    if cmd.startswith("/"):
        logger.info(f"Unknown command: {cmd}. Type /help for available commands.")
        return True

    return False


def _looks_like_pinyin_garbage(text: str) -> bool:
    """
    检测疑似"拼音未上屏成汉字"的乱码输入。

    典型场景: 用户用拼音输入法打字, 但字母没提交成汉字就回车,
    导致 'ni你好nemen...' 这种拉丁字母与中文夹杂的字符串被发出。

    判定规则(宽松, 避免误伤正常英文/中英混排):
      - 存在连续 >=2 个字母组成的"裸词"(如 ni, ting, nemen)
      - 且整体并非以正常中文为主(中文占比 < 30%)
    """
    if not text:
        return False
    # 连续拉丁字母词(长度>=2)
    import re
    latin_words = re.findall(r"[A-Za-z]{2,}", text)
    if not latin_words:
        return False
    han_count = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin_char_count = sum(len(w) for w in latin_words)
    total = len(text.strip())
    # 裸字母词较多, 且中文占比很低 → 视为拼音乱码
    if latin_char_count >= 4 and (han_count == 0 or han_count / total < 0.3):
        return True
    return False


async def interactive_input(sim: SensorSimulator):
    """交互式输入: 模拟人类对Animat说话 + 模拟器控制 + 环境刺激"""
    async with aiomqtt.Client(
        hostname=os.getenv("MQTT_BROKER_HOST", "localhost"),
        port=int(os.getenv("MQTT_BROKER_PORT", "1883")),
        identifier="animat-human-input",
    ) as client:
        loop = asyncio.get_event_loop()

        # 强制 stdin/stdout 使用 UTF-8，避免非 UTF-8 locale 下中文被截断/替换
        try:
            sys.stdin.reconfigure(encoding="utf-8")
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

        def _read_line() -> str:
            # 用二进制读取 stdin 再容错解码，避免非法 UTF-8 字节
            # (如输入法混合序列) 直接令 input() 抛出 UnicodeDecodeError
            raw = sys.stdin.buffer.readline()
            if not raw:
                return ""
            return raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")

        while True:
            try:
                text = await loop.run_in_executor(None, _read_line)
            except (EOFError, KeyboardInterrupt):
                sim._running = False
                break
            if not text.strip():
                continue

            # 1) 模拟器自身命令 (charge/stop/reset/status/help)
            if _handle_sim_command(text, sim):
                continue

            # 2) 环境刺激命令 (touch/hit/approach/leave/noise/battery)
            if text.strip().startswith("/"):
                if await _handle_stim_command(text, sim, client):
                    continue

            # 3) 普通语言输入 → 检测是否为拼音/乱码，避免误发
            cleaned = text.strip()
            if _looks_like_pinyin_garbage(cleaned):
                logger.warning(
                    "[INPUT] 检测到疑似拼音/乱码输入(含裸拉丁字母)，已拦截未发送。"
                    "请确认终端 locale 为 UTF-8 且输入法已把拼音上屏成汉字后再回车。"
                    f" 你输入的是: {cleaned!r}"
                )
                continue

            # 发送给 Animat
            await client.publish(
                "animat/external/speech",
                json.dumps({"text": cleaned, "timestamp": time.time()}).encode(),
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
