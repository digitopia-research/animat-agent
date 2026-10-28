"""
Animat Agent 启动入口

用法:
  uv run animat            # 通过 entry point
  uv run animat --reset    # 启动前清除Redis状态
  uv run python -m animat  # 通过模块

环境变量 (或 .env 文件):
  OPENAI_API_KEY    - LLM API密钥
  OPENAI_BASE_URL   - LLM API地址 (默认OpenAI)
  ANIMAT_LLM_MODEL  - LLM模型名 (默认gpt-4o-mini)
  MQTT_BROKER_HOST  - MQTT Broker地址
  REDIS_HOST        - Redis地址
"""

import argparse
import asyncio
import logging
import signal
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env (优先级: .env.local > .env)
load_dotenv(Path(__file__).resolve().parent.parent / ".env.local", override=True)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from animat.nervous import NervousSystem
from animat.memory import MemoryStore
from animat.brain import AnimatBrain
from animat.config import RedisKeys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("animat")


async def _reset_state():
    """清除Redis中的Animat状态（用于从死亡状态恢复）"""
    import redis.asyncio as aioredis
    import os
    r = aioredis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    keys_to_delete = [
        RedisKeys.STATE_HOMEOSTASIS,
        RedisKeys.STATE_EMOTION,
        RedisKeys.CORTEX_HISTORY,
        RedisKeys.MEMORY_SEMANTIC,
    ]
    for key in keys_to_delete:
        await r.delete(key)
    await r.aclose()
    logger.info("State reset complete — Redis keys cleared")


async def run(reset: bool = False):
    logger.info("╔═════════════════════════════════════════════╗")
    logger.info("║     Animat Agent — The agent that lives.    ║")
    logger.info("╚═════════════════════════════════════════════╝")

    if reset:
        logger.info("Resetting state...")
        await _reset_state()

    # 初始化核心系统
    nervous = NervousSystem()
    memory = MemoryStore()
    brain = AnimatBrain(nervous, memory)

    # 优雅退出
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows 不支持 add_signal_handler
            pass

    # 启动
    boot_task = asyncio.create_task(brain.boot())

    # 等待退出信号
    await shutdown_event.wait()

    # 关闭
    await brain.shutdown()
    boot_task.cancel()
    try:
        await boot_task
    except asyncio.CancelledError:
        pass

    logger.info("Goodbye. I was alive.")


def cli():
    """Entry point for `animat` command"""
    parser = argparse.ArgumentParser(description="Animat Agent")
    parser.add_argument("--reset", action="store_true",
                        help="清除Redis状态后启动（从死亡状态恢复）")
    args = parser.parse_args()

    try:
        asyncio.run(run(reset=args.reset))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
