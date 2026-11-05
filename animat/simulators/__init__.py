"""
外部世界与身体的模拟器

这些模块不是 Animat 核心的一部分，而是给没有真实硬件时用的替身：
它们扮演"环境"和"身体"向 animat/sensor/* 注入数据，让大脑以为
自己确实活在一个有昼夜、有路人、有电量的世界里。

之所以单独成包，是为了让核心目录保持干净：animat/ 下剩下来的
都是智能体本身的器官（感知、思维、记忆、效应器），而这里是体外
的布景。接真机时整个包可以直接不启动，核心无需任何改动。

两个模拟器职责互斥，不要同时承担同一类信号：
  bot_simulator  身体 —— 电量 / 体温 / 充电 / 倾角
  env_simulator  世界 —— 昼夜 / 光照 / 底噪 / 路人 / 触碰

用法:
  python -m animat.simulators.bot_simulator
  python -m animat.simulators.env_simulator

注意：这里刻意不做 eager import。两个模块都是可执行入口（`-m` 运行），
若在包初始化时就导入它们，runpy 会看到模块已在 sys.modules 中，
从而报 "found in sys.modules after import of package" 警告。
需要类对象时用下面的 __getattr__ 惰性获取。
"""

__all__ = ["BodySimulator", "EnvironmentSimulator"]


def __getattr__(name: str):
    """按需导入，兼顾 `from animat.simulators import BodySimulator` 的用法。"""
    if name == "BodySimulator":
        from animat.simulators.bot_simulator import BodySimulator
        return BodySimulator
    if name == "EnvironmentSimulator":
        from animat.simulators.env_simulator import EnvironmentSimulator
        return EnvironmentSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
