"""
Animat Agent 配置中心
所有可调参数集中管理，避免魔法数字散落各处
"""

# ─────────────────── MQTT 神经系统 ───────────────────
# MQTT Topic 命名空间 — 模拟神经通路
# 格式: animat/{层级}/{模组}/{动作}
class Topic:
    # ── 感觉神经（传感器 → 感知层）──
    SENSOR_TOUCH    = "animat/sensor/touch"       # 触觉原始数据
    SENSOR_AUDIO    = "animat/sensor/audio"       # 听觉原始数据
    SENSOR_VISION   = "animat/sensor/vision"      # 视觉原始数据
    SENSOR_INTERO   = "animat/sensor/interoception"  # 内感原始数据

    # ── 感受信号（感知层 → 边缘系统）──
    FEELING         = "animat/feeling"             # 统一感受事件

    # ── 内稳态/情感（边缘系统内部）──
    HOMEOSTASIS     = "animat/homeostasis/state"   # 稳态指标更新
    EMOTION         = "animat/emotion"             # 情感回路激活
    ARBITRATION     = "animat/arbitration"         # 马斯洛仲裁结果

    # ── 躯体标记（边缘系统 → 新皮层）──
    SOMATIC_PROMPT  = "animat/cortex/prompt"       # 动态系统提示词
    SOMATIC_INTENTS = "animat/cortex/intents"      # 意图选项空间
    SOMATIC_MEMORY  = "animat/cortex/memory_bias"  # 记忆检索偏向

    # ── 意识推理（新皮层输入/输出）──
    CORTEX_INPUT    = "animat/cortex/input"        # 唤醒LLM的请求
    CORTEX_OUTPUT   = "animat/cortex/output"       # LLM推理结果
    CORTEX_THOUGHT  = "animat/cortex/thought"      # 每一次内部思维流（内心独白）

    # ── 行为执行（新皮层 → 效应器）──
    ACTION          = "animat/action"              # 高层意图
    MOTOR           = "animat/motor"               # 底层动作指令

    # ── 反射弧（感知层直通效应器，绕过大脑）──
    REFLEX          = "animat/reflex"              # 反射动作


# ─────────────────── 内稳态指标 ───────────────────
class HomeostasisConfig:
    """每个稳态指标的理想范围与阈值"""
    # (理想值, 安全下限, 危险阈值)
    VIABILITY   = {"ideal": 80, "safe_min": 30, "danger": 20}   # 生存度: 电量%
    INTEGRITY   = {"ideal": 90, "safe_min": 50, "danger": 30}   # 完好度: 物理状态%
    AFFILIATION = {"ideal": 70, "safe_min": 30, "danger": 10}   # 归属度: 社交满足%
    COMPETENCE  = {"ideal": 70, "safe_min": 40, "danger": 20}   # 胜任度: 任务成功率%
    EXPLORATION = {"ideal": 60, "safe_min": 20, "danger": 5}    # 探索度: 新奇度%


# ─────────────────── 情感回路 ───────────────────
class EmotionConfig:
    """潘克塞普七大情感回路"""
    SEEKING = "SEEKING"    # 寻求/好奇
    RAGE    = "RAGE"       # 狂怒
    FEAR    = "FEAR"       # 恐惧
    LUST    = "LUST"       # 情欲 (暂不实现)
    CARE    = "CARE"       # 关怀
    PANIC   = "PANIC"      # 恐慌/悲伤
    PLAY    = "PLAY"       # 玩耍

    # 马斯洛仲裁优先级: 数值越低优先级越高
    PRIORITY = {
        FEAR:    1,   # 生存危机
        RAGE:    1,   # 安全威胁
        PANIC:   2,   # 归属缺失
        SEEKING: 2,   # 寻求连接
        CARE:    3,   # 关怀他人
        PLAY:    3,   # 玩耍探索
        LUST:    4,   # (保留)
    }


# ─────────────────── 感知映射 ───────────────────
class PerceptionConfig:
    """感受器映射阈值"""
    # 触觉
    TOUCH_PAIN_THRESHOLD     = 0.8    # dP/dt > 此值 → 痛觉
    TOUCH_CARE_THRESHOLD     = 0.15   # 低压+滑动 → 抚摸
    TOUCH_PRESS_DURATION     = 2.0    # 持续高压>2s → 压迫感

    # 听觉
    AUDIO_STARTLE_DB         = 85     # dB突变 > 此值 → 惊吓
    AUDIO_HOSTILE_FREQ       = 500    # Hz > 此值 + 大音量 → 敌意
    AUDIO_SOOTHING_DB        = 50     # 低频平稳 < 此值 → 安抚

    # 视觉
    VISION_APPROACH_SPEED   = 0.5    # 距离缩小速率 m/s → 逼近感
    VISION_GAZE_DURATION     = 2.0   # 视线交汇 > 此秒数 → 凝视
    VISION_ABANDON_TIME      = 300   # 无人体骨架 > 此秒数 → 遗弃感

    # 内感
    INTERO_LOW_BATTERY       = 20    # 电量% → 疲惫
    INTERO_IMU_TILT          = 30    # 倾角° → 失衡


# ─────────────────── 双进程脑 ───────────────────
class BrainConfig:
    SUBCONSCIOUS_TICK_MS   = 100    # 潜意识循环间隔 (ms)
    REFLEX_PRIORITY        = 99     # 反射弧最高优先级

    # 意识唤醒条件: 任一驱力超过阈值则唤醒LLM
    WAKE_DRIVE_THRESHOLD   = 0.6    # 驱力归一化值 ∈ [0,1]

    # 驱力唤醒检查间隔 (tick数, 1tick≈100ms)
    DRIVE_CHECK_INTERVAL   = 150    # ≈15秒检查一次驱力是否需要唤醒

    # 分级冷却: 不同触发源有不同的最短间隔 (秒)
    # 冷却从 LLM 产生输出后开始计算，而非从唤醒时计算
    COOLDOWN_SURVIVAL       = 10   # 生存危机冷却10秒 (紧急但仍需行动时间)
    COOLDOWN_DRIVE          = 30   # 驱力唤醒后冷却30秒 (让行为有"消化"时间)
    COOLDOWN_BOREDOM        = 60   # 无聊唤醒后冷却60秒
    COOLDOWN_EXTERNAL       = 5    # 外部语音冷却5秒 (人在等回复)


# ─────────────────── 共享常量（跨模块引用）───────────────
class SharedConstants:
    """模拟器与行为执行层之间的共享常量，避免 magic string"""
    DIRECTION_CHARGER = "charger"     # 向充电器移动的方向标识


# ─────────────────── Redis ───────────────────
class RedisKeys:
    STATE_HOMEOSTASIS  = "animat:homeostasis"     # Hash: 稳态指标当前值
    STATE_EMOTION      = "animat:emotion"          # String: 当前主导情感
    STATE_FEELINGS     = "animat:feelings"          # List: 最近感受时间线
    MEMORY_EPISODIC    = "animat:memory:episodic"   # Sorted Set: 情景记忆(按时间+情感权重排序)
    MEMORY_SEMANTIC    = "animat:memory:semantic"   # Hash: 语义记忆
    CORTEX_HISTORY     = "animat:cortex:history"    # List: LLM对话历史
    SENSOR_DERIVATIVES = "animat:sensor:derivatives" # Hash: 传感器导数值(用于变化率计算)
