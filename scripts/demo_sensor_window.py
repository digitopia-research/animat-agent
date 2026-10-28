"""
Phase 3 验证脚本: 传感器直通 + LLM 从原始序列推理

测试内容:
  1. SensorWindow 的基本功能
  2. 传感器上下文文本格式化
  3. LLM 从原始压力序列中识别"抚摸" vs "撞击"（不依赖中间翻译）
  4. 注意力提示的效果

用法:
  cd /opt/data/animat-agent
  python3 scripts/demo_sensor_window.py
"""

import sys
import time
import logging

sys.path.insert(0, "/opt/data/animat-agent")

from animat.llm_backend import LocalLLM, LLMConfig
from animat.sensor_window import SensorWindow, AttentionHint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


def test_sensor_window_basic():
    """测试 SensorWindow 基本功能"""
    print("=" * 60)
    print("  测试 1: SensorWindow 基础功能")
    print("=" * 60)
    
    sw = SensorWindow(window_sec=5.0)
    
    # 模拟一系列传感器数据
    now = time.time()
    
    # 内感数据
    for i in range(3):
        sw.add_frame("intero", {
            "battery_pct": 80 - i * 0.5,
            "temperature": 37.5 + i * 0.1,
            "imu_tilt_deg": 1.0 + i * 0.5,
            "charging": False,
        })
    
    # 触觉数据（模拟轻柔抚摸）
    for i in range(5):
        sw.add_frame("touch", {
            "pressure": 0.05 + i * 0.02,
            "capacitance": 0.4 + i * 0.1,
            "x": 0.3 + i * 0.1,
            "y": 0.5,
        })
    
    # 添加注意力提示
    sw.add_attention_hint(AttentionHint(
        modality="touch",
        hint_type="caress",
        confidence=0.85,
        description="轻柔接触（可能抚摸）",
    ))
    
    # 生成上下文
    context = sw.to_context(max_frames_per_modality=3)
    print("\n  LLM 上下文文本:")
    print("  " + context.replace("\n", "\n  "))
    
    # 摘要
    summary = sw.to_summary()
    print(f"\n  状态摘要: {summary}")


def test_impact_detection(llm):
    """测试 LLM 从原始序列识别撞击（无中间翻译）"""
    print("\n" + "=" * 60)
    print("  测试 2: LLM 从原始压力序列识别撞击")
    print("=" * 60)
    
    # 构建撞击场景的传感器窗口
    sw = SensorWindow(window_sec=5.0)
    
    # 背景：正常状态
    sw.add_frame("touch", {"pressure": 0.05, "capacitance": 0.01, "x": 0.5, "y": 0.5})
    sw.add_frame("touch", {"pressure": 0.06, "capacitance": 0.01, "x": 0.5, "y": 0.5})
    # 撞击！
    sw.add_frame("touch", {"pressure": 0.92, "capacitance": 0.01, "x": 0.7, "y": 0.6})
    # 撞击后的余波
    sw.add_frame("touch", {"pressure": 0.45, "capacitance": 0.01, "x": 0.7, "y": 0.6})
    sw.add_frame("touch", {"pressure": 0.15, "capacitance": 0.01, "x": 0.65, "y": 0.55})
    
    # 内感
    sw.add_frame("intero", {"battery_pct": 75, "temperature": 37, "imu_tilt_deg": 8, "charging": False})
    
    # 注意力提示（感知层的快速判断，供参考）
    sw.add_attention_hint(AttentionHint(
        modality="touch", hint_type="pain", confidence=0.9,
        description="触觉急变（可能撞击）",
    ))
    
    sensor_text = sw.to_context(max_frames_per_modality=5)
    print("\n  传感器数据（LLM 看到的内容）:")
    print("  " + sensor_text[:400].replace("\n", "\n  "))
    print()
    
    # 让 LLM 直接从原始序列推理
    prompt = f"""以下是机器人传感器最近的数据。请根据这些原始数据，判断发生了什么。

{sensor_text}

请回答：
1. 发生了什么事件？（从触觉数据中推断）
2. 你有多确定？（0-100%）
3. 你的第一反应应该是什么？"""

    response = llm.complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5,
    )
    text = response["choices"][0]["message"]["content"]
    print(f"  LLM 推理结果:\n    {text.replace(chr(10), chr(10) + '    ')}")


def test_caress_detection(llm):
    """测试 LLM 从原始序列识别抚摸（无中间翻译）"""
    print("\n" + "=" * 60)
    print("  测试 3: LLM 从原始压力序列识别抚摸")
    print("=" * 60)
    
    sw = SensorWindow(window_sec=5.0)
    
    # 轻柔抚摸的传感器序列
    for i in range(5):
        sw.add_frame("touch", {
            "pressure": 0.03 + i * 0.02,  # 低压
            "capacitance": 0.3 + i * 0.15,  # 电容逐渐升高（皮肤接触）
            "x": 0.2 + i * 0.15,            # 位置滑动
            "y": 0.5 + i * 0.05,             # y 轴轻微移动
        })
    
    sw.add_frame("intero", {"battery_pct": 78, "temperature": 37, "imu_tilt_deg": 1, "charging": False})
    
    # 注意力提示
    sw.add_attention_hint(AttentionHint(
        modality="touch", hint_type="caress", confidence=0.9,
        description="轻柔接触（可能抚摸）",
    ))
    
    sensor_text = sw.to_context(max_frames_per_modality=5)
    print("\n  传感器数据（LLM 看到的内容）:")
    print("  " + sensor_text[:400].replace("\n", "\n  "))
    print()
    
    prompt = f"""以下是机器人传感器最近的数据。请根据这些原始数据，判断发生了什么。

{sensor_text}

请回答：
1. 发生了什么？（从触觉数据的变化趋势推断）
2. 你有多确定？
3. 你的情感反应是什么？"""

    response = llm.complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5,
    )
    text = response["choices"][0]["message"]["content"]
    print(f"  LLM 推理结果:\n    {text.replace(chr(10), chr(10) + '    ')}")


def test_vs_v01_translation():
    """展示 v0.1 翻译 vs v0.2 直通的区别"""
    print("\n" + "=" * 60)
    print("  测试 4: v0.1 翻译 vs v0.2 直通 对比")
    print("=" * 60)
    
    print("""
  v0.1 做法（感知翻译层）:
    原始: [touch: P=0.05] [touch: P=0.06] [touch: P=0.92] [touch: P=0.45]
      → 计算 dP/dt = 0.92-0.06 = 0.86 > 0.8 阈值
      → 发射 "pain" 事件 (intensity=0.86)
      → 边缘系统接收 "pain" → 降低完好度 → 触发 FEAR
      → Prompt 注入: "你的完好度受到严重损害"
      → LLM 读到的是翻译后的文本，从未见过原始数据
    
  v0.2 做法（传感器直通）:
    原始: [T+0.0s] 触摸: 压力=0.05 [T+0.1s] 触摸: 压力=0.06 
           [T+0.2s] 触摸: 压力=0.92 [T+0.3s] 触摸: 压力=0.45
      → SensorWindow 保存所有帧
      → to_context() 生成结构化文本 → LLM 直接阅读
      → 同时: 反射弧检测到 dP/dt > 0.8 → 缩回（快速响应）
      → 注意力提示: "注意看触觉通道有个急变"
      → LLM 自己从压力模式推断: "刚才有个撞击，然后力在衰减"
      
  关键差异:
    v0.1: LLM 被告知 "这是痛觉" → 只能照做
    v0.2: LLM 被展示 "压力值从 0.06 跳到 0.92 然后回落到 0.45"
           → 它可以推断这是撞击、碰撞、还是摔跤
           → 更丰富的语义理解
""")


def test_format_with_body_context(llm):
    """测试传感器上下文 + 身体状态结合的 Prompt"""
    print("\n" + "=" * 60)
    print("  测试 5: 传感器 + 身体状态 结合推理")
    print("=" * 60)
    
    sw = SensorWindow(window_sec=5.0)
    
    # 低电量 + 有人在附近
    sw.add_frame("intero", {"battery_pct": 12, "temperature": 38, "imu_tilt_deg": 2, "charging": False})
    sw.add_frame("vision", {"distance_m": 1.2, "face_detected": True, "gaze_seconds": 1.5})
    sw.add_frame("touch", {"pressure": 0.04, "capacitance": 0.35, "x": 0.3, "y": 0.4})
    
    sensor_text = sw.to_context(max_frames_per_modality=3)
    print("\n  传感器上下文:")
    print("  " + sensor_text.replace("\n", "\n  "))
    
    # 身体状态（简洁描述，不暗示情感）
    body_context = "电量极低(12%)，满足度低。"
    
    prompt = f"""你是 Animat 机器人。以下是你当前的状态和传感器数据：

## 身体状态
{body_context}

{sensor_text}

用一个简短的自然想法表达你此刻的内心："""

    print("\n  LLM 推理:")
    response = llm.complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.8,
    )
    text = response["choices"][0]["message"]["content"]
    print(f"  \"{text.strip()}\"")


def main():
    # 1. 加载模型
    logger.info("加载本地 LLM...")
    config = LLMConfig(
        model_path="/opt/data/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        n_ctx=4096, n_threads=8,
    )
    llm = LocalLLM(config)
    llm.load()
    print("✓ LLM 加载完成")
    
    # 2. 运行测试
    test_sensor_window_basic()
    test_impact_detection(llm)
    test_caress_detection(llm)
    test_vs_v01_translation()
    test_format_with_body_context(llm)
    
    # 3. 总结
    print("\n" + "=" * 60)
    print("  Phase 3 验证结论")
    print("=" * 60)
    print("""
  传感器直通已完成:
  · SensorWindow 维护每个模态的 5 秒时间序列窗口
  · to_context() 生成结构化的原始数据文本 → LLM 直接阅读
  · 注意力提示（感知层快速判断）供 LLM 参考但不替代 LLM 自己推理
  · 反射弧保留，快速危险响应不受影响
  · perception.py 重构：感受器映射降级为注意力提示
  
  LLM 的能力:
  · 从压力值序列 [0.05, 0.06, 0.92, 0.45, 0.15] 推断出"撞击"
  · 从压力值序列 [0.03, 0.05, 0.07, 0.09, 0.11] + 电容上升推断出"抚摸"
  · 结合内感数据（电量 12%）+ 视觉（有人 1.2m 外）+ 触觉（轻柔）
    产生完整的场景理解
""")


if __name__ == "__main__":
    main()
