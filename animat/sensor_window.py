"""
传感器时间序列窗口 — SensorWindow

核心思想（v0.2）:
  传感器原始数据是 LLM 上下文的一等公民，不经过感知翻译层。
  
  LLM 应当直接从压力/声音/距离的时序模式中推断"发生了什么"——
  就像人脑中的感觉皮层直接处理原始信号，不等待某个子系统替你总结。

与 v0.1 的区别:
  v0.1:   压力值 → 计算 dP/dt → 如果 > 0.8 → 发射 "pain" 事件
  v0.2:   压力值序列直接写入 LLM 上下文 → LLM 自己判断"刚才那个力变化很大"
          同时保留反射弧（快速危险响应绕过 LLM）
          感受器映射降级为"注意力提示"（告诉 LLM 注意看压力通道）
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("animat.sensor_window")


@dataclass
class SensorFrame:
    """单帧传感器数据"""
    timestamp: float
    modality: str               # touch / audio / vision / intero
    values: dict                # 原始数值
    raw_payload: dict = field(default_factory=dict)  # 完整 MQTT payload

    def to_text(self) -> str:
        """转换为 LLM 可读的单行文本"""
        ts = f"T+{self.timestamp % 1000:.1f}s"  # 短时间戳
        
        if self.modality == "touch":
            return (
                f"[{ts}] 触摸: 压力={self.values.get('pressure', 0):.2f}, "
                f"电容={self.values.get('capacitance', 0):.2f}, "
                f"位置=({self.values.get('x', 0):.1f},{self.values.get('y', 0):.1f})"
            )
        elif self.modality == "audio":
            return (
                f"[{ts}] 声音: 音量={self.values.get('volume_db', 0):.1f}dB, "
                f"基频={self.values.get('fundamental_freq', 0):.0f}Hz"
            )
        elif self.modality == "vision":
            return (
                f"[{ts}] 视觉: 距离={self.values.get('distance_m', 10):.1f}m, "
                f"人脸={'有' if self.values.get('face_detected') else '无'}, "
                f"凝视={self.values.get('gaze_seconds', 0):.1f}s"
            )
        elif self.modality == "intero":
            parts = [f"电量={self.values.get('battery_pct', 100):.0f}%"]
            if self.values.get('charging'):
                parts.append("充电中")
            parts.append(f"温度={self.values.get('temperature', 37):.0f}°C")
            parts.append(f"倾角={self.values.get('imu_tilt_deg', 0):.0f}°")
            return f"[{ts}] 身体: " + ", ".join(parts)
        else:
            return f"[{ts}] {self.modality}: {self.values}"


@dataclass
class AttentionHint:
    """感知层的注意力提示（替代 v0.1 的感受器映射）"""
    modality: str               # 传感器模态
    hint_type: str              # pain / caress / startle / approach / ...
    confidence: float           # 置信度 [0, 1]
    description: str            # 人类可读描述
    timestamp: float = 0.0      # 创建时间戳（用于过期清理）

    def to_text(self) -> str:
        return f"[注意: {self.modality}通道可能发生了{self.description} (置信度{self.confidence:.0%})]"


class SensorWindow:
    """
    传感器时间序列窗口
    
    维护最近 N 秒的传感器帧，提供给 LLM 作为第一手感知材料。
    
    设计:
    - 环形缓冲 (deque)，每个模态独立窗口
    - 提供 to_context() 方法生成 LLM 可读的上下文文本
    - 提供 get_recent_changes() 计算最近一段时间的统计变化
    """

    # 默认窗口大小
    DEFAULT_WINDOW_SEC = 5.0
    MAX_FRAMES_PER_MODALITY = 50

    def __init__(self, window_sec: float = DEFAULT_WINDOW_SEC):
        self.window_sec = window_sec
        self._frames: dict[str, deque[SensorFrame]] = {
            "touch": deque(maxlen=self.MAX_FRAMES_PER_MODALITY),
            "audio": deque(maxlen=self.MAX_FRAMES_PER_MODALITY),
            "vision": deque(maxlen=self.MAX_FRAMES_PER_MODALITY),
            "intero": deque(maxlen=self.MAX_FRAMES_PER_MODALITY),
        }
        self._attention_hints: list[AttentionHint] = []
        self._latest_frame: Optional[SensorFrame] = None

    # ── 数据写入 ──
    def add_frame(self, modality: str, payload: dict):
        """添加一帧传感器数据"""
        now = time.time()
        
        # 提取标准字段
        values = {}
        if modality == "touch":
            values = {
                "pressure": payload.get("pressure", 0),
                "capacitance": payload.get("capacitance", 0),
                "x": payload.get("x", 0),
                "y": payload.get("y", 0),
            }
        elif modality == "audio":
            values = {
                "volume_db": payload.get("volume_db", 0),
                "fundamental_freq": payload.get("fundamental_freq", 0),
            }
        elif modality == "vision":
            values = {
                "distance_m": payload.get("distance_m", 10),
                "face_detected": payload.get("face_detected", False),
                "gaze_seconds": payload.get("gaze_seconds", 0),
            }
        elif modality == "intero":
            values = {
                "battery_pct": payload.get("battery_pct", 100),
                "temperature": payload.get("temperature", 37),
                "imu_tilt_deg": payload.get("imu_tilt_deg", 0),
                "charging": payload.get("charging", False),
            }
        
        frame = SensorFrame(
            timestamp=now,
            modality=modality,
            values=values,
            raw_payload=payload,
        )
        
        if modality in self._frames:
            self._frames[modality].append(frame)
        
        self._latest_frame = frame
        self._trim()

    def add_attention_hint(self, hint: AttentionHint):
        """添加注意力提示（来自感知层的快速判断）"""
        if hint.timestamp == 0.0:
            hint.timestamp = time.time()
        self._attention_hints.append(hint)
        # 只保留最近 5 条提示
        if len(self._attention_hints) > 5:
            self._attention_hints = self._attention_hints[-5:]

    def _trim(self):
        """移除超出窗口的旧帧和旧的 attention hint"""
        now = time.time()
        cutoff = now - self.window_sec
        for modality in self._frames:
            while self._frames[modality] and self._frames[modality][0].timestamp < cutoff:
                self._frames[modality].popleft()
        # attention hint 也随时间过期（5 秒窗口）
        self._attention_hints = [
            h for h in self._attention_hints
            if getattr(h, 'timestamp', now) > cutoff
        ]

    # ── 数据读取 ──
    def get_recent(self, modality: str, n: int = 5) -> list[SensorFrame]:
        """获取某模态最近 n 帧"""
        frames = list(self._frames.get(modality, []))
        return frames[-n:]

    def get_latest(self, modality: str) -> Optional[SensorFrame]:
        """获取某模态最新帧"""
        frames = self._frames.get(modality, deque())
        return frames[-1] if frames else None

    def get_all_modalities(self) -> list[str]:
        """获取有数据的模态列表"""
        return [m for m, f in self._frames.items() if len(f) > 0]

    # ── LLM 上下文生成 ──
    def to_context(self, max_frames_per_modality: int = 3) -> str:
        """
        生成供 LLM 阅读的传感器上下文文本
        
        格式: 结构化的时间序列，每帧一行
        
        这是 Phase 3 的核心接口: LLM 直接阅读传感器原始数据，
        不经过"pressure > 0.8 → pain"的翻译。
        """
        lines = []
        
        # 注意力提示（感知层的快速判断，供 LLM 参考但不替代自己的判断）
        if self._attention_hints:
            lines.append("## ⚡ 注意力提示")
            for hint in self._attention_hints[-3:]:  # 最近 3 条
                lines.append(hint.to_text())
            lines.append("")
        
        # 传感器数据
        modality_order = ["intero", "touch", "vision", "audio"]
        for modality in modality_order:
            frames = self.get_recent(modality, max_frames_per_modality)
            if not frames:
                continue
            
            label = {
                "intero": "## 🫀 身体内部感觉 (电量充足=80-100%, 正常温度=35-40°C)",
                "touch": "## ✋ 触觉 (<0.2=轻柔, >0.7=撞击)",
                "vision": "## 👁 视觉",
                "audio": "## 👂 听觉",
            }.get(modality, f"## {modality}")
            
            lines.append(label)
            for frame in frames:
                lines.append(frame.to_text())
            lines.append("")
        
        if not lines:
            return "（暂无传感器数据）"
        
        return "\n".join(lines)

    def to_summary(self) -> str:
        """生成简短的状态摘要（用于 System Prompt 而非完整上下文）"""
        parts = []
        
        # 内感
        intero = self.get_latest("intero")
        if intero:
            batt = intero.values.get("battery_pct", 100)
            charging = intero.values.get("charging", False)
            tilt = intero.values.get("imu_tilt_deg", 0)
            if batt < 20:
                parts.append(f"电量极低({batt:.0f}%)！")
            elif batt < 50:
                parts.append(f"电量偏低({batt:.0f}%)")
            if charging:
                parts.append("正在充电")
            if tilt > 20:
                parts.append("身体倾斜")
        
        # 触觉
        touch_frames = self.get_recent("touch", 3)
        if touch_frames:
            # 检测最近是否有高压事件
            high_pressure = any(
                f.values.get("pressure", 0) > 0.6 for f in touch_frames
            )
            low_pressure = all(
                f.values.get("pressure", 0) < 0.2 for f in touch_frames
            )
            if high_pressure:
                parts.append("感受到压力")
            elif low_pressure:
                parts.append("触觉轻柔")
        
        # 视觉
        vision = self.get_latest("vision")
        if vision:
            if vision.values.get("face_detected"):
                dist = vision.values.get("distance_m", 1)
                if dist < 1:
                    parts.append("有人很近")
                else:
                    parts.append("附近有人")
            else:
                # 检查是否有长时间无人的情况
                pass  # 由 homeostasis 处理归属度衰减
        
        return "；".join(parts) if parts else "感官正常"

    def get_recent_changes(self, modality: str, key: str) -> dict:
        """
        计算最近变化统计（用于感知层的注意力提示生成）
        
        Returns:
            {"min": ..., "max": ..., "avg": ..., "trend": "rising"/"falling"/"stable"}
        """
        frames = self.get_recent(modality, 10)
        if not frames:
            return {"trend": "none"}
        
        values = [f.values.get(key, 0) for f in frames if key in f.values]
        if not values:
            return {"trend": "none"}
        
        result = {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "count": len(values),
        }
        
        # 趋势判断
        if len(values) >= 2:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            if avg_second - avg_first > 0.1 * abs(avg_first or 1):
                result["trend"] = "rising"
            elif avg_first - avg_second > 0.1 * abs(avg_first or 1):
                result["trend"] = "falling"
            else:
                result["trend"] = "stable"
        else:
            result["trend"] = "stable"
        
        return result

    def clear(self):
        """清空所有缓冲"""
        for modality in self._frames:
            self._frames[modality].clear()
        self._attention_hints.clear()
        self._latest_frame = None
