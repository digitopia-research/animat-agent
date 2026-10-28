"""内稳态引擎核心逻辑测试"""
import pytest

from animat.homeostasis import (
    HomeostasisEngine,
    Vital,
    Drive,
    EmotionCircuit,
    EmotionalState,
    VITAL_CONFIGS,
)
from animat.config import HomeostasisConfig, EmotionConfig
from animat.perception import FeelingType


class TestDriveComputation:
    """驱力计算测试"""

    def test_all_ideal_returns_zero_deviation(self, mock_nervous, mock_memory):
        """所有指标在理想值时，驱力偏差为 0"""
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        drives = engine._compute_drives()
        for d in drives:
            assert d.deviation == pytest.approx(0.0, abs=0.01), f"{d.vital} should be 0 at ideal"

    def test_viability_danger(self, mock_nervous, mock_memory):
        """生存度降至危险阈值以下"""
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        engine._vitals[Vital.VIABILITY] = 10.0
        drives = engine._compute_drives()
        viability_drive = next(d for d in drives if d.vital == Vital.VIABILITY)
        assert viability_drive.is_danger is True
        assert viability_drive.deviation > 0.5

    def test_drives_sorted_by_deviation(self, mock_nervous, mock_memory):
        """驱力按偏差降序排列"""
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        engine._vitals[Vital.EXPLORATION] = 0.0  # 最大偏差
        engine._vitals[Vital.AFFILIATION] = 40.0  # 中等偏差
        engine._vitals[Vital.VIABILITY] = 80.0  # 零偏差
        drives = engine._compute_drives()
        deviations = [d.deviation for d in drives]
        for i in range(len(deviations) - 1):
            assert deviations[i] >= deviations[i + 1], "Drives should be sorted descending"


class TestMaslowArbitration:
    """马斯洛仲裁测试"""

    def test_survival_overrides_social(self, mock_nervous, mock_memory):
        """生存恐惧应压制社交需求"""
        emotions = [
            EmotionalState(
                circuit=EmotionCircuit.PLAY, intensity=0.8,
                cause="boredom", drives=["exploration"], feelings=[]
            ),
            EmotionalState(
                circuit=EmotionCircuit.FEAR, intensity=0.9,
                cause="low battery", drives=["viability"], feelings=[]
            ),
        ]
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        dominant, all_emotions = engine._maslow_arbitrate(emotions)
        assert dominant.circuit == EmotionCircuit.FEAR

    def test_secondary_emotions_preserved(self, mock_nervous, mock_memory):
        """混合情感应保留次导情感"""
        emotions = [
            EmotionalState(
                circuit=EmotionCircuit.FEAR, intensity=0.9,
                cause="charging", drives=["viability"], feelings=[]
            ),
            EmotionalState(
                circuit=EmotionCircuit.CARE, intensity=0.5,
                cause="caress", drives=["affiliation"], feelings=[]
            ),
        ]
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        dominant, all_emotions = engine._maslow_arbitrate(emotions)
        assert len(all_emotions) > 1, "Secondary emotions should be preserved"
        assert all_emotions[0].circuit == EmotionCircuit.FEAR
        # CARE 优先级低一级，强度应衰减
        assert all_emotions[1].intensity < 0.5

    def test_empty_emotions(self, mock_nervous, mock_memory):
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        dominant, all_emotions = engine._maslow_arbitrate([])
        assert dominant is None
        assert all_emotions == []


class TestFeelingDecay:
    """感受时效性衰减测试"""

    def test_decay_basic(self, mock_nervous, mock_memory):
        """基本时效性衰减"""
        import time
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        now = time.time()
        engine._current_feelings = [
            {"type": "caress", "intensity": 0.8, "timestamp": now},
            {"type": "caress", "intensity": 0.8, "timestamp": now - 6.0},  # 2个半衰期
        ]
        weights = engine._get_time_decayed_feelings()
        # 应该有2条感受
        assert len(weights) == 2
        # 按权重降序排列：第一条（刚发生的）> 第二条（6秒前）
        sorted_by_time = sorted(weights, key=lambda x: x[1], reverse=True)
        assert sorted_by_time[0][1] > sorted_by_time[-1][1]


class TestSomaticPromptBuilding:
    """躯体标记 Prompt 构建测试"""

    def test_mixed_emotions_in_prompt(self, mock_nervous, mock_memory):
        """混合情感应该都在 Prompt 中体现"""
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        engine._vitals = {
            Vital.VIABILITY: 15.0,
            Vital.INTEGRITY: 90.0,
            Vital.AFFILIATION: 70.0,
            Vital.COMPETENCE: 70.0,
            Vital.EXPLORATION: 60.0,
        }
        dominant = EmotionalState(
            circuit=EmotionCircuit.FEAR, intensity=0.8,
            cause="电量极低", drives=["viability"], feelings=[]
        )
        secondary = EmotionalState(
            circuit=EmotionCircuit.CARE, intensity=0.2,
            cause="被温柔对待", drives=["affiliation"], feelings=[]
        )
        prompt = engine._build_emotional_prompt(dominant, [dominant, secondary])
        assert "FEAR" in prompt
        assert "CARE" in prompt
        assert "viability" in prompt

    def test_danger_state_in_prompt(self, mock_nervous, mock_memory):
        """危险状态应在 Prompt 中提及"""
        engine = HomeostasisEngine(mock_nervous, mock_memory)
        engine._vitals[Vital.VIABILITY] = 10.0
        dominant = EmotionalState(
            circuit=EmotionCircuit.FEAR, intensity=0.7,
            cause="low battery", drives=["viability"], feelings=[]
        )
        prompt = engine._build_emotional_prompt(dominant, [dominant])
        assert "极度危险" in prompt
