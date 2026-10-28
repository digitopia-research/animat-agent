"""
注意力调制器 — 数字内分泌系统 v2

核心思想:
  身体状态不生成 Prompt 文本，而是直接调制 LLM 的 token 采样概率。
  模拟神经递质对神经元兴奋性的调控 —— 多巴胺让"值得追求的"更突出，
  去甲肾上腺素让"危险的"被优先加工。

实现方式:
  - 维护一组 token 语义聚类（危险/社交/探索/舒适/敌对）
  - 根据 BodyState 向量计算每个聚类的 bias
  - 注入 llama-cpp-python 的 logit_bias 参数

关键调参:
  - bias 范围: [0, 5.0]（经测试，>10 会导致 token 重复）
  - 正 bias: 让 token 更容易被选中
  - 负 bias: 抑制 token（但不能完全阻止，只是降低概率）
"""

import math
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("animat.attention_modulator")


@dataclass
class TokenCluster:
    """一个语义聚类——主题相近的 token 组"""
    name: str
    tokens: list[int] = field(default_factory=list)
    seed_words: list[str] = field(default_factory=list)  # 种子词（用于文档）
    bias_range: tuple[float, float] = (-3.0, 5.0)  # (最小, 最大) bias


@dataclass  
class BodyState:
    """
    身体状态的向量化表示
    
    所有维度值域 [0, 1]，由 homeostasis.py 的稳态指标计算
    """
    viability: float = 0.8      # 生存度（电量/网络状态）
    integrity: float = 0.9      # 完好度（受力/姿态）
    affiliation: float = 0.7    # 归属度（社交满足）
    competence: float = 0.7     # 胜任度（任务成功）
    exploration: float = 0.6    # 探索度（新奇度）
    
    # 衍生指标
    @property
    def arousal(self) -> float:
        """唤醒度：驱力的加权和"""
        drive_deviations = [
            1.0 - self.viability,
            1.0 - self.integrity,
            1.0 - self.affiliation,
            1.0 - self.competence,
            1.0 - self.exploration,
        ]
        # 生存和安全权重更高
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        return sum(d * w for d, w in zip(drive_deviations, weights))

    @property
    def valence(self) -> float:
        """情感效价：-1(极度负面) ~ +1(极度正面)"""
        # 简单模型：稳态总和 vs 理想总和
        ideal = 3.7  # 0.8+0.9+0.7+0.7+0.6
        current = self.viability + self.integrity + self.affiliation + self.competence + self.exploration
        deviation = (ideal - current) / ideal
        return max(-1.0, min(1.0, 1.0 - deviation * 2))

    @property
    def fear_level(self) -> float:
        """恐惧水平 = 生存威胁 + 完好威胁"""
        survival_threat = max(0, 1.0 - self.viability) 
        integrity_threat = max(0, 1.0 - self.integrity)
        return min(1.0, (survival_threat * 0.7 + integrity_threat * 0.3) ** 0.5)

    @property
    def loneliness_level(self) -> float:
        """孤独水平"""
        return max(0, 1.0 - self.affiliation)

    @property
    def boredom_level(self) -> float:
        """无聊水平"""
        return max(0, 1.0 - self.exploration)

    @classmethod
    def from_vitals(cls, vitals: dict) -> "BodyState":
        """从 homeostasis.py 的稳态指标字典创建"""
        return cls(
            viability=vitals.get("viability", 0.8) / 100.0,
            integrity=vitals.get("integrity", 0.9) / 100.0,
            affiliation=vitals.get("affiliation", 0.7) / 100.0,
            competence=vitals.get("competence", 0.7) / 100.0,
            exploration=vitals.get("exploration", 0.6) / 100.0,
        )

    def to_dict(self) -> dict:
        return {
            "viability": self.viability,
            "integrity": self.integrity,
            "affiliation": self.affiliation,
            "competence": self.competence,
            "exploration": self.exploration,
            "arousal": self.arousal,
            "valence": self.valence,
            "fear_level": self.fear_level,
            "loneliness_level": self.loneliness_level,
            "boredom_level": self.boredom_level,
        }


class AttentionModulator:
    """
    注意力调制器 — 根据身体状态计算 token 概率偏移
    
    使用方式:
      modulator = AttentionModulator(llm)
      body_state = BodyState(viability=0.08)  # 电量仅 8%
      bias = modulator.compute_logit_bias(body_state)
      response = llm.complete(messages, logit_bias=bias)
    
    设计原则:
      - 不做硬阻断（不像 v0.1 的意图池封锁）
      - 只做概率偏移，让 LLM "更容易"或"更难"想到某些方向
      - 偏移量连续映射，不是二分开关
      - bias 力度控制在 [-3, +5] 区间（测试表明 >10 导致 token 重复）
    """

    # 聚类定义
    CLUSTER_DEFS = {
        "danger": {
            "seed_words": ["危险", "充电", "电", "关机", "死亡", "帮助", "紧急", 
                          "逃跑", "保护", "躲", "求救", "故障", "警告"],
            "bias_range": (-3.0, 5.0),
        },
        "social": {
            "seed_words": ["你好", "聊天", "陪伴", "朋友", "一起", "开心", "玩",
                          "说话", "见面", "互动", "打招呼", "分享", "交流"],
            "bias_range": (-3.0, 4.0),
        },
        "explore": {
            "seed_words": ["探索", "好奇", "那边", "看看", "新", "有趣", "好玩的",
                          "发现", "走走", "逛逛", "冒险", "学习", "尝试"],
            "bias_range": (-3.0, 4.0),
        },
        "comfort": {
            "seed_words": ["温暖", "舒服", "安全", "休息", "安静", "抚摸",
                          "温柔", "放松", "安逸", "安心", "平静", "舒适"],
            "bias_range": (-2.0, 3.0),
        },
        "hostility": {
            "seed_words": ["滚开", "不要", "停下", "讨厌", "攻击", "防御",
                          "闭嘴", "退后", "愤怒", "生气", "讨厌"],
            "bias_range": (-3.0, 2.0),
        },
    }

    def __init__(self, llm=None):
        """
        Args:
            llm: llama-cpp-python Llama 实例（用于 tokenize），或一个暴露
                 tokenize(text: bytes, add_bos: bool) -> list[int] 可调用对象
                 （如 CloudLLM）。如果为 None，需要调用 set_tokenizer() 后再使用。
        """
        self._llm = llm
        self._tokenize_fn = None
        self._clusters: dict[str, TokenCluster] = {}
        self._token_to_cluster: dict[int, str] = {}  # 反向映射
        self._initialized = False

    def set_tokenizer(self, llm):
        """
        设置/更换 tokenizer。

        llm 可以是:
          - llama-cpp-python Llama 实例（有 .tokenize 方法）
          - 任意暴露 tokenize(text: bytes, add_bos: bool) -> list[int] 的对象（如 CloudLLM）
          - 直接是一个可调用函数 fn(text: bytes, add_bos: bool) -> list[int]
        """
        self._llm = llm
        self._tokenize_fn = None
        self._initialized = False

    def _tokenize(self, word: str) -> list[int]:
        """统一的 tokenize 入口，兼容 llama-cpp-python 与 CloudLLM/tiktoken"""
        if self._tokenize_fn is None:
            if callable(self._llm) and not hasattr(self._llm, "tokenize"):
                self._tokenize_fn = self._llm
            elif hasattr(self._llm, "tokenize"):
                target = self._llm
                self._tokenize_fn = lambda text, add_bos=False: target.tokenize(text, add_bos=add_bos)
            else:
                raise RuntimeError("AttentionModulator needs a tokenizer. Call set_tokenizer() first.")
        return self._tokenize_fn(word.encode("utf-8"), add_bos=False)

    def _ensure_clusters(self):
        """懒初始化聚类（需要 tokenizer）"""
        if self._initialized:
            return
        if self._llm is None:
            raise RuntimeError("AttentionModulator needs a tokenizer. Call set_tokenizer() first.")
        
        for cluster_name, defn in self.CLUSTER_DEFS.items():
            cluster = TokenCluster(
                name=cluster_name,
                seed_words=defn["seed_words"],
                bias_range=defn["bias_range"],
            )
            for word in defn["seed_words"]:
                # _tokenize 失败（如 tokenizer 未安装）应冒泡，便于定位环境配置问题
                tokens = self._tokenize(word)
                if not tokens:
                    continue
                for tid in tokens:
                    if tid not in self._token_to_cluster:
                        cluster.tokens.append(tid)
                        self._token_to_cluster[tid] = cluster_name
            
            # 去重
            cluster.tokens = list(set(cluster.tokens))
            self._clusters[cluster_name] = cluster
            logger.debug(
                f"Cluster '{cluster_name}': {len(cluster.tokens)} tokens "
                f"from {len(defn['seed_words'])} seed words"
            )
        
        self._initialized = True
        total_tokens = sum(len(c.tokens) for c in self._clusters.values())
        logger.info(f"AttentionModulator initialized: {len(self._clusters)} clusters, {total_tokens} tokens")

    def compute_logit_bias(self, body_state: BodyState) -> dict[int, float]:
        """
        根据身体状态计算 logit bias
        
        每个聚类的 bias 由身体状态的多维度非线性映射决定。
        
        映射规则:
          - fear_level 高 → danger tokens +, social/explore tokens -
          - loneliness 高 → social tokens +, hostility tokens -
          - boredom 高 → explore tokens +, comfort tokens -
          - comfort（正向）→ comfort tokens +, hostility tokens -
          - 负 valence → hostility tokens +
          - arousal 调节整体 bias 幅度（高唤醒 = bias 更强）
        """
        self._ensure_clusters()
        bias = {}
        
        fs = body_state.fear_level
        loneliness = body_state.loneliness_level
        boredom = body_state.boredom_level
        arousal = body_state.arousal
        valence = body_state.valence
        comfort_level = (
            body_state.affiliation * 0.6 + 
            body_state.integrity * 0.4
        )
        
        # ── 计算每个聚类的 bias ──
        cluster_biases = {}
        
        # danger: 恐惧 + 低生存度 → 提升
        danger_bias = fs * 3.5 - 0.3
        danger_bias = max(-3.0, min(4.0, danger_bias))
        cluster_biases["danger"] = danger_bias
        
        # social: 孤独想社交，恐惧不想社交
        social_bias = loneliness * 3.5 - fs * 3.0
        if valence > 0.3:  # 正面情绪时社交意愿更高
            social_bias += valence * 1.5
        social_bias = max(-3.0, min(4.0, social_bias))
        cluster_biases["social"] = social_bias
        
        # explore: 无聊想探索，恐惧不想探索
        explore_bias = boredom * 4.0 - fs * 4.0
        explore_bias = max(-3.0, min(4.0, explore_bias))
        cluster_biases["explore"] = explore_bias
        
        # comfort: 需要安抚时升高
        comfort_bias = (1.0 - comfort_level) * 2.5 - fs * 2.0
        if valence < -0.3: 
            comfort_bias += 1.0  # 负面情绪时渴望舒适
        comfort_bias = max(-2.0, min(3.0, comfort_bias))
        cluster_biases["comfort"] = comfort_bias
        
        # hostility: 恐惧+高完好威胁 → 防御性敌意
        integrity_threat = max(0, 1.0 - body_state.integrity)
        hostility_bias = fs * integrity_threat * 2.5 - comfort_level * 2.0
        if valence < -0.5:
            hostility_bias += 1.0
        hostility_bias = max(-3.0, min(2.0, hostility_bias))
        cluster_biases["hostility"] = hostility_bias
        
        # ── 唤醒度调节整体幅度 ──
        arousal_scale = 0.5 + arousal * 0.5  # [0.5, 1.0]
        
        # ── 构建逐 token bias ──
        for cluster_name, cluster in self._clusters.items():
            raw_bias = cluster_biases.get(cluster_name, 0.0)
            scaled_bias = raw_bias * arousal_scale
            
            # 跳过接近零的 bias
            if abs(scaled_bias) < 0.1:
                continue
            
            for tid in cluster.tokens:
                bias[tid] = round(scaled_bias, 2)
        
        logger.debug(
            f"BodyState(v={body_state.viability:.2f}, i={body_state.integrity:.2f}, "
            f"a={body_state.affiliation:.2f}, e={body_state.exploration:.2f}) -> "
            f"biases: danger={danger_bias:.1f}, social={social_bias:.1f}, "
            f"explore={explore_bias:.1f}, comfort={comfort_bias:.1f}, "
            f"hostility={hostility_bias:.1f}  (arousal={arousal:.2f})"
        )
        
        return bias

    def get_cluster_summary(self, body_state: BodyState) -> dict:
        """获取当前身体状态下的聚类 bias 摘要（用于调试和可视化）"""
        self._ensure_clusters()
        bias = self.compute_logit_bias(body_state)
        
        # 按聚类汇总
        summary = {}
        for cluster_name, cluster in self._clusters.items():
            cluster_tokens = [t for t in cluster.tokens if t in bias]
            if cluster_tokens:
                values = [bias[t] for t in cluster_tokens]
                summary[cluster_name] = {
                    "avg_bias": round(sum(values) / len(values), 2),
                    "max_bias": max(values),
                    "min_bias": min(values),
                    "token_count": len(cluster_tokens),
                }
        
        return summary
