# 本地小模型的工作原理

本文档详细分析 Animat Agent 中本地小模型（Local LLM）的工作机制，覆盖模型加载、推理接口、注意力调制（Phase 2）与意识流循环（Phase 1）四个层面。

---

## 一、模型加载与封装（`animat/llm_backend.py`）

本地模型通过 **`llama-cpp-python`** 加载 GGUF 量化模型（默认 `Qwen2.5-1.5B-Instruct` Q4_K_M，约 1GB，纯 CPU 推理）：

```python
self._llm = llama_cpp.Llama(
    model_path=self.config.model_path,
    n_ctx=self.config.n_ctx,        # 4096 上下文窗口
    n_threads=self.config.n_threads, # 8 线程
    n_gpu_layers=self.config.n_gpu_layers,  # 0 = 纯 CPU
    verbose=False,
)
```

封装层 `LocalLLM` 提供两类推理接口，且**接口与 OpenAI 兼容**，方便未来切换到云端或宿主机的 `llama-server`：

| 接口 | 模式 | 用途 |
|------|------|------|
| `complete()` | 非流式 | 回复人类输入（单轮对话） |
| `stream_complete()` / `stream_think()` | 流式 | 逐 token 产出，意识流核心 |

关键配置（`LLMConfig`）：

- `temperature = 0.7`（思维阶段乘 1.2 以鼓励多样性）
- `top_p = 0.9`
- `cache_prompt = True`（KV Cache 持久化，减少重复推理）
- `n_gpu_layers = 0`（默认 CPU；可设为 >0 卸载到 GPU）

`stream_think()` 与 `stream_complete()` 的区别：

- 输入不是对话，而是 `system_prompt + 上下文（记忆/身体感觉）`
- 输出是"内部思维"而非"对话回复"
- 使用略高的 temperature 鼓励多样性

---

## 二、注意力调制：身体状态如何"染色"思维（Phase 2）

这是本工程最独特的机制——**不修改 LLM 权重，也不写"你很害怕"的 prompt，而是直接偏置 token 采样概率**。

### 2.1 BodyState 向量化（`attention_modulator.py`）

把 5 个稳态维度（值域 [0,1]）映射为衍生指标：

```python
@dataclass
class BodyState:
    viability: float = 0.8      # 生存度（电量/网络状态）
    integrity: float = 0.9      # 完好度（受力/姿态）
    affiliation: float = 0.7    # 归属度（社交满足）
    competence: float = 0.7     # 胜任度（任务成功）
    exploration: float = 0.6    # 探索度（新奇度）
```

衍生指标：

- `fear_level`、`loneliness_level`、`boredom_level`（各自 [0,1]）
- `arousal`（唤醒度，加权驱力偏差）
- `valence`（情感效价 -1 ~ +1）

### 2.2 语义聚类 token 化

定义 5 个 token 簇（`danger / social / explore / comfort / hostility`），用种子词经 `llm.tokenize()` 转成 token ID，建立 `token_id → cluster` 反向映射：

```python
for word in defn["seed_words"]:
    tokens = self._llm.tokenize(word.encode("utf-8"), add_bos=False)
    for tid in tokens:
        if tid not in self._token_to_cluster:
            cluster.tokens.append(tid)
            self._token_to_cluster[tid] = cluster_name
```

### 2.3 计算 logit bias

`compute_logit_bias()` 把身体状态**非线性映射**到每个簇的偏置值，例如：

```python
danger_bias  = fs * 3.5 - 0.3              # 恐惧高 → 危险相关词更易出现
social_bias  = loneliness * 3.5 - fs * 3.0 # 孤独想社交、恐惧不想社交
explore_bias = boredom * 4.0 - fs * 4.0    # 无聊想探索、恐惧不想探索
comfort_bias = (1.0 - comfort_level) * 2.5 - fs * 2.0
hostility_bias = fs * integrity_threat * 2.5 - comfort_level * 2.0
```

最后用 `arousal_scale = 0.5 + arousal * 0.5` 缩放整体幅度。

### 2.4 注入推理

`logit_bias` 字典（`{token_id: bias}`）直接传给 `llm.create_chat_completion(logit_bias=...)`。bias 限制在 `[-3, +5]`（>10 会导致 token 重复）。

> **效果**：LLM 不会被告知"你害怕"，而是**自然地更多想到危险相关词汇**——类比神经递质对神经元兴奋性的调制（多巴胺让值得追求的更突出，去甲肾上腺素让危险被优先加工）。

---

## 三、意识流循环：模型"永不休眠"（Phase 1）

`StreamCortex.start()` 启动一个**永不返回的异步循环**（`while self._running: await self._think_cycle()`），每轮 `_think_cycle()` 执行：

1. **构建上下文**：`CONSCIOUSNESS_SYSTEM_PROMPT`（声明"你是持续运行的生命体，会产生内部想法"）+ 身体状态 + 最近记忆 + **原始传感器数据**（`sensor_window.to_context()`，Phase 3）。
2. **计算 logit_bias**：`self.modulator.compute_logit_bias(self._body_state)`。
3. **流式生成思维**：`llm.stream_think(...)` 内部构造 `system` + `user`（记忆/身体感觉 + "产生一个简短的内部想法"），以 `temperature * 1.2` 流式产出。
4. **门控评估**（`ActionGate`）：计算相关性评分 `_compute_relevance()`（含行动词/感受词加权），决定该思维被抑制（标签 💭）还是外化为行为（🗣️）。约 95% 被抑制，如人不会说出每个念头。
5. **存档与发射**：保留到 `ThoughtJournal`；若门控开放则经 MQTT 发射行为指令。
6. **动态间隔**：根据驱力水平调节下一轮等待（`1.5s` 高驱力 / `3s` 常规 / `5s` 安静），并加随机抖动避免机械节奏。

外部人类语音输入（`animat/external/speech`）通过异步锁入队，下一轮作为 `user` 消息走 `complete()` 生成回复（默认高相关性 0.85）。

---

## 四、整体工作链条

```
[传感器 / 稳态] → BodyState 向量 → AttentionModulator.compute_logit_bias()
                                      │ (token logit bias)
                                      ▼
StreamCortex 主循环 → stream_think(system_prompt + 记忆 + 原始传感)
                                      │ (llm.tokenize / create_chat_completion)
                                      ▼
           本地 GGUF 模型 (llama-cpp-python, CPU, KV cache)
                                      │
                                      ▼
                  Thought → ActionGate 门控 → 95% 抑制 / 5% 外化行为
                                      │
                                      ▼
                          ThoughtJournal + 行为发射(MQTT)
```

---

## 五、要点总结

- **模型本身是一个标准本地因果语言模型**（llama.cpp 推理），工程的创新不在改模型，而在**如何喂它、如何偏置它、如何筛选它的输出**。
- 身体状态通过 **logit bias（概率偏置）** 而非 prompt 文本影响认知，这是 v0.2 相对 v0.1「prompt 注入情绪」的核心升级。
- 「主动性」不是靠定时器唤醒，而是**不间断的意识流循环**自然涌现；门控机制模拟「人不会说出每个念头」。

---

## 六、如何切换到云端 API（DeepSeek / 阿里云百炼）

机制的本质是 **`logit_bias` 参数 + 同源 tokenizer**，与"本地还是云端"无关。本工程已内置 `CloudLLM` 后端（`animat/cloud_llm.py`），接口与 `LocalLLM` 完全对齐。

### 6.1 配置方式

设置环境变量即可切换后端：

```bash
export ANIMAT_LLM_BACKEND=cloud
export OPENAI_API_KEY=sk-xxxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1   # 或百炼的兼容地址
export ANIMAT_LLM_MODEL=deepseek-chat                 # 或 qwen-xxx
```

`brain.py` 通过 `create_llm()` 工厂（`animat/llm_backend.py`）按 `ANIMAT_LLM_BACKEND` 选择后端，默认仍是本地 GGUF。

### 6.2 两个前提条件

1. **云端 API 必须支持 `logit_bias`**
   - OpenAI 官方、DeepSeek、阿里云百炼（DashScope OpenAI 兼容）均支持。
   - 若某平台屏蔽该参数，`CloudLLM` 会捕获 400/错误并**自动降级为无偏置采样**（Phase 2 失效，退回 v0.1 的 prompt 注入方式）。
2. **tokenizer 必须同源**
   - `AttentionModulator` 用 `tokenize()` 把种子词切成 token_id，`CloudLLM` 默认用 `tiktoken (cl100k_base)`，与 GPT/DeepSeek 系一致。
   - 非 GPT 系模型（如 Qwen）若需精确 token_id，可在 `CloudLLMConfig.tokenizer_encoding` 指定，或后续接入对应 tokenizer 库。
   - `tiktoken` 已加入 `pyproject.toml` 依赖，运行 `uv sync` 即可安装。

### 6.3 依赖

```toml
dependencies = [
    "openai>=1.0.0",   # 云端后端所需
    "tiktoken>=0.7.0", # tokenize 所需
]
```

---

## 七、GPU 推理（Linux + NVIDIA）

本地 GGUF 模型可把 Transformer 层卸载到 GPU 加速。`LLMConfig` 会**自动探测** NVIDIA GPU（通过 `nvidia-smi -L`）：探测到则默认 `n_gpu_layers=99`（全量卸载），否则回落 CPU。也可用 `ANIMAT_N_GPU_LAYERS` 显式覆盖。

> Python 侧代码无需改动，但 **`llama-cpp-python` 必须是带 CUDA 后端的构建**。PyPI 的默认 wheel 是纯 CPU 版，装上后 `n_gpu_layers` 设多大都不会用 GPU。

### 7.1 判断当前是不是 CUDA 版

```bash
cd ~/Workspace/animat-agent
./.venv/bin/python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
ls .venv/lib/python3.12/site-packages/llama_cpp/lib/ | grep -i cuda
```

`True` + 存在 `libggml-cuda.so`（约 60MB）才是 CUDA 版。

### 7.2 踩坑记录（重要）

以下三个坑在实际部署中全部踩到过，直接决定成败：

**坑 1：`uv run pip` 可能装错环境**

若机器上有 conda 且 `PATH` 中 conda 的 `bin` 在前，`uv run pip` 里的 `pip` 会解析到 **conda base 的 pip**，包被装进 conda 环境，而 `.venv` 纹丝不动 —— 表现为「编译成功但 `gpu_offload` 仍是 False」。

**始终用绝对路径调用 venv 的解释器**：

```bash
./.venv/bin/python -m pip install ...
```

**坑 2：`uv pip install` 不传递 `CMAKE_ARGS`**

`uv pip` 的隔离构建不继承 `CMAKE_ARGS`，且对同版本包会直接 `Checked 1 package` 跳过重编。必须用 `python -m pip` + `--force-reinstall` + `--no-binary`。

**坑 3：CUDA Toolkit 版本必须匹配 GPU 架构**

Blackwell（RTX 50 系，`sm_120`）需要 **CUDA Toolkit ≥ 12.8**。系统自带的 12.0 会报：

```
nvcc fatal : Unsupported gpu architecture 'compute_120a'
```

注意 `nvidia-smi` 显示的 "CUDA Version: 13.0" 是**驱动支持的上限**，不是已装的 Toolkit 版本。以 `nvcc --version` 为准。

### 7.3 完整安装步骤

```bash
# ① 装 CUDA Toolkit（这里借 conda 提供 nvcc，不影响 .venv 的 Python 环境）
conda install -y -c nvidia cuda-toolkit=12.8
export PATH=$CONDA_PREFIX/bin:$PATH
nvcc --version    # 必须是 release 12.8

# ② 源码编译安装到 .venv
cd ~/Workspace/animat-agent
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc"
export FORCE_CMAKE=1
./.venv/bin/python -m pip install --no-binary llama-cpp-python --no-cache-dir --force-reinstall \
  "llama-cpp-python==0.3.34" -i https://mirrors.aliyun.com/pypi/simple/

# ③ 修复被顺带升级的 numpy
./.venv/bin/python -m pip install --no-cache-dir "numpy<2.2"
```

`CMAKE_CUDA_ARCHITECTURES` 按显卡填：

| 显卡 | 架构 | 值 |
|------|------|-----|
| RTX 50 系（5060/5090） | Blackwell | `120` |
| RTX 40 系（4090） | Ada | `89` |
| RTX 30 系（3090） | Ampere | `86` |

用纯数字（`120`）而非 `120a`，后者需要更新的 nvcc。

编译耗时约 10–20 分钟。

### 7.4 配置（`.env`）

```env
ANIMAT_LLM_BACKEND=local
ANIMAT_MODEL_PATH=/root/Workspace/animat-agent/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
# ANIMAT_N_GPU_LAYERS=99    # 留空即自动探测；显式设 0 可强制 CPU
ANIMAT_N_THREADS=8
ANIMAT_LLM_VERBOSE=0        # 设 1 会打印大量 llama.cpp 内部日志，仅调试时开
```

### 7.5 验证

```bash
uv run animat
```

日志出现以下内容即成功：

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 15847 MiB):
  Device 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, VMM: yes
Loading local LLM: ... [GPU mode, n_gpu_layers=99]
```

### 7.6 实测性能（RTX 5060 Ti 16GB / Qwen2.5-1.5B Q4_K_M）

| 指标 | GPU | CPU |
|------|-----|-----|
| 模型加载 | 195 ms | ~400 ms |
| Prompt eval | ~1000 tok/s | — |
| 生成速度 | ~70 tok/s | — |
| 显存占用 | 698 MB | — |
| 单轮思维循环 | 160–210 ms | — |

1.5B Q4 全量卸载仅占约 0.7GB 显存，16GB 卡跑 7B Q4（约 5GB）也很轻松。

### 7.7 避免被 `uv sync` 覆盖

`uv sync` 会按 lockfile 把 CUDA 版换回 PyPI 的 CPU wheel。执行过 sync 后需重新编译，建议存成脚本：

```bash
cat > rebuild_cuda_llama.sh <<'EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")"
export PATH=$CONDA_PREFIX/bin:$PATH
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc"
export FORCE_CMAKE=1
./.venv/bin/python -m pip install --no-binary llama-cpp-python --no-cache-dir --force-reinstall \
  "llama-cpp-python==0.3.34" -i https://mirrors.aliyun.com/pypi/simple/
./.venv/bin/python -m pip install --no-cache-dir "numpy<2.2"
EOF
chmod +x rebuild_cuda_llama.sh
```

---

## 附：相关源码文件

| 文件 | 职责 |
|------|------|
| `animat/llm_backend.py` | 本地 LLM 加载、流式/非流式推理、logit bias 支持 |
| `animat/attention_modulator.py` | 身体状态 → token 概率偏置（Phase 2） |
| `animat/stream_cortex.py` | 恒常意识流引擎（Phase 1） |
| `animat/thought.py` | Thought、ActionGate 三层门控、ThoughtJournal |
| `animat/sensor_window.py` | 原始传感器时间序列转文本上下文（Phase 3） |
| `animat/homeostasis.py` | 数字内分泌系统，产出 BodyState 稳态指标 |
