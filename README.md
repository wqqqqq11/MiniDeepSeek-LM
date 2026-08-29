# MiniDeepSeek-LM

针对大参数量模型在短句补全、意图识别任务上算力开销高的痛点，训练小型 Causal-LM，面向低资源场景下的垂直领域任务。项目基于 Transformer 实现哈希路由 MoE、CSA/HCA 混合注意力、mHC 流形约束超连接模块，完成模型预训练、调优与效果验证。

## 项目概述

大模型在短文本场景（语句补全、意图识别）上往往参数冗余、推理成本高，难以部署到单卡或边缘环境。本项目将模型规模压到约 **5M 参数**，用稀疏激活与压缩注意力降低计算与显存，服务低资源垂直任务。

当前实现重点：

- **架构**：Causal Transformer + 哈希 / 门控混合 MoE + CSA / HCA 混合注意力 + mHC 多残差流
- **训练**：预训练 → 扩窗续训 → 分领域 SFT / GRPO → 蒸馏合并 → 最终对齐；入口已接通预训练与领域 SFT
- **数据**：通用语料流水线（下载 / 清洗 / 去重 / 切分 / tokenize / 二值化）；数学、代码、科研等垂直领域 SFT 流水线
- **验证**：checkpoint 推理生成、困惑度评估、前向数值测试

词表使用 [jingyaogong/minimind-3](https://huggingface.co/jingyaogong/minimind-3)（`vocab_size=6400`），适配小模型与短序列场景。

## 技术栈

| 类别 | 选型 |
| --- | --- |
| 语言 | Python 3 |
| 深度学习 | PyTorch 2.5.0（CUDA 11.8） |
| 数据 | HuggingFace `datasets`、`transformers`、SentencePiece |
| Tokenizer | `jingyaogong/minimind-3` |
| 配置 | YAML（`configs/`）、dataclass（`src/models/config.py`、`src/data/config.py`） |
| 去重 | Bloom Filter（`pybloom-live`） |
| IO | `orjson`、`aiofiles`、`pandas`、`numpy` |
| 优化器 | AdamW（当前训练入口）；Muon 二阶优化器已实现，可按配置接入 |
| 精度 | 训练 / 推理使用 bfloat16；FP8 为结构预留，小模型默认不启用 |

主要依赖见 `requirements.txt`。安装示例：

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install datasets==2.17.0 transformers sentencepiece==0.2.1 pybloom-live aiofiles orjson pandas numpy pyyaml
```

## 项目结构

```
MiniDeepSeek-LM/
├── mian.py                          # 训练入口（stage 1 / stage 3）
├── inference.py                     # 推理入口
├── requirements.txt                 # Python 依赖
├── prompts/stage1/prompts.txt       # 推理输入样例
├── src/
│   ├── models/                      # 模型定义
│   │   ├── config.py                # ModelArgs（tiny / stage1 等预设）
│   │   ├── transformer.py           # Causal Transformer（HC + MLA + MoE）
│   │   ├── transformer_stage1.py    # 预训练可训练模型
│   │   ├── mla_attention.py         # MLA + CSA / HCA
│   │   ├── mla_attention_stage1.py  # 阶段 1 注意力实现
│   │   ├── moe.py                   # Hash / learned-gate MoE
│   │   ├── block.py                 # Transformer Block + mHC
│   │   ├── rotary_embedding.py      # Partial RoPE / YaRN
│   │   └── layers.py                # RMSNorm、并行线性层等
│   ├── data/                        # 数据预处理
│   │   ├── pipeline.py              # 阶段 1 流水线
│   │   ├── preprocess_run.py        # 阶段 1 预处理入口
│   │   ├── config.py                # 阶段 1 数据配置
│   │   └── stage3/                  # 阶段 3 SFT / GRPO 数据
│   ├── training/                    # 训练
│   │   ├── stages/                  # 五阶段训练脚本
│   │   ├── dataset.py               # TokenDataset
│   │   ├── muon.py                  # Muon 优化器
│   │   └── lr_scheduler.py          # 三段式学习率
│   ├── inferences/stage1_inference/ # 生成器与采样
│   ├── evaluations/stage1_evaluation/
│   └── tests/stage1_test/           # 前向 / 推理测试
├── datasets/                        # 预处理产物（git 忽略）
├── checkpoints/                     # 训练检查点
├── logs/                            # 训练日志
├── outputs/                         # 推理结果
├── plan_iteration/                  # 规划与阶段文档
│   ├── overall_strategy/strategy-v1.md
│   ├── stage_1.md
│   ├── stage_3.md
│   └── stage_4.md
└── study_materials/                 # 架构笔记与示意图
```

## 核心工作

### 1. 模型架构

默认预训练配置约 5M 参数，面向短序列 Causal-LM：

| 组件 | 实现要点 |
| --- | --- |
| **MLA** | 低秩 KV 压缩（`kv_lora_rank`），降低 KV cache |
| **CSA** | 压缩稀疏注意力：滑动窗口 + 轻量压缩，处理短距离依赖 |
| **HCA** | 重度压缩稠密注意力：历史 KV 压缩后 top-k 检索，处理长距离依赖 |
| **哈希路由 MoE** | 前若干层 Hash 路由，后续层 learned-gate；routed experts + shared expert；`sqrtsoftplus` 打分；aux-loss-free 动态偏置 |
| **mHC** | `hc_mult` 条并行残差流，Sinkhorn 约束混合权重 |
| **基础件** | RMSNorm、SwiGLU、Partial RoPE、YaRN 扩窗预留 |

层间注意力模式由 `attn_patterns` / `compress_ratios` 控制，例如预训练阶段交替 HCA / CSA。稀疏专家激活只计算被选中的专家，适合低资源推理。

### 2. 五阶段训练

| 阶段 | 目标 | 数据规模（规划） | 入口状态 |
| --- | --- | --- | --- |
| **1 预训练** | 通用语言 / 百科 / 常识 | 15M tokens，中英 6:4 | `python mian.py --stage 1` 已接通 |
| **2 扩窗续训** | 上下文 1K → 2K | 约 1.5M tokens | 模块已建，入口暂未接线 |
| **3 分领域专家** | 固定主干，只训领域专家 | 数学 / 代码 / 科研，各约 2M | SFT：`--stage 3 --domain` 已接通；GRPO 流水线已建 |
| **4 OPD 蒸馏** | 多专家合并进主模型 | 约 1.5M tokens | 规划已写，训练入口未接线 |
| **5 最终对齐** | 轻量指令 SFT | 约 1M tokens | 规划已写，训练入口未接线 |

阶段 3 领域与专家编号：

| 领域 | expert_id | SFT 数据 |
| --- | --- | --- |
| math | 0 | AI-MO/NuminaMath-CoT |
| code | 1 | Magicoder-OSS-Instruct-75K + CodeFeedback-Filtered-Instruction |
| science | 2 | qiaojin/PubMedQA |

原则：固定主干，强制路由到指定专家，只更新该专家 FFN。

### 3. 数据流水线

**阶段 1**（`src/data/pipeline.py`）：

1. 下载 Fineweb-Edu-Chinese-V2.1、RedPajama-Data-V2
2. 抽取文本字段
3. HTML / URL / 异常符号清洗与质量过滤
4. 近似去重
5. 按语言 8:1:1 切分后再合并
6. tokenize（文档末尾加 EOS）
7. 拼接为 token stream 并写成 `.bin`

**阶段 3 SFT**（`src/data/stage3/preprocess_run.py`）：下载 → 格式转换 → tokenize → 写出最终 Arrow 数据集。

### 4. 推理

`inference.py` 加载阶段 1 checkpoint，读取 `prompts/stage1/prompts.txt`，temperature 采样生成，结果写入 `outputs/stage1/`。

## 使用方法

### 环境

```bash
cd MiniDeepSeek-LM
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

如 `requirements.txt` 中的 PyTorch 索引与本机 CUDA 不符，按上一节技术栈单独安装对应 wheel。

### 阶段 1 数据预处理

编辑 `src/data/preprocess_run.py`，按需打开各 stage，或一次性跑完全流程：

```python
pipeline = DataPipeline(DataConfig(base_data_dir="datasets/stage_1_datasets"))
pipeline.run_stage(0)   # 下载
# pipeline.run_stage(1)  # 清洗
# pipeline.run_stage(2)  # 去重
# pipeline.run_stage(3)  # 切分
# pipeline.run_stage(4)  # tokenize
# pipeline.run_stage(5)  # 合并
# pipeline.run_stage(6)  # 二值化
# pipeline.run_all(start_stage=0)
```

```bash
python src/data/preprocess_run.py
```

产物目录：`datasets/stage_1_datasets/{0_raw ... 6_binarize}/`。训练读取：

- `datasets/stage_1_datasets/6_binarize/train.bin`
- `datasets/stage_1_datasets/6_binarize/val.bin`

### 阶段 3 SFT 数据预处理

```bash
python src/data/stage3/preprocess_run.py --stage 0 --domain all   # 下载
python src/data/stage3/preprocess_run.py --stage 1 --domain all   # 转换
python src/data/stage3/preprocess_run.py --stage 2 --domain all   # tokenize
python src/data/stage3/preprocess_run.py --stage 3 --domain all   # 最终数据集
```

`--domain` 可为 `all` / `math` / `code` / `science`。

### 训练

阶段 1 预训练：

```bash
python mian.py --stage 1 --config configs/stage1_pretrain.yaml
```

阶段 3 领域 SFT（需先有阶段 1 checkpoint，并改配置里的 `sft.base_checkpoint`）：

```bash
python mian.py --stage 3 --config configs/stage1_pretrain.yaml --domain math
python mian.py --stage 3 --config configs/stage1_pretrain.yaml --domain code
python mian.py --stage 3 --config configs/stage1_pretrain.yaml --domain science
```

检查点默认写到 `checkpoints/stage1/` 或 `checkpoints/stage3/`，日志在 `logs/stage1/`。

阶段 2 / 4 / 5 在 `src/training/stages/` 中有对应模块，训练入口尚未接入 `mian.py`。

### 推理

```bash
python inference.py ^
  --checkpoint checkpoints/stage1/checkpoint_epoch2_xxxx.pt ^
  --config configs/stage1_pretrain.yaml ^
  --input_file prompts/stage1/prompts.txt ^
  --output_dir outputs/stage1/ ^
  --temperature 0.6 ^
  --max_new_tokens 100
```

Linux / macOS 将 `^` 换成 `\`。

### 测试

```bash
python -c "from src.tests.stage1_test.test_model_forward import run_tests; run_tests()"
```

前向测试不需要 checkpoint；推理测试需要已训练权重。

## 配置说明

主配置文件：`configs/stage1_pretrain.yaml`。代码侧还有 `ModelArgs`（结构超参）和 `DataConfig`（数据流水线）。

### 模型（`model`）

| 字段 | 阶段 1 默认 | 说明 |
| --- | --- | --- |
| `vocab_size` | 6400 | 与 minimind tokenizer 一致 |
| `dim` | 256 | 隐藏层宽度 |
| `n_layers` | 8 | Transformer 层数 |
| `n_heads` | 8 | 注意力头数 |
| `max_seq_len` | 512 | 阶段 1 上下文长度 |
| `q_lora_rank` / `kv_lora_rank` | 64 / 32 | MLA 低秩压缩 |
| `n_hash_layers` | 1 | Hash 路由层数 |
| `n_routed_experts` | 3 | 路由专家数 |
| `n_shared_experts` | 1 | 共享专家数 |
| `n_activated_experts` | 1 | 每 token 激活专家数 |
| `score_func` | `sqrtsoftplus` | MoE 门控打分 |
| `attn_patterns` | `[0,1,0,1,...]` | 0=HCA，1=CSA |
| `compress_ratios` | `[4,2,4,2,...]` | 各层 KV 压缩比 |
| `hc_mult` | 2 | mHC 残差流条数 |
| `tie_word_embeddings` | true | 输入 / 输出词嵌入共享 |

结构预设见 `src/models/config.py`：

- `ModelArgs.stage1()`：当前训练用 5M 配置
- `ModelArgs.tiny()`：更小实验配置

### 训练（`training` / `optimizer` / `mtp`）

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `num_epochs` | 60 | 预训练轮数 |
| `batch_size` | 32 | 批次大小 |
| `max_lr` / `min_lr` | 1e-4 / 3e-5 | 三段式学习率：warmup → peak → cosine |
| `warmup_ratio` / `peak_ratio` | 0.05 / 0.50 | 其余为余弦衰减 |
| `grad_clip` | 0.5 | 梯度裁剪 |
| `seed` | 42 | 随机种子 |
| `optimizer.adamw_lr` | 3e-4 | 当前入口使用 AdamW |
| `optimizer.muon_lr` | 3e-4 | Muon 组学习率（配置预留） |
| `mtp.num_future_tokens` | 2 | 多 token 预测 |
| `early_stopping.patience` | 5 | 验证 PPL 不降则停 |

### 数据与硬件

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `data.train_bin` / `val_bin` | `datasets/stage_1_datasets/6_binarize/*.bin` | 预训练二进制语料 |
| `data.context_size` | 512 | 训练切窗长度，应与 `max_seq_len` 一致 |
| `hardware.device` | `cuda` | `cuda` 或 `cpu` |
| `logging.checkpoint_dir` | `checkpoints/stage1` | 权重保存目录 |

阶段 1 数据流水线默认在 `src/data/config.py`：中英 6:4、目标 15M tokens、`tokenizer_name=jingyaogong/minimind-3`。

### 阶段 3 SFT（同一 YAML 的 `sft` 段）

| 字段 | 说明 |
| --- | --- |
| `base_checkpoint` | 阶段 1（或阶段 2）权重路径 |
| `domains.*.expert_id` | 强制路由的专家编号 |
| `domains.*.train_data` / `val_data` | 最终 Arrow 目录 |
| `num_epochs` / `batch_size` / `lr` | SFT 超参 |
| `output_dir` | 领域专家 checkpoint 输出目录 |

改路径或超参时，优先改 YAML，避免直接改训练代码。模型宽度、层数、专家数等结构字段需与即将加载的 checkpoint 一致，否则无法 `load_state_dict`。
