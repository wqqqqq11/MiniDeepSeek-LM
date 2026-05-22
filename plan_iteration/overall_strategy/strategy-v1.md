# 目标：复现deepseek-v4模型（5M 参数），不追求参数规模和性能一致，重点复现以下机制：
        1. DeepSeekMoE：Hash-routed MoE + learned-gate MoE
        2. Hybrid Attention：CSA/HCA + sliding window KV
        3. mHC：多残差流 + Sinkhorn 约束
        4. 训练流程：预训练、扩窗续训、分领域专家 SFT/GRPO、统一蒸馏、最终对齐
        5. 推理流程：稀疏专家激活、压缩 KV、长上下文缓存

# 复现概述
分为 架构层、训练全阶段（5阶段）、推理层 三部分。

# 架构层面
## DeepSeekMoE 混合专家架构
- 前若干层：Hash-routed MoE
- 后续层：learned-gate DeepSeekMoE
- 每层 FFN 子层均采用 MoE FFN
- 专家结构：routed experts + shared expert
- 路由打分：sqrt(softplus)
- 负载均衡：aux-loss-free + 轻量 sequence-wise balance loss
## 混合稀疏注意力
    CSA 压缩稀疏注意力 + HCA 重度压缩稠密注意
    搭配滑动窗口注意力（swa）+ attention sink 支持 4k token 上下文 结构设计
## 流形约束超连接（mHC）
    残差链路特殊约束设计，解决超深 MoE 模型梯度消失
## 基础组件固定不变
    归一化：RMSNorm
    激活函数：SwiGLU
    位置编码：YaRN 扩展部分旋转位置编码（Partial RoPE），仅最后 64 维施加
    算子依赖：强制依赖 DeepGEMM 融合 CUDA 算子（不使用）
## 数值精度架构
    训练 / 推理统一：FP8 × FP4 混合精度 结构设计（由于模型参数量小，因此不使用混合精度训练和推理）
    主模块用 Muon 二阶优化，嵌入 / 分类头用 AdamW

# 训练策略
## 阶段 1：通用基座大规模预训练
    定位：打底通用语言、百科、常识、基础文理知识
    数据总量：15M Tokens
    数据集： 中文（60%）：https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
            英文（40%）：https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2

## 阶段 2：超长上下文扩窗续训（简化版）
    上下文：从 1K → 2K
    数据：使用阶段 1 的通用普通文本（1.5M tokens 数据量）,中英文（6:4）
    训练：按tokens计算

## 阶段 3：分领域独立专家训练（SFT + GRPO）
    独立拆分 5 大领域（为了简化训练，这里采用3大领域），每个领域用专属数据单独微调训练：
    数据总量：6M tokens
    数据：每个领域只用自己的垂直语料

    训练方式：固定 V4 主干，只精调对应领域专家分支
    目的：让每个专家在自己领域达到最优能力

    专家类型	总 Token	SFT（90%，基础能力）	GRPO（10%，专家强化）	数据集与组合比例
    数学推理	2M	        1.8M	                0.2M	             SFT：
                                                                    1. AI-MO/NuminaMath-CoT：100% 

                                                                  GRPO：
                                                                    1. openai/gsm8k：100%
          
    代码生成	2M	        1.8M	                0.2M	             SFT：
                                                                    1. ise-uiuc/Magicoder-OSS-Instruct-75K：60%
                                                                    3. m-a-p/CodeFeedback-Filtered-Instruction：40% 

                                                                  GRPO：
                                                                    1. codeparrot/apps（小 prompt pool）：100%
                
    科研学术	2M	        1.8M	                0.2M	       SFT：         
                                                                    1. qiaojin/PubMedQA：100%

                                                                  GRPO：
                                                                    1. allenai/scifact：100% 
    
    注意：SFT 和 GRPO 可以使用同源数据集，但必须切分不同样本池。

## 阶段 4：OPD 最优路径蒸馏合并
    把阶段 3 训练好的3 个独立领域专家，通过 OPD 蒸馏策略，合并融合进同一个主模型
    训练方式：知识蒸馏 + 路径权重学习
    效果：一个模型同时拥有所有专家的领域能力，且互相不干扰

    数据集构成（总共1.5M tokens）：
    数据类型	       	        Token占比	        数据来源
    通用数据	         	        45%	         阶段 1 的通用高质量子集
    代码数据	         	        20%	         阶段 3 代码专家的垂直数据核心子集
    数学数据	         	        20%	         阶段 3 数学专家的垂直数据核心子集
    科研数据	         	        15%	         阶段 3 科研专家的垂直数据核心子集

## 阶段 5：最终对齐（轻量 SFT）
### 有监督微调 SFT（1M tokens）：通用指令、对话、任务对齐
    领域	    token 占比		        数据集
    通用指令	45%	    	            https://huggingface.co/datasets/tatsu-lab/alpaca_cleaned（经典通用指令集）
    数学推理	20%	     	            https://huggingface.co/datasets/openbmb/UltraMath-Instruct（数学指令）
    代码生成	20%	     	            https://huggingface.co/datasets/HuggingFaceH4/code_alpaca_2k（代码指令）
    科研问答  15%     
# 项目代码结构 (src/models/)

```
src/models/
├── config.py                # 模型配置集中管理，定义 ModelArgs 数据类
├── RuntimeConfig.py         # 运行时环境配置（分布式、量化参数、实现模式）
├── layers.py                # 基础层组件：RMSNorm + 各类并行线性层
├── rotary_embedding.py      # YaRN 扩展的 RoPE 位置编码实现
├── kernel.py                # 量化 kernel fallback 实现（FP8/FP4 模拟）
├── mla_attention.py         # MLA 多头潜在注意力（含 CSA + HCA 稀疏注意力）
├── moe.py                   # MoE 混合专家架构（Gate + Expert + Shared Expert）
├── block.py                 # Transformer Decoder Block（MLA + MoE + Hyper-Connections）
└── transformer.py           # DeepSeek-V4 主模型入口，组装完整模型
```

## 模块依赖关系

```
transformer.py
    ├── ParallelEmbedding (layers.py, 词表分布式切分)
    ├── Block × N
    │   ├── RMSNorm (layers.py)              # Pre-Norm
    │   ├── MLA (mla_attention.py)           # 含 CSA + HCA + 低秩压缩
    │   │   ├── Compressor                   # KV 压缩器
    │   │   ├── Indexer (可选)               # 压缩索引检索
    │   │   └── apply_rotary_emb             # RoPE 应用
    │   ├── hc_pre/hc_post                   # Hyper-Connections 混合
    │   ├── RMSNorm (layers.py)              # Pre-Norm
    │   └── MoE (moe.py)                     # 混合专家
    │       ├── Gate                         # 门控路由
    │       ├── Expert × n_local_experts    # 本地专家
    │       └── MLP (shared_experts)         # 共享专家
    ├── hc_head                              # 最终 HC 合并
    ├── RMSNorm (layers.py)                  # 最终归一化
    └── ColumnParallelLinear (head)          # 输出头（词表分布式切分）

依赖注入:
    ├── ModelArgs (config.py)                # 模型结构参数
    └── RuntimeConfig (RuntimeConfig.py)     # 运行时配置（分布式、量化）

底层支持:
    └── kernel.py                            # 量化/HC 计算 fallback
    └── rotary_embedding.py                  # RoPE 预计算
```

## 关键接口约定

| 模块 | 输入 | 输出 | 备注 |
|------|------|------|------|
| Block | [B, S, hc_mult, D] | [B, S, hc_mult, D] | 带 Hyper-Connections 的 decoder 层 |
| MLA | [B, S, D] | [B, S, D] | 含 CSA(局部窗口) + HCA(全局压缩) |
| MoE | [B, S, D] | [B, S, D] | Gate 路由 + Expert 计算 + Shared Expert |
| Transformer | [B, S] | [B, V] | 输入 token ids，输出 logits |
| hc_pre | [B, S, hc, D] | ([B, S, D], [B, S, hc], [B, S, hc, hc]) | 混合为1个 + 返回 post/comb 权重 |
| hc_post | ([B, S, D], [B, S, hc, D], [B, S, hc], [B, S, hc, hc]) | [B, S, hc, D] | 扩展并混合残差流 |

## 推荐参数配置
```
vocab_size = 6400
hidden_size = 256
num_hidden_layers = 8
num_attention_heads = 8
num_key_value_heads = 4

intermediate_size = 128      # shared expert / dense fallback
moe_intermediate_size = 48   # 每个 routed expert 的 FFN hidden

num_routed_experts = 4
num_shared_experts = 1
num_experts_per_tok = 1   

n_hash_layers = 2             # 前 2 层 hash-routed
hc_mult = 2                  

max_position_embeddings = 4096
rope_dim = 32
tie_word_embeddings = True
```