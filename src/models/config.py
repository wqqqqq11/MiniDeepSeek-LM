from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelArgs:
    """
    模型参数数据类，定义所有模型结构和超参数。

    可通过 ModelArgs.original() 获取完整 DeepSeek V4 配置。
    """

    # ── 运行环境 / 部署相关 ────────────────────────────────────────────
    max_batch_size: int = 4
    """最大批次大小，用于预分配 KV 缓存缓冲区。"""

    max_seq_len: int = 4096
    """模型能处理的最大序列长度（同时决定 RoPE 预计算范围）。"""

    dtype: Literal["bf16", "fp8"] = "bf16"
    """计算数据类型：bf16（bfloat16）或 fp8（float8，需自定义 kernel 支持）。"""

    scale_fmt: Optional[str] = None
    """分块量化的缩放因子格式，None 表示不使用量化；仅在 dtype='fp8' 时生效。"""

    # ── 词表 ──────────────────────────────────────────────────────────
    vocab_size: int = 16000
    """词表大小（嵌入矩阵尺寸：vocab_size × dim）。"""

    # ── 核心模型维度 ──────────────────────────────────────────────────
    dim: int = 512
    """隐藏层维度（模型宽度 / 残差流维度），所有嵌入、注意力投影和 FFN 层均以此为基准。"""

    inter_dim: int = 1536
    """密集 MLP 前馈网络的中间层（隐藏）维度。
    通常约为 dim 的 3 倍；原始比例约为 5.34 倍（10944/2048）。"""

    moe_inter_dim: int = 192
    """每个 MoE 专家前馈网络的中间层维度。
    通常比 inter_dim 小，因为专家更专业化；此处约为 dim 的一半。"""

    n_layers: int = 8
    """Transformer 总层数（Block 个数）。"""

    n_dense_layers: int = 1
    """使用密集 MLP（而非 MoE）的前几层数量。
    第 0 到 n_dense_layers-1 层使用 MLP，其余层使用 MoE。"""

    n_hash_layers: int = 0
    """使用 hash 路由的前几层数量。
    前 n_hash_layers 层通过 token ID 直接查表获取专家，不使用门控网络。"""

    n_heads: int = 4
    """MLA（多头潜在注意力）的注意力头数。
    每个头的维度为 dim // n_heads = 128。"""

    # ── 专家混合（MoE） ──────────────────────────────────────────────
    n_routed_experts: int = 16
    """每个 MoE 层中路由专家的总数。
    当 world_size > 1 时，专家会在各 GPU 间切分。"""

    n_shared_experts: int = 2
    """共享专家数量（所有 token 都会经过，不参与路由）。
    共享专家的输出会加到路由专家输出之上。"""

    n_activated_experts: int = 4
    """每个 token 激活的专家数（top-k 路由）。
    每个 token 将被路由到门控得分最高的 k 个专家。"""

    n_expert_groups: int = 4
    """分层路由中的专家分组数。
    当 > 1 时，token 先路由到某个组，再在组内选择专家。"""

    n_limited_groups: int = 2
    """分层路由中每个 token 可选择的组数上限。"""

    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "softmax"
    """MoE 门控的评分函数：'softmax'、'sigmoid' 或 'sqrtsoftplus'（官方）。"""

    # ── Aux-Loss-Free 动态偏置 ─────────────────────────────────────
    bias_update_speed: float = 0.001
    """专家门控偏置更新速度（官方固定值 0.001）。"""

    ema_decay: float = 0.9
    """专家频率统计滑动平均衰减率。"""

    route_scale: float = 1.0
    """路由权重的缩放因子，在专家选择后乘以原始得分，用于控制专家输出大小。"""

    # ── 归一化 ─────────────────────────────────────────────────────
    norm_eps: float = 1e-6
    """RMSNorm 的数值稳定性参数，防止除零。
    用于所有 RMSNorm 层（Pre-Norm 和最终归一化）。"""

    # ── 多头潜在注意力（MLA） ────────────────────────────────────────
    q_lora_rank: int = 0
    """查询投影的低秩压缩维度。
    为 0 时，查询投影为直接线性层（无低秩分解）；
    大于 0 时，Q 先压缩到该维度再扩展到各头维度。"""

    kv_lora_rank: int = 128
    """键值投影的低秩压缩维度。
    KV 状态先压缩到此维度，可将 KV 缓存内存降低约 dim / kv_lora_rank 倍。"""

    head_dim: int = 128
    """注意力头的总维度（Query/Key/Value 的统一头维度）。
    在 DeepSeek-V4 中通常为 512，但小模型可适当减小。
    注意：qk_nope_head_dim + qk_rope_head_dim 可能不等于 head_dim，
    因为 MLA 使用低秩分解重构机制。"""

    qk_nope_head_dim: int = 64
    """查询和键中非位置编码部分的每头维度（内容部分）。"""

    qk_rope_head_dim: int = 32
    """查询和键中位置编码（RoPE）部分的每头维度。
    总 Q/K 头维度 = qk_nope_head_dim + qk_rope_head_dim = 96。"""

    v_head_dim: int = 64
    """值投影的每头维度。"""

    # ── YaRN 位置编码（长序列扩展） ──────────────────────────────────
    original_seq_len: int = 4096
    """预训练的原始序列长度，作为 YaRN 缩放的基准。
    序列长度 ≤ 此值时使用标准 RoPE。"""

    rope_theta: float = 10000.0
    """旋转位置编码（RoPE）的基础频率。
    值越大，频率衰减越慢，有利于改善长程衰减稳定性。"""

    rope_factor: float = 1.0
    """YaRN 扩展因子，用于在原序列长度基础上扩展。
    当 max_seq_len > original_seq_len 时，频率会除以此因子并配合平滑插值。
    设为 1.0 表示不需要长序列扩展。"""

    beta_fast: int = 32
    """YaRN 维度混合的快速 beta 修正参数。
    控制插值斜坡函数的下界（以旋转圈数为单位）。"""

    beta_slow: int = 1
    """YaRN 维度混合的慢速 beta 修正参数。
    控制插值斜坡函数的上界（以旋转圈数为单位）。"""

    mscale: float = 1.0
    """长序列扩展时注意力 softmax 缩放系数的乘数。
    当 max_seq_len > original_seq_len 时：
    softmax_scale *= 0.1 * mscale * ln(rope_factor) + 1.0。"""

    # ── 混合稀疏注意力（CSA + HCA）───────────────────────────────────
    window_size: int = 128
    """CSA 滑动窗口大小。局部注意力只关注最近 window_size 个 token。
    这是 DeepSeek-V4 处理短距离依赖的核心机制。"""

    attn_patterns: tuple = ()
    """每层注意力模式元组，长度应等于 n_layers。
    0 = HCA (纯全局压缩), 1 = CSA (局部窗口 + 轻量压缩)。
    示例: (0, 1, 0, 1, 0, 1, 0, 1) 表示交替 HCA/CSA。"""

    compress_ratios: tuple = ()
    """每层 HCA 的压缩比率元组，长度应等于 n_layers。
    0 表示该层不使用 HCA 压缩，仅使用 CSA 滑动窗口。
    4 或 128 表示每 N 个 token 压缩为 1 个，用于长距离依赖。
    示例: (128, 4, 128, 4, 128, 4, 128, 4) 表示 HCA=128, CSA=4。"""

    index_topk: int = 512
    """HCA 全局检索的 top-k 数量。
    从压缩后的历史 KV 中选择最相关的 512 个位置参与注意力。"""

    index_n_heads: int = 4
    """Indexer 用于评分检索的注意力头数。
    使用独立的一组头来计算与压缩 KV 的相关性分数。"""

    index_head_dim: int = 128
    """Indexer 每个头的维度。"""

    compress_rope_theta: float = 40000.0
    """压缩 KV 使用的位置编码频率。
    通常高于 rope_theta，以更好地区分压缩后的位置信息。"""

    # ── Hyper-Connections（流形超连接）────────────────────────────────
    hc_mult: int = 4
    """Hyper-Connections 的倍数。维护 hc_mult 个并行残差流。
    这是 DeepSeek-V4 解决超深 MoE 模型梯度消失的核心机制。
    输入/输出维度: [batch, seq, hc_mult, dim]"""

    hc_sinkhorn_iters: int = 20
    """Sinkhorn 正则化的迭代次数。用于学习残差流的最优混合权重。"""

    hc_eps: float = 1e-6
    """Hyper-Connections 的数值稳定常数。"""

    # ── 便捷预设 ──────────────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "ModelArgs":
        """返回 Tiny（方案A）配置：适合单 GPU 训练。"""
        return cls(
            n_layers=12,
            compress_ratios=(128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0)
        )

    @classmethod
    def stage1(cls) -> "ModelArgs":
        """返回阶段1预训练配置：5M 参数 Moe 模型。"""
        return cls(
            max_batch_size=32,
            max_seq_len=1024,
            vocab_size=6400,
            dim=256,
            inter_dim=688,  # 2/3 * 4 * dim for SwiGLU
            moe_inter_dim=48,
            n_layers=8,
            n_dense_layers=0,  # 不使用 Dense MLP
            n_hash_layers=1,  # 使用 Hash 路由
            n_routed_experts=3,  # 路由专家数量
            n_shared_experts=1,  # 共享专家数量
            n_activated_experts=1,  # 每个 token 激活的专家数
            n_heads=8,
            q_lora_rank=64,
            kv_lora_rank=32,
            head_dim=128,
            qk_nope_head_dim=96,
            qk_rope_head_dim=32,
            v_head_dim=96,
            original_seq_len=4096,
            rope_theta=10000.0,
            rope_factor=1.0,
            attn_patterns=(0, 1, 0, 1, 0, 1, 0, 1),  # 交替 HCA/CSA
            compress_ratios=(128, 4, 128, 4, 128, 4, 128, 4),  # HCA=128, CSA=4
            hc_mult=1,  # 不使用 Hyper-Connections
            score_func="sqrtsoftplus",  # 路由打分函数
            route_scale=1.0,  # 路由权重缩放
        )

    @classmethod
    def original(cls) -> "ModelArgs":
        """返回原始 DeepSeek V4 配置：约 671B 总参数，约 37B 激活参数。"""
        return cls(
            max_batch_size=8,
            max_seq_len=16384,
            vocab_size=129280,
            dim=7168,
            inter_dim=28672,
            moe_inter_dim=3072,
            n_layers=61,
            n_dense_layers=3,
            n_hash_layers=3,
            n_heads=128,
            n_routed_experts=384,
            n_shared_experts=1,
            n_activated_experts=6,
            n_expert_groups=8,
            n_limited_groups=4,
            score_func="sqrtsoftplus",
            route_scale=2.5,
            q_lora_rank=1536,
            kv_lora_rank=512,
            head_dim=512,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            v_head_dim=448,
            original_seq_len=65536,
            rope_theta=10000.0,
            rope_factor=16,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            compress_ratios=(128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0),
            index_topk=1024,
            index_n_heads=64,
            index_head_dim=128,
            compress_rope_theta=160000.0,
            hc_mult=4,
            hc_sinkhorn_iters=20,
        )