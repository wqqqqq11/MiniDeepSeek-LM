"""
阶段1 Transformer 模型（MoE 版本）

MoE架构：第0层使用Hash路由MoE，其余层使用Learned-gate MoE

结构：
    Embedding → [RMSNorm → MLA → RMSNorm → MoE] × n_layers
            → RMSNorm → Head → Logits
"""

import torch
import torch.nn as nn
from .config import ModelArgs
from .layers import RMSNorm
from .mla_attention_stage1 import MLAStage1
from .moe import MoE


class BlockStage1(nn.Module):
    """
    阶段1 Transformer Block（MoE版本）。

    结构（Pre-Norm）：
        x_norm = RMSNorm(x)
        h = x + MLA(x_norm)
        h_norm = RMSNorm(h)
        out = h + MoE(h_norm, input_ids)

    Args:
        layer_id: 层索引。
        args: 模型配置参数。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = MLAStage1(layer_id, args)
        self.mlp_norm = RMSNorm(args.dim, args.norm_eps)
        self.mlp = MoE(layer_id, args)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 [batch, seq, dim]。
            input_ids: 输入token IDs [batch, seq]，用于hash路由。
            start_pos: 序列起始位置（用于KV缓存）。

        Returns:
            torch.Tensor: 输出张量 [batch, seq, dim]。
        """
        # Attention 子层
        h = x + self.attn(self.attn_norm(x), start_pos)

        # MoE 子层（需要input_ids用于hash路由）
        out = h + self.mlp(self.mlp_norm(h), input_ids)

        return out


class TransformerStage1(nn.Module):
    """
    阶段1 Transformer 模型（MoE 版本）。

    结构：
        tokens → Embedding → [Block × n_layers] → RMSNorm → Head → logits

    特性：
        - 全部使用MoE（第0层hash路由，其余learned-gate路由）
        - 支持 MTP（多令牌预测）

    Args:
        args: 模型配置参数。
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim

        # 词嵌入
        self.embed = nn.Embedding(args.vocab_size, args.dim)

        # Transformer 层（全部使用MoE）
        self.layers = nn.ModuleList([
            BlockStage1(i, args) for i in range(args.n_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(args.dim, args.norm_eps)

        # 输出头
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 可选：MTP头
        self.mtp_enabled = getattr(args, "mtp_num_future_tokens", 0) > 0
        if self.mtp_enabled:
            self.mtp_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 权重初始化
        self._init_weights()

        # 权重共享
        if getattr(args, "tie_word_embeddings", True):
            self.head.weight = self.embed.weight

    def _init_weights(self):
        """初始化模型权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        return_mtp: bool = False
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            tokens: 输入 token ID [batch_size, seq_len]。
            start_pos: 序列起始位置（用于 KV 缓存）。
            return_mtp: 是否返回 MTP 预测结果。

        Returns:
            torch.Tensor: 主 logits [batch_size, seq_len, vocab_size]。
        """
        batch_size, seq_len = tokens.shape

        # 词嵌入
        h = self.embed(tokens)

        # 逐层传递（传递tokens用于hash路由）
        for layer in self.layers:
            h = layer(h, tokens, start_pos)

        # 最终归一化
        h = self.norm(h)

        # 主输出头
        logits = self.head(h)

        # MTP 输出
        if return_mtp and self.mtp_enabled:
            mtp_logits = self.mtp_head(h)
            return logits, mtp_logits

        return logits

    def get_num_params(self) -> int:
        """获取模型总参数量。"""
        return sum(p.numel() for p in self.parameters())


# 别名
DeepSeekV4Stage1 = TransformerStage1
