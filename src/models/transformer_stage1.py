"""
阶段1 Transformer 模型（支持 Hyper-Connections）

结构：
    Embedding → HC-expand → [Block × n_layers] → HC-merge → RMSNorm → Head → Logits

特性：
    - 全部使用 MoE（第0层 hash 路由，其余 learned-gate 路由）
    - 支持 Hyper-Connections（hc_mult 个并行残差流）
    - 支持 MTP（多令牌预测）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelArgs
from .layers import RMSNorm
from .mla_attention_stage1 import MLAStage1
from .moe import MoE
from .kernel import hc_split_sinkhorn


def _init_hc_params(module: nn.Module, mix_hc: int, hc_dim: int) -> None:
    """初始化 Hyper-Connections 参数。"""
    with torch.no_grad():
        module.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        module.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        module.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    nn.init.normal_(module.hc_fn, std=0.02)
    nn.init.zeros_(module.hc_base)
    nn.init.ones_(module.hc_scale)


def _compute_hc_mixes(x: torch.Tensor, hc_fn: torch.Tensor, eps: float) -> torch.Tensor:
    """计算 HC 混合分数。"""
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    return mixes


def _weighted_sum(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """用权重对 hc_mult 维度做加权求和。"""
    return torch.sum(weights.unsqueeze(-1) * x, dim=2)


class BlockStage1(nn.Module):
    """带 Hyper-Connections 的 Transformer Block。"""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult

        self.attn = MLAStage1(layer_id, args)
        self.mlp = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.mlp_norm = RMSNorm(args.dim, args.norm_eps)

        if args.hc_mult > 1:
            mix_hc = (2 + args.hc_mult) * args.hc_mult
            hc_dim = args.hc_mult * args.dim
            _init_hc_params(self, mix_hc, hc_dim)
            _init_hc_params(self, mix_hc, hc_dim)
            # 重命名以区分 attn/mlp
            self.hc_attn_fn = self.hc_fn
            self.hc_attn_base = self.hc_base
            self.hc_attn_scale = self.hc_scale
            delattr(self, "hc_fn")
            delattr(self, "hc_base")
            delattr(self, "hc_scale")

            _init_hc_params(self, mix_hc, hc_dim)
            self.hc_mlp_fn = self.hc_fn
            self.hc_mlp_base = self.hc_base
            self.hc_mlp_scale = self.hc_scale
            delattr(self, "hc_fn")
            delattr(self, "hc_base")
            delattr(self, "hc_scale")

    def _hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HC 预处理：将 hc_mult 个副本混合为 1 个。"""
        shape, dtype = x.size(), x.dtype

        mixes = _compute_hc_mixes(x, hc_fn, self.norm_eps)
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, 5, self.norm_eps)

        y = _weighted_sum(x.view(shape), pre)
        return y.to(dtype), post, comb

    def _hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """HC 后处理：将 1 个扩展回 hc_mult 个副本，与残差混合。"""
        y = post.unsqueeze(-1) * x.unsqueeze(-2)
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """前向传播。"""
        if self.hc_mult == 1:
            # 标准残差连接
            h = x + self.attn(self.attn_norm(x), start_pos)
            out = h + self.mlp(self.mlp_norm(h), input_ids)
            return out

        # Attention 子层（带 HC）
        residual = x
        x, post, comb = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn(self.attn_norm(x), start_pos)
        x = self._hc_post(x, residual, post, comb)

        # MoE 子层（带 HC）
        residual = x
        x, post, comb = self._hc_pre(residual, self.hc_mlp_fn, self.hc_mlp_scale, self.hc_mlp_base)
        x = self.mlp(self.mlp_norm(x), input_ids)
        x = self._hc_post(x, residual, post, comb)

        return x


class TransformerStage1(nn.Module):
    """阶段1 Transformer 模型（支持 Hyper-Connections）。"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        self.hc_mult = args.hc_mult

        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([BlockStage1(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # MTP 支持
        self.mtp_enabled = getattr(args, "mtp_num_future_tokens", 0) > 0
        if self.mtp_enabled:
            self.mtp_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # HC head 参数（当 hc_mult > 1 时）
        if args.hc_mult > 1:
            hc_dim = args.hc_mult * args.dim
            with torch.no_grad():
                self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, hc_dim, dtype=torch.float32))
                self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult, dtype=torch.float32))
                self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
            nn.init.normal_(self.hc_head_fn, std=0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)

        self._init_weights()

        if getattr(args, "tie_word_embeddings", True):
            self.head.weight = self.embed.weight

    def _init_weights(self):
        """初始化模型权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """HC 最终合并：将 hc_mult 个副本合并为 1 个。"""
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.args.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt

        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.args.norm_eps
        pre = pre / pre.sum(dim=-1, keepdim=True)

        y = _weighted_sum(x.view(shape), pre)
        return y.to(dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        return_mtp: bool = False,
    ) -> torch.Tensor:
        """前向传播。"""
        h = self.embed(tokens)

        # HC 扩展
        if self.hc_mult > 1:
            h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)

        # 逐层传递
        for layer in self.layers:
            h = layer(h, tokens, start_pos)

        # HC 合并
        if self.hc_mult > 1:
            h = self._hc_head(h)

        h = self.norm(h)
        logits = self.head(h)

        if return_mtp and self.mtp_enabled:
            mtp_logits = self.mtp_head(h)
            return logits, mtp_logits

        return logits

    def get_num_params(self) -> int:
        """获取模型总参数量。"""
        return sum(p.numel() for p in self.parameters())


DeepSeekV4Stage1 = TransformerStage1
