"""
MLA (Multi-Head Latent Attention) - 阶段1实现

支持交替 CSA/HCA 稀疏注意力模式：
- HCA (mode=0): 纯全局压缩，compress_ratio=128
- CSA (mode=1): 局部窗口 + 轻量压缩，window_size=128, compress_ratio=4

保持与阶段1兼容的接口，同时引入稀疏注意力机制。
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelArgs
from .layers import RMSNorm
from .rotary_embedding import apply_rotary_emb, precompute_freqs_cis


class Compressor(nn.Module):
    """KV 缓存压缩器 - 门控池化压缩为低频表示。"""

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = head_dim - args.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4

        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim))
        self.wkv = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, args.norm_eps)

        self.kv_cache: Optional[Tensor] = None
        self.freqs_cis: Optional[Tensor] = None

        max_bsz = args.max_batch_size
        state_len = coff * compress_ratio
        self.register_buffer(
            "kv_state",
            torch.zeros(max_bsz, state_len, coff * head_dim),
            persistent=False
        )
        self.register_buffer(
            "score_state",
            torch.full((max_bsz, state_len, coff * head_dim), float("-inf")),
            persistent=False
        )

    def overlap_transform(self, tensor: Tensor, value: float = 0) -> Tensor:
        """重叠窗口变换，使压缩边界更平滑。"""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def _compress_prefill(self, x: Tensor, seqlen: int, bsz: int) -> Tuple[Optional[Tensor], bool, int]:
        """预填充阶段压缩。"""
        ratio = self.compress_ratio
        kv = self.wkv(x)
        score = self.wgate(x)

        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if self.overlap else 0

        if self.overlap and cutoff >= ratio:
            self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio:cutoff]
            self.score_state[:bsz, :ratio] = score[:, cutoff - ratio:cutoff] + self.ape[:ratio]

        if remainder > 0:
            kv, self.kv_state[:bsz, offset:offset + remainder] = kv.split([cutoff, remainder], dim=1)
            self.score_state[:bsz, offset:offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
            score = score[:, :cutoff]

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if self.overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))

        kv = (kv * score.softmax(dim=2)).sum(dim=2)

        return kv, should_compress, cutoff

    def _compress_decode(self, x: Tensor, start_pos: int, bsz: int) -> Tuple[Optional[Tensor], bool]:
        """解码阶段压缩。"""
        ratio = self.compress_ratio
        kv = self.wkv(x)
        score = self.wgate(x) + self.ape[start_pos % ratio]

        should_compress = (start_pos + 1) % ratio == 0

        if self.overlap:
            self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
            self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)

            if should_compress:
                kv_state = torch.cat([
                    self.kv_state[:bsz, :ratio, :self.head_dim],
                    self.kv_state[:bsz, ratio:, self.head_dim:]
                ], dim=1)
                score_state = torch.cat([
                    self.score_state[:bsz, :ratio, :self.head_dim],
                    self.score_state[:bsz, ratio:, self.head_dim:]
                ], dim=1)
                kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
        else:
            self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
            self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)

            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

        return kv, should_compress

    def forward(self, x: Tensor, start_pos: int) -> Optional[Tensor]:
        """前向压缩，未达到压缩条件时返回 None。"""
        assert self.kv_cache is not None

        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        x = x.float()

        if start_pos == 0:
            kv, should_compress, cutoff = self._compress_prefill(x, seqlen, bsz)
        else:
            kv, should_compress = self._compress_decode(x, start_pos, bsz)
            cutoff = 0

        if not should_compress or kv is None:
            return None

        kv = self.norm(kv)

        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:self.compress_ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)

        # 确保 freqs_cis 与 kv 在同一设备
        if freqs_cis.device != kv.device:
            freqs_cis = freqs_cis.to(kv.device)

        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], dim=-1)

        # 使用 no_grad 避免计算图问题
        with torch.no_grad():
            if start_pos == 0:
                self.kv_cache[:bsz, :seqlen // self.compress_ratio, :kv.size(-1)].copy_(kv)
            else:
                self.kv_cache[:bsz, start_pos // self.compress_ratio, :kv.size(-1)].copy_(kv.squeeze(1))

        return kv


class Indexer(nn.Module):
    """压缩 KV 索引器 - 基于低秩 latent 学习选择最相关的压缩 KV 位置。"""

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.latent_dim = args.kv_lora_rank  # 使用低秩维度而非 head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.q_lora_rank = args.q_lora_rank

        # Query 投影到低秩维度进行检索
        q_input_dim = self.q_lora_rank if self.q_lora_rank > 0 else self.dim
        self.wq_b = nn.Linear(q_input_dim, self.n_heads * self.latent_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.latent_dim ** -0.5

        # 引用外部缓存（由 MLAStage1 设置）
        self.kv_cache: Optional[Tensor] = None
        self.freqs_cis: Optional[Tensor] = None

    def forward(self, x: Tensor, qr: Tensor, start_pos: int) -> Tensor:
        """检索最相关的压缩 KV 位置。返回 [0, cache_size-1] 范围内的索引。"""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        end_pos = start_pos + seqlen

        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.latent_dim))

        # 使用低秩 latent 计算检索分数
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        # kv_cache: [B, max_cache_len, latent_dim+rope_dim], 取 latent 部分
        kv_latent = self.kv_cache[:bsz, :end_pos // ratio, :self.latent_dim]

        # 确保 kv_latent 与 q 在同一设备
        if kv_latent.device != q.device:
            kv_latent = kv_latent.to(q.device)

        index_score = torch.einsum("bshd,btd->bsht", q, kv_latent)
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)

        if start_pos == 0:
            arange = torch.arange(seqlen // ratio, device=index_score.device)
            mask = arange.repeat(seqlen, 1) >= torch.arange(1, seqlen + 1, device=index_score.device).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)

        n_available = end_pos // ratio
        if n_available == 0:
            # 没有压缩位置，返回位置 0 作为回退
            return torch.zeros(bsz, seqlen, 1, device=x.device, dtype=torch.int64)

        k = min(self.index_topk, n_available)
        # 使用 .detach() 完全脱离计算图，避免反向传播问题
        topk_idxs = index_score.topk(k, dim=-1)[1].detach()

        # 应用因果 mask（未来位置标记为 -1）
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=topk_idxs.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs)

            # 确保每行至少有一个有效索引
            # 简单方法：将第一列的 -1 替换为 0（无论如何都替换，不影响有效值）
            first_col = topk_idxs[:, :, 0]
            first_col = torch.clamp(first_col, min=0)  # -1 变成 0，其他值不变
            topk_idxs = torch.cat([first_col.unsqueeze(-1), topk_idxs[:, :, 1:]], dim=-1)

        return topk_idxs


def get_window_indices(window_size: int, bsz: int, seqlen: int, start_pos: int) -> Tensor:
    """生成滑动窗口索引（CSA 局部注意力）。

    每个位置只关注最近的 window_size 个 token，实现局部稠密注意力。
    """
    win = window_size

    with torch.no_grad():
        if start_pos == 0:
            # 预填充阶段：返回绝对位置索引
            base = torch.arange(seqlen, device='cpu').unsqueeze(1)
            indices = (base - win + 1).clamp(0) + torch.arange(min(seqlen, win), device='cpu')
            indices = torch.where(indices > base, -1, indices)
            return indices.unsqueeze(0).expand(bsz, -1, -1).to(torch.int32)

        # 解码阶段
        if start_pos >= win - 1:
            start_pos %= win
            indices = torch.cat([
                torch.arange(start_pos + 1, win, device='cpu'),
                torch.arange(0, start_pos + 1, device='cpu')
            ], dim=0)
        else:
            indices = F.pad(torch.arange(start_pos + 1, device='cpu'),
                          (0, win - start_pos - 1), value=-1)
        return indices.unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1).to(torch.int64)


def get_compress_indices(compress_ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int) -> Tensor:
    """生成压缩 KV 的固定间隔采样索引（HCA 全局稀疏）。

    确保每个位置至少有一个有效索引，避免全 mask 导致 NaN。
    """
    with torch.no_grad():  # 确保不追踪计算图
        if start_pos > 0:
            n_compress = (start_pos + 1) // compress_ratio
            if n_compress == 0:
                indices = torch.zeros(1, device='cpu', dtype=torch.int64)
            else:
                indices = torch.arange(0, n_compress, device='cpu') + offset
        else:
            n_available = seqlen // compress_ratio
            if n_available == 0:
                indices = torch.zeros(seqlen, 1, device='cpu', dtype=torch.int64)
            else:
                base = torch.arange(seqlen, device='cpu').unsqueeze(1)
                indices = torch.arange(n_available, device='cpu').repeat(seqlen, 1)
                mask = indices >= torch.arange(1, seqlen + 1, device='cpu').unsqueeze(1) // compress_ratio
                indices = torch.where(mask, -1, indices + offset)
                all_masked = mask.all(dim=1)
                if all_masked.any():
                    indices[all_masked, 0] = offset

        return indices.unsqueeze(0).expand(bsz, -1, -1).to(torch.int64)


class MLAStage1(nn.Module):
    """
    阶段1 MLA 实现，支持交替 CSA/HCA 模式。

    Args:
        layer_id: 层索引，用于确定注意力模式。
        args: 模型配置参数。
    """

    def _init_query_proj(self, args: ModelArgs):
        """初始化 Query 投影层。"""
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

    def _init_kv_proj(self, args: ModelArgs):
        """初始化 KV 投影层。"""
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.norm_eps)
        self.wk_b = nn.Linear(self.kv_lora_rank, self.n_heads * self.nope_head_dim, bias=False)
        self.wv_b = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)

    def _init_caches(self, args: ModelArgs):
        """初始化 KV 缓存。"""
        # CSA 窗口缓存
        if self.mode == 1:
            cache_dim = self.kv_lora_rank + self.rope_head_dim
            self.register_buffer(
                "kv_window_cache",
                torch.zeros(args.max_batch_size, self.window_size, cache_dim),
                persistent=False
            )
        else:
            self.kv_window_cache = None

        # HCA 压缩缓存
        # 存储低秩 latent + k_pe，用于通过 wk_b/wv_b 恢复 K 和 V
        if self.compress_ratio > 0:
            max_cache_len = args.max_seq_len // self.compress_ratio
            cache_dim = self.kv_lora_rank + self.rope_head_dim
            self.register_buffer(
                "kv_compress_cache",
                torch.zeros(args.max_batch_size, max_cache_len, cache_dim),
                persistent=False
            )
            self.compressor = Compressor(args, self.compress_ratio, cache_dim)

            if self.mode == 1 and self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
                # 绑定缓存引用（freqs_cis 在 _init_rope 中绑定）
                self.indexer.kv_cache = self.kv_compress_cache
            else:
                self.indexer = None
        else:
            self.kv_compress_cache = None
            self.compressor = None
            self.indexer = None

    def _init_rope(self, args: ModelArgs):
        """初始化 RoPE 频率缓存。"""
        if self.compress_ratio > 0:
            orig_len, theta = args.original_seq_len, args.compress_rope_theta
        else:
            orig_len, theta = 0, args.rope_theta

        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, args.max_seq_len, orig_len,
            theta, args.rope_factor, args.beta_fast, args.beta_slow
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        if self.compressor is not None:
            self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads

        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim

        attn_patterns: List[int] = list(getattr(args, 'attn_patterns', [0] * args.n_layers))
        self.mode = attn_patterns[layer_id] if layer_id < len(attn_patterns) else 0

        compress_ratios: List[int] = list(getattr(args, 'compress_ratios', [0] * args.n_layers))
        self.compress_ratio = compress_ratios[layer_id] if layer_id < len(compress_ratios) else 0

        self.window_size = args.window_size if self.mode == 1 else 0
        self.softmax_scale = self.head_dim ** -0.5

        self._init_query_proj(args)
        self._init_kv_proj(args)
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self._init_caches(args)
        self._init_rope(args)

        self._init_attn_sink()

    def _init_attn_sink(self):
        """初始化 Attention Sink - 稳定长序列注意力的可学习偏差。"""
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads))

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """前向传播，根据 mode 选择 HCA 或 CSA 路径。"""
        if self.mode == 0:
            return self._forward_hca(x, start_pos)
        return self._forward_csa(x, start_pos)

    def _apply_attn_sink(self, scores: Tensor) -> Tensor:
        """应用 Attention Sink 偏差到注意力分数。[b,s,h,k] + [1,1,h,1]"""
        return scores + self.attn_sink.view(1, 1, -1, 1)

    def _forward_hca(self, x: Tensor, start_pos: int) -> Tensor:
        """纯 HCA 模式：只使用全局压缩 KV。"""
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        freqs_cis = self.freqs_cis[start_pos:end_pos]

        # Query 投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

        if freqs_cis.device != q.device:
            freqs_cis = freqs_cis.to(q.device)

        q = torch.cat([q[..., :-self.rope_head_dim], apply_rotary_emb(q[..., -self.rope_head_dim:], freqs_cis)], dim=-1)

        # KV 投影
        kv = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)

        # 更新压缩缓存
        if self.compressor is not None and self.kv_compress_cache is not None:
            self.compressor.kv_cache = self.kv_compress_cache
            self.compressor(x, start_pos)

        # 生成压缩索引
        compress_idxs = get_compress_indices(self.compress_ratio, bsz, seqlen, start_pos, 0)

        # 从压缩缓存收集 KV (低秩 latent + k_pe)
        sparse_kv = self._gather_kv(self.kv_compress_cache[:bsz], compress_idxs)

        # 投影恢复 K_nope 和 V
        latent = sparse_kv[..., :self.kv_lora_rank]
        k_nope_sparse = torch.einsum("bskc,hdc->bskhd", latent, self.wk_b.weight.view(self.n_local_heads, -1, self.kv_lora_rank))
        v_sparse = torch.einsum("bskc,hdc->bskhd", latent, self.wv_b.weight.view(self.n_local_heads, -1, self.kv_lora_rank))

        # k_pe 直接扩展
        k_pe_sparse = sparse_kv[..., self.kv_lora_rank:].unsqueeze(3).expand(-1, -1, -1, self.n_local_heads, -1)
        k_sparse = torch.cat([k_nope_sparse, k_pe_sparse], dim=-1)
        k_sparse = k_sparse.transpose(2, 3)
        v_sparse = v_sparse.transpose(2, 3)

        # 稀疏注意力
        scores = torch.einsum("bshd,bshkd->bshk", q, k_sparse) * self.softmax_scale
        scores = self._apply_attn_sink(scores)

        if compress_idxs.device != scores.device:
            compress_idxs = compress_idxs.to(scores.device)

        mask = compress_idxs < 0
        scores += torch.where(mask.unsqueeze(2), float("-inf"), 0)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        o = torch.einsum("bshk,bshkd->bshd", scores, v_sparse)

        return self.wo(o.flatten(2))

    def _update_window_cache(self, kv_full: Tensor, start_pos: int, bsz: int, seqlen: int):
        """更新 CSA 窗口缓存。"""
        win = self.window_size
        with torch.no_grad():  # 避免计算图问题
            if start_pos == 0:
                if seqlen <= win:
                    self.kv_window_cache[:bsz, :seqlen] = kv_full
                else:
                    cutoff = seqlen % win
                    self.kv_window_cache[:bsz, cutoff:win], self.kv_window_cache[:bsz, :cutoff] = \
                        kv_full[:, -win:].split([win - cutoff, cutoff], dim=1)
            else:
                self.kv_window_cache[:bsz, start_pos % win] = kv_full.squeeze(1)

    def _get_sparse_indices_csa(self, x: Tensor, qr: Tensor, start_pos: int, seqlen: int, bsz: int) -> Tensor:
        """生成 CSA 的双索引（窗口 + 压缩）。"""
        win = self.window_size
        window_idxs = get_window_indices(win, bsz, seqlen, start_pos)

        # 预填充阶段：将绝对位置索引映射到环形缓冲区位置，保留无效标记
        if start_pos == 0:
            valid_mask = window_idxs >= 0
            window_idxs = torch.where(valid_mask, window_idxs % win, window_idxs)

        if self.indexer is not None:
            compress_idxs = self.indexer(x, qr, start_pos)
        else:
            compress_idxs = get_compress_indices(self.compress_ratio, bsz, seqlen, start_pos, 0)

        device = x.device
        if window_idxs.device != device:
            window_idxs = window_idxs.to(device)
        if compress_idxs.device != device:
            compress_idxs = compress_idxs.to(device)

        return torch.cat([window_idxs, compress_idxs], dim=-1)

    def _compute_sparse_attention(self, q: Tensor, sparse_kv: Tensor, indices: Tensor) -> Tensor:
        """计算稀疏注意力。"""
        k_nope_sparse = torch.einsum("bskc,hdc->bskhd", sparse_kv[..., :self.kv_lora_rank], self.wk_b.weight.view(self.n_local_heads, -1, self.kv_lora_rank))

        v_sparse = torch.einsum("bskc,hdc->bskhd", sparse_kv[..., :self.kv_lora_rank], self.wv_b.weight.view(self.n_local_heads, -1, self.kv_lora_rank))

        k_pe_sparse = sparse_kv[..., self.kv_lora_rank:].unsqueeze(3).expand(-1, -1, -1, self.n_local_heads, -1)
        k_sparse = torch.cat([k_nope_sparse, k_pe_sparse], dim=-1).transpose(2, 3)
        v_sparse = v_sparse.transpose(2, 3)

        scores = torch.einsum("bshd,bshkd->bshk", q, k_sparse) * self.softmax_scale
        scores = self._apply_attn_sink(scores)

        if indices.device != scores.device:
            indices = indices.to(scores.device)

        mask = indices < 0
        scores += torch.where(mask.unsqueeze(2), float("-inf"), 0)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(q)

        return torch.einsum("bshk,bshkd->bshd", scores, v_sparse)

    def _forward_csa(self, x: Tensor, start_pos: int) -> Tensor:
        """CSA 模式：局部窗口 + 轻量压缩检索。"""
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        win = self.window_size

        freqs_cis = self.freqs_cis[start_pos:end_pos]

        # Query 投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
            qr = x
        else:
            qr = self.q_norm(self.wq_a(x))
            q = self.wq_b(qr)

        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

        if freqs_cis.device != q.device:
            freqs_cis = freqs_cis.to(q.device)

        q = torch.cat([q[..., :-self.rope_head_dim], apply_rotary_emb(q[..., -self.rope_head_dim:], freqs_cis)], dim=-1)

        # KV 投影与窗口缓存更新
        kv = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        kv_full = torch.cat([kv_latent, k_pe], dim=-1)
        self._update_window_cache(kv_full, start_pos, bsz, seqlen)

        # 更新压缩缓存
        if self.compressor is not None:
            self.compressor.kv_cache = self.kv_compress_cache
            self.compressor(x, start_pos)

        # 生成索引并收集 KV
        topk_idxs = self._get_sparse_indices_csa(x, qr, start_pos, seqlen, bsz)

        # 收集窗口 KV（前 win 个索引，环形缓冲区位置 [0, win-1]）
        kv_window = self._gather_kv(self.kv_window_cache[:bsz], topk_idxs[:, :, :win])

        # 收集压缩 KV（后 index_topk 个索引，范围 [0, cache_size-1]）
        kv_compress = self._gather_kv(self.kv_compress_cache[:bsz], topk_idxs[:, :, win:])

        sparse_kv = torch.cat([kv_window, kv_compress], dim=2)

        # 计算注意力
        o = self._compute_sparse_attention(q, sparse_kv, topk_idxs)

        return self.wo(o.flatten(2))

    def _gather_kv(self, kv_cache: Tensor, indices: Tensor) -> Tensor:
        """根据索引从 KV 缓存中收集稀疏 KV。"""
        bsz, seqlen, k = indices.size()
        head_dim = kv_cache.size(-1)

        # 确保索引与缓存设备一致
        if indices.device != kv_cache.device:
            indices = indices.to(kv_cache.device)

        safe_indices = indices.clamp(min=0)
        kv = kv_cache.gather(1, safe_indices.view(bsz, -1).unsqueeze(-1).expand(-1, -1, head_dim))
        kv = kv.view(bsz, seqlen, k, head_dim)

        mask = (indices < 0).unsqueeze(-1)
        kv = torch.where(mask, torch.zeros_like(kv), kv)

        return kv

    def get_kv_cache_size(self) -> int:
        """获取 KV 缓存大小（以元素计）。"""
        total = 0
        if self.kv_window_cache is not None:
            total += self.kv_window_cache.numel()
        if self.kv_compress_cache is not None:
            total += self.kv_compress_cache.numel()
        return total
