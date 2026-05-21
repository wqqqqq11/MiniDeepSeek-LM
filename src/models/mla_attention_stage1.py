"""
MLA (Multi-Head Latent Attention) - 阶段1实现

支持稀疏注意力机制：
- 所有层都有滑动窗口注意力 (window_size)
- 按层配置压缩比率 (compress_ratios):
  * 高压缩层 (128): 窗口 + 重度压缩检索
  * 低压缩层 (4): 窗口 + 轻量压缩检索
  * 无压缩层 (0): 纯窗口注意力

保持与阶段1兼容的接口。
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

        if self.overlap:
            max_groups = args.max_seq_len // compress_ratio
            self.register_buffer(
                "_ol_kv",
                torch.zeros(max_bsz, max_groups, 2 * compress_ratio, head_dim),
                persistent=False,
            )
            self.register_buffer(
                "_ol_score",
                torch.full((max_bsz, max_groups, 2 * compress_ratio, head_dim), float("-inf")),
                persistent=False,
            )

    def overlap_transform(self, tensor: Tensor, is_score: bool = False) -> Tensor:
        """重叠窗口变换 - 使用预分配 buffer 避免反复 malloc。"""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        with torch.no_grad():
            buf = self._ol_score[:b, :s] if is_score else self._ol_kv[:b, :s]
            buf[:, :, ratio:].copy_(tensor[:, :, :, d:])
            buf[:, 1:, :ratio].copy_(tensor[:, :-1, :, :d])
            if is_score:
                buf[:, 0, :ratio] = float("-inf")
            else:
                buf[:, 0, :ratio] = 0
        return buf.clone()

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
            kv = self.overlap_transform(kv, is_score=False)
            score = self.overlap_transform(score, is_score=True)

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

        if freqs_cis.device != kv.device:
            freqs_cis = freqs_cis.to(kv.device)

        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], dim=-1)

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
        self.latent_dim = args.kv_lora_rank
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.q_lora_rank = args.q_lora_rank

        q_input_dim = self.q_lora_rank if self.q_lora_rank > 0 else self.dim
        self.wq_b = nn.Linear(q_input_dim, self.n_heads * self.latent_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.latent_dim ** -0.5

        self.kv_cache: Optional[Tensor] = None
        self.freqs_cis: Optional[Tensor] = None

    def forward(self, x: Tensor, qr: Tensor, start_pos: int, offset: int, device: torch.device) -> Tensor:
        """检索最相关的压缩 KV 位置。返回加 offset 的索引。"""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        end_pos = start_pos + seqlen

        with torch.no_grad():
            q = self.wq_b(qr)
            q = q.unflatten(-1, (self.n_heads, self.latent_dim))

            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
            kv_latent = self.kv_cache[:bsz, :end_pos // ratio, :self.latent_dim]

            if kv_latent.device != device:
                kv_latent = kv_latent.to(device)

            index_score = torch.einsum("bshd,btd->bsht", q, kv_latent)
            index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)

            if start_pos == 0:
                arange = torch.arange(seqlen // ratio, device=device)
                mask = arange.repeat(seqlen, 1) >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
                index_score += torch.where(mask, float("-inf"), 0)

            n_available = end_pos // ratio
            if n_available == 0:
                return torch.zeros(bsz, seqlen, 1, device=device, dtype=torch.int64) + offset

            k = min(self.index_topk, n_available)
            topk_idxs = index_score.topk(k, dim=-1)[1]

            if start_pos == 0:
                mask = topk_idxs >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
                topk_idxs = torch.where(mask, -1, topk_idxs + offset)
                first_col = topk_idxs[:, :, 0]
                first_col = torch.clamp(first_col, min=0)
                topk_idxs = torch.cat([first_col.unsqueeze(-1), topk_idxs[:, :, 1:]], dim=-1)
            else:
                topk_idxs = topk_idxs + offset

            return topk_idxs


def get_window_indices(window_size: int, bsz: int, seqlen: int, start_pos: int, device: torch.device) -> Tensor:
    """生成滑动窗口索引（CSA 局部注意力）。"""
    win = window_size

    with torch.no_grad():
        if start_pos == 0:
            base = torch.arange(seqlen, device=device).unsqueeze(1)
            indices = (base - win + 1).clamp(0) + torch.arange(min(seqlen, win), device=device)
            indices = torch.where(indices > base, -1, indices)
            return indices.unsqueeze(0).expand(bsz, -1, -1).to(torch.int32)

        if start_pos >= win - 1:
            start_pos %= win
            indices = torch.cat([
                torch.arange(start_pos + 1, win, device=device),
                torch.arange(0, start_pos + 1, device=device)
            ], dim=0)
        else:
            indices = F.pad(torch.arange(start_pos + 1, device=device),
                          (0, win - start_pos - 1), value=-1)
        return indices.unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1).to(torch.int64)


def get_compress_indices(compress_ratio: int, bsz: int, seqlen: int, start_pos: int,
                         offset: int, device: torch.device) -> Tensor:
    """生成压缩 KV 的固定间隔采样索引（HCA 全局稀疏）。"""
    with torch.no_grad():
        if start_pos > 0:
            n_compress = (start_pos + 1) // compress_ratio
            if n_compress == 0:
                indices = torch.zeros(1, device=device, dtype=torch.int64)
            else:
                indices = torch.arange(0, n_compress, device=device) + offset
        else:
            n_available = seqlen // compress_ratio
            if n_available == 0:
                indices = torch.zeros(seqlen, 1, device=device, dtype=torch.int64)
            else:
                base = torch.arange(seqlen, device=device).unsqueeze(1)
                indices = torch.arange(n_available, device=device).repeat(seqlen, 1)
                mask = indices >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // compress_ratio
                indices = torch.where(mask, -1, indices + offset)
                all_masked = mask.all(dim=1)
                if all_masked.any():
                    indices[all_masked, 0] = offset

        return indices.unsqueeze(0).expand(bsz, -1, -1).to(torch.int64)


class MLAStage1(nn.Module):
    """阶段1 MLA 实现，支持交替 CSA/HCA 模式。"""

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

        compress_ratios: List[int] = list(getattr(args, 'compress_ratios', [0] * args.n_layers))
        self.compress_ratio = compress_ratios[layer_id] if layer_id < len(compress_ratios) else 0

        self.window_size = args.window_size
        self.softmax_scale = self.head_dim ** -0.5

        self._init_query_proj(args)
        self._init_kv_proj(args)
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self._init_caches(args)
        self._init_rope(args)
        self._init_attn_sink()

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
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, args.norm_eps)

    def _init_caches(self, args: ModelArgs):
        """初始化 KV 缓存：所有层都有窗口缓存，按需分配压缩缓存。"""
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, self.window_size, self.head_dim),
            persistent=False
        )

        if self.compress_ratio > 0:
            max_cache_len = args.max_seq_len // self.compress_ratio
            self.register_buffer(
                "kv_compress_cache",
                torch.zeros(args.max_batch_size, max_cache_len, self.head_dim),
                persistent=False
            )
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)

            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
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

    def _init_attn_sink(self):
        """初始化 Attention Sink - 稳定长序列注意力的可学习偏差。"""
        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))

    def _get_train_indices(self, bsz: int, seqlen: int, device: torch.device) -> Tensor:
        """训练阶段（start_pos=0）复用固定索引，避免重复计算。"""
        if not hasattr(self, '_idx_cache'):
            self._idx_cache: Optional[Tensor] = None
            self._idx_cache_key: tuple = (-1, -1)

        key = (bsz, seqlen)
        if self._idx_cache_key == key and self._idx_cache is not None:
            return self._idx_cache

        with torch.no_grad():
            idxs = get_window_indices(self.window_size, bsz, seqlen, 0, device)
            if self.compress_ratio > 0:
                offset = seqlen
                if self.indexer is not None:
                    # indexer 需要 x，训练时无法缓存，走固定索引
                    c_idxs = get_compress_indices(
                        self.compress_ratio, bsz, seqlen, 0, offset, device
                    )
                else:
                    c_idxs = get_compress_indices(
                        self.compress_ratio, bsz, seqlen, 0, offset, device
                    )
                idxs = torch.cat([idxs, c_idxs], dim=-1)
            idxs = idxs.long()

        self._idx_cache = idxs
        self._idx_cache_key = key
        return idxs

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """前向传播：所有层都有窗口注意力，可选压缩注意力。"""
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        win = self.window_size
        rd = self.rope_head_dim

        freqs_cis = self.freqs_cis[start_pos:end_pos]

        # Query 投影
        qr = q = self.q_norm(self.wq_a(x)) if self.q_lora_rank > 0 else x
        if self.q_lora_rank > 0:
            q = self.wq_b(q)
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

        # 应用 RoPE - 原地更新避免额外内存分配
        q[..., -rd:] = apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV 投影
        kv = self.wkv(x)
        kv = self.kv_norm(kv)

        # 应用 RoPE 到 KV - 原地更新
        kv[..., -rd:] = apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # 更新压缩缓存
        if self.compressor is not None and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_compress_cache
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # 生成索引 - 训练阶段缓存，推理阶段实时计算
        if start_pos == 0:
            topk_idxs = self._get_train_indices(bsz, seqlen, x.device)
            if self.compress_ratio > 0 and self.indexer is not None:
                # CSA 层：使用可学习 indexer，训练时每步需重新计算
                offset = kv.size(1)
                with torch.no_grad():
                    compress_idxs = self.indexer(x, qr, start_pos, offset, x.device)
                    topk_idxs = torch.cat([
                        get_window_indices(win, bsz, seqlen, 0, x.device),
                        compress_idxs
                    ], dim=-1).long()
        else:
            with torch.no_grad():
                topk_idxs = get_window_indices(win, bsz, seqlen, start_pos, x.device)
                if self.compress_ratio > 0:
                    offset = win
                    if self.indexer is not None:
                        compress_idxs = self.indexer(x, qr, start_pos, offset, x.device)
                    else:
                        compress_idxs = get_compress_indices(
                            self.compress_ratio, bsz, seqlen, start_pos, offset, x.device
                        )
                    topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
                topk_idxs = topk_idxs.long()

        # 缓存更新与注意力计算
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio > 0:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = self._sparse_attn(q, kv, topk_idxs)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio > 0:
                self.compressor(x, start_pos)
            o = self._sparse_attn(q, self.kv_cache[:bsz], topk_idxs)

        return self.wo(o.flatten(2))

    def _apply_attn_sink(self, scores: Tensor) -> Tensor:
        """应用 Attention Sink 偏差到注意力分数。"""
        return scores + self.attn_sink.view(1, 1, -1, 1)

    def _sparse_attn(self, q: Tensor, kv: Tensor, indices: Tensor) -> Tensor:
        """计算稀疏注意力 - 优化版：单次 matmul，合并 nope+rope 计算。"""
        bsz, seqlen, k = indices.size()

        # gather 稀疏 KV: [b, s*k, head_dim]
        safe_indices = indices.clamp(min=0)
        flat_idx = safe_indices.view(bsz, -1).unsqueeze(-1).expand(-1, -1, self.head_dim)
        k_sparse = kv.gather(1, flat_idx).view(bsz, seqlen, k, self.head_dim)
        v_sparse = k_sparse[..., :self.v_head_dim]

        # 单次 matmul：q[b,s,h,d] × k[b,s,k,d]ᵀ → [b,s,h,k]
        scores = torch.matmul(q, k_sparse.transpose(-2, -1)) * self.softmax_scale
        scores = self._apply_attn_sink(scores)
        scores.masked_fill_(indices.unsqueeze(2) < 0, float("-inf"))
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(q)

        # einsum 聚合，允许编译器融合
        return torch.einsum("bshk,bskd->bshd", scores, v_sparse)

    def get_kv_cache_size(self) -> int:
        """获取 KV 缓存大小（以元素计）。"""
        total = self.kv_cache.numel()
        if self.kv_compress_cache is not None:
            total += self.kv_compress_cache.numel()
        return total
