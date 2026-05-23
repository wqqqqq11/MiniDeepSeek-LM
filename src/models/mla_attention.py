"""
多头潜在注意力（MLA）模块

实现 DeepSeek-V4 中的 Multi-Head Latent Attention 机制，包含：
1. 低秩联合压缩（LoRA）减少 KV 缓存内存占用
2. CSA (Context-Sparse Attention): 局部滑动窗口处理短距离依赖
3. HCA (Hierarchical Context Attention): 全局压缩稀疏处理长距离依赖

MLA 作为主类集成所有稀疏注意力功能。
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelArgs
from .layers import Linear, ColumnParallelLinear, RowParallelLinear, RMSNorm
from .kernel import weight_dequant, act_quant, rotate_activation, fp4_act_quant
from .rotary_embedding import apply_rotary_emb, precompute_freqs_cis
from .RuntimeConfig import RuntimeConfig


class Compressor(nn.Module):
    """KV 缓存压缩器 - 将连续 token 通过门控池化压缩为低频表示。"""

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = head_dim - args.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.block_size = RuntimeConfig.default().block_size

        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32))
        nn.init.normal_(self.ape, mean=0.0, std=0.02)

        self.wkv = Linear(self.dim, coff * head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * head_dim, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, args.norm_eps)

        self.kv_cache: Optional[Tensor] = None
        self.freqs_cis: Optional[Tensor] = None

        max_bsz = args.max_batch_size
        state_len = coff * compress_ratio
        self.register_buffer("kv_state", torch.zeros(max_bsz, state_len, coff * head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((max_bsz, state_len, coff * head_dim), float("-inf"), dtype=torch.float32), persistent=False)

    def overlap_transform(self, tensor: Tensor, value: float = 0) -> Tensor:
        """重叠窗口变换，使压缩边界更平滑。"""
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: Tensor, start_pos: int) -> Optional[Tensor]:
        """前向压缩，返回压缩后的 KV 或在未达到压缩条件时返回 None。"""
        assert self.kv_cache is not None

        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        dtype = x.dtype

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
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

        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]

            if self.overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)

                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :self.head_dim], self.kv_state[:bsz, ratio:, self.head_dim:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :self.head_dim], self.score_state[:bsz, ratio:, self.head_dim:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)

                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = self.norm(kv.to(dtype))

        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)

        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], dim=-1)
        act_quant(kv[..., :-rd], 64, RuntimeConfig.default().scale_fmt, RuntimeConfig.default().scale_dtype, True)

        if start_pos == 0:
            # HCA缓存可能维度较大，只填充前head_dim列
            cache_slice = self.kv_cache[:bsz, :seqlen // ratio, :kv.size(-1)]
            cache_slice.copy_(kv)
        else:
            cache_slice = self.kv_cache[:bsz, start_pos // ratio, :kv.size(-1)]
            cache_slice.copy_(kv.squeeze(1))

        return kv


class Indexer(nn.Module):
    """压缩 KV 索引器 - 学习选择最相关的压缩 KV 位置。"""

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        world_size = RuntimeConfig.default().world_size
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.q_lora_rank = args.q_lora_rank

        # Indexer使用独立的查询投影，输入维度取决于是否使用低秩
        q_input_dim = self.q_lora_rank if self.q_lora_rank > 0 else self.dim
        self.wq_b = ColumnParallelLinear(q_input_dim, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5

        self.compressor = Compressor(args, compress_ratio, self.head_dim)

        max_cache_len = args.max_seq_len // compress_ratio
        kv_cache_dim = self.head_dim
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, max_cache_len, kv_cache_dim), persistent=False)
        self.freqs_cis = None

    def forward(self, x: Tensor, qr: Tensor, start_pos: int, offset: int) -> Tensor:
        """检索最相关的压缩 KV 位置。"""
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        q = torch.cat([q[..., :-rd], apply_rotary_emb(q[..., -rd:], freqs_cis)], dim=-1)
        q = rotate_activation(q)
        fp4_act_quant(q, RuntimeConfig.default().fp4_block_size, True)

        self.compressor(x, start_pos)

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        world_size = RuntimeConfig.default().world_size
        if world_size > 1:
            import torch.distributed as dist
            dist.all_reduce(index_score)

        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)

        k = min(self.index_topk, end_pos // ratio)
        topk_idxs = index_score.topk(k, dim=-1)[1]

        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset

        return topk_idxs


def get_window_indices(window_size: int, bsz: int, seqlen: int, start_pos: int) -> Tensor:
    """
    生成滑动窗口索引（CSA 局部注意力）。

    每个位置只关注最近的 window_size 个 token，实现局部稠密注意力。
    """
    win = window_size

    if start_pos == 0:
        # 预填充阶段
        base = torch.arange(seqlen, device='cpu').unsqueeze(1)
        indices = (base - win + 1).clamp(0) + torch.arange(min(seqlen, win), device='cpu')
        indices = torch.where(indices > base, -1, indices)
        return indices.unsqueeze(0).expand(bsz, -1, -1).to(torch.int32)
    else:
        # 解码阶段
        if start_pos >= win - 1:
            start_pos %= win
            indices = torch.cat([
                torch.arange(start_pos + 1, win, device='cpu'),
                torch.arange(0, start_pos + 1, device='cpu')
            ], dim=0)
        else:
            indices = F.pad(torch.arange(start_pos + 1, device='cpu'), (0, win - start_pos - 1), value=-1)

        return indices.unsqueeze(0).unsqueeze(0).expand(bsz, seqlen, -1).to(torch.int64)


def get_compress_indices(compress_ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int) -> Tensor:
    """
    生成压缩 KV 的固定间隔采样索引（HCA 全局稀疏）。

    用于 compress_ratio > 0 但不使用 Indexer 的情况（如 128 倍压缩）。
    """
    if start_pos > 0:
        indices = torch.arange(0, (start_pos + 1) // compress_ratio, device='cpu') + offset
    else:
        base = torch.arange(seqlen, device='cpu').unsqueeze(1)
        indices = torch.arange(seqlen // compress_ratio, device='cpu').repeat(seqlen, 1)
        mask = indices >= torch.arange(1, seqlen + 1, device='cpu').unsqueeze(1) // compress_ratio
        indices = torch.where(mask, -1, indices + offset)

    return indices.unsqueeze(0).expand(bsz, -1, -1).to(torch.int64)


class MLA(nn.Module):
    """
    DeepSeek-V4 多头潜在注意力（MLA）

    集成低秩 KV 压缩 + CSA 局部滑动窗口 + HCA 全局稀疏检索。
    每层同时使用三种机制，通过稀疏索引选择大幅降低计算复杂度。

    属性:
        layer_id: 层索引，用于选择 HCA 压缩比率
        window_size: CSA 滑动窗口大小（局部注意力）
        compress_ratio: HCA 压缩比率（全局稀疏，0 表示不使用）
        q/kv_lora_rank: 低秩压缩维度
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        world_size = RuntimeConfig.default().world_size
        self.n_local_heads = args.n_heads // world_size

        # 低秩压缩配置
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # CSA: 滑动窗口大小
        self.window_size = args.window_size

        # HCA: 每层独立的压缩比率
        compress_ratios = args.compress_ratios if args.compress_ratios else tuple([0] * args.n_layers)
        self.compress_ratio = compress_ratios[layer_id] if layer_id < len(compress_ratios) else 0

        # Query 投影（MLA 低秩）
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)

        # KV 投影（MLA 低秩联合压缩）
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_heads * (self.nope_head_dim + self.v_head_dim)
        )

        # 输出投影
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # Attention Sink（稳定长序列注意力的可学习偏差）
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))

        # Softmax 缩放（含 YaRN 长序列调整）
        self.softmax_scale = self.head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= mscale * mscale

        # HCA 模块（压缩 + 检索）
        if self.compress_ratio > 0:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            # compress_ratio == 4 时使用 Indexer 学习检索
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # KV 缓存：窗口部分（低秩）+ 压缩部分（全维度）
        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        # CSA窗口部分使用低秩维度，HCA压缩部分使用head_dim
        kv_cache_dim = max(self.kv_lora_rank + self.rope_head_dim, self.head_dim)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, kv_cache_dim), persistent=False)
        # RoPE 频率缓存
        if self.compress_ratio:
            orig_len, theta = args.original_seq_len, args.compress_rope_theta
        else:
            orig_len, theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, orig_len,
                                         theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # 绑定 HCA 的缓存引用
        if self.compress_ratio > 0:
            self.compressor.kv_cache = self.kv_cache[:, args.window_size:]
            self.compressor.freqs_cis = self.freqs_cis

        if self.compress_ratio > 0 and self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    def forward(self, x: Tensor, start_pos: int) -> Tensor:
        """
        MLA 前向传播 - 整合 CSA + HCA 稀疏注意力。

        Args:
            x: 输入 [batch_size, seq_len, dim]
            start_pos: 序列起始位置

        Returns:
            输出 [batch_size, seq_len, dim]
        """
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        end_pos = start_pos + seqlen

        # 延迟绑定 Indexer 的缓存
        if ratio > 0 and self.indexer is not None and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            self.indexer.compressor.kv_cache = self.kv_cache[:, win:]
            self.indexer.compressor.freqs_cis = self.freqs_cis

        # ---------- Query 投影 ----------
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # Query 归一化（V4 特有）
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

        # 应用 RoPE
        q = torch.cat([q[..., :-self.rope_head_dim], apply_rotary_emb(q[..., -self.rope_head_dim:], freqs_cis)], dim=-1)

        # ---------- KV 投影 ----------
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        act_quant(kv, 64, RuntimeConfig.default().scale_fmt, RuntimeConfig.default().scale_dtype, True)

        # ---------- CSA: 生成局部窗口索引 ----------
        window_idxs = get_window_indices(win, bsz, seqlen, start_pos)

        # ---------- HCA: 生成全局压缩索引 ----------
        if ratio > 0:
            offset = kv.size(1) if start_pos == 0 else win

            if self.indexer is not None:
                # 使用学习的 Indexer 检索最相关位置
                qr = self.q_norm(self.wq_a(x)) if self.q_lora_rank > 0 else x
                compress_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                # 使用固定间隔采样
                compress_idxs = get_compress_indices(ratio, bsz, seqlen, start_pos, offset)

            # 合并 CSA 和 HCA 索引
            topk_idxs = torch.cat([window_idxs, compress_idxs], dim=-1)
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.long()

        # ---------- 更新 KV 缓存 ----------
        # 合并 kv 和 k_pe 为完整 head_dim
        kv_full = torch.cat([kv, k_pe.squeeze(2)], dim=-1)

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv_full
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = \
                    kv_full[:, -win:].split([win - cutoff, cutoff], dim=1)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv_full.squeeze(1)

        # 执行压缩（更新 HCA 缓存）
        if ratio > 0:
            self.compressor(x, start_pos)

        # ---------- 稀疏注意力计算 ----------
        # 收集稀疏 KV（根据 topk_idxs）
        sparse_kv = self._gather_kv(self.kv_cache[:bsz], topk_idxs)

        # 从 wkv_b 展开 K_nope 和 V
        wkv_b_weight = self.wkv_b.weight
        if self.wkv_b.scale is not None:
            wkv_b_weight = weight_dequant(self.wkv_b.weight, self.wkv_b.scale, RuntimeConfig.default().block_size)
        wkv_b_weight = wkv_b_weight.view(self.n_local_heads, -1, self.kv_lora_rank)

        # 使用低秩投影从稀疏 KV 恢复 K_nope 和 V
        # sparse_kv: [b, s, k, head_dim] 其中 head_dim = nope + rope
        k_nope_sparse = torch.einsum("bskc,hdc->bskhd", sparse_kv[..., :self.kv_lora_rank], wkv_b_weight[:, :self.nope_head_dim])
        v_sparse = torch.einsum("bskc,hdc->bskhd", sparse_kv[..., :self.kv_lora_rank], wkv_b_weight[:, -self.v_head_dim:])

        k_pe_sparse = sparse_kv[..., self.kv_lora_rank:].unsqueeze(3).expand(-1, -1, -1, self.n_local_heads, -1)
        k_sparse = torch.cat([k_nope_sparse, k_pe_sparse], dim=-1)  # [b, s, k, n_heads, head_dim]
        k_sparse = k_sparse.transpose(2, 3)  # [b, s, n_heads, k, head_dim]
        v_sparse = v_sparse.transpose(2, 3)  # [b, s, n_heads, k, v_head_dim]

        # 计算稀疏注意力分数
        scores = torch.einsum("bshd,bshkd->bshk", q, k_sparse) * self.softmax_scale

        # 应用 causal mask（屏蔽无效索引）
        mask = topk_idxs < 0
        scores += torch.where(mask.unsqueeze(2), float("-inf"), 0)

        # Softmax + 聚合
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        o = torch.einsum("bshk,bshkd->bshd", scores, v_sparse)

        # ---------- 输出投影 ----------
        x = self.wo(o.flatten(2))
        return x

    def _gather_kv(self, kv_cache: Tensor, indices: Tensor) -> Tensor:
        """根据索引从 KV 缓存中收集稀疏 KV。"""
        bsz, seqlen, k = indices.size()
        head_dim = kv_cache.size(-1)

        safe_indices = indices.clamp(min=0)
        kv = kv_cache.gather(1, safe_indices.view(bsz, -1).unsqueeze(-1).expand(-1, -1, head_dim))
        kv = kv.view(bsz, seqlen, k, head_dim)

        # 将无效位置置零
        mask = (indices < 0).unsqueeze(-1)
        kv = torch.where(mask, torch.zeros_like(kv), kv)

        return kv
