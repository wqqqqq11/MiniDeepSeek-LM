"""
混合专家（MoE）模块

DeepSeek-V4 Pro架构：
    - Hash路由MoE：前n_hash_layers使用token_id % num_experts
    - Learned-gate MoE：后续层使用门控网络选择专家
    - Aux-Loss-Free：动态偏置更新实现负载均衡（无辅助损失）
    - 路由专家 + 共享专家结构
"""

from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelArgs


class MLP(nn.Module):
    """
    SwiGLU MLP（共享专家使用）。

    Args:
        dim: 输入/输出维度。
        inter_dim: 中间层维度。
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: w2(silu(w1(x)) * w3(x))"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """
    单个路由专家。

    Args:
        dim: 输入/输出维度。
        inter_dim: 中间层维度。
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    MoE门控路由（Aux-Loss-Free实现）。

    支持两种模式：
        - Hash路由（前n_hash_layers）：token_id % num_experts
        - Learned-gate路由：sqrt(softplus)打分 + 动态偏置均衡

    Aux-Loss-Free机制：
        - 使用buffer存储expert_bias（不参与梯度）
        - 基于批次频率统计动态更新偏置
        - 频率低→偏置升高，频率高→偏置降低
        - 无需辅助损失函数

    Args:
        layer_id: 层索引。
        args: 模型配置。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.n_routed_experts = args.n_routed_experts

        # Hash路由不需要参数
        if not self.hash:
            # 门控权重（可学习）
            self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
            nn.init.normal_(self.weight, std=0.02)

            # Aux-Loss-Free：动态偏置（buffer，不参与梯度）
            self.register_buffer('expert_bias', torch.zeros(args.n_routed_experts))
            # 频率统计滑动平均（buffer）
            self.register_buffer('freq_ema', torch.zeros(args.n_routed_experts))

            # 动态更新超参
            self.bias_update_speed = args.bias_update_speed
            self.ema_decay = args.ema_decay

    def _hash_route(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hash路由：token_id % num_experts。

        Returns:
            (weights, indices): 均匀权重和hash索引。
        """
        indices = (input_ids % self.n_routed_experts).unsqueeze(-1)
        weights = torch.ones_like(indices, dtype=torch.float32)
        return weights, indices

    def _learned_route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learned-gate路由（Aux-Loss-Free）。

        使用sqrt(softplus)打分 + 动态偏置均衡。
        """
        # 计算logits + 动态偏置（偏置为buffer，不参与梯度）
        x_fp32 = x.float()
        weight_fp32 = self.weight.float()
        scores = F.linear(x_fp32, weight_fp32) + self.expert_bias

        # 打分函数
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()

        # topk选择专家
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)

        # 归一化权重
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale

        # Aux-Loss-Free：动态更新偏置（无梯度）
        with torch.no_grad():
            self._update_bias(indices)

        return weights, indices

    def _update_bias(self, indices: torch.Tensor):
        """
        更新频率统计和动态偏置（Aux-Loss-Free核心）。

        基于当前批次频率，滑动平均更新偏置：
            freq_ema = decay * freq_ema + (1-decay) * current_freq
            bias += speed * (target_freq - freq_ema)
        """
        # 当前批次专家激活频率
        counts = torch.bincount(
            indices.flatten(),
            minlength=self.n_routed_experts
        ).float()
        current_freq = counts / indices.numel()

        # 滑动平均更新频率统计
        self.freq_ema = self.ema_decay * self.freq_ema + (1 - self.ema_decay) * current_freq

        # 目标频率（均匀分布）
        target_freq = 1.0 / self.n_routed_experts

        # 动态更新偏置（无梯度，纯统计更新）
        self.expert_bias += self.bias_update_speed * (target_freq - self.freq_ema)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None, forced_expert_id: Optional[int] = None):
        if forced_expert_id is not None:
            batch_size = x.size(0)
            indices = torch.full((batch_size, self.topk), forced_expert_id, dtype=torch.long, device=x.device)
            weights = torch.ones(batch_size, self.topk, dtype=torch.float32, device=x.device)
            return weights, indices

        if self.hash:
            return self._hash_route(input_ids)
        else:
            return self._learned_route(x)


class MoE(nn.Module):
    """
    混合专家模块（Aux-Loss-Free）。

    结构：路由专家（topk选择）+ 共享专家（所有token经过）
    负载均衡：通过Gate的动态偏置实现，无需辅助损失。

    Args:
        layer_id: 层索引。
        args: 模型配置。
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts

        self.gate = Gate(layer_id, args)

        # 路由专家（单GPU环境：全部创建）
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            for _ in range(args.n_routed_experts)
        ])

        # 共享专家
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None, forced_expert_id: Optional[int] = None) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, self.dim)

        if input_ids is not None:
            flat_ids = input_ids.flatten()
        else:
            flat_ids = None

        weights, indices = self.gate(x, flat_ids, forced_expert_id)

        expert_outs = []
        for i in range(self.n_routed_experts):
            mask = (indices == i).any(dim=-1)
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            expert = self.experts[i]

            expert_pos = (indices[mask] == i).float().argmax(dim=-1)
            w = weights[mask].gather(1, expert_pos.unsqueeze(1))

            out = expert(x[mask]) * w
            expert_outs.append((idx, out))

        if expert_outs:
            all_idx = torch.cat([idx for idx, _ in expert_outs])
            all_out = torch.cat([out for _, out in expert_outs], dim=0)
            base = torch.zeros_like(x)
            y = base.scatter_add(0, all_idx.unsqueeze(-1).expand(-1, x.size(-1)), all_out)
        else:
            y = torch.zeros_like(x)

        z = self.shared_experts(x)

        return (y + z).view(shape)
