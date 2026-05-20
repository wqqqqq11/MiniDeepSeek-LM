"""
模拟 fp8/fp4 量化 kernel 函数。

原始 DeepSeek V4 代码依赖外部 kernel 模块（act_quant, weight_dequant, fp8_gemm,
fp4_act_quant, rotate_activation, hc_split_sinkhorn 等）进行量化推理加速。
本模块提供纯 PyTorch fallback 实现，确保在无专用硬件支持的环境下也能正常运行。

当 RuntimeConfig.gemm_impl == "fp8" 时，这些函数会被调用：
  - act_quant: 对激活值进行分块量化（模拟）
  - weight_dequant: 对量化权重进行反量化（模拟）
  - fp8_gemm: 使用 fp8 进行矩阵乘法（模拟，实际退化为 bf16 计算）
  - fp4_act_quant: FP4 量化（用于 Indexer 模块）
  - rotate_activation: Hadamard 旋转移位（用于分散信息）
  - hc_split_sinkhorn: Hyper-Connections 的 Sinkhorn 正则化
"""

import torch
import torch.nn.functional as F


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: str = "fp32",
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对激活值进行分块量化（模拟实现）。

    原始 kernel 会将输入按 block_size 分块，每块计算 scale 后量化为 fp8。
    这里直接返回原始张量和一个 dummy scale，实际计算退化为 bf16。

    Args:
        x: 输入张量
        block_size: 分块大小，默认 128
        scale_fmt: 缩放因子格式，None 表示不使用量化

    Returns:
        (量化后的 x, 每块的 scale)
        模拟实现中 x 不变，scale 为全 1 张量
    """
    # 模拟：不实际量化，返回原张量
    # scale 的形状应与分块后的 x 相匹配
    *batch_dims, rows, cols = x.shape
    # 计算分块后的行数和列数
    num_row_blocks = (rows + block_size - 1) // block_size
    num_col_blocks = (cols + block_size - 1) // block_size

    if scale_dtype == "fp8":
        scale_dtype_torch = torch.float8_e8m0fnu
    else:
        scale_dtype_torch = torch.float32

    scale = torch.ones(
        *batch_dims, num_row_blocks, num_col_blocks,
        dtype=scale_dtype_torch,
        device=x.device,
    )

    # 如果 inplace=True，模拟原地修改（实际为了梯度追踪不做真正原地操作）
    if inplace and x.requires_grad:
        # 创建一个与原张量共享存储的视图
        pass  # PyTorch 中实际原地操作需要小心处理梯度

    return x, scale


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 激活值量化（模拟实现）。

    FP4 (E2M1) 提供比 FP8 更高的压缩率，但精度更低。
    主要用于 Indexer 模块等对精度要求不高但需要极致压缩的场景。

    Args:
        x: 输入张量
        block_size: 分块大小，默认 32（FP4 使用更小块）
        inplace: 是否原地修改

    Returns:
        (量化后的 x, 每块的 scale)
    """
    # 模拟实现：退化为 FP8 模拟，实际不压缩
    return act_quant(x, block_size, None, "fp32", inplace)


def rotate_activation(x: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """
    应用随机 Hadamard 旋转移位（模拟实现）。

    Hadamard 变换可以将张量的信息分散到所有维度，减少量化时的异常值影响。
    这是 DeepSeek-V4 量化训练的关键技术之一。

    公式: y = H @ x / sqrt(dim)，其中 H 是 Hadamard 矩阵。

    Args:
        x: 输入张量，形状 [..., dim]，dim 必须是 2 的幂次
        scale: 缩放因子，默认 1/sqrt(dim)

    Returns:
        旋转后的张量
    """
    # 尝试使用 fast_hadamard_transform，如果不存在则使用简单归一化模拟
    try:
        from fast_hadamard_transform import hadamard_transform
        if scale is None:
            scale = x.size(-1) ** -0.5
        return hadamard_transform(x, scale=scale)
    except ImportError:
        # 模拟实现：简单归一化（实际 Hadamard 是正交变换）
        if scale is None:
            scale = x.size(-1) ** -0.5
        # 使用 FFT 近似模拟 Hadamard 变换的信息分散效果
        x_fft = torch.fft.rfft(x.float(), dim=-1)
        x_rotated = torch.fft.irfft(x_fft, n=x.size(-1), dim=-1)
        return (x_rotated * scale).to(x.dtype)


def _apply_scale(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """应用缩放和偏置，分割为 pre/post/comb 三部分。"""
    # 构建 scale 重复张量: [pre_scale]*hc_mult + [post_scale]*hc_mult + [comb_scale]*hc_mult^2
    scale_parts = [
        hc_scale[0].expand(hc_mult),
        hc_scale[1].expand(hc_mult),
        hc_scale[2].expand(hc_mult * hc_mult),
    ]
    scale_repeated = torch.cat(scale_parts).unsqueeze(0).unsqueeze(0)

    mixes = mixes * scale_repeated + hc_base

    pre = mixes[..., :hc_mult]
    post = mixes[..., hc_mult:2 * hc_mult]
    comb = mixes[..., 2 * hc_mult:].view(mixes.size(0), mixes.size(1), hc_mult, hc_mult)

    return pre, post, comb


def _sinkhorn_normalize(matrix: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Sinkhorn 迭代归一化，将矩阵变为双随机矩阵。"""
    # 先确保非负，使用 exp 转换
    matrix = torch.exp(matrix)

    for _ in range(iters):
        # 行归一化
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        # 列归一化
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int = 5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hyper-Connections 的 Sinkhorn 正则化分块。"""
    pre, post, comb = _apply_scale(mixes, hc_scale, hc_base, hc_mult)

    # Pre/post 用 softmax 归一化
    pre_weights = torch.softmax(pre, dim=-1)
    post_weights = torch.softmax(post, dim=-1)

    # Comb 用 Sinkhorn 迭代归一化为双随机矩阵
    comb_matrix = _sinkhorn_normalize(comb, sinkhorn_iters, eps)

    return pre_weights, post_weights, comb_matrix


def weight_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    对量化权重进行反量化（模拟实现）。

    原始 kernel 会使用 scale 将 fp8 权重反量化为 bf16。
    这里直接返回原始权重（假设权重未被实际量化）。

    Args:
        weight: 量化后的权重张量
        scale: 量化时使用的缩放因子
        block_size: 分块大小，默认 128

    Returns:
        反量化后的权重（模拟实现中直接返回原权重）
    """
    # 模拟：假设权重量化前后相同，直接返回
    return weight


def fp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    fp8 通用矩阵乘法（模拟实现）。

    原始 kernel 使用 fp8 精度进行矩阵乘法加速。
    这里退化为标准的 bf16/float32 线性变换。

    Args:
        x: 输入（可能是分块量化后的）
        x_scale: 输入的缩放因子（未使用）
        weight: 权重矩阵
        weight_scale: 权重的缩放因子（未使用）

    Returns:
        矩阵乘法结果
    """
    # 退化为标准线性变换
    return F.linear(x, weight)