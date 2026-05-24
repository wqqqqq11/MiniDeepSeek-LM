"""
文本生成器模块

提供 temperature 采样生成功能，支持从检查点加载模型。
"""

import torch
from typing import Iterator

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.models.config import ModelArgs
from src.models.transformer_stage1 import TransformerStage1
from .sampler import sample


class Generator:
    """
    文本生成器。

    Args:
        model: 模型实例。
        tokenizer: tokenizer实例。
        device: 计算设备。
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: dict):
        """
        从检查点创建生成器。

        Args:
            checkpoint_path: 检查点文件路径。
            config: 配置字典（从yaml加载）。

        Returns:
            Generator: 配置好的生成器实例。
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers库未安装，无法加载tokenizer")

        device = torch.device(config["hardware"]["device"])

        # 加载tokenizer
        tokenizer_name = config["data"].get("tokenizer_name", "jingyaogong/minimind-3")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )

        # 模型配置
        model_args = ModelArgs.stage1()
        model_args.vocab_size = config["model"]["vocab_size"]
        model_args.dim = config["model"]["dim"]
        model_args.n_layers = config["model"]["n_layers"]
        model_args.n_heads = config["model"]["n_heads"]
        model_args.max_seq_len = config["model"]["max_seq_len"]
        model_args.q_lora_rank = config["model"]["q_lora_rank"]
        model_args.kv_lora_rank = config["model"]["kv_lora_rank"]
        model_args.head_dim = config["model"]["head_dim"]

        # 加载模型
        model = TransformerStage1(model_args).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])

        return cls(model, tokenizer, device)

    def encode(self, text: str) -> list:
        """将文本编码为token IDs。"""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list) -> str:
        """将token IDs解码为文本。"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.6,
    ) -> str:
        """
        生成文本。

        采用标准自回归模式：
        1. 预填充阶段：传入完整prompt，生成第一个token
        2. 解码阶段：每次只传入最后一个token，利用KV缓存

        Args:
            prompt: 输入提示文本。
            max_new_tokens: 最大生成token数。
            temperature: 温度参数，默认0.6（参考官方代码）。

        Returns:
            str: 生成的完整文本。
        """
        input_ids = self.encode(prompt)

        # 预填充阶段：传入完整prompt
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits = self.model(x, start_pos=0)
        next_logits = logits[0, -1, :]
        next_token = sample(next_logits, temperature)

        generated = [next_token]

        # 解码阶段：每次只传最后一个token，start_pos从prompt长度开始
        start_pos = len(input_ids)
        for _ in range(max_new_tokens - 1):
            x = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
            logits = self.model(x, start_pos=start_pos)

            next_logits = logits[0, -1, :]
            next_token = sample(next_logits, temperature)

            generated.append(next_token)
            start_pos += 1

        return self.decode(generated)
