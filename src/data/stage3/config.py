"""Stage 3 SFT 数据下载配置管理"""

from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path


@dataclass
class SFTDownloadConfig:
    """SFT 数据下载配置"""

    # 基础路径
    base_data_dir: str = "datasets/stage_3_datasets"
    tokenizer_name: str = "jingyaogong/minimind-3"

    # 数学数据集 (1.8M tokens, expert 0)
    math_dataset: str = "AI-MO/NuminaMath-CoT"
    math_target_tokens: int = 1_800_000
    math_expert_id: int = 0

    # 代码数据集 (1.08M + 0.72M = 1.8M tokens, expert 1)
    code_dataset_primary: str = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    code_primary_ratio: float = 0.6
    code_primary_tokens: int = 1_080_000

    code_dataset_secondary: str = "m-a-p/CodeFeedback-Filtered-Instruction"
    code_secondary_ratio: float = 0.4
    code_secondary_tokens: int = 720_000

    code_expert_id: int = 1

    # 科研数据集 (1.8M tokens, expert 2)
    science_dataset: str = "qiaojin/PubMedQA"
    science_target_tokens: int = 1_800_000
    science_expert_id: int = 2

    # 并发配置
    batch_size: int = 1000
    write_queue_size: int = 3
    max_retries: int = 3

    def get_domain_dir(self, domain: str) -> Path:
        """获取领域数据目录"""
        return Path(self.base_data_dir) / domain / "sft"

    def get_stage_dir(self, domain: str, stage: int) -> Path:
        """获取指定阶段的目录"""
        stage_names = {0: "0_download", 1: "1_convert", 2: "2_tokenized", 3: "3_final"}
        return self.get_domain_dir(domain) / stage_names[stage]

    def get_checkpoint_path(self) -> Path:
        """获取 checkpoint 文件路径"""
        return Path(self.base_data_dir) / "sft_download_checkpoint.json"

    def ensure_dirs(self) -> None:
        """确保所有目录存在"""
        for domain in ["math", "code", "science"]:
            for stage in range(4):
                self.get_stage_dir(domain, stage).mkdir(parents=True, exist_ok=True)


# 字段映射配置
FIELD_MAPPINGS = {
    "AI-MO/NuminaMath-CoT": {
        "problem_field": "problem",
        "answer_field": "solution",
    },
    "ise-uiuc/Magicoder-OSS-Instruct-75K": {
        "problem_field": "problem",
        "answer_field": "solution",
    },
    "m-a-p/CodeFeedback-Filtered-Instruction": {
        "problem_field": "query",
        "answer_field": "answer",
    },
    "qiaojin/PubMedQA": {
        "problem_field": "question",
        "answer_field": "long_answer",
    },
}

# 数据集子配置（HuggingFace 数据集有多个 config 时需要）
DATASET_CONFIGS = {
    "qiaojin/PubMedQA": "pqa_labeled",  # 使用标注数据
}

# System Prompts
SYSTEM_PROMPTS = {
    "math": "你是一个数学推理专家。请给出清晰、严谨的解题过程，并在最后给出答案。",
    "code": "你是一个代码生成专家。请根据要求生成正确、高效的代码，并添加必要的注释。",
    "science": "你是一个科研学术专家。请基于证据给出准确的科学回答，不确定时明确说明。",
}
