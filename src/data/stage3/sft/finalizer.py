"""Stage 3: SFT 数据集最终处理

将 tokenized 数据转换为 HuggingFace Dataset 格式（Arrow），
并划分训练集和验证集。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from datasets import Dataset, DatasetDict
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from ..config import SFTDownloadConfig
from ..utils.downloader_utils import CheckpointManager

logger = logging.getLogger(__name__)

VAL_RATIO = 0.05  # 验证集比例


class SFTFinalizer:
    """SFT 数据集最终处理器"""

    def __init__(self, config: Optional[SFTDownloadConfig] = None):
        self.config = config or SFTDownloadConfig()
        self.checkpoint = CheckpointManager(
            self._get_checkpoint_path()
        )
        if not HAS_DATASETS:
            raise RuntimeError("需要安装 datasets 库")

    def _get_checkpoint_path(self) -> Path:
        """获取 checkpoint 路径"""
        return Path(self.config.base_data_dir) / "sft_finalize_checkpoint.json"

    def finalize_all(self) -> Dict[str, Any]:
        """处理所有领域数据"""
        logger.info("开始 Stage 3 Finalization")
        results = {}

        for domain in ["math", "code", "science"]:
            results[domain] = self._finalize_domain(domain)

        logger.info("全部 Finalization 完成")
        return results

    def _finalize_domain(self, domain: str) -> Dict[str, Any]:
        """处理单个领域数据"""
        stage_key = f"{domain}_3"

        if self.checkpoint.is_completed(stage_key):
            logger.info(f"{domain} 数据已 Finalize，跳过")
            return self.checkpoint.get_stats(stage_key)

        input_file = (
            self.config.get_stage_dir(domain, 2) / "tokenized.jsonl"
        )
        output_dir = self.config.get_stage_dir(domain, 3)

        if not input_file.exists():
            logger.warning(f"{domain}: 未找到输入文件 {input_file}")
            return {"train_samples": 0, "val_samples": 0}

        output_dir.mkdir(parents=True, exist_ok=True)

        stats = self._process_and_save(input_file, output_dir)

        self.checkpoint.mark_completed(stage_key, stats)
        logger.info(
            f"{domain} Finalize 完成: train={stats['train_samples']}, "
            f"val={stats['val_samples']}"
        )
        return stats

    def _process_and_save(
        self,
        input_file: Path,
        output_dir: Path
    ) -> Dict[str, int]:
        """读取、处理并保存数据集"""
        records = self._load_records(input_file)

        if not records:
            return {"train_samples": 0, "val_samples": 0}

        dataset = Dataset.from_list(records)
        dataset = self._add_metadata(dataset)

        split_dataset = dataset.train_test_split(
            test_size=VAL_RATIO,
            shuffle=True,
            seed=42
        )

        train_dir = output_dir / "train"
        val_dir = output_dir / "val"

        split_dataset["train"].save_to_disk(train_dir)
        split_dataset["test"].save_to_disk(val_dir)

        self._save_dataset_info(output_dir, split_dataset)

        return {
            "train_samples": len(split_dataset["train"]),
            "val_samples": len(split_dataset["test"]),
        }

    def _load_records(self, input_file: Path) -> List[Dict]:
        """从 jsonl 加载记录"""
        records = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        return records

    def _add_metadata(self, dataset: Dataset) -> Dataset:
        """添加元数据（如有效长度）"""
        def add_length(example):
            mask = example["attention_mask"]
            example["length"] = sum(mask)
            return example

        return dataset.map(add_length)

    def _save_dataset_info(
        self,
        output_dir: Path,
        dataset_dict: DatasetDict
    ) -> None:
        """保存数据集信息"""
        info = {
            "train": {
                "num_samples": len(dataset_dict["train"]),
                "features": list(dataset_dict["train"].features.keys()),
            },
            "validation": {
                "num_samples": len(dataset_dict["test"]),
                "features": list(dataset_dict["test"].features.keys()),
            },
        }

        info_file = output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
