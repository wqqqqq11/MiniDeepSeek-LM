"""Stage 3 SFT 数据下载器

从 HuggingFace Hub 流式下载 SFT 数据集，按 token budget 精确控制。
同步版本，避免 asyncio SSL 问题。
"""

import time
import logging
from typing import Dict, Optional, Tuple, Any, Iterator
from pathlib import Path

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from ..config import SFTDownloadConfig, FIELD_MAPPINGS, SYSTEM_PROMPTS, DATASET_CONFIGS
from ..utils.downloader_utils import (
    CheckpointManager,
    generate_sample_id,
    create_messages_format,
    extract_fields,
    format_duration,
)

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """Tokenizer 包装器"""

    def __init__(self, tokenizer_name: str):
        if not HAS_DEPS:
            raise RuntimeError("需要安装 datasets 和 transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = 2

    def count_tokens(self, text: str) -> int:
        """计算文本 token 数量"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class SFTDownloader:
    """SFT 数据下载器"""

    def __init__(self, config: Optional[SFTDownloadConfig] = None):
        self.config = config or SFTDownloadConfig()
        self.config.ensure_dirs()
        self.checkpoint = CheckpointManager(self.config.get_checkpoint_path())
        self.tokenizer = TokenizerWrapper(self.config.tokenizer_name)

    def download_all(self) -> Dict[str, Any]:
        """下载所有领域数据"""
        logger.info("开始下载 Stage 3 SFT 数据集")
        results = {}

        results["math"] = self._download_math()
        results["code"] = self._download_code()
        results["science"] = self._download_science()

        total_docs = sum(r.get("docs", 0) for r in results.values())
        total_tokens = sum(r.get("tokens", 0) for r in results.values())

        logger.info(f"全部下载完成: {total_docs} 条, {total_tokens} tokens")
        return results

    def _download_math(self) -> Dict[str, Any]:
        """下载数学领域数据"""
        stage_key = "math_0"
        if self.checkpoint.is_completed(stage_key):
            logger.info("数学数据已下载，跳过")
            return self.checkpoint.get_stats(stage_key)

        logger.info("开始下载数学数据集...")
        output_dir = self.config.get_stage_dir("math", 0)

        result = self._download_single_dataset(
            dataset_name=self.config.math_dataset,
            target_tokens=self.config.math_target_tokens,
            domain="math",
            expert_id=self.config.math_expert_id,
            output_dir=output_dir,
            field_mapping=FIELD_MAPPINGS[self.config.math_dataset],
            system_prompt=SYSTEM_PROMPTS["math"],
        )

        self.checkpoint.mark_completed(stage_key, result)
        logger.info(f"数学数据下载完成: {result['docs']} 条, {result['tokens']} tokens")
        return result

    def _download_code(self) -> Dict[str, Any]:
        """下载代码领域数据（混合两个数据集）"""
        stage_key = "code_0"
        if self.checkpoint.is_completed(stage_key):
            logger.info("代码数据已下载，跳过")
            return self.checkpoint.get_stats(stage_key)

        logger.info("开始下载代码数据集...")
        output_dir = self.config.get_stage_dir("code", 0)

        # 下载主数据集 (60%)
        primary_result = self._download_single_dataset(
            dataset_name=self.config.code_dataset_primary,
            target_tokens=self.config.code_primary_tokens,
            domain="code",
            expert_id=self.config.code_expert_id,
            output_dir=output_dir,
            field_mapping=FIELD_MAPPINGS[self.config.code_dataset_primary],
            system_prompt=SYSTEM_PROMPTS["code"],
            filename_prefix="magicoder",
        )

        # 下载次数据集 (40%)
        secondary_result = self._download_single_dataset(
            dataset_name=self.config.code_dataset_secondary,
            target_tokens=self.config.code_secondary_tokens,
            domain="code",
            expert_id=self.config.code_expert_id,
            output_dir=output_dir,
            field_mapping=FIELD_MAPPINGS[self.config.code_dataset_secondary],
            system_prompt=SYSTEM_PROMPTS["code"],
            filename_prefix="codefeedback",
            start_serial=primary_result["docs"],
        )

        result = {
            "docs": primary_result["docs"] + secondary_result["docs"],
            "tokens": primary_result["tokens"] + secondary_result["tokens"],
            "primary": primary_result,
            "secondary": secondary_result,
            "elapsed_time": primary_result["elapsed_time"] + secondary_result["elapsed_time"],
        }

        self.checkpoint.mark_completed(stage_key, result)
        logger.info(f"代码数据下载完成: {result['docs']} 条, {result['tokens']} tokens")
        return result

    def _download_science(self) -> Dict[str, Any]:
        """下载科研领域数据"""
        stage_key = "science_0"
        if self.checkpoint.is_completed(stage_key):
            logger.info("科研数据已下载，跳过")
            return self.checkpoint.get_stats(stage_key)

        logger.info("开始下载科研数据集...")
        output_dir = self.config.get_stage_dir("science", 0)

        result = self._download_single_dataset(
            dataset_name=self.config.science_dataset,
            target_tokens=self.config.science_target_tokens,
            domain="science",
            expert_id=self.config.science_expert_id,
            output_dir=output_dir,
            field_mapping=FIELD_MAPPINGS[self.config.science_dataset],
            system_prompt=SYSTEM_PROMPTS["science"],
        )

        self.checkpoint.mark_completed(stage_key, result)
        logger.info(f"科研数据下载完成: {result['docs']} 条, {result['tokens']} tokens")
        return result

    def _download_single_dataset(
        self,
        dataset_name: str,
        target_tokens: int,
        domain: str,
        expert_id: int,
        output_dir: Path,
        field_mapping: Dict,
        system_prompt: str,
        filename_prefix: Optional[str] = None,
        start_serial: int = 0,
    ) -> Dict[str, Any]:
        """下载单个数据集"""
        start_time = time.time()

        # 确定输出文件名
        if filename_prefix:
            output_file = output_dir / f"{filename_prefix}_part_0000.jsonl"
        else:
            output_file = output_dir / "part_0000.jsonl"

        # 检查是否已存在
        if output_file.exists():
            logger.info(f"文件已存在: {output_file}")
            return {"docs": 0, "tokens": 0, "skipped": True, "elapsed_time": 0}

        try:
            # 检查是否需要指定 config
            config_name = DATASET_CONFIGS.get(dataset_name)
            if config_name:
                dataset = load_dataset(
                    dataset_name, config_name, streaming=True, split="train"
                )
            else:
                dataset = load_dataset(dataset_name, streaming=True, split="train")
        except Exception as e:
            logger.error(f"加载数据集失败 {dataset_name}: {e}")
            raise

        result = self._process_stream(
            dataset=dataset,
            output_file=output_file,
            target_tokens=target_tokens,
            domain=domain,
            expert_id=expert_id,
            dataset_name=dataset_name,
            field_mapping=field_mapping,
            system_prompt=system_prompt,
            start_serial=start_serial,
        )

        elapsed = time.time() - start_time
        result["elapsed_time"] = elapsed
        result["dataset"] = dataset_name

        logger.info(f"数据集 {dataset_name}: {result['docs']} 条, "
                   f"{result['tokens']} tokens, 耗时 {format_duration(elapsed)}")

        return result

    def _process_stream(
        self,
        dataset: Iterator,
        output_file: Path,
        target_tokens: int,
        domain: str,
        expert_id: int,
        dataset_name: str,
        field_mapping: Dict,
        system_prompt: str,
        start_serial: int = 0,
    ) -> Dict[str, Any]:
        """处理数据流，按 token budget 采样"""
        current_tokens = 0
        doc_count = 0
        skip_count = 0
        serial = start_serial

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for raw_sample in dataset:
                # 提取字段
                fields = extract_fields(raw_sample, field_mapping)
                if fields is None:
                    skip_count += 1
                    continue

                problem, answer = fields

                # 计算 token 数（包含 system prompt 和格式开销）
                full_text = self._build_full_text(problem, answer, system_prompt)
                n_tokens = self.tokenizer.count_tokens(full_text)

                if n_tokens <= 0:
                    skip_count += 1
                    continue

                # 检查预算（关键逻辑：允许超预算下载完整最后一条）
                remaining = target_tokens - current_tokens
                if remaining <= 0:
                    break

                # 构建样本
                sample = self._build_sample(
                    serial=serial,
                    domain=domain,
                    expert_id=expert_id,
                    dataset_name=dataset_name,
                    problem=problem,
                    answer=answer,
                    system_prompt=system_prompt,
                    token_count=n_tokens,
                )

                # 同步写入
                import json
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

                serial += 1
                doc_count += 1
                current_tokens += n_tokens

                # 进度日志
                if doc_count % 1000 == 0:
                    logger.info(f"已处理 {doc_count} 条, {current_tokens}/{target_tokens} tokens")

        return {
            "docs": doc_count,
            "tokens": current_tokens,
            "skipped": skip_count,
        }

    def _build_full_text(self, problem: str, answer: str, system_prompt: str) -> str:
        """构建完整文本用于 token 计数"""
        messages = create_messages_format(problem, answer, system_prompt)
        # 简单拼接估算，实际格式由 tokenizer chat_template 处理
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts)

    def _build_sample(
        self,
        serial: int,
        domain: str,
        expert_id: int,
        dataset_name: str,
        problem: str,
        answer: str,
        system_prompt: str,
        token_count: int,
    ) -> Dict[str, Any]:
        """构建样本记录"""
        messages = create_messages_format(problem, answer, system_prompt)

        return {
            "id": generate_sample_id(domain, serial),
            "domain": domain,
            "expert_id": expert_id,
            "sample_type": "task",
            "source": dataset_name,
            "messages": messages,
            "token_count": token_count,
        }
