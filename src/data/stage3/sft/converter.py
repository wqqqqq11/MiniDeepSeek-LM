"""Stage 1: SFT 数据转换器

将 Stage 0 下载的原始数据转换为标准化格式，进行验证和清洗。
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..config import SFTDownloadConfig
from ..utils.downloader_utils import CheckpointManager, format_duration

logger = logging.getLogger(__name__)


class SFTConverter:
    """SFT 数据转换器"""

    def __init__(self, config: Optional[SFTDownloadConfig] = None):
        self.config = config or SFTDownloadConfig()
        self.checkpoint = CheckpointManager(
            self._get_checkpoint_path()
        )

    def _get_checkpoint_path(self) -> Path:
        """获取转换阶段 checkpoint 路径"""
        return Path(self.config.base_data_dir) / "sft_convert_checkpoint.json"

    def convert_all(self) -> Dict[str, Any]:
        """转换所有领域数据"""
        logger.info("开始 Stage 1 数据转换")
        results = {}

        for domain in ["math", "code", "science"]:
            results[domain] = self._convert_domain(domain)

        total = sum(r.get("output_docs", 0) for r in results.values())
        logger.info(f"全部转换完成: {total} 条")
        return results

    def _convert_domain(self, domain: str) -> Dict[str, Any]:
        """转换单个领域数据"""
        stage_key = f"{domain}_1"

        if self.checkpoint.is_completed(stage_key):
            logger.info(f"{domain} 数据已转换，跳过")
            return self.checkpoint.get_stats(stage_key)

        input_dir = self.config.get_stage_dir(domain, 0)
        output_dir = self.config.get_stage_dir(domain, 1)

        input_files = list(input_dir.glob("*.jsonl"))
        if not input_files:
            logger.warning(f"{domain}: 未找到输入文件")
            return {"input_docs": 0, "output_docs": 0, "filtered": 0}

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "converted.jsonl"

        stats = self._process_files(input_files, output_file)

        self.checkpoint.mark_completed(stage_key, stats)
        logger.info(
            f"{domain} 转换完成: {stats['input_docs']} -> "
            f"{stats['output_docs']} (过滤 {stats['filtered']})"
        )
        return stats

    def _process_files(
        self,
        input_files: List[Path],
        output_file: Path
    ) -> Dict[str, int]:
        """处理输入文件列表"""
        total_input = 0
        total_output = 0
        total_filtered = 0

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for input_file in input_files:
                in_count, out_count, filt_count = self._process_single_file(
                    input_file, out_f
                )
                total_input += in_count
                total_output += out_count
                total_filtered += filt_count

                logger.info(
                    f"处理 {input_file.name}: {in_count} -> "
                    f"{out_count} (过滤 {filt_count})"
                )

        return {
            "input_docs": total_input,
            "output_docs": total_output,
            "filtered": total_filtered,
        }

    def _process_single_file(
        self,
        input_file: Path,
        out_f
    ) -> tuple:
        """处理单个文件，返回 (输入数, 输出数, 过滤数)"""
        input_count = 0
        output_count = 0
        filtered_count = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                input_count += 1

                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    filtered_count += 1
                    continue

                # 验证和转换记录
                converted = self._convert_record(record)
                if converted is None:
                    filtered_count += 1
                    continue

                # 写入输出
                out_f.write(
                    json.dumps(converted, ensure_ascii=False) + '\n'
                )
                output_count += 1

                # 进度日志
                if input_count % 1000 == 0:
                    logger.info(f"已处理 {input_count} 条")

        return input_count, output_count, filtered_count

    def _convert_record(self, record: Dict) -> Optional[Dict]:
        """转换单条记录，验证并标准化格式"""
        # 必填字段检查
        required = ["id", "domain", "expert_id", "messages"]
        for field in required:
            if field not in record:
                return None

        messages = record.get("messages", [])

        # 检查 messages 格式
        if not self._validate_messages(messages):
            return None

        # 标准化输出格式
        return {
            "id": record["id"],
            "domain": record["domain"],
            "expert_id": record["expert_id"],
            "sample_type": record.get("sample_type", "task"),
            "source": record.get("source", "unknown"),
            "messages": messages,
            "token_count": record.get("token_count", 0),
        }

    def _validate_messages(self, messages: List[Dict]) -> bool:
        """验证 messages 格式"""
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        # 检查必需角色
        roles = [m.get("role") for m in messages]
        if "assistant" not in roles:
            return False

        # 检查每个 message 结构
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if not isinstance(msg["content"], str):
                return False
            if not msg["content"].strip():
                return False

        return True
