"""Stage 2: SFT 数据 Tokenizer

将 converted 数据转换为模型训练格式，生成 input_ids 和 labels。
关键规则：system 和 user 部分 labels = -100，assistant 部分正常 token_ids。
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..config import SFTDownloadConfig
from ..utils.downloader_utils import CheckpointManager, format_duration

logger = logging.getLogger(__name__)

# 最大序列长度
MAX_SEQ_LEN = 1536


class SFTTokenizer:
    """SFT 数据 Tokenizer"""

    def __init__(self, config: Optional[SFTDownloadConfig] = None):
        self.config = config or SFTDownloadConfig()
        self.checkpoint = CheckpointManager(
            self._get_checkpoint_path()
        )
        self.tokenizer = self._load_tokenizer()
        self.pad_token_id = 0  # <|endoftext|>

    def _get_checkpoint_path(self) -> Path:
        """获取 tokenizer 阶段 checkpoint 路径"""
        return Path(self.config.base_data_dir) / "sft_tokenize_checkpoint.json"

    def _load_tokenizer(self):
        """加载 tokenizer"""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("需要安装 transformers")

        tok = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            trust_remote_code=True
        )
        if tok.eos_token_id is None:
            tok.eos_token_id = 2
        return tok

    def tokenize_all(self) -> Dict[str, Any]:
        """Tokenize 所有领域数据"""
        logger.info("开始 Stage 2 Tokenization")
        results = {}

        for domain in ["math", "code", "science"]:
            results[domain] = self._tokenize_domain(domain)

        total = sum(r.get("output_docs", 0) for r in results.values())
        logger.info(f"全部 Tokenize 完成: {total} 条")
        return results

    def _tokenize_domain(self, domain: str) -> Dict[str, Any]:
        """Tokenize 单个领域数据"""
        stage_key = f"{domain}_2"

        if self.checkpoint.is_completed(stage_key):
            logger.info(f"{domain} 数据已 Tokenize，跳过")
            return self.checkpoint.get_stats(stage_key)

        input_file = (
            self.config.get_stage_dir(domain, 1) / "converted.jsonl"
        )
        output_dir = self.config.get_stage_dir(domain, 2)

        if not input_file.exists():
            logger.warning(f"{domain}: 未找到输入文件 {input_file}")
            return {"input_docs": 0, "output_docs": 0}

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "tokenized.jsonl"

        stats = self._process_file(input_file, output_file)

        self.checkpoint.mark_completed(stage_key, stats)
        logger.info(
            f"{domain} Tokenize 完成: {stats['input_docs']} -> "
            f"{stats['output_docs']}"
        )
        return stats

    def _process_file(
        self,
        input_file: Path,
        output_file: Path
    ) -> Dict[str, int]:
        """处理单个文件"""
        input_count = 0
        output_count = 0

        with open(input_file, 'r', encoding='utf-8') as in_f:
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for line in in_f:
                    input_count += 1

                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    # Tokenize 记录
                    tokenized = self._tokenize_record(record)
                    if tokenized is None:
                        continue

                    out_f.write(
                        json.dumps(tokenized, ensure_ascii=False) + '\n'
                    )
                    output_count += 1

                    if input_count % 1000 == 0:
                        logger.info(f"已处理 {input_count} 条")

        return {
            "input_docs": input_count,
            "output_docs": output_count,
        }

    def _tokenize_record(self, record: Dict) -> Optional[Dict]:
        """Tokenize 单条记录，生成 input_ids 和 labels"""
        messages = record.get("messages", [])
        if not messages:
            return None

        # 应用 chat template 获取完整文本
        try:
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"应用 chat template 失败: {e}")
            # 回退：简单拼接
            full_text = self._simple_concat_messages(messages)

        # 编码为 token ids（apply_chat_template 已包含 special tokens）
        input_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False
        )

        # 生成 labels: system/user = -100, assistant = token_ids
        labels = self._generate_labels(messages, input_ids)

        if len(input_ids) != len(labels):
            logger.warning(f"input_ids 和 labels 长度不匹配")
            return None

        # 统一序列长度
        input_ids, attention_mask, labels = self._pad_sequence(
            input_ids, labels
        )

        return {
            "id": record["id"],
            "domain": record["domain"],
            "expert_id": record["expert_id"],
            "sample_type": record.get("sample_type", "task"),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _pad_sequence(
        self,
        input_ids: List[int],
        labels: List[int]
    ) -> tuple:
        """统一序列长度：截断或填充到 MAX_SEQ_LEN"""
        seq_len = len(input_ids)
        eos_id = self.tokenizer.eos_token_id

        if seq_len > MAX_SEQ_LEN:
            # 截断：保留最后一个 token 给 eos
            truncate_pos = MAX_SEQ_LEN - 1

            # 判断截断位置是否在 assistant 中
            is_in_assistant = labels[truncate_pos] != -100

            # 截断并强制以 eos 结尾
            input_ids = input_ids[:truncate_pos] + [eos_id]
            labels = labels[:truncate_pos] + [eos_id if is_in_assistant else -100]

            attention_mask = [1] * MAX_SEQ_LEN
        else:
            # 填充
            pad_len = MAX_SEQ_LEN - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return input_ids, attention_mask, labels

    def _simple_concat_messages(self, messages: List[Dict]) -> str:
        """简单拼接 messages 作为回退方案"""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _generate_labels(
        self,
        messages: List[Dict],
        input_ids: List[int]
    ) -> List[int]:
        """生成 labels。system/user = -100，assistant = token_ids，eos 需要学习"""
        labels = [-100] * len(input_ids)
        eos_id = self.tokenizer.eos_token_id

        # 找到最后一条 assistant 消息的内容
        assistant_content = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
                break

        if not assistant_content:
            return labels

        # 编码 assistant 内容
        assistant_tokens = self.tokenizer.encode(
            assistant_content,
            add_special_tokens=False
        )

        if not assistant_tokens:
            return labels

        # 在 input_ids 中查找 assistant_tokens 的位置
        pos = self._find_subsequence(input_ids, assistant_tokens)

        if pos >= 0:
            # 设置 assistant tokens 的 labels
            end_pos = min(pos + len(assistant_tokens), len(input_ids))
            for i in range(pos, end_pos):
                labels[i] = input_ids[i]

            # eos token 也要学习（如果存在且紧跟 assistant 内容）
            if end_pos < len(input_ids) and input_ids[end_pos] == eos_id:
                labels[end_pos] = eos_id

        return labels

    def _find_subsequence(
        self,
        sequence: List[int],
        subsequence: List[int]
    ) -> int:
        """
        在 sequence 中查找 subsequence 最后出现的位置。
        返回起始索引，未找到返回 -1。
        """
        if not subsequence or len(subsequence) > len(sequence):
            return -1

        # 从后往前找
        for i in range(len(sequence) - len(subsequence), -1, -1):
            if sequence[i:i + len(subsequence)] == subsequence:
                return i

        return -1
