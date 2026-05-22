"""Stage 3 SFT 下载工具函数"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Checkpoint 管理器"""

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path
        self.data = self._load()

    def _load(self) -> Dict:
        """加载 checkpoint"""
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"completed_stages": []}

    def save(self) -> None:
        """保存 checkpoint"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def is_completed(self, stage_key: str) -> bool:
        """检查阶段是否已完成"""
        return stage_key in self.data.get("completed_stages", [])

    def mark_completed(self, stage_key: str, stats: Dict[str, Any]) -> None:
        """标记阶段完成"""
        self.data["completed_stages"].append(stage_key)
        self.data[stage_key] = stats
        self.save()

    def get_stats(self, stage_key: str) -> Optional[Dict]:
        """获取阶段统计信息"""
        return self.data.get(stage_key)


class AsyncWriter:
    """异步 JSONL 写入器"""

    def __init__(self, filepath: Path, batch_size: int = 1000):
        self.filepath = filepath
        self.batch_size = batch_size
        self.buffer = []
        self.total_written = 0
        self._lock = asyncio.Lock()

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, record: Dict[str, Any]) -> None:
        """写入单条记录"""
        async with self._lock:
            self.buffer.append(record)
            if len(self.buffer) >= self.batch_size:
                await self._flush()

    async def close(self) -> int:
        """关闭写入器，返回总写入数"""
        async with self._lock:
            if self.buffer:
                await self._flush()
        return self.total_written

    async def _flush(self) -> None:
        """批量写入缓冲区"""
        if not self.buffer:
            return

        lines = []
        for record in self.buffer:
            if HAS_ORJSON:
                lines.append(orjson.dumps(record).decode('utf-8'))
            else:
                lines.append(json.dumps(record, ensure_ascii=False))

        content = '\n'.join(lines) + '\n'

        if HAS_AIOFILES:
            async with aiofiles.open(self.filepath, 'a', encoding='utf-8') as f:
                await f.write(content)
        else:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(content)

        self.total_written += len(self.buffer)
        self.buffer.clear()

        if self.total_written % 10000 == 0:
            logger.info(f"已写入 {self.total_written} 条记录到 {self.filepath.name}")


def generate_sample_id(domain: str, serial: int) -> str:
    """生成样本唯一 ID"""
    return f"{domain}_sft_{serial:08d}"


def create_messages_format(
    problem: str,
    answer: str,
    system_prompt: str
) -> list:
    """创建 messages 格式"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": answer},
    ]


def extract_fields(raw_sample: Dict, field_mapping: Dict) -> Optional[tuple]:
    """从原始样本中提取问题与答案"""
    problem_field = field_mapping.get("problem_field", "problem")
    answer_field = field_mapping.get("answer_field", "solution")

    problem = raw_sample.get(problem_field, "").strip()
    answer = raw_sample.get(answer_field, "").strip()

    if not problem or not answer:
        return None

    return problem, answer


def format_duration(seconds: float) -> str:
    """格式化时长显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
