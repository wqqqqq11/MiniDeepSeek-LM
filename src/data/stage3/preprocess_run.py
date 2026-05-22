"""Stage 3 SFT 数据预处理运行入口

Usage:
    python preprocess_run.py --stage 0 --domain all
    python preprocess_run.py --stage 0 --domain math
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.stage3.config import SFTDownloadConfig
from src.data.stage3.sft.downloader import SFTDownloader
from src.data.stage3.sft.converter import SFTConverter


def setup_logging(level: int = logging.INFO) -> None:
    """配置日志输出"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_stage_0_download(domain: str) -> None:
    """运行 Stage 0: 下载原始数据"""
    config = SFTDownloadConfig()
    downloader = SFTDownloader(config)

    if domain == "all":
        downloader.download_all()
    elif domain == "math":
        downloader._download_math()
    elif domain == "code":
        downloader._download_code()
    elif domain == "science":
        downloader._download_science()
    else:
        raise ValueError(f"未知领域: {domain}")


def run_stage_1_convert(domain: str) -> None:
    """运行 Stage 1: 转换数据格式"""
    converter = SFTConverter()

    if domain == "all":
        converter.convert_all()
    else:
        converter._convert_domain(domain)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Stage 3 SFT 数据预处理"
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=[0, 1, 2, 3],
        required=True,
        help="处理阶段: 0=下载, 1=转换, 2=Tokenize, 3=最终数据集"
    )

    parser.add_argument(
        "--domain",
        type=str,
        choices=["all", "math", "code", "science"],
        default="all",
        help="处理领域 (默认: all)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志"
    )

    return parser.parse_args()


def main() -> int:
    """主入口函数"""
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)

    logger = logging.getLogger(__name__)
    logger.info(f"启动 Stage 3 预处理: stage={args.stage}, domain={args.domain}")

    try:
        if args.stage == 0:
            run_stage_0_download(args.domain)
        elif args.stage == 1:
            run_stage_1_convert(args.domain)
        else:
            logger.warning(f"Stage {args.stage} 尚未实现")
            return 1

        logger.info("处理完成")
        return 0

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
