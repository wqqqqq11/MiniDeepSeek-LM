"""
推理入口

用法:
    python inference.py --checkpoint checkpoints/stage1/checkpoint_epoch1_20260524_133702.pt --config configs/stage1_pretrain.yaml --temperature 0.6  --max_new_tokens 100
"""

import argparse
import logging
import sys
import yaml
import time
from pathlib import Path
from datetime import datetime

from src.inferences.stage1_inference import Generator


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger():
    """设置日志记录器。"""
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


def load_prompts(input_file):
    """从文件加载prompt列表。"""
    with open(input_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_results(output_file, results):
    """保存生成结果到文件。"""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (prompt, result) in enumerate(results):
            f.write(f"=== 样例 {i + 1} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Output: {result}\n")
            f.write("\n")


def inference_stage1(config, checkpoint_path, input_file, output_dir, logger, args):
    """阶段1推理主函数。"""
    logger.info("\n【阶段1推理】")
    logger.info("=" * 50)

    # 加载生成器
    logger.info("\n[1] 加载模型...")
    start_time = time.time()
    generator = Generator.from_checkpoint(checkpoint_path, config)
    load_time = time.time() - start_time
    logger.info(f"  ✓ 模型加载完成 ({load_time:.2f}s)")

    # 加载prompts
    logger.info("\n[2] 加载输入文件...")
    prompts = load_prompts(input_file)
    logger.info(f"  ✓ 共 {len(prompts)} 个prompt")

    # 生成结果
    logger.info("\n[3] 开始生成...")
    results = []
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        start = time.time()
        output = generator.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - start

        # 统计生成token数
        gen_tokens = len(generator.encode(output))
        total_tokens += gen_tokens

        results.append((prompt, output))

        logger.info(f"\n  样例 {i + 1}/{len(prompts)} ({elapsed:.2f}s)")
        logger.info(f"    Prompt: {prompt[:50]}...")
        logger.info(f"    Output: {output[:100]}...")

    # 保存结果
    logger.info("\n[4] 保存结果...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.txt"

    save_results(output_file, results)
    logger.info(f"  ✓ 结果已保存: {output_file}")

    # 统计信息
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 50)
    logger.info("【推理完成】")
    logger.info(f"  总样本数: {len(prompts)}")
    logger.info(f"  总时间:   {total_time:.2f}s")
    logger.info(f"  平均每条: {total_time / len(prompts):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="阶段1模型推理")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--input_file", type=str, default="prompts/stage1/prompts.txt", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs/stage1/", help="输出目录")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.6, help="温度参数，默认0.6（参考官方代码）")
    args = parser.parse_args()

    # 设置日志
    logger = setup_logger()

    # 加载配置
    config = load_yaml(args.config)

    # 执行推理
    inference_stage1(
        config,
        args.checkpoint,
        args.input_file,
        args.output_dir,
        logger,
        args
    )


if __name__ == "__main__":
    main()
