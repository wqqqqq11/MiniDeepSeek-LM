"""
测试入口

用法:
    python test.py --stage 1 --config configs/stage1_pretrain.yaml
"""

import argparse
import logging
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime

from src.models.config import ModelArgs
from src.models.transformer_stage1 import TransformerStage1
from src.training.dataset import TokenDataset
from src.evaluations.stage1_evaluation.perplexity import compute_ppl
from src.tests.stage1_test import test_model_forward, test_inference


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir):
    """
    设置日志记录器。

    Args:
        log_dir: 日志目录路径。

    Returns:
        logger: 配置好的日志记录器。
        log_file: 日志文件路径。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage_1_test_{timestamp}.log"

    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger, str(log_file)


def test_stage1(config, logger):
    cfg = config
    device = torch.device(cfg["hardware"]["device"])

    # 模型配置
    model_args = ModelArgs.stage1()
    model_args.vocab_size = cfg["model"]["vocab_size"]
    model_args.dim = cfg["model"]["dim"]
    model_args.n_layers = cfg["model"]["n_layers"]
    model_args.n_heads = cfg["model"]["n_heads"]
    model_args.max_seq_len = cfg["model"]["max_seq_len"]
    model_args.q_lora_rank = cfg["model"]["q_lora_rank"]
    model_args.kv_lora_rank = cfg["model"]["kv_lora_rank"]
    model_args.head_dim = cfg["model"]["head_dim"]

    logger.info("\n【阶段1测试】")
    logger.info("=" * 50)

    # 模型前向测试（无需检查点）
    logger.info("\n[1] 模型架构测试")
    test_model_forward.run_tests()
    logger.info("  ✓ 模型架构测试完成")

    # 加载检查点
    logger.info("\n[2] 加载模型检查点")
    ckpt_path = Path(cfg["logging"]["checkpoint_dir"]) / "checkpoint_epoch0_20260523_184731.pt"

    if not ckpt_path.exists():
        logger.info(f"  检查点不存在: {ckpt_path}")
        logger.info("  跳过后续测试")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = TransformerStage1(model_args).to(device)
    model.load_state_dict(checkpoint["model"])

    logger.info(f"  ✓ 加载检查点: {ckpt_path}")
    logger.info(f"  最优epoch: {checkpoint['epoch']}")
    logger.info(f"  保存时的val_ppl: {checkpoint['val_ppl']:.2f}")

    # 验证集PPL（验证一致性）
    logger.info("\n[3] 验证集PPL")
    val_dataset = TokenDataset(
        cfg["data"]["val_bin"],
        context_size=cfg["data"]["context_size"],
        shuffle_seed=cfg["data"]["shuffle_seed"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    current_val_ppl = compute_ppl(model, val_loader, device, model_args.vocab_size)
    logger.info(f"  当前计算: {current_val_ppl:.2f}")
    logger.info(f"  保存时:   {checkpoint['val_ppl']:.2f}")

    diff = abs(current_val_ppl - checkpoint["val_ppl"])
    if diff > 1.0:
        logger.info(f"  ⚠ 偏差较大({diff:.2f})，请检查环境一致性")
    else:
        logger.info(f"  ✓ 偏差在合理范围({diff:.2f})")

    # 测试集PPL
    logger.info("\n[4] 测试集PPL")
    test_dataset = TokenDataset(
        cfg["data"]["test_bin"],
        context_size=cfg["data"]["context_size"],
        shuffle_seed=cfg["data"]["shuffle_seed"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    test_ppl = compute_ppl(model, test_loader, device, model_args.vocab_size)
    logger.info(f"  Test PPL: {test_ppl:.2f}")

    # 推理测试
    logger.info("\n[5] 推理功能测试")
    test_inference.run_tests(model, model_args.vocab_size)
    logger.info("  ✓ 推理功能测试完成")

    # 测试报告
    logger.info("\n" + "=" * 50)
    logger.info("【测试完成】")
    logger.info(f"  Val PPL:  {current_val_ppl:.2f}")
    logger.info(f"  Test PPL: {test_ppl:.2f}")
    logger.info(f"  Gap:      {test_ppl - current_val_ppl:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, help="测试阶段")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    if args.stage != 1:
        print(f"阶段 {args.stage} 暂未实现")
        return

    config = load_yaml(args.config)

    # 补充test_bin路径（训练配置中可能只有train/val）
    if "test_bin" not in config["data"]:
        test_bin = config["data"]["train_bin"].replace("train.bin", "test.bin")
        config["data"]["test_bin"] = test_bin

    # 设置日志
    log_dir = config["logging"]["log_dir"]
    logger, log_file = setup_logger(log_dir)

    logger.info(f"测试日志文件: {log_file}")

    test_stage1(config, logger)

    logger.info(f"\n日志已保存: {log_file}")


if __name__ == "__main__":
    main()
