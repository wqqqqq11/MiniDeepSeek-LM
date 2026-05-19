"""
训练入口

用法:
    python mian.py --stage 1 --config configs/stage1_pretrain.yaml
"""

import argparse
import time
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.config import ModelArgs
from src.models.transformer_stage1 import TransformerStage1
from src.training.dataset import TokenDataset
from src.training.lr_scheduler import create_scheduler
from src.training.logger import TrainingLogger, get_gpu_memory
from src.training.utils.util import set_seed, count_parameters, clip_gradients


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_stage1(config):
    cfg = config

    # 设置种子
    set_seed(cfg["training"]["seed"])

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

    # 创建模型
    device = torch.device(cfg["hardware"]["device"])
    model = TransformerStage1(model_args).to(device)

    total_params, _ = count_parameters(model)
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 优化器（使用 AdamW）
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["adamw_lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt_cfg["weight_decay"],
    )

    # 学习率调度
    sched_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # 计算实际训练总步数
    train_dataset = TokenDataset(
        data_cfg["train_bin"],
        context_size=data_cfg["context_size"],
        shuffle_seed=data_cfg["shuffle_seed"],
    )
    steps_per_epoch = len(train_dataset) // sched_cfg["batch_size"]
    total_steps = steps_per_epoch * sched_cfg["num_epochs"]

    # 按比例计算各阶段步数
    warmup_steps = int(total_steps * sched_cfg["warmup_ratio"])
    peak_steps = int(total_steps * sched_cfg["peak_ratio"])
    cosine_steps = total_steps - warmup_steps - peak_steps

    print(f"学习率调度: 总步数={total_steps}, 预热={warmup_steps}, 峰值={peak_steps}, 衰减={cosine_steps}")

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        peak_steps=peak_steps,
        cosine_steps=cosine_steps,
        max_lr=sched_cfg["max_lr"],
        min_lr=sched_cfg["min_lr"],
        mtp_initial_weight=cfg["mtp"]["loss_weight"],
        mtp_final_weight=cfg["mtp"]["loss_weight_decay_end"],
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = len(train_dataset) // cfg["training"]["batch_size"]
    print(f"Steps per epoch: {steps_per_epoch}")

    # 日志
    log_cfg = cfg["logging"]
    logger = TrainingLogger(
        log_cfg["log_dir"],
        log_every_steps=log_cfg["log_every_steps"],
        prefix="stage_1_train",
    )

    # 训练参数
    num_epochs = cfg["training"]["num_epochs"]
    grad_clip = cfg["training"]["grad_clip"]

    # 最优模型跟踪
    best_ppl = float("inf")

    # 训练循环
    step_start_time = time.time()
    
    for epoch in range(num_epochs):
        train_dataset.set_epoch(epoch)
        logger.current_epoch = epoch + 1
        logger.reset_timer()

        model.train()
        epoch_loss = 0.0
        actual_steps = 0

        for step, (x, y) in enumerate(train_loader):
            batch_size = x.size(0)
            x, y = x.to(device), y.to(device)

            # 前向
            logits = model(x)

            # 检查数值
            if torch.isnan(logits).any():
                print(f"\n[NaN detected] step={step}")
                raise RuntimeError("NaN in logits")

            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, model_args.vocab_size),
                y.view(-1),
                ignore_index=-1,
            )

            if torch.isnan(loss):
                raise RuntimeError("NaN loss")

            # 反向
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_norm = clip_gradients(model, grad_clip)

            # 更新
            optimizer.step()
            sched_info = scheduler.step()

            # 记录
            actual_steps += 1
            epoch_loss += loss.item()

            # 计算 tok/s
            step_time = time.time() - step_start_time
            tokens_per_sec = (batch_size * data_cfg["context_size"]) / max(step_time, 1e-6)

            # 日志
            metrics = {
                "loss": loss.item(),
                "lr": sched_info["lr"],
                "grad_norm": grad_norm,
                "tokens_per_sec": tokens_per_sec,
            }
            logger.log_train(step, steps_per_epoch, metrics)

        # 计算 epoch 平均损失
        avg_loss = epoch_loss / actual_steps

        # 验证
        val_loss = evaluate(model, cfg, device, model_args.vocab_size)
        val_ppl = torch.exp(torch.tensor(val_loss)).item()

        logger.log_eval(epoch + 1, val_loss, val_ppl)
        logger.log_epoch_end(epoch + 1, avg_loss, val_loss)

        # 保存检查点（基于 PPL）
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            ckpt_path = Path(log_cfg["checkpoint_dir"]) / "best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, ckpt_path)
            logger.logger.info(f"[SAVE] 新最优模型 saved (ppl: {val_ppl:.2f})")

    print(f"\n训练完成！最优 PPL: {best_ppl:.2f}，日志文件: {logger.get_log_file()}")


def evaluate(model, cfg, device, vocab_size):
    """验证"""
    data_cfg = cfg["data"]

    try:
        val_dataset = TokenDataset(
            data_cfg["val_bin"],
            context_size=data_cfg["context_size"],
            shuffle_seed=data_cfg["shuffle_seed"],
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    except FileNotFoundError:
        return 0.0

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=-1,
            )

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 100:
                break

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, help="训练阶段")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    if args.stage != 1:
        print(f"阶段 {args.stage} 暂未实现")
        return

    config = load_yaml(args.config)
    train_stage1(config)


if __name__ == "__main__":
    main()
