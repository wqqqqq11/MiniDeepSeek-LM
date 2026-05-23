import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from src.training.sft.dataset import SFTArrowDataset, collate_fn
from src.training.logger import TrainingLogger
from src.training.utils.util import set_seed, clip_gradients
from src.training.utils.early_stopping import EarlyStopping


def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_expert_params(model, expert_id):
    for name, param in model.named_parameters():
        if f"experts.{expert_id}." in name:
            param.requires_grad = True


def evaluate(model, val_loader, device, vocab_size, expert_id):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            logits = model(x, forced_expert_id=expert_id)
            
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=-100,
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_expert_checkpoint(model, expert_id, save_path):
    expert_params = {}
    
    for name, param in model.named_parameters():
        if f"experts.{expert_id}." in name:
            expert_params[name] = param.data.cpu()
    
    checkpoint = {
        "expert_id": expert_id,
        "expert_params": expert_params,
    }
    
    torch.save(checkpoint, save_path)


def train_domain_sft(model, train_loader, val_loader, cfg, device, expert_id, output_dir, domain):
    set_seed(cfg["training"]["seed"])
    
    model = model.to(device)
    
    freeze_all_params(model)
    unfreeze_expert_params(model, expert_id)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Expert {expert_id} trainable params: {trainable:,}")
    
    sft_cfg = cfg["sft"]
    lr = float(sft_cfg["lr"])
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    
    es_cfg = sft_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 3),
        enabled=es_cfg.get("enabled", True),
    )
    
    logger = TrainingLogger(
        "logs/stage3",
        log_every_steps=cfg["logging"]["log_every_steps"],
        prefix=f"stage3_{domain}",
    )
    
    vocab_size = cfg["model"]["vocab_size"]
    grad_clip = cfg["training"]["grad_clip"]
    
    steps_per_epoch = len(train_loader)
    
    for epoch in range(sft_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        logger.current_epoch = epoch + 1
        logger.reset_timer()
        step_start_time = time.time()
        
        for step, (x, y, mask) in enumerate(train_loader):
            batch_size = x.size(0)
            seq_len = x.size(1)
            x, y = x.to(device), y.to(device)
            
            logits = model(x, forced_expert_id=expert_id)
            
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=-100,
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = clip_gradients(model, grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            step_time = time.time() - step_start_time
            tokens_per_sec = (batch_size * seq_len) / max(step_time, 1e-6)
            step_start_time = time.time()
            
            metrics = {
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "tokens_per_sec": tokens_per_sec,
            }
            logger.log_train(step, steps_per_epoch, metrics)
        
        avg_loss = epoch_loss / steps_per_epoch
        val_loss = evaluate(model, val_loader, device, vocab_size, expert_id)
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        
        logger.log_eval(epoch + 1, val_loss, val_ppl)
        logger.log_epoch_end(epoch + 1, avg_loss, val_loss)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{domain}_expert_{expert_id}_{timestamp}.pt"
        save_path = Path(output_dir) / save_name
        save_expert_checkpoint(model, expert_id, save_path)
        
        logger.logger.info(f"[SAVE] epoch {epoch+1} saved: {save_path.name} (ppl: {val_ppl:.2f})")
        
        if early_stopping.step(val_ppl):
            logger.logger.info(f"[EarlyStop] triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete! Log: {logger.get_log_file()}")
