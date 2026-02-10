"""
Training loop for Neural Prediction Market.

Two‑track optimisation:
  Track 1 (backprop) — optimise backbone + agent weights via market loss
  Track 2 (capital)  — update capital via log‑wealth rule (no gradient)

Optional:
  Track 3 (evolution) — replace bankrupt agents periodically
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.vit_npm import NeuralPredictionMarket
from npm_core.market import MarketAggregator
from npm_core.capital import CapitalManager
from npm_core.uncertainty import uncertainty_report
from data_utils import get_loaders, num_classes_for


# ══════════════════════════════════════════════════════════════════════
#  A100 / GPU optimizations
# ══════════════════════════════════════════════════════════════════════

def setup_cuda_optimizations():
    """Enable hardware-level speedups (TF32, cudnn benchmark)."""
    if not torch.cuda.is_available():
        return
    # TF32 — free ~3× speedup on A100 matmuls (19-bit mantissa, fine for training)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN auto-tuner — pick fastest conv algorithm for fixed input size
    torch.backends.cudnn.benchmark = True
    print("CUDA optimizations: TF32=ON, cudnn.benchmark=ON")


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, cfg: dict):
    name = cfg["training"]["optimizer"].lower()
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                               momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
    warmup_steps = cfg["training"]["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════
#  Main training function
# ══════════════════════════════════════════════════════════════════════

def train(cfg: dict):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    setup_cuda_optimizations()

    # ── Data ──
    dataset = cfg["data"].get("dataset", "cifar10")
    # Auto-set num_classes from dataset if not explicitly overridden
    auto_classes = num_classes_for(dataset)
    if cfg["model"]["num_classes"] != auto_classes:
        print(f"Auto-setting num_classes={auto_classes} for dataset={dataset}")
        cfg["model"]["num_classes"] = auto_classes

    train_loader, test_loader = get_loaders(
        dataset=dataset,
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augmentation=cfg["data"]["augmentation"],
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
    )

    # ── Model ──
    mc = cfg["model"]
    model = NeuralPredictionMarket(
        image_size=mc["image_size"],
        patch_size=mc["patch_size"],
        embed_dim=mc["embed_dim"],
        depth=mc["depth"],
        num_heads=mc["num_heads"],
        num_agents=mc["num_agents"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
        bet_temperature=cfg["market"]["bet_temperature"],
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # ── torch.compile (PyTorch 2.0+) — fused kernels, ~20-40% speedup ──
    use_compile = cfg.get("performance", {}).get("compile", False)
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled (reduce-overhead mode)")

    # ── AMP (mixed precision) ──
    use_amp = cfg.get("performance", {}).get("amp", False) and device.type == "cuda"
    amp_dtype = getattr(torch, cfg.get("performance", {}).get("amp_dtype", "bfloat16"))
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"AMP enabled (dtype={amp_dtype})")

    # ── Capital ──
    mkt = cfg["market"]
    capital_mgr = CapitalManager(
        num_agents=mc["num_agents"],
        initial_capital=mkt["initial_capital"],
        lr=mkt["capital_lr"],
        ema=mkt["capital_ema"],
        min_capital=mkt["min_capital"],
        max_capital=mkt["max_capital"],
        device=device,
    )

    # ── Optimizer ──
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # ── Logging ──
    log_dir = Path(cfg["logging"]["log_dir"]) / time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir) if cfg["logging"]["tensorboard"] else None
    log_interval = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]

    market = MarketAggregator()
    diversity_w = mkt["diversity_weight"]
    grad_clip = cfg["training"]["gradient_clip"]

    global_step = 0
    running_liquidity = None  # for normalised uncertainty

    # ── Training Loop ──
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # ── Forward (with optional AMP) ──
            capital = capital_mgr.get_capital()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(images, capital)

                # ── Market loss (backprop) ──
                loss_market = market.compute_market_loss(
                    out["market_probs"], targets,
                    label_smoothing=cfg["training"]["label_smoothing"],
                )

                # Diversity regularizer (anti‑herding)
                loss_div = market.diversity_loss(out["all_probs"])
                loss = loss_market + diversity_w * loss_div

            # ── Backprop (with GradScaler for fp16, no-op for bf16) ──
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── Capital update (no grad) ──
            with torch.no_grad():
                payoffs = market.agent_payoffs(
                    out["all_probs"].detach(), targets,
                )
                capital_mgr.update(payoffs)

            # ── Running stats ──
            with torch.no_grad():
                pred = out["market_probs"].argmax(dim=-1)
                correct = (pred == targets).sum().item()
                epoch_correct += correct
                epoch_total += targets.size(0)
                epoch_loss += loss.item() * targets.size(0)

                # Track liquidity running mean
                from npm_core.uncertainty import market_liquidity
                liq = market_liquidity(out["all_bets"], capital).mean().item()
                if running_liquidity is None:
                    running_liquidity = liq
                else:
                    running_liquidity = 0.99 * running_liquidity + 0.01 * liq

            # ── Logging ──
            global_step += 1
            if global_step % log_interval == 0 and writer:
                cap_info = capital_mgr.summary()
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_market", loss_market.item(), global_step)
                writer.add_scalar("train/loss_diversity", loss_div.item(), global_step)
                writer.add_scalar("train/accuracy",
                                  correct / targets.size(0), global_step)
                writer.add_scalar("train/lr",
                                  scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("capital/mean", cap_info["mean"], global_step)
                writer.add_scalar("capital/std", cap_info["std"], global_step)
                writer.add_scalar("capital/gini", cap_info["gini"], global_step)
                writer.add_scalar("capital/num_bankrupt",
                                  cap_info["num_bankrupt"], global_step)
                writer.add_scalar("market/liquidity", liq, global_step)

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             acc=f"{correct / targets.size(0):.2%}")

        # ── End of epoch ──
        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total
        print(f"Epoch {epoch:3d}  |  loss {train_loss:.4f}  |  "
              f"acc {train_acc:.2%}  |  "
              f"capital gini {capital_mgr.summary()['gini']:.3f}")

        # ── Evolutionary selection ──
        if (mkt["evolution_enabled"]
                and epoch % mkt["evolution_interval"] == 0):
            capital_mgr.evolutionary_step(
                model.agent_pool.agents,
                kill_fraction=mkt["kill_fraction"],
                mutation_std=mkt["mutation_std"],
            )
            print(f"  → Evolution: replaced bottom "
                  f"{mkt['kill_fraction']:.0%} of agents")

        # ── Validation ──
        if epoch % save_interval == 0 or epoch == cfg["training"]["epochs"]:
            val_acc, val_nll = evaluate(model, test_loader, capital_mgr, device,
                                         use_amp=use_amp, amp_dtype=amp_dtype)
            print(f"  → Val acc {val_acc:.2%}  |  NLL {val_nll:.4f}")
            if writer:
                writer.add_scalar("val/accuracy", val_acc, epoch)
                writer.add_scalar("val/nll", val_nll, epoch)

            # Save checkpoint (unwrap torch.compile prefix if present)
            ckpt_path = log_dir / f"ckpt_epoch{epoch:03d}.pt"
            raw_sd = model.state_dict()
            # torch.compile wraps keys with _orig_mod. — strip it for portability
            clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": clean_sd,
                "optimizer_state_dict": optimizer.state_dict(),
                "capital": capital_mgr.state_dict(),
                "config": cfg,
            }, ckpt_path)

    if writer:
        writer.close()

    print(f"\nTraining complete. Checkpoints saved to {log_dir}")
    return model, capital_mgr


# ══════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: NeuralPredictionMarket,
    loader,
    capital_mgr: CapitalManager,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    model.eval()
    correct = 0
    total = 0
    total_nll = 0.0
    market = MarketAggregator()

    capital = capital_mgr.get_capital()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(images, capital)

        pred = out["market_probs"].argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

        log_probs = torch.log(out["market_probs"].clamp(min=1e-8))
        nll = F.nll_loss(log_probs, targets, reduction="sum")
        total_nll += nll.item()

    return correct / total, total_nll / total


# ══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Neural Prediction Market")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size

    train(cfg)


if __name__ == "__main__":
    main()
