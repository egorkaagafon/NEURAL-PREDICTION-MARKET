"""
Phase 3 — Ablation Study.

Train NPM variants to show each component matters:
  1. Full NPM           (capital + bets + evolution)
  2. No Evolution       (capital + bets, no bankruptcy)
  3. No Capital         (uniform capital=1, bets only)
  4. No Bets            (capital dynamics, but bet=1 always)
  5. No Diversity Loss  (capital + bets + evolution, λ_div=0)

This answers: "Is the market just decoration?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_utils import get_cifar10_loaders
from models.vit_npm import NeuralPredictionMarket
from npm_core.capital import CapitalManager
from npm_core.market import MarketAggregator
from evaluate import full_evaluation, selective_risk_curve


def train_npm_variant(
    cfg: dict,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    *,
    use_evolution: bool = True,
    use_capital: bool = True,
    use_bets: bool = True,
    use_diversity: bool = True,
    variant_name: str = "full",
):
    """Train a single NPM variant and return evaluation results."""
    mc = cfg["model"]
    mkt = cfg["market"]

    model = NeuralPredictionMarket(
        image_size=mc["image_size"], patch_size=mc["patch_size"],
        embed_dim=mc["embed_dim"], depth=mc["depth"],
        num_heads=mc["num_heads"], num_agents=mc["num_agents"],
        num_classes=mc["num_classes"], dropout=mc["dropout"],
        bet_temperature=mkt["bet_temperature"],
    ).to(device)

    capital_mgr = CapitalManager(
        num_agents=mc["num_agents"],
        initial_capital=mkt["initial_capital"],
        lr=mkt["capital_lr"] if use_capital else 0.0,
        ema=mkt["capital_ema"],
        min_capital=mkt["min_capital"],
        max_capital=mkt["max_capital"],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    market = MarketAggregator()
    diversity_w = mkt["diversity_weight"] if use_diversity else 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0; correct = 0; total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            capital = capital_mgr.get_capital() if use_capital else \
                      torch.ones(mc["num_agents"], device=device)

            out = model(images, capital)

            # If no bets — override bet to 1.0
            if not use_bets:
                K, B = out["all_bets"].shape
                fake_bets = torch.ones_like(out["all_bets"])
                market_probs = market.clearing_price(
                    out["all_probs"], fake_bets, capital,
                )
            else:
                market_probs = out["market_probs"]

            loss_market = market.compute_market_loss(market_probs, targets)
            loss_div = market.diversity_loss(out["all_probs"])
            loss = loss_market + diversity_w * loss_div

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if use_capital:
                with torch.no_grad():
                    payoffs = market.agent_payoffs(out["all_probs"].detach(), targets)
                    capital_mgr.update(payoffs)

            pred = market_probs.argmax(-1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)

        scheduler.step()

        if use_evolution and use_capital and epoch % mkt["evolution_interval"] == 0:
            capital_mgr.evolutionary_step(
                model.agent_pool.agents,
                kill_fraction=mkt["kill_fraction"],
                mutation_std=mkt["mutation_std"],
            )

        if epoch % 20 == 0:
            print(f"  [{variant_name}] Epoch {epoch:3d}  "
                  f"loss={total_loss/total:.4f}  acc={correct/total:.2%}")

    # ── Evaluate ──
    eval_res = full_evaluation(model, test_loader, capital_mgr, device)
    sr = selective_risk_curve(
        eval_res["probs"], eval_res["targets"],
        eval_res["uncertainty"]["epistemic_unc"],
    )

    return {
        "accuracy": eval_res["accuracy"],
        "nll": eval_res["nll"],
        "brier": eval_res["brier"],
        "ece": eval_res["ece"],
        "aurc": sr["aurc"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.device:
        cfg["device"] = args.device
    cfg["training"]["epochs"] = args.epochs

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_cifar10_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    variants = {
        "full_npm":       dict(use_evolution=True,  use_capital=True,  use_bets=True,  use_diversity=True),
        "no_evolution":   dict(use_evolution=False, use_capital=True,  use_bets=True,  use_diversity=True),
        "no_capital":     dict(use_evolution=False, use_capital=False, use_bets=True,  use_diversity=True),
        "no_bets":        dict(use_evolution=True,  use_capital=True,  use_bets=False, use_diversity=True),
        "no_diversity":   dict(use_evolution=True,  use_capital=True,  use_bets=True,  use_diversity=False),
    }

    results = {}
    for name, flags in variants.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")

        res = train_npm_variant(
            cfg, train_loader, test_loader, device, args.epochs,
            variant_name=name, **flags,
        )
        results[name] = res
        print(f"  → acc={res['accuracy']:.2%}  nll={res['nll']:.4f}  "
              f"brier={res['brier']:.4f}  ece={res['ece']:.4f}  "
              f"aurc={res['aurc']:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("PHASE 3 ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Variant':20s} {'Acc':>8s} {'NLL':>8s} {'Brier':>8s} {'ECE':>8s} {'AURC':>8s}")
    for name, r in results.items():
        print(f"{name:20s} {r['accuracy']:8.2%} {r['nll']:8.4f} "
              f"{r['brier']:8.4f} {r['ece']:8.4f} {r['aurc']:8.4f}")

    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "phase3_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path / 'phase3_ablation.json'}")


if __name__ == "__main__":
    main()
