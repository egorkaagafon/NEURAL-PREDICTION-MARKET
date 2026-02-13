"""
Phase 1 — Sanity Check: NPM vs Ensemble vs MC-Dropout vs MoE on CIFAR-10.

Metrics: accuracy, NLL, ECE, Brier, selective risk (AURC).

This script trains all four models and compares them head‑to‑head.
For a quick sanity check, reduce --epochs to 20-50.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_utils import get_cifar10_loaders
from models.vit_npm import NeuralPredictionMarket
from models.baselines import DeepEnsemble, MCDropoutViT, MoEClassifier
from npm_core.capital import CapitalManager
from npm_core.market import MarketAggregator
from evaluate import full_evaluation, selective_risk_curve


def _baseline_metrics(probs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute accuracy, NLL, Brier, ECE, entropy for a baseline."""
    preds = probs.argmax(-1)
    acc = (preds == targets).float().mean().item()
    nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), targets).item()
    one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()
    brier = ((probs - one_hot) ** 2).sum(-1).mean().item()
    # ECE (15 bins)
    confs, preds2 = probs.max(-1)
    corrects = (preds2 == targets).float()
    ece = 0.0
    boundaries = torch.linspace(0, 1, 16)
    for i in range(15):
        mask = (confs > boundaries[i]) & (confs <= boundaries[i + 1])
        if mask.sum() > 0:
            ece += mask.float().mean() * (corrects[mask].mean() - confs[mask].mean()).abs()
    ece = ece.item()
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(-1)
    return {"accuracy": acc, "nll": nll, "brier": brier, "ece": ece, "entropy": entropy}


def train_baseline(model, train_loader, device, epochs=100, lr=3e-4):
    """Generic training loop for CE‑based baselines."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            probs = out["market_probs"]

            log_probs = torch.log(probs.clamp(min=1e-8))
            loss = F.nll_loss(log_probs, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = probs.argmax(-1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)

        scheduler.step()
        if epoch % 10 == 0:
            print(f"  [Baseline] Epoch {epoch:3d}  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.2%}")


def train_npm(cfg, train_loader, device, epochs=100):
    """Train NPM with capital dynamics."""
    from train import build_optimizer, build_scheduler

    mc = cfg["model"]
    mkt = cfg["market"]

    model = NeuralPredictionMarket(
        image_size=mc["image_size"], patch_size=mc["patch_size"],
        embed_dim=mc["embed_dim"], depth=mc["depth"],
        num_heads=mc["num_heads"], num_agents=mc["num_agents"],
        num_classes=mc["num_classes"], dropout=mc["dropout"],
        bet_temperature=mkt["bet_temperature"],
        feature_keep_prob=mkt.get("feature_keep_prob", 0.7),
    ).to(device)

    capital_mgr = CapitalManager(
        num_agents=mc["num_agents"],
        initial_capital=mkt["initial_capital"],
        lr=mkt["capital_lr"],
        decay=mkt.get("capital_decay", 0.9),
        normalize_payoffs=mkt.get("normalize_payoffs", True),
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )
    market = MarketAggregator()
    agent_aux_w = mkt.get("agent_aux_weight", 0.3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            capital = capital_mgr.get_capital()
            out = model(images, capital)

            loss_market = market.compute_market_loss(out["market_probs"], targets)
            loss_div = market.diversity_loss(out["all_probs"])

            # Per-agent auxiliary loss (matches train.py)
            K, B, C = out["all_probs"].shape
            agent_log_probs = torch.log(out["all_probs"].clamp(min=1e-8))
            tgt_expanded = targets.unsqueeze(0).expand(K, B)
            loss_agent_aux = F.nll_loss(
                agent_log_probs.reshape(K * B, C),
                tgt_expanded.reshape(K * B),
            )

            loss = (loss_market
                    + agent_aux_w * loss_agent_aux
                    + mkt["diversity_weight"] * loss_div)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                payoffs = market.agent_payoffs(
                    out["all_probs"].detach(), targets,
                    bets=out["all_bets"].detach(),
                )
                capital_mgr.accumulate(payoffs)

            pred = out["market_probs"].argmax(-1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)

        scheduler.step()

        # Consolidate epoch capital
        capital_mgr.step()

        if mkt["evolution_enabled"] and epoch % mkt["evolution_interval"] == 0:
            capital_mgr.evolutionary_step(
                model.agent_pool.agents,
                kill_fraction=mkt["kill_fraction"],
                mutation_std=mkt["mutation_std"],
            )

        if epoch % 10 == 0:
            print(f"  [NPM] Epoch {epoch:3d}  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.2%}  "
                  f"gini={capital_mgr.summary()['gini']:.3f}")

    return model, capital_mgr


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
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    vit_kwargs = dict(
        image_size=cfg["model"]["image_size"],
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )

    # ── Parameter-matched baseline configs ──
    # Target: ~7.5M params each (same as NPM).
    # MC-Dropout: deeper backbone (depth=9) → 7.21M
    mc_kwargs = dict(vit_kwargs, depth=9)
    # Ensemble(5): 5 smaller ViTs (embed=160, depth=5, heads=8) → 7.84M
    ens_kwargs = dict(vit_kwargs, embed_dim=160, depth=5, num_heads=8)
    # MoE: bigger expert hidden → 6.97M
    moe_hidden = 512

    results = {}

    # ── 1. NPM ──
    print("\n" + "="*60)
    print("Training NPM")
    print("="*60)
    t0 = time.perf_counter()
    npm_model, capital_mgr = train_npm(cfg, train_loader, device, args.epochs)
    npm_train_time = time.perf_counter() - t0

    npm_eval = full_evaluation(npm_model, test_loader, capital_mgr, device)
    npm_sr_epist = selective_risk_curve(
        npm_eval["probs"], npm_eval["targets"],
        npm_eval["uncertainty"]["epistemic_unc"],
    )
    npm_sr_entropy = selective_risk_curve(
        npm_eval["probs"], npm_eval["targets"],
        npm_eval["uncertainty"]["entropy_market"],
    )
    npm_sr_market = selective_risk_curve(
        npm_eval["probs"], npm_eval["targets"],
        npm_eval["uncertainty"]["market_unc"],
    )
    results["npm"] = {
        "accuracy": npm_eval["accuracy"],
        "nll": npm_eval["nll"],
        "brier": npm_eval["brier"],
        "ece": npm_eval["ece"],
        "aurc_epistemic": npm_sr_epist["aurc"],
        "aurc_entropy": npm_sr_entropy["aurc"],
        "aurc_market": npm_sr_market["aurc"],
        "train_time_s": round(npm_train_time, 1),
    }
    print(f"NPM: {results['npm']}")

    # ── 2. Deep Ensemble ──
    print("\n" + "="*60)
    print("Training Deep Ensemble (5 members, param-matched)")
    print("="*60)
    ensemble = DeepEnsemble(num_members=5, **ens_kwargs).to(device)
    ens_params = sum(p.numel() for p in ensemble.parameters())
    print(f"  Ensemble params: {ens_params:,}")

    # Train each member independently (proper Ensemble protocol)
    t0 = time.perf_counter()
    for mi in range(ensemble.num_members):
        member = ensemble.members[mi]
        opt_m = torch.optim.AdamW(member.parameters(), lr=3e-4, weight_decay=0.05)
        sch_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=args.epochs)
        for epoch in range(1, args.epochs + 1):
            member.train()
            total_loss = 0; correct = 0; total = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                logits = member(images)
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs.clamp(min=1e-8))
                loss = F.nll_loss(log_probs, targets)
                opt_m.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                opt_m.step()
                pred = probs.argmax(-1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item() * targets.size(0)
            sch_m.step()
            if epoch % 10 == 0:
                print(f"  [Ens member {mi}] Epoch {epoch:3d}  "
                      f"loss={total_loss/total:.4f}  acc={correct/total:.2%}")
    ens_train_time = time.perf_counter() - t0

    # Fake capital manager for interface compatibility
    class FakeCapital:
        def get_capital(self):
            return torch.ones(5, device=device)
        def summary(self):
            return {"gini": 0.0, "num_bankrupt": 0}

    # Direct evaluation for ensemble
    ensemble.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            out = ensemble(images)
            all_probs.append(out["market_probs"].cpu())
            all_targets.append(targets)
    probs = torch.cat(all_probs)
    tgts = torch.cat(all_targets)
    m = _baseline_metrics(probs, tgts)
    sr = selective_risk_curve(probs, tgts, m["entropy"])
    results["ensemble"] = {
        "accuracy": m["accuracy"], "nll": m["nll"],
        "brier": m["brier"], "ece": m["ece"],
        "aurc_entropy": sr["aurc"],
        "train_time_s": round(ens_train_time, 1),
    }
    print(f"Ensemble: {results['ensemble']}")

    # ── 3. MC-Dropout ──
    print("\n" + "="*60)
    print("Training MC-Dropout (10 samples, param-matched)")
    print("="*60)
    mc_model = MCDropoutViT(mc_samples=10, **mc_kwargs).to(device)
    mc_params = sum(p.numel() for p in mc_model.parameters())
    print(f"  MC-Dropout params: {mc_params:,}")
    t0 = time.perf_counter()
    train_baseline(mc_model, train_loader, device, args.epochs)
    mc_train_time = time.perf_counter() - t0

    mc_model.eval()          # triggers MC multi-sample inference
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            out = mc_model(images)
            all_probs.append(out["market_probs"].cpu())
            all_targets.append(targets)
    probs = torch.cat(all_probs)
    tgts = torch.cat(all_targets)
    m = _baseline_metrics(probs, tgts)
    sr = selective_risk_curve(probs, tgts, m["entropy"])
    results["mc_dropout"] = {
        "accuracy": m["accuracy"], "nll": m["nll"],
        "brier": m["brier"], "ece": m["ece"],
        "aurc_entropy": sr["aurc"],
        "train_time_s": round(mc_train_time, 1),
    }
    print(f"MC-Dropout: {results['mc_dropout']}")

    # ── 4. MoE ──
    print("\n" + "="*60)
    print("Training MoE (16 experts, top-4, param-matched)")
    print("="*60)
    moe = MoEClassifier(
        **vit_kwargs, num_experts=16, top_k=4,
        expert_hidden_dim=moe_hidden,
    ).to(device)
    moe_params = sum(p.numel() for p in moe.parameters())
    print(f"  MoE params: {moe_params:,}")
    t0 = time.perf_counter()
    train_baseline(moe, train_loader, device, args.epochs)
    moe_train_time = time.perf_counter() - t0

    all_probs, all_targets = [], []
    with torch.no_grad():
        moe.eval()
        for images, targets in test_loader:
            images = images.to(device)
            out = moe(images)
            all_probs.append(out["market_probs"].cpu())
            all_targets.append(targets)
    probs = torch.cat(all_probs)
    tgts = torch.cat(all_targets)
    m = _baseline_metrics(probs, tgts)
    sr = selective_risk_curve(probs, tgts, m["entropy"])
    results["moe"] = {
        "accuracy": m["accuracy"], "nll": m["nll"],
        "brier": m["brier"], "ece": m["ece"],
        "aurc_entropy": sr["aurc"],
        "train_time_s": round(moe_train_time, 1),
    }
    print(f"MoE: {results['moe']}")

    # ── Summary ──
    print("\n" + "="*60)
    print("PHASE 1 RESULTS")
    print("="*60)
    for name, r in results.items():
        t = r.get('train_time_s', 0)
        aurc_e = r.get('aurc_entropy', r.get('aurc', 0))
        brier = r.get('brier', 0)
        ece = r.get('ece', 0)
        extra = ""
        if 'aurc_market' in r:
            extra = (f"  aurc_market={r['aurc_market']:.4f}"
                     f"  aurc_epist={r['aurc_epistemic']:.4f}")
        print(f"  {name:15s}: acc={r['accuracy']:.2%}  nll={r['nll']:.4f}  "
              f"brier={brier:.4f}  ece={ece:.4f}  aurc={aurc_e:.4f}{extra}  "
              f"time={t:.0f}s")

    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "phase1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path / 'phase1_results.json'}")


if __name__ == "__main__":
    main()
