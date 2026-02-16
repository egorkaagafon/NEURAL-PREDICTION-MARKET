"""
Pretrained Backbone Experiment — NPM vs Baselines on frozen ViT features.

Same structure as run_phase1.py but:
  - Uses frozen pretrained ViT (DeiT-Tiny / ViT-Small) from timm
  - Only trains agent/classifier heads
  - 224×224 input with ImageNet normalization
  - Much faster training (heads only)

This isolates the NPM contribution from backbone quality.
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

from data_utils import (
    get_cifar10_loaders_224_with_val,
    get_ood_loader,
)
from models.pretrained_npm import (
    PretrainedNPM,
    PretrainedBackbone,
    PretrainedEnsemble,
    PretrainedMCDropout,
    PretrainedMoE,
)
from npm_core.capital import CapitalManager
from npm_core.market import MarketAggregator
from npm_core.uncertainty import uncertainty_report
from evaluate import (
    selective_risk_curve,
    ood_detection_scores,
    baseline_ood_scores,
    ood_detection_volatility,
)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _baseline_metrics(probs, targets):
    """Compute accuracy, NLL, Brier, ECE, entropy for a baseline."""
    preds = probs.argmax(-1)
    acc = (preds == targets).float().mean().item()
    nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), targets).item()
    one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()
    brier = ((probs - one_hot) ** 2).sum(-1).mean().item()
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


@torch.no_grad()
def _eval_val_npm(model, val_loader, capital_mgr, market, device):
    """Compute val loss & accuracy for NPM."""
    model.eval()
    capital = capital_mgr.get_capital()
    total_loss = 0; correct = 0; total = 0
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        out = model(images, capital)
        loss = market.compute_market_loss(out["market_probs"], targets)
        total_loss += loss.item() * targets.size(0)
        correct += (out["market_probs"].argmax(-1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_val_baseline(model, val_loader, device):
    """Compute val loss & accuracy for a baseline model."""
    model.eval()
    total_loss = 0; correct = 0; total = 0
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        probs = out["market_probs"]
        log_probs = torch.log(probs.clamp(min=1e-8))
        total_loss += F.nll_loss(log_probs, targets, reduction='sum').item()
        correct += (probs.argmax(-1) == targets).sum().item()
        total += targets.size(0)
    model.train()
    return total_loss / total, correct / total


@torch.no_grad()
def full_evaluation_npm(model, test_loader, capital_mgr, device):
    """Run all ID metrics for pretrained NPM."""
    model.eval()
    capital = capital_mgr.get_capital()

    all_probs, all_targets = [], []
    all_unc = {
        "liquidity": [], "epistemic_unc": [], "herding": [],
        "gini": [], "entropy_market": [], "market_unc": [],
        "market_unc_sum": [], "market_unc_max": [], "market_unc_temp": [],
        "pred_variance": [], "mutual_info": [],
    }

    for images, targets in tqdm(test_loader, desc="Evaluating NPM", leave=False):
        images = images.to(device)
        out = model(images, capital)
        all_probs.append(out["market_probs"].cpu())
        all_targets.append(targets)

        unc = uncertainty_report(out["all_probs"], out["all_bets"], capital)
        for k in all_unc:
            all_unc[k].append(unc[k].cpu())

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    for k in all_unc:
        all_unc[k] = torch.cat(all_unc[k])

    preds = probs.argmax(-1)
    accuracy = (preds == targets).float().mean().item()
    nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), targets).item()
    one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()
    brier = ((probs - one_hot) ** 2).sum(-1).mean().item()
    confs, preds2 = probs.max(-1)
    corrects = (preds2 == targets).float()
    ece = 0.0
    boundaries = torch.linspace(0, 1, 16)
    for i in range(15):
        mask = (confs > boundaries[i]) & (confs <= boundaries[i + 1])
        if mask.sum() > 0:
            ece += mask.float().mean() * (corrects[mask].mean() - confs[mask].mean()).abs()

    return {
        "accuracy": accuracy,
        "nll": nll,
        "brier": brier.item() if hasattr(brier, 'item') else brier,
        "ece": ece.item() if hasattr(ece, 'item') else ece,
        "probs": probs,
        "targets": targets,
        "uncertainty": all_unc,
    }


# ══════════════════════════════════════════════════════════════════════
#  Training loops
# ══════════════════════════════════════════════════════════════════════

def train_pretrained_npm(cfg, train_loader, device, epochs, val_loader=None):
    """Train NPM heads on frozen pretrained backbone."""
    mkt = cfg["market"]
    mc = cfg["model"]
    backbone_name = cfg["backbone"]["name"]

    model = PretrainedNPM(
        backbone_name=backbone_name,
        num_agents=mc["num_agents"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
        bet_temperature=mkt["bet_temperature"],
        feature_keep_prob=mkt.get("feature_keep_prob", 0.5),
        freeze_backbone=cfg["backbone"].get("freeze", True),
    ).to(device)

    trainable = model.count_parameters(trainable_only=True)
    total = model.count_parameters(trainable_only=False)
    print(f"  Backbone: {backbone_name} ({total - trainable:,} frozen)")
    print(f"  NPM heads: {trainable:,} trainable")
    print(f"  Total: {total:,}")

    capital_mgr = CapitalManager(
        num_agents=mc["num_agents"],
        initial_capital=mkt["initial_capital"],
        lr=mkt["capital_lr"],
        decay=mkt.get("capital_decay", 0.9),
        normalize_payoffs=mkt.get("normalize_payoffs", True),
        device=device,
    )

    # Only optimize trainable params (agent heads)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    market = MarketAggregator()
    agent_aux_w = mkt.get("agent_aux_weight", 0.3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0; correct = 0; total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            capital = capital_mgr.get_capital()
            out = model(images, capital)

            loss_market = market.compute_market_loss(out["market_probs"], targets)
            loss_div = market.diversity_loss(out["all_probs"])

            K, B, C = out["all_probs"].shape
            agent_log_probs = torch.log(out["all_probs"].clamp(min=1e-8))
            tgt_expanded = targets.unsqueeze(0).expand(K, B)
            loss_agent_aux = F.nll_loss(
                agent_log_probs.reshape(K * B, C),
                tgt_expanded.reshape(K * B),
            )
            loss_bet_cal = market.bet_calibration_loss(
                out["all_probs"], out["all_bets"], targets,
            )
            bet_cal_w = mkt.get("bet_calibration_weight", 0.2)

            loss = (loss_market
                    + agent_aux_w * loss_agent_aux
                    + mkt["diversity_weight"] * loss_div
                    + bet_cal_w * loss_bet_cal)

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
        capital_mgr.step()

        if mkt["evolution_enabled"] and epoch % mkt["evolution_interval"] == 0:
            capital_mgr.evolutionary_step(
                model.agent_pool.agents,
                kill_fraction=mkt["kill_fraction"],
                mutation_std=mkt["mutation_std"],
            )

        val_str = ""
        if val_loader is not None and epoch % 5 == 0:
            val_loss, val_acc = _eval_val_npm(model, val_loader, capital_mgr, market, device)
            val_str = f"  val_loss={val_loss:.4f}  val_acc={val_acc:.2%}"

        if epoch % 5 == 0:
            print(f"  [PretrainedNPM] Epoch {epoch:3d}  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.2%}  "
                  f"gini={capital_mgr.summary()['gini']:.3f}{val_str}")

    return model, capital_mgr


def train_pretrained_baseline(model, train_loader, device, epochs,
                              val_loader=None, name="Baseline"):
    """Generic training for pretrained baseline heads."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0; correct = 0; total = 0

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

        val_str = ""
        if val_loader is not None and epoch % 5 == 0:
            val_loss, val_acc = _eval_val_baseline(model, val_loader, device)
            val_str = f"  val_loss={val_loss:.4f}  val_acc={val_acc:.2%}"

        if epoch % 5 == 0:
            print(f"  [{name}] Epoch {epoch:3d}  "
                  f"loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.2%}{val_str}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pretrained backbone experiment")
    parser.add_argument("--config", default="configs/pretrained.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Models to skip: ensemble mc_dropout moe")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    epochs = cfg["training"]["epochs"]

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"PRETRAINED BACKBONE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {cfg['backbone']['name']} (frozen={cfg['backbone'].get('freeze', True)})")

    image_size = cfg["data"].get("image_size", 224)
    train_loader, val_loader, test_loader = get_cifar10_loaders_224_with_val(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        image_size=image_size,
    )
    print(f"Data: train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}  image_size={image_size}")

    # OOD loaders (224px)
    ood_datasets = cfg.get("evaluation", {}).get("ood_datasets", ["cifar100", "svhn"])
    ood_loaders = {}
    for ood_name in ood_datasets:
        ood_loaders[ood_name] = get_ood_loader(
            ood_name, root=cfg["data"]["root"],
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            image_size=image_size,
        )

    results = {}
    backbone_name = cfg["backbone"]["name"]

    # ── 1. Pretrained NPM ──
    print(f"\n{'='*60}")
    print("Training NPM (pretrained backbone)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    npm_model, capital_mgr = train_pretrained_npm(
        cfg, train_loader, device, epochs, val_loader=val_loader,
    )
    npm_train_time = time.perf_counter() - t0

    npm_eval = full_evaluation_npm(npm_model, test_loader, capital_mgr, device)

    # AURC for all NPM metrics
    aurc_keys = [
        ("epistemic_unc", "aurc_epistemic"),
        ("entropy_market", "aurc_entropy"),
        ("market_unc", "aurc_market"),
        ("market_unc_sum", "aurc_market_sum"),
        ("market_unc_max", "aurc_market_max"),
        ("market_unc_temp", "aurc_market_temp"),
        ("pred_variance", "aurc_pred_var"),
        ("mutual_info", "aurc_mutual_info"),
    ]
    npm_results = {
        "accuracy": npm_eval["accuracy"],
        "nll": npm_eval["nll"],
        "brier": npm_eval["brier"],
        "ece": npm_eval["ece"],
        "train_time_s": round(npm_train_time, 1),
    }
    for unc_key, result_key in aurc_keys:
        sr = selective_risk_curve(
            npm_eval["probs"], npm_eval["targets"],
            npm_eval["uncertainty"][unc_key],
        )
        npm_results[result_key] = sr["aurc"]

    # NPM OOD detection
    npm_ood = {}
    npm_score_fns = ["epistemic_unc", "entropy_market", "market_unc",
                     "market_unc_sum", "market_unc_max", "market_unc_temp",
                     "pred_variance", "mutual_info"]
    for ood_name, ood_loader in ood_loaders.items():
        npm_ood[ood_name] = {}
        for sfn in npm_score_fns:
            res = ood_detection_scores(
                npm_model, test_loader, ood_loader,
                capital_mgr, device, score_fn=sfn,
            )
            npm_ood[ood_name][sfn] = {
                "auroc": round(res["auroc"], 4),
                "aupr": round(res["aupr"], 4),
            }
    npm_results["ood"] = npm_ood
    results["npm"] = npm_results
    print(f"NPM: {results['npm']}")

    # ── Shared frozen backbone for baselines ──
    shared_backbone = PretrainedBackbone(backbone_name).to(device)

    # ── 2. Deep Ensemble ──
    if "ensemble" not in args.skip:
        print(f"\n{'='*60}")
        print("Training Deep Ensemble (5 heads, shared backbone)")
        print(f"{'='*60}")
        ensemble = PretrainedEnsemble(
            shared_backbone, num_members=5,
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
        trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        print(f"  Ensemble trainable params: {trainable:,}")

        t0 = time.perf_counter()
        # Train each head independently
        for mi in range(ensemble.num_members):
            head = ensemble.heads[mi]
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            for epoch in range(1, epochs + 1):
                head.train()
                total_loss = 0; correct = 0; total = 0
                for images, targets in train_loader:
                    images, targets = images.to(device), targets.to(device)
                    features = shared_backbone(images)
                    logits = head(features)
                    probs = F.softmax(logits, dim=-1)
                    loss = F.nll_loss(torch.log(probs.clamp(1e-8)), targets)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()
                    correct += (probs.argmax(-1) == targets).sum().item()
                    total += targets.size(0)
                    total_loss += loss.item() * targets.size(0)
                sch.step()
                if epoch % 10 == 0:
                    print(f"  [Ens head {mi}] Epoch {epoch:3d}  "
                          f"loss={total_loss/total:.4f}  acc={correct/total:.2%}")
        ens_train_time = time.perf_counter() - t0

        ensemble.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                out = ensemble(images)
                all_probs.append(out["market_probs"].cpu())
                all_targets.append(targets)
        probs = torch.cat(all_probs); tgts = torch.cat(all_targets)
        m = _baseline_metrics(probs, tgts)
        sr = selective_risk_curve(probs, tgts, m["entropy"])
        results["ensemble"] = {
            "accuracy": m["accuracy"], "nll": m["nll"],
            "brier": m["brier"], "ece": m["ece"],
            "aurc_entropy": sr["aurc"],
            "train_time_s": round(ens_train_time, 1),
        }
        ens_ood = {}
        for ood_name, ood_loader in ood_loaders.items():
            ens_ood[ood_name] = baseline_ood_scores(
                ensemble, test_loader, ood_loader, device,
            )
        results["ensemble"]["ood"] = ens_ood
        print(f"Ensemble: {results['ensemble']}")

    # ── 3. MC-Dropout ──
    if "mc_dropout" not in args.skip:
        print(f"\n{'='*60}")
        print("Training MC-Dropout (10 samples, shared backbone)")
        print(f"{'='*60}")
        mc_model = PretrainedMCDropout(
            shared_backbone, mc_samples=10,
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
        trainable = sum(p.numel() for p in mc_model.parameters() if p.requires_grad)
        print(f"  MC-Dropout trainable params: {trainable:,}")

        t0 = time.perf_counter()
        train_pretrained_baseline(
            mc_model, train_loader, device, epochs,
            val_loader=val_loader, name="MC-Dropout",
        )
        mc_train_time = time.perf_counter() - t0

        mc_model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                out = mc_model(images)
                all_probs.append(out["market_probs"].cpu())
                all_targets.append(targets)
        probs = torch.cat(all_probs); tgts = torch.cat(all_targets)
        m = _baseline_metrics(probs, tgts)
        sr = selective_risk_curve(probs, tgts, m["entropy"])
        results["mc_dropout"] = {
            "accuracy": m["accuracy"], "nll": m["nll"],
            "brier": m["brier"], "ece": m["ece"],
            "aurc_entropy": sr["aurc"],
            "train_time_s": round(mc_train_time, 1),
        }
        mc_ood = {}
        for ood_name, ood_loader in ood_loaders.items():
            mc_ood[ood_name] = baseline_ood_scores(
                mc_model, test_loader, ood_loader, device,
            )
        results["mc_dropout"]["ood"] = mc_ood
        print(f"MC-Dropout: {results['mc_dropout']}")

    # ── 4. MoE ──
    if "moe" not in args.skip:
        print(f"\n{'='*60}")
        print("Training MoE (16 experts, top-4, shared backbone)")
        print(f"{'='*60}")
        moe = PretrainedMoE(
            shared_backbone, num_experts=16, top_k=4,
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
        trainable = sum(p.numel() for p in moe.parameters() if p.requires_grad)
        print(f"  MoE trainable params: {trainable:,}")

        t0 = time.perf_counter()
        train_pretrained_baseline(
            moe, train_loader, device, epochs,
            val_loader=val_loader, name="MoE",
        )
        moe_train_time = time.perf_counter() - t0

        all_probs, all_targets = [], []
        with torch.no_grad():
            moe.eval()
            for images, targets in test_loader:
                images = images.to(device)
                out = moe(images)
                all_probs.append(out["market_probs"].cpu())
                all_targets.append(targets)
        probs = torch.cat(all_probs); tgts = torch.cat(all_targets)
        m = _baseline_metrics(probs, tgts)
        sr = selective_risk_curve(probs, tgts, m["entropy"])
        results["moe"] = {
            "accuracy": m["accuracy"], "nll": m["nll"],
            "brier": m["brier"], "ece": m["ece"],
            "aurc_entropy": sr["aurc"],
            "train_time_s": round(moe_train_time, 1),
        }
        moe_ood = {}
        for ood_name, ood_loader in ood_loaders.items():
            moe_ood[ood_name] = baseline_ood_scores(
                moe, test_loader, ood_loader, device,
                score_fns=["entropy"],
            )
        results["moe"]["ood"] = moe_ood
        print(f"MoE: {results['moe']}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"PRETRAINED RESULTS — ID Metrics (backbone={backbone_name})")
    print(f"{'='*60}")
    for name, r in results.items():
        t = r.get('train_time_s', 0)
        aurc_e = r.get('aurc_entropy', 0)
        extra = ""
        if 'aurc_market' in r:
            extra = (f"  aurc_market={r['aurc_market']:.4f}"
                     f"  aurc_epist={r['aurc_epistemic']:.4f}"
                     f"  aurc_pvar={r['aurc_pred_var']:.4f}"
                     f"  aurc_mi={r['aurc_mutual_info']:.4f}")
        print(f"  {name:15s}: acc={r['accuracy']:.2%}  nll={r['nll']:.4f}  "
              f"brier={r.get('brier',0):.4f}  ece={r.get('ece',0):.4f}  "
              f"aurc={aurc_e:.4f}{extra}  time={t:.0f}s")

    # ── OOD Summary ──
    print(f"\n{'='*60}")
    print(f"PRETRAINED RESULTS — OOD Detection (AUROC)")
    print(f"{'='*60}")
    for ood_name in ood_datasets:
        print(f"\n  --- {ood_name} ---")
        score_keys_seen = set()
        for name, r in results.items():
            if "ood" in r and ood_name in r["ood"]:
                for sk in r["ood"][ood_name]:
                    score_keys_seen.add(sk)
        score_keys = sorted(score_keys_seen)
        header = f"  {'model':15s}"
        for sk in score_keys:
            header += f"  {sk:>15s}"
        print(header)
        for name, r in results.items():
            if "ood" not in r or ood_name not in r["ood"]:
                continue
            row = f"  {name:15s}"
            for sk in score_keys:
                ood_data = r["ood"][ood_name]
                if sk in ood_data:
                    v = ood_data[sk]
                    auroc = v["auroc"] if isinstance(v, dict) else v
                    row += f"  {auroc:>15.4f}"
                else:
                    row += f"  {'—':>15s}"
            print(row)

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pretrained_results.json"

    def _ser(obj):
        import numpy as np
        if isinstance(obj, (torch.Tensor, np.generic)):
            return obj.item() if hasattr(obj, 'item') else float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_ser)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
