"""
Pretrained Backbone Experiment — NPM vs Baselines on frozen ViT features.

Optimised for T4 (Google Colab):
  1. Feature caching: extract 192-d features ONCE, train heads on cached tensors
     → eliminates 224×224 resizing + backbone forward every epoch (~10× speedup)
  2. AMP (float16): T4 has good fp16 throughput
  3. TensorDataset on GPU: cached features stay in VRAM, zero data-loading overhead

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
from torch.utils.data import TensorDataset, DataLoader
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
)


# ══════════════════════════════════════════════════════════════════════
#  Feature caching
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _extract_features(backbone, loader, device):
    """Extract features from frozen backbone ONCE. Returns (features, targets)."""
    backbone.eval()
    all_feats, all_targets = [], []
    for images, targets in tqdm(loader, desc="Caching features", leave=False):
        images = images.to(device, non_blocking=True)
        feats = backbone(images)          # [B, D]
        all_feats.append(feats)           # keep on GPU
        all_targets.append(targets.to(device, non_blocking=True))
    return torch.cat(all_feats), torch.cat(all_targets)


def _make_cached_loader(features, targets, batch_size, shuffle=True):
    """TensorDataset on GPU → zero CPU↔GPU transfer overhead."""
    ds = TensorDataset(features, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False, drop_last=False)


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
def _eval_val_npm_cached(model, val_feats, val_targets, capital_mgr, market,
                         batch_size=512):
    """Val loss & accuracy on cached features."""
    model.eval()
    capital = capital_mgr.get_capital()
    loader = _make_cached_loader(val_feats, val_targets, batch_size, shuffle=False)
    total_loss = 0; correct = 0; total = 0
    for feats, tgts in loader:
        out = model.forward_on_features(feats, capital)
        loss = market.compute_market_loss(out["market_probs"], tgts)
        total_loss += loss.item() * tgts.size(0)
        correct += (out["market_probs"].argmax(-1) == tgts).sum().item()
        total += tgts.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_val_baseline_cached(model, val_feats, val_targets, batch_size=512):
    """Val loss & accuracy for baseline on cached features."""
    model.eval()
    loader = _make_cached_loader(val_feats, val_targets, batch_size, shuffle=False)
    total_loss = 0; correct = 0; total = 0
    for feats, tgts in loader:
        out = model.forward_on_features(feats)
        probs = out["market_probs"]
        log_probs = torch.log(probs.clamp(min=1e-8))
        total_loss += F.nll_loss(log_probs, tgts, reduction='sum').item()
        correct += (probs.argmax(-1) == tgts).sum().item()
        total += tgts.size(0)
    model.train()
    return total_loss / total, correct / total


@torch.no_grad()
def full_evaluation_npm_cached(model, test_feats, test_targets, capital_mgr,
                               batch_size=512):
    """Run all ID metrics for pretrained NPM on cached features."""
    model.eval()
    capital = capital_mgr.get_capital()
    loader = _make_cached_loader(test_feats, test_targets, batch_size, shuffle=False)

    all_probs, all_targets_list = [], []
    all_unc = {
        "liquidity": [], "epistemic_unc": [], "herding": [],
        "gini": [], "entropy_market": [], "market_unc": [],
        "market_unc_sum": [], "market_unc_max": [], "market_unc_temp": [],
        "pred_variance": [], "mutual_info": [],
    }

    for feats, tgts in tqdm(loader, desc="Evaluating NPM", leave=False):
        out = model.forward_on_features(feats, capital)
        all_probs.append(out["market_probs"].cpu())
        all_targets_list.append(tgts.cpu())

        unc = uncertainty_report(out["all_probs"], out["all_bets"], capital)
        for k in all_unc:
            all_unc[k].append(unc[k].cpu())

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets_list)
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
        "brier": brier if isinstance(brier, float) else brier.item(),
        "ece": ece if isinstance(ece, float) else ece.item(),
        "probs": probs,
        "targets": targets,
        "uncertainty": all_unc,
    }


@torch.no_grad()
def full_evaluation_baseline_cached(model, test_feats, test_targets,
                                    batch_size=512):
    """Run ID metrics for a baseline on cached features."""
    model.eval()
    loader = _make_cached_loader(test_feats, test_targets, batch_size, shuffle=False)

    all_probs, all_targets_list = [], []
    for feats, tgts in loader:
        out = model.forward_on_features(feats)
        all_probs.append(out["market_probs"].cpu())
        all_targets_list.append(tgts.cpu())

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets_list)
    return _baseline_metrics(probs, targets), probs, targets


# ══════════════════════════════════════════════════════════════════════
#  Training loops (on cached features — no backbone forward)
# ══════════════════════════════════════════════════════════════════════

def train_npm_cached(cfg, train_feats, train_targets, device, epochs,
                     val_feats=None, val_targets=None):
    """Train NPM heads on cached features with AMP, early stopping, label smoothing."""
    mkt = cfg["market"]
    mc = cfg["model"]
    backbone_name = cfg["backbone"]["name"]
    batch_size = cfg["data"]["batch_size"]
    label_smoothing = mc.get("label_smoothing", 0.0)
    patience = cfg["training"].get("early_stopping_patience", 0)

    # Create model (backbone loaded but never used during training)
    model = PretrainedNPM(
        backbone_name=backbone_name,
        num_agents=mc["num_agents"],
        num_classes=mc["num_classes"],
        dropout=mc["dropout"],
        bet_temperature=mkt["bet_temperature"],
        feature_keep_prob=mkt.get("feature_keep_prob", 0.5),
        freeze_backbone=True,
    ).to(device)

    trainable = model.count_parameters(trainable_only=True)
    total = model.count_parameters(trainable_only=False)
    print(f"  Backbone: {backbone_name} ({total - trainable:,} frozen)")
    print(f"  NPM heads: {trainable:,} trainable")

    capital_mgr = CapitalManager(
        num_agents=mc["num_agents"],
        initial_capital=mkt["initial_capital"],
        lr=mkt["capital_lr"],
        decay=mkt.get("capital_decay", 0.9),
        normalize_payoffs=mkt.get("normalize_payoffs", True),
        device=device,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    market = MarketAggregator()
    agent_aux_w = mkt.get("agent_aux_weight", 0.3)
    bet_cal_w = mkt.get("bet_calibration_weight", 0.2)

    # Label smoothing CE for agent auxiliary loss
    ce_smooth = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # AMP scaler for T4
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = _make_cached_loader(train_feats, train_targets, batch_size)

    # Early stopping state
    best_val_loss = float("inf")
    best_state = None
    best_capital_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0; correct = 0; total = 0

        for feats, targets in train_loader:
            capital = capital_mgr.get_capital()

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model.forward_on_features(feats, capital)

                loss_market = market.compute_market_loss(out["market_probs"], targets)
                loss_div = market.diversity_loss(out["all_probs"])

                K, B, C = out["all_probs"].shape
                # Agent auxiliary loss with label smoothing
                agent_logits = torch.log(out["all_probs"].clamp(min=1e-8))
                tgt_expanded = targets.unsqueeze(0).expand(K, B)
                loss_agent_aux = ce_smooth(
                    agent_logits.reshape(K * B, C),
                    tgt_expanded.reshape(K * B),
                )
                loss_bet_cal = market.bet_calibration_loss(
                    out["all_probs"], out["all_bets"], targets,
                )

                loss = (loss_market
                        + agent_aux_w * loss_agent_aux
                        + mkt["diversity_weight"] * loss_div
                        + bet_cal_w * loss_bet_cal)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

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
        if val_feats is not None:
            vl, va = _eval_val_npm_cached(
                model, val_feats, val_targets, capital_mgr, market,
            )
            val_str = f"  val_loss={vl:.4f}  val_acc={va:.2%}"

            # Early stopping
            if patience > 0:
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_capital_state = capital_mgr.get_capital().clone()
                    wait = 0
                else:
                    wait += 1

        if epoch % 5 == 0:
            gini = capital_mgr.summary()['gini']
            print(f"  [NPM] Ep {epoch:3d}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.2%}  gini={gini:.3f}{val_str}")

    # Restore best model if early stopping was used
    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
        capital_mgr._capital.data.copy_(best_capital_state)
        print(f"  [NPM] Restored best model (val_loss={best_val_loss:.4f}, "
              f"waited {wait} epochs)")

    return model, capital_mgr


def train_baseline_cached(model, train_feats, train_targets, device, epochs,
                          val_feats=None, val_targets=None,
                          batch_size=512, name="Baseline",
                          label_smoothing=0.0, patience=0):
    """Generic training for baseline heads on cached features with AMP + early stopping."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    train_loader = _make_cached_loader(train_feats, train_targets, batch_size)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0; correct = 0; total = 0

        for feats, targets in train_loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model.forward_on_features(feats)
                probs = out["market_probs"]
                # Use log-probs with label-smoothing CE
                log_probs = torch.log(probs.clamp(min=1e-8))
                loss = ce(log_probs, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pred = probs.argmax(-1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)

        scheduler.step()

        val_str = ""
        if val_feats is not None:
            vl, va = _eval_val_baseline_cached(model, val_feats, val_targets)
            val_str = f"  val_loss={vl:.4f}  val_acc={va:.2%}"
            if patience > 0:
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1

        if epoch % 5 == 0:
            print(f"  [{name}] Ep {epoch:3d}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.2%}{val_str}")

    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [{name}] Restored best model (val_loss={best_val_loss:.4f}, "
              f"waited {wait} epochs)")


def train_ensemble_cached(heads, train_feats, train_targets, device, epochs,
                          batch_size=512, val_feats=None, val_targets=None,
                          label_smoothing=0.0, patience=0):
    """Train each ensemble head independently on cached features."""
    use_amp = device.type == "cuda"
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for mi, head in enumerate(heads):
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        train_loader = _make_cached_loader(train_feats, train_targets, batch_size)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, epochs + 1):
            head.train()
            total_loss = 0; correct = 0; total = 0
            for feats, targets in train_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = head(feats)
                    loss = ce(logits, targets)
                    probs = F.softmax(logits, dim=-1)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                correct += (probs.argmax(-1) == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item() * targets.size(0)
            sch.step()

            # Per-head val for early stopping
            if val_feats is not None and patience > 0:
                head.eval()
                with torch.no_grad():
                    vl_logits = head(val_feats)
                    vl_loss = F.cross_entropy(vl_logits, val_targets).item()
                if vl_loss < best_val_loss:
                    best_val_loss = vl_loss
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                    wait = 0
                else:
                    wait += 1

            if epoch % 10 == 0:
                print(f"  [Ens head {mi}] Ep {epoch:3d}  "
                      f"loss={total_loss/total:.4f}  acc={correct/total:.2%}")

        if patience > 0 and best_state is not None:
            head.load_state_dict(best_state)
            print(f"  [Ens head {mi}] Restored best (val_loss={best_val_loss:.4f})")


# ══════════════════════════════════════════════════════════════════════
#  OOD wrappers (still need backbone forward for new images)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _npm_ood_cached(model, backbone, id_feats, ood_loader, capital_mgr, device,
                    score_fn="epistemic_unc"):
    """OOD detection using cached ID features + fresh OOD backbone forward."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    import numpy as np

    model.eval()
    capital = capital_mgr.get_capital()

    # ID scores from cached features
    id_scores = []
    for i in range(0, len(id_feats), 512):
        feats = id_feats[i:i+512]
        out = model.forward_on_features(feats, capital)
        unc = uncertainty_report(out["all_probs"], out["all_bets"], capital)
        id_scores.append(unc[score_fn].cpu())
    id_scores = torch.cat(id_scores)

    # OOD scores via backbone forward
    ood_scores = []
    for images, _ in ood_loader:
        images = images.to(device, non_blocking=True)
        feats = backbone(images)
        out = model.forward_on_features(feats, capital)
        unc = uncertainty_report(out["all_probs"], out["all_bets"], capital)
        ood_scores.append(unc[score_fn].cpu())
    ood_scores = torch.cat(ood_scores)

    if score_fn == "herding":
        id_scores = -id_scores
        ood_scores = -ood_scores

    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])
    return {
        "auroc": roc_auc_score(labels, scores),
        "aupr": average_precision_score(labels, scores),
    }


@torch.no_grad()
def _baseline_ood_cached(model, id_feats, ood_loader, backbone, device,
                         score_fns=None):
    """OOD detection for baselines: cached ID features + fresh OOD forward."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    import numpy as np

    if score_fns is None:
        score_fns = ["entropy", "pred_variance", "mutual_info"]

    model.eval()

    def _compute(feats_iter, is_cached=False):
        all_scores = {k: [] for k in score_fns}
        for batch in feats_iter:
            if is_cached:
                feats = batch
            else:
                images, _ = batch
                images = images.to(device, non_blocking=True)
                feats = backbone(images)
            out = model.forward_on_features(feats)
            mp = out["market_probs"]
            ap = out.get("all_probs")

            if "entropy" in score_fns:
                ent = -(mp * mp.clamp(min=1e-8).log()).sum(-1)
                all_scores["entropy"].append(ent.cpu())
            if ap is not None and ap.shape[0] > 1:
                if "pred_variance" in score_fns:
                    var = ((ap - mp.unsqueeze(0))**2).sum(-1).mean(0)
                    all_scores["pred_variance"].append(var.cpu())
                if "mutual_info" in score_fns:
                    total_ent = -(mp * mp.clamp(min=1e-8).log()).sum(-1)
                    member_ent = -(ap * ap.clamp(min=1e-8).log()).sum(-1)
                    mi = (total_ent - member_ent.mean(0)).clamp(min=0)
                    all_scores["mutual_info"].append(mi.cpu())
            else:
                B = mp.shape[0]
                if "pred_variance" in score_fns:
                    all_scores["pred_variance"].append(torch.zeros(B))
                if "mutual_info" in score_fns:
                    all_scores["mutual_info"].append(torch.zeros(B))
        return {k: torch.cat(v) for k, v in all_scores.items() if v}

    # ID: iterate cached features in chunks
    id_chunks = [id_feats[i:i+512] for i in range(0, len(id_feats), 512)]
    id_all = _compute(id_chunks, is_cached=True)
    ood_all = _compute(ood_loader, is_cached=False)

    results = {}
    for sfn in score_fns:
        if sfn not in id_all:
            continue
        id_s = id_all[sfn].numpy()
        ood_s = ood_all[sfn].numpy()
        labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
        scores = np.concatenate([id_s, ood_s])
        results[sfn] = {
            "auroc": roc_auc_score(labels, scores),
            "aupr": average_precision_score(labels, scores),
        }
    return results


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

    # T4 optimisations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    backbone_name = cfg["backbone"]["name"]
    image_size = cfg["data"].get("image_size", 224)
    batch_size = cfg["data"]["batch_size"]

    print(f"{'='*60}")
    print(f"PRETRAINED BACKBONE EXPERIMENT (T4-optimised)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {backbone_name} (frozen)")
    print(f"AMP: {'enabled' if device.type == 'cuda' else 'disabled'}")

    # ── Step 1: Load image data ──
    train_loader, val_loader, test_loader = get_cifar10_loaders_224_with_val(
        root=cfg["data"]["root"],
        batch_size=batch_size,
        num_workers=cfg["data"]["num_workers"],
        image_size=image_size,
    )
    print(f"Data: train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}  image_size={image_size}")

    # ── Step 2: Cache features (one-time backbone forward) ──
    print(f"\nExtracting features with {backbone_name}...")
    t0_cache = time.perf_counter()

    shared_backbone = PretrainedBackbone(backbone_name).to(device)

    train_feats, train_targets = _extract_features(shared_backbone, train_loader, device)
    val_feats, val_targets = _extract_features(shared_backbone, val_loader, device)
    test_feats, test_targets = _extract_features(shared_backbone, test_loader, device)

    cache_time = time.perf_counter() - t0_cache
    feat_mb = train_feats.nelement() * train_feats.element_size() / 1024**2
    print(f"  Cached in {cache_time:.1f}s — "
          f"train: {train_feats.shape}  ({feat_mb:.1f} MB on {train_feats.device})")
    print(f"  val: {val_feats.shape}  test: {test_feats.shape}")

    # OOD loaders (still need backbone forward for these)
    ood_datasets = cfg.get("evaluation", {}).get("ood_datasets", ["cifar100", "svhn"])
    ood_loaders = {}
    for ood_name in ood_datasets:
        ood_loaders[ood_name] = get_ood_loader(
            ood_name, root=cfg["data"]["root"],
            batch_size=batch_size,
            num_workers=cfg["data"]["num_workers"],
            image_size=image_size,
        )

    results = {}
    label_smoothing = cfg["model"].get("label_smoothing", 0.0)
    patience = cfg["training"].get("early_stopping_patience", 0)
    bl_cfg = cfg.get("baselines", {})

    # ══════════════════════════════════════════════════════════════════
    #  1. Pretrained NPM
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Training NPM (cached features)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    npm_model, capital_mgr = train_npm_cached(
        cfg, train_feats, train_targets, device, epochs,
        val_feats=val_feats, val_targets=val_targets,
    )
    npm_train_time = time.perf_counter() - t0
    print(f"  NPM training: {npm_train_time:.1f}s")

    # ID evaluation
    npm_eval = full_evaluation_npm_cached(
        npm_model, test_feats, test_targets, capital_mgr,
    )

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
            res = _npm_ood_cached(
                npm_model, shared_backbone, test_feats, ood_loader,
                capital_mgr, device, score_fn=sfn,
            )
            npm_ood[ood_name][sfn] = {
                "auroc": round(res["auroc"], 4),
                "aupr": round(res["aupr"], 4),
            }
    npm_results["ood"] = npm_ood
    results["npm"] = npm_results
    print(f"\nNPM: acc={npm_results['accuracy']:.2%}  nll={npm_results['nll']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  2. Deep Ensemble
    # ══════════════════════════════════════════════════════════════════
    if "ensemble" not in args.skip:
        print(f"\n{'='*60}")
        print("Training Deep Ensemble (5 heads, cached features)")
        print(f"{'='*60}")
        ens_cfg = bl_cfg.get("ensemble", {})
        ensemble = PretrainedEnsemble(
            shared_backbone,
            num_members=ens_cfg.get("num_members", 5),
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"].get("dropout", 0.1),
            hidden_dim=ens_cfg.get("hidden_dim", 0),
            num_layers=ens_cfg.get("num_layers", 1),
        ).to(device)
        trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,}")

        t0 = time.perf_counter()
        train_ensemble_cached(
            ensemble.heads, train_feats, train_targets, device, epochs,
            batch_size=batch_size,
            val_feats=val_feats, val_targets=val_targets,
            label_smoothing=label_smoothing, patience=patience,
        )
        ens_time = time.perf_counter() - t0
        print(f"  Ensemble training: {ens_time:.1f}s")

        m, probs, tgts = full_evaluation_baseline_cached(
            ensemble, test_feats, test_targets,
        )
        sr = selective_risk_curve(probs, tgts, m["entropy"])
        results["ensemble"] = {
            "accuracy": m["accuracy"], "nll": m["nll"],
            "brier": m["brier"], "ece": m["ece"],
            "aurc_entropy": sr["aurc"],
            "train_time_s": round(ens_time, 1),
        }
        ens_ood = {}
        for ood_name, ood_loader in ood_loaders.items():
            ens_ood[ood_name] = _baseline_ood_cached(
                ensemble, test_feats, ood_loader, shared_backbone, device,
            )
        results["ensemble"]["ood"] = ens_ood
        print(f"  Ensemble: acc={m['accuracy']:.2%}  nll={m['nll']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  3. MC-Dropout
    # ══════════════════════════════════════════════════════════════════
    if "mc_dropout" not in args.skip:
        print(f"\n{'='*60}")
        print("Training MC-Dropout (10 samples, cached features)")
        print(f"{'='*60}")
        mc_cfg = bl_cfg.get("mc_dropout", {})
        mc_model = PretrainedMCDropout(
            shared_backbone,
            mc_samples=mc_cfg.get("mc_samples", 10),
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"].get("dropout", 0.1),
            hidden_dim=mc_cfg.get("hidden_dim", 0),
            num_layers=mc_cfg.get("num_layers", 1),
        ).to(device)
        trainable = sum(p.numel() for p in mc_model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,}")

        t0 = time.perf_counter()
        train_baseline_cached(
            mc_model, train_feats, train_targets, device, epochs,
            val_feats=val_feats, val_targets=val_targets,
            batch_size=batch_size, name="MC-Dropout",
            label_smoothing=label_smoothing, patience=patience,
        )
        mc_time = time.perf_counter() - t0
        print(f"  MC-Dropout training: {mc_time:.1f}s")

        m, probs, tgts = full_evaluation_baseline_cached(
            mc_model, test_feats, test_targets,
        )
        sr = selective_risk_curve(probs, tgts, m["entropy"])
        results["mc_dropout"] = {
            "accuracy": m["accuracy"], "nll": m["nll"],
            "brier": m["brier"], "ece": m["ece"],
            "aurc_entropy": sr["aurc"],
            "train_time_s": round(mc_time, 1),
        }
        mc_ood = {}
        for ood_name, ood_loader in ood_loaders.items():
            mc_ood[ood_name] = _baseline_ood_cached(
                mc_model, test_feats, ood_loader, shared_backbone, device,
            )
        results["mc_dropout"]["ood"] = mc_ood
        print(f"  MC-Dropout: acc={m['accuracy']:.2%}  nll={m['nll']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  4. MoE
    # ══════════════════════════════════════════════════════════════════
    if "moe" not in args.skip:
        print(f"\n{'='*60}")
        print("Training MoE (16 experts, top-4, cached features)")
        print(f"{'='*60}")
        moe_cfg = bl_cfg.get("moe", {})
        moe = PretrainedMoE(
            shared_backbone,
            num_experts=moe_cfg.get("num_experts", 16),
            top_k=moe_cfg.get("top_k", 4),
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"].get("dropout", 0.1),
            expert_hidden_dim=moe_cfg.get("expert_hidden_dim", 0),
        ).to(device)
        trainable = sum(p.numel() for p in moe.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,}")

        t0 = time.perf_counter()
        train_baseline_cached(
            moe, train_feats, train_targets, device, epochs,
            val_feats=val_feats, val_targets=val_targets,
            batch_size=batch_size, name="MoE",
            label_smoothing=label_smoothing, patience=patience,
        )
        moe_time = time.perf_counter() - t0
        print(f"  MoE training: {moe_time:.1f}s")

        m, probs, tgts = full_evaluation_baseline_cached(
            moe, test_feats, test_targets,
        )
        sr = selective_risk_curve(probs, tgts, m["entropy"])
        results["moe"] = {
            "accuracy": m["accuracy"], "nll": m["nll"],
            "brier": m["brier"], "ece": m["ece"],
            "aurc_entropy": sr["aurc"],
            "train_time_s": round(moe_time, 1),
        }
        moe_ood = {}
        for ood_name, ood_loader in ood_loaders.items():
            moe_ood[ood_name] = _baseline_ood_cached(
                moe, test_feats, ood_loader, shared_backbone, device,
                score_fns=["entropy"],
            )
        results["moe"]["ood"] = moe_ood
        print(f"  MoE: acc={m['accuracy']:.2%}  nll={m['nll']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PRETRAINED RESULTS — ID Metrics (backbone={backbone_name})")
    print(f"{'='*60}")
    for name, r in results.items():
        t = r.get('train_time_s', 0)
        aurc_e = r.get('aurc_entropy', 0)
        extra = ""
        if 'aurc_market' in r:
            extra = (f"  aurc_mkt={r['aurc_market']:.4f}"
                     f"  aurc_epi={r['aurc_epistemic']:.4f}"
                     f"  aurc_pv={r['aurc_pred_var']:.4f}"
                     f"  aurc_mi={r['aurc_mutual_info']:.4f}")
        print(f"  {name:15s}: acc={r['accuracy']:.2%}  nll={r['nll']:.4f}  "
              f"brier={r.get('brier',0):.4f}  ece={r.get('ece',0):.4f}  "
              f"aurc_ent={aurc_e:.4f}{extra}  time={t:.0f}s")

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

    # Total time
    total_train = sum(r.get("train_time_s", 0) for r in results.values())
    print(f"Total training time: {total_train:.0f}s  "
          f"(+ {cache_time:.0f}s feature caching)")


if __name__ == "__main__":
    main()
