"""
Evaluation suite: accuracy, calibration, selective prediction, OOD detection.

Provides all metrics needed for a solid paper:
  1. Standard metrics:  accuracy, NLL, ECE, Brier score
  2. Selective risk:    risk–coverage curves
  3. OOD detection:     AUROC / AUPR using market signals vs. entropy
  4. Market diagnostics: per‑sample liquidity, herding, volatility
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from models.vit_npm import NeuralPredictionMarket
from npm_core.capital import CapitalManager
from npm_core.market import MarketAggregator
from npm_core.uncertainty import (
    uncertainty_report,
    estimate_volatility,
    market_liquidity,
)
from data_utils import get_cifar10_loaders, get_ood_loader


# ══════════════════════════════════════════════════════════════════════
#  1.  In‑Distribution Metrics
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def full_evaluation(
    model: NeuralPredictionMarket,
    test_loader,
    capital_mgr: CapitalManager,
    device: torch.device,
) -> dict:
    """Run all ID metrics on the test set."""

    model.eval()
    capital = capital_mgr.get_capital()

    all_probs = []
    all_targets = []
    all_unc = {
        "liquidity": [], "epistemic_unc": [], "herding": [],
        "gini": [], "entropy_market": [], "market_unc": [],
    }
    all_agent_probs = []
    all_bets = []

    for images, targets in tqdm(test_loader, desc="Evaluating", leave=False):
        images, targets = images.to(device), targets.to(device)
        out = model(images, capital)

        all_probs.append(out["market_probs"].cpu())
        all_targets.append(targets.cpu())
        all_agent_probs.append(out["all_probs"].cpu())
        all_bets.append(out["all_bets"].cpu())

        unc = uncertainty_report(
            out["all_probs"], out["all_bets"], capital,
        )
        for k in all_unc:
            all_unc[k].append(unc[k].cpu())

    probs = torch.cat(all_probs, dim=0)        # [N, C]
    targets = torch.cat(all_targets, dim=0)     # [N]
    for k in all_unc:
        all_unc[k] = torch.cat(all_unc[k], dim=0)

    # Accuracy
    preds = probs.argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()

    # NLL
    log_probs = torch.log(probs.clamp(min=1e-8))
    nll = F.nll_loss(log_probs, targets).item()

    # Brier score
    one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()

    # ECE (15 bins)
    ece = _expected_calibration_error(probs, targets, n_bins=15)

    return {
        "accuracy": accuracy,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "probs": probs,
        "targets": targets,
        "uncertainty": all_unc,
    }


def _expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15,
) -> float:
    """Compute ECE."""
    confs, preds = probs.max(dim=-1)
    corrects = (preds == targets).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = corrects[mask].mean()
            bin_conf = confs[mask].mean()
            ece += mask.float().mean() * (bin_acc - bin_conf).abs()
    return ece.item()


# ══════════════════════════════════════════════════════════════════════
#  2.  Selective Prediction (Risk–Coverage)
# ══════════════════════════════════════════════════════════════════════

def selective_risk_curve(
    probs: torch.Tensor,
    targets: torch.Tensor,
    uncertainty_scores: torch.Tensor,
    n_points: int = 100,
) -> dict:
    """Compute risk (1 - accuracy) at different coverage levels,
    rejecting samples with highest uncertainty first.

    Returns
    -------
    dict with 'coverage', 'risk', 'aurc'
    """
    N = probs.shape[0]
    preds = probs.argmax(dim=-1)
    correct = (preds == targets).float()

    # Sort by uncertainty (ascending → most certain first)
    order = uncertainty_scores.argsort()
    correct_sorted = correct[order]

    coverages = np.linspace(0.05, 1.0, n_points)
    risks = []
    for cov in coverages:
        n = max(1, int(cov * N))
        risk = 1.0 - correct_sorted[:n].mean().item()
        risks.append(risk)

    risks = np.array(risks)
    aurc = np.trapezoid(risks, coverages)

    return {"coverage": coverages, "risk": risks, "aurc": aurc}


# ══════════════════════════════════════════════════════════════════════
#  3.  OOD Detection
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ood_detection_scores(
    model: NeuralPredictionMarket,
    id_loader,
    ood_loader,
    capital_mgr: CapitalManager,
    device: torch.device,
    score_fn: str = "epistemic_unc",
) -> dict:
    """Compute OOD detection AUROC and AUPR using market signals.

    Parameters
    ----------
    score_fn : str
        Which uncertainty metric to use:
        'epistemic_unc', 'liquidity' (inverted), 'herding', 'entropy_market'
    """
    model.eval()
    capital = capital_mgr.get_capital()

    def collect_scores(loader):
        scores = []
        for images, _ in loader:
            images = images.to(device)
            out = model(images, capital)
            unc = uncertainty_report(out["all_probs"], out["all_bets"], capital)
            scores.append(unc[score_fn].cpu())
        return torch.cat(scores, dim=0)

    id_scores = collect_scores(id_loader)
    ood_scores = collect_scores(ood_loader)

    # Herding is an ID-confidence score (high = agents agree strongly).
    # For OOD detection we need higher scores = more likely OOD, so invert.
    if score_fn == "herding":
        id_scores = -id_scores
        ood_scores = -ood_scores

    # Labels: 0 = ID, 1 = OOD
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    return {
        "auroc": auroc,
        "aupr": aupr,
        "score_fn": score_fn,
        "id_scores": id_scores,
        "ood_scores": ood_scores,
    }


# ══════════════════════════════════════════════════════════════════════
#  4.  Volatility analysis (batch‑level)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def volatility_analysis(
    model: NeuralPredictionMarket,
    loader,
    capital_mgr: CapitalManager,
    device: torch.device,
    eps: float = 0.01,
    n_samples: int = 5,
    max_batches: int = 20,
) -> dict:
    """Run volatility estimation on a few batches."""
    model.eval()
    capital = capital_mgr.get_capital()

    all_js = []
    all_gini_std = []

    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        vol = estimate_volatility(model, images, capital, eps=eps,
                                  n_samples=n_samples)
        all_js.append(vol["mean_js_div"].cpu())
        all_gini_std.append(vol["influence_gini_std"].cpu())

    return {
        "mean_js_div": torch.cat(all_js),
        "influence_gini_std": torch.cat(all_gini_std),
    }
