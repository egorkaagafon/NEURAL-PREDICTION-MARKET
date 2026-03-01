"""
Post-hoc OOD detection methods applied on cached features + logits.

These methods do NOT require separate training -- they are scoring functions
applied to the outputs of any trained backbone + head.

Implemented methods (state-of-the-art as of OpenOOD v1.5):

  1. Energy Score       (Liu et al., 2020)  -- -logsumexp(logits)
  2. ODIN               (Liang et al., 2018) -- temperature-scaled softmax
  3. Mahalanobis        (Lee et al., 2018)  -- class-conditional Gaussian distance
  4. ViM                (Wang et al., 2022) -- Virtual-logit Matching
  5. ReAct              (Sun et al., 2021)  -- Rectified Activations + Energy
  6. KNN                (Sun et al., 2022)  -- k-nearest-neighbour distance

All methods operate on **cached features** [B, D] and/or logits [B, C],
making them cheap to evaluate (no backbone forward needed for ID data).
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


# ══════════════════════════════════════════════════════════════════════
#  1. Energy Score  (Liu et al., NIPS 2020)
# ══════════════════════════════════════════════════════════════════════

def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy score: -T * logsumexp(logits / T).

    Higher energy = more likely OOD.

    Parameters
    ----------
    logits : [B, C] raw logits (before softmax)
    temperature : float, default 1.0

    Returns
    -------
    scores : [B] energy scores (higher = more OOD)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


# ══════════════════════════════════════════════════════════════════════
#  2. ODIN  (Liang et al., ICLR 2018)
# ══════════════════════════════════════════════════════════════════════

def odin_score(
    logits: torch.Tensor,
    temperature: float = 1000.0,
) -> torch.Tensor:
    """ODIN score: max softmax probability with temperature scaling.

    In the original ODIN, input perturbation is also used. Here we only
    implement temperature scaling (applicable to cached features).
    Score = -max_c softmax(logits/T)  (negated so higher = more OOD).

    Parameters
    ----------
    logits : [B, C]
    temperature : float

    Returns
    -------
    scores : [B] (higher = more OOD)
    """
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    max_prob, _ = probs.max(dim=-1)
    return -max_prob  # negate so higher = OOD


# ══════════════════════════════════════════════════════════════════════
#  3. Mahalanobis Distance  (Lee et al., NIPS 2018)
# ══════════════════════════════════════════════════════════════════════

class MahalanobisDetector:
    """Class-conditional Mahalanobis distance OOD detector.

    Fits class-conditional Gaussians on cached training features,
    then scores test samples by minimum Mahalanobis distance.
    """

    def __init__(self):
        self.class_means: Optional[torch.Tensor] = None   # [C, D]
        self.precision: Optional[torch.Tensor] = None      # [D, D]
        self.fitted = False
        self.num_classes = 0

    @torch.no_grad()
    def fit(
        self,
        features: torch.Tensor,     # [N, D]
        targets: torch.Tensor,      # [N]
        num_classes: int,
        reg: float = 1e-5,
    ) -> "MahalanobisDetector":
        """Fit class-conditional Gaussians from training features."""
        D = features.shape[1]
        self.num_classes = num_classes
        device = features.device

        # Class means
        means = torch.zeros(num_classes, D, device=device)
        for c in range(num_classes):
            mask = targets == c
            if mask.sum() > 0:
                means[c] = features[mask].mean(dim=0)
        self.class_means = means

        # Shared covariance (tied covariance model)
        centered = features - means[targets]  # [N, D]
        cov = (centered.T @ centered) / features.shape[0]
        cov += reg * torch.eye(D, device=device)

        # Precision matrix via Cholesky
        try:
            L = torch.linalg.cholesky(cov)
            self.precision = torch.cholesky_inverse(L)
        except Exception:
            # Fallback: pseudo-inverse
            self.precision = torch.linalg.pinv(cov)

        self.fitted = True
        return self

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance score (higher = more OOD).

        Returns the minimum Mahalanobis distance over all classes.
        """
        assert self.fitted, "Call .fit() first"
        B, D = features.shape

        # [B, C, D]
        diff = features.unsqueeze(1) - self.class_means.unsqueeze(0)
        # Mahalanobis: diff @ precision @ diff.T per class
        # [B, C]
        mahal = (diff @ self.precision).mul_(diff).sum(dim=-1)
        # Minimum distance across classes
        min_dist, _ = mahal.min(dim=1)
        return min_dist


# ══════════════════════════════════════════════════════════════════════
#  4. ViM  (Wang et al., CVPR 2022)
# ══════════════════════════════════════════════════════════════════════

class ViMDetector:
    """Virtual-logit Matching OOD detector.

    ViM uses the "virtual logit" -- a logit computed from the residual
    of features projected onto the principal subspace of the weight matrix.
    The virtual logit captures information *not* represented by any class.
    """

    def __init__(self):
        self.principal_space: Optional[torch.Tensor] = None  # [D, d]
        self.train_mean: Optional[torch.Tensor] = None       # [D]
        self.alpha: float = 1.0
        self.fitted = False

    @torch.no_grad()
    def fit(
        self,
        features: torch.Tensor,   # [N, D]
        weight: torch.Tensor,     # [C, D] -- classifier weight matrix
        dim_ratio: float = 0.5,    # fraction of principal dims to keep
    ) -> "ViMDetector":
        """Fit ViM from training features and classifier weights."""

        D = features.shape[1]
        d = max(1, int(D * dim_ratio))

        self.train_mean = features.mean(dim=0)

        # Principal subspace from classifier weights
        # U, S, Vh = svd(W^T)  →  first d right-singular vectors
        U, S, Vh = torch.linalg.svd(weight.T, full_matrices=False)
        self.principal_space = U[:, :d]  # [D, d]

        # Compute alpha: scales virtual logit to match logit magnitudes
        centered = features - self.train_mean
        proj = centered @ self.principal_space  # [N, d]
        residual = centered - proj @ self.principal_space.T  # [N, D]
        vlogit_norm = residual.norm(dim=-1)  # [N]

        logits = (centered @ weight.T)  # [N, C]
        max_logit = logits.max(dim=-1).values
        self.alpha = (max_logit.mean() / vlogit_norm.mean().clamp(min=1e-8)).item()

        self.fitted = True
        return self

    @torch.no_grad()
    def score(
        self,
        features: torch.Tensor,     # [B, D]
        weight: torch.Tensor,       # [C, D]
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score using ViM (higher = more OOD)."""
        assert self.fitted, "Call .fit() first"

        centered = features - self.train_mean
        proj = centered @ self.principal_space
        residual = centered - proj @ self.principal_space.T
        vlogit = self.alpha * residual.norm(dim=-1, keepdim=True)  # [B, 1]

        # Original logits
        logits = centered @ weight.T  # [B, C]
        if bias is not None:
            logits = logits + bias

        # Append virtual logit → energy on augmented logits
        aug_logits = torch.cat([logits, vlogit], dim=-1)  # [B, C+1]
        vim_energy = -torch.logsumexp(aug_logits, dim=-1)
        return vim_energy


# ══════════════════════════════════════════════════════════════════════
#  5. ReAct  (Sun et al., ICML 2021)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def react_score(
    features: torch.Tensor,      # [B, D]
    weight: torch.Tensor,        # [C, D]
    bias: Optional[torch.Tensor] = None,
    percentile: float = 90.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """ReAct: clip feature activations at a percentile, then compute energy.

    On ID data activations are moderate; OOD data has high-magnitude
    activations that get clipped, causing a drop in energy (OOD signal).

    Parameters
    ----------
    features : [B, D]
    weight : [C, D]  -- last linear layer weights
    bias : [C] or None
    percentile : float, percentile threshold for clipping
    temperature : float

    Returns
    -------
    scores : [B] (higher = more OOD)
    """
    # Compute threshold on the input batch (in practice, use train set)
    thresh = torch.quantile(features, percentile / 100.0)
    clipped = features.clamp(max=thresh)
    logits = clipped @ weight.T
    if bias is not None:
        logits = logits + bias
    return energy_score(logits, temperature)


class ReActDetector:
    """Stateful ReAct: fits threshold from training features."""

    def __init__(self, percentile: float = 90.0, temperature: float = 1.0):
        self.percentile = percentile
        self.temperature = temperature
        self.threshold: Optional[float] = None

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> "ReActDetector":
        """Compute activation threshold from training features."""
        self.threshold = torch.quantile(
            features, self.percentile / 100.0
        ).item()
        return self

    @torch.no_grad()
    def score(
        self,
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.threshold is not None, "Call .fit() first"
        clipped = features.clamp(max=self.threshold)
        logits = clipped @ weight.T
        if bias is not None:
            logits = logits + bias
        return energy_score(logits, self.temperature)


# ══════════════════════════════════════════════════════════════════════
#  6. KNN Distance  (Sun et al., ICML 2022)
# ══════════════════════════════════════════════════════════════════════

class KNNDetector:
    """k-Nearest Neighbour OOD detector on cached features.

    Scores = mean distance to k nearest training neighbours.
    Simple but competitive, especially on pretrained features.
    """

    def __init__(self, k: int = 50):
        self.k = k
        self.train_features: Optional[torch.Tensor] = None
        self.fitted = False

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> "KNNDetector":
        """Store (normalized) training features."""
        self.train_features = F.normalize(features, dim=-1)
        self.fitted = True
        return self

    @torch.no_grad()
    def score(self, features: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """Compute mean k-NN distance (higher = more OOD)."""
        assert self.fitted, "Call .fit() first"
        test_norm = F.normalize(features, dim=-1)
        scores = []
        for i in range(0, len(test_norm), batch_size):
            batch = test_norm[i:i+batch_size]
            # Cosine similarity → distance = 1 - sim
            sim = batch @ self.train_features.T  # [b, N_train]
            k_eff = min(self.k, sim.shape[-1])
            topk_sim, _ = sim.topk(k_eff, dim=-1)  # [b, k]
            knn_dist = (1 - topk_sim).mean(dim=-1)  # [b]
            scores.append(knn_dist)
        return torch.cat(scores)


# ══════════════════════════════════════════════════════════════════════
#  Unified evaluation helper
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_posthoc_ood(
    method_name: str,
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> Dict[str, float]:
    """Compute AUROC and AUPR from ID and OOD score arrays."""
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([
        id_scores.cpu().numpy(),
        ood_scores.cpu().numpy(),
    ])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
    }


@torch.no_grad()
def run_all_posthoc_ood(
    head: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    ood_features: torch.Tensor,
    num_classes: int,
    k_knn: int = 50,
    react_percentile: float = 90.0,
    vim_dim_ratio: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Run all 6 post-hoc OOD methods on cached features.

    Parameters
    ----------
    head : nn.Module
        A trained classifier head that accepts features.
        Must have a final nn.Linear layer whose .weight / .bias are extracted.
    train_features : [N_train, D]
    train_targets : [N_train]
    test_features : [N_test, D]  -- ID test
    ood_features : [N_ood, D]   -- OOD test
    num_classes : int
    k_knn : int, k for KNN
    react_percentile : float

    Returns
    -------
    dict : {method_name: {"auroc": float, "aupr": float}}
    """
    # Extract classifier weight/bias from the head
    weight, bias = _extract_last_linear(head)
    device = train_features.device

    results = {}

    # 1. Get logits for ID and OOD
    id_logits = _get_logits(head, test_features)
    ood_logits = _get_logits(head, ood_features)

    # -- Energy Score --
    id_e = energy_score(id_logits)
    ood_e = energy_score(ood_logits)
    results["energy"] = evaluate_posthoc_ood("energy", id_e, ood_e)

    # -- ODIN --
    id_o = odin_score(id_logits, temperature=1000.0)
    ood_o = odin_score(ood_logits, temperature=1000.0)
    results["odin"] = evaluate_posthoc_ood("odin", id_o, ood_o)

    # -- Mahalanobis --
    maha = MahalanobisDetector()
    maha.fit(train_features, train_targets, num_classes)
    id_m = maha.score(test_features)
    ood_m = maha.score(ood_features)
    results["mahalanobis"] = evaluate_posthoc_ood("mahalanobis", id_m, ood_m)

    # -- ViM --
    if weight is not None and weight.shape[1] == train_features.shape[1]:
        vim = ViMDetector()
        vim.fit(train_features, weight, dim_ratio=vim_dim_ratio)
        id_v = vim.score(test_features, weight, bias)
        ood_v = vim.score(ood_features, weight, bias)
        results["vim"] = evaluate_posthoc_ood("vim", id_v, ood_v)

    # -- ReAct --
    if weight is not None and weight.shape[1] == train_features.shape[1]:
        ra = ReActDetector(percentile=react_percentile)
        ra.fit(train_features)
        id_r = ra.score(test_features, weight, bias)
        ood_r = ra.score(ood_features, weight, bias)
        results["react"] = evaluate_posthoc_ood("react", id_r, ood_r)

    # -- KNN --
    knn = KNNDetector(k=k_knn)
    knn.fit(train_features)
    id_k = knn.score(test_features)
    ood_k = knn.score(ood_features)
    results["knn"] = evaluate_posthoc_ood("knn", id_k, ood_k)

    return results


def _extract_last_linear(module: nn.Module) -> Tuple[Optional[torch.Tensor],
                                                       Optional[torch.Tensor]]:
    """Walk a module and extract the last nn.Linear's weight/bias."""
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        return None, None
    return last_linear.weight.detach(), (
        last_linear.bias.detach() if last_linear.bias is not None else None
    )


@torch.no_grad()
def _get_logits(head: nn.Module, features: torch.Tensor,
                batch_size: int = 2048) -> torch.Tensor:
    """Get raw logits from a head (handles both Sequential and custom heads)."""
    head.eval()
    all_logits = []
    # Check if head uses forward_on_features (our UQ heads)
    use_fof = hasattr(head, "forward_on_features") and not isinstance(head, nn.Sequential)
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        out = head.forward_on_features(batch) if use_fof else head(batch)
        if isinstance(out, dict):
            # Some heads return dicts -- try to get logits or raw probs
            if "logits" in out:
                all_logits.append(out["logits"])
            elif "market_probs" in out:
                # Convert probs back to log-space as pseudo-logits
                all_logits.append(torch.log(out["market_probs"].clamp(min=1e-8)))
            else:
                all_logits.append(next(iter(out.values())))
        else:
            all_logits.append(out)
    return torch.cat(all_logits, dim=0)
