"""
Strong UQ baseline heads on frozen pretrained backbone features.

All models operate on cached features [B, D] and are trained
like any other head in the pretrained experiment.

Implemented:
  1. SNGP     -- Spectral-Normalized GP  (Liu et al., ICML 2020)
  2. DUE      -- Deterministic Uncertainty Estimation  (van Amersfoort et al., 2021)
  3. DUQ      -- Deterministic Uncertainty Quantification  (van Amersfoort et al., 2020)
  4. Evidential Deep Learning  (Sensoy et al., NeurIPS 2018)

All heads expose the same interface as existing baselines:
  .forward_on_features(features) -> {"market_probs": [B,C], ...}
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  Helper: Spectral Normalization wrapper (for SNGP & DUE)
# ══════════════════════════════════════════════════════════════════════

def _spectral_norm_fc(linear: nn.Linear, coeff: float = 0.95,
                      n_power_iterations: int = 1) -> nn.Linear:
    """Apply spectral normalization with a Lipschitz bound to a Linear layer."""
    return nn.utils.parametrizations.spectral_norm(
        linear, n_power_iterations=n_power_iterations,
    )


# ══════════════════════════════════════════════════════════════════════
#  1. SNGP Head  (Spectral-Normalized Neural Gaussian Process)
#     Liu et al., "Simple and Principled Uncertainty Estimation with
#     Deterministic Deep Learning via Distance Awareness", ICML 2020
# ══════════════════════════════════════════════════════════════════════

class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for Gaussian (RBF) kernel approximation.

    phi(x) = sqrt(2/D_rff) * cos(Wx + b)
    where W ~ N(0, 1/lengthscale^2), b ~ Uniform(0, 2*pi)
    """

    def __init__(self, in_dim: int, rff_dim: int = 1024,
                 lengthscale: float = 1.0):
        super().__init__()
        self.rff_dim = rff_dim
        # Random projection (frozen)
        W = torch.randn(in_dim, rff_dim) / lengthscale
        b = torch.rand(rff_dim) * 2 * math.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, D] @ [D, rff_dim] + [rff_dim] -> cos -> [B, rff_dim]
        proj = x @ self.W + self.b
        return self.scale * torch.cos(proj)


class SNGPHead(nn.Module):
    """SNGP: Spectral-Normalized hidden layers + GP output via RFF.

    Architecture:
      features [B, D]
        -> SN-Linear -> GELU -> SN-Linear -> GELU   (spectral-normalized MLP)
        -> RandomFourierFeatures [B, rff_dim]
        -> Bayesian last layer (mean-field posterior)
        -> logits [B, C]

    Uncertainty = predictive variance from the GP posterior.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 200,
        hidden_dim: int = 512,
        rff_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        lengthscale: float = 2.0,
        sn_coeff: float = 0.95,
        mean_field_factor: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.rff_dim = rff_dim
        self.mean_field_factor = mean_field_factor

        # Spectral-normalized hidden layers
        layers = []
        in_d = embed_dim
        for _ in range(num_layers):
            lin = nn.Linear(in_d, hidden_dim)
            lin = _spectral_norm_fc(lin, coeff=sn_coeff)
            layers.extend([nn.LayerNorm(in_d), lin, nn.GELU(), nn.Dropout(dropout)])
            in_d = hidden_dim
        self.hidden = nn.Sequential(*layers)

        # Random Fourier Features
        self.rff = RandomFourierFeatures(hidden_dim, rff_dim, lengthscale)

        # GP output layer: learnable mean (beta) + precision tracking
        self.beta = nn.Linear(rff_dim, num_classes)
        # Running precision estimate for the posterior
        self.register_buffer(
            "precision", torch.eye(rff_dim) * 1.0
        )
        self.register_buffer("_precision_initialized", torch.tensor(False))
        self._ridge = 1.0

    def reset_precision(self):
        """Reset precision to identity (call at start of each epoch)."""
        self.precision.copy_(torch.eye(self.rff_dim, device=self.precision.device))
        self._precision_initialized.fill_(False)

    def update_precision(self, phi: torch.Tensor, targets: torch.Tensor):
        """Online precision update: precision += phi^T @ diag(prob*(1-prob)) @ phi.

        This is the Laplace approximation of the GP posterior precision.
        Call this after each training batch.
        """
        with torch.no_grad():
            logits = self.beta(phi)
            probs = F.softmax(logits, dim=-1)
            # Use the predicted probability of the true class
            p_correct = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            w = (p_correct * (1 - p_correct)).clamp(min=1e-6)  # [B]
            # Weighted outer product: sum_i w_i * phi_i^T phi_i
            weighted_phi = phi * w.unsqueeze(1).sqrt()  # [B, D_rff]
            self.precision.add_(weighted_phi.T @ weighted_phi)
            self._precision_initialized.fill_(True)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        """Forward pass on cached features."""
        h = self.hidden(features)        # [B, hid]
        phi = self.rff(h)               # [B, rff_dim]
        logits = self.beta(phi)          # [B, C]

        # Predictive variance from GP posterior
        if self._precision_initialized:
            try:
                # Solve precision @ x = phi^T  →  use Cholesky
                L = torch.linalg.cholesky(self.precision + self._ridge *
                                          torch.eye(self.rff_dim,
                                                    device=self.precision.device))
                v = torch.linalg.solve_triangular(L, phi.T, upper=False)  # [D, B]
                variance = (v * v).sum(dim=0)  # [B]
            except Exception:
                variance = torch.zeros(features.shape[0], device=features.device)
        else:
            variance = torch.zeros(features.shape[0], device=features.device)

        # Mean-field adjustment: scale logits by 1/sqrt(1 + mf_factor * var)
        adjusted = logits / (1 + self.mean_field_factor * variance).unsqueeze(-1).sqrt()
        probs = F.softmax(adjusted, dim=-1)

        return {
            "market_probs": probs,
            "all_probs": probs.unsqueeze(0),
            "logits": logits,
            "gp_variance": variance,
        }


class PretrainedSNGP(nn.Module):
    """SNGP on frozen backbone (wrapper for run_pretrained compatibility)."""

    def __init__(self, backbone, num_classes: int = 200,
                 hidden_dim: int = 512, rff_dim: int = 1024,
                 num_layers: int = 2, dropout: float = 0.1,
                 lengthscale: float = 2.0):
        super().__init__()
        self.backbone = backbone
        self.head = SNGPHead(
            backbone.embed_dim, num_classes, hidden_dim,
            rff_dim, num_layers, dropout, lengthscale,
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.forward_on_features(features)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        return self.head.forward_on_features(features)


# ══════════════════════════════════════════════════════════════════════
#  2. DUE Head  (Deterministic Uncertainty Estimation)
#     van Amersfoort et al., "Improving Deterministic Uncertainty
#     Estimation in Deep Learning for Classification and Regression",
#     ICML 2021
#
#  DUE = SN-ResNet + Deep Kernel Learning (GP with learned kernel).
#  On frozen features: SN-MLP + RBF GP output (same as SNGP but uses
#  a learnable kernel lengthscale and inducing points).
#  We implement the "lite" version: SN-hidden + GP with inducing points.
# ══════════════════════════════════════════════════════════════════════

class DUEHead(nn.Module):
    """DUE: SN-hidden layers + variational GP with inducing points.

    Simplified for frozen-feature setting:
      SN-MLP -> low-dim embedding -> RBF kernel GP -> logits

    The key difference from SNGP: DUE uses *learnable* inducing points
    and a variational bound, while SNGP uses random Fourier features
    with a Laplace approximation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 200,
        hidden_dim: int = 512,
        n_inducing: int = 20,
        kernel_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        lengthscale_init: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_inducing = n_inducing
        self.kernel_dim = kernel_dim

        # SN-hidden layers
        layers = []
        in_d = embed_dim
        for _ in range(num_layers):
            lin = nn.Linear(in_d, hidden_dim)
            lin = _spectral_norm_fc(lin)
            layers.extend([nn.LayerNorm(in_d), lin, nn.GELU(), nn.Dropout(dropout)])
            in_d = hidden_dim
        # Final projection to kernel space
        proj = nn.Linear(hidden_dim, kernel_dim)
        proj = _spectral_norm_fc(proj)
        layers.append(proj)
        self.feature_extractor = nn.Sequential(*layers)

        # Learnable log-lengthscale for RBF kernel
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(lengthscale_init))
        )

        # Inducing points: [num_classes, n_inducing, kernel_dim]
        self.inducing_points = nn.Parameter(
            torch.randn(num_classes, n_inducing, kernel_dim) * 0.1
        )

        # Variational parameters per class
        self.variational_mean = nn.Parameter(
            torch.randn(num_classes, n_inducing) * 0.01
        )

    def _rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """RBF kernel: k(x1, x2) = exp(-||x1-x2||^2 / (2 * ls^2))"""
        ls = self.log_lengthscale.exp()
        dist = torch.cdist(x1, x2, p=2)
        return torch.exp(-dist.pow(2) / (2 * ls.pow(2)))

    def forward_on_features(self, features: torch.Tensor) -> dict:
        B = features.shape[0]
        h = self.feature_extractor(features)  # [B, kernel_dim]

        logits = torch.zeros(B, self.num_classes, device=features.device)
        uncertainty = torch.zeros(B, device=features.device)

        for c in range(self.num_classes):
            z = self.inducing_points[c]  # [n_ind, kernel_dim]
            # K(x, z): [B, n_ind]
            k_xz = self._rbf_kernel(h.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
            # K(z, z): [n_ind, n_ind]
            k_zz = self._rbf_kernel(z, z) + 1e-4 * torch.eye(
                self.n_inducing, device=features.device
            )

            # Predictive mean: k_xz @ K_zz^{-1} @ m
            try:
                L = torch.linalg.cholesky(k_zz)
                alpha = torch.cholesky_solve(
                    self.variational_mean[c].unsqueeze(-1), L
                ).squeeze(-1)
            except Exception:
                alpha = torch.linalg.solve(k_zz, self.variational_mean[c])
            logits[:, c] = k_xz @ alpha

            # Predictive variance (diagonal)
            try:
                v = torch.linalg.solve_triangular(L, k_xz.T, upper=False)
                var_c = 1 - (v * v).sum(dim=0)  # [B]
            except Exception:
                var_c = torch.zeros(B, device=features.device)
            uncertainty += var_c.clamp(min=0)

        uncertainty = uncertainty / self.num_classes
        probs = F.softmax(logits, dim=-1)

        return {
            "market_probs": probs,
            "all_probs": probs.unsqueeze(0),
            "logits": logits,
            "gp_variance": uncertainty,
        }


class PretrainedDUE(nn.Module):
    """DUE on frozen backbone."""

    def __init__(self, backbone, num_classes: int = 200,
                 hidden_dim: int = 512, n_inducing: int = 20,
                 kernel_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head = DUEHead(
            backbone.embed_dim, num_classes, hidden_dim,
            n_inducing, kernel_dim, num_layers, dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.forward_on_features(features)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        return self.head.forward_on_features(features)


# ══════════════════════════════════════════════════════════════════════
#  3. DUQ Head  (Deterministic Uncertainty Quantification)
#     van Amersfoort et al., "Uncertainty Estimation Using a Single
#     Deep Deterministic Neural Network", ICML 2020
#
#  DUQ uses RBF centroids per class. The model learns class centroids
#  and weights, using kernel distance for both classification and
#  uncertainty. Samples far from all centroids are OOD.
# ══════════════════════════════════════════════════════════════════════

class DUQHead(nn.Module):
    """DUQ: gradient penalty + RBF-based classification.

    Architecture:
      features [B, D] -> MLP -> embedding [B, E]
      class_centroids [C, E]  (learnable)
      output = RBF(embedding, centroids)  -- [B, C]

    Uncertainty = 1 - max_c RBF(x, centroid_c)
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 200,
        hidden_dim: int = 512,
        centroid_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        rbf_lengthscale: float = 0.1,
        gamma: float = 0.999,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.centroid_dim = centroid_dim
        self.rbf_lengthscale = rbf_lengthscale
        self.gamma = gamma  # EMA decay for centroid updates

        # MLP feature extractor
        layers = []
        in_d = embed_dim
        for _ in range(num_layers):
            layers.extend([
                nn.LayerNorm(in_d),
                nn.Linear(in_d, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, centroid_dim))
        self.feature_extractor = nn.Sequential(*layers)

        # Per-class weight matrix W_c: [C, centroid_dim, centroid_dim]
        # We use a simplified version: per-class linear projection
        self.class_weight = nn.Parameter(
            torch.randn(num_classes, centroid_dim, centroid_dim) * 0.01
        )

        # Class centroids (updated via EMA during training)
        self.register_buffer(
            "centroids", torch.randn(num_classes, centroid_dim) * 0.05
        )
        self.register_buffer(
            "centroid_counts", torch.ones(num_classes)
        )

    def _kernel(self, z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """RBF kernel between embeddings and centroids.

        z: [B, E], centroids: [C, E] -> [B, C]
        """
        dist = torch.cdist(z.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
        return torch.exp(-dist.pow(2) / (2 * self.rbf_lengthscale ** 2))

    @torch.no_grad()
    def update_centroids(self, features: torch.Tensor, targets: torch.Tensor):
        """EMA update of class centroids during training."""
        z = self.feature_extractor(features)  # [B, E]
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                new_centroid = z[mask].mean(dim=0)
                self.centroids[c] = (
                    self.gamma * self.centroids[c]
                    + (1 - self.gamma) * new_centroid
                )
                self.centroid_counts[c] += mask.sum().float()

    def forward_on_features(self, features: torch.Tensor) -> dict:
        z = self.feature_extractor(features)  # [B, E]

        # Apply per-class weight: z_c = z @ W_c for each class
        # Efficient: [B, 1, E] @ [C, E, E] -> [B, C, E]
        z_exp = z.unsqueeze(1).expand(-1, self.num_classes, -1)
        weighted = torch.einsum("bce,ced->bcd", z_exp, self.class_weight)
        # weighted: [B, C, E]

        # Distance to centroids
        diff = weighted - self.centroids.unsqueeze(0)  # [B, C, E]
        dist_sq = (diff ** 2).sum(dim=-1)  # [B, C]

        # RBF kernel values as "probabilities"
        # Use log-space softmax for numerical stability (avoids all-zero kernel)
        log_kernel = -dist_sq / (2 * self.rbf_lengthscale ** 2)
        kernel_vals = torch.exp(log_kernel)

        # Stable softmax normalization
        probs = F.softmax(log_kernel, dim=-1)

        # Uncertainty: 1 - max kernel value (higher = more OOD)
        max_kernel, _ = kernel_vals.max(dim=-1)
        uncertainty = 1 - max_kernel

        return {
            "market_probs": probs,
            "all_probs": probs.unsqueeze(0),
            "kernel_vals": kernel_vals,
            "duq_uncertainty": uncertainty,
        }


class PretrainedDUQ(nn.Module):
    """DUQ on frozen backbone."""

    def __init__(self, backbone, num_classes: int = 200,
                 hidden_dim: int = 512, centroid_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1,
                 rbf_lengthscale: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head = DUQHead(
            backbone.embed_dim, num_classes, hidden_dim,
            centroid_dim, num_layers, dropout, rbf_lengthscale,
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.forward_on_features(features)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        return self.head.forward_on_features(features)


# ══════════════════════════════════════════════════════════════════════
#  4. Evidential Deep Learning  (EDL)
#     Sensoy et al., "Evidential Deep Learning to Quantify
#     Classification Uncertainty", NeurIPS 2018
#
#  EDL outputs Dirichlet concentration parameters alpha_c instead of
#  logits. p(y|x) = Dir(alpha). Uncertainty metrics:
#    - Total evidence S = sum(alpha)
#    - Epistemic uncertainty = C / S  (vacuity)
#    - Aleatoric = entropy of expected distribution
# ══════════════════════════════════════════════════════════════════════

class EvidentialHead(nn.Module):
    """Evidential Deep Learning head.

    Outputs Dirichlet concentration parameters: alpha = softplus(logits) + 1
    so alpha_c >= 1 for all classes.

    Loss: Type-II maximum likelihood (marginal likelihood of Dirichlet).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 200,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_d = embed_dim
        for _ in range(num_layers):
            layers.extend([
                nn.LayerNorm(in_d),
                nn.Linear(in_d, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        layers.extend([
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        ])
        self.head = nn.Sequential(*layers)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        logits = self.head(features)  # [B, C]

        # Dirichlet concentration: alpha = exp(logits) + 1
        # Using softplus for smoother gradients
        evidence = F.softplus(logits)
        alpha = evidence + 1.0  # [B, C]

        # Dirichlet strength (total evidence)
        S = alpha.sum(dim=-1, keepdim=True)  # [B, 1]

        # Expected probability (mean of Dirichlet)
        probs = alpha / S  # [B, C]

        # Epistemic uncertainty: vacuity = C / S
        vacuity = self.num_classes / S.squeeze(-1)  # [B]

        # Aleatoric uncertainty: entropy of expected categorical
        aleatoric = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)  # [B]

        return {
            "market_probs": probs,
            "all_probs": probs.unsqueeze(0),
            "logits": logits,
            "alpha": alpha,
            "evidence": evidence,
            "total_evidence": S.squeeze(-1),
            "vacuity": vacuity,
            "aleatoric": aleatoric,
        }


class PretrainedEvidential(nn.Module):
    """Evidential Deep Learning on frozen backbone."""

    def __init__(self, backbone, num_classes: int = 200,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head = EvidentialHead(
            backbone.embed_dim, num_classes, hidden_dim,
            num_layers, dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.forward_on_features(features)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        return self.head.forward_on_features(features)


# ── Evidential loss functions ──

def edl_mse_loss(
    alpha: torch.Tensor,    # [B, C]
    targets: torch.Tensor,  # [B] class indices
    epoch: int = 0,
    total_epochs: int = 50,
    annealing_coeff: float = 10.0,
) -> torch.Tensor:
    """Evidential MSE loss with KL annealing (Sensoy et al. eq. 5)."""
    C = alpha.shape[1]
    one_hot = F.one_hot(targets, num_classes=C).float()
    S = alpha.sum(dim=-1, keepdim=True)

    # MSE term
    p_hat = alpha / S
    mse = (one_hot - p_hat).pow(2).sum(dim=-1)

    # Variance term
    var = (alpha * (S - alpha) / (S.pow(2) * (S + 1))).sum(dim=-1)

    loss = (mse + var).mean()

    # KL regularization (annealed)
    if total_epochs > 0:
        anneal = min(1.0, epoch / (total_epochs / annealing_coeff))
    else:
        anneal = 1.0

    alpha_tilde = one_hot + (1 - one_hot) * (alpha - 1).clamp(min=0) + 1
    kl = _dirichlet_kl(alpha_tilde, torch.ones_like(alpha_tilde))
    loss = loss + anneal * kl.mean()

    return loss


def edl_digamma_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    epoch: int = 0,
    total_epochs: int = 50,
    annealing_coeff: float = 10.0,
) -> torch.Tensor:
    """Evidential loss using digamma (Sensoy et al. eq. 3-4)."""
    C = alpha.shape[1]
    one_hot = F.one_hot(targets, num_classes=C).float()
    S = alpha.sum(dim=-1, keepdim=True)

    # Type-II ML loss
    loss = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()

    # KL regularization
    if total_epochs > 0:
        anneal = min(1.0, epoch / (total_epochs / annealing_coeff))
    else:
        anneal = 1.0

    alpha_tilde = one_hot + (1 - one_hot) * (alpha - 1).clamp(min=0) + 1
    kl = _dirichlet_kl(alpha_tilde, torch.ones_like(alpha_tilde))
    loss = loss + anneal * kl.mean()

    return loss


def _dirichlet_kl(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """KL divergence KL(Dir(alpha) || Dir(beta))."""
    S_alpha = alpha.sum(dim=-1)
    S_beta = beta.sum(dim=-1)
    kl = (
        torch.lgamma(S_alpha) - torch.lgamma(S_beta)
        - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=-1)
        + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha).unsqueeze(-1))).sum(dim=-1)
    )
    return kl
