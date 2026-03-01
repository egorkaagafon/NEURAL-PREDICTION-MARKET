"""
NPM with a frozen pretrained backbone (from timm).

Architecture:
  Frozen backbone (ViT, DeiT, ResNet, …)  →  feature vector [B, D]
       ↓
  AgentPool (K agents, each with feature_mask + head + bet)
       ↓
  MarketAggregator  →  P(y|x)

Only the agent heads are trained.  The backbone is frozen (no gradient).
This isolates the contribution of the NPM market mechanism from
backbone quality, enabling clean ablation and fair comparison with
other uncertainty methods on the *same* features.

Supported backbone families (via timm):
  ViT / DeiT:
    - deit_tiny_patch16_224   (5.7 M, embed_dim=192)
    - deit_small_patch16_224  (22 M, embed_dim=384)  — "small transformer"
    - vit_small_patch16_224   (22 M, embed_dim=384)
  ResNet:
    - resnet18                (11.7 M, embed_dim=512)
    - resnet50                (25.6 M, embed_dim=2048)
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

import timm

from npm_core.agents import AgentPool
from npm_core.market import MarketAggregator


# ══════════════════════════════════════════════════════════════════════
#  Backbone registry — quick look-up of supported backbones
# ══════════════════════════════════════════════════════════════════════

BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Vision Transformers ──
    "deit_tiny_patch16_224": {
        "family": "vit", "embed_dim": 192, "params_m": 5.7,
        "description": "DeiT-Tiny — lightweight ViT (default)",
    },
    "deit_small_patch16_224": {
        "family": "vit", "embed_dim": 384, "params_m": 22,
        "description": "DeiT-Small — larger ViT with richer features",
    },
    "vit_small_patch16_224": {
        "family": "vit", "embed_dim": 384, "params_m": 22,
        "description": "ViT-Small (original ViT, patch-16)",
    },
    # ── ResNets ──
    "resnet18": {
        "family": "cnn", "embed_dim": 512, "params_m": 11.7,
        "description": "ResNet-18 — classic CNN baseline",
    },
    "resnet50": {
        "family": "cnn", "embed_dim": 2048, "params_m": 25.6,
        "description": "ResNet-50 — strong CNN baseline",
    },
}


def list_backbones() -> None:
    """Print a table of supported backbone models."""
    print(f"{'name':<30s} {'family':<6s} {'dim':>5s} {'params':>8s}  description")
    print("-" * 90)
    for name, info in BACKBONE_REGISTRY.items():
        print(f"{name:<30s} {info['family']:<6s} {info['embed_dim']:>5d} "
              f"{info['params_m']:>7.1f}M  {info['description']}")


# ══════════════════════════════════════════════════════════════════════
#  Parameter-budget utilities for fair comparison
# ══════════════════════════════════════════════════════════════════════

def _head_params(embed_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int) -> int:
    """Exact trainable-param count for a PretrainedClassifierHead."""
    # LayerNorm(embed_dim): 2 * embed_dim
    p = 2 * embed_dim
    # First linear: embed_dim -> hidden_dim
    p += embed_dim * hidden_dim + hidden_dim
    # Additional hidden layers: hidden_dim -> hidden_dim
    for _ in range(num_layers - 1):
        p += hidden_dim * hidden_dim + hidden_dim
    # Output linear: hidden_dim -> num_classes
    p += hidden_dim * num_classes + num_classes
    return p


def _moe_params(embed_dim: int, expert_hidden_dim: int, num_classes: int,
                num_experts: int) -> int:
    """Exact trainable-param count for a PretrainedMoE."""
    # Router: Linear(embed_dim, num_experts)
    p = embed_dim * num_experts + num_experts
    # Each expert: LN(embed_dim) + Linear(D,h) + Linear(h,h) + Linear(h,C)
    per_expert = (2 * embed_dim                               # LayerNorm
                  + embed_dim * expert_hidden_dim + expert_hidden_dim   # Linear 1
                  + expert_hidden_dim ** 2 + expert_hidden_dim         # Linear 2
                  + expert_hidden_dim * num_classes + num_classes)      # Linear 3
    p += num_experts * per_expert
    return p


def solve_ensemble_hidden_dim(target_params: int, embed_dim: int,
                              num_classes: int, num_members: int,
                              num_layers: int) -> int:
    """Find hidden_dim so that num_members × per-head params ≈ target_params.

    Binary search for the closest integer hidden_dim.
    """
    per_head_target = target_params / num_members
    lo, hi = 8, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        if _head_params(embed_dim, mid, num_classes, num_layers) < per_head_target:
            lo = mid + 1
        else:
            hi = mid
    # Return the value whose total is closest to target
    best_h, best_diff = lo, float("inf")
    for h in range(max(8, lo - 2), lo + 3):
        total = num_members * _head_params(embed_dim, h, num_classes, num_layers)
        if abs(total - target_params) < best_diff:
            best_diff = abs(total - target_params)
            best_h = h
    return best_h


def solve_mc_hidden_dim(target_params: int, embed_dim: int,
                        num_classes: int, num_layers: int) -> int:
    """Find hidden_dim for MC-Dropout (single head) ≈ target_params."""
    lo, hi = 8, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        if _head_params(embed_dim, mid, num_classes, num_layers) < target_params:
            lo = mid + 1
        else:
            hi = mid
    best_h, best_diff = lo, float("inf")
    for h in range(max(8, lo - 2), lo + 3):
        diff = abs(_head_params(embed_dim, h, num_classes, num_layers) - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
    return best_h


def solve_moe_hidden_dim(target_params: int, embed_dim: int,
                         num_classes: int, num_experts: int) -> int:
    """Find expert_hidden_dim for MoE ≈ target_params."""
    lo, hi = 8, 2048
    while lo < hi:
        mid = (lo + hi) // 2
        if _moe_params(embed_dim, mid, num_classes, num_experts) < target_params:
            lo = mid + 1
        else:
            hi = mid
    best_h, best_diff = lo, float("inf")
    for h in range(max(8, lo - 2), lo + 3):
        diff = abs(_moe_params(embed_dim, h, num_classes, num_experts) - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
    return best_h


class PretrainedNPM(nn.Module):
    """NPM with a frozen pretrained backbone.

    Parameters
    ----------
    backbone_name : str
        timm model name.  See ``BACKBONE_REGISTRY`` for tested options:
        - ViT/DeiT:  'deit_tiny_patch16_224', 'deit_small_patch16_224',
                      'vit_small_patch16_224'
        - ResNet:     'resnet18', 'resnet50'
        Any timm model with ``num_classes=0`` that produces a 1-D feature
        vector is supported.
    num_agents : int
        Number of agent heads.
    num_classes : int
        Number of output classes (e.g. 200 for Tiny ImageNet).
    dropout : float
        Dropout in agent heads.
    bet_temperature : float
        Sigmoid temperature for bet head.
    feature_keep_prob : float
        Fraction of backbone features each agent can see.
    freeze_backbone : bool
        If True (default), backbone params are frozen.
    """

    def __init__(
        self,
        backbone_name: str = "deit_tiny_patch16_224",
        num_agents: int = 16,
        num_classes: int = 10,
        dropout: float = 0.1,
        bet_temperature: float = 1.0,
        feature_keep_prob: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_classes = num_classes

        # ── Frozen backbone ──
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,  # remove head
        )
        self.embed_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # keep BN/Dropout in eval mode

        self._freeze_backbone = freeze_backbone

        # ── NPM agent pool ──
        self.agent_pool = AgentPool(
            num_agents=num_agents,
            in_dim=self.embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            bet_temperature=bet_temperature,
            feature_keep_prob=feature_keep_prob,
        )

        self.market = MarketAggregator()

    def train(self, mode: bool = True):
        """Override to keep backbone frozen in eval mode."""
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        capital: Optional[torch.Tensor] = None,
    ) -> dict:
        # Backbone forward (no grad if frozen)
        if self._freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)  # [B, D]
        else:
            features = self.backbone(images)

        all_logits, all_probs, all_bets = self.agent_pool(features)

        if capital is None:
            capital = torch.ones(self.num_agents, device=images.device)
        else:
            capital = capital.detach().clone()

        market_probs = self.market.clearing_price(
            all_probs, all_bets, capital,
        )

        return {
            "features": features,
            "all_logits": all_logits,
            "all_probs": all_probs,
            "all_bets": all_bets,
            "market_probs": market_probs,
        }

    def forward_on_features(
        self,
        features: torch.Tensor,
        capital: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass on pre-extracted features (skip backbone)."""
        all_logits, all_probs, all_bets = self.agent_pool(features)

        if capital is None:
            capital = torch.ones(self.num_agents, device=features.device)
        else:
            capital = capital.detach().clone()

        market_probs = self.market.clearing_price(
            all_probs, all_bets, capital,
        )

        return {
            "features": features,
            "all_logits": all_logits,
            "all_probs": all_probs,
            "all_bets": all_bets,
            "market_probs": market_probs,
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
#  Pretrained backbone wrapper for baselines
# ══════════════════════════════════════════════════════════════════════

class PretrainedBackbone(nn.Module):
    """Frozen pretrained backbone that returns feature vectors.

    Shared by all pretrained baselines for fair comparison.
    Supports both ViT/DeiT and CNN (ResNet) backbones via timm.
    """

    def __init__(self, backbone_name: str = "deit_tiny_patch16_224"):
        super().__init__()
        self.backbone_name = backbone_name
        self.model = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
        )
        self.embed_dim = self.model.num_features

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        family = BACKBONE_REGISTRY.get(backbone_name, {}).get("family", "unknown")
        print(f"  Backbone: {backbone_name}  family={family}  "
              f"embed_dim={self.embed_dim}  (frozen)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()  # always eval
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # [B, D]


# ══════════════════════════════════════════════════════════════════════
#  Pretrained Baselines — heads on frozen backbone
# ══════════════════════════════════════════════════════════════════════

class PretrainedClassifierHead(nn.Module):
    """Classifier head with configurable width and depth for param matching."""

    def __init__(self, embed_dim: int, num_classes: int = 10,
                 dropout: float = 0.1, hidden_dim: int = 0,
                 num_layers: int = 1):
        super().__init__()
        hid = hidden_dim if hidden_dim > 0 else embed_dim
        layers = [nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hid),
                  nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hid, hid), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hid, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)  # [B, C]


class PretrainedEnsemble(nn.Module):
    """Deep ensemble: M independent heads on shared frozen backbone.

    Each member has its own randomly-initialised head.

    **Parameter matching policy** (important for fair comparison):
    Total trainable params = M x per_head_params.  When comparing to
    NPM (or any single model), hidden_dim must be set so that the
    *aggregate* trainable budget matches, NOT the per-member budget.
    Use ``solve_ensemble_hidden_dim()`` to compute the correct value.
    """

    def __init__(self, backbone: PretrainedBackbone,
                 num_members: int = 5, num_classes: int = 10,
                 dropout: float = 0.1, hidden_dim: int = 0,
                 num_layers: int = 1):
        super().__init__()
        self.backbone = backbone
        self.num_members = num_members
        self.heads = nn.ModuleList([
            PretrainedClassifierHead(backbone.embed_dim, num_classes, dropout,
                                     hidden_dim=hidden_dim,
                                     num_layers=num_layers)
            for _ in range(num_members)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.forward_on_features(features)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        """Forward on pre-extracted features (skip backbone)."""
        import torch.nn.functional as F
        logits_list = [h(features) for h in self.heads]
        logits_stack = torch.stack(logits_list, dim=0)  # [M, B, C]
        probs_stack = F.softmax(logits_stack, dim=-1)
        mean_probs = probs_stack.mean(dim=0)
        return {
            "market_probs": mean_probs,
            "all_probs": probs_stack,
        }


class PretrainedMCDropout(nn.Module):
    """MC-Dropout head on frozen backbone.

    Dropout is kept ON at inference for MC sampling.
    """

    def __init__(self, backbone: PretrainedBackbone,
                 mc_samples: int = 10, num_classes: int = 10,
                 dropout: float = 0.1, hidden_dim: int = 0,
                 num_layers: int = 1):
        super().__init__()
        self.backbone = backbone
        self.mc_samples = mc_samples
        self.head = PretrainedClassifierHead(
            backbone.embed_dim, num_classes, dropout,
            hidden_dim=hidden_dim, num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.forward_on_features(features)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        """Forward on pre-extracted features (skip backbone)."""
        import torch.nn.functional as F

        if self.training:
            logits = self.head(features)
            probs = F.softmax(logits, dim=-1)
            return {"market_probs": probs, "all_probs": probs.unsqueeze(0)}

        # MC inference: keep dropout ON in head
        self.head.train()
        with torch.no_grad():
            logits_list = [self.head(features) for _ in range(self.mc_samples)]
        logits_stack = torch.stack(logits_list, dim=0)
        probs_stack = F.softmax(logits_stack, dim=-1)
        mean_probs = probs_stack.mean(dim=0)
        self.head.eval()
        return {"market_probs": mean_probs, "all_probs": probs_stack}


class PretrainedMoE(nn.Module):
    """MoE with learned router on frozen backbone.

    Same architecture as MoEClassifier but uses shared frozen features.
    """

    def __init__(self, backbone: PretrainedBackbone,
                 num_experts: int = 16, top_k: int = 4,
                 num_classes: int = 10, dropout: float = 0.1,
                 expert_hidden_dim: int = 0):
        super().__init__()
        self.backbone = backbone
        self.num_experts = num_experts
        self.top_k = top_k
        embed_dim = backbone.embed_dim

        self.router = nn.Linear(embed_dim, num_experts)
        ehid = expert_hidden_dim if expert_hidden_dim > 0 else embed_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, ehid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ehid, ehid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ehid, num_classes),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.forward_on_features(features)

    def forward_on_features(self, features: torch.Tensor) -> dict:
        """Forward on pre-extracted features (skip backbone)."""
        import torch.nn.functional as F

        gate_logits = self.router(features)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = gate_probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        all_logits = torch.stack([e(features) for e in self.experts], dim=1)
        C = all_logits.shape[-1]
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        selected = all_logits.gather(1, idx_exp)
        weighted = (topk_vals.unsqueeze(-1) * F.softmax(selected, dim=-1)).sum(1)

        return {
            "market_probs": weighted,
            "all_probs": F.softmax(all_logits.permute(1, 0, 2), dim=-1),
            "gate_probs": gate_probs,
        }
