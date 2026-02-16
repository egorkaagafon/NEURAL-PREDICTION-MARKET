"""
NPM with a frozen pretrained ViT backbone (from timm).

Architecture:
  Frozen ViT backbone (e.g. DeiT-Tiny, ViT-Small)  →  CLS token [B, D]
       ↓
  AgentPool (K agents, each with feature_mask + head + bet)
       ↓
  MarketAggregator  →  P(y|x)

Only the agent heads are trained.  The backbone is frozen (no gradient).
This isolates the contribution of the NPM market mechanism from
backbone quality, enabling clean ablation and fair comparison with
other uncertainty methods on the *same* features.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import timm

from npm_core.agents import AgentPool
from npm_core.market import MarketAggregator


class PretrainedNPM(nn.Module):
    """NPM with a frozen pretrained ViT backbone.

    Parameters
    ----------
    backbone_name : str
        timm model name (e.g. 'deit_tiny_patch16_224', 'vit_small_patch16_224').
    num_agents : int
        Number of agent heads.
    num_classes : int
        Number of output classes.
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

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
#  Pretrained backbone wrapper for baselines
# ══════════════════════════════════════════════════════════════════════

class PretrainedBackbone(nn.Module):
    """Frozen pretrained backbone that returns CLS features.

    Shared by all pretrained baselines for fair comparison.
    """

    def __init__(self, backbone_name: str = "deit_tiny_patch16_224"):
        super().__init__()
        self.model = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
        )
        self.embed_dim = self.model.num_features

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

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
    """Simple linear head on top of pretrained features."""

    def __init__(self, embed_dim: int, num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)  # [B, C]


class PretrainedEnsemble(nn.Module):
    """Deep ensemble: M independent heads on shared frozen backbone.

    Each member has its own randomly-initialised head.
    Total trainable params ≈ M × head_params.
    """

    def __init__(self, backbone: PretrainedBackbone,
                 num_members: int = 5, num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.num_members = num_members
        self.heads = nn.ModuleList([
            PretrainedClassifierHead(backbone.embed_dim, num_classes, dropout)
            for _ in range(num_members)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        logits_list = [h(features) for h in self.heads]
        logits_stack = torch.stack(logits_list, dim=0)  # [M, B, C]
        import torch.nn.functional as F
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
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.mc_samples = mc_samples
        self.head = PretrainedClassifierHead(
            backbone.embed_dim, num_classes, dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        import torch.nn.functional as F
        features = self.backbone(x)

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
                nn.Linear(ehid, num_classes),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        import torch.nn.functional as F
        features = self.backbone(x)

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
