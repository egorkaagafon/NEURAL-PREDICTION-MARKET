"""
Baselines for fair comparison:
  1. Deep Ensemble (Lakshminarayanan et al., 2017)
  2. MC-Dropout (Gal & Ghahramani, 2016)
  3. Mixture of Experts with learned router (MoE)

All baselines are parameter-matched to NPM (~7.5M params) for fair
comparison.  Each model does exactly 1 backbone forward pass per
training step.  Multi-sample inference (MC-Dropout, Ensemble averaging)
happens only at eval time.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_npm import ViTBackbone


# ══════════════════════════════════════════════════════════════════════
#  1.  Standard ViT classifier (used inside Ensemble & MC‑Dropout)
# ══════════════════════════════════════════════════════════════════════

class ViTClassifier(nn.Module):
    """Simple ViT + linear head (single model)."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = ViTBackbone(
            image_size=image_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# ══════════════════════════════════════════════════════════════════════
#  2.  Deep Ensemble
# ══════════════════════════════════════════════════════════════════════

class DeepEnsemble(nn.Module):
    """Collection of M independently‑initialised ViT classifiers.

    Parameter‑matched: each member uses a smaller backbone so total
    parameter count ≈ 7.5M (same as NPM).

    Training:  Each member runs its own forward + backward independently
               (1× backbone FLOPS per member per step — but members are
               trained in a round‑robin loop, so total wall‑clock is M×).

    Inference: All members run, predictions averaged.
    """

    def __init__(self, num_members: int = 5, **vit_kwargs):
        super().__init__()
        self.num_members = num_members
        self.members = nn.ModuleList([
            ViTClassifier(**vit_kwargs)
            for _ in range(num_members)
        ])

    def forward(self, x: torch.Tensor) -> dict:
        logits_list = [m(x) for m in self.members]
        logits_stack = torch.stack(logits_list, dim=0)       # [M, B, C]
        probs_stack = F.softmax(logits_stack, dim=-1)        # [M, B, C]
        mean_probs = probs_stack.mean(dim=0)                 # [B, C]
        return {
            "market_probs": mean_probs,   # compatible interface
            "all_probs": probs_stack,
        }

    def forward_member(self, x: torch.Tensor, member_idx: int) -> dict:
        """Forward a single member (used during training for proper
        independent gradient updates)."""
        logits = self.members[member_idx](x)
        probs = F.softmax(logits, dim=-1)
        return {
            "market_probs": probs,
            "all_probs": probs.unsqueeze(0),
        }


# ══════════════════════════════════════════════════════════════════════
#  3.  MC‑Dropout
# ══════════════════════════════════════════════════════════════════════

class MCDropoutViT(nn.Module):
    """ViT with dropout kept ON at inference for Monte‑Carlo sampling.

    TRAINING:  Standard single forward pass (dropout is part of normal
               training anyway).  1× backbone FLOPS.

    INFERENCE: M stochastic forward passes with dropout ON, averaged.
               This is the standard Gal & Ghahramani (2016) protocol.
    """

    def __init__(self, mc_samples: int = 10, **vit_kwargs):
        super().__init__()
        self.mc_samples = mc_samples
        self.model = ViTClassifier(**vit_kwargs)

    def forward(self, x: torch.Tensor) -> dict:
        if self.training:
            # Training: single forward pass, normal dropout
            logits = self.model(x)                             # [B, C]
            probs = F.softmax(logits, dim=-1)                  # [B, C]
            return {
                "market_probs": probs,
                "all_probs": probs.unsqueeze(0),               # [1, B, C]
            }

        # Inference: multiple stochastic forward passes
        self.model.train()   # enable dropout
        with torch.no_grad():
            logits_list = [self.model(x) for _ in range(self.mc_samples)]
        logits_stack = torch.stack(logits_list, dim=0)
        probs_stack = F.softmax(logits_stack, dim=-1)
        mean_probs = probs_stack.mean(dim=0)
        self.model.eval()
        return {
            "market_probs": mean_probs,
            "all_probs": probs_stack,
        }


# ══════════════════════════════════════════════════════════════════════
#  4.  Mixture of Experts (with learned router — «socialist MoE»)
# ══════════════════════════════════════════════════════════════════════

class MoEClassifier(nn.Module):
    """Standard top‑k MoE with a learned router.

    This is the 'socialist' baseline: a centralised controller decides
    which experts to use.  Parameter‑matched to NPM via expert_hidden_dim.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        num_classes: int = 10,
        num_experts: int = 16,
        top_k: int = 4,
        dropout: float = 0.1,
        expert_hidden_dim: int = 0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.backbone = ViTBackbone(
            image_size=image_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, dropout=dropout,
        )

        # Router: centralised gating
        self.router = nn.Linear(embed_dim, num_experts)

        # Expert heads (hidden dim configurable for param matching)
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
        features = self.backbone(x)                    # [B, D]

        # Router gating
        gate_logits = self.router(features)            # [B, E]
        gate_probs = F.softmax(gate_logits, dim=-1)    # [B, E]

        # Top-k selection
        topk_vals, topk_idx = gate_probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        B = features.shape[0]
        all_logits = torch.stack([e(features) for e in self.experts], dim=1)
        # [B, E, C]

        # Gather top-k expert outputs
        C = all_logits.shape[-1]
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        selected = all_logits.gather(1, idx_expanded)  # [B, k, C]

        # Weighted combination
        weighted = (topk_vals.unsqueeze(-1) * F.softmax(selected, dim=-1)).sum(1)

        return {
            "market_probs": weighted,
            "all_probs": F.softmax(all_logits.permute(1, 0, 2), dim=-1),
            "gate_probs": gate_probs,
        }
