"""
Baselines for fair comparison:
  1. Deep Ensemble (Lakshminarayanan et al., 2017)
  2. MC-Dropout (Gal & Ghahramani, 2016)
  3. Mixture of Experts with learned router (MoE)

All baselines share the same ViT backbone size as NPM for apples‑to‑apples.
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

    At inference, predictions are averaged.
    """

    def __init__(self, num_members: int = 5, **vit_kwargs):
        super().__init__()
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


# ══════════════════════════════════════════════════════════════════════
#  3.  MC‑Dropout
# ══════════════════════════════════════════════════════════════════════

class MCDropoutViT(nn.Module):
    """ViT with dropout kept ON at inference for Monte‑Carlo sampling."""

    def __init__(self, mc_samples: int = 10, **vit_kwargs):
        super().__init__()
        self.mc_samples = mc_samples
        self.model = ViTClassifier(**vit_kwargs)

    def forward(self, x: torch.Tensor) -> dict:
        # Keep dropout on
        self.model.train()
        logits_list = [self.model(x) for _ in range(self.mc_samples)]
        logits_stack = torch.stack(logits_list, dim=0)
        probs_stack = F.softmax(logits_stack, dim=-1)
        mean_probs = probs_stack.mean(dim=0)
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
    which experts to use.
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

        # Expert heads
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
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
