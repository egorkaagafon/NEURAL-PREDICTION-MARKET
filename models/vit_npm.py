"""
Vision Transformer backbone + Neural Prediction Market head.

Architecture summary:
────────────────────
  Input image  (3 × 32 × 32  for CIFAR‑10)
       │
  ┌────▼─────────────────┐
  │  Patch Embedding      │  (patch_size=4  → 8×8 = 64 patches)
  └────┬─────────────────┘
       │  + CLS token + positional encoding
  ┌────▼─────────────────┐
  │  N × Transformer     │  standard multi‑head self‑attention
  │      Encoder Blocks  │
  └────┬─────────────────┘
       │  CLS token features  [B, D]
  ┌────▼─────────────────┐
  │  AgentPool            │  K independent «traders»
  │  (classifier + bet)   │  each outputs [logits, probs, bet]
  └────┬─────────────────┘
       │
  ┌────▼─────────────────┐
  │  MarketAggregator     │  capital‑weighted clearing price
  └──────────────────────┘
       ↓
    P(y | x)  — ensemble prediction
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from npm_core.agents import AgentPool
from npm_core.market import MarketAggregator


# ══════════════════════════════════════════════════════════════════════
#  Simple ViT backbone (small, suitable for CIFAR-scale experiments)
# ══════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] → [B, num_patches, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 3, H, W]

        Returns
        -------
        features : Tensor [B, D]   (CLS token output)
        """
        B = x.shape[0]
        patches = self.patch_embed(x)                      # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)             # [B, 1, D]
        x = torch.cat([cls, patches], dim=1)               # [B, N+1, D]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]                                     # CLS token [B, D]


# ══════════════════════════════════════════════════════════════════════
#  Full NPM model
# ══════════════════════════════════════════════════════════════════════

class NeuralPredictionMarket(nn.Module):
    """Complete NPM model: ViT backbone → Agent Pool → Market Aggregator.

    Forward pass returns everything needed for both backprop and
    capital updates.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        num_agents: int = 16,
        num_classes: int = 10,
        dropout: float = 0.1,
        bet_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_classes = num_classes

        self.backbone = ViTBackbone(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.agent_pool = AgentPool(
            num_agents=num_agents,
            in_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            bet_temperature=bet_temperature,
        )

        self.market = MarketAggregator()

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,          # [B, 3, H, W]
        capital: Optional[torch.Tensor] = None,  # [K]
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            features     : [B, D]
            all_logits   : [K, B, C]
            all_probs    : [K, B, C]
            all_bets     : [K, B]
            market_probs : [B, C]      (clearing price)
        """
        features = self.backbone(images)                     # [B, D]

        all_logits, all_probs, all_bets = self.agent_pool(features)
        # all_logits: [K, B, C], all_probs: [K, B, C], all_bets: [K, B]

        if capital is None:
            capital = torch.ones(self.num_agents, device=images.device)
        else:
            capital = capital.detach().clone()  # never backprop through capital

        market_probs = self.market.clearing_price(
            all_probs, all_bets, capital,
        )

        return {
            "features":     features,
            "all_logits":   all_logits,
            "all_probs":    all_probs,
            "all_bets":     all_bets,
            "market_probs": market_probs,
        }

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
