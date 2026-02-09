"""
Agent heads — independent «traders» that produce class predictions + bet sizes.

Each agent is a small classifier head that receives shared features from a
backbone and independently outputs:
  - logits  z_i(x)  →  class probabilities  p_i(y|x)
  - bet     b_i(x)  ∈ (0, 1]  — how much capital it is willing to risk

Design note:
  Confidence ≠ Probability.  An agent may output a peaked distribution
  (high probability on one class) but still bet conservatively if it has
  learned that this region of feature space is unreliable.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentHead(nn.Module):
    """Single trader / classifier head.

    Parameters
    ----------
    in_dim : int
        Dimensionality of backbone features.
    num_classes : int
        Number of output classes (e.g. 10 for CIFAR-10).
    hidden_dim : int | None
        If given, add a hidden layer before the logit projection.
    dropout : float
        Dropout probability in hidden layer.
    bet_temperature : float
        Controls sharpness of the learned bet size.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        bet_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bet_temperature = bet_temperature

        hid = hidden_dim or in_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_classes),
        )

        # Bet head — scalar that controls how much capital this agent risks.
        # Learned independently from the classifier logits.
        self.bet_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid // 2, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor):
        """
        Parameters
        ----------
        features : Tensor [B, D]
            Shared backbone features.

        Returns
        -------
        logits : Tensor [B, C]
        probs  : Tensor [B, C]   — softmax(logits)
        bet    : Tensor [B]      — ∈ (0, 1]
        """
        logits = self.classifier(features)                    # [B, C]
        probs = F.softmax(logits, dim=-1)                     # [B, C]

        raw_bet = self.bet_head(features).squeeze(-1)         # [B]
        # Sigmoid + small ε to keep bet in (ε, 1]
        bet = torch.sigmoid(raw_bet / self.bet_temperature)   # [B]
        bet = bet.clamp(min=1e-6)

        return logits, probs, bet


class AgentPool(nn.Module):
    """A pool of K independent agents sharing a backbone feature space.

    Parameters
    ----------
    num_agents : int
    in_dim     : int
    num_classes: int
    dropout    : float
    bet_temperature : float
    """

    def __init__(
        self,
        num_agents: int,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        bet_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.agents = nn.ModuleList([
            AgentHead(
                in_dim=in_dim,
                num_classes=num_classes,
                dropout=dropout,
                bet_temperature=bet_temperature,
            )
            for _ in range(num_agents)
        ])

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor):
        """
        Parameters
        ----------
        features : Tensor [B, D]

        Returns
        -------
        all_logits : Tensor [K, B, C]
        all_probs  : Tensor [K, B, C]
        all_bets   : Tensor [K, B]
        """
        all_logits, all_probs, all_bets = [], [], []
        for agent in self.agents:
            logits, probs, bet = agent(features)
            all_logits.append(logits)
            all_probs.append(probs)
            all_bets.append(bet)

        return (
            torch.stack(all_logits, dim=0),   # [K, B, C]
            torch.stack(all_probs, dim=0),     # [K, B, C]
            torch.stack(all_bets, dim=0),      # [K, B]
        )
