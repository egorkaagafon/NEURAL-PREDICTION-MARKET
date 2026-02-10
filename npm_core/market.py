"""
Market aggregation — the «exchange floor» that turns individual agent
predictions + bets + capital into a single ensemble prediction.

The market clearing price:

    P(y | x) = Σ_i  C_i · b_i(x) · p_i(y | x)
               ─────────────────────────────────
                      Σ_i  C_i · b_i(x)

This is NOT a simple average — it is a capital‑weighted, bet‑weighted
combination where agents with more accumulated capital and higher
current confidence have proportionally more influence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MarketAggregator:
    """Stateless helper that computes the market clearing price and
    individual agent payoffs.

    All methods are pure functions (no learnable parameters).
    """

    # ------------------------------------------------------------------
    @staticmethod
    def clearing_price(
        probs: torch.Tensor,     # [K, B, C]
        bets: torch.Tensor,      # [K, B]
        capital: torch.Tensor,   # [K]
    ) -> torch.Tensor:
        """Compute aggregate market prediction.

        Returns
        -------
        market_probs : Tensor [B, C]
        """
        K, B, C = probs.shape

        # Expand capital to [K, B, 1]
        cap = capital.view(K, 1, 1).expand(K, B, 1)
        bet = bets.unsqueeze(-1)            # [K, B, 1]

        weights = cap * bet                 # [K, B, 1]
        weighted = (weights * probs).sum(0) # [B, C]
        denom = weights.sum(0).clamp(min=1e-8)  # [B, 1]

        market_probs = weighted / denom     # [B, C]
        return market_probs

    # ------------------------------------------------------------------
    @staticmethod
    def compute_market_loss(
        market_probs: torch.Tensor,   # [B, C]
        targets: torch.Tensor,        # [B]  (long)
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Cross‑entropy loss on the market prediction (used for backprop
        through the backbone + agent weights)."""

        log_probs = torch.log(market_probs.clamp(min=1e-8))
        if label_smoothing > 0:
            C = market_probs.shape[-1]
            one_hot = F.one_hot(targets, C).float()
            smooth = one_hot * (1 - label_smoothing) + label_smoothing / C
            loss = -(smooth * log_probs).sum(dim=-1).mean()
        else:
            loss = F.nll_loss(log_probs, targets)
        return loss

    # ------------------------------------------------------------------
    @staticmethod
    def agent_payoffs(
        probs: torch.Tensor,          # [K, B, C]
        targets: torch.Tensor,        # [B]
        bets: torch.Tensor = None,    # [K, B]  (optional)
    ) -> torch.Tensor:
        """Per‑agent bet‑weighted payoff (used for capital updates, NOT backprop).

        R_i = mean_over_batch [ b_i · log p_i(y_true) + (1 - b_i) · 0 ]

        If an agent bets b_i≈0 it neither gains nor loses capital
        ("safe haven" / cash position).  Only the fraction of capital
        that is actually staked is at risk.

        Without bets (backward compat): R_i = mean log p_i(y_true).

        Returns
        -------
        payoffs : Tensor [K]  — mean reward per agent for this batch.
        """
        K, B, C = probs.shape
        # Gather true‑class probabilities: [K, B]
        idx = targets.unsqueeze(0).expand(K, B).unsqueeze(-1)  # [K, B, 1]
        p_true = probs.gather(dim=2, index=idx).squeeze(-1)    # [K, B]
        log_p = torch.log(p_true.clamp(min=1e-8))              # [K, B]

        if bets is not None:
            # Bet‑weighted: staked fraction earns/loses, unstaked is safe
            log_p = bets * log_p   # [K, B]  — b_i≈0 → payoff≈0

        payoffs = log_p.mean(dim=1)                             # [K]
        return payoffs

    # ------------------------------------------------------------------
    @staticmethod
    def diversity_loss(
        probs: torch.Tensor,   # [K, B, C]
    ) -> torch.Tensor:
        """Encourage agents to maintain diverse hypotheses.

        Penalise when all agents agree too much:
            L_div = - mean pairwise KL divergence

        Maximising diversity ≈ minimising herding risk.
        """
        K, B, C = probs.shape
        # Mean KL between every pair (i, j)
        # For efficiency, we compare each agent to the ensemble mean.
        mean_probs = probs.mean(dim=0, keepdim=True)  # [1, B, C]
        # KL(p_i || mean)
        kl = (probs * (probs.clamp(min=1e-8).log() - mean_probs.clamp(min=1e-8).log()))
        kl = kl.sum(dim=-1).mean()  # scalar
        # We want to maximise KL → so loss = -KL
        return -kl
