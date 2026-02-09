"""
Market‑based uncertainty metrics.

This module extracts interpretable uncertainty signals from the internal
market dynamics of NPM.  Three families of signals:

1. **Liquidity** — Total capital× bet wagered on a sample.
   Low liquidity ≈ epistemic uncertainty (nobody wants to bet).

2. **Concentration / Herding** — Gini coefficient of agent influence.
   High concentration + wrong answer ≈ systematic bias / adversarial.

3. **Volatility** — Sensitivity of the market to input perturbations.
   High volatility ≈ decision boundary instability.

These are motivated by real‑world financial market analytics, not by
Bayesian inference, yet they capture similar phenomena:  liquidity ↔
posterior uncertainty, herding ↔ mode collapse, volatility ↔ gradient
norm.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  1.  Liquidity
# ══════════════════════════════════════════════════════════════════════

def market_liquidity(
    bets: torch.Tensor,       # [K, B]
    capital: torch.Tensor,    # [K]
) -> torch.Tensor:
    """Per‑sample total effective stake.

    L(x) = Σ_i  C_i · b_i(x)

    Returns
    -------
    liquidity : Tensor [B]
    """
    cap = capital.unsqueeze(1)            # [K, 1]
    return (cap * bets).sum(dim=0)        # [B]


def liquidity_uncertainty(
    bets: torch.Tensor,
    capital: torch.Tensor,
    running_mean: Optional[float] = None,
) -> torch.Tensor:
    """Normalised epistemic uncertainty score ∈ [0, 1].

    If running_mean of liquidity is provided, uncertainty =
       1 - clamp(L(x) / running_mean, 0, 1).

    Otherwise returns 1 / (1 + L(x))  — monotone decreasing in L.

    Returns
    -------
    uncertainty : Tensor [B]
    """
    L = market_liquidity(bets, capital)    # [B]
    if running_mean is not None and running_mean > 0:
        return (1.0 - (L / running_mean).clamp(0, 1))
    return 1.0 / (1.0 + L)


# ══════════════════════════════════════════════════════════════════════
#  2.  Concentration / Herding
# ══════════════════════════════════════════════════════════════════════

def agent_influence_weights(
    bets: torch.Tensor,       # [K, B]
    capital: torch.Tensor,    # [K]
) -> torch.Tensor:
    """Normalised influence of each agent on each sample.

    w_i(x) = C_i · b_i(x) / Σ_j C_j · b_j(x)

    Returns
    -------
    weights : Tensor [K, B]  — sums to 1 along dim 0.
    """
    cap = capital.unsqueeze(1)             # [K, 1]
    raw = cap * bets                       # [K, B]
    return raw / raw.sum(dim=0, keepdim=True).clamp(min=1e-8)


def gini_per_sample(weights: torch.Tensor) -> torch.Tensor:
    """Gini coefficient of agent influence for each sample.

    Parameters
    ----------
    weights : Tensor [K, B]

    Returns
    -------
    gini : Tensor [B]   ∈ [0, 1]  (0=equal, 1=single agent dominates)
    """
    K, B = weights.shape
    sorted_w, _ = weights.sort(dim=0)              # [K, B]
    index = torch.arange(1, K + 1, dtype=weights.dtype,
                         device=weights.device).unsqueeze(1)  # [K, 1]
    return (2.0 * (index * sorted_w).sum(0) / (K * sorted_w.sum(0).clamp(min=1e-8))) - (K + 1) / K


def herding_score(
    probs: torch.Tensor,                  # [K, B, C]
    bets: torch.Tensor,                   # [K, B]
    capital: torch.Tensor,                # [K]
) -> torch.Tensor:
    """Detects consensus‑failure: all agents agree AND bet heavily,
    yet the consensus could be wrong.

    Score = Gini(influence) × max_class_prob(market)

    High score + wrong prediction → herding / adversarial.

    Returns
    -------
    score : Tensor [B]
    """
    w = agent_influence_weights(bets, capital)       # [K, B]
    gini = gini_per_sample(w)                        # [B]

    # Market‑level certainty (without capital, just raw consensus)
    avg_probs = probs.mean(dim=0)                    # [B, C]
    max_conf = avg_probs.max(dim=-1).values          # [B]

    return gini * max_conf


# ══════════════════════════════════════════════════════════════════════
#  3.  Volatility (requires model forward pass)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_volatility(
    model,                    # NeuralPredictionMarket
    images: torch.Tensor,     # [B, 3, H, W]
    capital: torch.Tensor,    # [K]
    eps: float = 0.01,
    n_samples: int = 5,
) -> Dict[str, torch.Tensor]:
    """Assess decision stability under small Gaussian perturbations.

    For each perturbation ε·N(0,1):
      • recompute forward pass
      • compare market_probs, agent rankings

    Returns
    -------
    dict:
        prob_std       : [B, C]  — std of market_probs across perturbations
        mean_js_div    : [B]     — mean Jensen–Shannon divergence vs clean
        rank_kendall   : [B]     — mean Kendall‑τ change in agent ranking
        influence_gini_std : [B] — std of Gini coefficient across perturbations
    """
    device = images.device
    B = images.shape[0]

    # Clean forward
    out_clean = model(images, capital)
    p_clean = out_clean["market_probs"]          # [B, C]
    bets_clean = out_clean["all_bets"]           # [K, B]

    prob_samples = []
    gini_samples = []

    for _ in range(n_samples):
        noise = torch.randn_like(images) * eps
        out = model(images + noise, capital)
        prob_samples.append(out["market_probs"])

        w = agent_influence_weights(out["all_bets"], capital)
        gini_samples.append(gini_per_sample(w))

    probs_stack = torch.stack(prob_samples, dim=0)   # [S, B, C]
    ginis_stack = torch.stack(gini_samples, dim=0)   # [S, B]

    prob_std = probs_stack.std(dim=0)                # [B, C]

    # JS divergence between clean and each perturbed
    js_divs = []
    for ps in prob_samples:
        m = 0.5 * (p_clean + ps)
        kl1 = (p_clean * (p_clean.clamp(min=1e-8).log() - m.clamp(min=1e-8).log())).sum(-1)
        kl2 = (ps * (ps.clamp(min=1e-8).log() - m.clamp(min=1e-8).log())).sum(-1)
        js_divs.append(0.5 * (kl1 + kl2))
    mean_js = torch.stack(js_divs, dim=0).mean(dim=0)  # [B]

    gini_std = ginis_stack.std(dim=0)                   # [B]

    return {
        "prob_std":        prob_std,
        "mean_js_div":     mean_js,
        "influence_gini_std": gini_std,
    }


# ══════════════════════════════════════════════════════════════════════
#  Aggregate uncertainty report
# ══════════════════════════════════════════════════════════════════════

def uncertainty_report(
    probs: torch.Tensor,                  # [K, B, C]
    bets: torch.Tensor,                   # [K, B]
    capital: torch.Tensor,                # [K]
    running_liquidity_mean: float = None,
) -> Dict[str, torch.Tensor]:
    """One‑call function that returns all market uncertainty signals.

    Returns
    -------
    dict:
        liquidity       : [B]  — total effective stake (higher = more certain)
        epistemic_unc   : [B]  — normalised liquidity uncertainty
        herding         : [B]  — consensus + concentration score
        gini            : [B]  — capital concentration per sample
        entropy_market  : [B]  — Shannon entropy of market prediction
    """
    liq = market_liquidity(bets, capital)
    epi = liquidity_uncertainty(bets, capital, running_liquidity_mean)
    herd = herding_score(probs, bets, capital)

    w = agent_influence_weights(bets, capital)
    gini = gini_per_sample(w)

    # Market clearing price (approx — unweighted mean for speed)
    cap = capital.view(-1, 1, 1)
    bet = bets.unsqueeze(-1)
    ww = cap * bet
    mp = (ww * probs).sum(0) / ww.sum(0).clamp(min=1e-8)
    ent = -(mp * mp.clamp(min=1e-8).log()).sum(-1)

    return {
        "liquidity": liq,
        "epistemic_unc": epi,
        "herding": herd,
        "gini": gini,
        "entropy_market": ent,
    }
