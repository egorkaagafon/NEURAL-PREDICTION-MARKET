"""
Capital dynamics, bankruptcy detection and evolutionary selection.

Capital is tracked in **log‑space** with a decayed accumulator:

    S_i  ←  γ · S_i  +  η · z_i          (once per epoch)
    C_i  =  exp(S_i)                       (always positive)

where z_i is the z‑score of agent i's mean epoch payoff and γ < 1 is
the decay that makes the system mean‑reverting.  This eliminates the
absorbing‑barrier problem of multiplicative capital with hard clamping
(where agents hit min/max and stay there permanently).

Steady‑state capital ratio ≈ exp(η · 3 / (1 − γ)) between best and
worst agent.  With η=0.1, γ=0.9 → ratio ≈ exp(3) ≈ 20×.

The log‑space representation also means no agent ever has zero capital
— even the worst performer retains a voice in the market clearing price.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class CapitalManager:
    """Manages the capital vector for K agents using decayed log‑capital.

    Parameters
    ----------
    num_agents : int
    lr : float
        Step size η for log‑capital updates (per epoch).
    decay : float
        Decay factor γ for mean‑reverting dynamics (0.9 = 10‑epoch window).
    normalize_payoffs : bool
        Whether to z‑score normalise epoch payoffs before updating.
    device : str or torch.device
    """

    def __init__(
        self,
        num_agents: int,
        initial_capital: float = 1.0,
        lr: float = 0.1,
        decay: float = 0.9,
        normalize_payoffs: bool = True,
        device: str = "cpu",
        # Legacy kwargs (ignored but accepted for backward compat)
        ema: float = 0.95,
        min_capital: float = 0.01,
        max_capital: float = 10.0,
    ):
        self.num_agents = num_agents
        self.lr = lr
        self.decay = decay
        self.normalize_payoffs = normalize_payoffs
        self._device = device

        # Core state: log‑capital (additive, unbounded, mean‑reverting)
        self.log_capital = torch.zeros(
            num_agents, dtype=torch.float32, device=device
        )

        # Epoch accumulator: collect batch payoffs, average at epoch end
        self._epoch_payoffs: list[torch.Tensor] = []

        # History for analysis / plotting
        self.history: list[torch.Tensor] = []

        # Diagnostic: last payoff stats
        self._last_payoff_raw: Optional[torch.Tensor] = None
        self._last_payoff_norm: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def accumulate(self, payoffs: torch.Tensor):
        """Store per‑batch payoffs for epoch‑level averaging.

        Call this once per training batch.

        Parameters
        ----------
        payoffs : Tensor [K]
            Per‑agent mean reward for the current batch.
        """
        self._last_payoff_raw = payoffs.clone()
        self._epoch_payoffs.append(payoffs.cpu())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self):
        """Epoch‑level log‑capital update.

        Computes mean payoff over all accumulated batches, z‑normalises,
        and applies the decayed update:

            S_i = γ · S_i + η · z_i

        Call this once at the end of each epoch, after all accumulate()
        calls.
        """
        if not self._epoch_payoffs:
            return

        mean_payoff = torch.stack(self._epoch_payoffs).mean(dim=0)
        mean_payoff = mean_payoff.to(self.log_capital.device)

        # ── Payoff standardisation ────────────────────────────────────
        if self.normalize_payoffs:
            mu = mean_payoff.mean()
            sigma = mean_payoff.std()
            if sigma > 1e-12:
                z = ((mean_payoff - mu) / sigma).clamp(-3.0, 3.0)
            else:
                z = torch.zeros_like(mean_payoff)
        else:
            z = mean_payoff

        self._last_payoff_norm = z.clone()

        # Decayed update in log‑space (mean‑reverting dynamics)
        self.log_capital = self.decay * self.log_capital + self.lr * z

        # Snapshot for analysis
        self.history.append(self.get_capital().clone().cpu())

        # Reset accumulator
        self._epoch_payoffs = []

    # ------------------------------------------------------------------
    # Backward‑compatible alias: update() acts as accumulate() so old
    # code that calls update() per batch still works (step() must still
    # be called at epoch end).
    def update(self, payoffs: torch.Tensor):
        """Alias for accumulate() — backward compatible."""
        self.accumulate(payoffs)

    # ------------------------------------------------------------------
    def get_capital(self) -> torch.Tensor:
        """Return current capital vector [K] = exp(log_capital)."""
        return torch.exp(self.log_capital)

    def get_raw_capital(self) -> torch.Tensor:
        """Alias for get_capital()."""
        return self.get_capital()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def bankruptcy_mask(self) -> torch.Tensor:
        """Boolean mask [K]: True where agent has very low capital."""
        return self.get_capital() < 0.1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evolutionary_step(
        self,
        agent_pool: nn.ModuleList,
        kill_fraction: float = 0.1,
        mutation_std: float = 0.02,
    ) -> list[int]:
        """Replace the worst‑performing agents with mutated copies of the best.

        This implements replicator–mutator dynamics:
          1. Sort agents by log_capital.
          2. Kill bottom `kill_fraction` %.
          3. Clone top `kill_fraction` % into their slots.
          4. Add Gaussian noise to cloned weights (exploration).
          5. Reset log_capital of new agents to 0 (= capital 1.0).

        Returns
        -------
        list[int]
            Indices of the agents that were replaced (for optimizer state reset).
        """
        K = self.num_agents
        n_kill = max(1, int(K * kill_fraction))

        # Rank agents by log_capital (ascending)
        ranking = torch.argsort(self.log_capital)
        worst_idx = ranking[:n_kill].tolist()
        best_idx = ranking[-n_kill:].tolist()

        for w, b in zip(worst_idx, best_idx):
            # Clone best → worst slot
            agent_pool[w].load_state_dict(
                copy.deepcopy(agent_pool[b].state_dict())
            )
            # Mutate: add noise to all parameters
            with torch.no_grad():
                for param in agent_pool[w].parameters():
                    param.add_(torch.randn_like(param) * mutation_std)

            # Reset log_capital to 0 (= capital 1.0, fresh start)
            self.log_capital[w] = 0.0

        return worst_idx

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "log_capital": self.log_capital.cpu(),
        }

    def load_state_dict(self, d: dict):
        if "log_capital" in d:
            self.log_capital = d["log_capital"].to(self.log_capital.device)
        elif "capital" in d:
            # Backward compat: convert old multiplicative capital to log
            old_cap = d["capital"].float().clamp(min=1e-8)
            self.log_capital = torch.log(old_cap).to(self.log_capital.device)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Diagnostic snapshot."""
        c = self.get_capital()
        info = {
            "mean": c.mean().item(),
            "std": c.std().item(),
            "min": c.min().item(),
            "max": c.max().item(),
            "gini": _gini(c).item(),
            "num_bankrupt": int(self.bankruptcy_mask().sum().item()),
            "log_capital_spread": (self.log_capital.max() - self.log_capital.min()).item(),
        }
        # Include payoff diagnostics if available
        if self._last_payoff_raw is not None:
            r = self._last_payoff_raw
            info["payoff_raw_mean"] = r.mean().item()
            info["payoff_raw_std"] = r.std().item()
            info["payoff_raw_spread"] = (r.max() - r.min()).item()
        if self._last_payoff_norm is not None:
            n = self._last_payoff_norm
            info["payoff_norm_spread"] = (n.max() - n.min()).item()
        return info


# ── Utility ──────────────────────────────────────────────────────────
def _gini(x: torch.Tensor) -> torch.Tensor:
    """Gini coefficient — measures capital concentration.
    0 = perfectly equal, 1 = one agent owns everything."""
    x = x.float()
    if x.sum() == 0:
        return torch.tensor(0.0)
    sorted_x, _ = torch.sort(x)
    n = x.numel()
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return (2.0 * (index * sorted_x).sum() / (n * sorted_x.sum())) - (n + 1) / n
