"""
Capital dynamics, bankruptcy detection and evolutionary selection.

Capital lives *outside* the computational graph — it is a state variable
updated after each batch via multiplicative log‑wealth updates:

    C_i  ←  C_i · exp(η · R_i)

where R_i = log p_i(y_true)  (proper scoring rule / Kelly criterion).

This separation is critical:
  • Weights are trained with backprop (gradient descent on market loss).
  • Capital is updated with replicator dynamics (reward signal).
  • The two interact via the market clearing price (forward pass).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class CapitalManager:
    """Manages the capital vector for K agents.

    Parameters
    ----------
    num_agents : int
    initial_capital : float
    lr : float
        Step size η for multiplicative updates.
    ema : float
        Exponential moving average smoothing for capital.
    min_capital : float
        Bankruptcy threshold.
    max_capital : float
        Cap to prevent any single agent from dominating.
    device : str or torch.device
    """

    def __init__(
        self,
        num_agents: int,
        initial_capital: float = 1.0,
        lr: float = 0.02,
        ema: float = 0.999,
        min_capital: float = 1e-4,
        max_capital: float = 100.0,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.lr = lr
        self.ema = ema
        self.min_capital = min_capital
        self.max_capital = max_capital

        self.capital = torch.full(
            (num_agents,), initial_capital, dtype=torch.float32, device=device
        )
        # Running EMA for smoother dynamics
        self.capital_ema = self.capital.clone()
        # History for analysis / plotting
        self.history: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, payoffs: torch.Tensor):
        """Multiplicative log‑wealth update.

        Parameters
        ----------
        payoffs : Tensor [K]
            Per‑agent mean log p_i(y_true) for the current batch.
        """
        # Multiplicative update: C_i *= exp(η * R_i)
        self.capital = self.capital * torch.exp(self.lr * payoffs)

        # Clamp to [min, max]
        self.capital.clamp_(min=self.min_capital, max=self.max_capital)

        # EMA smoothing
        self.capital_ema = (
            self.ema * self.capital_ema + (1 - self.ema) * self.capital
        )

        # Snapshot for later analysis
        self.history.append(self.capital.clone().cpu())

    # ------------------------------------------------------------------
    def get_capital(self) -> torch.Tensor:
        """Return current (smoothed) capital vector [K]."""
        return self.capital_ema

    # ------------------------------------------------------------------
    def get_raw_capital(self) -> torch.Tensor:
        return self.capital

    # ------------------------------------------------------------------
    @torch.no_grad()
    def bankruptcy_mask(self) -> torch.Tensor:
        """Boolean mask [K]: True where agent is bankrupt."""
        return self.capital <= self.min_capital * 1.5

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
          1. Sort agents by capital.
          2. Kill bottom `kill_fraction` %.
          3. Clone top `kill_fraction` % into their slots.
          4. Add Gaussian noise to cloned weights (exploration).
          5. Reset capital of new agents to median capital.

        Parameters
        ----------
        agent_pool : nn.ModuleList
            The list of AgentHead modules to mutate in‑place.
        kill_fraction : float
            Fraction of agents to replace (e.g. 0.1 = bottom 10 %).
        mutation_std : float
            Standard deviation of Gaussian noise added to cloned weights.

        Returns
        -------
        list[int]
            Indices of the agents that were replaced (for optimizer state reset).
        """
        K = self.num_agents
        n_kill = max(1, int(K * kill_fraction))

        # Rank agents by capital (ascending)
        ranking = torch.argsort(self.capital)
        worst_idx = ranking[:n_kill].tolist()
        best_idx = ranking[-n_kill:].tolist()

        median_capital = self.capital.median().item()

        for w, b in zip(worst_idx, best_idx):
            # Clone best → worst slot
            agent_pool[w].load_state_dict(
                copy.deepcopy(agent_pool[b].state_dict())
            )
            # Mutate: add noise to all parameters
            with torch.no_grad():
                for param in agent_pool[w].parameters():
                    param.add_(torch.randn_like(param) * mutation_std)

            # Reset capital to median (fresh start, not zero)
            self.capital[w] = median_capital
            self.capital_ema[w] = median_capital

        return worst_idx

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "capital": self.capital.cpu(),
            "capital_ema": self.capital_ema.cpu(),
        }

    def load_state_dict(self, d: dict):
        self.capital = d["capital"].to(self.capital.device)
        self.capital_ema = d["capital_ema"].to(self.capital_ema.device)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Diagnostic snapshot."""
        c = self.capital
        return {
            "mean": c.mean().item(),
            "std": c.std().item(),
            "min": c.min().item(),
            "max": c.max().item(),
            "gini": _gini(c).item(),
            "num_bankrupt": int(self.bankruptcy_mask().sum().item()),
        }


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
