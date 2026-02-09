"""
Visualization utilities for NPM experiments.

Generates publication‑quality plots for:
  - Capital dynamics over training
  - Market uncertainty heatmaps
  - OOD detection histograms
  - Risk–coverage curves
  - Agent specialisation analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_capital_history(
    capital_history: List[torch.Tensor],
    save_path: Optional[str] = None,
):
    """Plot capital trajectories of all agents over training steps.

    Parameters
    ----------
    capital_history : list of Tensor [K]
        From CapitalManager.history.
    """
    data = torch.stack(capital_history).numpy()  # [T, K]
    T, K = data.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(K):
        ax.plot(data[:, i], alpha=0.6, linewidth=1.0, label=f"Agent {i}")
    ax.set_xlabel("Training Step (batch)")
    ax.set_ylabel("Capital")
    ax.set_title("Capital Dynamics: Agent Trajectories")
    ax.set_yscale("log")
    if K <= 16:
        ax.legend(fontsize=7, ncol=4, loc="upper right")
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_capital_gini(
    capital_history: List[torch.Tensor],
    save_path: Optional[str] = None,
):
    """Plot Gini coefficient of capital over time."""
    from npm_core.capital import _gini
    ginis = [_gini(c).item() for c in capital_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ginis, color="crimson", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Capital Concentration Over Training")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="High concentration")
    ax.legend()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_ood_histograms(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    score_name: str = "Epistemic Uncertainty",
    ood_name: str = "CIFAR-100",
    save_path: Optional[str] = None,
):
    """Side‑by‑side histograms of uncertainty scores for ID vs OOD."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(id_scores, bins=60, alpha=0.6, density=True, label="ID (CIFAR-10)",
            color="steelblue")
    ax.hist(ood_scores, bins=60, alpha=0.6, density=True, label=f"OOD ({ood_name})",
            color="coral")
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.set_title(f"OOD Detection via {score_name}")
    ax.legend()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_risk_coverage(
    curves: Dict[str, dict],
    save_path: Optional[str] = None,
):
    """Risk–coverage curves for multiple methods.

    Parameters
    ----------
    curves : dict { method_name: {"coverage": array, "risk": array, "aurc": float} }
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in curves.items():
        label = f"{name} (AURC={data['aurc']:.4f})"
        ax.plot(data["coverage"], data["risk"], linewidth=1.5, label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 - Accuracy)")
    ax.set_title("Selective Prediction: Risk vs Coverage")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_agent_specialisation(
    model,
    test_loader,
    capital: torch.Tensor,
    device: torch.device,
    num_classes: int = 10,
    save_path: Optional[str] = None,
):
    """Heatmap: which agents specialise in which classes.

    For each agent, compute mean bet × probability for each class
    across the test set.
    """
    model.eval()
    K = model.num_agents
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    # Accumulate: [K, C] — mean (bet_i × p_i(c)) weighted by target
    spec = torch.zeros(K, num_classes)
    counts = torch.zeros(num_classes)

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            out = model(images, capital)
            probs = out["all_probs"].cpu()   # [K, B, C]
            bets = out["all_bets"].cpu()     # [K, B]

            for c in range(num_classes):
                mask = (targets == c)
                if mask.sum() == 0:
                    continue
                # Average bet × prob for class c when target IS c
                bp = bets[:, mask].unsqueeze(-1) * probs[:, mask, :]  # [K, n, C]
                spec[:, c] += bp[:, :, c].mean(dim=1)
                counts[c] += 1

    spec = spec / counts.unsqueeze(0).clamp(min=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        spec.numpy(), ax=ax, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=[f"Agent {i}" for i in range(K)],
        cmap="YlOrRd",
    )
    ax.set_title("Agent Specialisation: Mean Bet × P(correct class)")
    ax.set_xlabel("True Class")
    ax.set_ylabel("Agent")
    if save_path:
        fig.savefig(save_path)
    plt.show()


def plot_ablation_bars(
    results: Dict[str, dict],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
):
    """Bar chart comparing ablation variants."""
    names = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(names))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation Study — {metric}")
    ax.set_xticklabels(names, rotation=30, ha="right")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", fontsize=9)

    if save_path:
        fig.savefig(save_path)
    plt.show()
