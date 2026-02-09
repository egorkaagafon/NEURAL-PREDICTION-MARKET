# Neural Prediction Market (NPM)

> **Endogenous hypothesis selection via betting-based credit assignment**

A novel neural network architecture where internal sub-networks ("agents") compete in a prediction market. Instead of centralized routing (MoE) or random averaging (ensemble), agents stake learned capital on their predictions—and the market dynamics produce interpretable uncertainty signals for free.

## Key Ideas

| Concept | Standard NN | NPM |
|---------|------------|-----|
| Aggregation | Average / learned router | Capital-weighted market clearing price |
| Confidence | = Probability output | = Bet size (how much capital to risk) |
| Regularisation | Dropout / weight decay | Bankruptcy: bad agents die, good ones reproduce |
| Uncertainty | Entropy of output | Market signals: liquidity, herding, volatility |
| Credit assignment | Backprop only | Backprop (weights) + replicator dynamics (capital) |

## Architecture

```
Input Image → ViT Backbone → Shared Features [B, D]
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
               Agent 0          Agent 1    ...  Agent K-1
            ┌──────────┐    ┌──────────┐    ┌──────────┐
            │ Classifier│    │ Classifier│    │ Classifier│
            │ → p_i(y|x)│   │ → p_i(y|x)│   │ → p_i(y|x)│
            │ Bet Head  │    │ Bet Head  │    │ Bet Head  │
            │ → b_i(x)  │    │ → b_i(x)  │    │ → b_i(x)  │
            └──────────┘    └──────────┘    └──────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    Market Aggregator (capital-weighted)
                    P(y|x) = Σ C_i·b_i·p_i / Σ C_i·b_i
                                    │
                                    ▼
                            Final Prediction
```

## Two-Track Optimisation

```
Track 1 (Backprop):   ∇_θ L_market(P(y|x), y_true)  →  update weights
Track 2 (Capital):    C_i ← C_i · exp(η · log p_i(y_true))  →  replicator dynamics
Track 3 (Evolution):  Kill bottom α%, clone top α% + noise  →  regularisation
```

## Uncertainty Signals

- **Liquidity** `Σ C_i·b_i(x)` — low = nobody wants to bet = epistemic uncertainty
- **Herding** (Gini of agent influence) — high concentration + wrong = systematic bias
- **Volatility** (JS divergence under perturbation) — high = unstable decision boundary

## Project Structure

```
├── npm_core/
│   ├── agents.py          # AgentHead, AgentPool — independent traders
│   ├── market.py          # MarketAggregator — clearing price, payoffs, diversity
│   ├── capital.py         # CapitalManager — log-wealth updates, bankruptcy, evolution
│   └── uncertainty.py     # Liquidity, herding, volatility metrics
├── models/
│   ├── vit_npm.py         # ViT backbone + NPM model
│   └── baselines.py       # Deep Ensemble, MC-Dropout, MoE
├── experiments/
│   ├── run_phase1.py      # Sanity check: NPM vs baselines on CIFAR-10
│   ├── run_phase2.py      # OOD detection: CIFAR-100, SVHN
│   └── run_phase3.py      # Ablation: what happens without market components
├── configs/
│   └── default.yaml       # All hyperparameters
├── train.py               # Main training loop
├── evaluate.py            # Full evaluation suite (accuracy, NLL, ECE, OOD, selective)
├── visualize.py           # Publication-quality plots
├── data_utils.py          # CIFAR-10/100, SVHN loaders
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train NPM on CIFAR-10
python train.py --config configs/default.yaml --device cuda

# Compare with baselines (Phase 1)
python experiments/run_phase1.py --epochs 100

# OOD detection (Phase 2, needs checkpoint)
python experiments/run_phase2.py --checkpoint runs/<timestamp>/ckpt_epoch200.pt

# Ablation study (Phase 3)
python experiments/run_phase3.py --epochs 100
```

## Research Plan

### Phase 1 — Sanity Check
- CIFAR-10 accuracy, NLL, ECE, Brier, AURC
- Compare: NPM vs Deep Ensemble vs MC-Dropout vs MoE

### Phase 2 — Uncertainty & OOD
- OOD datasets: CIFAR-100, SVHN
- Market signals: liquidity, herding, volatility
- AUROC/AUPR comparison with entropy and mutual information

### Phase 3 — Ablation
| Variant | Capital | Bets | Evolution | Diversity |
|---------|---------|------|-----------|-----------|
| Full NPM | ✓ | ✓ | ✓ | ✓ |
| No evolution | ✓ | ✓ | ✗ | ✓ |
| No capital | ✗ | ✓ | ✗ | ✓ |
| No bets | ✓ | ✗ | ✓ | ✓ |
| No diversity | ✓ | ✓ | ✓ | ✗ |

## Theoretical Grounding

- **Proper scoring rules**: Log-loss payoff ensures honest prediction is the dominant strategy (Kelly criterion)
- **Replicator dynamics**: Capital updates implement multiplicative weights → converges to best hypothesis
- **Evolutionary regularisation**: Bankruptcy + cloning = replicator-mutator equation (not random dropout)
- **Market efficiency**: Clearing price aggregates information efficiently (Hayek, 1945; Hanson, 2003)

## Defending Against Reviewers

| Objection | Response |
|-----------|----------|
| "This is just MoE" | No router. No centralised optimisation. Endogenous weighting. Replicator dynamics ≠ gating. |
| "Training instability" | Capital lives outside backprop. Bounded bets. Log-wealth EMA smoothing. |
| "Why not ensemble?" | Ensemble doesn't explain *why*. No market signals. No endogenous adaptation. |
| "Too many hyperparameters" | Ablation study shows each component matters. Core math is clean (Kelly + replicator). |

## Citation

```bibtex
@article{npm2026,
  title={Neural Prediction Market: Endogenous Hypothesis Selection via Betting-Based Credit Assignment},
  year={2026}
}
```