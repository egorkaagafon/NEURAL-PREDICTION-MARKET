"""
Phase 2 — Uncertainty & OOD Detection.

Uses a trained NPM checkpoint and compares market‑based uncertainty
signals against standard entropy / mutual information for OOD detection.

OOD datasets: CIFAR-100, SVHN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from data_utils import get_cifar10_loaders, get_ood_loader
from models.vit_npm import NeuralPredictionMarket
from npm_core.capital import CapitalManager
from npm_core.uncertainty import uncertainty_report, estimate_volatility
from evaluate import ood_detection_scores, ood_detection_volatility, volatility_analysis


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load NPM model + capital from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    mkt = cfg["market"]

    model = NeuralPredictionMarket(
        image_size=mc["image_size"], patch_size=mc["patch_size"],
        embed_dim=mc["embed_dim"], depth=mc["depth"],
        num_heads=mc["num_heads"], num_agents=mc["num_agents"],
        num_classes=mc["num_classes"], dropout=mc["dropout"],
        bet_temperature=mkt["bet_temperature"],
        feature_keep_prob=mkt.get("feature_keep_prob", 0.7),
    ).to(device)
    # Handle torch.compile _orig_mod. prefix in older checkpoints
    sd = ckpt["model_state_dict"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load_checkpoint] Missing keys (will use random init): "
              f"{[k.split('.')[-1] for k in missing]}")
    if unexpected:
        print(f"  [load_checkpoint] Unexpected keys (ignored): "
              f"{[k.split('.')[-1] for k in unexpected]}")

    capital_mgr = CapitalManager(
        num_agents=mc["num_agents"],
        initial_capital=mkt["initial_capital"],
        lr=mkt["capital_lr"],
        decay=mkt.get("capital_decay", 0.9),
        normalize_payoffs=mkt.get("normalize_payoffs", True),
        device=device,
    )
    capital_mgr.load_state_dict(ckpt["capital"])

    return model, capital_mgr, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to NPM checkpoint .pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device_str = args.device or "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model, capital_mgr, cfg = load_checkpoint(args.checkpoint, device)
    model.eval()

    _, test_loader = get_cifar10_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augmentation=False,
    )

    results = {}

    # ── OOD Detection ──
    ood_datasets = cfg.get("evaluation", {}).get("ood_datasets", ["cifar100", "svhn"])
    score_fns = ["epistemic_unc", "herding", "entropy_market", "market_unc",
                  "market_unc_sum", "market_unc_max", "market_unc_temp",
                  "pred_variance", "mutual_info"]

    for ood_name in ood_datasets:
        print(f"\n{'='*60}")
        print(f"OOD: {ood_name}")
        print(f"{'='*60}")

        ood_loader = get_ood_loader(
            ood_name, root=cfg["data"]["root"],
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
        )

        results[ood_name] = {}
        for sfn in score_fns:
            ood_res = ood_detection_scores(
                model, test_loader, ood_loader,
                capital_mgr, device, score_fn=sfn,
            )
            results[ood_name][sfn] = {
                "auroc": ood_res["auroc"],
                "aupr": ood_res["aupr"],
            }
            print(f"  {sfn:20s}  AUROC={ood_res['auroc']:.4f}  "
                  f"AUPR={ood_res['aupr']:.4f}")

    # ── Volatility as OOD Detector ──
    eval_cfg = cfg.get("evaluation", {})
    vol_eps = eval_cfg.get("volatility_eps", 0.01)
    vol_steps = eval_cfg.get("volatility_steps", 5)
    vol_max_batches = eval_cfg.get("volatility_max_batches", 50)

    for ood_name in ood_datasets:
        print(f"\n{'='*60}")
        print(f"Volatility OOD Detection: {ood_name}")
        print(f"{'='*60}")

        ood_loader = get_ood_loader(
            ood_name, root=cfg["data"]["root"],
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
        )

        for vkey in ["mean_js_div", "influence_gini_std"]:
            vres = ood_detection_volatility(
                model, test_loader, ood_loader,
                capital_mgr, device,
                score_key=vkey,
                eps=vol_eps, n_samples=vol_steps,
                max_batches=vol_max_batches,
            )
            tag = f"volatility_{vkey}"
            results.setdefault(ood_name, {})[tag] = {
                "auroc": vres["auroc"],
                "aupr": vres["aupr"],
                "id_mean": vres["id_mean"],
                "ood_mean": vres["ood_mean"],
            }
            print(f"  {tag:30s}  AUROC={vres['auroc']:.4f}  "
                  f"AUPR={vres['aupr']:.4f}  "
                  f"ID_mean={vres['id_mean']:.6f}  "
                  f"OOD_mean={vres['ood_mean']:.6f}")

    # ── Save ──
    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "phase2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path / 'phase2_results.json'}")


if __name__ == "__main__":
    main()
