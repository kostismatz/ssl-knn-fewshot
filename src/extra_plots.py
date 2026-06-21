import os
import sys
import csv
import math
import argparse
import collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import load_config


class ExtraPlotGenerator:
    """Generates the extended analysis figures from a results.csv produced by evaluate.py.

    Figures:
      - all_models_<pool>_comparison.png : accuracy-vs-shots, one line per model, shaded std band
      - pooling_comparison_bars.png      : CLS vs mean-pool bars (1-shot & 100-shot) per model
      - k_sensitivity_heatmap.png        : accuracy heatmap over k (at a fixed N) per dataset/pool
      - gain_over_random.png             : accuracy gain over random chance per model and shot count

    The model set, datasets, shot counts and k values are discovered automatically from the CSV,
    so the same class works whether you evaluated 4, 6 or 7 backbones.
    """

    # Canonical display order; any model not listed here is appended in CSV order.
    CANONICAL_ORDER = [
        "dino_vits16", "dinov2_vits14", "dinov2_vits14_reg", "dinov3_vits16",
        "moco_v3_vits16", "ibot_vits16", "attmask_vits16",
    ]
    DISPLAY_NAME = {
        "dino_vits16": "DINO ViT-S/16",
        "dinov2_vits14": "DINOv2 ViT-S/14",
        "dinov2_vits14_reg": "DINOv2 ViT-S/14 (reg)",
        "dinov3_vits16": "DINOv3 ViT-S/16",
        "moco_v3_vits16": "MoCo v3 ViT-S/16",
        "ibot_vits16": "iBOT ViT-S/16",
        "attmask_vits16": "AttMask ViT-S/16",
    }
    COLOR = {
        "dino_vits16": "#1f77b4",
        "dinov2_vits14": "#2ca02c",
        "dinov2_vits14_reg": "#9467bd",
        "dinov3_vits16": "#ff7f0e",
        "moco_v3_vits16": "#d62728",
        "ibot_vits16": "#8c564b",
        "attmask_vits16": "#e377c2",
    }
    # Random-chance accuracy per dataset (1 / num_classes).
    RANDOM_CHANCE = {"cifar10": 0.10, "cifar100": 0.01}

    def __init__(self, results_file=None, figures_dir=None, config_path="configs/default.yaml"):
        if results_file is None or figures_dir is None:
            cfg = load_config(config_path)
            results_file = results_file or cfg["paths"]["results_file"]
            figures_dir = figures_dir or cfg["paths"]["figures_dir"]
        self.results_file = Path(results_file)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ---------- data loading / aggregation ----------
    def _load(self):
        acc = collections.defaultdict(list)
        datasets, models, shots, ks = [], [], set(), set()
        with open(self.results_file, "r") as f:
            for row in csv.DictReader(f):
                key = (row["dataset"], row["model"], row["pooling"],
                       int(row["n_shots"]), int(row["k"]))
                acc[key].append(float(row["accuracy"]))
                if row["dataset"] not in datasets:
                    datasets.append(row["dataset"])
                if row["model"] not in models:
                    models.append(row["model"])
                shots.add(int(row["n_shots"]))
                ks.add(int(row["k"]))
        self._acc = acc
        self.datasets = datasets
        self.models = self._order_models(models)
        self.shots = sorted(shots)
        self.ks = sorted(ks)

    def _order_models(self, models):
        ordered = [m for m in self.CANONICAL_ORDER if m in models]
        ordered += [m for m in models if m not in ordered]
        return ordered

    def _name(self, m):
        return self.DISPLAY_NAME.get(m, m)

    def _color(self, m, idx):
        return self.COLOR.get(m, plt.cm.tab10(idx % 10))

    def mean(self, ds, m, pool, n, k):
        vals = self._acc.get((ds, m, pool, n, k))
        return float(np.mean(vals)) if vals else math.nan

    def std(self, ds, m, pool, n, k):
        vals = self._acc.get((ds, m, pool, n, k))
        return float(np.std(vals)) if vals else 0.0  # population std, matches plot.py band

    # ---------- figures ----------
    def all_models_comparison(self, pool="cls", k=5):
        N = self.shots
        fig, axes = plt.subplots(1, len(self.datasets), figsize=(5.85 * len(self.datasets), 5.0))
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, self.datasets):
            for idx, m in enumerate(self.models):
                me = np.array([self.mean(ds, m, pool, n, k) for n in N])
                sd = np.array([self.std(ds, m, pool, n, k) for n in N])
                ax.plot(N, me, "-o", color=self._color(m, idx), label=self._name(m), ms=4, lw=1.8)
                ax.fill_between(N, me - sd, me + sd, color=self._color(m, idx), alpha=0.15)
            ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(N)
            ax.set_title(ds.upper()); ax.set_xlabel("N Shots Per Class")
            ax.set_ylim(0, 1); ax.grid(True, which="both", ls="--", alpha=0.3)
        axes[0].set_ylabel("Accuracy")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"All Models — {'CLS Token' if pool == 'cls' else 'Mean Pool'} (k={k})", fontweight="bold")
        self._save(fig, f"all_models_{pool}_comparison.png")

    def pooling_comparison(self, k=5, low=1, high=100):
        x = np.arange(len(self.models)); w = 0.2
        fig, axes = plt.subplots(1, len(self.datasets), figsize=(5.75 * len(self.datasets), 5.0))
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, self.datasets):
            c_lo = [self.mean(ds, m, "cls", low, k) for m in self.models]
            p_lo = [self.mean(ds, m, "mean_pool", low, k) for m in self.models]
            c_hi = [self.mean(ds, m, "cls", high, k) for m in self.models]
            p_hi = [self.mean(ds, m, "mean_pool", high, k) for m in self.models]
            ax.bar(x - 1.5 * w, c_lo, w, color="#4C72B0", label=f"CLS ({low}-shot)")
            ax.bar(x - 0.5 * w, p_lo, w, facecolor="#4C72B0", alpha=0.45, hatch="//",
                   edgecolor="#4C72B0", label=f"MeanPool ({low}-shot)")
            ax.bar(x + 0.5 * w, c_hi, w, color="#C44E52", label=f"CLS ({high}-shot)")
            ax.bar(x + 1.5 * w, p_hi, w, facecolor="#C44E52", alpha=0.45, hatch="//",
                   edgecolor="#C44E52", label=f"MeanPool ({high}-shot)")
            ax.set_xticks(x); ax.set_xticklabels([self._name(m) for m in self.models],
                                                 rotation=30, ha="right", fontsize=7)
            ax.set_title(ds.upper()); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
            ax.grid(axis="y", ls="--", alpha=0.3)
        axes[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"CLS Token vs. Mean Pooling: Accuracy Comparison (k={k})", fontweight="bold")
        self._save(fig, "pooling_comparison_bars.png")

    def k_sensitivity_heatmap(self, n_shots=None):
        n_shots = n_shots if n_shots is not None else max(self.shots)
        pools = ["cls", "mean_pool"]
        panels = [(ds, pool) for ds in self.datasets for pool in pools]
        ncols = len(pools)
        nrows = len(self.datasets)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows))
        axes = np.atleast_2d(axes)
        for ax, (ds, pool) in zip(axes.flat, panels):
            M = np.array([[self.mean(ds, m, pool, n_shots, k) for k in self.ks] for m in self.models])
            im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(self.ks))); ax.set_xticklabels([f"k={k}" for k in self.ks])
            ax.set_yticks(range(len(self.models)))
            ax.set_yticklabels([self._name(m) for m in self.models], fontsize=7)
            for i in range(len(self.models)):
                for j in range(len(self.ks)):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6, fontweight="bold")
            ax.set_title(f"{ds.upper()} — pool={pool}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Effect of k on kNN Accuracy (N={n_shots} shots)", fontweight="bold")
        self._save(fig, "k_sensitivity_heatmap.png")

    def gain_over_random(self, pool="cls", k=5, shots=(1, 5, 20, 100)):
        shots = [s for s in shots if s in self.shots] or self.shots
        x = np.arange(len(shots)); w = min(0.8 / len(self.models), 0.18)
        offset = (len(self.models) - 1) / 2.0
        fig, axes = plt.subplots(1, len(self.datasets), figsize=(5.4 * len(self.datasets), 5.2))
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, self.datasets):
            rand = self.RANDOM_CHANCE.get(ds, 0.1)
            for idx, m in enumerate(self.models):
                vals = [self.mean(ds, m, pool, n, k) - rand for n in shots]
                ax.bar(x + (idx - offset) * w, vals, w, color=self._color(m, idx), label=self._name(m))
            ax.set_xticks(x); ax.set_xticklabels([f"{s}-shot" for s in shots])
            ax.set_title(f"{ds.upper()} (random={rand:.0%})")
            ax.set_ylabel("Accuracy gain over random"); ax.grid(axis="y", ls="--", alpha=0.3)
        axes[0].legend(fontsize=6)
        fig.suptitle(f"Accuracy vs. Random-Chance Baseline ({'CLS' if pool == 'cls' else 'Mean'}, k={k})",
                     fontweight="bold")
        self._save(fig, "gain_over_random.png")

    def generate_all(self, pool="cls", k=5):
        self.all_models_comparison(pool=pool, k=k)
        self.pooling_comparison(k=k)
        self.k_sensitivity_heatmap()
        self.gain_over_random(pool=pool, k=k)

    def _save(self, fig, name):
        fig.tight_layout()
        path = self.figures_dir / name
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def main(results_file=None, figures_dir=None, k=5, pooling="cls"):
    parser = argparse.ArgumentParser(description="Generate extended analysis figures from results.csv")
    parser.add_argument("--results", type=str, default=results_file, help="Path to results.csv")
    parser.add_argument("--figures", type=str, default=figures_dir, help="Output figures directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config (for default paths)")
    parser.add_argument("--k", type=int, default=k, help="Number of neighbours k for the line/bar figures")
    parser.add_argument("--pooling", type=str, default=pooling, choices=["cls", "mean_pool"],
                        help="Pooling mode for the line/bar/gain figures")
    args, _ = parser.parse_known_args()

    gen = ExtraPlotGenerator(results_file=args.results, figures_dir=args.figures, config_path=args.config)
    print(f"Loaded {len(gen.models)} models, {len(gen.datasets)} datasets from {gen.results_file}")
    gen.generate_all(pool=args.pooling, k=args.k)
    print("Done.")


if __name__ == "__main__":
    main()
