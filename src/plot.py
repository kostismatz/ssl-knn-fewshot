import os
import sys
import csv
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / max(1, n - 1)
    std = math.sqrt(variance)
    return mean, std

def load_results_data(results_file):
    data = {}
    with open(results_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Mapping column indices
        col_map = {col: i for i, col in enumerate(header)}
        
        for row in reader:
            if not row:
                continue
            dataset = row[col_map["dataset"]]
            model = row[col_map["model"]]
            pooling = row[col_map["pooling"]]
            n_shots = int(row[col_map["n_shots"]])
            seed = int(row[col_map["seed"]])
            k = int(row[col_map["k"]])
            accuracy = float(row[col_map["accuracy"]])
            macro_f1 = float(row[col_map["macro_f1"]])
            
            key = (dataset, model, pooling, n_shots, k)
            if key not in data:
                data[key] = {"acc": [], "f1": []}
            data[key]["acc"].append(accuracy)
            data[key]["f1"].append(macro_f1)
            
    # Compute mean and std
    aggregated = {}
    for key, vals in data.items():
        mean_acc, std_acc = mean_std(vals["acc"])
        mean_f1, std_f1 = mean_std(vals["f1"])
        aggregated[key] = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1
        }
    return aggregated

def plot_accuracy_curves(aggregated_data, figures_dir, k=5, pooling="cls"):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify unique datasets and models
    datasets_set = set()
    models_set = set()
    for (dataset, model, pool, n_shots, num_k) in aggregated_data.keys():
        datasets_set.add(dataset)
        if pool == pooling and num_k == k:
            models_set.add(model)
            
    for dataset_name in datasets_set:
        plt.figure(figsize=(8, 6))
        
        # Color palette for distinct visual aesthetics
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        
        for idx, model_name in enumerate(sorted(models_set)):
            # Filter and sort data points by N shots
            model_points = []
            for (d, m, p, n, num_k), metrics in aggregated_data.items():
                if d == dataset_name and m == model_name and p == pooling and num_k == k:
                    model_points.append((n, metrics["mean_acc"], metrics["std_acc"]))
                    
            if not model_points:
                continue
                
            model_points.sort(key=lambda x: x[0])
            n_shots = [p[0] for p in model_points]
            means = [p[1] for p in model_points]
            stds = [p[2] for p in model_points]
            
            color = colors[idx % len(colors)]
            
            # Plot line
            plt.plot(n_shots, means, marker="o", label=model_name, linewidth=2, color=color)
            
            # Shaded standard deviation band
            lower_bound = [m - s for m, s in zip(means, stds)]
            upper_bound = [m + s for m, s in zip(means, stds)]
            plt.fill_between(n_shots, lower_bound, upper_bound, alpha=0.15, color=color)
            
        plt.xscale("log")
        plt.xticks(n_shots, [str(n) for n in n_shots])
        
        # Labels and design styling
        plt.title(f"Few-Shot kNN Accuracy on {dataset_name.upper()} (k={k}, pool={pooling})", fontsize=12, pad=15)
        plt.xlabel("N Shots Per Class", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)
        plt.ylim(0, 1)
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.legend(frameon=True, facecolor="white", edgecolor="none")
        
        save_path = figures_dir / f"{dataset_name}_knn_{k}_{pooling}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved accuracy plot to {save_path}")

def main(results_file="results/results.csv", figures_dir="results/figures", k=5, pooling="cls"):
    # Parse CLI args if present (works both standalone and via cli.py)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=results_file)
    parser.add_argument("--figures", type=str, default=figures_dir)
    parser.add_argument("--k", type=int, default=k)
    parser.add_argument("--pooling", type=str, default=pooling, choices=["cls", "mean_pool"])
    args, _ = parser.parse_known_args()
    
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"[ERROR] Results file not found at {results_file}. Run evaluate.py first.")
        return
        
    print(f"Loading results from {results_file}...")
    agg = load_results_data(results_file)
    
    print(f"Generating plots for k={args.k}, pooling={args.pooling}...")
    plot_accuracy_curves(agg, args.figures, args.k, args.pooling)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/results.csv", help="Path to results file")
    parser.add_argument("--figures", type=str, default="results/figures", help="Dir to save figures")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors k to plot")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean_pool"], help="Pooling mode to plot")
    args = parser.parse_args()
    
    main(args.results, args.figures, args.k, args.pooling)