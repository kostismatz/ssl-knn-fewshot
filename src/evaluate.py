import os
import sys
import csv
import torch
import argparse
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.knn import knn_predict, accuracy
from src.config import load_config

def calculate_macro_f1(preds, labels, num_classes):
    f1_scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
    return sum(f1_scores) / len(f1_scores)

def sample_few_shot(features, labels, N, seed, num_classes):
    torch.manual_seed(seed)
    
    selected_indices = []
    for c in range(num_classes):
        class_idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(class_idx) < N:

            chosen = class_idx
        else:
            perm = torch.randperm(len(class_idx))
            chosen = class_idx[perm[:N]]
        selected_indices.append(chosen)
        
    selected_indices = torch.cat(selected_indices)
    return features[selected_indices], labels[selected_indices]

def main(config_path="configs/default.yaml", force=False):
    config = load_config(config_path)
    
    features_dir = Path(config["paths"]["features_dir"])
    results_file = Path(config["paths"]["results_file"])
    results_file.parent.mkdir(parents=True, exist_ok=True)

    datasets_list = config["datasets"]
    models_list = config["models"]
    n_shots_list = config["n_shots"]
    k_neighbors_list = config["k_neighbors"]
    num_seeds = config["seeds"]
    pooling_modes = config.get("pooling_modes", ["cls", "mean_pool"])

    file_exists = results_file.exists()

    mode = "w" if (not file_exists or force) else "a"
    csv_file = open(results_file, mode, newline="")
    csv_writer = csv.writer(csv_file)
    
    if mode == "w":
        csv_writer.writerow(["dataset", "model", "pooling", "n_shots", "seed", "k", "accuracy", "macro_f1"])
        csv_file.flush()

    completed_runs = set()
    if mode == "a" and file_exists:
        with open(results_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    completed_runs.add((row[0], row[1], row[2], int(row[3]), int(row[4]), int(row[5])))

    for dataset_name in datasets_list:
        num_classes = 10 if dataset_name == "cifar10" else 100
        
        for model_name in models_list:
            model_feat_dir = features_dir / dataset_name / model_name
            train_path = model_feat_dir / "train.pt"
            test_path = model_feat_dir / "test.pt"
            
            if not train_path.exists() or not test_path.exists():
                print(f"Skipping model {model_name} on {dataset_name}: Cached features not found.")
                continue
                
            print(f"Evaluating {model_name} on {dataset_name}...")

            train_cache = torch.load(train_path, map_location="cpu")
            test_cache = torch.load(test_path, map_location="cpu")
            
            for pool in pooling_modes:
                train_feats = train_cache[pool]
                test_feats = test_cache[pool]
                train_labels = train_cache["labels"]
                test_labels = test_cache["labels"]
                
                for N in n_shots_list:
                    for seed in range(num_seeds):
                        
                        # Prepare run list of k neighbors to sweep for this subset
                        k_list = [k for k in k_neighbors_list if (dataset_name, model_name, pool, N, seed, k) not in completed_runs]
                        if not k_list:
                            continue
                            
                        # Stratified sample
                        sampled_train_feats, sampled_train_labels = sample_few_shot(
                            train_feats, train_labels, N, seed, num_classes
                        )
                        
                        for k in k_list:
                            clamped_k = min(k, len(sampled_train_feats))
                            
                            preds = knn_predict(
                                sampled_train_feats,
                                sampled_train_labels,
                                test_feats,
                                k=clamped_k
                            )
                            
                            acc = accuracy(preds, test_labels)
                            macro_f1 = calculate_macro_f1(preds, test_labels, num_classes)
                            
                            # Log and write row
                            csv_writer.writerow([dataset_name, model_name, pool, N, seed, k, acc, macro_f1])
                            csv_file.flush()
                            
    csv_file.close()
    print(f"Evaluation complete. Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--force", action="store_true", help="Force recalculation and overwrite results file")
    args = parser.parse_args()
    
    main(args.config, args.force)
