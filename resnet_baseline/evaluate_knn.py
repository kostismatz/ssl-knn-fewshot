import torch
import sys
import os

# Add src to path to import knn and plot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from plot import plot_results
from knn import knn_predict, accuracy

def sample_few_shot(features, labels, N, num_classes=10, seed=0):
    torch.manual_seed(seed)
    selected_idx = []

    for c in range(num_classes):
        class_idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))
        chosen = class_idx[perm[:N]]
        selected_idx.append(chosen)

    selected_idx = torch.cat(selected_idx)
    return features[selected_idx], labels[selected_idx]

def run_eval(train, test, N_values=[1,2,5,10,50], k=5, seeds=3):
    results = []
    test_feats = test["features"]
    test_labels = test["labels"]

    for N in N_values:
        accs = []
        for seed in range(seeds):
            train_feats, train_labels = sample_few_shot(
                train["features"],
                train["labels"],
                N,
                seed=seed
            )
            preds = knn_predict(
                train_feats,
                train_labels,
                test_feats,
                k=k
            )
            accs.append(accuracy(preds, test_labels))
        results.append((N, sum(accs)/len(accs)))

    return results

if __name__ == "__main__":
    train = torch.load("features_resnet/cifar10_train.pt")
    test = torch.load("features_resnet/cifar10_test.pt")

    results = run_eval(train, test)

    print("\nFINAL RESULTS (ResNet-50 Baseline):")
    for r in results:
        print(r)

    # Save to a separate plot
    import matplotlib.pyplot as plt
    
    # Re-implement plot_results slightly to save to results_resnet.png instead of results.png
    plt.figure(figsize=(8,6))
    Ns, accs = zip(*results)
    
    plt.plot(Ns, accs, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.xticks(Ns, Ns)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.title('Few-Shot KNN Accuracy (ResNet-50 Baseline)', fontsize=14)
    plt.xlabel('Number of Shots per Class (N)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results_resnet.png', dpi=300)
    print("Saved plot to results_resnet.png")
