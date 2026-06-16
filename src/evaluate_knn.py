import torch
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

    train = torch.load("features/cifar10_train.pt")
    test = torch.load("features/cifar10_test.pt")

    results = run_eval(train, test)

    print("\nFINAL RESULTS:")
    for r in results:
        print(r)

    plot_results(results)

from src.plot import plot_results