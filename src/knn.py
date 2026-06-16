import torch
import torch.nn.functional as F


def knn_predict(train_feats, train_labels, test_feats, k=5):

    # cosine similarity (since features are L2-normalized)
    sims = test_feats @ train_feats.T   # [N_test, N_train]

    topk = sims.topk(k=k, dim=1)

    top_labels = train_labels[topk.indices]  # [N_test, k]

    preds = []

    for lbls in top_labels:
        preds.append(torch.mode(lbls).values.item())

    return torch.tensor(preds)


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()