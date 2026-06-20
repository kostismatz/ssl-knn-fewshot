import torch

def knn_predict(train_feats, train_labels, test_feats, k=5, tau=0.07):

    train_feats = torch.as_tensor(train_feats)
    train_labels = torch.as_tensor(train_labels)
    test_feats = torch.as_tensor(test_feats)

    k = min(k, len(train_feats))
    if k <= 0:
        raise ValueError("Number of neighbors k must be at least 1")

    # Compute cosine similarity matrix
    sims = test_feats @ train_feats.T

    # Find the top-k nearest neighbors
    topk = sims.topk(k=k, dim=1)
    top_sims = topk.values
    top_indices = topk.indices

    # DINO-style weighted voting: exp(similarity / tau)
    weights = torch.exp(top_sims / tau)

    # Map indices to labels
    top_labels = train_labels[top_indices]

    num_classes = int(train_labels.max().item() + 1)
    N_test = test_feats.shape[0]
    scores = torch.zeros(N_test, num_classes, device=test_feats.device)
    
    scores.scatter_add_(dim=1, index=top_labels, src=weights)

    preds = scores.argmax(dim=1)
    return preds

def accuracy(preds, labels):
    preds = torch.as_tensor(preds)
    labels = torch.as_tensor(labels)
    return (preds == labels).float().mean().item()