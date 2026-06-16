import torch
from src.knn import knn_predict, accuracy

train = torch.load('features/cifar10_train.pt')
test = torch.load('features/cifar10_test.pt')

preds = knn_predict(
    train['features'],
    train['labels'],
    test['features'],
    k=5
)

print('Accuracy:', accuracy(preds, test['labels']))