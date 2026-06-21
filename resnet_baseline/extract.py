import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from models import load_resnet50_for_extraction

@torch.no_grad()
def extract(loader, model, device):
    feats = []
    labels = []

    for i, (imgs, lbls) in enumerate(tqdm(loader)):
        imgs = imgs.to(device)

        # Forward pass through resnet to get pooled features
        # Note: we replaced fc with Identity, so output shape is (B, 2048)
        out = model(imgs)

        out = F.normalize(out, dim=1)

        feats.append(out.cpu())
        labels.append(lbls)

    return torch.cat(feats), torch.cat(labels)

def main(args):
    device = torch.device(args.device)
    print("Device:", device)

    # Use the same transforms as in training (for evaluation, just normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    model = load_resnet50_for_extraction(args.checkpoint)
    model = model.to(device)

    print("Extracting train features...")
    train_feats, train_labels = extract(train_loader, model, device)

    print("Extracting test features...")
    test_feats, test_labels = extract(test_loader, model, device)

    os.makedirs("features_resnet", exist_ok=True)
    
    train_out_path = "features_resnet/cifar10_train.pt"
    test_out_path = "features_resnet/cifar10_test.pt"

    torch.save({"features": train_feats, "labels": train_labels}, train_out_path)
    torch.save({"features": test_feats, "labels": test_labels}, test_out_path)

    print("DONE extracting features.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_resnet/best_model.pt", help="Path to trained checkpoint")
    args = parser.parse_args()

    main(args)
