import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import load_dinov2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction (lowered for 4GB VRAM)")
parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
args = parser.parse_args()

# =========================
# DEVICE
# =========================
DEVICE = args.device
print("Device:", DEVICE)


# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# =========================
# DATASET LOAD
# =========================
train_ds_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_ds_full = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# FULL DATASET
train_ds = train_ds_full
test_ds = test_ds_full


# =========================
# DATALOADERS
# =========================
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)


# =========================
# MODEL
# =========================
model = load_dinov2()
model = model.to(DEVICE)
model.eval()


# =========================
# FEATURE EXTRACTION
# =========================
@torch.no_grad()
def extract(loader):
    feats = []
    labels = []

    for i, (imgs, lbls) in enumerate(tqdm(loader)):
        imgs = imgs.to(DEVICE)

        with torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu"):
            out = model.forward_features(imgs)
            cls = out["x_norm_clstoken"]

        cls = F.normalize(cls, dim=1)

        feats.append(cls.cpu())
        labels.append(lbls)

        # debug progress
        if i % 10 == 0:
            print("batch:", i)

    return torch.cat(feats), torch.cat(labels)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("Extracting train...")
    train_feats, train_labels = extract(train_loader)

    print("Extracting test...")
    test_feats, test_labels = extract(test_loader)


    # =========================
    # SAVE FEATURES
    # =========================
    torch.save(
        {"features": train_feats, "labels": train_labels},
        "features/cifar10_train.pt"
    )

    torch.save(
        {"features": test_feats, "labels": test_labels},
        "features/cifar10_test.pt"
    )

    print("DONE")