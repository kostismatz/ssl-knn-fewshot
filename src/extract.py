import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets
from pathlib import Path
from tqdm import tqdm
import ssl

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.models import load_backbone
from src.config import load_config

@torch.no_grad()
def extract_dataset_gpu(dataset, model, device, batch_size):
    # Convert entire dataset (numpy array [N, 32, 32, 3]) to float tensor [N, 3, 32, 32] in [0, 1]
    data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    
    cls_feats = []
    mp_feats = []
    
    # ImageNet normalization parameters on GPU
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    num_batches = math.ceil(len(data) / batch_size)
    
    for i in tqdm(range(0, len(data), batch_size), desc="Extracting", total=num_batches, leave=False):
        batch_imgs = data[i : i + batch_size].to(device)
        
        # Normalize and resize on GPU
        batch_imgs = (batch_imgs - mean) / std
        batch_imgs = F.interpolate(batch_imgs, size=(224, 224), mode="bicubic", align_corners=False)
        
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        with torch.autocast(device_type=device_type):
            out = model.forward_features(batch_imgs)

            cls = out[:, 0]
            
            # Extract mean pool (excl. prefix tokens like CLS and registers)
            num_prefix = getattr(model, "num_prefix_tokens", 1)
            mp = out[:, num_prefix:].mean(dim=1)
            
            # L2-normalize
            cls = F.normalize(cls, dim=1)
            mp = F.normalize(mp, dim=1)
            
        cls_feats.append(cls.cpu())
        mp_feats.append(mp.cpu())
        
    return torch.cat(cls_feats), torch.cat(mp_feats), labels

def main(config_path="configs/default.yaml", force=False):
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extraction Device: {device}")
    
    data_dir = config["paths"]["data_dir"]
    features_dir = Path(config["paths"]["features_dir"])
    batch_size = config.get("batch_size", 128)
    
    for dataset_name in config["datasets"]:
        print(f"\n========================================\nDATASET: {dataset_name}\n========================================")
        
        # Load raw dataset (we don't pass transforms here since we preprocess on GPU)
        if dataset_name == "cifar10":
            train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True)
            test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True)
        elif dataset_name == "cifar100":
            train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True)
            test_ds = datasets.CIFAR100(root=data_dir, train=False, download=True)
        else:
            print(f"Skipping unknown dataset: {dataset_name}")
            continue
            
        for model_name in config["models"]:
            model_feat_dir = features_dir / dataset_name / model_name
            train_cache = model_feat_dir / "train.pt"
            test_cache = model_feat_dir / "test.pt"
            
            if train_cache.exists() and test_cache.exists() and not force:
                print(f"Features already cached for {model_name} on {dataset_name}. Skipping.")
                continue
                
            try:
                model = load_backbone(model_name)
                model = model.to(device)
            except FileNotFoundError as e:
                print(e)
                print(f"Skipping model {model_name} because weight file is missing.")
                continue
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                continue
                
            model_feat_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Extracting train features...")
            train_cls, train_mp, train_labels = extract_dataset_gpu(train_ds, model, device, batch_size)
            torch.save({
                "cls": train_cls,
                "mean_pool": train_mp,
                "labels": train_labels
            }, train_cache)
            
            print(f"Extracting test features...")
            test_cls, test_mp, test_labels = extract_dataset_gpu(test_ds, model, device, batch_size)
            torch.save({
                "cls": test_cls,
                "mean_pool": test_mp,
                "labels": test_labels
            }, test_cache)
            
            print(f"Saved cached features to {model_feat_dir}")
            
            # Clean up GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--force", action="store_true", help="Force recalculating features")
    args = parser.parse_args()
    
    main(args.config, args.force)