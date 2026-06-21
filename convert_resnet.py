import torch
import os

mappings = [
    # (source_dir, prefix, target_dataset, target_model_name)
    ("features_resnet", "cifar10", "cifar10", "resnet50_baseline"),
    ("features_resnet_aug", "cifar10", "cifar10", "resnet50_aug"),
    ("features_resnet100", "cifar100", "cifar100", "resnet50_100"),
    ("features_resnet100_aug", "cifar100", "cifar100", "resnet50_100_aug")
]

for src_dir, prefix, target_ds, target_model in mappings:
    if not os.path.exists(src_dir): continue
    
    os.makedirs(f"features/{target_ds}/{target_model}", exist_ok=True)
    
    for split in ["train", "test"]:
        src_path = f"{src_dir}/{prefix}_{split}.pt"
        tgt_path = f"features/{target_ds}/{target_model}/{split}.pt"
        
        if os.path.exists(src_path):
            print(f"Loading {src_path}...")
            data = torch.load(src_path)
            # Map the "features" key to "cls" and "mean_pool" to match SSL models
            new_data = {
                "cls": data["features"],
                "mean_pool": data["features"],
                "labels": data["labels"]
            }
            torch.save(new_data, tgt_path)
            print(f"Converted {src_path} -> {tgt_path}")
