import os
import urllib.request
import torch
import timm
from pathlib import Path
import ssl

# Bypass SSL certificate verification for downloading models
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    
    print(f"Downloading checkpoint from {url} to {dest_path}...")
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            if percent % 20 == 0 and percent > 0:
                print(f"Progress: {percent}%", end="\r")
                
    urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
    print("\nDownload complete.")

def load_backbone(model_name, checkpoints_dir="checkpoints"):
    checkpoints_dir = Path(checkpoints_dir)
    print(f"Loading model: {model_name}...")

    if model_name == "dino_vits16":
        # DINOv1 ViT-S/16
        model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
        
    elif model_name == "dinov2_vits14":
        # DINOv2 ViT-S/14 (requires img_size=224 to override default 518 in timm)
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224)
        
    elif model_name == "dinov2_vits14_reg":
        # DINOv2 ViT-S/14 with registers (requires img_size=224 to override default 518 in timm)
        model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True, img_size=224)
        
    elif model_name == "moco_v3_vits16":
        # MoCo v3 ViT-S/16
        moco_url = "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        checkpoint_path = checkpoints_dir / "moco_v3_vit_s_300ep.pth.tar"
        
        # Download if not present
        download_file(moco_url, checkpoint_path)
        
        # Load state dict
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Strip prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module.base_encoder."):
                new_state_dict[k.replace("module.base_encoder.", "")] = v
            else:
                new_state_dict[k] = v
                
        model = timm.create_model("vit_small_patch16_224", pretrained=False)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"MoCo v3 loaded. Strict=False message: {msg}")
        
    elif model_name in ["ibot_vits16", "attmask_vits16"]:
        # Custom models that are hosted on Google Drive / Box and require manual download
        checkpoint_path = checkpoints_dir / f"{model_name}.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Checkpoint for {model_name} not found at '{checkpoint_path}'.\n"
                f"Because official weights for iBOT and AttMask are hosted on Google Drive, "
                f"please download them manually and place them in the checkpoints folder:\n"
                f" - For ibot_vits16: Download 'checkpoint_teacher.pth' from https://github.com/bytedance/ibot\n"
                f" - For attmask_vits16: Download the ViT-S teacher checkpoint from https://github.com/gkakogeorgiou/attmask\n"
                f"Rename them and place them at: {checkpoint_path.resolve()}\n"
            )
            
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Look for teacher state dict (standard in iBOT/AttMask)
        state_dict = None
        for key in ["teacher", "state_dict", "model"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
            
        # Strip prefixes like 'backbone.' or 'module.'
        new_state_dict = {}
        for k, v in state_dict.items():
            k_new = k
            if k.startswith("backbone."):
                k_new = k.replace("backbone.", "")
            elif k.startswith("module.backbone."):
                k_new = k.replace("module.backbone.", "")
            elif k.startswith("module."):
                k_new = k.replace("module.", "")
            new_state_dict[k_new] = v
            
        model = timm.create_model("vit_small_patch16_224", pretrained=False)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"{model_name} loaded. Strict=False message: {msg}")
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    model.eval()
    return model

@torch.no_grad()
def get_features(model, x, pool="cls"):
    out = model.forward_features(x)  # [B, N, C]
    num_prefix = getattr(model, "num_prefix_tokens", 1)
    
    if pool == "cls":
        return out[:, 0]
    elif pool == "mean_pool":
        return out[:, num_prefix:].mean(dim=1)
    else:
        raise ValueError(f"Unknown pool mode: {pool}")