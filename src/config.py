import os
import yaml
from pathlib import Path

def load_config(config_path="configs/default.yaml"):
    # Determine the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    
    # Try direct path, then try relative to project root
    path = Path(config_path)
    if not path.is_absolute():
        path = root_dir / config_path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Convert output paths to absolute paths relative to project root
    paths = config.get("paths", {})
    for k, v in paths.items():
        v_path = Path(v)
        if not v_path.is_absolute():
            paths[k] = str((root_dir / v_path).resolve())
            
    config["paths"] = paths
    return config
