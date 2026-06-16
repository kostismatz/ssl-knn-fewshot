import torch
import torch.nn as nn
from torchvision.models import resnet50

def get_cifar_resnet50(num_classes=10):
    """
    Returns a ResNet-50 adapted for 32x32 CIFAR images.
    """
    model = resnet50(weights=None)
    
    # Adapt conv1 and maxpool for 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Change final fc for CIFAR num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def load_resnet50_for_extraction(checkpoint_path=None):
    """
    Returns a ResNet-50 model ready for feature extraction.
    Replaces the final fc layer with an Identity layer.
    """
    model = get_cifar_resnet50(num_classes=10)
    
    if checkpoint_path is not None:
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    
    # Replace the final fully connected layer with an Identity layer
    # to output the raw pooled features
    model.fc = nn.Identity()
    
    model.eval()
    return model
