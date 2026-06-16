import torch


def load_dinov2():

    print("Loading DINOv2...")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    )

    model.eval()

    return model