import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import argparse

from models import get_cifar_resnet50

def train(args):
    device = torch.device(args.device)
    print(f"Training on device: {device}")

    # Optimize CuDNN for faster convolutions
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Augmented CIFAR-10 transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.5)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))
    ])

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_cifar_resnet50(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.device == 'cuda'))

    best_acc = 0.0

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda" if args.device == "cuda" else "cpu", enabled=(args.device == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({"Loss": train_loss/(pbar.n+1), "Acc": 100.*correct/total})

        scheduler.step()

        # Evaluate
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        print(f"Test Accuracy: {acc:.2f}% | Test Loss: {test_loss/len(test_loader):.4f}")

        if acc > best_acc:
            print("Saving best model...")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            best_acc = acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_resnet_aug")
    args = parser.parse_args()

    train(args)
