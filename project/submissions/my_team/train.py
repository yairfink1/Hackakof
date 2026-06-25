#!/usr/bin/env python3
import sys
from pathlib import Path
import random
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Add project root to sys.path so we can import base_model
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from base_model import ImageNetSubset
from model import ModelArchitecture

# Configuration
DATA_ROOT = project_root / "dataset"
OUTPUT_WEIGHTS = current_dir / "weights.joblib"

# --- Training Configurations ---
BATCH_SIZE = 128
EPOCHS = 10          
LEARNING_RATE = 1e-3
DATA_FRACTION = 1.0  # Train on 100% of the train_65 split (13,000 images total)


def get_fraction_subset(dataset, fraction=0.05, seed=42):
    if fraction >= 1.0:
        return dataset

    # Stratified split: group indices by label
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    random.seed(seed)
    selected_indices = []
    for label, indices in class_indices.items():
        k = int(len(indices) * fraction)
        k = max(1, k)  # Ensure at least one image per class
        selected_indices.extend(random.sample(indices, k))

    return Subset(dataset, selected_indices)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Data transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    # Train transforms with robustness augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("Loading datasets...")
    # Loader reads from dataset/train_65/
    full_train_dataset = ImageNetSubset(DATA_ROOT, split="train_65", transform=train_transform)

    # Loader reads from dataset/val_10/
    val_dataset = ImageNetSubset(DATA_ROOT, split="val_10", transform=val_transform)

    # Subsample training dataset for speed on CPU
    train_dataset = get_fraction_subset(full_train_dataset, fraction=DATA_FRACTION)
    print(f"Training subset size: {len(train_dataset)} samples (Fraction: {DATA_FRACTION})")
    print(f"Validation subset size: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Instantiate model
    model = ModelArchitecture(num_classes=20).to(device, memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_val_acc = 0.0
    best_state_dict = None

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, memory_format=torch.channels_last)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Epoch {epoch}/{EPOCHS} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Keep best weights
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            # Copy to CPU to be hardware independent as required
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save to weights.joblib
    if best_state_dict is not None:
        joblib.dump(best_state_dict, OUTPUT_WEIGHTS)
        print(f"Saved best weights with Val Acc: {best_val_acc:.4f} to {OUTPUT_WEIGHTS}")
    else:
        # Fallback
        joblib.dump(model.cpu().state_dict(), OUTPUT_WEIGHTS)
        print(f"Saved default weights to {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()