#!/usr/bin/env python3
import sys
from pathlib import Path
import random
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import ImageFilter

torch.backends.cudnn.benchmark = True

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
EPOCHS = 20          
LEARNING_RATE = 1.2*(1e-3)
DATA_FRACTION = 1.0  # Train on 100% of the train_65 split

# --- Mixup / CutMix Configuration ---
MIXUP_ALPHA = 0.2       
CUTMIX_ALPHA = 1.0      
MIXUP_PROB = 0.3         
CUTMIX_PROB = 0.2        

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ── Mixup & CutMix helpers ───────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x_cut, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ── Data subset helper ────────────────────────────────────────────────────────

class RandomEdgeDetection:
    """Randomly converts the image to an edge-detected version to force shape-bias learning."""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, pil_img):
        if random.random() < self.p:
            edges = pil_img.filter(ImageFilter.FIND_EDGES)
            return edges.convert("RGB")
        return pil_img

def get_fraction_subset(dataset, fraction=0.05, seed=42):
    if fraction >= 1.0:
        return dataset

    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    random.seed(seed)
    selected_indices = []
    for label, indices in class_indices.items():
        k = int(len(indices) * fraction)
        k = max(1, k)
        selected_indices.extend(random.sample(indices, k))

    return Subset(dataset, selected_indices)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Slightly less aggressive crop
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Reduced from 30 to preserve shape integrity
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        RandomEdgeDetection(p=0.05), # Added shape-bias
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Reduced scale slightly
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("Loading datasets...")
    full_train_dataset = ImageNetSubset(DATA_ROOT, split="train_85", transform=train_transform)
    val_dataset = ImageNetSubset(DATA_ROOT, split="val_15", transform=val_transform)

    train_dataset = get_fraction_subset(full_train_dataset, fraction=DATA_FRACTION)
    print(f"Training subset size: {len(train_dataset)} samples (Fraction: {DATA_FRACTION})")
    print(f"Validation subset size: {len(val_dataset)} samples")

    # num_workers=2 explicitly configured here for optimal loading on Colab
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Instantiate model
    model = ModelArchitecture(num_classes=20).to(device, memory_format=torch.channels_last)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Using AdamW for better weight decay handling + OneCycleLR for faster convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler() # VRAM optimization via AMP
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_val_acc = 0.0
    best_state_dict = None

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Mixup prob: {MIXUP_PROB}, CutMix prob: {CUTMIX_PROB}")
    print(f"LR: {LEARNING_RATE}, Weight Decay: 1e-4")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            r = random.random()
            if r < MIXUP_PROB:
                images_mixed, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images_mixed)
                    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            elif r < MIXUP_PROB + CUTMIX_PROB:
                images_mixed, targets_a, targets_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images_mixed)
                    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            
            # 1. Scale the loss and compute gradients
            scaler.scale(loss).backward()
            
            # 2. Unscale the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            
            # 3. Clip the unscaled gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 4. Step the optimizer and update the scaler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            
            if r < MIXUP_PROB + CUTMIX_PROB:
                dominant_targets = targets_a if lam > 0.5 else targets_b
                correct_train += (preds == dominant_targets).sum().item()
            else:
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
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}"
              + (" ★" if epoch_val_acc >= best_val_acc else ""))

        # Keep best weights
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            # Copy to CPU to be hardware independent as required
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    print("-" * 70)

    # Save both uncompiled and compiled versions
    OUTPUT_COMPILED = current_dir / "weights_compiled.joblib"
    
    state_dict_to_save = best_state_dict if best_state_dict is not None else model.cpu().state_dict()
    
    # Save the original (compiled) version
    joblib.dump(state_dict_to_save, OUTPUT_COMPILED)
    print(f"Saved compiled weights to {OUTPUT_COMPILED}")
    
    # Create uncompiled version
    uncompiled_state_dict = {}
    for k, v in state_dict_to_save.items():
        new_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        uncompiled_state_dict[new_key] = v
        
    joblib.dump(uncompiled_state_dict, OUTPUT_WEIGHTS)
    print(f"Saved uncompiled weights to {OUTPUT_WEIGHTS}")
    if best_state_dict is not None:
        print(f"Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()