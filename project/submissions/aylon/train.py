#!/usr/bin/env python3
"""
Aylon's robust training script — Phase 3: ResNet + AugMix + Shape Bias.

Key improvements:
  - Uses Teammate A's ResNet-9 architecture (high capacity).
  - AugMix: Creates multiple augmented branches of each image and mixes them,
    with a Jensen-Shannon consistency loss to force identical predictions
    across all branches.
  - Mixup & CutMix in the training loop for background-invariant learning.
  - Shape-bias: Random edge detection transform forces the model to learn
    object outlines rather than textures/colors.
  - Cosine Annealing LR scheduler + weight decay.
  - Best-model checkpointing based on validation accuracy.
"""
import sys
from pathlib import Path
import random
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import ImageFilter, ImageOps, Image

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
EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DATA_FRACTION = 1.0  # Use all available training data

# --- Mixup / CutMix Configuration ---
MIXUP_ALPHA = 0.2       # Beta distribution parameter for Mixup
CUTMIX_ALPHA = 1.0      # Beta distribution parameter for CutMix
MIXUP_PROB = 0.3         # Probability of applying Mixup on a batch
CUTMIX_PROB = 0.2        # Probability of applying CutMix on a batch

# --- AugMix Configuration ---
AUGMIX_SEVERITY = 3      # Max severity of individual augmentation ops
AUGMIX_WIDTH = 3         # Number of augmentation chains to mix
AUGMIX_DEPTH = -1        # -1 means random depth (1-3) per chain
AUGMIX_ALPHA_DIRICHLET = 1.0  # Dirichlet mixing weight
JSD_LAMBDA = 12.0        # Weight for Jensen-Shannon consistency loss

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── AugMix augmentation operations ───────────────────────────────────────────

def int_parameter(level, maxval):
    """Helper to scale level to a discrete int parameter."""
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper to scale level to a float parameter."""
    return float(level) * maxval / 10.0

# Individual augmentation ops used by AugMix (PIL-based)
def autocontrast(pil_img, level):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, level):
    return ImageOps.equalize(pil_img)

def posterize(pil_img, level):
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img, max(1, 4 - level))

def rotate_aug(pil_img, level):
    degrees = float_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def solarize(pil_img, level):
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def shear_y(pil_img, level):
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def translate_x(pil_img, level):
    level = int_parameter(level, pil_img.size[0] // 3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def translate_y(pil_img, level):
    level = int_parameter(level, pil_img.size[1] // 3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))

def color_aug(pil_img, level):
    from PIL import ImageEnhance
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)

def contrast_aug(pil_img, level):
    from PIL import ImageEnhance
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)

def brightness_aug(pil_img, level):
    from PIL import ImageEnhance
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)

def sharpness_aug(pil_img, level):
    from PIL import ImageEnhance
    level = float_parameter(level, 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

# List of all augmentation ops for AugMix
AUGMIX_OPS = [
    autocontrast, equalize, posterize, rotate_aug, solarize,
    shear_x, shear_y, translate_x, translate_y,
    color_aug, contrast_aug, brightness_aug, sharpness_aug,
]


def augmix_chain(image, severity, depth):
    """Apply a chain of random augmentation ops to a PIL image."""
    if depth <= 0:
        depth = random.randint(1, 3)
    for _ in range(depth):
        op = random.choice(AUGMIX_OPS)
        level = random.randint(1, severity)
        image = op(image, level)
    return image


class AugMixTransform:
    """
    AugMix: Creates multiple augmented branches and mixes them.

    Returns a tuple of (clean_tensor, augmix_tensor1, augmix_tensor2)
    so the training loop can compute Jensen-Shannon divergence
    between the predictions of all three versions.
    """
    def __init__(self, preprocess, severity=3, width=3, depth=-1, alpha=1.0):
        self.preprocess = preprocess  # transforms.Compose for Resize+CenterCrop
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, pil_img):
        # Preprocess to standard size first
        pil_img = self.preprocess(pil_img)

        # Clean version
        x_clean = self.to_tensor(pil_img)

        # Generate two AugMix branches for JSD
        augmix_imgs = []
        for _ in range(2):
            # Mix multiple augmented chains using Dirichlet weights
            ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
            m = np.float32(np.random.beta(self.alpha, self.alpha))

            mix = torch.zeros_like(x_clean)
            for w_i in range(self.width):
                aug_img = augmix_chain(pil_img.copy(), self.severity, self.depth)
                mix += ws[w_i] * self.to_tensor(aug_img)

            # Interpolate between clean and augmented
            augmixed = (1 - m) * x_clean + m * mix
            augmix_imgs.append(augmixed)

        return x_clean, augmix_imgs[0], augmix_imgs[1]


# ── Shape-bias: Edge detection transform ─────────────────────────────────────

class RandomEdgeDetection:
    """
    Randomly converts the image to an edge-detected version.
    This forces the model to learn object SHAPES rather than textures/colors.
    """
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, pil_img):
        if random.random() < self.p:
            edges = pil_img.filter(ImageFilter.FIND_EDGES)
            # Convert back to RGB (edges are often grayscale-ish)
            return edges.convert("RGB")
        return pil_img


# ── Standard training transforms (for non-AugMix batches) ────────────────────

STANDARD_PREPROCESS = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
])

STANDARD_AUGMENTATIONS = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.15),
    RandomEdgeDetection(p=0.08),  # Shape-bias!
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPosterize(bits=4, p=0.1),
    transforms.RandomSolarize(threshold=200, p=0.1),
])

STANDARD_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])


class HybridTransform:
    """
    For each image, randomly chooses between:
      - AugMix (returns 3 tensors for JSD loss): 40% of the time
      - Standard augmentations (returns 1 tensor): 60% of the time

    When AugMix is NOT used, returns (augmented_tensor, None, None)
    so the DataLoader can collate consistently.
    """
    def __init__(self, augmix_prob=0.4):
        self.augmix_prob = augmix_prob
        self.augmix = AugMixTransform(
            preprocess=STANDARD_PREPROCESS,
            severity=AUGMIX_SEVERITY,
            width=AUGMIX_WIDTH,
            depth=AUGMIX_DEPTH,
            alpha=AUGMIX_ALPHA_DIRICHLET,
        )
        self.standard = transforms.Compose([
            STANDARD_PREPROCESS,
            STANDARD_AUGMENTATIONS,
            STANDARD_TO_TENSOR,
        ])

    def __call__(self, pil_img):
        if random.random() < self.augmix_prob:
            return self.augmix(pil_img)
        else:
            return self.standard(pil_img), None, None


# ── Custom collate to handle the hybrid (clean, aug1, aug2) tuples ────────────

def hybrid_collate(batch):
    """
    Custom collate function that handles both:
      - Standard samples: (tensor, label) where the transform returned (tensor, None, None)
      - AugMix samples: ((clean, aug1, aug2), label)

    Returns: images, labels, aug1_batch (or None), aug2_batch (or None)
    """
    images = []
    labels = []
    aug1_list = []
    aug2_list = []
    has_augmix = False

    for sample, label in batch:
        if isinstance(sample, tuple) and len(sample) == 3:
            clean, aug1, aug2 = sample
            images.append(clean)
            labels.append(label)
            if aug1 is not None:
                aug1_list.append(aug1)
                aug2_list.append(aug2)
                has_augmix = True
            else:
                aug1_list.append(clean)  # Placeholder
                aug2_list.append(clean)
        else:
            images.append(sample)
            labels.append(label)
            aug1_list.append(sample)
            aug2_list.append(sample)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    if has_augmix:
        aug1_batch = torch.stack(aug1_list)
        aug2_batch = torch.stack(aug2_list)
        return images, labels, aug1_batch, aug2_batch
    else:
        return images, labels, None, None


# ── Mixup & CutMix helpers ───────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    """Mixup: blend two images and their labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate a random bounding box for CutMix."""
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
    """CutMix: cut a patch from one image and paste it onto another."""
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
    """Compute the mixed loss for Mixup or CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Jensen-Shannon Divergence loss for AugMix ────────────────────────────────

def jsd_loss(logits_clean, logits_aug1, logits_aug2):
    """
    Jensen-Shannon Divergence between three predictions.
    Forces the model to make consistent predictions regardless of augmentation.
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)

    # Average distribution (the "M" in JSD)
    p_mixture = (p_clean + p_aug1 + p_aug2) / 3.0

    # KL divergence from each to the mixture
    loss = (F.kl_div(p_mixture.log(), p_clean, reduction='batchmean') +
            F.kl_div(p_mixture.log(), p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture.log(), p_aug2, reduction='batchmean')) / 3.0
    return loss


# ── Data subset helper ────────────────────────────────────────────────────────

def get_fraction_subset(dataset, fraction=0.05, seed=42):
    if fraction >= 1.0:
        return dataset
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)
    random.seed(seed)
    selected_indices = []
    for label, indices in class_indices.items():
        k = max(1, int(len(indices) * fraction))
        selected_indices.extend(random.sample(indices, k))
    return Subset(dataset, selected_indices)


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hybrid transform: 40% AugMix, 60% standard augmentations
    train_transform = HybridTransform(augmix_prob=0.4)

    # Validation transforms (no augmentation, clean evaluation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("Loading datasets...")
    full_train_dataset = ImageNetSubset(DATA_ROOT, split="train_50", transform=train_transform)
    val_dataset = ImageNetSubset(DATA_ROOT, split="val_10", transform=val_transform)

    train_dataset = get_fraction_subset(full_train_dataset, fraction=DATA_FRACTION)
    print(f"Training subset size: {len(train_dataset)} samples (Fraction: {DATA_FRACTION})")
    print(f"Validation subset size: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=hybrid_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Instantiate ResNet-9 model
    model = ModelArchitecture(num_classes=20).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    best_val_acc = 0.0
    best_state_dict = None

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Mixup prob: {MIXUP_PROB}, CutMix prob: {CUTMIX_PROB}")
    print(f"AugMix prob: 0.4, JSD Lambda: {JSD_LAMBDA}")
    print(f"LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_data in train_loader:
            images, labels, aug1, aug2 = batch_data
            images, labels = images.to(device), labels.to(device)

            # ── Mixup / CutMix on the clean images ──
            r = random.random()
            if r < MIXUP_PROB:
                images, targets_a, targets_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                optimizer.zero_grad()
                logits = model(images)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            elif r < MIXUP_PROB + CUTMIX_PROB:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                optimizer.zero_grad()
                logits = model(images)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)

            # ── AugMix JSD consistency loss ──
            if aug1 is not None and aug2 is not None:
                aug1, aug2 = aug1.to(device), aug2.to(device)
                logits_aug1 = model(aug1)
                logits_aug2 = model(aug2)
                loss += JSD_LAMBDA * jsd_loss(logits.detach(), logits_aug1, logits_aug2)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
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

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    print("-" * 70)

    if best_state_dict is not None:
        joblib.dump(best_state_dict, OUTPUT_WEIGHTS)
        print(f"Saved best weights with Val Acc: {best_val_acc:.4f} to {OUTPUT_WEIGHTS}")
    else:
        joblib.dump(model.cpu().state_dict(), OUTPUT_WEIGHTS)
        print(f"Saved default weights to {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()