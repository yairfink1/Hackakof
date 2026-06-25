#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import joblib

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX, TARGET_HF_INDICES
from submissions.my_team.model import ModelArchitecture

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RobustnessSubset(Dataset):
    """Loads a specific pre-generated augmentation split from dataset/augmentations/."""

    def __init__(self, root: Path, augmentation_name: str, transform=None):
        self.transform = transform
        self.samples = []

        split_root = root / "augmentations" / augmentation_name

        if not split_root.exists():
            raise FileNotFoundError(f"Augmentation folder not found: {split_root}")

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            local_idx = HF_INDEX_TO_IDX[hf_idx]

            image_paths = []
            image_paths.extend(class_dir.glob("*.jpg"))
            image_paths.extend(class_dir.glob("*.jpeg"))
            image_paths.extend(class_dir.glob("*.JPEG"))
            image_paths.extend(class_dir.glob("*.png"))

            for img_path in sorted(image_paths):
                self.samples.append((img_path, local_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_on_set(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running robustness evaluation on device: {device}")

    # Standard preprocessing
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load our team's trained model
    team_dir = project_root / "submissions" / "my_team"
    weights_path = team_dir / "weights.joblib"

    if not weights_path.exists():
        print(f"Error: Could not find weights at {weights_path}")
        sys.exit(1)

    print("Loading model and weights...")
    model = ModelArchitecture(num_classes=20)
    state_dict = joblib.load(weights_path)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Evaluate standard validation first for reference
    from base_model import ImageNetSubset
    val_dataset = ImageNetSubset(project_root / "dataset", split="val_15", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_acc = evaluate_on_set(model, val_loader, device)
    print(f"\n[Baseline] Val_15 Clean Accuracy: {val_acc:.4f}")

    # Evaluate color_jitter
    try:
        cj_dataset = RobustnessSubset(project_root / "dataset", "color_jitter", transform=transform)
        cj_loader = DataLoader(cj_dataset, batch_size=32, shuffle=False)
        cj_acc = evaluate_on_set(model, cj_loader, device)
        print(f"[Stress Test] Color Jitter Robustness Accuracy: {cj_acc:.4f}")
    except Exception as e:
        print(f"Error evaluating color_jitter: {e}")

    # Evaluate random_rotation
    try:
        rr_dataset = RobustnessSubset(project_root / "dataset", "random_rotation", transform=transform)
        rr_loader = DataLoader(rr_dataset, batch_size=32, shuffle=False)
        rr_acc = evaluate_on_set(model, rr_loader, device)
        print(f"[Stress Test] Random Rotation Robustness Accuracy: {rr_acc:.4f}")
    except Exception as e:
        print(f"Error evaluating random_rotation: {e}")


if __name__ == "__main__":
    main()
