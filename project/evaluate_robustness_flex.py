#!/usr/bin/env python3
"""
Evaluate a team's model on robustness augmentation sets.

Usage:
    python evaluate_robustness_flex.py [team_name]

    team_name: Name of the submission folder under submissions/ (default: my_team)

Examples:
    python evaluate_robustness_flex.py aylon
    python evaluate_robustness_flex.py my_team
"""
import importlib.util
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model_from_team(team_name: str):
    """Dynamically load ModelArchitecture from a team's model.py."""
    team_dir = project_root / "submissions" / team_name
    model_path = team_dir / "model.py"
    weights_path = team_dir / "weights.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"model.py not found in {team_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights.joblib not found in {team_dir}")

    # Dynamically import model.py
    spec = importlib.util.spec_from_file_location(f"{team_name}_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model.py from {team_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "ModelArchitecture"):
        raise AttributeError(f"model.py in {team_dir} must define ModelArchitecture")

    model = module.ModelArchitecture(num_classes=20)
    state_dict = joblib.load(weights_path)
    model.load_state_dict(state_dict)
    return model


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
    # Parse team name from CLI args
    team_name = sys.argv[1] if len(sys.argv) > 1 else "my_team"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running robustness evaluation for '{team_name}' on device: {device}")

    # Standard preprocessing
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load the team's trained model dynamically
    print("Loading model and weights...")
    model = load_model_from_team(team_name)
    model = model.to(device)

    # Evaluate standard validation first for reference
    from base_model import ImageNetSubset
    val_dataset = ImageNetSubset(project_root / "dataset", split="val_10", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_acc = evaluate_on_set(model, val_loader, device)
    print(f"\n[Baseline] Val_10 Clean Accuracy: {val_acc:.4f}")

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

    # Summary
    print(f"\n--- Robustness Summary for '{team_name}' ---")
    print(f"  Clean Accuracy:           {val_acc:.4f}")
    try:
        print(f"  Color Jitter Accuracy:    {cj_acc:.4f}")
    except NameError:
        print(f"  Color Jitter Accuracy:    FAILED")
    try:
        print(f"  Random Rotation Accuracy: {rr_acc:.4f}")
    except NameError:
        print(f"  Random Rotation Accuracy: FAILED")


if __name__ == "__main__":
    main()
