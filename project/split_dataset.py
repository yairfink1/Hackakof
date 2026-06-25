import os
import random
from pathlib import Path
import shutil

def main():
    # Seeds for reproducibility
    random.seed(42)

    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    train_dir = dataset_dir / "train"

    if not train_dir.exists():
        print(f"Error: Train directory {train_dir} does not exist.")
        return

    # Define target split directories
    train_85_dir = dataset_dir / "train_85"
    val_15_dir = dataset_dir / "val_15"
    validation_dir = dataset_dir / "validation"  # Needed for evaluate.py

    # Clean existing splits if they exist to allow re-running
    for d in [train_85_dir, val_15_dir, validation_dir]:
        if d.exists():
            print(f"Removing existing directory: {d}")
            shutil.rmtree(d)

    # Get all class subdirectories
    classes = [d for d in train_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes.")

    total_train = 0
    total_val = 0

    for class_path in sorted(classes):
        class_name = class_path.name
        
        # Get all images in this class
        image_paths = []
        image_paths.extend(class_path.glob("*.jpg"))
        image_paths.extend(class_path.glob("*.jpeg"))
        image_paths.extend(class_path.glob("*.JPEG"))
        image_paths.extend(class_path.glob("*.png"))
        image_paths = sorted(image_paths)

        # Shuffle deterministically
        random.shuffle(image_paths)

        n_total = len(image_paths)
        if n_total == 0:
            print(f"Warning: No images found in {class_path}")
            continue

        # Splits: 85% train, 15% val
        n_train = int(n_total * 0.85)
        n_val = n_total - n_train
        
        train_set = image_paths[:n_train]
        val_set = image_paths[n_train:]

        # Helper function to link images
        def link_images(src_list, target_sub_dir):
            target_sub_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_list:
                dst_file = target_sub_dir / src_file.name
                try:
                    # Try creating a hard link first (fast, 0 disk space, works on Windows)
                    os.link(src_file, dst_file)
                except OSError:
                    # Fallback to copy if hard link fails (e.g. crossing device boundaries)
                    shutil.copy2(src_file, dst_file)

        # Link files to their respective split directories
        link_images(train_set, train_85_dir / class_name)
        link_images(val_set, val_15_dir / class_name)
        link_images(val_set, validation_dir / class_name)  # evaluate.py reads from dataset/validation

        print(f"Class '{class_name}': {len(train_set)} train, {len(val_set)} val.")
        total_train += len(train_set)
        total_val += len(val_set)

    print("\nSplitting complete!")
    print(f"Total linked: {total_train} train_85, {total_val} val_15.")

if __name__ == "__main__":
    main()
