"""
VinBigData subset creation utilities.

Builds balanced training/val/test subsets from the full VinBigData YOLO dataset
by capping over-represented classes and prioritising rare ones.
"""

import os
import shutil
import random
from collections import defaultdict

CLASS_NAMES = {
    0: "Aortic_enlargement", 1: "Atelectasis",  2: "Calcification",
    3: "Cardiomegaly",       4: "Consolidation", 5: "ILD",
    6: "Infiltration",       7: "Lung_Opacity",  8: "Nodule/Mass",
    9: "Other_lesion",      10: "Pleural_effusion", 11: "Pleural_thickening",
    12: "Pneumothorax",     13: "Pulmonary_fibrosis", 14: "No finding",
}

# Classes that are rare and should be prioritised when extending a subset
PRIORITY_CLASSES = {1, 12, 4, 2, 6}


def collect_all_pairs(base_dir: str) -> list[tuple[str, str]]:
    """
    Walk train/val/test splits under *base_dir* and return (img_path, lbl_path) pairs.
    """
    pairs = []
    for split in ("train", "val", "test"):
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            img_path = os.path.join(img_dir, fname.replace(".txt", ".jpg"))
            lbl_path = os.path.join(lbl_dir, fname)
            if os.path.exists(img_path):
                pairs.append((img_path, lbl_path))
    return pairs


def _read_classes(lbl_path: str) -> list[int]:
    with open(lbl_path) as f:
        return [int(line.split()[0]) for line in f if line.strip()]


def build_balanced_subset(
    source_dir: str,
    output_dir: str,
    total_images: int = 5000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    max_class_14: int = 1500,
    max_other: int = 1200,
    seed: int = 42,
) -> dict:
    """
    Build a class-balanced subset of *total_images* images.

    Args:
        source_dir:    Root of the original VinBigData YOLO dataset.
        output_dir:    Where to write the subset (train/val/test splits).
        total_images:  Target total number of images.
        train_ratio:   Fraction for training.
        val_ratio:     Fraction for validation (remainder goes to test).
        max_class_14:  Cap for the dominant "No finding" class.
        max_other:     Cap for all other classes.
        seed:          Random seed for reproducibility.

    Returns:
        Dict with split sizes and class distribution.
    """
    random.seed(seed)

    train_size = int(total_images * train_ratio)
    val_size   = int(total_images * val_ratio)

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    all_pairs = collect_all_pairs(source_dir)
    random.shuffle(all_pairs)
    print(f"Total available images: {len(all_pairs)}")

    selected = []
    class_counts: dict[int, int] = defaultdict(int)

    for img_path, lbl_path in all_pairs:
        if len(selected) >= total_images:
            break
        classes = _read_classes(lbl_path)
        if not classes:
            continue
        max_cap = max_class_14 if 14 in classes else max_other
        if any(class_counts[c] >= max_cap for c in classes):
            continue
        selected.append((img_path, lbl_path))
        for c in classes:
            class_counts[c] += 1

    # Distribute into splits
    train_pairs = selected[:train_size]
    val_pairs   = selected[train_size:train_size + val_size]
    test_pairs  = selected[train_size + val_size:]

    split_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    for split, pairs in split_map.items():
        for img_path, lbl_path in pairs:
            fname_base = os.path.basename(img_path).replace(".jpg", "")
            shutil.copy(img_path, os.path.join(output_dir, split, "images", os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(output_dir, split, "labels", fname_base + ".txt"))

    stats = {
        "total_selected": len(selected),
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
        "class_distribution": dict(class_counts),
    }
    print(f"Subset created: {stats['train']} train / {stats['val']} val / {stats['test']} test")
    return stats


def extend_subset(
    source_dir: str,
    existing_subset_dir: str,
    output_dir: str,
    target_new: int = 2500,
    seed: int = 42,
) -> dict:
    """
    Extend an existing subset with *target_new* additional images,
    prioritising rare classes (PRIORITY_CLASSES).

    Args:
        source_dir:           Root of the original VinBigData dataset.
        existing_subset_dir:  Path to the already-built subset (to avoid duplicates).
        output_dir:           Where to write the extended subset.
        target_new:           How many new images to add.
        seed:                 Random seed.

    Returns:
        Dict with counts of new images added per split.
    """
    random.seed(seed)

    # Collect already-selected image filenames
    already_selected: set[str] = set()
    for split in ("train", "val", "test"):
        img_dir = os.path.join(existing_subset_dir, split, "images")
        if os.path.isdir(img_dir):
            already_selected.update(os.listdir(img_dir))
    print(f"Already selected: {len(already_selected)} images")

    # Collect remaining candidates
    remaining = []
    for split in ("train", "val", "test"):
        img_dir = os.path.join(source_dir, split, "images")
        lbl_dir = os.path.join(source_dir, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            img_name = fname.replace(".txt", ".jpg")
            if img_name in already_selected:
                continue
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, fname)
            if os.path.exists(img_path):
                remaining.append((img_path, lbl_path))

    random.shuffle(remaining)
    print(f"Remaining candidates: {len(remaining)}")

    # Pass 1: prioritise rare classes
    selected_new = []
    selected_new_paths: set[str] = set()

    for img_path, lbl_path in remaining:
        if len(selected_new) >= target_new:
            break
        classes = set(_read_classes(lbl_path))
        if classes & PRIORITY_CLASSES:
            selected_new.append((img_path, lbl_path))
            selected_new_paths.add(img_path)

    # Pass 2: fill the rest
    for img_path, lbl_path in remaining:
        if len(selected_new) >= target_new:
            break
        if img_path not in selected_new_paths:
            selected_new.append((img_path, lbl_path))
            selected_new_paths.add(img_path)

    print(f"New images selected: {len(selected_new)}")

    # Copy existing subset + new images into output_dir
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)
        src_img = os.path.join(existing_subset_dir, split, "images")
        src_lbl = os.path.join(existing_subset_dir, split, "labels")
        if os.path.isdir(src_img):
            for f in os.listdir(src_img):
                shutil.copy(os.path.join(src_img, f), os.path.join(output_dir, split, "images", f))
        if os.path.isdir(src_lbl):
            for f in os.listdir(src_lbl):
                shutil.copy(os.path.join(src_lbl, f), os.path.join(output_dir, split, "labels", f))

    # Distribute new images into train (70%) and val/test proportionally
    n = len(selected_new)
    train_new = selected_new[: int(n * 0.70)]
    val_new   = selected_new[int(n * 0.70) : int(n * 0.90)]
    test_new  = selected_new[int(n * 0.90) :]

    for split, pairs in [("train", train_new), ("val", val_new), ("test", test_new)]:
        for img_path, lbl_path in pairs:
            fname_base = os.path.basename(img_path).replace(".jpg", "")
            shutil.copy(img_path, os.path.join(output_dir, split, "images", os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(output_dir, split, "labels", fname_base + ".txt"))

    return {"new_train": len(train_new), "new_val": len(val_new), "new_test": len(test_new)}
