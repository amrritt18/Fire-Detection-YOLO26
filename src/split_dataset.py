from pathlib import Path
from collections import Counter
import random
import shutil


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train"
    / "images"
)

SOURCE_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train"
    / "labels"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

CLASS_NAMES = {
    0: "fire",
    1: "other",
    2: "smoke"
}


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    splits = ["train", "valid", "test"]

    for split in splits:

        image_dir = OUTPUT_DIR / split / "images"
        label_dir = OUTPUT_DIR / split / "labels"

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 4. FIND IMAGE-LABEL PAIRS
# ============================================================

def get_image_label_pairs():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    pairs = []

    images = [
        image
        for image in SOURCE_IMAGE_DIR.iterdir()
        if image.suffix.lower() in image_extensions
    ]

    for image in images:

        label = SOURCE_LABEL_DIR / f"{image.stem}.txt"

        if not label.exists():

            print(
                f"WARNING: Missing label for {image.name}"
            )

            continue

        pairs.append(
            (image, label)
        )

    return pairs


# ============================================================
# 5. GET CLASSES PRESENT IN AN IMAGE
# ============================================================

def get_image_classes(label_path):

    classes = set()

    with open(label_path, "r") as file:

        for line in file:

            parts = line.strip().split()

            if not parts:
                continue

            class_id = int(parts[0])

            if class_id in CLASS_NAMES:
                classes.add(class_id)

    return classes


# ============================================================
# 6. CREATE MULTI-LABEL GROUPS
# ============================================================

def create_groups(pairs):

    groups = {}

    for image_path, label_path in pairs:

        classes = get_image_classes(label_path)

        # Represent the combination of classes
        # Example:
        # {0}       -> fire
        # {2}       -> smoke
        # {0, 2}    -> fire + smoke
        # {0, 1, 2} -> fire + other + smoke

        group = tuple(sorted(classes))

        if group not in groups:
            groups[group] = []

        groups[group].append(
            (image_path, label_path)
        )

    return groups


# ============================================================
# 7. SPLIT EACH GROUP
# ============================================================

def split_group(items):

    random.shuffle(items)

    total = len(items)

    train_count = int(total * TRAIN_RATIO)

    valid_count = int(total * VALID_RATIO)

    train_items = items[:train_count]

    valid_items = items[
        train_count:
        train_count + valid_count
    ]

    test_items = items[
        train_count + valid_count:
    ]

    return (
        train_items,
        valid_items,
        test_items
    )


# ============================================================
# 8. COPY FILES
# ============================================================

def copy_files(items, split):

    image_output_dir = (
        OUTPUT_DIR
        / split
        / "images"
    )

    label_output_dir = (
        OUTPUT_DIR
        / split
        / "labels"
    )

    for image_path, label_path in items:

        shutil.copy2(
            image_path,
            image_output_dir / image_path.name
        )

        shutil.copy2(
            label_path,
            label_output_dir / label_path.name
        )


# ============================================================
# 9. CALCULATE DISTRIBUTION
# ============================================================

def calculate_distribution(items):

    annotation_count = Counter()
    image_count = Counter()

    for image_path, label_path in items:

        classes_in_image = set()

        with open(label_path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                if class_id in CLASS_NAMES:

                    annotation_count[class_id] += 1
                    classes_in_image.add(class_id)

        for class_id in classes_in_image:
            image_count[class_id] += 1

    return annotation_count, image_count


# ============================================================
# 10. PRINT DISTRIBUTION
# ============================================================

def print_distribution(
    split,
    items
):

    annotation_count, image_count = (
        calculate_distribution(items)
    )

    print(
        f"\n========== {split.upper()} =========="
    )

    print(
        f"Images: {len(items)}"
    )

    print("\nAnnotations:")

    for class_id, name in CLASS_NAMES.items():

        print(
            f"{name:<6}: "
            f"{annotation_count[class_id]}"
        )

    print("\nImages containing class:")

    for class_id, name in CLASS_NAMES.items():

        print(
            f"{name:<6}: "
            f"{image_count[class_id]}"
        )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("========================================")
    print("       DATASET SPLITTING")
    print("========================================")

    # Check source directories

    if not SOURCE_IMAGE_DIR.exists():

        print(
            f"ERROR: Image directory not found:\n"
            f"{SOURCE_IMAGE_DIR}"
        )

        return

    if not SOURCE_LABEL_DIR.exists():

        print(
            f"ERROR: Label directory not found:\n"
            f"{SOURCE_LABEL_DIR}"
        )

        return

    # Set seed

    random.seed(RANDOM_SEED)

    # Create directories

    create_directories()

    # Get image-label pairs

    pairs = get_image_label_pairs()

    print(
        f"\nValid image-label pairs: {len(pairs)}"
    )

    # Create multi-label groups

    groups = create_groups(pairs)

    print(
        f"Unique class combinations: "
        f"{len(groups)}"
    )

    print("\nClass combinations:")

    for group, items in groups.items():

        names = [
            CLASS_NAMES[class_id]
            for class_id in group
        ]

        print(
            f"{' + '.join(names):<20}"
            f": {len(items)} images"
        )

    # Create split lists

    train_items = []
    valid_items = []
    test_items = []

    for group, items in groups.items():

        train_group, valid_group, test_group = (
            split_group(items)
        )

        train_items.extend(train_group)
        valid_items.extend(valid_group)
        test_items.extend(test_group)

    # Shuffle final lists

    random.shuffle(train_items)
    random.shuffle(valid_items)
    random.shuffle(test_items)

    # Copy data

    print("\nCopying files...")

    copy_files(train_items, "train")
    copy_files(valid_items, "valid")
    copy_files(test_items, "test")

    # Print distributions

    print_distribution(
        "train",
        train_items
    )

    print_distribution(
        "valid",
        valid_items
    )

    print_distribution(
        "test",
        test_items
    )

    # Final summary

    print("\n========================================")
    print("          SPLIT COMPLETE")
    print("========================================")

    print(
        f"\nTotal: {len(pairs)}"
    )

    print(
        f"Train: {len(train_items)} "
        f"({len(train_items) / len(pairs) * 100:.2f}%)"
    )

    print(
        f"Valid: {len(valid_items)} "
        f"({len(valid_items) / len(pairs) * 100:.2f}%)"
    )

    print(
        f"Test : {len(test_items)} "
        f"({len(test_items) / len(pairs) * 100:.2f}%)"
    )

    print(
        f"\nDataset created at:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()