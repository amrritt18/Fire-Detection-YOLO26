from pathlib import Path
from collections import Counter


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "fire",
    1: "other",
    2: "smoke"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(split):

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    print("\n" + "=" * 50)
    print(f"{split.upper()} DATASET")
    print("=" * 50)

    if not image_dir.exists():
        print(f"ERROR: Missing image directory: {image_dir}")
        return

    if not label_dir.exists():
        print(f"ERROR: Missing label directory: {label_dir}")
        return

    # --------------------------------------------------------
    # Get files
    # --------------------------------------------------------

    images = [
        file
        for file in image_dir.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    # --------------------------------------------------------
    # Check image-label correspondence
    # --------------------------------------------------------

    image_stems = {
        image.stem
        for image in images
    }

    label_stems = {
        label.stem
        for label in labels
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print("\nImage-label validation:")

    print(
        f"Images without labels : "
        f"{len(missing_labels)}"
    )

    print(
        f"Labels without images : "
        f"{len(missing_images)}"
    )

    # --------------------------------------------------------
    # Analyze annotations
    # --------------------------------------------------------

    annotation_count = Counter()
    image_count = Counter()

    empty_labels = []
    invalid_labels = []

    total_annotations = 0

    for label_file in labels:

        classes_in_image = set()

        with open(label_file, "r") as file:

            lines = file.readlines()

        if not lines:

            empty_labels.append(
                label_file.name
            )

            continue

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.strip().split()

            if not parts:
                continue

            # YOLO format:
            # class x_center y_center width height

            if len(parts) != 5:

                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Expected 5 values"
                    )
                )

                continue

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Non-numeric value"
                    )
                )

                continue

            # Check class ID

            if class_id not in CLASS_NAMES:

                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        f"Invalid class ID: {class_id}"
                    )
                )

                continue

            # Check coordinates

            coordinates = [
                x_center,
                y_center,
                width,
                height
            ]

            if not all(
                0 <= value <= 1
                for value in coordinates
            ):

                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Coordinate outside [0,1]"
                    )
                )

                continue

            # Check width and height > 0

            if width <= 0 or height <= 0:

                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Width/height <= 0"
                    )
                )

                continue

            annotation_count[class_id] += 1

            classes_in_image.add(class_id)

            total_annotations += 1

        # Count image once per class

        for class_id in classes_in_image:

            image_count[class_id] += 1

    # --------------------------------------------------------
    # Print distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    print(
        f"{'Class':<10}"
        f"{'Annotations':>15}"
        f"{'Images':>15}"
    )

    print("-" * 40)

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_name:<10}"
            f"{annotation_count[class_id]:>15}"
            f"{image_count[class_id]:>15}"
        )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    print("\nData quality:")

    print(
        f"Empty labels    : "
        f"{len(empty_labels)}"
    )

    print(
        f"Invalid labels  : "
        f"{len(invalid_labels)}"
    )

    # --------------------------------------------------------
    # Show errors
    # --------------------------------------------------------

    if missing_labels:

        print("\nExample missing labels:")

        for item in list(missing_labels)[:5]:
            print(item)

    if missing_images:

        print("\nExample missing images:")

        for item in list(missing_images)[:5]:
            print(item)

    if empty_labels:

        print("\nExample empty labels:")

        for item in empty_labels[:5]:
            print(item)

    if invalid_labels:

        print("\nExample invalid labels:")

        for item in invalid_labels[:5]:
            print(item)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 50)
    print("       PROCESSED DATASET VALIDATION")
    print("=" * 50)

    for split in SPLITS:

        validate_split(split)

    print("\n" + "=" * 50)
    print("       VALIDATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
    