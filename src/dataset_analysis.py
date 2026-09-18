from pathlib import Path
from collections import Counter
import yaml


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "images"
LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "labels"

DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"


# ============================================================
# 2. CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "fire",
    1: "other",
    2: "smoke"
}


# ============================================================
# 3. BASIC DATASET COUNTS
# ============================================================

def count_files():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    images = [
        file for file in IMAGE_DIR.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    labels = list(LABEL_DIR.glob("*.txt"))

    print("\n========== DATASET SIZE ==========")

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    return images, labels


# ============================================================
# 4. CHECK IMAGE-LABEL PAIRS
# ============================================================

def check_image_label_pairs(images, labels):

    image_stems = {
        image.stem for image in images
    }

    label_stems = {
        label.stem for label in labels
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print("\n========== IMAGE-LABEL PAIRS ==========")

    print(f"Images without labels : {len(missing_labels)}")
    print(f"Labels without images : {len(missing_images)}")

    if missing_labels:
        print("\nExamples of images without labels:")
        for name in list(missing_labels)[:10]:
            print(name)

    if missing_images:
        print("\nExamples of labels without images:")
        for name in list(missing_images)[:10]:
            print(name)


# ============================================================
# 5. ANALYZE YOLO LABELS
# ============================================================

def analyze_labels():

    annotation_count = Counter()
    image_count = Counter()

    empty_labels = []
    invalid_labels = []

    total_annotations = 0

    for label_file in LABEL_DIR.glob("*.txt"):

        classes_in_image = set()

        with open(label_file, "r") as file:

            lines = file.readlines()

        # Check empty label files
        if not lines:
            empty_labels.append(label_file.name)
            continue

        for line_number, line in enumerate(lines, start=1):

            parts = line.strip().split()

            # Skip blank lines
            if not parts:
                continue

            # YOLO format:
            # class_id x_center y_center width height

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

            # Check YOLO coordinates
            values = [
                x_center,
                y_center,
                width,
                height
            ]

            if not all(0 <= value <= 1 for value in values):

                invalid_labels.append(
                    (
                        label_file.name,
                        line_number,
                        "Coordinate outside [0, 1]"
                    )
                )

                continue

            annotation_count[class_id] += 1
            classes_in_image.add(class_id)

            total_annotations += 1

        # Count image once per class
        for class_id in classes_in_image:
            image_count[class_id] += 1

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n========== CLASS DISTRIBUTION ==========")

    print(f"Total annotations : {total_annotations}")

    print("\nAnnotations per class:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_id} - {class_name:<6}: "
            f"{annotation_count[class_id]}"
        )

    print("\nImages containing each class:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_id} - {class_name:<6}: "
            f"{image_count[class_id]}"
        )

    print("\n========== DATA QUALITY ==========")

    print(f"Empty label files   : {len(empty_labels)}")
    print(f"Invalid annotations: {len(invalid_labels)}")

    if empty_labels:

        print("\nExample empty labels:")

        for file in empty_labels[:10]:
            print(file)

    if invalid_labels:

        print("\nExample invalid annotations:")

        for item in invalid_labels[:10]:
            print(item)

    return annotation_count, image_count


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("========================================")
    print("       FIRE DATASET ANALYSIS")
    print("========================================")

    print(f"\nImage directory:")
    print(IMAGE_DIR)

    print(f"\nLabel directory:")
    print(LABEL_DIR)

    # Check directories
    if not IMAGE_DIR.exists():

        print("\nERROR: Image directory does not exist.")
        return

    if not LABEL_DIR.exists():

        print("\nERROR: Label directory does not exist.")
        return

    images, labels = count_files()

    check_image_label_pairs(images, labels)

    analyze_labels()

    print("\n========================================")
    print("          ANALYSIS COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()