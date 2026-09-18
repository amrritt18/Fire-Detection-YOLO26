from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "images"
LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "fire",
    1: "other",
    2: "smoke"
}


# ============================================================
# 3. DRAW YOLO BOUNDING BOXES
# ============================================================

def draw_annotations(image_path):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    if not label_path.exists():
        return image

    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # YOLO normalized coordinates
        # Convert to pixel coordinates

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Draw class name
        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return image


# ============================================================
# 4. SELECT RANDOM IMAGES
# ============================================================

def get_images():

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    images = [
        file
        for file in IMAGE_DIR.iterdir()
        if file.suffix.lower() in extensions
    ]

    return images


# ============================================================
# 5. VISUALIZE
# ============================================================

def visualize(num_images=12):

    images = get_images()

    if len(images) == 0:
        print("No images found.")
        return

    num_images = min(num_images, len(images))

    selected_images = random.sample(images, num_images)

    print(f"\nVisualizing {num_images} images...")

    for i, image_path in enumerate(selected_images, start=1):

        annotated_image = draw_annotations(image_path)

        if annotated_image is None:
            continue

        # Save image
        output_path = (
            OUTPUT_DIR /
            f"annotation_{i}_{image_path.name}"
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                annotated_image,
                cv2.COLOR_RGB2BGR
            )
        )

        # Display
        plt.figure(figsize=(10, 7))

        plt.imshow(annotated_image)
        plt.title(image_path.name)
        plt.axis("off")

        plt.show()

        print(f"Saved: {output_path}")


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("     YOLO ANNOTATION VISUALIZATION")
    print("========================================")

    visualize(num_images=12)

    print("\nVisualization complete.")