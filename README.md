# 🔥 Fire Detection using YOLO26

A computer vision project for detecting **fire and smoke in images** using the **YOLO26 object detection architecture**.

The project includes a complete dataset preparation and validation pipeline, followed by YOLO26 model training and evaluation. The dataset contains three object classes:

- **Fire**
- **Smoke**
- **Other**

---

## 📌 Project Overview

Early detection of fire and smoke is important for surveillance, industrial safety, smart buildings, and automated emergency response systems.

This project aims to develop an object detection system capable of identifying fire and smoke directly from images using a modern YOLO-based architecture.

The complete workflow is:

```text
Raw Dataset
     ↓
Dataset Audit
     ↓
Annotation Visualization
     ↓
Multi-label Dataset Splitting
     ↓
Dataset Validation
     ↓
YOLO Dataset Configuration
     ↓
YOLO26 Training
     ↓
Model Evaluation
     ↓
Inference
```

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a reliable fire and smoke detection system.
2. Prepare and validate an object detection dataset.
3. Maintain class distribution across training, validation, and test sets.
4. Train a YOLO26 object detection model.
5. Evaluate detection performance using standard object detection metrics.
6. Perform inference on unseen images.
7. Develop a project structure suitable for further deployment.

---

## 🗂️ Dataset

The dataset was obtained from **Roboflow Universe** and contains approximately **8,939 images** with three object classes.

### Classes

| Class ID | Class |
|---:|---|
| 0 | Fire |
| 1 | Other |
| 2 | Smoke |

The dataset uses the standard **YOLO annotation format**.

Each image has a corresponding `.txt` annotation file containing:

```text
class_id x_center y_center width height
```

The bounding-box coordinates are normalized to the range `[0, 1]`.

---

## 📊 Dataset Statistics

The original dataset contains:

- **8,939 images**
- **8,939 label files**
- **24,091 total annotations**

### Annotation Distribution

| Class | Annotations | Images containing class |
|---|---:|---:|
| Fire | 13,914 | 7,704 |
| Other | 4,845 | 2,182 |
| Smoke | 5,332 | 3,601 |

> The number of images containing each class does not add up to the total number of images because a single image can contain multiple classes.

### Class Combinations

The dataset contains images with different combinations of objects, including:

- Fire
- Smoke
- Other
- Fire + Smoke
- Fire + Other
- Other + Smoke
- Fire + Other + Smoke

---

## 🔍 Dataset Quality Analysis

Before training, the dataset was systematically checked for annotation and file-level problems.

The following checks were performed:

- Missing images
- Missing label files
- Empty label files
- Invalid YOLO annotations
- Invalid class IDs
- Bounding-box coordinates outside the valid range
- Image-label correspondence
- Class distribution
- Multi-class combinations

### Audit Result

```text
Images without labels       : 0
Labels without images       : 0
Empty label files           : 0
Invalid annotations         : 0
```

This ensured that the dataset was structurally suitable for object detection training.

---

## 🖼️ Annotation Visualization

Sample images were visualized with their corresponding bounding boxes and class labels before training.

This step was performed to verify that:

- Bounding boxes are correctly positioned.
- Class IDs correspond to the intended classes.
- Images and labels are correctly paired.
- Annotations are visually meaningful.

The visualization code is available in:

```text
src/visualize_annotations.py
```

---

## ✂️ Dataset Splitting

The original dataset contained only a `train` directory. Therefore, the dataset was split manually into:

```text
Train      → 70%
Validation → 20%
Test       → 10%
```

A class-combination-aware splitting strategy was used so that images containing the same combination of classes were distributed across the three subsets while maintaining approximately similar class distributions.

### Final Split

| Split | Images |
|---|---:|
| Train | 6,254 |
| Validation | 1,784 |
| Test | 901 |
| **Total** | **8,939** |

The total number of annotations after splitting remained **24,091**, confirming that no annotations were lost during dataset preparation.

---

## 📁 Project Structure

```text
Fire-Detection-YOLO26/
│
├── data/
│   ├── raw/
│   │   └── train/
│   │       ├── images/
│   │       └── labels/
│   │
│   └── processed/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       │
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       │
│       └── test/
│           ├── images/
│           └── labels/
│
├── visualizations/
│   └── Sample Images                 
│
├── configs/
│   └── data.yaml
│
├── src/
│   ├── dataset_analysis.py
│   ├── visualize_annotations.py
│   ├── split_dataset.py
│   └── validate_dataset.py
│
├── notebooks/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ YOLO Dataset Configuration

The dataset configuration is stored in:

```text
configs/data.yaml
```

Configuration:

```yaml
path: ../data/processed

train: train/images
val: valid/images
test: test/images

names:
  0: fire
  1: other
  2: smoke
```

---

## 🐍 Environment Setup

Python **3.12** was used for the local project environment.

Create the environment using `uv`:

```bash
uv venv --python 3.12
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

Install the project dependencies:

```bash
uv pip install -r requirements.txt
```

---

## 📦 Main Technologies

The project uses:

- **Python**
- **YOLO26**
- **Ultralytics**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **PyYAML**
- **Google Colab** for model training

---

## 🔬 Dataset Preparation Pipeline

### 1. Dataset Audit

The initial dataset was analyzed using:

```bash
python src/dataset_analysis.py
```

This script checks the dataset structure, annotation statistics, class distribution, and annotation validity.

### 2. Annotation Visualization

```bash
python src/visualize_annotations.py
```

This generates sample images with bounding boxes and class labels.

### 3. Dataset Splitting

```bash
python src/split_dataset.py
```

This creates the final:

```text
train/
valid/
test/
```

dataset structure.

### 4. Dataset Validation

```bash
python src/validate_dataset.py
```

This performs a final structural and annotation validation before model training.

---

# 🚀 Model Training

Model training will be performed using **Google Colab** to take advantage of GPU acceleration.

The planned training pipeline is:

```text
Load Dataset
     ↓
Install Ultralytics
     ↓
Load YOLO26 Model
     ↓
Load data.yaml
     ↓
Train Model
     ↓
Validate Model
     ↓
Evaluate on Test Set
     ↓
Save Best Weights
```

The training notebook will be added to:

```text
notebooks/
```

---

## 📈 Model Evaluation

The trained model will be evaluated using standard object detection metrics, including:

- **Precision**
- **Recall**
- **mAP@50**
- **mAP@50-95**
- **Confusion Matrix**
- Per-class detection performance

The final model results will be added to this README after training.

---

## 🔎 Inference

After training, the model will be tested on previously unseen images.

Example inference workflow:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test_image.jpg")

for result in results:
    result.show()
```

The model should identify objects such as:

```text
🔥 Fire
💨 Smoke
📦 Other
```

along with their corresponding bounding boxes and confidence scores.

---

## 📌 Future Work

Possible extensions of this project include:

- Real-time fire detection using CCTV/video streams.
- Webcam-based detection.
- Video inference.
- Deployment using Streamlit.
- Docker-based deployment.
- Edge-device deployment.
- Integration with an automated alert system.
- Evaluation on real-world surveillance footage.
- Model optimization for faster inference.

---

## 💡 Key Learning Outcomes

Through this project, the following concepts are covered:

- Object detection
- YOLO annotation format
- Dataset auditing
- Bounding-box visualization
- Multi-label dataset splitting
- Dataset validation
- YOLO dataset configuration
- YOLO26 training
- Object detection evaluation
- Model inference
- Computer vision pipeline development

---

## 👨‍💻 Author

**Amrit Raj**

M.Tech — Robotics & Artificial Intelligence  
IIT Bhubaneswar

---

## 📄 Project Status

```text
✅ Dataset collected
✅ Dataset audited
✅ Annotation visualization completed
✅ Dataset split completed
✅ Dataset validated
✅ YOLO configuration completed
✅ Python environment configured
⬜ YOLO26 model training
⬜ Model evaluation
⬜ Test-set inference
⬜ Deployment
```

---

## ⭐ Acknowledgements

Dataset sourced from **Roboflow Universe**.

Model training and object detection are implemented using the **Ultralytics YOLO framework**.