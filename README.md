# 🔥 Real-Time Fire Detection System using YOLO26

A complete computer vision system for **real-time fire, smoke, and other-object detection** using the **YOLO26 object detection architecture**.

The project covers the complete pipeline from dataset preparation and validation to model training, evaluation, real-time webcam inference, automated email alerts, snapshot generation, and persistent alert history.

---

## 📌 Project Overview

Early detection of fire and smoke is important for surveillance, industrial safety, smart buildings, and automated emergency-response systems.

This project develops a YOLO26-based object detection system capable of detecting:

- 🔥 Fire
- 💨 Smoke
- 📦 Other

The trained model is integrated into a **Streamlit + WebRTC real-time application** that can process webcam frames continuously.

When fire is detected, the system:

1. Detects the fire using YOLO26.
2. Draws the detection bounding box.
3. Generates exactly one snapshot for the fire event.
4. Sends an automated email alert.
5. Attaches the detection snapshot to the email.
6. Records the event in a persistent alert-history CSV file.
7. Prevents repeated alerts while the same fire event remains visible.

---

# 🏗️ Complete System Architecture

```text
                    Input Dataset
                         │
                         ▼
                 Dataset Audit
                         │
                         ▼
             Annotation Visualization
                         │
                         ▼
                Dataset Splitting
                         │
                         ▼
                Dataset Validation
                         │
                         ▼
              YOLO Dataset Config
                         │
                         ▼
                  YOLO26 Training
                         │
                         ▼
                Model Evaluation
                         │
                         ▼
                   best.pt
                         │
                         ▼
              Real-Time Inference
                         │
                         ▼
                  WebRTC Camera
                         │
                         ▼
                    YOLO26n
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
            Fire       Smoke      Other
              │
              ▼
         Alert Manager
              │
       ┌──────┼─────────┐
       │      │         │
       ▼      ▼         ▼
   Snapshot Email   Alert History
              │         │
              ▼         ▼
           Gmail      CSV Log
🎯 Objectives
The main objectives of this project are:
1. Build a fire and smoke object detection system.
2. Prepare and validate an object detection dataset.
3. Maintain class distribution across train, validation, and test sets.
4. Train a YOLO26 object detection model.
5. Evaluate the trained model using standard object detection metrics.
6. Perform inference on unseen images.
7. Develop real-time webcam-based detection.
8. Generate automated fire alerts.
9. Capture a detection snapshot when a new fire event occurs.
10. Maintain persistent alert history.
11. Build a modular application suitable for deployment.
🗂️ Dataset
The dataset was obtained from Roboflow Universe and contains approximately 8,939 images with three object classes.
Classes
Class ID	Class
0	Fire
1	Other
2	Smoke


The dataset uses the standard YOLO annotation format.
Each image has a corresponding .txt annotation file containing:
class_id x_center y_center width height
The bounding-box coordinates are normalized to the range [0, 1].
📊 Dataset Statistics
The original dataset contains:
- 8,939 images
- 8,939 label files
- 24,091 total annotations
Annotation Distribution
Class	Annotations	Images containing class
Fire	13,914	7,704
Other	4,845	2,182
Smoke	5,332	3,601


The number of images containing each class does not add up to the total number of images because a single image can contain multiple classes.
Class Combinations
The dataset contains different combinations of objects, including:
- Fire
- Smoke
- Other
- Fire + Smoke
- Fire + Other
- Other + Smoke
- Fire + Other + Smoke
🔍 Dataset Quality Analysis
Before model training, the dataset was systematically checked for annotation and file-level problems.
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
Audit Result
Images without labels       : 0
Labels without images       : 0
Empty label files           : 0
Invalid annotations         : 0
This ensured that the dataset was structurally suitable for object detection training.
🖼️ Annotation Visualization
Sample images were visualized with their corresponding bounding boxes and class labels before training.
This was used to verify:
- Bounding boxes are correctly positioned.
- Class IDs correspond to the intended classes.
- Images and labels are correctly paired.
- Annotations are visually meaningful.
The visualization code is available in:
src/visualize_annotation.py
✂️ Dataset Splitting
The original dataset contained only a train directory.
The dataset was split into:
Train       → 70%
Validation  → 20%
Test        → 10%
A class-combination-aware splitting strategy was used to maintain approximately similar class distributions across the three subsets.
Final Split
Split	Images
Train	6,254
Validation	1,784
Test	901
Total	8,939


The total number of annotations remained 24,091, confirming that no annotations were lost during dataset preparation.
📁 Project Structure
Fire-Detection-YOLO26/
│
├── app.py
├── app_backup.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
│
├── models/
│   └── best.pt
│
├── configs/
│   └── data.yaml
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
├── src/
│   ├── __init__.py
│   │
│   ├── config.py
│   ├── model.py
│   ├── detector.py
│   ├── snapshot.py
│   ├── alert_manager.py
│   ├── alert_history.py
│   ├── email_alert.py
│   │
│   ├── data_analysis.py
│   ├── split_dataset.py
│   ├── validate_dataset.py
│   └── visualize_annotation.py
│
├── notebooks/
│
├── visualizations/
│
├── snapshots/
│
└── logs/
    └── alert_history.csv
⚙️ YOLO Dataset Configuration
The dataset configuration is stored in:
configs/data.yaml
Configuration:
path: ../data/processed

train: train/images
val: valid/images
test: test/images

names:
  0: fire
  1: other
  2: smoke
🐍 Environment Setup
Python 3.12 was used for the project environment.
The project uses uv for Python environment and dependency management.
Create the environment:
uv venv --python 3.12
Activate the environment on Windows:
.venv\Scripts\activate
Install dependencies:
uv pip install -r requirements.txt
Alternatively, if dependencies are managed through the project configuration:
uv sync
📦 Main Technologies
The project uses:
- Python 3.12
- YOLO26
- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- PyYAML
- Streamlit
- streamlit-webrtc
- SMTP / Gmail
- uv
Model training was performed using an NVIDIA GPU environment.
🔬 Dataset Preparation Pipeline
1. Dataset Audit
The dataset was analyzed using:
python src/data_analysis.py
The script checks dataset structure, annotation statistics, class distribution, and annotation validity.
2. Annotation Visualization
python src/visualize_annotation.py
This generates sample images with bounding boxes and class labels.
3. Dataset Splitting
python src/split_dataset.py
This creates:
train/
valid/
test/
with the required image and label structure.
4. Dataset Validation
python src/validate_dataset.py
This performs final structural and annotation validation before model training.
🚀 YOLO26 Model Training
The final object detection model is:
YOLO26n
The trained model contains:
Parameters : 2,375,421
GFLOPs     : 5.3
Layers     : 120
The best checkpoint was obtained at:
Epoch: 55
The trained model is stored as:
models/best.pt
📈 Model Evaluation
The model was evaluated using:
- Precision
- Recall
- mAP@50
- mAP@50-95
- Per-class performance
Validation Performance
Class	Precision	Recall	mAP50	mAP50-95
Fire	0.6215	0.6060	0.6364	0.3247
Other	0.4324	0.3284	0.3199	0.1270
Smoke	0.5608	0.4252	0.4602	0.2207
Overall	0.5382	0.4532	0.4722	0.2241


Test Performance
Class	Precision	Recall	mAP50	mAP50-95
Fire	0.5982	0.5865	0.6017	0.3061
Other	0.4259	0.2815	0.2741	0.1066
Smoke	0.6136	0.4451	0.4953	0.2323
Overall	0.5459	0.4377	0.4570	0.2150


The results show that the model provides meaningful detection performance across the three classes, with fire detection achieving a test mAP@50 of 0.6017.
🔎 Image Inference
The trained model can be used for inference on unseen images.
Example:
from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model("test_image.jpg")

for result in results:
    result.show()

The model predicts:
🔥 Fire
💨 Smoke
📦 Other
along with bounding boxes and confidence scores.
📷 Real-Time Webcam Detection
The trained YOLO26 model is integrated into a Streamlit application using streamlit-webrtc.
Run the application using:
uv run streamlit run app.py
The application provides:
- Real-time webcam inference
- YOLO bounding boxes
- Fire detection
- Smoke detection
- Other-object detection
- Confidence threshold control
- Detection counts
- Highest detection confidence
- FPS monitoring
- Real-time detection status
🚨 Automated Fire Alert System
When a new fire event is detected, the application automatically performs the following:
Fire detected
     ↓
Create one snapshot
     ↓
Create alert-history record
     ↓
Start background email thread
     ↓
Send Gmail alert
     ↓
Attach snapshot
     ↓
Update alert history
The email contains:
- Detection class
- Confidence
- Detection timestamp
- Attached detection snapshot
📸 Snapshot System
The system saves a snapshot when a new fire event occurs.
Snapshots are stored in:
snapshots/
Example:
snapshots/
└── fire_2026-09-23_16-30-42.jpg
Only one snapshot is generated for one continuous fire event.
If fire remains visible for several seconds, additional snapshots and emails are not generated.
When the fire disappears and a new fire event occurs later, a new snapshot and alert can be generated.
📧 Background Email Processing
Email transmission is performed using a background thread.
This prevents SMTP operations from blocking the real-time video-processing pipeline.
The architecture is:
WebRTC Camera
      │
      ▼
YOLO Inference
      │
      ▼
Fire Detection
      │
      ├──────────────► Continue Video Processing
      │
      ▼
Alert Manager
      │
      ▼
Background Email Thread
      │
      ▼
Gmail SMTP
This helps maintain responsive real-time video processing while the alert email is being sent.
📋 Alert History
Every new fire event is recorded in:
logs/alert_history.csv
The history contains:
- Alert ID
- Timestamp
- Detection class
- Confidence
- Snapshot filename
- Email status
Example:
alert_id,timestamp,class,confidence,snapshot,email_status
20260923163042123456,2026-09-23 16:30:42,fire,87.40%,fire_2026-09-23_16-30-42.jpg,Sent
The latest alerts are also displayed directly in the Streamlit dashboard.
🛡️ Duplicate Alert Prevention
A major part of the application is preventing repeated alerts for the same continuous fire event.
The logic is:
Fire appears
     ↓
New fire event?
     ↓
YES
     ↓
One snapshot
     ↓
One email
     ↓
Fire remains visible
     ↓
No additional alerts
     ↓
Fire disappears
     ↓
Reset alert state
     ↓
New fire appears later
     ↓
New snapshot + new email
An alert cooldown is also used to reduce repeated notifications.
🖥️ Streamlit Dashboard
The dashboard contains:
Live Camera
Real-time webcam feed with YOLO detections.
Detection Status
Displays:
- Fire detected
- Smoke detected
- No fire detected
Detection Counts
🔥 Fire
💨 Smoke
📦 Other
Detection Details
Highest Confidence
Detected Class
FPS
Alert Status
Displays the current email status.
Alert History
Displays the latest fire events recorded by the system.
🔐 Environment Variables
Email credentials are stored in a .env file.
Example:
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_google_app_password
EMAIL_RECEIVER=receiver_email@gmail.com
The .env file should never be committed to GitHub.
A safe .env.example file can be provided instead:
EMAIL_SENDER=
EMAIL_PASSWORD=
EMAIL_RECEIVER=
🐳 Docker Deployment
Dockerization is planned as the final deployment step.
The target architecture is:
Docker Container
      │
      ├── Streamlit
      ├── YOLO26
      ├── OpenCV
      ├── WebRTC
      └── Alert System
The application can then be packaged into a reproducible deployment environment.

🧠 Key Engineering Concepts
This project covers several important computer-vision and software-engineering concepts:
Computer Vision
- Object detection
- Bounding-box detection
- Real-time inference
- Confidence thresholds
- FPS measurement
Deep Learning
- YOLO architecture
- YOLO26
- Transfer/model training workflow
- Precision
- Recall
- mAP@50
- mAP@50-95
Dataset Engineering
- Dataset auditing
- Annotation validation
- Bounding-box visualization
- Multi-label dataset splitting
- Class-distribution analysis
Application Development
- Streamlit
- WebRTC
- Modular Python architecture
- Background threading
- SMTP email alerts
- Snapshot generation
- CSV-based persistent logging
📚 Project Learning Outcomes
Through this project, the following concepts were implemented:
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
- Real-time computer vision
- Streamlit application development
- WebRTC video processing
- Background task processing
- Automated email alerts
- Snapshot generation
- Persistent alert logging
- Modular application architecture
📌 Project Status
✅ Dataset collected
✅ Dataset audited
✅ Annotation visualization completed
✅ Dataset split completed
✅ Dataset validated
✅ YOLO configuration completed
✅ Python environment configured
✅ YOLO26 model trained
✅ Model evaluated
✅ Test-set inference completed
✅ Real-time webcam inference completed
✅ Streamlit dashboard completed
✅ Fire/smoke/other detection completed
✅ Automated email alert completed
✅ Background email processing completed
✅ Fire-detection snapshot completed
✅ Email snapshot attachment completed
✅ Duplicate alert prevention completed
✅ Persistent alert history completed
⬜ Docker deployment
⬜ Final GitHub cleanup/documentation
🚀 Future Improvements
Possible future extensions include:
- Docker-based deployment
- CCTV/RTSP stream integration
- Edge-device deployment
- GPU/CPU inference optimization
- Real-world surveillance evaluation
- Alert notification through additional communication channels
- Database-backed alert storage
- Model optimization for resource-constrained devices
👨‍💻 Author
Amrit Raj
M.Tech — Robotics & Artificial Intelligence
IIT Bhubaneswar
⭐ Acknowledgements
Dataset sourced from Roboflow Universe.
Model training and object detection are implemented using the Ultralytics YOLO framework.

### A couple of important changes I made

Your original README described things like **model training, evaluation, Streamlit deployment, and automated alerts as future work**. For example, the old status section still had YOLO26 training and deployment unchecked. :chatgpt-content-reference{index="1"}

I've changed that to reflect what you've **actually built now**:

- YOLO26n trained and evaluated
- Real-time WebRTC camera
- Streamlit dashboard
- Background email
- One snapshot per fire event
- Snapshot attached to email
- Alert history
- Duplicate-alert prevention

I also corrected the source filenames to match your **current project**: `data_analysis.py` and `visualize_annotation.py`, rather than the older names in the uploaded README. Your current README had `dataset_analysis.py` / `visualize_annotations.py` in several places. :chatgpt-content-reference{index="2"}

**One thing I would not claim yet:** Docker is still listed as pending, because we haven't actually built