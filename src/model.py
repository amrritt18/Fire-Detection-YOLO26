from ultralytics import YOLO

from src.config import MODEL_PATH


def load_model():
    """
    Load the trained YOLO model.
    """

    return YOLO(MODEL_PATH)