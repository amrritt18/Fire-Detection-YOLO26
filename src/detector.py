import time

import cv2
from ultralytics import YOLO

from src.alert_manager import AlertManager


class FireDetector:
    """
    Handles YOLO inference, detection statistics,
    FPS calculation, and fire alert processing.
    """

    def __init__(self, model: YOLO, confidence: float = 0.25):
        self.model = model
        self.confidence = confidence

        # Detection statistics
        self.fire_count = 0
        self.smoke_count = 0
        self.other_count = 0

        self.max_confidence = 0.0
        self.detected_class = "None"

        # FPS
        self.fps = 0.0
        self.previous_time = time.time()

        # Email alert manager
        self.alert_manager = AlertManager()

    def process_frame(self, frame):
        """
        Run YOLO inference on one frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image/frame.

        Returns
        -------
        annotated_frame : numpy.ndarray
            Frame with YOLO detections drawn.
        """

        # Reset statistics for current frame
        self.fire_count = 0
        self.smoke_count = 0
        self.other_count = 0
        self.max_confidence = 0.0
        self.detected_class = "None"

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]

        # Process detections
        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = self.model.names.get(
                    class_id,
                    str(class_id)
                )

                # Count detections
                if class_name == "fire":
                    self.fire_count += 1

                elif class_name == "smoke":
                    self.smoke_count += 1

                elif class_name == "other":
                    self.other_count += 1

                # Highest confidence
                if confidence > self.max_confidence:
                    self.max_confidence = confidence
                    self.detected_class = class_name

        # Check fire alert
        fire_detected = self.fire_count > 0

        self.alert_status = self.alert_manager.process_fire_detection(
            fire_detected=fire_detected,
            confidence=self.max_confidence,
        )

        # Draw YOLO annotations
        annotated_frame = result.plot()

        # Calculate FPS
        current_time = time.time()

        elapsed_time = current_time - self.previous_time

        if elapsed_time > 0:
            self.fps = 1.0 / elapsed_time

        self.previous_time = current_time

        # Draw FPS on frame
        cv2.putText(
            annotated_frame,
            f"FPS: {self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        return annotated_frame

    def get_status(self):
        """
        Return current detection statistics.
        """

        return {
            "fire_count": self.fire_count,
            "smoke_count": self.smoke_count,
            "other_count": self.other_count,
            "max_confidence": self.max_confidence,
            "detected_class": self.detected_class,
            "fps": self.fps,
            "alert_status": self.alert_status,
        }