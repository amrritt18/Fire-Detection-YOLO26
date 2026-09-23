import time

import cv2
from ultralytics import YOLO

from src.alert_manager import AlertManager
from src.snapshot import save_snapshot


class FireDetector:
    """
    Handles YOLO inference, detection statistics,
    FPS calculation, snapshots and alerts.
    """

    def __init__(
        self,
        model: YOLO,
        confidence: float = 0.25,
    ):

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

        # Alert manager
        self.alert_manager = AlertManager()

        # Current snapshot
        self.last_snapshot_path = None

    def process_frame(self, frame):
        """
        Run YOLO inference on one frame.
        """

        # ====================================================
        # RESET FRAME STATISTICS
        # ====================================================

        self.fire_count = 0
        self.smoke_count = 0
        self.other_count = 0

        self.max_confidence = 0.0
        self.detected_class = "None"

        # ====================================================
        # YOLO INFERENCE
        # ====================================================

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]

        # ====================================================
        # DETECTION PROCESSING
        # ====================================================

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = self.model.names.get(
                    class_id,
                    str(class_id),
                )

                # -----------------------------
                # Count detections
                # -----------------------------

                if class_name == "fire":

                    self.fire_count += 1

                elif class_name == "smoke":

                    self.smoke_count += 1

                elif class_name == "other":

                    self.other_count += 1

                # -----------------------------
                # Highest confidence
                # -----------------------------

                if confidence > self.max_confidence:

                    self.max_confidence = confidence

                    self.detected_class = class_name

        # ====================================================
        # DRAW YOLO DETECTIONS
        # ====================================================

        annotated_frame = result.plot()

        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        elapsed_time = (
            current_time
            - self.previous_time
        )

        if elapsed_time > 0:

            self.fps = 1.0 / elapsed_time

        self.previous_time = current_time

        cv2.putText(
            annotated_frame,
            f"FPS: {self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # ====================================================
        # FIRE ALERT DECISION
        # ====================================================

        fire_detected = (
            self.fire_count > 0
        )

        should_alert = (
            self.alert_manager.should_create_alert(
                fire_detected
            )
        )

        # ====================================================
        # CREATE EXACTLY ONE SNAPSHOT
        # ====================================================

        snapshot_path = None

        if should_alert:

            snapshot_path = save_snapshot(
                annotated_frame
            )

            self.last_snapshot_path = snapshot_path

            # ------------------------------------------------
            # Start background email
            # ------------------------------------------------

            self.alert_manager.send_alert(
                confidence=self.max_confidence,
                snapshot_path=snapshot_path,
            )

        # ====================================================
        # UPDATE ALERT STATUS
        # ====================================================

        self.alert_status = (
            self.alert_manager.email_status
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
            "snapshot_path": self.last_snapshot_path,
        }