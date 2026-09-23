import threading
import time

from src.alert_history import (
    create_alert,
    update_email_status,
)
from src.config import ALERT_COOLDOWN
from src.email_alert import send_fire_alert


class AlertManager:
    """
    Manages fire alerts.

    One continuous fire event generates:
        - One snapshot
        - One email
        - One history record
    """

    def __init__(self):

        self.last_alert_time = 0

        self.fire_alert_sent = False

        self.email_status = "No alert sent"

        self.email_thread = None

        # Current alert history ID
        self.current_alert_id = None

    def should_create_alert(
        self,
        fire_detected,
    ):
        """
        Decide whether a new fire alert should be created.
        """

        current_time = time.time()

        # ----------------------------------------------------
        # No fire
        # ----------------------------------------------------

        if not fire_detected:

            self.fire_alert_sent = False

            return False

        # ----------------------------------------------------
        # Fire detected
        # ----------------------------------------------------

        cooldown_finished = (
            current_time - self.last_alert_time
            >= ALERT_COOLDOWN
        )

        thread_finished = (
            self.email_thread is None
            or not self.email_thread.is_alive()
        )

        # ----------------------------------------------------
        # New alert
        # ----------------------------------------------------

        if (
            not self.fire_alert_sent
            and cooldown_finished
            and thread_finished
        ):

            self.fire_alert_sent = True

            self.last_alert_time = current_time

            return True

        return False

    def send_alert(
        self,
        confidence,
        snapshot_path,
    ):
        """
        Create one history record and start
        the email in a background thread.
        """

        # ====================================================
        # CREATE HISTORY RECORD
        # ====================================================

        self.current_alert_id = create_alert(
            confidence=confidence,
            snapshot_path=snapshot_path,
        )

        self.email_status = (
            "📧 Sending fire alert..."
        )

        # ====================================================
        # BACKGROUND EMAIL
        # ====================================================

        self.email_thread = threading.Thread(
            target=self._send_alert_in_background,
            args=(
                confidence,
                snapshot_path,
                self.current_alert_id,
            ),
            daemon=True,
        )

        self.email_thread.start()

    def _send_alert_in_background(
        self,
        confidence,
        snapshot_path,
        alert_id,
    ):
        """
        Send email in the background and update
        the corresponding history record.
        """

        success = send_fire_alert(
            confidence=confidence,
            snapshot_path=snapshot_path,
        )

        if success:

            self.email_status = (
                "📧 Fire alert email sent"
            )

            update_email_status(
                alert_id,
                "Sent",
            )

        else:

            self.email_status = (
                "⚠️ Email failed"
            )

            update_email_status(
                alert_id,
                "Failed",
            )

            # Allow a future attempt after failure.
            self.fire_alert_sent = False

    def process_fire_detection(
        self,
        fire_detected,
        confidence,
        snapshot_path=None,
    ):
        """
        Backward-compatible alert processing method.
        """

        should_alert = (
            self.should_create_alert(
                fire_detected
            )
        )

        if should_alert:

            self.send_alert(
                confidence=confidence,
                snapshot_path=snapshot_path,
            )

        return self.email_status