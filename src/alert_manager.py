import time

from src.config import ALERT_COOLDOWN
from src.email_alert import send_fire_alert


class AlertManager:

    def __init__(self):

        self.last_alert_time = 0

        self.fire_alert_sent = False

        self.email_status = "No alert sent"


    def process_fire_detection(
        self,
        fire_detected,
        confidence,
    ):
        """
        Decide whether a fire email should be sent.
        """

        current_time = time.time()

        # ----------------------------------------------------
        # Fire detected
        # ----------------------------------------------------

        if fire_detected:

            cooldown_finished = (
                current_time - self.last_alert_time
                >= ALERT_COOLDOWN
            )

            if (
                not self.fire_alert_sent
                and cooldown_finished
            ):

                success = send_fire_alert(
                    confidence
                )

                if success:

                    self.fire_alert_sent = True

                    self.last_alert_time = (
                        current_time
                    )

                    self.email_status = (
                        "📧 Fire alert email sent"
                    )

                else:

                    self.email_status = (
                        "⚠️ Email failed"
                    )

        # ----------------------------------------------------
        # No fire
        # ----------------------------------------------------

        else:

            # Reset the current fire event.
            self.fire_alert_sent = False

        return self.email_status