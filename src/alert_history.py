import csv
import os
import threading
from datetime import datetime


LOG_DIR = "logs"
HISTORY_FILE = os.path.join(
    LOG_DIR,
    "alert_history.csv",
)

# Protect CSV operations when the background
# email thread updates the history.
_history_lock = threading.Lock()


def _ensure_history_file():
    """
    Create the logs directory and CSV file
    if they do not already exist.
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(HISTORY_FILE):

        with open(
            HISTORY_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "alert_id",
                    "timestamp",
                    "class",
                    "confidence",
                    "snapshot",
                    "email_status",
                ]
            )


def create_alert(
    confidence,
    snapshot_path,
):
    """
    Create one alert-history record.

    Returns
    -------
    str
        Unique alert ID.
    """

    _ensure_history_file()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    alert_id = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    snapshot_name = ""

    if snapshot_path:
        snapshot_name = os.path.basename(
            snapshot_path
        )

    with _history_lock:

        with open(
            HISTORY_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    alert_id,
                    timestamp,
                    "fire",
                    f"{confidence * 100:.2f}%",
                    snapshot_name,
                    "Sending...",
                ]
            )

    return alert_id


def update_email_status(
    alert_id,
    email_status,
):
    """
    Update the email status of an existing alert.
    """

    _ensure_history_file()

    with _history_lock:

        rows = []

        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["alert_id"] == alert_id:

                    row["email_status"] = (
                        email_status
                    )

                rows.append(row)

        with open(
            HISTORY_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            fieldnames = [
                "alert_id",
                "timestamp",
                "class",
                "confidence",
                "snapshot",
                "email_status",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)


def get_alert_history(limit=20):
    """
    Return the latest alert records.

    Parameters
    ----------
    limit : int
        Maximum number of records to return.

    Returns
    -------
    list
        Latest alert records.
    """

    _ensure_history_file()

    with _history_lock:

        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            rows = list(reader)

    # Newest first
    rows.reverse()

    return rows[:limit]