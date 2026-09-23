import os
from datetime import datetime

import cv2


SNAPSHOT_DIR = "snapshots"


def save_snapshot(frame):
    """
    Save a detection frame to the snapshots directory.

    Parameters
    ----------
    frame : numpy.ndarray
        Annotated BGR frame.

    Returns
    -------
    str
        Path of the saved snapshot.
    """

    # Create directory if it doesn't exist
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = f"fire_{timestamp}.jpg"

    filepath = os.path.join(
        SNAPSHOT_DIR,
        filename,
    )

    # Save image
    success = cv2.imwrite(
        filepath,
        frame,
    )

    if not success:
        print("Failed to save fire snapshot.")
        return None

    print(f"Fire snapshot saved: {filepath}")

    return filepath