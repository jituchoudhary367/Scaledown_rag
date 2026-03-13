"""Metrics logger — appends results to JSON log."""
import json
from datetime import datetime
from src.config import LOG_PATH


def log_metrics(entry):
    import os
    entry["timestamp"] = str(datetime.now())
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    try:
        with open(LOG_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        data = []

    data.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)
