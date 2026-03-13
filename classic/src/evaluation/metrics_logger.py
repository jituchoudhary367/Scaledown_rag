import json
from datetime import datetime
from src.config import LOG_PATH

def log_metrics(entry):
    entry["timestamp"] = str(datetime.now())

    try:
        with open(LOG_PATH, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)