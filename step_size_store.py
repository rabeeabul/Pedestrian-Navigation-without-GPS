"""Stride length per person, saved in user_step_sizes.json and keyed by an ID."""

import os
import json
import datetime

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_step_sizes.json")


def load_all():
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r") as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def load_user_step_size(user_id):
    """Stride in metres for this user, or None if they have not calibrated yet."""
    entry = load_all().get(str(user_id))
    if not entry:
        return None
    try:
        return float(entry["step_size"])
    except (KeyError, TypeError, ValueError):
        return None


def save_user(user_id, step_size, path_length_m, steps):
    data = load_all()
    entry = {
        "step_size": round(float(step_size), 3),
        "path_length_m": round(float(path_length_m), 2),
        "steps": int(steps),
        "date": datetime.date.today().isoformat(),
    }
    data[str(user_id)] = entry
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return entry
