"""Measures one person's stride length.

Walk a path you have measured, with forward steps only, and press Enter at the end.
The steps are counted from AccY with the same thresholds the main program uses, so
the count matches what the main program would have counted, and

    stride = path length / steps

is saved under your ID in user_step_sizes.json. Run this once per person, before
the first real run. It never edits config.py.

    python calibrate_step_length.py
    python calibrate_step_length.py --id 1 --distance 10
"""

import time
import argparse

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import step_size_store
from buffer_utils import make_acc_dataframe
from signal_utils import filter_acc_axes
from config import (PHONE_IP, PHONE_PORT, POLL_INTERVAL, MIN_FILTER_SAMPLES,
                    FORWARD_STEP_HEIGHT, IGNORE_FIRST_SECONDS)


def count_forward_steps(acc_df):
    if len(acc_df) < MIN_FILTER_SAMPLES:
        return 0, []

    t, x_f, y_f, z_f = filter_acc_axes(acc_df)
    t_arr = np.asarray(t, dtype=float)
    dts = np.diff(t_arr)
    dts = dts[dts > 0]
    if len(dts) == 0:
        return 0, []
    fs = 1.0 / np.median(dts)

    peaks, _ = find_peaks(np.asarray(y_f, dtype=float),
                          height=FORWARD_STEP_HEIGHT,
                          distance=max(1, int(0.6 * fs)))

    times = [round(float(t.iloc[p]), 2) for p in peaks if t.iloc[p] >= IGNORE_FIRST_SECONDS]
    return len(times), times


def collect_live(client):
    # Imported here so the module only needs them in live mode.
    from main_ipwebcam import map_accel_row
    from keypress_stop import start_enter_to_stop

    print("\n>>> Walk the calibration path NOW, forward steps only, no crab or turns.")
    print(">>> Keep the phone on your chest as usual. Press Enter when you reach the end.\n")

    stop_requested = start_enter_to_stop()
    samples = []
    t0_ms = None
    last_ts = -1.0
    try:
        while not stop_requested():
            data = client.get_sensors_json(timeout=2.0)
            if data is None:
                print("[warn] no data this poll; is the phone connected? retrying...")
                time.sleep(POLL_INTERVAL)
                continue
            accel_arr, _ = client.parse_imu(data)
            if t0_ms is None and len(accel_arr) > 0:
                t0_ms = float(accel_arr[0][0])
            for row in accel_arr:
                ts_ms = float(row[0])
                if t0_ms is not None and ts_ms > last_ts:
                    last_ts = ts_ms
                    samples.append(map_accel_row((ts_ms - t0_ms) / 1000.0, row[1:4]))
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        pass
    print("\nStopped collecting. Analysing the walk...")

    return make_acc_dataframe(samples)


def load_accel_log(path):
    """Offline mode: calibrate from an accel_log.csv saved by an earlier run."""
    raw = pd.read_csv(path)
    needed = {"t_s", "lateral", "forward", "vertical"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError("%s missing columns %s" % (path, sorted(missing)))
    return pd.DataFrame({
        "seconds_elapsed": raw["t_s"],
        "x": raw["lateral"],
        "y": raw["forward"],
        "z": raw["vertical"],
    })


def _ask_float(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
            if val > 0:
                return val
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a number, e.g. 10")


def run(argv=None):
    ap = argparse.ArgumentParser(description="Per-user step-size calibration.")
    ap.add_argument("--id", default=None, help="user ID (skip the prompt)")
    ap.add_argument("--distance", type=float, default=None, help="path length in metres")
    ap.add_argument("--replay", default=None, help="calibrate from a saved accel_log.csv")
    args = ap.parse_args(argv)

    print("========  PER-USER STEP-SIZE CALIBRATION  ========")
    user_id = args.id if args.id is not None else input("Enter user ID (name or number, e.g. 1): ").strip()
    if not user_id:
        print("No ID entered. Nothing done.")
        return
    distance = args.distance if args.distance is not None else _ask_float(
        "Length of the calibration path in metres (about 10 is best): ")

    if args.replay:
        acc_df = load_accel_log(args.replay)
    else:
        from ipwebcam_client import IPWebcamClient
        client = IPWebcamClient(PHONE_IP, PHONE_PORT)
        acc_df = collect_live(client)

    n_steps, step_times = count_forward_steps(acc_df)
    print("\nDetected %d forward steps." % n_steps)
    if step_times:
        print("  step times (s): %s" % step_times)

    if n_steps <= 0:
        print("\nNo steps detected, nothing saved. Re-run and walk with clear forward steps.")
        return

    step_size = distance / n_steps
    step_size_store.save_user(user_id, step_size, distance, n_steps)

    print("\n==================================================")
    print(" User '%s' calibrated." % user_id)
    print("   path length : %.2f m" % distance)
    print("   steps       : %d" % n_steps)
    print("   STEP SIZE   : %.3f m   (= %.2f m / %d steps)" % (step_size, distance, n_steps))
    print(" Saved to user_step_sizes.json.")
    print("")
    print(" Enter '%s' as the ID when you run main_ipwebcam.py" % user_id)
    print(" and it will use %.3f m for every step." % step_size)
    print("==================================================")
    return step_size


if __name__ == "__main__":
    run()
