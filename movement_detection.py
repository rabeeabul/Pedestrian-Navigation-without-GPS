import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from config import (IGNORE_FIRST_SECONDS, STEP_MIN_TIME_BETWEEN_MOVEMENTS, FORWARD_STEP_HEIGHT,
                    CRAB_STEP_HEIGHT, CRAB_STEP_PROMINENCE, CRAB_RATIO_TO_FORWARD,
                    CRAB_LEFT_IS_POSITIVE_X, TURN_MOVEMENT_REMOVE_PADDING, MIN_FILTER_SAMPLES)
from signal_utils import local_max_abs


def merge_close_peaks(candidate_peaks, t, score_signal):
    # One step can raise a peak on both the forward and the crab signal. Whichever
    # peak is taller wins, and the other is dropped.
    if len(candidate_peaks) == 0:
        return []

    candidate_peaks = sorted(set(candidate_peaks))

    t_arr = np.asarray(t)

    time_diff = np.diff(t_arr)
    time_diff = time_diff[time_diff > 0]

    if len(time_diff) == 0:
        return candidate_peaks

    dt = np.median(time_diff)
    fs = 1.0 / dt

    min_gap_samples = max(1, int(STEP_MIN_TIME_BETWEEN_MOVEMENTS * fs))

    merged = []
    current_best = candidate_peaks[0]

    for p in candidate_peaks[1:]:
        if p - current_best < min_gap_samples:
            if score_signal.iloc[p] > score_signal.iloc[current_best]:
                current_best = p
        else:
            merged.append(current_best)
            current_best = p

    merged.append(current_best)
    return merged


def detect_movement_peaks(acc_t, x_f, y_f, z_f):
    forward_signal = pd.Series(y_f).reset_index(drop=True)
    crab_signed = pd.Series(x_f).reset_index(drop=True)
    crab_signal = crab_signed.abs()

    if len(acc_t) < MIN_FILTER_SAMPLES:
        movement_signal = pd.Series(np.zeros(len(acc_t)))
        return [], movement_signal, forward_signal, crab_signal

    time_diff = np.diff(acc_t)
    time_diff = time_diff[time_diff > 0]

    if len(time_diff) == 0:
        movement_signal = pd.Series(np.zeros(len(acc_t)))
        return [], movement_signal, forward_signal, crab_signal

    dt = np.median(time_diff)
    fs = 1.0 / dt

    forward_peaks, _ = find_peaks(
        forward_signal,
        height=FORWARD_STEP_HEIGHT,
        distance=int(0.6 * fs)
    )

    crab_peaks, _ = find_peaks(
        crab_signal,
        height=CRAB_STEP_HEIGHT,
        prominence=CRAB_STEP_PROMINENCE,
        distance=int(0.5 * fs)
    )

    candidate_peaks = []

    for p in forward_peaks:
        if acc_t.iloc[p] >= IGNORE_FIRST_SECONDS:
            # A walking peak has to clearly beat the sideways signal at the same moment.
            if forward_signal.iloc[p] > CRAB_RATIO_TO_FORWARD * crab_signal.iloc[p]:
                candidate_peaks.append(p)

    for p in crab_peaks:
        if acc_t.iloc[p] >= IGNORE_FIRST_SECONDS:
            candidate_peaks.append(p)

    movement_signal = pd.Series(np.maximum(np.maximum(forward_signal, 0), crab_signal))

    peaks = merge_close_peaks(candidate_peaks, acc_t, movement_signal)

    return peaks, movement_signal, forward_signal, crab_signal


def estimate_lateral_displacement(acc_t, x_f, center_time, window_before=0.75, window_after=0.75):
    # Integrating acceleration twice to get which way the crab step went. Over a
    # window this short the drift is small, and the linear term is removed below.
    start_time = center_time - window_before
    end_time = center_time + window_after
    mask = (acc_t >= start_time) & (acc_t <= end_time)
    t_local = acc_t[mask].reset_index(drop=True)
    x_local = x_f[mask].reset_index(drop=True)

    if len(x_local) < 5:
        return 0.0

    x_local = x_local - x_local.mean()
    t_vals = t_local.to_numpy()
    a_vals = x_local.to_numpy()
    velocity = np.zeros(len(a_vals))

    for i in range(1, len(a_vals)):
        dt = t_vals[i] - t_vals[i - 1]
        velocity[i] = velocity[i - 1] + 0.5 * (a_vals[i] + a_vals[i - 1]) * dt

    velocity_drift = np.linspace(velocity[0], velocity[-1], len(velocity))
    velocity_corrected = velocity - velocity_drift

    displacement = 0.0
    for i in range(1, len(velocity_corrected)):
        dt = t_vals[i] - t_vals[i - 1]
        displacement += 0.5 * (velocity_corrected[i] + velocity_corrected[i - 1]) * dt

    return displacement


def classify_movement_peak(peak_idx, acc_t, x_f, y_f, z_f):
    peak_time = acc_t.iloc[peak_idx]

    x_max = local_max_abs(x_f, acc_t, peak_time - 0.30, peak_time + 0.30)

    if x_max >= CRAB_STEP_HEIGHT:
        lateral_displacement = estimate_lateral_displacement(acc_t, x_f, peak_time)

        if CRAB_LEFT_IS_POSITIVE_X:
            return "left_crab" if lateral_displacement > 0 else "right_crab"
        else:
            return "right_crab" if lateral_displacement > 0 else "left_crab"

    return "step_forward"


def remove_movements_inside_turns(movement_events, turn_events):
    # While you turn, the body sways enough to look like steps. Everything inside
    # the turn window plus a margin is thrown away.
    cleaned_events = []

    for movement in movement_events:
        movement_time = movement["time"]
        inside_turn = False

        for turn in turn_events:
            if turn["start"] - TURN_MOVEMENT_REMOVE_PADDING <= movement_time <= turn["end"] + TURN_MOVEMENT_REMOVE_PADDING:
                inside_turn = True
                break

        if not inside_turn:
            cleaned_events.append(movement)

    return cleaned_events
