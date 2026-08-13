import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from config import (MIN_FILTER_SAMPLES, ACC_FILTER_LOW, ACC_FILTER_HIGH, ACC_FILTER_ORDER,
                    GYRO_FILTER_CUTOFF, GYRO_FILTER_ORDER)


def butter_bandpass_filter(signal, t, lowcut, highcut, order=3):
    signal = np.asarray(signal, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(signal) < MIN_FILTER_SAMPLES or len(np.unique(t)) < 3:
        return pd.Series(np.zeros(len(signal)))

    # The sample rate is not fixed, so it is measured from the timestamps each time.
    dt = np.median(np.diff(t))
    if dt <= 0 or not np.isfinite(dt):
        return pd.Series(np.zeros(len(signal)))

    fs = 1.0 / dt
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.99)

    if high <= low:
        return pd.Series(np.zeros(len(signal)))

    b, a = butter(order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return pd.Series(np.zeros(len(signal)))

    return pd.Series(filtfilt(b, a, signal))


def butter_lowpass_filter(signal, t, cutoff, order=3):
    signal = np.asarray(signal, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(signal) < MIN_FILTER_SAMPLES or len(np.unique(t)) < 3:
        return pd.Series(np.zeros(len(signal)))

    dt = np.median(np.diff(t))
    if dt <= 0 or not np.isfinite(dt):
        return pd.Series(np.zeros(len(signal)))

    fs = 1.0 / dt
    nyquist = 0.5 * fs
    normal_cutoff = min(max(cutoff / nyquist, 0.001), 0.99)

    b, a = butter(order, normal_cutoff, btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return pd.Series(np.zeros(len(signal)))

    return pd.Series(filtfilt(b, a, signal))


def local_max_abs(signal, t, start_time, end_time):
    mask = (t >= start_time) & (t <= end_time)
    if mask.sum() == 0:
        return 0.0
    return float(np.max(np.abs(signal[mask])))


def filter_acc_axes(acc):
    t = acc["seconds_elapsed"].reset_index(drop=True)
    x_f = butter_bandpass_filter(acc["x"].reset_index(drop=True), t,
                                 ACC_FILTER_LOW, ACC_FILTER_HIGH, ACC_FILTER_ORDER)
    y_f = butter_bandpass_filter(acc["y"].reset_index(drop=True), t,
                                 ACC_FILTER_LOW, ACC_FILTER_HIGH, ACC_FILTER_ORDER)
    z_f = butter_bandpass_filter(acc["z"].reset_index(drop=True), t,
                                 ACC_FILTER_LOW, ACC_FILTER_HIGH, ACC_FILTER_ORDER)
    return t, x_f, y_f, z_f


def filter_gyro_y(gyro):
    t = gyro["seconds_elapsed"].reset_index(drop=True)
    gy_f = butter_lowpass_filter(gyro["y"].reset_index(drop=True), t,
                                 GYRO_FILTER_CUTOFF, GYRO_FILTER_ORDER)
    return t, gy_f
