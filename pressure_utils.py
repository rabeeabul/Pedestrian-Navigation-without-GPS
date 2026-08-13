import numpy as np
import pandas as pd


def make_pressure_dataframe(pressure_buffer):
    return pd.DataFrame(list(pressure_buffer), columns=["seconds_elapsed", "pressure"])


def pressure_to_altitude_series(pressure_values):
    # Barometric formula, taken relative to the first sample so every run starts at z = 0.
    p = pd.Series(pressure_values).astype(float)

    if len(p.dropna()) == 0:
        return pd.Series([], dtype=float)

    # Android usually reports hPa, but some phones report Pa.
    median_p = float(np.nanmedian(p))
    if median_p < 2000.0:
        p_pa = p * 100.0
    else:
        p_pa = p

    p0 = float(p_pa.dropna().iloc[0])

    altitude = 44330.0 * (1.0 - (p_pa / p0) ** (1.0 / 5.255))

    return altitude


def relative_altitude_at_time(pressure_buffer, event_time, smooth_seconds):
    # Mean altitude over the seconds before event_time. Averaging a couple of
    # seconds takes this phone's jitter from about 0.13 m down to about 0.03 m.
    pressure_df = make_pressure_dataframe(pressure_buffer)
    if len(pressure_df) == 0:
        return 0.0

    pressure_df = pressure_df.copy()
    pressure_df["altitude"] = pressure_to_altitude_series(pressure_df["pressure"])

    seg = pressure_df[(pressure_df["seconds_elapsed"] >= event_time - smooth_seconds) &
                      (pressure_df["seconds_elapsed"] <= event_time)]
    if len(seg) == 0:
        before = pressure_df[pressure_df["seconds_elapsed"] <= event_time]
        if len(before) == 0:
            return float(pressure_df["altitude"].iloc[0])
        return float(before["altitude"].iloc[-1])
    return float(seg["altitude"].mean())


def centered_altitude_at_time(pressure_buffer, event_time, half_window):
    # Same idea but averaging both sides of event_time, which removes the lag the
    # backward-looking version has. Only usable after the run, when the whole
    # pressure record exists, and it is what dates a climb to the right step.
    pressure_df = make_pressure_dataframe(pressure_buffer)
    if len(pressure_df) == 0:
        return 0.0

    pressure_df = pressure_df.copy()
    pressure_df["altitude"] = pressure_to_altitude_series(pressure_df["pressure"])

    seg = pressure_df[(pressure_df["seconds_elapsed"] >= event_time - half_window) &
                      (pressure_df["seconds_elapsed"] <= event_time + half_window)]
    if len(seg) == 0:
        nearest = (pressure_df["seconds_elapsed"] - event_time).abs().idxmin()
        return float(pressure_df["altitude"].loc[nearest])
    return float(seg["altitude"].mean())
