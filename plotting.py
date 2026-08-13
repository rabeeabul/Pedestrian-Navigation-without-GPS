"""Live windows during a run, and the figures saved when it stops.

Live: the 3D path as it is built, and the movement signal with detected steps marked.
At the end: 3D map, filtered accelerometer, detected movements and turns, the two
detection signals, turn detection, a 2D map, and the barometer trace.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from config import (
    FORWARD_STEP_HEIGHT, CRAB_STEP_HEIGHT, TURN_THRESHOLD,
    MIN_FILTER_SAMPLES, LIVE_SMOOTH_WINDOW,
)
from buffer_utils import make_acc_dataframe, make_gyro_dataframe
from signal_utils import filter_acc_axes, filter_gyro_y
from movement_detection import detect_movement_peaks
from turn_detection import find_turn_candidates, accept_real_turns
from pressure_utils import make_pressure_dataframe, pressure_to_altitude_series

MOVEMENT_TYPES = {"step_forward", "step_backward", "left_crab", "right_crab"}
TURN_TYPES = {"left_turn", "right_turn"}


def moving_average(signal, window):
    if len(signal) < window:
        return signal
    return np.convolve(signal, np.ones(window) / window, mode="same")


def setup_live_plots():
    plt.ion()

    fig_map = plt.figure(figsize=(7, 6))
    ax_map = fig_map.add_subplot(111, projection="3d")
    ax_map.set_title("Live 3D PDR Map")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_zlabel("Z [m]")
    route_line, = ax_map.plot([], [], [], marker="o", label="Route")
    ax_map.legend(loc="upper right")

    fig_signal, ax_signal = plt.subplots(figsize=(11, 5))
    ax_signal.set_title("Live Movement Signal")
    ax_signal.set_xlabel("Time [s]")
    ax_signal.set_ylabel("Filtered signal")
    ax_signal.grid(True)
    line_forward, = ax_signal.plot([], [], label="Forward AccY")
    line_crab, = ax_signal.plot([], [], label="Crab abs(AccX)")
    peak_points, = ax_signal.plot([], [], "o", label="Detected step")
    ax_signal.axhline(FORWARD_STEP_HEIGHT, linestyle="--", label="Forward threshold")
    ax_signal.axhline(CRAB_STEP_HEIGHT, linestyle="--", label="Crab threshold")
    ax_signal.legend(loc="upper right")

    return {
        "fig_map": fig_map, "ax_map": ax_map, "route_line": route_line,
        "fig_signal": fig_signal, "ax_signal": ax_signal,
        "line_forward": line_forward, "line_crab": line_crab,
        "peak_points": peak_points, "live_artists": [],
    }


def _update_3d_map(h, route_rows):
    route = pd.DataFrame(route_rows)
    xs = route["x_value"].to_numpy()
    ys = route["y_value"].to_numpy()
    zs = route["z_value"].to_numpy()

    line = h["route_line"]
    line.set_data(xs, ys)
    line.set_3d_properties(zs)

    ax = h["ax_map"]

    def lim(a):
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi - lo < 1.0:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 1.0, mid + 1.0
        m = 0.1 * (hi - lo)
        return lo - m, hi + m
    ax.set_xlim(*lim(xs))
    ax.set_ylim(*lim(ys))
    ax.set_zlim(*lim(zs))


def _update_signal(h, acc_t, forward_signal, crab_signal, confirmed_events):
    if len(acc_t) == 0:
        return
    ax = h["ax_signal"]

    # The markers are redrawn every frame, so the old ones have to go first.
    for artist in h["live_artists"]:
        try:
            artist.remove()
        except Exception:
            pass
    h["live_artists"] = []

    t_min = float(acc_t.iloc[0])
    t_max = float(acc_t.iloc[-1])

    fwd = moving_average(forward_signal.to_numpy(), LIVE_SMOOTH_WINDOW)
    crab = moving_average(crab_signal.to_numpy(), LIVE_SMOOTH_WINDOW)
    h["line_forward"].set_data(acc_t, fwd)
    h["line_crab"].set_data(acc_t, crab)

    acc_arr = acc_t.to_numpy()
    moves = [e for e in confirmed_events
             if e["type"] in MOVEMENT_TYPES and t_min <= e["time"] <= t_max]
    if moves:
        times, vals = [], []
        for e in moves:
            idx = int(np.argmin(np.abs(acc_arr - e["time"])))
            times.append(e["time"])
            vals.append(max(float(fwd[idx]), float(crab[idx])))
            marker = ax.axvline(e["time"], linestyle=":", linewidth=1.3)
            h["live_artists"].append(marker)
            label = {"step_forward": "fwd", "step_backward": "back",
                     "left_crab": "L crab", "right_crab": "R crab"}.get(e["type"], "step")
            h["live_artists"].append(
                ax.text(e["time"], FORWARD_STEP_HEIGHT + 0.15, label,
                        rotation=90, fontsize=8, ha="center"))
        h["peak_points"].set_data(times, vals)
    else:
        h["peak_points"].set_data([], [])

    for e in [e for e in confirmed_events if e["type"] in TURN_TYPES and t_min <= e["time"] <= t_max]:
        h["live_artists"].append(ax.axvspan(e["time"] - 0.5, e["time"] + 0.5, alpha=0.2))
        h["live_artists"].append(
            ax.text(e["time"], FORWARD_STEP_HEIGHT + 0.45, e["type"].replace("_", " "),
                    rotation=90, fontsize=8, ha="center"))

    ax.set_xlim(t_min, t_max)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)


def update_live_plots(handles, acc_t, forward_signal, crab_signal, confirmed_events, route_rows):
    _update_signal(handles, acc_t, forward_signal, crab_signal, confirmed_events)
    _update_3d_map(handles, route_rows)
    plt.pause(0.001)


def close_live_plots():
    try:
        plt.close("all")
    except Exception:
        pass


MOVE_STYLE = {
    "step_forward":  ("#1f77b4", "Forward"),
    "step_backward": ("#d62728", "Backward"),
    "left_crab":     ("#2ca02c", "Crab"),
    "right_crab":    ("#2ca02c", "Crab"),
    "stair_up":      ("#ff7f0e", "Stairs"),
    "stair_down":    ("#ff7f0e", "Stairs"),
    "elevator_up":   ("#9467bd", "Elevator"),
    "elevator_down": ("#9467bd", "Elevator"),
}
TURN_STYLE = {
    "left_turn":  ("^", "black", "Turn"),
    "right_turn": ("^", "black", "Turn"),
}
DEFAULT_MOVE_COLOR = "#7f7f7f"


def _draw_colored_route(ax, route, is_3d):
    # Each segment is coloured by the movement that produced it, and the legend
    # gets one entry per movement type that actually appears in the run.
    xs = route["x_value"].to_numpy()
    ys = route["y_value"].to_numpy()
    zs = route["z_value"].to_numpy()
    types = route["step_type"].tolist()
    seen = {}

    def _plot(i, j, color, **kw):
        if is_3d:
            ax.plot(xs[i:j], ys[i:j], zs[i:j], color=color, **kw)
        else:
            ax.plot(xs[i:j], ys[i:j], color=color, **kw)

    def _scatter(i, color, marker, s, z):
        if is_3d:
            ax.scatter(xs[i], ys[i], zs[i], color=color, marker=marker, s=s)
        else:
            ax.scatter(xs[i], ys[i], color=color, marker=marker, s=s, zorder=z)

    for i in range(1, len(route)):
        stype = types[i]
        if stype in TURN_STYLE:
            continue
        color, label = MOVE_STYLE.get(stype, (DEFAULT_MOVE_COLOR, stype))
        _plot(i - 1, i + 1, color, marker="o", markersize=3, linewidth=2)
        if label not in seen:
            seen[label] = Line2D([0], [0], color=color, marker="o", label=label)

    for i in range(1, len(route)):
        stype = types[i]
        if stype in TURN_STYLE:
            marker, color, label = TURN_STYLE[stype]
            _scatter(i, color, marker, 70, 5)
            if label not in seen:
                seen[label] = Line2D([0], [0], color=color, marker=marker,
                                     linestyle="None", label=label)

    _scatter(0, "black", "o", 90, 6)
    _scatter(len(route) - 1, "black", "X", 110, 6)
    seen["Start"] = Line2D([0], [0], color="black", marker="o", linestyle="None", label="Start")
    seen["End"] = Line2D([0], [0], color="black", marker="X", linestyle="None", label="End")
    return list(seen.values())


def save_analysis_plots(route_rows, full_acc_buffer, full_gyro_buffer,
                        confirmed_events, pressure_buffer=None, show=True):
    # The signals are filtered again here over the whole run, not the rolling
    # window, so the saved figures show the complete walk.
    route = pd.DataFrame(route_rows)
    acc = make_acc_dataframe(full_acc_buffer)
    gyro = make_gyro_dataframe(full_gyro_buffer)

    if len(acc) < MIN_FILTER_SAMPLES:
        print("Not enough accelerometer data for analysis plots.")
        return

    acc_t, x_f, y_f, z_f = filter_acc_axes(acc)
    _, movement_signal, forward_signal, crab_signal = detect_movement_peaks(acc_t, x_f, y_f, z_f)

    if len(gyro) >= MIN_FILTER_SAMPLES:
        gyro_t, gyro_y_f = filter_gyro_y(gyro)
        accepted_turns, _ = accept_real_turns(find_turn_candidates(gyro_t, gyro_y_f),
                                              acc_t, x_f, y_f, z_f)
    else:
        gyro_t, gyro_y_f, accepted_turns = pd.Series([], dtype=float), pd.Series([], dtype=float), []

    fig1 = plt.figure(figsize=(9, 7))
    ax1 = fig1.add_subplot(111, projection="3d")
    handles1 = _draw_colored_route(ax1, route, is_3d=True)
    ax1.set_title("3D PDR Map, colored by movement type")
    ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]"); ax1.set_zlabel("Z [m]")
    ax1.legend(handles=handles1, loc="upper left", fontsize=8)
    fig1.savefig("fig1_3d_pdr_map.png", dpi=200, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(acc_t, x_f, label="Acc X (lateral)")
    ax2.plot(acc_t, y_f, label="Acc Y (forward/bounce)")
    ax2.plot(acc_t, z_f, label="Acc Z")
    ax2.set_title("Filtered Accelerometer X, Y, Z")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Filtered acceleration"); ax2.grid(True); ax2.legend()
    fig2.savefig("fig2_filtered_acc_xyz.png", dpi=200, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(acc_t, movement_signal, label="Movement signal")
    moves = [e for e in confirmed_events if e["type"] in MOVEMENT_TYPES]
    if moves:
        acc_arr = acc_t.to_numpy()
        ax3.scatter([e["time"] for e in moves],
                    [float(movement_signal.iloc[int(np.argmin(np.abs(acc_arr - e["time"])))]) for e in moves],
                    label="Detected movements", zorder=5)
    for turn in accepted_turns:
        ax3.axvspan(turn["start"], turn["end"], alpha=0.2, label="Turn region")
    ax3.set_title("Detected Movements and Turn Regions")
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Movement signal"); ax3.grid(True)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
    fig3.savefig("fig3_detected_movements_turns.png", dpi=200, bbox_inches="tight")

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(acc_t, forward_signal, label="Forward: filtered AccY")
    ax4.plot(acc_t, crab_signal, label="Crab: abs(filtered AccX)")
    ax4.axhline(FORWARD_STEP_HEIGHT, linestyle="--", label=f"Forward threshold {FORWARD_STEP_HEIGHT}")
    ax4.axhline(CRAB_STEP_HEIGHT, linestyle="--", label=f"Crab threshold {CRAB_STEP_HEIGHT}")
    ax4.set_title("Forward and Crab Detection Signals")
    ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Filtered signal"); ax4.grid(True); ax4.legend()
    fig4.savefig("fig4_forward_crab_signals.png", dpi=200, bbox_inches="tight")

    if len(gyro_t) > 0:
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        ax5.plot(gyro_t, gyro_y_f, label="Filtered Gyro Y")
        ax5.axhline(TURN_THRESHOLD, linestyle="--", label="Left turn threshold")
        ax5.axhline(-TURN_THRESHOLD, linestyle="--", label="Right turn threshold")
        for turn in accepted_turns:
            ax5.scatter(turn["time"], turn["value"], zorder=5)
        ax5.set_title("Turn Detection (Gyro Y)")
        ax5.set_xlabel("Time [s]"); ax5.set_ylabel("Filtered gyro Y"); ax5.grid(True); ax5.legend()
        fig5.savefig("fig5_turn_detection.png", dpi=200, bbox_inches="tight")

    fig6, ax6 = plt.subplots(figsize=(8, 7))
    handles6 = _draw_colored_route(ax6, route, is_3d=False)
    for i in range(len(route)):
        # The small offset keeps consecutive labels from sitting on top of each other.
        ax6.text(route["x_value"].iloc[i] + 0.03 * (i % 3),
                 route["y_value"].iloc[i] + 0.03 * (i % 4), str(i), fontsize=7)
    ax6.set_title("2D PDR Map, colored by movement type")
    ax6.set_xlabel("X [m]"); ax6.set_ylabel("Y [m]"); ax6.axis("equal"); ax6.grid(True)
    ax6.legend(handles=handles6, loc="best", fontsize=8)
    fig6.savefig("fig6_2d_pdr_map.png", dpi=200, bbox_inches="tight")

    if pressure_buffer is not None and len(pressure_buffer) > 0:
        pdf = make_pressure_dataframe(pressure_buffer)
        pdf["altitude"] = pressure_to_altitude_series(pdf["pressure"])
        fig7, ax7 = plt.subplots(figsize=(12, 5))
        ax7.plot(pdf["seconds_elapsed"], pdf["pressure"], label="Pressure")
        ax7.set_title("Barometer Pressure and Estimated Elevation")
        ax7.set_xlabel("Time [s]"); ax7.set_ylabel("Pressure"); ax7.grid(True)
        ax7b = ax7.twinx()
        ax7b.plot(pdf["seconds_elapsed"], pdf["altitude"], linestyle="--", label="Relative altitude [m]")
        ax7b.set_ylabel("Relative altitude [m]")
        for e in confirmed_events:
            ax7.axvline(e["time"], linestyle=":", linewidth=0.8)
        lines_a, labels_a = ax7.get_legend_handles_labels()
        lines_b, labels_b = ax7b.get_legend_handles_labels()
        ax7.legend(lines_a + lines_b, labels_a + labels_b, loc="best")
        fig7.savefig("fig7_barometer_elevation.png", dpi=200, bbox_inches="tight")
        print("Saved fig7_barometer_elevation.png")

    print("Saved analysis figures: fig1..fig6 (.png in this folder).")
    if show:
        keep = fig1.number
        for num in plt.get_fignums():
            if num != keep:
                plt.close(num)
        plt.ioff()
        plt.show()
