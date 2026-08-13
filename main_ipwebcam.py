"""Entry point. Run this to record a walk.

Each pass of the loop:
  1. read the newest IMU and barometer samples from the phone
  2. map the raw phone axes to lateral, forward and vertical
  3. filter, then find step peaks and turns
  4. for a step, ask the camera whether the scene expanded or contracted
  5. move the position and write a row to data.csv

When the run stops, the whole recording is available at once, so stairs, lifts and
the forward/backward direction are worked out again with everything in hand. That
second pass is what gets saved.
"""

import time
import csv
from collections import deque

from config import (
    PHONE_IP, PHONE_PORT,
    ACC_FORWARD_INDEX, ACC_LATERAL_INDEX, ACC_VERTICAL_INDEX, GYRO_TURN_INDEX,
    ACC_FORWARD_SIGN, ACC_LATERAL_SIGN, ACC_VERTICAL_SIGN, GYRO_TURN_SIGN,
    POLL_INTERVAL, BUFFER_SECONDS, MIN_FILTER_SAMPLES, EVENT_DELAY_SECONDS,
    STEP_MIN_TIME_BETWEEN_MOVEMENTS, TURN_MOVEMENT_REMOVE_PADDING,
    STOP_IF_NO_NEW_DATA_SECONDS, STEP_SIZE, SAVE_CSV_FILE,
    USE_CAMERA, OPTICAL_FLOW_CONFIRMS_DIRECTION, CAMERA_FRAME_TIMEOUT, FRAME_HISTORY,
    CAMERA_ROTATED_180, CAMERA_MIN_INTERVAL, PLOT_UPDATE_INTERVAL,
    CAMERA_FLOW_WINDOW_BEFORE, CAMERA_FLOW_WINDOW_AFTER,
    STAIR_DETECTION_ENABLED, STAIR_HEIGHT_M, STAIR_GAP_RESET,
    SAVE_PRESSURE_CSV, SAVE_ACCEL_CSV, STAIR_FLIGHT_MIN_RISE, STAIR_BARO_LAG,
    ELEVATOR_MIN_RISE, ELEVATOR_MIN_SECONDS, ELEVATOR_STEP_FRACTION,
    CAMERA_DIR_SMOOTHING, DIRECTION_BOUT_GAP,
    USE_AUDIO, AUDIO_STAIR_CUE, AUDIO_STAIR_WINDOW_S, AUDIO_STAIR_MIN_STEPS,
)

from ipwebcam_client import IPWebcamClient
from buffer_utils import trim_deque_by_time, make_acc_dataframe, make_gyro_dataframe
from signal_utils import filter_acc_axes, filter_gyro_y
from turn_detection import find_turn_candidates, accept_real_turns
from movement_detection import (
    detect_movement_peaks,
    classify_movement_peak,
    remove_movements_inside_turns,
)
import mapping
from mapping import update_position, save_data_csv
import step_size_store
from keypress_stop import start_enter_to_stop
from audio_feedback import Speaker, MovementAnnouncer, StairVoiceCue
from plotting import setup_live_plots, update_live_plots, close_live_plots, save_analysis_plots
from pressure_utils import (
    relative_altitude_at_time, centered_altitude_at_time,
    make_pressure_dataframe, pressure_to_altitude_series,
)

try:
    from optical_flow import MotionEstimator, FlowParams, direction_from_divergence
    _HAVE_FLOW = True
except ImportError:
    _HAVE_FLOW = False


MOVEMENT_TYPES = {"step_forward", "step_backward", "left_crab", "right_crab"}
TURN_TYPES = {"left_turn", "right_turn"}

CSV_FIELDS = ["x_value", "y_value", "z_value", "step_type"]


def is_movement_event(t):
    return t in MOVEMENT_TYPES


def is_turn_event(t):
    return t in TURN_TYPES


def same_movement_family(a, b):
    return is_movement_event(a) and is_movement_event(b)


def movement_near_turn(event, confirmed_events):
    et = event["time"]
    for ev in confirmed_events:
        if not is_turn_event(ev["type"]):
            continue
        start = ev.get("start", ev["time"])
        end = ev.get("end", ev["time"])
        if start - TURN_MOVEMENT_REMOVE_PADDING <= et <= end + TURN_MOVEMENT_REMOVE_PADDING:
            return True
    return False


def step_backward_position(position, facing):
    """One step against the facing direction. update_position only walks forward."""
    directions = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
    dx, dy = directions[facing]
    position[0] -= STEP_SIZE * dx
    position[1] -= STEP_SIZE * dy
    return position


def recent_altitude_rise(pressure_buffer, now_s, window_s):
    """Altitude change over the last window_s seconds, for the live voice cue."""
    if not pressure_buffer:
        return None
    # Only the tail of the buffer is used, so this stays cheap inside the loop.
    tail = [s for s in pressure_buffer[-400:] if s[0] >= now_s - (window_s + 3.0)]
    if len(tail) < 5 or tail[-1][0] - tail[0][0] < window_s * 0.8:
        return None
    alt_now = relative_altitude_at_time(tail, now_s, 2.0)
    alt_then = relative_altitude_at_time(tail, now_s - window_s, 2.0)
    return alt_now - alt_then


def write_csv_header(path, init_row):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(init_row)


def append_csv_row(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def _find_altitude_ramps(pressure_buffer, grid_dt=0.5, backtrack=0.10, trim_eps=0.15):
    """Find sustained altitude changes in the barometer record, ignoring the steps.

    The altitude is sampled on a time grid and walked greedily: a rising run keeps
    going while new highs appear, small dips are tolerated so noise does not cut a
    real climb in half, and the run is kept only if it gained at least
    STAIR_FLIGHT_MIN_RISE. This is what separates a real change of floor, which
    ends higher and stays there, from the barometer wobbling around one level.

    Each ramp is then trimmed back to where the altitude is actually moving,
    otherwise a ride that follows some flat walking would start its ramp inside the
    flat part and swallow the steps taken there.

    Returns a list of (t_start, t_end, dz, 'up' or 'down').
    """
    if len(pressure_buffer) < 5:
        return []
    ts = [p[0] for p in pressure_buffer]
    t0, t1 = float(min(ts)), float(max(ts))
    if t1 - t0 < 1.0:
        return []
    grid = [t0 + i * grid_dt for i in range(int((t1 - t0) / grid_dt) + 1)]
    alt = [centered_altitude_at_time(pressure_buffer, float(t), 0.75) for t in grid]
    n = len(grid)

    def trim(i0, i1, bottom, top):
        s = i0
        for idx in range(i0, i1 + 1):
            if alt[idx] <= bottom + trim_eps:
                s = idx
        e = i1
        for idx in range(i1, i0 - 1, -1):
            if alt[idx] >= top - trim_eps:
                e = idx
        return (s, e) if s <= e else (i0, i1)

    ramps = []
    i = 0
    while i < n - 1:
        hi, hi_idx, k = alt[i], i, i
        while k + 1 < n and alt[k + 1] >= hi - backtrack:
            if alt[k + 1] > hi:
                hi, hi_idx = alt[k + 1], k + 1
            k += 1
        if hi - alt[i] >= STAIR_FLIGHT_MIN_RISE:
            s, e = trim(i, hi_idx, alt[i], hi)
            ramps.append((float(grid[s]), float(grid[e]), hi - alt[i], "up"))
            i = hi_idx + 1
            continue
        lo, lo_idx, k = alt[i], i, i
        while k + 1 < n and alt[k + 1] <= lo + backtrack:
            if alt[k + 1] < lo:
                lo, lo_idx = alt[k + 1], k + 1
            k += 1
        if alt[i] - lo >= STAIR_FLIGHT_MIN_RISE:
            s, e = trim(i, lo_idx, lo, alt[i])
            ramps.append((float(grid[s]), float(grid[e]), alt[i] - lo, "down"))
            i = lo_idx + 1
            continue
        i += 1
    return ramps


def classify_altitude_changes(confirmed_events, pressure_buffer):
    """Decide stairs or lift for every altitude ramp found above.

    A flight of stairs needs roughly one step per riser. In a lift you stand still.
    So the number of steps during the rise, compared with the number the height
    would require, tells the two apart.

    Returns:
        seg_dir   which step events belong to a flight, and in which direction
        elevators the lift rides, with their height taken from the barometer
        flights   one entry per flight, for the printed summary
        drop      steps to delete, the jolts the IMU picked up inside a lift
    """
    seg_dir, elevators, flights, drop = {}, [], [], set()
    if not STAIR_DETECTION_ENABLED or len(pressure_buffer) < 5:
        return seg_dir, elevators, flights, drop

    step_events = [(k, float(ev["time"])) for k, ev in enumerate(confirmed_events)
                   if ev.get("type") == "step_forward"]

    for (t_start, t_end, dz, direction) in _find_altitude_ramps(pressure_buffer):
        n_expected = max(1, int(round(abs(dz) / STAIR_HEIGHT_M)))
        # The pressure ramp trails the steps that caused it, so the search window
        # starts before the ramp does. This is what catches the first stairs of a
        # flight, which the live loop can never see.
        cand = sorted([(t, k) for (k, t) in step_events
                       if t_start - STAIR_BARO_LAG <= t <= t_end])
        if len(cand) < ELEVATOR_STEP_FRACTION * n_expected:
            if abs(dz) >= ELEVATOR_MIN_RISE and (t_end - t_start) >= ELEVATOR_MIN_SECONDS:
                elevators.append({"time": 0.5 * (t_start + t_end),
                                  "type": "elevator_up" if direction == "up" else "elevator_down",
                                  "dz": abs(dz)})
                for (t, k) in cand:
                    drop.add(k)
            continue
        # A step separated from the flight by a real pause was taken before the
        # stairs, not on them.
        while len(cand) >= 2 and cand[1][0] - cand[0][0] > STAIR_GAP_RESET:
            cand.pop(0)
        cand = cand[-n_expected:]
        label = "stair_up" if direction == "up" else "stair_down"
        for (t, k) in cand:
            seg_dir[k] = label
        flights.append((label, len(cand), round(dz, 2)))
    return seg_dir, elevators, flights, drop


def smooth_step_directions(confirmed_events, seg_dir, drop):
    """Vote one forward/backward direction per stretch of walking.

    Every wrong camera call in testing was a weak reading at the start or end of a
    stretch, where the chest rocks against the direction of travel. Mid-stretch
    readings were strong and always right. Direction belongs to the stretch rather
    than to the individual step, so the majority call fixes exactly those edges.

    A stretch is consecutive plain steps with no turn or crab between them and no
    gap longer than DIRECTION_BOUT_GAP. Backward has to win clearly, since it is
    the deliberate and much rarer movement.

    Returns the direction chosen per step, and the list of steps it overturned.
    """
    if not CAMERA_DIR_SMOOTHING:
        return {}, []

    bouts, cur, last_t = [], [], None
    for k, ev in enumerate(confirmed_events):
        etype = ev.get("type")
        if etype == "step_forward" and k not in seg_dir and k not in drop:
            t = float(ev["time"])
            if cur and t - last_t > DIRECTION_BOUT_GAP:
                bouts.append(cur)
                cur = []
            cur.append(k)
            last_t = t
        elif etype in ("left_turn", "right_turn", "left_crab", "right_crab") or k in seg_dir:
            if cur:
                bouts.append(cur)
                cur = []
            last_t = None
    if cur:
        bouts.append(cur)

    dir_override, fixes = {}, []
    for bout in bouts:
        n_fwd = sum(1 for k in bout if confirmed_events[k].get("camera_dir") == "forward")
        n_back = sum(1 for k in bout if confirmed_events[k].get("camera_dir") == "backward")
        voted = ("backward"
                 if (n_back >= 2 and n_back > n_fwd and n_back >= 0.5 * len(bout))
                 else "forward")
        for k in bout:
            dir_override[k] = voted
            live = "backward" if confirmed_events[k].get("camera_dir") == "backward" else "forward"
            if live != voted:
                fixes.append({"time": float(confirmed_events[k]["time"]),
                              "from": live, "to": voted, "n_fwd": n_fwd, "n_back": n_back})
    return dir_override, fixes


def rebuild_path_from_events(confirmed_events, pressure_buffer):
    """Rebuild the route once the run is over. This is the version that gets saved.

    A flight of stairs or a lift ride is only visible once you can see the whole
    altitude change, and the direction of a stretch of walking is only clear once
    you can see the whole stretch. The live loop does its best step by step; this
    pass corrects it with everything in hand.

    Crabs and turns are left exactly as they were detected, because the IMU is
    reliable for those.
    """
    seg_dir, elevators, flights, drop = classify_altitude_changes(confirmed_events, pressure_buffer)
    dropped = [confirmed_events[k] for k in sorted(drop)]

    dir_override, dir_fixes = smooth_step_directions(confirmed_events, seg_dir, drop)

    timeline = [(float(ev["time"]), "ev", k, ev)
                for k, ev in enumerate(confirmed_events) if k not in drop]
    timeline += [(float(el["time"]), "elev", None, el) for el in elevators]
    timeline.sort(key=lambda x: x[0])

    position = [0.0, 0.0, 0.0]
    facing = 0
    route_rows = [{"x_value": 0.0, "y_value": 0.0, "z_value": 0.0, "step_type": "Init"}]

    for _t, kind, k, obj in timeline:
        if kind == "elev":
            label = obj["type"]
            position[2] += obj["dz"] if label == "elevator_up" else -obj["dz"]
        else:
            ev = obj
            t = ev.get("type")
            if t in ("left_turn", "right_turn", "left_crab", "right_crab"):
                label = t
                position, facing = update_position(position, facing, t)
            elif t == "step_forward" and k in seg_dir:
                label = seg_dir[k]
                position, facing = update_position(position, facing, "step_forward")
                position[2] += STAIR_HEIGHT_M if label == "stair_up" else -STAIR_HEIGHT_M
            elif t == "step_forward":
                live_dir = "backward" if ev.get("camera_dir") == "backward" else "forward"
                if dir_override.get(k, live_dir) == "backward":
                    label = "step_backward"
                    position = step_backward_position(position, facing)
                else:
                    label = "step_forward"
                    position, facing = update_position(position, facing, "step_forward")
            else:
                label = t or "step_forward"
                position, facing = update_position(position, facing, "step_forward")

        route_rows.append({
            "x_value": round(position[0], 2),
            "y_value": round(position[1], 2),
            "z_value": round(position[2], 2),
            "step_type": label,
        })
    return route_rows, flights, elevators, dropped, dir_fixes


def save_pressure_log(pressure_buffer, path, verbose=True):
    """Raw pressure plus the altitude derived from it, so a walk can be replayed later."""
    if not pressure_buffer:
        return
    df = make_pressure_dataframe(pressure_buffer)
    df["altitude_m"] = pressure_to_altitude_series(df["pressure"]).round(4)
    df = df.rename(columns={"seconds_elapsed": "t_s"})
    df.to_csv(path, index=False)
    if verbose:
        print(f"Saved pressure log to {path} ({len(df)} samples).")


def save_accel_log(acc_buffer, path, verbose=True):
    """Raw accelerometer stream. If a walk misses steps, this shows whether the
    samples were there and the threshold was too high, or the data never arrived."""
    if not acc_buffer:
        return
    import pandas as pd
    df = pd.DataFrame(acc_buffer, columns=["t_s", "lateral", "forward", "vertical"])
    df.to_csv(path, index=False)
    if verbose:
        print(f"Saved accel log to {path} ({len(df)} samples).")


def map_accel_row(ts_seconds, raw_xyz):
    lateral = ACC_LATERAL_SIGN * raw_xyz[ACC_LATERAL_INDEX]
    forward = ACC_FORWARD_SIGN * raw_xyz[ACC_FORWARD_INDEX]
    vertical = ACC_VERTICAL_SIGN * raw_xyz[ACC_VERTICAL_INDEX]
    return (ts_seconds, float(lateral), float(forward), float(vertical))


def map_gyro_row(ts_seconds, raw_xyz):
    turn = GYRO_TURN_SIGN * raw_xyz[GYRO_TURN_INDEX]
    return (ts_seconds, float(turn))


def confirm_direction_with_camera(frame_history, event_time, motion_estimator):
    """Forward or backward for the step at event_time.

    One pair of frames can be flipped by a single bad frame, so the flow is summed
    over the consecutive pairs in a short window around the step instead. Summing
    small movements is both more accurate and far less sensitive to one bad pair.

    Returns (direction, summed divergence, summed vertical flow), or None if there
    are not enough frames yet.
    """
    if motion_estimator is None or len(frame_history) < 2:
        return None

    window = [(ts, fr) for ts, fr in frame_history
              if event_time - CAMERA_FLOW_WINDOW_BEFORE <= ts <= event_time + CAMERA_FLOW_WINDOW_AFTER]
    if len(window) < 2:
        # Not enough frames inside the window, so fall back to the last frame
        # taken before the step against the newest one.
        before = frame_history[0][1]
        for ts, fr in frame_history:
            if ts <= event_time:
                before = fr
            else:
                break
        r = motion_estimator.compare(before, frame_history[-1][1])
        return direction_from_divergence(r.divergence, motion_estimator.params), r.divergence, r.mean_dy

    total_div = 0.0
    total_vert = 0.0
    for i in range(1, len(window)):
        res = motion_estimator.compare(window[i - 1][1], window[i][1])
        total_div += res.divergence
        total_vert += res.mean_dy
    return direction_from_divergence(total_div, motion_estimator.params), total_div, total_vert


def apply_user_step_size(user_id):
    """Load this user's calibrated stride. Both this module and mapping read
    STEP_SIZE as a global, so both are updated."""
    global STEP_SIZE
    if user_id is not None:
        user_ss = step_size_store.load_user_step_size(user_id)
        if user_ss is not None:
            STEP_SIZE = user_ss
            mapping.STEP_SIZE = user_ss
        else:
            print(f"[user {user_id}] not calibrated yet; using default step size "
                  f"{STEP_SIZE:.3f} m. Run calibrate_step_length.py first.")
    return STEP_SIZE


def main(client=None, motion_estimator=None, max_iterations=None, use_camera=None, make_plots=True,
         user_id=None):
    apply_user_step_size(user_id)

    if client is None:
        client = IPWebcamClient(PHONE_IP, PHONE_PORT, rotate_180=CAMERA_ROTATED_180)
    if use_camera is None:
        use_camera = USE_CAMERA

    if use_camera and OPTICAL_FLOW_CONFIRMS_DIRECTION and motion_estimator is None:
        if _HAVE_FLOW:
            motion_estimator = MotionEstimator(FlowParams())
        else:
            print("[warn] optical_flow/OpenCV not available; running sensor-only.")
            use_camera = False

    print("IP Webcam PDR starting.")
    print(f"  phone:  http://{getattr(client, 'ip', '?')}:{getattr(client, 'port', '?')}")
    print(f"  camera: {'ON' if use_camera else 'OFF'}")
    print(f"  user:   {user_id if user_id is not None else '(default)'}  |  step size: {STEP_SIZE:.3f} m")

    speaker = Speaker(enabled=USE_AUDIO)
    announcer = MovementAnnouncer(speaker)
    stair_cue = (StairVoiceCue(STAIR_FLIGHT_MIN_RISE, AUDIO_STAIR_MIN_STEPS)
                 if (speaker.active and AUDIO_STAIR_CUE and STAIR_DETECTION_ENABLED) else None)
    last_cue_wall = 0.0
    print(f"  audio:  {'ON' if speaker.active else 'OFF'}"
          + (f"  (stair cue {'ON' if stair_cue else 'OFF'})" if speaker.active else ""))

    # The deques are the rolling window the detection works on; the full lists keep
    # the whole run for the end-of-run rebuild and the figures.
    accel_buf = deque()
    gyro_buf = deque()
    full_acc_buffer = []
    full_gyro_buffer = []
    full_pressure_buffer = []
    frame_history = deque(maxlen=FRAME_HISTORY)

    confirmed_events = []
    confirmed_keys = set()
    last_confirmed_time = -1.0

    t0_ms = None
    last_accel_ts = -1.0
    last_gyro_ts = -1.0
    last_pressure_ts = -1.0
    last_new_data_wall = time.time()
    last_frame_wall = 0.0
    last_plot_wall = 0.0
    last_autosave_wall = 0.0
    received_any = False
    plot_handles = None

    position = [0.0, 0.0, 0.0]
    facing = 0
    route_rows = [{
        "x_value": 0.0, "y_value": 0.0, "z_value": 0.0, "step_type": "Init",
    }]
    write_csv_header(SAVE_CSV_FILE, route_rows[0])

    counts = {"step_forward": 0, "step_backward": 0, "stair_up": 0, "stair_down": 0,
              "elevator_up": 0, "elevator_down": 0,
              "left_crab": 0, "right_crab": 0, "left_turn": 0, "right_turn": 0}

    if make_plots:
        plot_handles = setup_live_plots()
        print("Live windows open: 3D path map + movement signal.")

    stop_requested = start_enter_to_stop() if max_iterations is None else (lambda: False)
    if max_iterations is None:
        print("  (press Enter in this console to stop and save)")

    iteration = 0
    try:
        while True:
            if stop_requested():
                print("\nStop requested (Enter). Saving...")
                break
            if max_iterations is not None and iteration >= max_iterations:
                break
            iteration += 1

            data = client.get_sensors_json(timeout=2.0)
            if data is None:
                print("Lost connection and could not recover. Stopping.")
                break

            accel_arr, gyro_arr = client.parse_imu(data)

            # Phone timestamps are absolute, so the first one becomes t = 0.
            if t0_ms is None:
                if len(accel_arr) > 0:
                    t0_ms = float(accel_arr[0][0])
                elif len(gyro_arr) > 0:
                    t0_ms = float(gyro_arr[0][0])
                else:
                    time.sleep(POLL_INTERVAL)
                    continue

            # Each poll returns a buffer that overlaps the last one, so only
            # samples newer than the last timestamp are taken.
            new_samples = 0
            for row in accel_arr:
                ts_ms = float(row[0])
                if ts_ms > last_accel_ts:
                    last_accel_ts = ts_ms
                    sample = map_accel_row((ts_ms - t0_ms) / 1000.0, row[1:4])
                    accel_buf.append(sample)
                    full_acc_buffer.append(sample)
                    new_samples += 1
            for row in gyro_arr:
                ts_ms = float(row[0])
                if ts_ms > last_gyro_ts:
                    last_gyro_ts = ts_ms
                    gsample = map_gyro_row((ts_ms - t0_ms) / 1000.0, row[1:4])
                    gyro_buf.append(gsample)
                    full_gyro_buffer.append(gsample)
                    new_samples += 1
            for row in client.parse_pressure(data):
                ts_ms = float(row[0])
                if ts_ms > last_pressure_ts:
                    last_pressure_ts = ts_ms
                    full_pressure_buffer.append(((ts_ms - t0_ms) / 1000.0, float(row[1])))

            now_s = (max(last_accel_ts, last_gyro_ts) - t0_ms) / 1000.0

            if new_samples > 0:
                received_any = True
                last_new_data_wall = time.time()
            elif received_any and time.time() - last_new_data_wall > STOP_IF_NO_NEW_DATA_SECONDS:
                print("No new data for a while. Stopping and saving.")
                break

            # Fetching a frame every pass would slow the sensor polling enough to
            # miss steps. This rate is still fast enough to bracket one.
            if use_camera and time.time() - last_frame_wall >= CAMERA_MIN_INTERVAL:
                last_frame_wall = time.time()
                frame = client.get_frame(timeout=CAMERA_FRAME_TIMEOUT)
                if frame is not None:
                    frame_history.append((now_s, frame))

            trim_deque_by_time(accel_buf, now_s, BUFFER_SECONDS)
            trim_deque_by_time(gyro_buf, now_s, BUFFER_SECONDS)

            if len(accel_buf) < MIN_FILTER_SAMPLES or len(gyro_buf) < MIN_FILTER_SAMPLES:
                time.sleep(POLL_INTERVAL)
                continue

            acc = make_acc_dataframe(accel_buf)
            gyro = make_gyro_dataframe(gyro_buf)

            acc_t, x_f, y_f, z_f = filter_acc_axes(acc)
            gyro_t, gyro_y_f = filter_gyro_y(gyro)

            turn_candidates = find_turn_candidates(gyro_t, gyro_y_f)
            accepted_turns, _ = accept_real_turns(turn_candidates, acc_t, x_f, y_f, z_f)

            peaks, _, forward_signal, crab_signal = detect_movement_peaks(acc_t, x_f, y_f, z_f)

            # Anything too close to the end of the window can still change as more
            # samples arrive, so it is left alone until it has settled.
            movement_events = []
            for p in peaks:
                pt = float(acc_t.iloc[p])
                if pt > now_s - EVENT_DELAY_SECONDS:
                    continue
                movement_events.append({"time": pt, "type": classify_movement_peak(p, acc_t, x_f, y_f, z_f)})

            confirmed_turns = [t for t in accepted_turns
                               if t["end"] <= now_s - EVENT_DELAY_SECONDS]

            movement_events = remove_movements_inside_turns(movement_events, accepted_turns)

            current = list(movement_events)
            for t in confirmed_turns:
                current.append({"time": t["time"], "type": t["type"],
                                "start": t["start"], "end": t["end"]})
            current.sort(key=lambda e: e["time"])

            for ev in current:
                # The same event reappears on every pass while it is inside the
                # rolling window, so it is matched by time and type and skipped.
                if ev["time"] <= last_confirmed_time + 0.15:
                    continue
                if confirmed_events:
                    last_ev = confirmed_events[-1]
                    if same_movement_family(ev["type"], last_ev["type"]) and \
                            ev["time"] - last_ev["time"] < STEP_MIN_TIME_BETWEEN_MOVEMENTS:
                        continue
                key = (round(ev["time"], 1), ev["type"])
                if key in confirmed_keys:
                    continue
                if is_movement_event(ev["type"]) and movement_near_turn(ev, confirmed_events):
                    continue

                movement_type = ev["type"]
                flow_div = None
                camera_direction = None

                if use_camera and OPTICAL_FLOW_CONFIRMS_DIRECTION and movement_type == "step_forward":
                    flow = confirm_direction_with_camera(frame_history, ev["time"], motion_estimator)
                    if flow is not None:
                        camera_direction, flow_div, _ = flow

                ev["flow_div"] = flow_div
                ev["camera_dir"] = camera_direction

                confirmed_keys.add(key)
                confirmed_events.append(ev)
                last_confirmed_time = ev["time"]

                if movement_type != "step_forward":
                    position, facing = update_position(position, facing, movement_type)
                else:
                    if camera_direction == "backward":
                        movement_type = "step_backward"
                        position = step_backward_position(position, facing)
                    else:
                        position, facing = update_position(position, facing, "step_forward")

                if movement_type in counts:
                    counts[movement_type] += 1

                route_rows.append({
                    "x_value": round(position[0], 2),
                    "y_value": round(position[1], 2),
                    "z_value": round(position[2], 2),
                    "step_type": movement_type,
                })
                append_csv_row(SAVE_CSV_FILE, route_rows[-1])

                notes = ""
                if flow_div is not None:
                    notes += f"  flow_div={flow_div:+.2f}"
                print(f"{len(confirmed_events):02d}. t={ev['time']:.2f}s  {movement_type:12s}  "
                      f"pos={[round(p,2) for p in position]}  facing={facing}{notes}")
                announcer.movement(movement_type)

            if make_plots and time.time() - last_plot_wall >= PLOT_UPDATE_INTERVAL:
                last_plot_wall = time.time()
                try:
                    update_live_plots(plot_handles, acc_t, forward_signal, crab_signal,
                                      confirmed_events, route_rows)
                except Exception:
                    pass

            # Autosave, so a hard stop or a crash cannot lose the raw logs.
            if time.time() - last_autosave_wall >= 4.0:
                last_autosave_wall = time.time()
                save_accel_log(full_acc_buffer, SAVE_ACCEL_CSV, verbose=False)
                save_pressure_log(full_pressure_buffer, SAVE_PRESSURE_CSV, verbose=False)

            if stair_cue is not None and time.time() - last_cue_wall >= 1.0:
                last_cue_wall = time.time()
                rise = recent_altitude_rise(full_pressure_buffer, now_s, AUDIO_STAIR_WINDOW_S)
                steps_in_window = sum(1 for e in confirmed_events
                                      if e["type"] == "step_forward"
                                      and e["time"] >= now_s - AUDIO_STAIR_WINDOW_S)
                phrase = stair_cue.update(rise, steps_in_window)
                if phrase:
                    announcer.stairs(phrase)
                    print(f"[voice] {phrase}  (altitude {rise:+.2f} m over last "
                          f"{AUDIO_STAIR_WINDOW_S:.0f} s, {steps_in_window} steps)")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped by keyboard.")
    finally:
        speaker.close()

        save_pressure_log(full_pressure_buffer, SAVE_PRESSURE_CSV)
        save_accel_log(full_acc_buffer, SAVE_ACCEL_CSV)

        route_rows, flights, elevators, dropped, dir_fixes = rebuild_path_from_events(
            confirmed_events, full_pressure_buffer)
        counts = {k: 0 for k in counts}
        for row in route_rows[1:]:
            counts[row["step_type"]] = counts.get(row["step_type"], 0) + 1
        save_data_csv(route_rows)

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        if STAIR_DETECTION_ENABLED:
            for direction, n_steps, baro_net in flights:
                sign = 1 if direction == "stair_up" else -1
                print(f"  {direction}: flight of {n_steps} steps -> "
                      f"{sign * n_steps * STAIR_HEIGHT_M:+.2f} m  (barometer net {baro_net:+.2f} m)")
            for el in elevators:
                s = "+" if el["type"] == "elevator_up" else "-"
                print(f"  {el['type']}: {s}{el['dz']:.2f} m  (sustained altitude change, no steps)")
            for ev in dropped:
                print(f"  ignored a step shown live at t={float(ev['time']):.2f}s "
                      f"(taken inside the lift)")
            if not flights and not elevators:
                print("  flat route, no stairs or lift detected")
        for fx in dir_fixes:
            print(f"  corrected step at t={fx['time']:.2f}s: {fx['from']} -> {fx['to']} "
                  f"(its walking stretch voted {fx['n_fwd']} forward vs {fx['n_back']} backward)")
        if STAIR_DETECTION_ENABLED or dir_fixes:
            print("")
        for k, v in counts.items():
            print(f"    {k}: {v}")
        z_final = route_rows[-1]["z_value"] if route_rows else 0.0
        print(f"    final z (height): {z_final:+.2f} m")
        print("=" * 60)

        if make_plots:
            close_live_plots()
            save_analysis_plots(route_rows, full_acc_buffer, full_gyro_buffer, confirmed_events,
                                pressure_buffer=full_pressure_buffer)

    return route_rows, counts


if __name__ == "__main__":
    uid = input("Enter user ID (name or number; leave blank for default step size): ").strip()
    main(user_id=uid if uid else None)
