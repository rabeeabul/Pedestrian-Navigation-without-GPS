import numpy as np
from config import TURN_THRESHOLD, TURN_MIN_DURATION, POSITIVE_GYRO_Y_IS_LEFT


def find_turn_candidates(gyro_t, gyro_y_f):
    # A turn is a stretch where the filtered yaw rate stays over the threshold.
    time_arr = gyro_t.to_numpy()
    signal = gyro_y_f.to_numpy()
    turn_mask = np.abs(signal) >= TURN_THRESHOLD
    turn_candidates = []
    in_region = False
    start_idx = 0

    for i in range(len(turn_mask)):
        if turn_mask[i] and not in_region:
            in_region = True
            start_idx = i

        is_last = i == len(turn_mask) - 1
        if in_region and ((not turn_mask[i]) or is_last):
            end_idx = i if not turn_mask[i] else i + 1
            start_time = time_arr[start_idx]
            end_time = time_arr[end_idx - 1]
            duration = end_time - start_time

            # Short spikes are noise, not a turn of the body.
            if duration >= TURN_MIN_DURATION:
                region_signal = signal[start_idx:end_idx]
                local_peak_idx = np.argmax(np.abs(region_signal))
                peak_idx = start_idx + local_peak_idx
                peak_time = time_arr[peak_idx]
                peak_value = signal[peak_idx]

                if POSITIVE_GYRO_Y_IS_LEFT:
                    turn_type = "left_turn" if peak_value > 0 else "right_turn"
                else:
                    turn_type = "right_turn" if peak_value > 0 else "left_turn"

                turn_candidates.append({
                    "time": float(peak_time),
                    "type": turn_type,
                    "start": float(start_time),
                    "end": float(end_time),
                    "value": float(peak_value),
                })

            in_region = False

    return turn_candidates


def accept_real_turns(turn_candidates, acc_t, x_f, y_f, z_f):
    # A turn always wins over a crab step. Earlier versions threw turns away when
    # AccX was strong, but a real turn also swings the phone sideways, so genuine
    # turns were being counted as crab steps. Instead every candidate is kept and
    # the steps inside its time window are dropped in movement_detection.
    accepted_turns = list(turn_candidates)
    rejected_turns = []
    return accepted_turns, rejected_turns
