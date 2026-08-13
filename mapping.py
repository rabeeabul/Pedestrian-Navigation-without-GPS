import pandas as pd
from config import STEP_SIZE, SAVE_CSV_FILE


def update_position(position, facing, movement_type):
    # Heading is kept as one of four compass directions, so a turn is a quarter
    # turn and indoor corridors stay square. This also means no gyro drift builds up.
    directions = {
        0: (0, 1),
        1: (1, 0),
        2: (0, -1),
        3: (-1, 0),
    }

    dx, dy = directions[facing]

    if movement_type == "step_forward":
        position[0] += STEP_SIZE * dx
        position[1] += STEP_SIZE * dy
    elif movement_type == "left_crab":
        position[0] += STEP_SIZE * (-dy)
        position[1] += STEP_SIZE * dx
    elif movement_type == "right_crab":
        position[0] += STEP_SIZE * dy
        position[1] += STEP_SIZE * (-dx)
    elif movement_type == "left_turn":
        facing = (facing - 1) % 4
    elif movement_type == "right_turn":
        facing = (facing + 1) % 4

    return position, facing


def save_data_csv(route_rows):
    df = pd.DataFrame(route_rows)
    df.to_csv(SAVE_CSV_FILE, index=False)
    print(f"\nSaved route to {SAVE_CSV_FILE}")
