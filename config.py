# All tunable constants for the system live here.
# The thresholds are calibrated for one phone carried in a chest holder, so they
# are not general-purpose numbers. If you change phone or mount, re-measure them.

# --- phone connection ---
PHONE_IP = "10.194.198.210"     # phone address over USB tethering
PHONE_PORT = 8080

# --- sensor axes ---
# Chest holder, portrait, screen facing forward.
# AccY is the up/down bounce of a step and gives the cleanest peak, so steps are
# counted from it. The camera is what decides forward vs backward.
ACC_FORWARD_INDEX = 1           # AccY, step bounce
ACC_LATERAL_INDEX = 0           # AccX, crab walking
ACC_VERTICAL_INDEX = 2          # AccZ
GYRO_TURN_INDEX = 1             # gyro yaw

# Set to -1 if an axis points the opposite way on your phone.
ACC_FORWARD_SIGN = 1
ACC_LATERAL_SIGN = 1
ACC_VERTICAL_SIGN = 1
GYRO_TURN_SIGN = 1

# --- dead reckoning ---
STEP_SIZE = 0.70                # fallback stride; a calibrated user overrides this
IGNORE_FIRST_SECONDS = 2.0      # skip the start of the run, where the filter settles
STOP_IF_NO_NEW_DATA_SECONDS = 30.0

# --- filters ---
ACC_FILTER_LOW = 0.45
ACC_FILTER_HIGH = 3.0
ACC_FILTER_ORDER = 3
GYRO_FILTER_CUTOFF = 2.5
GYRO_FILTER_ORDER = 3

# --- step detection ---
STEP_MIN_TIME_BETWEEN_MOVEMENTS = 0.7   # seconds; two peaks closer than this are one step
FORWARD_STEP_HEIGHT = 0.95
CRAB_STEP_HEIGHT = 1.78                 # a crab peak is much taller than a walking peak
CRAB_STEP_PROMINENCE = 0.70
CRAB_RATIO_TO_FORWARD = 1.20            # raise this if crab steps come out as forward steps

# --- turn detection ---
TURN_THRESHOLD = 1.0            # rad/s
TURN_MIN_DURATION = 0.25        # s
TURN_MOVEMENT_REMOVE_PADDING = 1.0      # steps this close to a turn are body sway, not steps

POSITIVE_GYRO_Y_IS_LEFT = True  # flip either of these if left/right come out mirrored
CRAB_LEFT_IS_POSITIVE_X = False

# --- real-time loop ---
POLL_INTERVAL = 0.08            # s between reads of /sensors.json
BUFFER_SECONDS = 10.0           # length of the rolling window kept in memory
MIN_FILTER_SAMPLES = 40         # below this the filter output is meaningless
EVENT_DELAY_SECONDS = 0.85      # wait this long before confirming an event, so it can't move
LIVE_SMOOTH_WINDOW = 5          # samples, live plot only

# --- camera ---
USE_CAMERA = True
OPTICAL_FLOW_CONFIRMS_DIRECTION = True
CAMERA_FRAME_TIMEOUT = 3.0
CAMERA_ROTATED_180 = False      # True only if the phone is mounted upside down
FRAME_HISTORY = 12              # frames kept so a step can be bracketed
CAMERA_MIN_INTERVAL = 0.25      # s between frame fetches, about 4 fps

# Flow is read only over the window where the body is actually moving. Frames that
# arrive after you have stopped are still, and they drag the sum towards zero.
CAMERA_FLOW_WINDOW_BEFORE = 0.5
CAMERA_FLOW_WINDOW_AFTER = 0.35

# One forward/backward decision per walking stretch instead of one per step.
# Wrong camera calls were always at the start or end of a stretch, where the chest
# rocks against the direction of travel; mid-stretch calls were always right.
CAMERA_DIR_SMOOTHING = True
DIRECTION_BOUT_GAP = 2.0        # s without a step splits one stretch from the next

# --- voice feedback ---
USE_AUDIO = True
AUDIO_STAIR_CUE = True
AUDIO_STAIR_WINDOW_S = 9.0      # s of altitude history the stair cue looks back over
AUDIO_STAIR_MIN_STEPS = 2       # a rise with no steps is a lift, so stay quiet

# --- stairs and lifts (barometer) ---
STAIR_DETECTION_ENABLED = True
PRESSURE_KEYS = ["pressure", "barometer", "baro", "press"]

STAIR_HEIGHT_M = 0.17           # measured riser: 2.20 m over 13 stairs
STAIR_GAP_RESET = 3.0           # s; a change measured across a longer pause is not trusted

# This barometer's flat-ground noise is about the size of one stair, so single
# stairs cannot be picked out. A whole flight nets around 2.2 m, which is far
# above the noise, so flights are detected as one sustained altitude change.
STAIR_FLIGHT_MIN_RISE = 0.55    # m; below this it is noise, not a flight
STAIR_BARO_LAG = 3.0            # s; the pressure ramp trails the steps that caused it

SAVE_PRESSURE_CSV = "pressure_log.csv"
SAVE_ACCEL_CSV = "accel_log.csv"

# A lift is a sustained altitude change with almost no steps during it.
ELEVATOR_MIN_RISE = 1.5         # m
ELEVATOR_MIN_SECONDS = 2.5      # s
ELEVATOR_STEP_FRACTION = 0.30   # steps found vs steps a real climb would need

# --- output ---
PLOT_UPDATE_INTERVAL = 0.4      # s between live plot redraws; 3D redraws are slow
SAVE_CSV_FILE = "data.csv"
