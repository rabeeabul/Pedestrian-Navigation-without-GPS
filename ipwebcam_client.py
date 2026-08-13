"""HTTP client for the IP Webcam Android app.

Two endpoints are used, both served on port 8080:
    /sensors.json   recent IMU and barometer samples
    /shot.jpg       one JPEG frame from the camera

Each sample in sensors.json looks like [timestamp_ms, [x, y, z]].
"""

import time
import json

import numpy as np
import requests

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# Different phones name the same sensor differently, so several names are tried.
ACCEL_KEYS = ["lin_accel", "accel", "linear_acceleration", "accelerometer"]
GYRO_KEYS = ["gyro", "gyroscope", "rot_rate"]
PRESSURE_KEYS = ["pressure", "barometer", "baro", "press"]


class IPWebcamClient:
    def __init__(self, ip, port=8080, accel_keys=None, gyro_keys=None, pressure_keys=None,
                 max_retries=5, base_backoff=1.0, max_backoff=10.0, rotate_180=False):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"
        self.rotate_180 = rotate_180
        self.accel_keys = accel_keys or ACCEL_KEYS
        self.gyro_keys = gyro_keys or GYRO_KEYS
        self.pressure_keys = pressure_keys or PRESSURE_KEYS
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.session = requests.Session()

    def _get(self, path, timeout=2.0, retry=True):
        # Wi-Fi drops for a second or two now and then, so a failed request waits
        # and tries again instead of ending the run.
        url = self.base_url + path
        attempts = self.max_retries if retry else 1
        delay = self.base_backoff
        for attempt in range(attempts):
            try:
                resp = self.session.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                if not retry:
                    return None
                if attempt == 0:
                    print(f"[ipwebcam] connection problem on {path}: {e}. Retrying...")
                else:
                    print(f"[ipwebcam] retry {attempt}/{attempts - 1} failed; waiting {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, self.max_backoff)
        print(f"[ipwebcam] gave up on {path} after {attempts} attempts")
        return None

    def get_sensors_json(self, timeout=2.0):
        resp = self._get("/sensors.json", timeout=timeout)
        if resp is None:
            return None
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[ipwebcam] could not parse sensors.json: {e}")
            return None

    @staticmethod
    def _find_key(data_dict, candidates):
        for key in candidates:
            if key in data_dict and isinstance(data_dict[key], dict) and "data" in data_dict[key]:
                return key
        return None

    @staticmethod
    def _samples_to_array(samples):
        # Turn [[ts, [x, y, z]], ...] into an (N, 4) array. Rows that are short or
        # malformed are skipped rather than allowed to stop the run.
        rows = []
        for s in samples:
            try:
                ts = float(s[0])
                vals = s[1]
                if isinstance(vals, (list, tuple)):
                    row = [ts] + [float(v) for v in vals]
                else:
                    row = [ts, float(vals)]
                rows.append(row)
            except (TypeError, ValueError, IndexError):
                continue
        if not rows:
            return np.empty((0, 4))
        width = max(len(r) for r in rows)
        out = np.full((len(rows), width), np.nan)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
        return out

    def parse_imu(self, data_dict):
        """Returns (accel, gyro), each an (N, 4) array of [timestamp_ms, x, y, z]."""
        if data_dict is None:
            return np.empty((0, 4)), np.empty((0, 4))

        accel_key = self._find_key(data_dict, self.accel_keys)
        gyro_key = self._find_key(data_dict, self.gyro_keys)

        accel = self._samples_to_array(data_dict[accel_key]["data"]) if accel_key else np.empty((0, 4))
        gyro = self._samples_to_array(data_dict[gyro_key]["data"]) if gyro_key else np.empty((0, 4))
        return accel, gyro

    def parse_pressure(self, data_dict):
        """Returns an (N, 2) array of [timestamp_ms, pressure]."""
        if data_dict is None:
            return np.empty((0, 2))
        key = self._find_key(data_dict, self.pressure_keys)
        if key is None:
            return np.empty((0, 2))
        return self._samples_to_array(data_dict[key]["data"])

    def get_frame(self, timeout=3.0):
        """One camera frame as a BGR image, or None if it could not be fetched."""
        if not _HAVE_CV2:
            raise RuntimeError("OpenCV (cv2) is required for get_frame(); pip install opencv-python")
        resp = self._get("/shot.jpg", timeout=timeout)
        if resp is None:
            return None
        buf = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            print("[ipwebcam] /shot.jpg did not decode to an image")
            return None
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def test_connection(self, timeout=5.0):
        """Returns (ok, message). Used by the connection check below."""
        data = self.get_sensors_json(timeout=timeout)
        if data is None:
            return False, f"No response from {self.base_url}/sensors.json"
        accel, gyro = self.parse_imu(data)
        sensors = sorted(data.keys())
        msg = (f"Connected to {self.base_url}. Sensors: {sensors}. "
               f"accel samples={len(accel)}, gyro samples={len(gyro)}")
        return True, msg


if __name__ == "__main__":
    # python ipwebcam_client.py 10.194.198.210
    import sys
    from config import PHONE_IP, PHONE_PORT

    ip = sys.argv[1] if len(sys.argv) > 1 else PHONE_IP
    print(f"Testing the connection to {ip} ...")
    client = IPWebcamClient(ip, PHONE_PORT)
    ok, message = client.test_connection()
    print(("[OK] " if ok else "[FAIL] ") + message)
    if ok and _HAVE_CV2:
        frame = client.get_frame()
        if frame is not None:
            print(f"[OK] camera frame received, size {frame.shape}")
