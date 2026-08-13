"""Camera motion estimation.

The accelerometer tells us when a step happened but not which way it went: a
forward step and a backward step have the same signature. The camera does know.
Walking forward, the scene expands away from a point ahead of you; walking
backward it contracts. A sideways shift or a turn slides the whole image instead.

So the flow field is split in two:
    mean of the flow            -> sideways or vertical slide
    radial part of the rest     -> expansion, which is forward or backward
"""

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class FlowParams:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    resize_scale: float = 0.5       # half size frames, fast enough to keep up with the sensor loop

    center_ignore_frac: float = 0.15    # the middle of the image barely moves, so ignore it

    # Forward fires on a small expansion, backward needs a much clearer contraction,
    # because backward is the rare movement and a false one is expensive.
    forward_min_divergence: float = 0.15
    backward_min_divergence: float = 1.0
    lateral_min_flow: float = 0.30
    vertical_min_flow: float = 0.30

    forward_is_positive_divergence: bool = True
    right_is_positive_x: bool = True
    up_is_negative_y: bool = True

    use_alignment: bool = False
    align_max_features: int = 500
    align_min_matches: int = 12


def to_gray(image):
    if image is None:
        raise ValueError("to_gray received None")
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _maybe_resize(gray, scale):
    if scale is None or abs(scale - 1.0) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def align_frame(prev_gray, curr_gray, params: FlowParams):
    # Optional shake cancelling with ORB features. If there are too few matches the
    # current frame is returned untouched, since a bad warp is worse than none.
    orb = cv2.ORB_create(nfeatures=params.align_max_features)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < params.align_min_matches or len(kp2) < params.align_min_matches:
        return curr_gray

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < params.align_min_matches:
        return curr_gray

    matches = sorted(matches, key=lambda m: m.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None or H.shape != (3, 3):
        return curr_gray

    h, w = prev_gray.shape[:2]
    return cv2.warpPerspective(curr_gray, H, (w, h))


def dense_flow(prev_gray, curr_gray, params: FlowParams):
    """Farneback flow, shape (H, W, 2): [..., 0] is dx and [..., 1] is dy."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        params.pyr_scale, params.levels, params.winsize,
        params.iterations, params.poly_n, params.poly_sigma, params.flags,
    )


def mean_translation(flow):
    return float(np.mean(flow[..., 0])), float(np.mean(flow[..., 1]))


def radial_divergence(flow, center_ignore_frac=0.15):
    """Positive when the scene expands (forward), negative when it contracts (backward)."""
    h, w = flow.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # Subtracting the mean first is what makes this work: a pure sideways slide
    # then has almost no radial part and cannot be read as forward motion.
    tx, ty = mean_translation(flow)
    fx = flow[..., 0] - tx
    fy = flow[..., 1] - ty

    yy, xx = np.mgrid[0:h, 0:w]
    rx = xx - cx
    ry = yy - cy
    radius = np.sqrt(rx * rx + ry * ry)

    half_diag = np.sqrt(cx * cx + cy * cy)
    mask = radius > (center_ignore_frac * half_diag)

    safe_r = np.where(radius > 1e-6, radius, 1.0)
    ux = rx / safe_r
    uy = ry / safe_r

    radial_component = fx * ux + fy * uy
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(radial_component[mask]))


@dataclass
class MotionResult:
    forward_backward: str
    lateral: str
    vertical: str
    divergence: float
    mean_dx: float
    mean_dy: float
    flow: np.ndarray = None


def direction_from_divergence(divergence, params):
    """'forward', 'backward', or 'none' when the reading is too weak to trust."""
    signed = divergence if params.forward_is_positive_divergence else -divergence
    if signed >= params.forward_min_divergence:
        return "forward"
    if signed <= -params.backward_min_divergence:
        return "backward"
    return "none"


def classify_motion(prev_bgr, curr_bgr, params: FlowParams = None, keep_flow=False):
    if params is None:
        params = FlowParams()

    prev_gray = _maybe_resize(to_gray(prev_bgr), params.resize_scale)
    curr_gray = _maybe_resize(to_gray(curr_bgr), params.resize_scale)

    if params.use_alignment:
        curr_gray = align_frame(prev_gray, curr_gray, params)

    flow = dense_flow(prev_gray, curr_gray, params)

    divergence = radial_divergence(flow, params.center_ignore_frac)
    mean_dx, mean_dy = mean_translation(flow)

    fb = direction_from_divergence(divergence, params)

    lat = "none"
    signed_dx = mean_dx if params.right_is_positive_x else -mean_dx
    if abs(mean_dx) >= params.lateral_min_flow:
        lat = "right" if signed_dx > 0 else "left"

    vert = "none"
    up_sign = -1.0 if params.up_is_negative_y else 1.0
    signed_dy = up_sign * mean_dy
    if abs(mean_dy) >= params.vertical_min_flow:
        vert = "up" if signed_dy > 0 else "down"

    return MotionResult(
        forward_backward=fb,
        lateral=lat,
        vertical=vert,
        divergence=divergence,
        mean_dx=mean_dx,
        mean_dy=mean_dy,
        flow=flow if keep_flow else None,
    )


class MotionEstimator:
    """Holds the flow settings so the main loop can compare any two frames it likes."""

    def __init__(self, params: FlowParams = None):
        self.params = params or FlowParams()

    def compare(self, frame_a_bgr, frame_b_bgr, keep_flow=False):
        return classify_motion(frame_a_bgr, frame_b_bgr, self.params, keep_flow=keep_flow)
