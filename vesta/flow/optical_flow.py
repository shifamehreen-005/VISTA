"""
Module 1: Optical Flow — Camera Motion Estimation

Estimates camera translation (dx, dy) and rotation (d_theta) between consecutive
frames using ORB feature matching with affine decomposition.

This is the "vestibular system" of VESTA — it tells the agent how the camera moved
so that previously detected hazards can be re-projected into the current viewpoint.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraMotion:
    """Camera motion between two consecutive frames."""
    dx: float        # horizontal translation (pixels, right = positive)
    dy: float        # vertical translation (pixels, down = positive)
    d_theta: float   # rotation (degrees, clockwise = positive)
    confidence: float  # 0-1, based on number of inlier matches

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.dx**2 + self.dy**2)

    @property
    def is_significant(self) -> bool:
        """True if camera moved enough to matter."""
        return self.magnitude > 2.0 or abs(self.d_theta) > 0.5


# ORB detector — created once, reused
_orb = cv2.ORB_create(nfeatures=500)  # 500 features is enough, faster than 1000
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Resolution for optical flow computation (width in pixels)
# ORB scales as O(w*h) — processing at 640px instead of 1920px is ~4.4x faster
FLOW_PROCESSING_WIDTH = 640

# Farneback params for dense flow fallback
_farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _downscale_for_flow(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Downscale a frame for faster optical flow processing.
    Returns (downscaled_frame, scale_factor) where scale_factor maps
    downscaled coords back to original resolution.
    """
    h, w = frame.shape[:2]
    if w <= FLOW_PROCESSING_WIDTH:
        return frame, 1.0
    scale = FLOW_PROCESSING_WIDTH / w
    new_w = FLOW_PROCESSING_WIDTH
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def estimate_camera_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    method: str = "orb",
) -> CameraMotion:
    """
    Estimate how the camera moved between two consecutive frames.

    Frames are automatically downscaled to FLOW_PROCESSING_WIDTH for speed.
    Translation (dx, dy) is scaled back to original resolution.
    Rotation (d_theta) is scale-invariant.

    Args:
        prev_frame: Previous frame (BGR or grayscale)
        curr_frame: Current frame (BGR or grayscale)
        method: "orb" for feature-based (better for rotation),
                "dense" for Farneback optical flow (better for translation)

    Returns:
        CameraMotion with dx, dy, d_theta
    """
    # Downscale for speed — ORB goes from ~65ms to ~15ms at 640px
    prev_small, scale = _downscale_for_flow(prev_frame)
    curr_small, _ = _downscale_for_flow(curr_frame)

    if method == "orb":
        motion = _estimate_orb(prev_small, curr_small)
    elif method == "dense":
        motion = _estimate_dense(prev_small, curr_small)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Invert signs: the affine/flow measures how FEATURES moved (scene-relative),
    # but we want how the CAMERA moved. Camera right → features left → negate.
    motion = CameraMotion(
        dx=-motion.dx,
        dy=-motion.dy,
        d_theta=-motion.d_theta,
        confidence=motion.confidence,
    )

    # Scale translation back to original resolution (rotation is scale-invariant)
    if scale != 1.0:
        motion = CameraMotion(
            dx=motion.dx / scale,
            dy=motion.dy / scale,
            d_theta=motion.d_theta,
            confidence=motion.confidence,
        )
    return motion


def _estimate_orb(prev_frame: np.ndarray, curr_frame: np.ndarray) -> CameraMotion:
    """ORB feature matching → affine transform → decompose to dx, dy, dθ."""
    gray_prev = _to_gray(prev_frame)
    gray_curr = _to_gray(curr_frame)

    kp1, des1 = _orb.detectAndCompute(gray_prev, None)
    kp2, des2 = _orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return CameraMotion(0, 0, 0, 0.0)

    # KNN match with ratio test
    matches = _bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 6:
        # Not enough matches — fall back to dense flow
        return _estimate_dense(prev_frame, curr_frame)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimate rigid (rotation + translation) transform
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if M is None:
        return CameraMotion(0, 0, 0, 0.0)

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    confidence = min(1.0, inlier_count / max(len(good), 1))

    # Decompose 2x3 affine into rotation + translation
    # M = [[cos(θ) -sin(θ) tx],
    #      [sin(θ)  cos(θ) ty]]
    dx = M[0, 2]
    dy = M[1, 2]
    d_theta = math.degrees(math.atan2(M[1, 0], M[0, 0]))

    return CameraMotion(dx=dx, dy=dy, d_theta=d_theta, confidence=confidence)


def _estimate_dense(prev_frame: np.ndarray, curr_frame: np.ndarray) -> CameraMotion:
    """Farneback dense optical flow → average motion vector."""
    gray_prev = _to_gray(prev_frame)
    gray_curr = _to_gray(curr_frame)

    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, **_farneback_params)

    # Average flow gives global camera motion
    dx = float(np.median(flow[..., 0]))
    dy = float(np.median(flow[..., 1]))

    # Estimate rotation from flow field curl
    # Sample flow at image quadrants to detect rotational component
    h, w = flow.shape[:2]
    cx, cy = w // 2, h // 2

    # Flow vectors at 4 cardinal points
    top = flow[h // 4, cx]
    bottom = flow[3 * h // 4, cx]
    left = flow[cy, w // 4]
    right = flow[cy, 3 * w // 4]

    # Rotational component: cross product of position vector and flow vector
    # Positive = clockwise rotation
    curl = (
        (right[1] - left[1]) / w +  # horizontal shear
        (top[0] - bottom[0]) / h     # vertical shear
    )
    d_theta = math.degrees(curl)

    return CameraMotion(dx=dx, dy=dy, d_theta=d_theta, confidence=0.6)


def process_video_motion(
    video_path: str,
    max_frames: int | None = None,
) -> list[CameraMotion]:
    """
    Process an entire video and return per-frame camera motion estimates.

    Args:
        video_path: Path to video file
        max_frames: Optional limit on frames to process

    Returns:
        List of CameraMotion, one per frame (first frame has zero motion)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    motions = []
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if prev_frame is None:
            motions.append(CameraMotion(0, 0, 0, 1.0))
        else:
            motion = estimate_camera_motion(prev_frame, frame, method="orb")
            motions.append(motion)

        prev_frame = frame
        frame_idx += 1

    cap.release()
    return motions


def compute_cumulative_rotation(motions: list[CameraMotion]) -> list[float]:
    """
    Compute cumulative heading angle from a sequence of frame-to-frame motions.

    Returns list of absolute heading angles in degrees (0 = initial facing direction).
    """
    headings = [0.0]
    for motion in motions[1:]:
        headings.append(headings[-1] + motion.d_theta)
    return headings
