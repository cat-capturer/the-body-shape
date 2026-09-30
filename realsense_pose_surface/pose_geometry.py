"""Camera-independent depth sampling and temporal filtering (meters)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Landmark3D:
    point: np.ndarray
    visibility: float


def normalized_pixel(x: float, y: float, width: int, height: int):
    """Map normalized coordinates to pixels without clamping off-screen points."""
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < 1 and 0 <= y < 1):
        return None
    return int(x * width), int(y * height)


def median_depth(depth_image, x, y, window, units, min_depth=0.0,
                 max_depth=float("inf"), tolerance=None):
    """Median of valid depths, optionally gated around the valid center pixel.

    A missing center falls back to the valid neighborhood median. This cannot
    distinguish a person from background when the whole foreground is missing.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("depth window must be a positive odd integer")
    if not math.isfinite(units) or units <= 0:
        raise ValueError("depth units must be finite and positive")
    height, width = depth_image.shape
    if not (0 <= x < width and 0 <= y < height):
        return 0.0
    radius = window // 2
    patch = depth_image[max(0, y-radius):min(height, y+radius+1),
                        max(0, x-radius):min(width, x+radius+1)].astype(float) * units
    valid = patch[np.isfinite(patch) & (patch > 0) &
                  (patch >= min_depth) & (patch <= max_depth)]
    center = float(depth_image[y, x]) * units
    if tolerance is not None and math.isfinite(center) and center > 0 and min_depth <= center <= max_depth:
        valid = valid[np.abs(valid - center) <= tolerance]
    return float(np.median(valid)) if valid.size else 0.0


def smooth_landmarks(previous, current, smooth, dt=1/30, reset_distance=0.75,
                     reset_after=0.5):
    """EMA with a 30 Hz reference weight; reset on gaps and large jumps.

    Missing points are immediately dropped. A large jump is accepted as a new
    position rather than blending two potentially different people together.
    """
    if not math.isfinite(dt) or dt <= 0 or dt > reset_after:
        return current
    alpha = min(max(smooth, 0.0), 0.95) ** (dt * 30)
    output = {}
    for index, landmark in current.items():
        point = landmark.point
        old = previous.get(index)
        if old is not None and np.linalg.norm(old.point - point) <= reset_distance:
            point = old.point * alpha + point * (1 - alpha)
        output[index] = Landmark3D(point=point, visibility=landmark.visibility)
    return output


def bone_transform(start, end, radius):
    """Transform a unit cylinder centered on Z to a segment; None if degenerate."""
    direction = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length < 1e-4:
        return None
    z_axis = direction / length
    reference = [0, 0, 1] if abs(z_axis[2]) < 0.9 else [0, 1, 0]
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis * radius, y_axis * radius, direction))
    transform[:3, 3] = (np.asarray(start) + np.asarray(end)) * 0.5
    return transform
