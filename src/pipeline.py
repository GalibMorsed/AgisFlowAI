"""Pipeline helpers: grid mapping, direction entropy, scoring.

These functions are pure-Python / NumPy-only where possible so unit
tests can run without OpenCV or heavy ML libraries installed.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np


def make_grid(frame_shape: Tuple[int, int], rows: int, cols: int) -> List[Tuple[int, int, int, int]]:
    """Return list of grid cell rectangles (x0, y0, x1, y1).

    frame_shape: (height, width)
    rows, cols: number of grid rows and columns
    """
    h, w = frame_shape
    cell_h = h / rows
    cell_w = w / cols
    rects: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * cell_w))
            y0 = int(round(r * cell_h))
            # ensure last cell reaches the border
            x1 = int(round((c + 1) * cell_w)) if c < cols - 1 else w
            y1 = int(round((r + 1) * cell_h)) if r < rows - 1 else h
            rects.append((x0, y0, x1, y1))
    return rects


def bbox_center_to_cell(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int], rows: int, cols: int) -> int:
    """Given bbox (x0,y0,x1,y1) return flat cell index (row-major).

    If center exactly on boundary, it will go to the lower index.
    """
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    h, w = frame_shape
    col = min(int(cx * cols / w), cols - 1)
    row = min(int(cy * rows / h), rows - 1)
    return row * cols + col


def angles_from_flow(u: Iterable[float], v: Iterable[float]) -> np.ndarray:
    """Convert flow vector components to angles in radians in range [0, 2pi).

    u, v are sequences (same length) of horizontal and vertical flow.
    Returns numpy array of angles.
    """
    u_a = np.asarray(u, dtype=float)
    v_a = np.asarray(v, dtype=float)
    angles = np.arctan2(v_a, u_a)  # range [-pi, pi]
    angles = np.mod(angles + 2 * math.pi, 2 * math.pi)
    return angles


def direction_histogram(angles: np.ndarray, bins: int = 8) -> np.ndarray:
    """Compute histogram of angles with `bins` uniform bins over [0,2pi).

    Returns counts normalized to sum=1 (probability distribution). If no
    angles provided, returns zeros.
    """
    if angles.size == 0:
        return np.zeros(bins, dtype=float)
    counts, _ = np.histogram(angles, bins=bins, range=(0.0, 2 * math.pi))
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return np.zeros(bins, dtype=float)
    return counts / total


def entropy_of_distribution(p: np.ndarray) -> float:
    """Compute Shannon entropy (base e) of a probability vector p.

    p should sum to 1; zeros are ignored in entropy calculation.
    Normalized entropy (0..1) can be obtained by dividing by log(n_bins).
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return 0.0
    p_nonzero = p[p > 0]
    if p_nonzero.size == 0:
        return 0.0
    return -float(np.sum(p_nonzero * np.log(p_nonzero)))


def normalized_entropy(p: np.ndarray) -> float:
    """Normalized entropy between 0 and 1 for distribution p."""
    e = entropy_of_distribution(p)
    max_e = math.log(len(p)) if len(p) > 0 else 1.0
    if max_e == 0:
        return 0.0
    return e / max_e


def normalize_feature(x: np.ndarray, clip_min: float = 1.0, clip_max: float = 1.0) -> np.ndarray:
    """Normalize a 1D array to 0..1 using min/max (optionally clipped).

    If all values equal, returns zeros.
    """
    x = np.asarray(x, dtype=float)
    if clip_min is not None:
        x = np.maximum(x, clip_min)
    if clip_max is not None:
        x = np.minimum(x, clip_max)
    mn = x.min() if x.size > 0 else 0.0
    mx = x.max() if x.size > 0 else 0.0
    if mx - mn <= 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def instability_score(d: float, a: float, e: float, w1: float = 0.4, w2: float = 0.4, w3: float = 0.2) -> float:
    """Compute rule-based instability score from normalized features.

    d,a,e should be in [0,1]. Output clipped to [0,1].
    """
    raw = w1 * a + w2 * e + w3 * d
    return float(max(0.0, min(1.0, raw)))


def risk_category(score: float) -> str:
    """Map score to category string: Green, Yellow, Red."""
    if score < 0.3:
        return "Green"
    if score < 0.6:
        return "Yellow"
    return "Red"
