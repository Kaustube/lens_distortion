"""
target_free.py
--------------
Target-free lens distortion estimation from straight lines.

Idea
  Straight 3-D lines project to straight 2-D lines in an undistorted image.
  Distortion curves them.  We minimise the sum of squared distances from each
  edge point to its fitted line in the *undistorted* image (straightness cost).

Pipeline
  1. Detect line segments with LSD (or Canny + Hough as fallback).
  2. Keep long segments; sample interior points.
  3. Optimise [k1, k2, k3, p1, p2] (or a subset) with scipy.
  4. Report final straightness residual.

Author : Sukhbeer Kaur (Team KASAT)
Course : CSET381 – Applications of AI, Semester 5
"""

import numpy as np
import cv2
from scipy.optimize import minimize
from typing import List, Tuple, Optional

from distortion_model import DistortionParams, undistort_pixel_coords


# ---------------------------------------------------------------------------
# Line / edge detection
# ---------------------------------------------------------------------------

def detect_line_segments_lsd(gray: np.ndarray,
                               min_length: float = 60.0
                               ) -> List[np.ndarray]:
    """
    Detect line segments with OpenCV's LSD detector.
    Falls back to Canny+HoughP if LSD is unavailable.

    Returns
    -------
    segments : list of (N_i, 2) arrays of pixel coords along each segment
    """
    segments = []

    try:
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(gray)
        if lines is None:
            raise ValueError("LSD returned None")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= min_length:
                n = max(int(length / 2), 2)
                xs = np.linspace(x1, x2, n)
                ys = np.linspace(y1, y2, n)
                segments.append(np.stack([xs, ys], axis=1))
    except Exception:
        # Fallback: Canny + HoughP
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines_h = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                   minLineLength=int(min_length),
                                   maxLineGap=10)
        if lines_h is not None:
            for line in lines_h:
                x1, y1, x2, y2 = line[0]
                length = np.hypot(x2 - x1, y2 - y1)
                n = max(int(length / 2), 2)
                xs = np.linspace(x1, x2, n)
                ys = np.linspace(y1, y2, n)
                segments.append(np.stack([xs, ys], axis=1))

    return segments


def detect_lines_from_image(image_path: str,
                              min_length: float = 80.0
                              ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Load image, detect line segments.

    Returns
    -------
    gray     : grayscale image
    segments : list of (N_i, 2) point arrays
    """
    img  = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segs = detect_line_segments_lsd(gray, min_length=min_length)
    return gray, segs


# ---------------------------------------------------------------------------
# Straightness cost
# ---------------------------------------------------------------------------

def _point_to_line_distances(pts: np.ndarray) -> float:
    """
    Mean squared distance of each point to the best-fit line through pts.

    Parameters
    ----------
    pts : (N, 2)

    Returns
    -------
    msd : float  (mean squared orthogonal distance)
    """
    if len(pts) < 2:
        return 0.0
    mu  = pts.mean(axis=0)
    cen = pts - mu
    # PCA: smallest eigenvalue direction is the normal
    _, s, _ = np.linalg.svd(cen, full_matrices=False)
    if s[-1] < 1e-12:
        return 0.0
    # Orthogonal residual = smallest singular value / sqrt(N-1)
    residual = s[-1] ** 2 / max(len(pts) - 1, 1)
    return float(residual)


def straightness_cost(params_vec: np.ndarray,
                       segments: List[np.ndarray],
                       base_params: DistortionParams,
                       active_coefs: List[str]) -> float:
    """
    Objective function: sum of mean-squared straightness residuals across all
    line segments after undistorting with the given coefficients.

    Parameters
    ----------
    params_vec   : 1-D array of the active coefficients
    segments     : list of (N_i, 2) distorted pixel arrays
    base_params  : DistortionParams with fx,fy,cx,cy (and defaults for dist)
    active_coefs : e.g. ['k1','k2'] — which coefficients are optimised
    """
    p = _vec_to_params(params_vec, base_params, active_coefs)

    total = 0.0
    for seg in segments:
        undist = undistort_pixel_coords(seg, p)
        total += _point_to_line_distances(undist)
    return total / max(len(segments), 1)


def _vec_to_params(vec: np.ndarray, base: DistortionParams,
                   active: List[str]) -> DistortionParams:
    """Inject optimisation vector back into a DistortionParams."""
    d = dict(k1=base.k1, k2=base.k2, k3=base.k3,
             p1=base.p1, p2=base.p2)
    for name, val in zip(active, vec):
        d[name] = val
    return DistortionParams(base.fx, base.fy, base.cx, base.cy,
                             d['k1'], d['k2'], d['k3'], d['p1'], d['p2'])


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def estimate_distortion_from_lines(
        image_paths: List[str],
        fx: float, fy: float, cx: float, cy: float,
        active_coefs: Optional[List[str]] = None,
        min_length: float = 80.0,
        method: str = 'Nelder-Mead') -> DistortionParams:
    """
    Estimate distortion coefficients from straight-line scenes.

    Parameters
    ----------
    image_paths  : one or more scene images
    fx,fy,cx,cy  : initial intrinsics (from chessboard or image metadata)
    active_coefs : subset of ['k1','k2','k3','p1','p2'] to optimise.
                   Defaults to ['k1','k2'].
    min_length   : minimum segment length in pixels to keep
    method       : scipy.optimize.minimize method

    Returns
    -------
    DistortionParams with estimated coefficients
    """
    if active_coefs is None:
        active_coefs = ['k1', 'k2']

    base = DistortionParams(fx, fy, cx, cy)

    # Collect all segments from all images
    all_segments: List[np.ndarray] = []
    for path in image_paths:
        try:
            _, segs = detect_lines_from_image(path, min_length=min_length)
            all_segments.extend(segs)
            print(f"  {path}: {len(segs)} segments")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")

    if not all_segments:
        print("  [WARN] No segments found. Returning zero distortion.")
        return base

    print(f"\n  Total segments: {len(all_segments)}")

    x0 = np.zeros(len(active_coefs))

    result = minimize(
        straightness_cost,
        x0,
        args=(all_segments, base, active_coefs),
        method=method,
        options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000},
    )

    params_opt = _vec_to_params(result.x, base, active_coefs)
    print(f"\n  Optimisation: {result.message}")
    print(f"  Final cost: {result.fun:.6f}")
    print(f"  Estimated: {params_opt}")
    return params_opt


# ---------------------------------------------------------------------------
# Straightness residual reporter
# ---------------------------------------------------------------------------

def compute_straightness_residual(segments: List[np.ndarray],
                                   params: DistortionParams) -> float:
    """
    Mean pixel deviation of undistorted segment points from their best-fit line.

    Returns
    -------
    mean_px : average orthogonal distance in pixels
    """
    residuals = []
    for seg in segments:
        undist = undistort_pixel_coords(seg, params)
        mu     = undist.mean(axis=0)
        cen    = undist - mu
        _, s, _ = np.linalg.svd(cen, full_matrices=False)
        # RMS orthogonal distance
        rms = float(s[-1]) / max(np.sqrt(len(seg) - 1), 1)
        residuals.append(rms)
    return float(np.mean(residuals)) if residuals else 0.0


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob, sys
    from pathlib import Path

    img_dir = Path(__file__).parent.parent / "data" / "straight_line_scenes"
    paths   = sorted(glob.glob(str(img_dir / "*.jpg")))
    if not paths:
        paths = sorted(glob.glob(str(img_dir / "*.png")))

    if not paths:
        print("No scene images found in data/straight_line_scenes/. "
              "Place .jpg or .png images there and re-run.")
        sys.exit(0)

    # Placeholder intrinsics – replace with chessboard result
    FX, FY = 800.0, 800.0
    CX, CY = 320.0, 240.0

    print("=== Target-free estimation ===")
    params = estimate_distortion_from_lines(
        paths, FX, FY, CX, CY,
        active_coefs=['k1', 'k2'],
    )
    print(params)
