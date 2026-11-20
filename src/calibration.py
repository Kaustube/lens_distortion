"""
calibration.py
--------------
Target-based (chessboard) camera calibration.

Pipeline
  1. Detect corners with OpenCV (sub-pixel refinement).
  2. Compute an initial homography per view, extract an initial K.
  3. Refine all parameters with Levenberg–Marquardt (scipy).
  4. Compare with OpenCV's built-in calibrateCamera.

Author : Aryan Parmar (Team KASAT)
Course : CSET381 – Applications of AI, Semester 5
"""

import numpy as np
import cv2
from scipy.optimize import least_squares
from pathlib import Path
from typing import List, Tuple, Optional

from distortion_model import DistortionParams, project_points, _rodrigues


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------

def detect_chessboard_corners(image_paths: List[str],
                               board_size: Tuple[int, int] = (9, 6),
                               square_size_mm: float = 25.0
                               ) -> Tuple[List, List]:
    """
    Detect inner corners on chessboard images.

    Parameters
    ----------
    image_paths    : list of file paths
    board_size     : (cols, rows) of inner corners
    square_size_mm : physical side length of one square

    Returns
    -------
    obj_points : list of (N,3) arrays – 3-D world coords (Z=0 plane)
    img_points : list of (N,2) arrays – detected pixel coords
    """
    cols, rows = board_size
    # World coordinates on the Z=0 plane
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = (np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                   * square_size_mm)

    obj_points, img_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [WARN] Could not read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            # Sub-pixel refinement
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           criteria)
            obj_points.append(objp)
            img_points.append(corners_sub.reshape(-1, 2))
            print(f"  [OK]  {Path(path).name}")
        else:
            print(f"  [MISS] {Path(path).name}")

    print(f"\n  Detected in {len(img_points)}/{len(image_paths)} images.")
    return obj_points, img_points


# ---------------------------------------------------------------------------
# Homography-based initial estimate of K
# ---------------------------------------------------------------------------

def _compute_homography(obj_pts: np.ndarray,
                         img_pts: np.ndarray) -> np.ndarray:
    """DLT homography from Z=0 plane to image."""
    H, _ = cv2.findHomography(obj_pts[:, :2], img_pts)
    return H


def _vij(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """Helper vector for Zhang's constraint."""
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j],
    ])


def estimate_initial_K(obj_points: List, img_points: List,
                        image_size: Tuple[int, int]) -> np.ndarray:
    """
    Zhang (2000) closed-form K from multiple homographies.

    image_size : (width, height)
    """
    V = []
    for op, ip in zip(obj_points, img_points):
        H = _compute_homography(np.array(op), np.array(ip))
        V.append(_vij(H, 0, 1))
        V.append(_vij(H, 0, 0) - _vij(H, 1, 1))

    V = np.array(V)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]                        # solution: last row of Vt

    # Extract K from b (B = K^{-T} K^{-1})
    B11, B12, B22, B13, B23, B33 = b
    denom = B11 * B22 - B12 * B12

    cy = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11
    fx  = np.sqrt(lam / B11)
    fy  = np.sqrt(lam * B11 / denom)
    s   = -B12 * fx * fx * fy / lam
    cx  = s * cy / fy - B13 * fx * fx / lam

    K = np.array([[fx, s,  cx],
                  [0,  fy, cy],
                  [0,  0,  1 ]], dtype=np.float64)
    return K


# ---------------------------------------------------------------------------
# Extrinsic estimation per view
# ---------------------------------------------------------------------------

def estimate_extrinsics(obj_pts: np.ndarray, img_pts: np.ndarray,
                         K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rvec, tvec) from a single view using solvePnP."""
    ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                    K, np.zeros(5))
    return rvec.ravel(), tvec.ravel()


# ---------------------------------------------------------------------------
# Levenberg–Marquardt bundle adjustment
# ---------------------------------------------------------------------------

def _pack_params(K: np.ndarray, dist: np.ndarray,
                 rvecs: List, tvecs: List) -> np.ndarray:
    """Flatten all optimisation variables into a 1-D vector."""
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])  # fx,fy,cx,cy
    extrinsics = np.concatenate([np.concatenate([r, t])
                                  for r, t in zip(rvecs, tvecs)])
    return np.concatenate([intrinsics, dist, extrinsics])


def _unpack_params(x: np.ndarray, n_views: int, n_dist: int = 5):
    """Unpack the 1-D parameter vector."""
    fx, fy, cx, cy = x[0], x[1], x[2], x[3]
    dist = x[4:4 + n_dist]
    ext  = x[4 + n_dist:]
    rvecs, tvecs = [], []
    for i in range(n_views):
        rvecs.append(ext[6 * i: 6 * i + 3])
        tvecs.append(ext[6 * i + 3: 6 * i + 6])
    return fx, fy, cx, cy, dist, rvecs, tvecs


def _residuals(x: np.ndarray, obj_points: List, img_points: List,
               n_dist: int) -> np.ndarray:
    n_views = len(obj_points)
    fx, fy, cx, cy, dist, rvecs, tvecs = _unpack_params(x, n_views, n_dist)

    if n_dist == 4:
        k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], 0.0
    else:
        k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]

    params = DistortionParams(fx, fy, cx, cy, k1, k2, k3, p1, p2)

    res = []
    for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
        projected = project_points(op, params, rv, tv)
        res.append((projected - ip).ravel())
    return np.concatenate(res)


def refine_calibration(K_init: np.ndarray,
                        obj_points: List, img_points: List,
                        fix_k3: bool = False) -> DistortionParams:
    """
    Full bundle adjustment with Levenberg–Marquardt.

    Parameters
    ----------
    K_init     : initial intrinsic matrix (3×3)
    fix_k3     : if True, optimise only k1,k2,p1,p2 (k3 = 0)
    """
    n_dist = 4 if fix_k3 else 5
    dist_init = np.zeros(n_dist)

    rvecs, tvecs = [], []
    for op, ip in zip(obj_points, img_points):
        rv, tv = estimate_extrinsics(np.array(op), np.array(ip), K_init)
        rvecs.append(rv)
        tvecs.append(tv)

    x0 = _pack_params(K_init, dist_init, rvecs, tvecs)

    result = least_squares(
        _residuals, x0,
        args=(obj_points, img_points, n_dist),
        method='lm',
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
        max_nfev=5000,
    )

    fx, fy, cx, cy, dist, _, _ = _unpack_params(result.x,
                                                   len(obj_points), n_dist)
    if fix_k3:
        k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], 0.0
    else:
        k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]

    params = DistortionParams(fx, fy, cx, cy, k1, k2, k3, p1, p2)
    print(f"\n  LM converged: cost={result.cost:.4f}  "
          f"nfev={result.nfev}  status={result.status}")
    return params


# ---------------------------------------------------------------------------
# OpenCV comparison
# ---------------------------------------------------------------------------

def calibrate_opencv(obj_points: List, img_points: List,
                      image_size: Tuple[int, int],
                      fix_k3: bool = False) -> DistortionParams:
    """
    Run cv2.calibrateCamera and return a DistortionParams.

    image_size : (width, height)
    """
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    ret, K, dist, _, _ = cv2.calibrateCamera(
        [np.array(op, dtype=np.float32) for op in obj_points],
        [np.array(ip, dtype=np.float32) for ip in img_points],
        image_size, None, None, flags=flags
    )
    k1, k2, p1, p2 = dist[0, :4]
    k3 = 0.0 if fix_k3 else float(dist[0, 4])
    params = DistortionParams(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                               k1, k2, k3, p1, p2)
    print(f"\n  OpenCV RMS reprojection error: {ret:.6f} px")
    return params


# ---------------------------------------------------------------------------
# Quick self-test  (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob, sys

    img_dir = Path(__file__).parent.parent / "data" / "chessboard"
    paths = sorted(glob.glob(str(img_dir / "*.jpg")))
    if not paths:
        paths = sorted(glob.glob(str(img_dir / "*.png")))

    if not paths:
        print("No chessboard images found in data/chessboard/. "
              "Place .jpg or .png files there and re-run.")
        sys.exit(0)

    # Read image size from first image
    sample = cv2.imread(paths[0])
    h, w   = sample.shape[:2]

    print("=== Detecting corners ===")
    obj_pts, img_pts = detect_chessboard_corners(paths)

    if len(obj_pts) < 4:
        print("Need at least 4 valid views. Exiting.")
        sys.exit(1)

    print("\n=== Initial K (Zhang) ===")
    K0 = estimate_initial_K(obj_pts, img_pts, (w, h))
    print(K0)

    print("\n=== LM refinement ===")
    params_lm = refine_calibration(K0, obj_pts, img_pts, fix_k3=False)
    print(params_lm)

    print("\n=== OpenCV reference ===")
    params_cv = calibrate_opencv(obj_pts, img_pts, (w, h))
    print(params_cv)
