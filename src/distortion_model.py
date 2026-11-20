"""
distortion_model.py
-------------------
Brown–Conrady lens distortion model (forward and inverse).

Five coefficients
  k1, k2, k3  – radial
  p1, p2       – tangential

Author : Kaustubh Singh (Team KASAT)
Course : CSET381 – Applications of AI, Semester 5
"""

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class DistortionParams:
    """Holds camera intrinsics + distortion coefficients."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 k1: float = 0.0, k2: float = 0.0, k3: float = 0.0,
                 p1: float = 0.0, p2: float = 0.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0,       0,       1]], dtype=np.float64)

    @property
    def dist_vec(self) -> np.ndarray:
        """OpenCV-compatible distortion vector [k1,k2,p1,p2,k3]."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                        dtype=np.float64)

    def __repr__(self) -> str:
        return (f"DistortionParams(fx={self.fx:.4f}, fy={self.fy:.4f}, "
                f"cx={self.cx:.4f}, cy={self.cy:.4f}, "
                f"k1={self.k1:.6f}, k2={self.k2:.6f}, k3={self.k3:.6f}, "
                f"p1={self.p1:.6f}, p2={self.p2:.6f})")


# ---------------------------------------------------------------------------
# Forward model  (ideal → distorted)
# ---------------------------------------------------------------------------

def distort_points(pts_norm: np.ndarray, params: DistortionParams) -> np.ndarray:
    """
    Apply Brown–Conrady distortion to normalised image coordinates.

    Parameters
    ----------
    pts_norm : (N, 2)  –  undistorted normalised coords (x_u, y_u)
    params   : DistortionParams

    Returns
    -------
    pts_dist : (N, 2)  –  distorted normalised coords
    """
    x = pts_norm[:, 0]
    y = pts_norm[:, 1]

    r2 = x * x + y * y          # r² = x² + y²
    r4 = r2 * r2
    r6 = r4 * r2

    k1, k2, k3 = params.k1, params.k2, params.k3
    p1, p2     = params.p1, params.p2

    # Radial scale factor
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    # Tangential offsets
    dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    xd = x * radial + dx
    yd = y * radial + dy

    return np.stack([xd, yd], axis=1)


def project_points(pts_3d: np.ndarray, params: DistortionParams,
                   rvec: np.ndarray = None,
                   tvec: np.ndarray = None) -> np.ndarray:
    """
    Full projection: 3-D world → 2-D distorted pixel.

    Parameters
    ----------
    pts_3d  : (N, 3)
    params  : DistortionParams
    rvec    : (3,) rotation vector (Rodrigues); identity if None
    tvec    : (3,) translation vector; zero if None

    Returns
    -------
    pts_px  : (N, 2)
    """
    pts_3d = np.asarray(pts_3d, dtype=np.float64)

    # Rigid-body transform
    if rvec is not None and tvec is not None:
        R = _rodrigues(np.asarray(rvec, dtype=np.float64))
        pts_cam = (R @ pts_3d.T).T + np.asarray(tvec, dtype=np.float64)
    else:
        pts_cam = pts_3d.copy()

    # Normalised coordinates
    z = pts_cam[:, 2]
    pts_norm = pts_cam[:, :2] / z[:, np.newaxis]

    # Distort
    pts_dist = distort_points(pts_norm, params)

    # Back to pixels
    px = pts_dist[:, 0] * params.fx + params.cx
    py = pts_dist[:, 1] * params.fy + params.cy

    return np.stack([px, py], axis=1)


# ---------------------------------------------------------------------------
# Inverse model  (distorted → ideal, iterative)
# ---------------------------------------------------------------------------

def undistort_points(pts_dist_norm: np.ndarray, params: DistortionParams,
                     max_iter: int = 20, tol: float = 1e-9) -> np.ndarray:
    """
    Invert Brown–Conrady distortion iteratively (no closed form exists).

    Parameters
    ----------
    pts_dist_norm : (N, 2)  –  distorted normalised coords
    params        : DistortionParams
    max_iter      : maximum Newton iterations
    tol           : convergence tolerance (pixels in normalised space)

    Returns
    -------
    pts_undist : (N, 2)  –  undistorted normalised coords
    """
    xd = pts_dist_norm[:, 0].copy()
    yd = pts_dist_norm[:, 1].copy()

    # Initialise with the distorted coords (zeroth-order guess)
    xu = xd.copy()
    yu = yd.copy()

    for _ in range(max_iter):
        r2 = xu * xu + yu * yu
        r4 = r2 * r2
        r6 = r4 * r2

        radial = 1.0 + params.k1 * r2 + params.k2 * r4 + params.k3 * r6
        dx = 2.0 * params.p1 * xu * yu + params.p2 * (r2 + 2.0 * xu * xu)
        dy = params.p1 * (r2 + 2.0 * yu * yu) + 2.0 * params.p2 * xu * yu

        xu_new = (xd - dx) / radial
        yu_new = (yd - dy) / radial

        delta = np.max(np.abs(xu_new - xu) + np.abs(yu_new - yu))
        xu, yu = xu_new, yu_new

        if delta < tol:
            break

    return np.stack([xu, yu], axis=1)


def undistort_pixel_coords(pts_px: np.ndarray, params: DistortionParams,
                           **kwargs) -> np.ndarray:
    """
    Undistort 2-D pixel coordinates → corrected pixel coordinates.

    Parameters
    ----------
    pts_px : (N, 2)  –  distorted pixel coords

    Returns
    -------
    (N, 2)  –  undistorted pixel coords
    """
    # Pixel → normalised
    xn = (pts_px[:, 0] - params.cx) / params.fx
    yn = (pts_px[:, 1] - params.cy) / params.fy
    pts_norm = np.stack([xn, yn], axis=1)

    # Iterative inversion
    pts_undist = undistort_points(pts_norm, params, **kwargs)

    # Normalised → pixel
    pu = pts_undist[:, 0] * params.fx + params.cx
    pv = pts_undist[:, 1] * params.fy + params.cy
    return np.stack([pu, pv], axis=1)


# ---------------------------------------------------------------------------
# Image remapping
# ---------------------------------------------------------------------------

def build_undistort_map(height: int, width: int,
                        params: DistortionParams) -> tuple:
    """
    Build dense (map_x, map_y) for cv2.remap-style undistortion.

    Returns
    -------
    map_x, map_y : (H, W) float32 arrays
        Each output pixel (r, c) maps to source pixel (map_x[r,c], map_y[r,c]).
    """
    # Grid of destination pixel coordinates
    xs = np.arange(width,  dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)

    pts_dst = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Find where each destination pixel came from in the distorted image
    pts_src = undistort_pixel_coords(pts_dst, params)

    map_x = pts_src[:, 0].reshape(height, width).astype(np.float32)
    map_y = pts_src[:, 1].reshape(height, width).astype(np.float32)

    return map_x, map_y


def remap_bilinear(image: np.ndarray,
                   map_x: np.ndarray,
                   map_y: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy bilinear remapping (no OpenCV dependency).

    Parameters
    ----------
    image : (H, W) or (H, W, C)
    map_x, map_y : (H_out, W_out) float32

    Returns
    -------
    output : same dtype and channels as image
    """
    H, W = image.shape[:2]
    H_out, W_out = map_x.shape

    # Clamp source coordinates
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts
    wx = (map_x - x0).astype(np.float32)
    wy = (map_y - y0).astype(np.float32)

    # Clamp to valid range
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    # Mask out-of-bounds
    valid = (map_x >= 0) & (map_x < W) & (map_y >= 0) & (map_y < H)

    if image.ndim == 2:
        out = (image[y0c, x0c] * (1 - wy) * (1 - wx)
             + image[y0c, x1c] * (1 - wy) *      wx
             + image[y1c, x0c] *      wy  * (1 - wx)
             + image[y1c, x1c] *      wy  *      wx)
        out[~valid] = 0
    else:
        wx  = wx[:, :, np.newaxis]
        wy  = wy[:, :, np.newaxis]
        out = (image[y0c, x0c] * (1 - wy) * (1 - wx)
             + image[y0c, x1c] * (1 - wy) *      wx
             + image[y1c, x0c] *      wy  * (1 - wx)
             + image[y1c, x1c] *      wy  *      wx)
        out[~valid] = 0

    return np.clip(out, 0, 255).astype(image.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation vector → 3×3 rotation matrix."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K_ = np.array([[ 0,    -k[2],  k[1]],
                   [ k[2],  0,    -k[0]],
                   [-k[1],  k[0],  0   ]])
    return np.eye(3) + np.sin(theta) * K_ + (1 - np.cos(theta)) * (K_ @ K_)
