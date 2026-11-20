"""
evaluation.py
-------------
Evaluation metrics and model-order study.

Metrics
  1. Reprojection error   – mean pixel distance between detected corners and
                            projected corners, computed per view and overall.
  2. Straightness residual – mean orthogonal distance of undistorted line points
                             from their best-fit line.
  3. Model-order study     – repeat calibration with [k1], [k1,k2], [k1,k2,k3]
                             and report errors to detect overfitting.

Author : Yashwanth (Team KASAT)
Course : CSET381 – Applications of AI, Semester 5
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')          # headless – safe for all environments
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

from distortion_model import DistortionParams, project_points
from target_free import compute_straightness_residual, detect_lines_from_image
from calibration import (calibrate_opencv, estimate_initial_K,
                          refine_calibration, _unpack_params)


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------

def reprojection_error(obj_points: List, img_points: List,
                        params: DistortionParams,
                        rvecs: List, tvecs: List
                        ) -> Tuple[float, List[float]]:
    """
    Compute mean reprojection error over all views.

    Returns
    -------
    mean_err  : overall mean pixel error
    per_view  : list of per-view mean errors
    """
    per_view = []
    for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
        projected = project_points(np.array(op), params,
                                    np.array(rv), np.array(tv))
        err = np.linalg.norm(projected - np.array(ip), axis=1).mean()
        per_view.append(float(err))
    mean_err = float(np.mean(per_view))
    return mean_err, per_view


def reprojection_error_opencv(obj_points: List, img_points: List,
                               K: np.ndarray, dist: np.ndarray,
                               rvecs_cv, tvecs_cv) -> Tuple[float, List[float]]:
    """
    Reprojection error using OpenCV projectPoints (for fair comparison).
    """
    per_view = []
    for op, ip, rv, tv in zip(obj_points, img_points, rvecs_cv, tvecs_cv):
        proj, _ = cv2.projectPoints(
            np.array(op, dtype=np.float32),
            rv, tv, K, dist)
        err = np.linalg.norm(
            proj.reshape(-1, 2) - np.array(ip), axis=1).mean()
        per_view.append(float(err))
    return float(np.mean(per_view)), per_view


# ---------------------------------------------------------------------------
# Model-order study
# ---------------------------------------------------------------------------

def model_order_study(obj_points: List, img_points: List,
                       image_size: Tuple[int, int],
                       out_dir: Path) -> Dict[str, float]:
    """
    Calibrate with increasing numbers of radial coefficients and
    record reprojection error on a held-out set (last 20 % of views).

    Parameters
    ----------
    image_size : (width, height)
    out_dir    : directory to save the error plot

    Returns
    -------
    results : dict  model_name -> mean_reprojection_error
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(obj_points)
    n_train = max(int(n * 0.8), 3)
    obj_tr, img_tr = obj_points[:n_train], img_points[:n_train]
    obj_ho, img_ho = obj_points[n_train:], img_points[n_train:]

    if not obj_ho:
        # Not enough views for a split – use all data
        obj_ho, img_ho = obj_tr, img_tr

    models = {
        'k1 only':    cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
        'k1,k2':      cv2.CALIB_FIX_K3,
        'k1,k2,k3':   0,
    }

    results = {}
    for name, flags in models.items():
        ret, K, dist, rvecs_cv, tvecs_cv = cv2.calibrateCamera(
            [np.array(op, dtype=np.float32) for op in obj_tr],
            [np.array(ip, dtype=np.float32) for ip in img_tr],
            image_size, None, None, flags=flags
        )

        # Evaluate on held-out views
        ho_errors = []
        for op, ip in zip(obj_ho, img_ho):
            ret2, rv, tv = cv2.solvePnP(
                np.array(op, dtype=np.float32),
                np.array(ip, dtype=np.float32),
                K, dist)
            proj, _ = cv2.projectPoints(
                np.array(op, dtype=np.float32), rv, tv, K, dist)
            err = np.linalg.norm(
                proj.reshape(-1, 2) - np.array(ip), axis=1).mean()
            ho_errors.append(err)

        mean_err = float(np.mean(ho_errors))
        results[name] = mean_err
        print(f"  {name:12s}  held-out reprojection error = {mean_err:.4f} px")

    # Plot
    _plot_model_order(results, out_dir / "model_order_study.png")
    return results


def _plot_model_order(results: Dict[str, float], save_path: Path):
    names  = list(results.keys())
    errors = [results[n] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, errors, color=['#4C72B0', '#DD8452', '#55A868'],
                  edgecolor='black')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_ylabel("Mean Reprojection Error (px)")
    ax.set_title("Model-Order Study: Held-Out Reprojection Error")
    ax.set_ylim(0, max(errors) * 1.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results_lm: Dict, results_cv: Dict,
                             results_tf: Dict = None):
    """
    Print a formatted comparison table of LM, OpenCV, and target-free results.
    """
    headers = ["Method", "k1", "k2", "k3", "p1", "p2", "Rep. err (px)"]
    row_fmt = "{:<22} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>14.6f}"
    print("\n" + "=" * 90)
    print(f"{'Method':<22} {'k1':>10} {'k2':>10} {'k3':>10} "
          f"{'p1':>10} {'p2':>10} {'Rep.err(px)':>14}")
    print("=" * 90)

    for label, p, err in [
        ("LM (ours)", results_lm['params'], results_lm['error']),
        ("OpenCV",    results_cv['params'], results_cv['error']),
    ]:
        print(row_fmt.format(label,
                              p.k1, p.k2, p.k3, p.p1, p.p2, err))

    if results_tf is not None:
        p   = results_tf['params']
        err = results_tf.get('straightness', float('nan'))
        print(f"{'Target-free':<22} {p.k1:>10.6f} {p.k2:>10.6f} "
              f"{p.k3:>10.6f} {p.p1:>10.6f} {p.p2:>10.6f} "
              f"{'(straight):':>12} {err:>6.6f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Per-view error plot
# ---------------------------------------------------------------------------

def plot_per_view_errors(per_view_lm: List[float],
                          per_view_cv: List[float],
                          out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(1, len(per_view_lm) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, per_view_lm, 'o-', label='LM (ours)', color='#4C72B0')
    ax.plot(xs, per_view_cv, 's--', label='OpenCV',   color='#DD8452')
    ax.set_xlabel("View index")
    ax.set_ylabel("Mean reprojection error (px)")
    ax.set_title("Per-View Reprojection Error")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = out_dir / "per_view_errors.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob, sys
    from pathlib import Path

    img_dir = Path(__file__).parent.parent / "data" / "chessboard"
    paths   = sorted(glob.glob(str(img_dir / "*.jpg")))
    if not paths:
        paths = sorted(glob.glob(str(img_dir / "*.png")))

    if not paths:
        print("No chessboard images found in data/chessboard/. "
              "Place .jpg or .png files there and re-run.")
        sys.exit(0)

    from calibration import detect_chessboard_corners
    sample = cv2.imread(paths[0])
    h, w   = sample.shape[:2]

    obj_pts, img_pts = detect_chessboard_corners(paths)
    if len(obj_pts) < 4:
        print("Need at least 4 views.")
        sys.exit(1)

    K0         = estimate_initial_K(obj_pts, img_pts, (w, h))
    params_lm  = refine_calibration(K0, obj_pts, img_pts)
    params_cv  = calibrate_opencv(obj_pts, img_pts, (w, h))

    out = Path(__file__).parent.parent / "results"

    print("\n=== Model-order study ===")
    model_order_study(obj_pts, img_pts, (w, h), out)

    print("\n=== Comparison table ===")
    print_comparison_table(
        {'params': params_lm, 'error': 0.0},
        {'params': params_cv, 'error': 0.0},
    )
