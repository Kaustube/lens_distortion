"""
demo_tool.py
------------
End-to-end demo tool.

  Distorted image in → corrected image + parameters out.

Usage examples
--------------
  # Chessboard calibration then correct a photo:
  python demo_tool.py --mode chessboard \
      --board-dir ../data/chessboard \
      --input-image ../data/sample.jpg \
      --output-image ../data/output/corrected.jpg

  # Target-free estimation then correct a photo:
  python demo_tool.py --mode target-free \
      --scene-dir ../data/straight_line_scenes \
      --input-image ../data/sample.jpg \
      --output-image ../data/output/corrected.jpg \
      --fx 800 --fy 800 --cx 320 --cy 240

  # Load saved parameters and correct a photo:
  python demo_tool.py --mode load \
      --params-file ../results/params.npz \
      --input-image ../data/sample.jpg \
      --output-image ../data/output/corrected.jpg

Authors : B.T. Ashish + Kaustubh Singh (integration) — Team KASAT
Course  : CSET381 – Applications of AI, Semester 5
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path when running from root
sys.path.insert(0, str(Path(__file__).parent))

from distortion_model import DistortionParams, build_undistort_map, remap_bilinear
from calibration import (detect_chessboard_corners, estimate_initial_K,
                          refine_calibration, calibrate_opencv)
from target_free import (estimate_distortion_from_lines,
                          detect_lines_from_image,
                          compute_straightness_residual)
from evaluation import (reprojection_error_opencv, model_order_study,
                         print_comparison_table, plot_per_view_errors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_params(params: DistortionParams, path: str):
    np.savez(path,
             fx=params.fx, fy=params.fy,
             cx=params.cx, cy=params.cy,
             k1=params.k1, k2=params.k2, k3=params.k3,
             p1=params.p1, p2=params.p2)
    print(f"  Parameters saved → {path}")


def load_params(path: str) -> DistortionParams:
    d = np.load(path)
    return DistortionParams(
        float(d['fx']), float(d['fy']),
        float(d['cx']), float(d['cy']),
        float(d['k1']), float(d['k2']), float(d['k3']),
        float(d['p1']), float(d['p2']),
    )


def correct_image(image_path: str, params: DistortionParams,
                   output_path: str, use_opencv: bool = False):
    """
    Undistort image and save.  Optionally compare with cv2.undistort.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Cannot read {image_path}")
        return

    h, w = img.shape[:2]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if use_opencv:
        corrected = cv2.undistort(img, params.K, params.dist_vec)
    else:
        map_x, map_y = build_undistort_map(h, w, params)
        corrected    = remap_bilinear(img, map_x, map_y)

    cv2.imwrite(output_path, corrected)
    print(f"  Corrected image → {output_path}")

    # Side-by-side comparison
    comp_path = str(Path(output_path).parent /
                    (Path(output_path).stem + "_comparison.jpg"))
    separator = np.ones((h, 4, img.shape[2] if img.ndim == 3 else 1),
                         dtype=np.uint8) * 180
    comparison = np.concatenate([img, separator, corrected], axis=1)
    cv2.imwrite(comp_path, comparison)
    print(f"  Side-by-side    → {comp_path}")


# ---------------------------------------------------------------------------
# Mode: chessboard
# ---------------------------------------------------------------------------

def run_chessboard(args):
    board_dir = Path(args.board_dir)
    paths = (sorted(glob.glob(str(board_dir / "*.jpg"))) +
             sorted(glob.glob(str(board_dir / "*.png"))))
    if not paths:
        print(f"[ERROR] No images in {board_dir}")
        sys.exit(1)

    sample = cv2.imread(paths[0])
    h, w   = sample.shape[:2]

    print(f"\n{'='*60}")
    print("STEP 1 – Detecting chessboard corners")
    print('='*60)
    obj_pts, img_pts = detect_chessboard_corners(
        paths,
        board_size=(args.board_cols, args.board_rows),
        square_size_mm=args.square_size,
    )
    if len(obj_pts) < 4:
        print("[ERROR] Need at least 4 valid views.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("STEP 2 – Initial intrinsics (Zhang)")
    print('='*60)
    K0 = estimate_initial_K(obj_pts, img_pts, (w, h))
    print(K0)

    print(f"\n{'='*60}")
    print("STEP 3 – LM refinement (our implementation)")
    print('='*60)
    params_lm = refine_calibration(K0, obj_pts, img_pts,
                                    fix_k3=args.fix_k3)
    print(params_lm)

    print(f"\n{'='*60}")
    print("STEP 4 – OpenCV calibrateCamera (reference)")
    print('='*60)
    params_cv = calibrate_opencv(obj_pts, img_pts, (w, h),
                                  fix_k3=args.fix_k3)
    print(params_cv)

    print(f"\n{'='*60}")
    print("STEP 5 – Reprojection errors")
    print('='*60)
    flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0
    _, K_cv, dist_cv, rvecs_cv, tvecs_cv = cv2.calibrateCamera(
        [np.array(op, dtype=np.float32) for op in obj_pts],
        [np.array(ip, dtype=np.float32) for ip in img_pts],
        (w, h), None, None, flags=flags)

    err_cv, pv_cv = reprojection_error_opencv(
        obj_pts, img_pts, K_cv, dist_cv, rvecs_cv, tvecs_cv)

    # LM reprojection error (re-estimate extrinsics with LM params)
    from calibration import estimate_extrinsics
    rvecs_lm, tvecs_lm = [], []
    for op, ip in zip(obj_pts, img_pts):
        rv, tv = estimate_extrinsics(np.array(op), np.array(ip), params_lm.K)
        rvecs_lm.append(rv)
        tvecs_lm.append(tv)
    from evaluation import reprojection_error as _rep_err
    err_lm, pv_lm = _rep_err(obj_pts, img_pts, params_lm, rvecs_lm, tvecs_lm)

    print(f"  LM   mean reprojection error : {err_lm:.6f} px")
    print(f"  OpenCV mean reprojection error: {err_cv:.6f} px")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_per_view_errors(pv_lm, pv_cv, out_dir)

    print(f"\n{'='*60}")
    print("STEP 6 – Model-order study")
    print('='*60)
    model_order_study(obj_pts, img_pts, (w, h), out_dir)

    print_comparison_table(
        {'params': params_lm, 'error': err_lm},
        {'params': params_cv, 'error': err_cv},
    )

    # Save parameters
    save_params(params_lm, str(out_dir / "params_lm.npz"))
    save_params(params_cv, str(out_dir / "params_cv.npz"))

    # Correct supplied image
    if args.input_image:
        print(f"\n{'='*60}")
        print("STEP 7 – Correcting input image")
        print('='*60)
        correct_image(args.input_image, params_lm,
                       args.output_image or str(
                           Path(args.results_dir) / "corrected_lm.jpg"))

    return params_lm


# ---------------------------------------------------------------------------
# Mode: target-free
# ---------------------------------------------------------------------------

def run_target_free(args):
    scene_dir = Path(args.scene_dir)
    paths = (sorted(glob.glob(str(scene_dir / "*.jpg"))) +
             sorted(glob.glob(str(scene_dir / "*.png"))))
    if not paths:
        print(f"[ERROR] No images in {scene_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("STEP 1 – Target-free distortion estimation")
    print('='*60)
    params = estimate_distortion_from_lines(
        paths,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        active_coefs=args.active_coefs.split(',') if args.active_coefs
                     else ['k1', 'k2'],
        min_length=args.min_length,
    )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_params(params, str(out_dir / "params_tf.npz"))

    if args.input_image:
        print(f"\n{'='*60}")
        print("STEP 2 – Correcting input image")
        print('='*60)
        correct_image(args.input_image, params,
                       args.output_image or str(
                           Path(args.results_dir) / "corrected_tf.jpg"))

    return params


# ---------------------------------------------------------------------------
# Mode: load
# ---------------------------------------------------------------------------

def run_load(args):
    params = load_params(args.params_file)
    print(f"  Loaded: {params}")
    if args.input_image:
        correct_image(args.input_image, params,
                       args.output_image or str(
                           Path(args.results_dir) / "corrected.jpg"))
    return params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lens distortion correction demo (Team KASAT, CSET381)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--mode', choices=['chessboard', 'target-free', 'load'],
                   required=True,
                   help='Calibration mode')

    # Chessboard options
    p.add_argument('--board-dir',   default='../data/chessboard')
    p.add_argument('--board-cols',  type=int, default=9)
    p.add_argument('--board-rows',  type=int, default=6)
    p.add_argument('--square-size', type=float, default=25.0,
                   help='Square size in mm')
    p.add_argument('--fix-k3',      action='store_true',
                   help='Fix k3=0 (use for mild lenses)')

    # Target-free options
    p.add_argument('--scene-dir',    default='../data/straight_line_scenes')
    p.add_argument('--fx',   type=float, default=800.0)
    p.add_argument('--fy',   type=float, default=800.0)
    p.add_argument('--cx',   type=float, default=320.0)
    p.add_argument('--cy',   type=float, default=240.0)
    p.add_argument('--active-coefs', default='k1,k2',
                   help='Comma-separated list: k1,k2,k3,p1,p2')
    p.add_argument('--min-length',   type=float, default=80.0)

    # Load mode
    p.add_argument('--params-file', default='../results/params_lm.npz')

    # Common
    p.add_argument('--input-image',  default=None)
    p.add_argument('--output-image', default=None)
    p.add_argument('--results-dir',  default='../results')

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == 'chessboard':
        run_chessboard(args)
    elif args.mode == 'target-free':
        run_target_free(args)
    elif args.mode == 'load':
        run_load(args)


if __name__ == "__main__":
    main()
