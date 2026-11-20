"""
generate_synthetic_data.py
--------------------------
Generates synthetic chessboard and straight-line images with known distortion
so the pipeline can be tested without a real camera.

Run this first if you have no real images:
    python generate_synthetic_data.py

Outputs
  ../data/chessboard/synth_*.png   – 20 synthetic chessboard views
  ../data/straight_line_scenes/synth_lines_*.png – 5 grid scenes

Author : B.T. Ashish (Team KASAT)
"""

import numpy as np
import cv2
from pathlib import Path

# ---- Ground-truth parameters ------------------------------------------------
GT_FX, GT_FY = 800.0, 800.0
GT_CX, GT_CY = 320.0, 240.0
GT_K1, GT_K2, GT_K3 = -0.35, 0.12, 0.0
GT_P1, GT_P2         = 0.001, -0.0005

IMG_W, IMG_H = 640, 480
BOARD_COLS, BOARD_ROWS = 9, 6
SQ = 25.0  # mm


def distort_pixel(xu: float, yu: float) -> tuple:
    """Apply Brown–Conrady to a single pixel (normalised coords)."""
    r2 = xu*xu + yu*yu
    r4 = r2*r2; r6 = r4*r2
    rad = 1 + GT_K1*r2 + GT_K2*r4 + GT_K3*r6
    dx  = 2*GT_P1*xu*yu + GT_P2*(r2 + 2*xu*xu)
    dy  = GT_P1*(r2 + 2*yu*yu) + 2*GT_P2*xu*yu
    xd  = xu*rad + dx
    yd  = yu*rad + dy
    px  = xd*GT_FX + GT_CX
    py  = yd*GT_FY + GT_CY
    return px, py


def render_distorted_chessboard(rvec, tvec) -> np.ndarray:
    """Render a chessboard with known distortion."""
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 200
    pts3d = []
    for r in range(BOARD_ROWS + 1):
        for c in range(BOARD_COLS + 1):
            pts3d.append([c * SQ, r * SQ, 0.0])
    pts3d = np.array(pts3d, dtype=np.float64)

    # Rigid transform
    theta = np.linalg.norm(rvec)
    if theta > 1e-9:
        k = rvec / theta
        K_ = np.array([[ 0,    -k[2],  k[1]],
                        [ k[2],  0,    -k[0]],
                        [-k[1],  k[0],  0   ]])
        R = np.eye(3) + np.sin(theta)*K_ + (1-np.cos(theta))*(K_@K_)
    else:
        R = np.eye(3)

    pts_cam = (R @ pts3d.T).T + tvec

    # Draw squares
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            idx = r*(BOARD_COLS+1) + c
            corners_3d = pts_cam[[idx, idx+1,
                                   idx+BOARD_COLS+2,
                                   idx+BOARD_COLS+1]]
            poly = []
            for pt in corners_3d:
                xu = pt[0] / pt[2]; yu = pt[1] / pt[2]
                px, py = distort_pixel(xu, yu)
                poly.append([px, py])
            poly = np.array(poly, dtype=np.int32)
            color = (0, 0, 0) if (r + c) % 2 == 0 else (255, 255, 255)
            cv2.fillConvexPoly(img, poly, color)
    return img


def generate_chessboards(n: int = 20, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "data" / "chessboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for i in range(n):
        angle  = rng.uniform(-0.5, 0.5, 3)
        tvec   = np.array([
            rng.uniform(-50, 50),
            rng.uniform(-40, 40),
            rng.uniform(300, 500),
        ])
        img = render_distorted_chessboard(angle, tvec)
        path = out_dir / f"synth_{i:02d}.png"
        cv2.imwrite(str(path), img)
    print(f"  Generated {n} chessboard images → {out_dir}")


def generate_line_scenes(n: int = 5, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "data" / "straight_line_scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 240
        # Draw horizontal and vertical lines (straight in ideal coords)
        for y_world in range(50, IMG_H, 60):
            pts = []
            for x_px in range(0, IMG_W, 2):
                xu = (x_px - GT_CX) / GT_FX
                yu = (y_world - GT_CY) / GT_FY
                px, py = distort_pixel(xu, yu)
                pts.append((int(px), int(py)))
            for j in range(len(pts)-1):
                cv2.line(img, pts[j], pts[j+1], (0, 0, 180), 2)
        for x_world in range(50, IMG_W, 80):
            pts = []
            for y_px in range(0, IMG_H, 2):
                xu = (x_world - GT_CX) / GT_FX
                yu = (y_px - GT_CY) / GT_FY
                px, py = distort_pixel(xu, yu)
                pts.append((int(px), int(py)))
            for j in range(len(pts)-1):
                cv2.line(img, pts[j], pts[j+1], (180, 0, 0), 2)
        path = out_dir / f"synth_lines_{i:02d}.png"
        cv2.imwrite(str(path), img)
    print(f"  Generated {n} line-scene images  → {out_dir}")


if __name__ == "__main__":
    print("Generating synthetic data …")
    generate_chessboards()
    generate_line_scenes()
    print("\nGround-truth parameters:")
    print(f"  fx={GT_FX}, fy={GT_FY}, cx={GT_CX}, cy={GT_CY}")
    print(f"  k1={GT_K1}, k2={GT_K2}, k3={GT_K3}, p1={GT_P1}, p2={GT_P2}")
    print("\nDone. Now run demo_tool.py --mode chessboard")
