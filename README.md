# Lens Distortion Correction
### Team KASAT | CSET381 – Applications of AI | Semester 5
**Guide: Dr. Naveen Kumar**

---

## Team & Contributions

| Member | Role | Files owned |
|---|---|---|
| **Kaustubh Singh** (Lead) | Overall architecture, Brown–Conrady forward/inverse model, integration | `distortion_model.py`, `demo_tool.py` (integration) |
| **Aryan Parmar** | Chessboard capture, corner detection, target-based calibration, OpenCV comparison | `calibration.py` |
| **Sukhbeer Kaur** | Target-free estimator — edge detection, line grouping, robust cost minimisation | `target_free.py` |
| **Yashwanth** | Evaluation metrics, model-order study, error plots | `evaluation.py` |
| **B.T. Ashish** | Data capture (cameras 2 & 3), synthetic data generator, demo tool CLI, documentation | `generate_synthetic_data.py`, `demo_tool.py` (CLI/IO) |

---

## Project Summary

We study how camera lenses warp images and implement the standard **Brown–Conrady model** to undo the distortion.  
Real lenses bend straight lines into curves (radial distortion) and slightly skew the image when the lens is misaligned (tangential distortion).

We implement the full pipeline **from scratch in NumPy/SciPy**, estimate the five distortion coefficients in two ways, and compare both against OpenCV.

---

## The Five Coefficients

| Symbol | Type | Effect |
|---|---|---|
| k1, k2, k3 | Radial | Bend magnitude grows with distance from centre |
| p1, p2 | Tangential | Skew from lens–sensor misalignment |

---

## Project Structure

```
lens_distortion_correction/
├── data/
│   ├── chessboard/            ← Place chessboard .jpg/.png images here
│   ├── straight_line_scenes/  ← Place scene images with long edges here
│   └── output/                ← Corrected images written here
├── src/
│   ├── distortion_model.py    ← Brown–Conrady forward/inverse, remapping
│   ├── calibration.py         ← Corner detection, Zhang init-K, LM refinement
│   ├── target_free.py         ← Line detection, straightness cost, optimizer
│   ├── evaluation.py          ← Reprojection error, model-order study, plots
│   ├── demo_tool.py           ← End-to-end CLI tool
│   └── generate_synthetic_data.py  ← Synthetic test data (no camera needed)
├── results/                   ← Plots and saved parameters written here
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic test data (no camera needed)
```bash
cd src
python generate_synthetic_data.py
```
This writes 20 synthetic distorted chessboard images and 5 line-grid scenes.

### 3. Run chessboard calibration
```bash
python demo_tool.py --mode chessboard \
    --board-dir ../data/chessboard \
    --board-cols 9 --board-rows 6 \
    --square-size 25
```

### 4. Run target-free estimation
```bash
python demo_tool.py --mode target-free \
    --scene-dir ../data/straight_line_scenes \
    --fx 800 --fy 800 --cx 320 --cy 240 \
    --active-coefs k1,k2
```

### 5. Correct an image with saved parameters
```bash
python demo_tool.py --mode load \
    --params-file ../results/params_lm.npz \
    --input-image ../data/sample.jpg \
    --output-image ../data/output/corrected.jpg
```

---

## Pipeline Overview

```
Step 1  Capture          25–30 chessboard shots + straight-edge scenes
Step 2  Forward model    Brown–Conrady polynomial (NumPy)
Step 3  Inverse model    Iterative inversion (no closed form)
Step 4  Calibration      Corners → Homography init → LM bundle adjustment
Step 5  Target-free      Edges → line segments → minimise curvature
Step 6  Evaluation       Reprojection error, straightness residual, model-order study
Step 7  Tool             Image in → corrected image + parameters out
```

---

## Key Concepts

**Reprojection error** — mean pixel distance between detected chessboard corners and where the fitted model says they should be. The optimiser minimises this directly.

**Straightness residual** — after correction, points on a real straight edge should be collinear. We measure mean orthogonal distance to the best-fit line.

**Model order** — more coefficients are not always better. k3 on a mild lens can overfit near the corners. We report results for `[k1]`, `[k1,k2]`, and `[k1,k2,k3]`.

**Direction problem** — the Brown–Conrady polynomial maps ideal points to distorted points. Undistortion needs the reverse; since there is no closed form, we invert iteratively (Newton-style fixed-point iteration, converges in <20 steps).

---

## Viva Quick Reference

| Question | Answer |
|---|---|
| What is r in the model? | Normalised radius: r² = x² + y² (in normalised image coords) |
| What does radial distortion look like? | Barrel (lines bend inward) or pincushion (lines bend outward) |
| Why is undistortion harder? | The polynomial has no closed-form inverse → iterative solver needed |
| Why reprojection error? | Same quantity the optimiser minimises; cheap, pattern-independent |
| Why target-free? | Works on any photo with long straight edges; no calibration object |
| Why not deep learning? | Transparent, inspectable, no training data needed |

---

## Milestones

| Stage | Focus | Status |
|---|---|---|
| Milestone 1 | Topic, literature review, problem definition | ✅ Submitted 30 Sept |
| Milestone 2 | Forward/inverse model, chessboard calibration, OpenCV comparison | 🔄 Upcoming |
| Final | Target-free estimator, model-order study, demo tool, final report | 🔄 Upcoming |

---

## References

- Brown, D.C. (1966). *Decentering distortion of lenses.* Photometric Engineering, 32(3), 444–462.
- Zhang, Z. (2000). *A flexible new technique for camera calibration.* IEEE TPAMI, 22(11), 1330–1334.
- OpenCV documentation: `cv2.calibrateCamera`, `cv2.undistort`
