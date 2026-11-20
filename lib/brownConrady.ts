/**
 * brownConrady.ts
 * ---------------
 * Brown–Conrady lens distortion model in pure TypeScript.
 * Mirrors the Python implementation in distortion_model.py.
 *
 * Five coefficients: k1,k2,k3 (radial) + p1,p2 (tangential)
 * Plus intrinsics:   fx, fy, cx, cy
 */

export interface BrownConradyParams {
  fx: number; fy: number;
  cx: number; cy: number;
  k1: number; k2: number; k3: number;
  p1: number; p2: number;
}

/** Iteratively invert Brown–Conrady: distorted normalised → undistorted normalised */
function undistortNorm(
  xd: number, yd: number,
  p: BrownConradyParams,
  maxIter = 20,
  tol = 1e-9,
): [number, number] {
  let xu = xd, yu = yd;
  for (let i = 0; i < maxIter; i++) {
    const r2  = xu * xu + yu * yu;
    const r4  = r2 * r2;
    const r6  = r4 * r2;
    const rad = 1 + p.k1 * r2 + p.k2 * r4 + p.k3 * r6;
    const dx  = 2 * p.p1 * xu * yu + p.p2 * (r2 + 2 * xu * xu);
    const dy  = p.p1 * (r2 + 2 * yu * yu) + 2 * p.p2 * xu * yu;
    const xn  = (xd - dx) / rad;
    const yn  = (yd - dy) / rad;
    if (Math.abs(xn - xu) + Math.abs(yn - yu) < tol) { xu = xn; yu = yn; break; }
    xu = xn; yu = yn;
  }
  return [xu, yu];
}

/** Bilinear sample from a flat RGBA pixel buffer */
function sampleBilinear(
  data: Uint8ClampedArray,
  w: number, h: number,
  sx: number, sy: number,
): [number, number, number, number] {
  if (sx < 0 || sx >= w - 1 || sy < 0 || sy >= h - 1) return [0, 0, 0, 0];
  const x0 = Math.floor(sx), y0 = Math.floor(sy);
  const x1 = x0 + 1,         y1 = y0 + 1;
  const wx = sx - x0,         wy = sy - y0;

  const idx = (r: number, c: number) => (r * w + c) * 4;
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

  const res: [number, number, number, number] = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    res[c] = lerp(
      lerp(data[idx(y0, x0) + c], data[idx(y0, x1) + c], wx),
      lerp(data[idx(y1, x0) + c], data[idx(y1, x1) + c], wx),
      wy,
    );
  }
  return res;
}

/**
 * Undistort an image given Brown–Conrady params.
 * Input/output: flat Uint8ClampedArray in RGBA order (same as canvas ImageData).
 */
export function undistortImage(
  src: Uint8ClampedArray,
  width: number,
  height: number,
  p: BrownConradyParams,
): { dst: Uint8ClampedArray; metrics: Record<string, number> } {
  const dst = new Uint8ClampedArray(src.length);

  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      // Destination pixel → normalised coords
      const xn = (px - p.cx) / p.fx;
      const yn = (py - p.cy) / p.fy;

      // Invert: find where this came from in the distorted image
      const [xu, yu] = undistortNorm(xn, yn, p);

      // Back to pixel coords
      const sx = xu * p.fx + p.cx;
      const sy = yu * p.fy + p.cy;

      const [r, g, b, a] = sampleBilinear(src, width, height, sx, sy);
      const i = (py * width + px) * 4;
      dst[i] = r; dst[i + 1] = g; dst[i + 2] = b; dst[i + 3] = a || 255;
    }
  }

  // Compute a simple "straightness" proxy: corner displacement magnitude
  const cornerDisplace = (() => {
    const corners = [
      [0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1],
    ];
    let total = 0;
    for (const [cx2, cy2] of corners) {
      const xn = (cx2 - p.cx) / p.fx;
      const yn = (cy2 - p.cy) / p.fy;
      const [xu, yu] = undistortNorm(xn, yn, p);
      const sx = xu * p.fx + p.cx;
      const sy = yu * p.fy + p.cy;
      total += Math.hypot(sx - cx2, sy - cy2);
    }
    return total / 4;
  })();

  return {
    dst,
    metrics: {
      "Corner shift (px)": Math.round(cornerDisplace * 100) / 100,
      "k1·r²_max":         Math.round(p.k1 * 100) / 100,
      "Width":             width,
      "Height":            height,
    },
  };
}
