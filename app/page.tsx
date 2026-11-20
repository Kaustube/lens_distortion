"use client";

import { useState, useRef, useCallback } from "react";
import CoefficientSlider from "@/components/CoefficientSlider";
import ImageCompare from "@/components/ImageCompare";
import ResultPanel from "@/components/ResultPanel";
import HeroSection from "@/components/HeroSection";
import { undistortImage, BrownConradyParams } from "@/lib/brownConrady";

export interface Coefficients extends BrownConradyParams {}

const DEFAULT_COEFS: Coefficients = {
  k1: -0.35, k2: 0.12, k3: 0.0,
  p1: 0.001,  p2: -0.0005,
  fx: 800,    fy: 800,
  cx: 320,    cy: 240,
};

/** Process image entirely in the browser using Canvas API */
async function processClientSide(
  dataUrl: string,
  p: BrownConradyParams,
): Promise<{ correctedDataUrl: string; metrics: Record<string, number> }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width  = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const raw = ctx.getImageData(0, 0, img.width, img.height);

      const { dst, metrics } = undistortImage(
        raw.data, img.width, img.height, p,
      );

      // Write corrected pixels
      const outCanvas = document.createElement("canvas");
      outCanvas.width  = img.width;
      outCanvas.height = img.height;
      const outCtx = outCanvas.getContext("2d")!;
      const outData = outCtx.createImageData(img.width, img.height);
      outData.data.set(dst);
      outCtx.putImageData(outData, 0, 0);

      resolve({
        correctedDataUrl: outCanvas.toDataURL("image/png"),
        metrics,
      });
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

export default function Home() {
  const [file, setFile]           = useState<File | null>(null);
  const [preview, setPreview]     = useState<string | null>(null);
  const [corrected, setCorrected] = useState<string | null>(null);
  const [coefs, setCoefs]         = useState<Coefficients>(DEFAULT_COEFS);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [metrics, setMetrics]     = useState<Record<string, number> | null>(null);
  const [dragging, setDragging]   = useState(false);
  const fileInputRef              = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (jpg, png, webp).");
      return;
    }
    setFile(f);
    setError(null);
    setCorrected(null);
    setMetrics(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const onCorrect = async () => {
    if (!file || !preview) return;
    setLoading(true);
    setError(null);
    setCorrected(null);

    try {
      // Always process client-side first (instant, no server needed)
      const result = await processClientSide(preview, coefs);
      setCorrected(result.correctedDataUrl);
      setMetrics(result.metrics);
    } catch (err: unknown) {
      // Fallback to server API (for environments without Canvas)
      try {
        const form = new FormData();
        form.append("image", file);
        form.append("coefficients", JSON.stringify(coefs));
        const res  = await fetch("/api/correct", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Correction failed");
        setCorrected(data.correctedDataUrl);
        setMetrics(data.metrics ?? null);
      } catch (err2: unknown) {
        setError(err2 instanceof Error ? err2.message : "Correction failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const setCoef = (key: keyof Coefficients) => (val: number) =>
    setCoefs((prev) => ({ ...prev, [key]: val }));

  return (
    <div className="min-h-screen flex flex-col">
      {/* Nav */}
      <nav className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center text-sm font-bold">L</div>
          <span className="font-semibold text-white">LensCorrect</span>
          <span className="hidden sm:block text-xs text-white/40 ml-2">Brown–Conrady Distortion Demo</span>
        </div>
        <div className="flex items-center gap-4 text-sm text-white/50">
          <a href="https://github.com/Kaustube/lens_distortion" target="_blank"
             className="hover:text-white transition-colors">GitHub</a>
          <span className="text-white/20">|</span>
          <span className="hidden sm:inline">Team KASAT · CSET381</span>
        </div>
      </nav>

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-10 grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-8">

        {/* LEFT COLUMN */}
        <div className="flex flex-col gap-6">
          <HeroSection />

          {/* Upload zone */}
          <div
            className={`relative rounded-2xl border-2 border-dashed transition-all cursor-pointer
              ${dragging ? "border-indigo-400 bg-indigo-500/10" : "border-white/20 hover:border-white/40"}
              ${preview ? "p-2" : "p-12 flex flex-col items-center justify-center gap-3"}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Input" className="w-full max-h-[380px] object-contain rounded-xl" />
            ) : (
              <>
                <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center text-2xl">📷</div>
                <p className="text-white/70 font-medium">Drop an image or click to upload</p>
                <p className="text-white/30 text-sm">JPG, PNG, WEBP — processed in your browser</p>
              </>
            )}
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
          </div>

          {/* Corrected result */}
          {corrected && preview && (
            <ImageCompare original={preview} corrected={corrected} />
          )}

          {error && (
            <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 text-red-400 text-sm">
              ⚠ {error}
            </div>
          )}
        </div>

        {/* RIGHT COLUMN — controls */}
        <div className="flex flex-col gap-5">
          <div className="rounded-2xl bg-white/5 border border-white/10 p-5">
            <h2 className="text-sm font-semibold text-white/60 uppercase tracking-widest mb-4">
              Distortion Coefficients
            </h2>
            <div className="space-y-4">
              <p className="text-xs text-white/40 mb-2">Radial</p>
              <CoefficientSlider label="k₁" value={coefs.k1} min={-1}    max={1}    step={0.001}  onChange={setCoef("k1")} description="Primary radial (barrel/pincushion)" />
              <CoefficientSlider label="k₂" value={coefs.k2} min={-1}    max={1}    step={0.001}  onChange={setCoef("k2")} description="Secondary radial" />
              <CoefficientSlider label="k₃" value={coefs.k3} min={-1}    max={1}    step={0.001}  onChange={setCoef("k3")} description="Tertiary radial (fix at 0 for mild lenses)" />
              <p className="text-xs text-white/40 mt-4 mb-2">Tangential</p>
              <CoefficientSlider label="p₁" value={coefs.p1} min={-0.1}  max={0.1}  step={0.0001} onChange={setCoef("p1")} description="Tangential — y-axis tilt" />
              <CoefficientSlider label="p₂" value={coefs.p2} min={-0.1}  max={0.1}  step={0.0001} onChange={setCoef("p2")} description="Tangential — x-axis tilt" />
              <p className="text-xs text-white/40 mt-4 mb-2">Intrinsics</p>
              <CoefficientSlider label="fx"  value={coefs.fx} min={200}  max={2000} step={1}      onChange={setCoef("fx")} description="Focal length x (px)" />
              <CoefficientSlider label="fy"  value={coefs.fy} min={200}  max={2000} step={1}      onChange={setCoef("fy")} description="Focal length y (px)" />
              <CoefficientSlider label="cx"  value={coefs.cx} min={0}    max={2000} step={1}      onChange={setCoef("cx")} description="Principal point x" />
              <CoefficientSlider label="cy"  value={coefs.cy} min={0}    max={2000} step={1}      onChange={setCoef("cy")} description="Principal point y" />
            </div>
            <button
              onClick={() => setCoefs(DEFAULT_COEFS)}
              className="mt-4 w-full text-xs text-white/30 hover:text-white/60 transition-colors"
            >
              Reset to defaults
            </button>
          </div>

          {/* Action button */}
          <button
            onClick={onCorrect}
            disabled={!file || loading}
            className="w-full py-4 rounded-2xl font-semibold text-white
              bg-indigo-500 hover:bg-indigo-400 disabled:opacity-40 disabled:cursor-not-allowed
              transition-all active:scale-[0.98] flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Processing…
              </>
            ) : "✦ Correct Image"}
          </button>

          {corrected && <ResultPanel corrected={corrected} metrics={metrics} />}

          {/* Theory card */}
          <div className="rounded-2xl bg-white/5 border border-white/10 p-5 text-sm text-white/50 space-y-2">
            <p className="text-white/80 font-medium text-xs uppercase tracking-widest">How it works</p>
            <p>The Brown–Conrady model maps each ideal pixel through a radial polynomial <span className="text-indigo-300">(1 + k₁r² + k₂r⁴ + k₃r⁶)</span> and tangential offsets.</p>
            <p>Undistortion inverts this iteratively — no closed form exists. Each destination pixel traces back to its source in the distorted image via bilinear interpolation.</p>
            <p className="text-xs text-white/30">Processing runs entirely in your browser — no image is uploaded to any server.</p>
            <a href="https://github.com/Kaustube/lens_distortion" target="_blank"
               className="block mt-2 text-indigo-400 hover:text-indigo-300 transition-colors">
              Full Python source on GitHub →
            </a>
          </div>
        </div>
      </main>

      <footer className="border-t border-white/10 py-5 text-center text-xs text-white/30">
        Team KASAT — Kaustubh · Aryan · Sukhbeer · Yashwanth · Ashish &nbsp;·&nbsp; CSET381 Applications of AI · Sem 5
      </footer>
    </div>
  );
}
