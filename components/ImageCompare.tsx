"use client";

import { useState, useRef, useCallback } from "react";

interface Props {
  original: string;
  corrected: string;
}

export default function ImageCompare({ original, corrected }: Props) {
  const [split, setSplit] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const onMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct  = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSplit(pct);
  }, []);

  return (
    <div className="rounded-2xl overflow-hidden border border-white/10">
      <div className="flex items-center justify-between px-4 py-2 bg-white/5 text-xs text-white/40">
        <span>← Original</span>
        <span className="text-white/20">drag to compare</span>
        <span>Corrected →</span>
      </div>

      <div
        ref={containerRef}
        className="relative select-none cursor-col-resize"
        style={{ userSelect: "none" }}
        onMouseDown={() => { dragging.current = true; }}
        onMouseMove={(e) => { if (dragging.current) onMove(e.clientX); }}
        onMouseUp={() => { dragging.current = false; }}
        onMouseLeave={() => { dragging.current = false; }}
        onTouchMove={(e) => onMove(e.touches[0].clientX)}
      >
        {/* Corrected (full width, behind) */}
        <img src={corrected} alt="Corrected" className="w-full block max-h-[400px] object-contain bg-black" />

        {/* Original (clipped left side) */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ width: `${split}%` }}
        >
          <img src={original} alt="Original"
            className="w-full block max-h-[400px] object-contain bg-black"
            style={{ minWidth: `${100 / (split / 100)}%` }}
          />
        </div>

        {/* Divider */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-indigo-400 shadow-[0_0_8px_#6366f1]"
          style={{ left: `${split}%`, transform: "translateX(-50%)" }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
            w-7 h-7 rounded-full bg-indigo-500 border-2 border-white shadow-lg
            flex items-center justify-center text-xs text-white font-bold">
            ⇔
          </div>
        </div>
      </div>
    </div>
  );
}
