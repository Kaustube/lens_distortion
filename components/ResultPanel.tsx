"use client";

interface Props {
  corrected: string;
  metrics: Record<string, number> | null;
}

export default function ResultPanel({ corrected, metrics }: Props) {
  const download = () => {
    const a = document.createElement("a");
    a.href = corrected;
    a.download = "corrected_lens.png";
    a.click();
  };

  return (
    <div className="rounded-2xl bg-white/5 border border-white/10 p-5 space-y-4">
      <p className="text-xs font-semibold text-white/60 uppercase tracking-widest">Result</p>

      {metrics && (
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(metrics).map(([key, val]) => (
            <div key={key} className="rounded-xl bg-white/5 p-3">
              <p className="text-xs text-white/40 mb-1">{key}</p>
              <p className="text-sm font-mono text-indigo-300">{typeof val === "number" ? val.toFixed(4) : val}</p>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={download}
        className="w-full py-3 rounded-xl bg-indigo-500/20 hover:bg-indigo-500/30
          border border-indigo-500/40 text-indigo-300 text-sm font-medium
          transition-all flex items-center justify-center gap-2"
      >
        ↓ Download corrected image
      </button>
    </div>
  );
}
