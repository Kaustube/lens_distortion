"use client";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (val: number) => void;
  description?: string;
}

export default function CoefficientSlider({
  label, value, min, max, step, onChange, description,
}: Props) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="group">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-indigo-300 w-6">{label}</span>
          {description && (
            <span className="text-xs text-white/30 hidden group-hover:inline transition-all">
              — {description}
            </span>
          )}
        </div>
        <input
          type="number"
          value={value}
          step={step}
          min={min}
          max={max}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          className="w-24 text-right text-xs font-mono bg-white/5 border border-white/10
            rounded px-2 py-1 text-white/80 focus:outline-none focus:border-indigo-500"
        />
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full"
          style={{
            background: `linear-gradient(to right, #6366f1 ${pct}%, #2a2a3a ${pct}%)`,
          }}
        />
      </div>
    </div>
  );
}
