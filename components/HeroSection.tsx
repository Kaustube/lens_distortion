export default function HeroSection() {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2 py-1 rounded-full">
          CSET381 · Team KASAT
        </span>
        <span className="text-xs text-white/30">Milestone 2 Demo</span>
      </div>
      <h1 className="text-3xl font-bold tracking-tight text-white">
        Lens Distortion{" "}
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">
          Corrector
        </span>
      </h1>
      <p className="text-white/50 text-sm max-w-xl">
        Upload a distorted photo, dial in the five Brown–Conrady coefficients,
        and get a corrected image instantly — computed entirely in your browser
        via our pure-JavaScript implementation.
      </p>
    </div>
  );
}
