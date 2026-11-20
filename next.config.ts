import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Use webpack (stable) instead of Turbopack to avoid native-addon issues
  turbopack: undefined,
};

export default nextConfig;
