import type { NextConfig } from "next";

// GITHUB_PAGES is set by the deploy workflow: static export served from
// https://<user>.github.io/rag-playground/ needs the basePath prefix.
const isPages = process.env.GITHUB_PAGES === "true";
const basePath = isPages ? "/rag-playground" : "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  images: { unoptimized: true },
  // plain asset URLs (sample PDF / chunk JSON) don't get basePath rewriting,
  // so expose it for manual prefixing
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
