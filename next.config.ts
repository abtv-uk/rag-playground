import type { NextConfig } from "next";

// GITHUB_PAGES is set by the deploy workflow: static export served from
// https://<user>.github.io/rag-playground/ needs the basePath prefix.
const isPages = process.env.GITHUB_PAGES === "true";

const nextConfig: NextConfig = {
  output: "export",
  basePath: isPages ? "/rag-playground" : "",
  images: { unoptimized: true },
};

export default nextConfig;
