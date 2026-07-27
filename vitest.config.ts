// Vitest exists here for one load-bearing reason beyond running tests:
// lib/*.ts uses extensionless relative imports ("./constants"), which plain
// `node --test` cannot resolve even with type stripping — earlier
// measurement scripts had to sed-rewrite imports into throwaway copies to
// run at all. Vitest resolves them natively, so the oracles in tests/ can
// import the real modules.
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // Defensive: no test needs the "@/" alias today (lib modules import
    // each other relatively), but hooks/ uses it — this keeps a future
    // hook-level test from failing mysteriously.
    alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
