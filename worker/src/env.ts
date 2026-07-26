// Hand-declared rather than pulled from @cloudflare/workers-types, whose
// exact global name for the rate-limit binding varies by package version.
export interface RateLimiterBinding {
  limit(options: { key: string }): Promise<{ success: boolean }>;
}

export interface Env {
  AI: Ai;
  QUOTA: KVNamespace;
  RATE: KVNamespace;
  GENERATE_LIMITER: RateLimiterBinding;
  // secrets — set via `wrangler secret put`, read locally from .dev.vars
  GEMINI_API_KEY: string;
  IP_HASH_SALT: string;
  // vars
  ALLOWED_ORIGINS: string;
}
