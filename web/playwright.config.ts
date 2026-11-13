import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright tests for the docs site, against two builds of it.
 *
 * - ``chromium`` (``e2e/smoke.spec.ts``): runtime coverage the static contract
 *   audit (scripts/audit-docs.mjs) can't see - client hydration, console / page
 *   errors, interactive behavior (the scroll-spy ToC).  Runs against
 *   ``pnpm dev:fast`` (``SKIP_API_BUILD=1`` so it needs neither Griffe nor a
 *   built Lucid); dev mode leaves basePath empty (it's prod-only), so URLs are
 *   plain.
 * - ``export`` (``e2e/export.spec.ts``): what a visitor's browser meets - the
 *   static export (``out/``, ``pnpm build`` first) served under ``/lucid`` by
 *   ``e2e/serve-export.mjs``.  A ``#fragment`` link has to land there, with
 *   the real fonts, the real streaming shape and no dev server's timing
 *   (LCD-1005); a page has to be readable without JavaScript.
 *
 * ``E2E_TARGET=dev|export`` runs one of them (``pnpm e2e:dev``,
 * ``pnpm e2e:export``); the default runs both.
 */
const target = process.env.E2E_TARGET ?? "all";
const wantsDev = target !== "export";
const wantsExport = target !== "dev";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: { trace: "on-first-retry" },
  projects: [
    ...(wantsDev
      ? [
          {
            name: "chromium",
            testMatch: "smoke.spec.ts",
            use: { ...devices["Desktop Chrome"], baseURL: "http://localhost:3000" },
          },
        ]
      : []),
    ...(wantsExport
      ? [
          {
            name: "export",
            testMatch: "export.spec.ts",
            use: { ...devices["Desktop Chrome"], baseURL: "http://localhost:3100" },
          },
        ]
      : []),
  ],
  webServer: [
    ...(wantsDev
      ? [
          {
            command: "pnpm dev:fast",
            url: "http://localhost:3000",
            reuseExistingServer: !process.env.CI,
            timeout: 180_000,
          },
        ]
      : []),
    ...(wantsExport
      ? [
          {
            command: "node e2e/serve-export.mjs 3100",
            url: "http://localhost:3100/lucid/",
            reuseExistingServer: !process.env.CI,
            timeout: 30_000,
          },
        ]
      : []),
  ],
});
