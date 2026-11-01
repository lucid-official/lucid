import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test, expect, type Page } from "@playwright/test";

/**
 * Runtime smoke: load representative pages and assert they hydrate without
 * uncaught exceptions or console errors, then exercise the scroll-spy ToC.
 * These are failure modes the static-HTML contract audit cannot observe.
 */

// Console noise that isn't a real defect (dev-only fast-refresh / resource
// 404s for assets not served by `next dev`).  Keep this list tight.
const BENIGN = [
  /Download the React DevTools/i,
  /\[Fast Refresh\]/i,
  /favicon/i,
];

function attachErrorCollectors(page: Page): { errors: string[] } {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const text = msg.text();
    if (BENIGN.some((re) => re.test(text))) return;
    errors.push(`console.error: ${text}`);
  });
  return { errors };
}

const PAGES = [
  { path: "/", name: "home" },
  { path: "/api", name: "api landing" },
  { path: "/api/lucid.nn", name: "module overview" },
  { path: "/api/lucid.models.vision.resnet/resnet_50", name: "model member" },
  { path: "/api/lucid.peft", name: "peft package" },
  { path: "/api/lucid.peft.tuners", name: "peft tuners overview" },
  { path: "/api/lucid.peft.tuners.lora", name: "peft tuner" },
  { path: "/api/lucid.peft.tuners.lora/LoraConfig", name: "peft tuner member" },
  { path: "/architecture", name: "architecture landing" },
  { path: "/architecture/layer-map", name: "architecture diagram" },
];

for (const { path, name } of PAGES) {
  test(`${name} hydrates without errors`, async ({ page }) => {
    const { errors } = attachErrorCollectors(page);
    await page.goto(path, { waitUntil: "networkidle" });
    await expect(page.locator("#main-content, main").first()).toBeVisible();
    expect(errors, errors.join("\n")).toEqual([]);
  });
}

test("model member page renders the Model Size card", async ({ page }) => {
  await page.goto("/api/lucid.models.vision.resnet/resnet_50", {
    waitUntil: "networkidle",
  });
  await expect(page.getByText("Model Size", { exact: false })).toBeVisible();
});

// lucid.peft is documented as a package section (web/data/api-sections.json):
// the sidebar groups the 22 tuners by family, each tuner page holds its config,
// model and layers, and the generator fixes of LCD-725 show on the real pages.
test.describe("lucid.peft reference", () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
  });

  test("the sidebar groups the tuners by family, with one Tuners entry", async ({ page }) => {
    await page.goto("/api/lucid.peft.tuners.lora/LoraConfig", { waitUntil: "networkidle" });
    const sidebar = page.locator('aside[aria-label="Documentation navigation"]');
    for (const group of [
      "Core",
      "LoRA family",
      "Orthogonal family",
      "LyCORIS family",
      "Prompt learning",
      "Shared-table tuners",
      "Structured and sparse updates",
      "Scaling and selective tuning",
      "Utilities",
    ]) {
      await expect(sidebar.getByText(group, { exact: true }), group).toBeVisible();
    }
    // The overview is one link; a "Tuners" source-folder group beside it was
    // the duplicate entry of the unfixed sidebar.
    await expect(sidebar.getByText("All Tuners", { exact: true })).toBeVisible();
    await expect(sidebar.getByText("Tuners", { exact: true })).toHaveCount(0);
    // The active path is open down to the page: PEFT > LoRA family > LoRA > LoraConfig.
    await expect(sidebar.locator('a[aria-current="page"][href$="/LoraConfig"]')).toBeVisible();
  });

  test("a tuner page lists its config, model and layer classes", async ({ page }) => {
    await page.goto("/api/lucid.peft.tuners.lora", { waitUntil: "networkidle" });
    const main = page.locator("#main-content");
    for (const name of ["LoraConfig", "LoraModel", "LoraLayer", "MultiheadAttention"]) {
      await expect(main.getByText(name, { exact: true }).first(), name).toBeVisible();
    }
  });

  test("members inherited from private bases and documented attributes are listed", async ({
    page,
  }) => {
    await page.goto("/api/lucid.peft.tuners.lora/LoraConfig", { waitUntil: "networkidle" });
    const main = page.locator("#main-content");
    for (const name of ["to_dict", "from_dict", "save_pretrained"]) {
      await expect(main.locator(`h3#${name}`), name).toHaveCount(1);
    }
    await page.goto("/api/lucid.peft.tuners.lora/LoraLayer", { waitUntil: "networkidle" });
    for (const name of ["merged", "disable_adapters", "active_adapter"]) {
      await expect(page.locator(`#main-content h3#${name}`), name).toHaveCount(1);
    }
  });

  test("a reST literal block renders as code, not as a paragraph", async ({ page }) => {
    await page.goto("/api/lucid.peft", { waitUntil: "networkidle" });
    const code = page.locator("#main-content pre", { hasText: "get_peft_model(base_model, config)" });
    await expect(code.first()).toBeVisible();
    // Line breaks survive: the example is several lines, not one run-on.
    expect((await code.first().innerText()).split("\n").length).toBeGreaterThan(4);
  });
});

test("ToC entry click scrolls + updates the hash", async ({ page }) => {
  await page.goto("/api/lucid.nn", { waitUntil: "networkidle" });
  // The "On this page" ToC is an <aside> of in-page anchor links (xl only).
  await page.setViewportSize({ width: 1440, height: 900 });
  const tocLinks = page.locator('aside a[href*="#"]');
  const count = await tocLinks.count();
  test.skip(count === 0, "no ToC on this page");
  await tocLinks.first().click();
  await expect(page).toHaveURL(/#.+/);
});

// The package imports as ``lucid`` but installs as ``lucid-dl``; a copied
// ``pip install lucid`` fetches an unrelated project.  Read the name straight
// from pyproject.toml here, independently of src/lib/distribution.ts.
test("install commands name the PyPI distribution from pyproject.toml", async ({ page }) => {
  const toml = readFileSync(join(process.cwd(), "..", "pyproject.toml"), "utf8");
  const dist = toml.split(/^\[project\]\s*$/m)[1]?.match(/^name\s*=\s*"([^"]+)"/m)?.[1];
  expect(dist, "[project].name in ../pyproject.toml").toBeTruthy();

  await page.goto("/", { waitUntil: "networkidle" });
  await expect(page.locator("[data-install-command]")).toHaveText(`pip install ${dist}`);

  // Every Lucid install line in the guide, extras and flags included.
  await page.goto("/docs/installation", { waitUntil: "networkidle" });
  const text = await page.locator("#main-content").innerText();
  const targets = [...text.matchAll(/pip install(?:\s+--?\w[\w-]*)*\s+"?(lucid[\w.-]*)/g)].map(
    (m) => m[1],
  );
  expect(targets.length).toBeGreaterThan(0);
  for (const name of targets) expect(name).toBe(dist);
});

// Brand kit wiring (scripts/sync-brand.mjs → public/brand/, linked by the root
// layout, the manifest and BrandLockup).  A kit file that didn't sync, or a
// link that lost its path, only shows up as a broken image or a missing icon.
test("every head icon and the manifest resolve", async ({ page, request }) => {
  await page.goto("/", { waitUntil: "networkidle" });
  const hrefs = await page
    .locator('link[rel~="icon"], link[rel="apple-touch-icon"], link[rel="manifest"]')
    .evaluateAll((links) => links.map((l) => (l as HTMLLinkElement).href));
  expect(hrefs.length).toBeGreaterThanOrEqual(4);
  const manifestHref = hrefs.find((h) => h.includes("manifest"));
  expect(manifestHref).toBeDefined();
  const manifest = await (await request.get(manifestHref!)).json();
  const icons = (manifest.icons as { src: string }[]).map((i) => new URL(i.src, page.url()).href);
  for (const href of [...hrefs, ...icons]) {
    expect((await request.get(href)).ok(), href).toBe(true);
  }
});

test("header lockup loads and follows the site's theme toggle", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/", { waitUntil: "networkidle" });
  const header = page.locator("header").first();
  const onDark = header.locator('img[data-theme-only="dark"]');
  const onLight = header.locator('img[data-theme-only="light"]');
  const loaded = (img: typeof onDark) =>
    img.evaluate((el) => (el as HTMLImageElement).naturalWidth);

  await expect(onDark).toBeVisible();
  await expect(onLight).toBeHidden();
  await expect.poll(() => loaded(onDark)).toBeGreaterThan(0);

  // The site's own toggle, not the OS preference, picks the variant.
  await page.getByRole("button", { name: "Switch to light mode" }).click();
  await expect(onLight).toBeVisible();
  await expect(onDark).toBeHidden();
  await expect.poll(() => loaded(onLight)).toBeGreaterThan(0);
});
