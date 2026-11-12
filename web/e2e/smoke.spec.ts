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

/** Open ``path`` and fail if the site answered with its not-found page.  Under
 *  ``next dev`` a missing page is still HTTP 200 (the 404 is streamed in), so
 *  the status code cannot tell; the not-found page is the one marked
 *  ``noindex``.  Without this, a test on a path the static export does not
 *  generate (``/api/lucid.tensor/Tensor`` — ``Tensor`` is the class page
 *  ``/api/lucid.tensor``, not one of its methods) passes by inspecting a 404,
 *  and ``next dev`` logs "Failed to generate static paths for /api/[...slug]". */
async function visit(page: Page, path: string): Promise<void> {
  await page.goto(path, { waitUntil: "networkidle" });
  await expect(
    page.locator('meta[name="robots"][content="noindex"]'),
    `${path} is the 404 page — the static export has no such page`,
  ).toHaveCount(0);
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
    await visit(page, path);
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
// the sidebar groups the tuners by family, each tuner page holds its config,
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

  // The overview page of a sectioned package (api-sections.json "overview")
  // defines classes of its own — ``MixedModel`` on /api/lucid.peft.tuners —
  // that no section of the package claims.  Each one needs a card on the page
  // and a leaf under the sidebar's "All …" link, or it is reachable from
  // neither.
  test("an overview page lists its own classes and gives each a sidebar leaf", async ({ page }) => {
    const sections = JSON.parse(
      readFileSync(join(process.cwd(), "data", "api-sections.json"), "utf8"),
    ) as Record<string, { overview?: string }>;
    const overviews = Object.values(sections)
      .map((plan) => plan.overview)
      .filter((slug): slug is string => Boolean(slug));
    expect(overviews.length).toBeGreaterThan(0);

    for (const slug of overviews) {
      const data = JSON.parse(
        readFileSync(join(process.cwd(), "public", "api-data", `${slug}.json`), "utf8"),
      ) as { members: { name: string; kind: string }[] };
      const own = data.members.filter((m) => m.kind === "class" || m.kind === "function");
      await page.goto(`/api/${slug}`, { waitUntil: "networkidle" });
      const sidebar = page.locator('aside[aria-label="Documentation navigation"]');
      for (const { name } of own) {
        const href = `/api/${slug}/${name}`;
        await expect(page.locator(`#main-content a[href$="${href}"]`), `${name} card`).toHaveCount(1);
        await expect(sidebar.locator(`a[href$="${href}"]`), `${name} sidebar leaf`).toBeVisible();
      }
    }
    // MixedModel is the reason the test exists; name it in case the data moves.
    await expect(page.locator('#main-content a[href$="/lucid.peft.tuners/MixedModel"]')).toHaveCount(1);
  });

  // A module page puts its constants under one "Attributes" section title; the
  // table beneath it used to repeat the word.  And a name wider than the name
  // column (``TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING``) was
  // drawn over its own description at 1440 px and clipped at 390 px.
  test("the attributes table sits under one heading", async ({ page }) => {
    await page.goto("/api/lucid.peft", { waitUntil: "networkidle" });
    await expect(
      page.locator("section:has(> h2#section-attributes)").getByText("Attributes", { exact: true }),
    ).toHaveCount(1);
  });

  for (const width of [1440, 390]) {
    test(`no attribute name spills out of its column at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.goto("/api/lucid.peft", { waitUntil: "networkidle" });
      const section = page.locator("section:has(> h2#section-attributes)");

      const spills = await section.evaluate((el) =>
        [...el.querySelectorAll(".divide-y > div")].flatMap((row) => {
          const code = row.querySelector("code");
          const column = code?.parentElement;
          if (!code || !column) return [];
          // The text's own extent: an inline box reports its column's width.
          const range = document.createRange();
          range.selectNodeContents(code);
          const text = range.getBoundingClientRect();
          const col = column.getBoundingClientRect();
          return text.right > col.right + 0.5 ? [`${code.textContent}: ${text.right} > ${col.right}`] : [];
        }),
      );
      expect(spills, spills.join("\n")).toEqual([]);
    });
  }

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

// A section's one anchor is its heading.  Every section of a package page used
// to put the same id on the ``<section>`` and on its own ``<h2>``, which left
// ``#fragment`` navigation, ``getElementById`` and the ToC to choose between
// two elements.  These pages cover each place an id is minted: a module's
// member sections, the titled sub-package sections, the family groups of the
// model zoo, a class page's collapsible method groups, and a model member.
const ID_PAGES = [
  "/api",
  "/api/lucid.nn",
  "/api/lucid.peft",
  "/api/lucid.peft.tuners",
  "/api/lucid.models",
  "/api/lucid.nn/Linear",
  "/api/lucid.tensor",
  "/api/lucid.optim/Adam",
  "/api/lucid.peft.tuners.lora/LoraConfig",
  "/api/lucid.models.vision.resnet/resnet_50",
];

for (const path of ID_PAGES) {
  test(`no id is repeated on ${path}`, async ({ page }) => {
    await visit(page, path);
    const repeated = await page.evaluate(() => {
      const count = new Map<string, number>();
      for (const el of document.querySelectorAll("[id]")) {
        count.set(el.id, (count.get(el.id) ?? 0) + 1);
      }
      return [...count].filter(([, n]) => n > 1).map(([id, n]) => `${id} x${n}`);
    });
    expect(repeated, repeated.join(", ")).toEqual([]);
  });
}

test("a section anchor is its heading, and a fragment lands it below the header", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/api/lucid.peft", { waitUntil: "networkidle" });
  const anchor = page.locator("#section-attributes");
  await expect(anchor).toHaveCount(1);
  expect(await anchor.evaluate((el) => el.tagName)).toBe("H2");
  // The browser's own fragment navigation (what a sidebar or tutorial link does).
  // ``scroll-mt-24`` = 96 px: the heading stops under the fixed header, not behind it.
  await page.evaluate(() => {
    window.location.hash = "section-attributes";
  });
  await expect
    .poll(async () => Math.abs((await anchor.boundingBox())!.y - 96))
    .toBeLessThanOrEqual(2);

  // A class page's method group: its ToC link scrolls to the one element carrying the id.
  await page.goto("/api/lucid.nn/Linear", { waitUntil: "networkidle" });
  const group = page.locator("article h2[id]").first();
  const id = await group.getAttribute("id");
  await expect(page.locator(`[id="${id}"]`)).toHaveCount(1);
  await page.locator(`aside a[href="#${id}"]`).first().click();
  await expect.poll(async () => Math.round((await group.boundingBox())!.y)).toBeLessThanOrEqual(100);
});

// A display formula wider than the column scrolls inside its own box.  Left to
// KaTeX's single nowrap line it ran out of the column and widened the whole
// page: 416 px on the LoRA tutorial at a 390 px viewport, so the page itself
// scrolled sideways.  The pages are the formula-heavy ones of /docs plus API
// pages whose docstrings carry display math.
const FORMULA_PAGES = [
  "/docs/tutorials/lora-fine-tuning",
  "/docs/special",
  "/docs/signal",
  "/docs/distributions",
  "/docs/fft",
  "/docs/linalg",
  "/api/lucid.distributions/Normal",
  "/api/lucid.nn/TransformerDecoderLayer",
];

test.describe("display math at a phone width", () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 800 });
  });

  for (const path of FORMULA_PAGES) {
    test(`${path} is not wider than the viewport`, async ({ page }) => {
      await visit(page, path);
      expect(await page.locator(".katex-display").count(), "page has display math").toBeGreaterThan(0);
      const { viewport, document: width } = await page.evaluate(() => ({
        viewport: window.innerWidth,
        document: document.documentElement.scrollWidth,
      }));
      expect(width, `document ${width}px in a ${viewport}px viewport`).toBeLessThanOrEqual(viewport);
    });
  }

  test("a formula wider than its column scrolls in its own box; inline math is untouched", async ({
    page,
  }) => {
    await page.goto("/docs/tutorials/lora-fine-tuning", { waitUntil: "networkidle" });
    const wide = await page.locator(".katex-display").evaluateAll((els) =>
      els
        .filter((el) => el.scrollWidth > el.clientWidth + 1)
        .map((el) => getComputedStyle(el).overflowX),
    );
    expect(wide.length, "a display formula wider than the column").toBeGreaterThan(0);
    expect(new Set(wide)).toEqual(new Set(["auto"]));

    const inline = page.locator(".katex:not(.katex-display .katex)").first();
    expect(await inline.evaluate((el) => getComputedStyle(el).overflowX)).toBe("visible");
  });
});

// The copy-link icon of a titled sub-package section (the "LoRA family" heading
// of /api/lucid.peft.tuners) is hidden until the heading is hovered.  It never
// showed: the icon reveals on ``group-hover`` and no ancestor of the heading was
// a ``group`` (LCD-978).  The group is the heading, not the section — a
// section-wide one would also fire the ``group-hover`` styles of every card below.
test("a sub-package section heading shows its copy-link icon on hover", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await visit(page, "/api/lucid.peft.tuners");
  for (const [title, id] of [
    ["LoRA family", "section-lora-family"],
    ["Refused", "section-refused"],
  ]) {
    const heading = page.locator(`h2#${id}`);
    await expect(heading, title).toHaveText(title);
    const icon = heading.locator('button[aria-label="Copy link to section"]');
    await page.mouse.move(0, 0);
    await expect(icon, `${title}: hidden at rest`).toBeHidden();
    await heading.hover();
    await expect(icon, `${title}: shown while the heading is hovered`).toBeVisible();
    await heading.locator("xpath=..").locator("a").first().hover();
    await expect(icon, `${title}: hidden again while a card is hovered`).toBeHidden();
  }

  // Using it deep-links the section.
  const heading = page.locator("h2#section-lora-family");
  await heading.hover();
  await heading.locator('button[aria-label="Copy link to section"]').click();
  await expect(page).toHaveURL(/#section-lora-family$/);
});

// The /api index counts ``lucid.peft``'s methods.  The package names one more
// than builds an adapter (ShadowPEFT is documented but refused) and the guides
// say how many build, so the card said "46 methods" next to guides saying 45
// (LCD-978).  The audit contract ``index.card-count`` checks the built site; this
// reads the same sections file independently of the card's own code.
test("the lucid.peft cards count the methods that build an adapter", async ({ page }) => {
  const peft = (
    JSON.parse(readFileSync(join(process.cwd(), "data", "api-sections.json"), "utf8")) as Record<
      string,
      { sections: { id: string; packages?: unknown[] }[] }
    >
  )["lucid.peft"];
  const size = (keep: (id: string) => boolean) =>
    peft.sections.filter((s) => keep(s.id)).reduce((n, s) => n + (s.packages?.length ?? 0), 0);
  const built = size((id) => id !== "refused");
  const named = size(() => true);
  expect(built, "a refused section exists").toBeLessThan(named);

  await visit(page, "/api");
  const card = page.locator('#main-content a[href$="/api/lucid.peft.tuners"]');
  await expect(card).toHaveCount(1);
  await expect(card.getByText(/^\d+ methods$/)).toHaveText(`${built} methods`);
  await expect(card).not.toContainText(`${named} methods`);
});
