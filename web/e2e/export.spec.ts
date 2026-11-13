import { test, expect, type Locator, type Page } from "@playwright/test";

/**
 * The static export, as a visitor's browser meets it (LCD-1005).  Run by the
 * ``export`` project of playwright.config.ts: ``pnpm build`` first, then
 * ``pnpm e2e:export``.  ``e2e/serve-export.mjs`` serves ``out/`` under
 * ``/lucid`` like Pages does.
 *
 * Why not just ``next dev``: three things that decide whether a copied section
 * link works only exist in the export.  Its HTML is complete before any script
 * runs (``next dev`` streams it); its web fonts arrive after first paint and
 * re-wrap a long page by hundreds of pixels; and its page content is either in
 * the document or behind a Suspense boundary that an inline script reveals later
 * - the browser follows a ``#fragment`` once, at load, so a heading that is not
 * rendered yet is never scrolled to.
 */

const BASE = "/lucid";

/** The headings a copied link points at: a package section near the end of a
 *  21 000 px page, and a method group on a class page. */
const FRAGMENTS = [
  { path: "/api/lucid.peft", id: "section-attributes" },
  { path: "/api/lucid.nn/Linear", id: "forward" },
];

/** CPU slowdown factors of Chromium's own throttle: 1 is this machine, 4 is a
 *  slow laptop or a hosted runner - hydration, fonts and the entrance all land
 *  later, in a different order. */
const CPU_RATES = [1, 4];

/** Each load is a test of its own, in a fresh browser context (cold cache, like
 *  a first visit): the old failures were races, so one pass proves little. */
const RUNS = 4;

test.describe("a copied section link lands on its heading", () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  for (const { path, id } of FRAGMENTS) {
    for (const rate of CPU_RATES) {
      for (let run = 1; run <= RUNS; run++) {
        test(`${path}#${id} at ${rate}x CPU (run ${run}/${RUNS})`, async ({ page }) => {
          const cdp = await page.context().newCDPSession(page);
          await cdp.send("Emulation.setCPUThrottlingRate", { rate });

          await page.goto(`${BASE}${path}#${id}`, { waitUntil: "load" });

          const heading = page.locator(`[id="${id}"]`);
          await expect(heading).toHaveCount(1);
          // ``scroll-mt-24``: the heading stops under the fixed header, not behind it.
          const line = await heading.evaluate((el) => parseFloat(getComputedStyle(el).scrollMarginTop));
          expect(line, "the heading has a scroll margin to land on").toBeGreaterThan(0);

          const distance = async () => Math.abs((await heading.boundingBox())!.y - line);
          await expect
            .poll(distance, { timeout: 15_000, message: `the heading never reached its ${line}px line` })
            .toBeLessThanOrEqual(2);
          // ... and stays: by now the fonts, hydration and the entrance have played out.
          await page.waitForTimeout(1_000);
          expect(await distance(), "the heading moved off its line after landing").toBeLessThanOrEqual(2);
        });
      }
    }
  }
});

// The entrance is markup the server writes and React must adopt unchanged; a
// mismatch there is a production console error (and a re-render of the page).
test.describe("the export hydrates without errors", () => {
  for (const path of ["/", "/api/lucid.peft", "/api/lucid.nn/Linear"]) {
    test(path, async ({ page }) => {
      const errors: string[] = [];
      page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
      page.on("console", (msg) => {
        if (msg.type() === "error" && !/favicon/i.test(msg.text())) errors.push(`console.error: ${msg.text()}`);
      });
      await page.goto(`${BASE}${path}`, { waitUntil: "networkidle" });
      // React marks every node it has adopted with a ``__reactFiber$…`` property.
      await page.waitForFunction(() => {
        const heading = document.querySelector("#main-content h1");
        return !!heading && Object.keys(heading).some((key) => key.startsWith("__reactFiber$"));
      });
      expect(errors, errors.join("\n")).toEqual([]);
    });
  }
});

/** Is this element - and everything above it - actually drawn?  ``toBeVisible``
 *  calls an ``opacity: 0`` element visible, which is exactly the failure here. */
async function drawn(el: Locator): Promise<boolean> {
  return el.evaluate((node) => {
    for (let n: Element | null = node; n; n = n.parentElement) {
      const style = getComputedStyle(n);
      if (style.opacity !== "1" || style.display === "none" || style.visibility === "hidden") return false;
    }
    const box = node.getBoundingClientRect();
    return box.width > 0 && box.height > 0 && (node.textContent ?? "").trim().length > 0;
  });
}

async function expectDrawn(page: Page, selector: string, what: string): Promise<void> {
  const el = page.locator(selector).first();
  await expect(el, `${what}: ${selector} is in the document`).toHaveCount(1);
  // The entrance is a 0.3 s CSS animation; wait it out rather than race it.
  await expect.poll(() => drawn(el), { message: `${what} is not drawn`, timeout: 5_000 }).toBe(true);
}

// The server writes the page into the HTML; JavaScript only adds behaviour.  An
// entrance that starts from ``opacity: 0`` in the markup (framer-motion's
// ``initial``) leaves it blank until the bundle has loaded and hydrated: with
// scripts off, for a crawler that does not run them, and for a slow connection.
test.describe("a page is readable without JavaScript", () => {
  test.use({ javaScriptEnabled: false, viewport: { width: 1440, height: 900 } });

  const PAGES = [
    { path: "/", name: "home" },
    { path: "/api", name: "api landing" },
    { path: "/api/lucid.peft", name: "package overview" },
    { path: "/api/lucid.nn/Linear", name: "class page" },
    { path: "/docs/tutorials/lora-fine-tuning", name: "tutorial" },
  ];

  for (const { path, name } of PAGES) {
    test(`${name} (${path}) shows its heading and text`, async ({ page }) => {
      await page.goto(`${BASE}${path}`, { waitUntil: "load" });
      await expectDrawn(page, "#main-content h1", "the page heading");
      await expectDrawn(page, "#main-content p", "the first paragraph");
    });
  }
});
