/**
 * Copy the brand kit's web files into the site.
 *
 *   node scripts/sync-brand.mjs      (run by predev / prebuild / dev:fast)
 *
 * Source of truth is the repository brand kit, ../assets (see its README).
 * The copies land in public/brand/ (gitignored), served at /brand/ — icons,
 * the web-manifest icons, the social card and the logo artwork.  Nothing is
 * committed twice, so the site can never ship a stale or hand-edited logo.
 *
 * Why not Next's file-based metadata (app/favicon.ico, app/icon.svg): under a
 * basePath, Next stops recognising app/favicon.ico as the favicon (it matches
 * the literal "/favicon.ico" URL) and the root page then drops it whenever
 * app/icon.svg also exists.  The layout links these files explicitly instead.
 *
 * Copies whose source left the list are removed, and a missing kit fails the
 * build rather than deploying a site without icons.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const KIT = join(process.cwd(), "..", "assets");
const OUT = join(process.cwd(), "public", "brand");

/** Paths inside the kit; each lands in public/brand/ under its base name. */
const FILES = [
  "favicon.ico",
  "svg/favicon.svg",
  "png/apple-touch-icon.png",
  "png/favicon-16.png",
  "png/favicon-32.png",
  "png/favicon-48.png",
  "png/icon-192.png",
  "png/icon-512.png",
  "png/icon-maskable-512.png",
  "png/og-image.png",
  "svg/lucid-icon-dark-square.svg",
  "svg/lucid-icon-dark.svg",
  "svg/lucid-icon-light.svg",
  "svg/lucid-icon-violet.svg",
  "svg/lucid-lockup-horizontal-on-dark.svg",
  "svg/lucid-lockup-horizontal-on-light.svg",
  "svg/lucid-lockup-stacked-on-dark.svg",
  "svg/lucid-lockup-stacked-on-light.svg",
  "svg/lucid-mark-black.svg",
  "svg/lucid-mark-duotone.svg",
  "svg/lucid-mark-white.svg",
  "svg/lucid-mark.svg",
  "svg/lucid-og.svg",
  "svg/lucid-wordmark-on-dark.svg",
  "svg/lucid-wordmark-on-light.svg",
];

if (!existsSync(KIT)) {
  throw new Error(`[sync-brand] brand kit not found at ${KIT}`);
}
mkdirSync(OUT, { recursive: true });

const names = new Set();
let written = 0;
for (const rel of FILES) {
  const src = join(KIT, rel);
  if (!existsSync(src)) throw new Error(`[sync-brand] ${rel} is missing from the brand kit`);
  const name = rel.split("/").pop();
  names.add(name);
  const bytes = readFileSync(src);
  const dest = join(OUT, name);
  // Leave identical copies alone so the dev server's watcher stays quiet.
  if (existsSync(dest) && readFileSync(dest).equals(bytes)) continue;
  writeFileSync(dest, bytes);
  written += 1;
}

for (const name of readdirSync(OUT)) {
  if (!names.has(name)) rmSync(join(OUT, name));
}

console.log(`[sync-brand] ${FILES.length} file(s) from ../assets → public/brand/ (${written} updated)`);
