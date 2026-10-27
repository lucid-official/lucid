import { readFileSync } from "fs";
import { join, resolve, sep } from "path";
import type {
  ApiData,
  ApiModule,
  ApiClassModule,
  ApiMember,
  ModuleSlug,
  MODULE_SLUGS,
} from "./types";

const API_DATA_DIR = join(process.cwd(), "public", "api-data");

// ---------------------------------------------------------------------------
// Core loaders (build-time only — called from generateStaticParams / page)
// ---------------------------------------------------------------------------

/** Lightweight runtime shape check.  We can't pull in zod for a docs
 *  site, but a raw ``as ApiData`` cast over arbitrary JSON would let
 *  malformed payloads (mid-broken cache, partial fetch, corrupted
 *  file) flow into every renderer and crash deep in the tree with
 *  cryptic errors.  Asserting the top-level invariants right here at
 *  the I/O boundary turns those into a single legible error message
 *  that the per-route ``error.tsx`` surfaces.
 *
 *  Returned shape:
 *    * ``kind`` must be one of the known string literals.
 *    * Either ``members`` (modules) or ``methods`` (class-modules)
 *      must be an array — the renderer relies on that. */
function _validateApiData(data: unknown, slug: string): asserts data is ApiData {
  if (typeof data !== "object" || data === null) {
    throw new Error(
      `API data for "${slug}" is not an object — got ${typeof data}`,
    );
  }
  const obj = data as Record<string, unknown>;
  if (typeof obj.kind !== "string") {
    throw new Error(
      `API data for "${slug}" is missing the "kind" field (or it's not a string).`,
    );
  }
  if (obj.kind !== "module" && obj.kind !== "class-module") {
    throw new Error(
      `API data for "${slug}" has unrecognised kind "${obj.kind}" — expected "module" or "class-module".`,
    );
  }
  if (obj.kind === "module" && !Array.isArray(obj.members)) {
    throw new Error(
      `API data for "${slug}" is a module but "members" is not an array (got ${typeof obj.members}).`,
    );
  }
  if (obj.kind === "class-module" && !Array.isArray(obj.methods)) {
    throw new Error(
      `API data for "${slug}" is a class-module but "methods" is not an array (got ${typeof obj.methods}).`,
    );
  }
}

export function loadApiData(slug: string): ApiData {
  // The slug comes from the route, which ``next dev`` does not restrict to
  // ``generateStaticParams``; keep it from naming a file outside api-data.
  const filePath = resolve(API_DATA_DIR, `${slug}.json`);
  if (!filePath.startsWith(API_DATA_DIR + sep)) {
    throw new Error(`API data not found for slug: "${slug}".`);
  }
  let raw: string;
  try {
    raw = readFileSync(filePath, "utf-8");
  } catch {
    throw new Error(`API data not found for slug: "${slug}". Run pnpm build:api first.`);
  }
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    throw new Error(`API data for "${slug}" is not valid JSON: ${message}`);
  }
  _validateApiData(data, slug);
  return data;
}

export function loadApiModule(slug: string): ApiModule {
  const data = loadApiData(slug);
  if (data.kind !== "module") {
    throw new Error(`Expected module but got "${data.kind}" for slug: "${slug}"`);
  }
  return data;
}

export function loadApiClassModule(slug: string): ApiClassModule {
  const data = loadApiData(slug);
  if (data.kind !== "class-module") {
    throw new Error(`Expected class-module but got "${data.kind}" for slug: "${slug}"`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Member lookup
// ---------------------------------------------------------------------------

export function findMember(data: ApiData, memberName: string): ApiMember | undefined {
  if (data.kind === "module") {
    return data.members.find((m) => m.name === memberName);
  }
  if (data.kind === "class-module") {
    // Tensor class — methods are top-level
    const asMethod = data.methods.find((m) => m.name === memberName);
    return asMethod as ApiMember | undefined;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Static params helpers (used in generateStaticParams)
// ---------------------------------------------------------------------------

/** Return all valid module slugs that have a corresponding JSON file.
 *
 *  Convention: files whose basename begins with ``_`` are side-channel
 *  caches (e.g. ``_summaries.json``, ``_summaries.meta.json`` written by
 *  ``tools/build_model_summaries.py``), not module data — they share the
 *  directory only because the consumers happen to need them at the same
 *  build-time read path.  Excluding them here is what keeps the sidebar
 *  from sprouting phantom ``Summaries`` / ``Summaries.meta`` entries. */
export function getAllModuleSlugs(): string[] {
  const fs = require("fs") as typeof import("fs");
  try {
    return fs
      .readdirSync(API_DATA_DIR)
      .filter((f: string) => f.endsWith(".json") && !f.startsWith("_"))
      .map((f: string) => encodeURIComponent(f.slice(0, -5)));
  } catch {
    return [];
  }
}

/** Load all module metadata (name, slug, memberCount, summary) for index pages. */
export function loadModuleIndex(): Array<{
  slug: string;
  name: string;
  memberCount: number;
  summary: string | null;
}> {
  const slugs = getAllModuleSlugs();
  return slugs
    .map((slug) => {
      try {
        const data = loadApiData(slug);
        const memberCount =
          data.kind === "module"
            ? data.members.length
            : data.kind === "class-module"
              ? data.methods.length
              : 0;
        return { slug, name: data.name, memberCount, summary: data.summary ?? null };
      } catch {
        return null;
      }
    })
    .filter((x): x is NonNullable<typeof x> => x !== null);
}
