/**
 * Type-name → docs URL resolver, used to linkify type annotations in
 * function signatures, parameter tables, and return blocks.
 *
 * Built once at request time from the emitted API JSONs.  Cached at
 * module scope so repeated calls in a single render pass don't re-walk
 * the entire tree.  The cache is per Node worker; that's fine for SSG.
 */

import { getAllModuleSlugs, loadApiData } from "@/lib/api-loader";
import { BASE_PATH } from "@/lib/site";
import { isApiClass, isApiClassModule, isApiModule } from "@/lib/types";

// The type-link map stores base-relative hrefs (``/api/…``) because its other
// consumer — ``TypeAnnotation`` — feeds them to Next's ``<Link>``, which
// prepends the basePath itself.  ``linkifyTypesInHtml`` instead emits a raw
// ``<a>`` into a ``dangerouslySetInnerHTML`` string, which Next does NOT
// rewrite, so it must prepend ``BASE_PATH`` manually (otherwise every linked
// type in a code signature 404s on GitHub Pages, escaping the ``/lucid`` base).

/** Type name (basename, e.g. ``"Tensor"`` / ``"Module"``) → docs URL. */
let _typeLinkCache: ReadonlyMap<string, string> | null = null;

function _buildTypeLinkMap(): ReadonlyMap<string, string> {
  const map = new Map<string, string>();
  // Class-modules (Tensor is the only one today) — the slug IS the
  // landing page, so we link the bare class name to ``/api/<slug>``.
  // Class members of regular modules get one link each to
  // ``/api/<module>/<class>``.
  for (const slug of getAllModuleSlugs()) {
    try {
      const data = loadApiData(slug);
      if (isApiClassModule(data)) {
        // ``lucid.tensor`` exposes ``Tensor`` — case-sensitive,
        // single canonical form.
        map.set(data.name, `/api/${encodeURIComponent(slug)}`);
      } else if (isApiModule(data)) {
        for (const m of data.members) {
          if (isApiClass(m)) {
            // First-write wins so a name shared by two modules
            // (rare — would be a registry collision) keeps the
            // module that appears first in the slug list.  In
            // practice every Lucid class name is unique.
            if (!map.has(m.name)) {
              map.set(
                m.name,
                `/api/${encodeURIComponent(slug)}/${encodeURIComponent(m.name)}`,
              );
            }
          }
        }
      }
    } catch {
      // Missing JSON — skip.
    }
  }
  return map;
}

export function getTypeLinkMap(): ReadonlyMap<string, string> {
  if (_typeLinkCache === null) {
    _typeLinkCache = _buildTypeLinkMap();
  }
  return _typeLinkCache;
}

/** Resolve a single type name to a docs URL, or ``null`` when no class
 *  by that name is documented.  Recognises a few common type-spec
 *  prefixes (``Optional[Tensor]`` → ``Tensor``).  Stays read-only —
 *  callers handle escaping. */
export function resolveTypeLink(name: string): string | null {
  return getTypeLinkMap().get(name) ?? null;
}

/** Post-process shiki-highlighted HTML so every recognised type name
 *  becomes a hyperlink.  The Python lexer emits each identifier as a
 *  ``<span style="…">Name</span>`` token, so we match that exact shape
 *  and wrap the span in an ``<a>``.  Unrecognised identifiers (locals
 *  / params / keywords) are left untouched.
 *
 *  Server-side string transform — runs once per function signature
 *  during the render pass.  No DOM, no client-side hydration. */
export function linkifyTypesInHtml(html: string): string {
  const map = getTypeLinkMap();
  // Capture the *contents* of each token span so we can decide whether
  // it's a known type.  Shiki's span output is ``<span style="...">X</span>``
  // — no nested children, so the inner text is captured directly.
  return html.replace(
    /(<span style="[^"]*">)([A-Za-z_][A-Za-z0-9_]*)(<\/span>)/g,
    (full, openTag, name, closeTag) => {
      const href = map.get(name);
      if (!href) return full;
      // ``decoration-dotted`` separates type links from regular hover
      // underlines so users can tell linked types apart from primitive
      // shiki tokens.
      // The href lands in raw HTML, so escape it as an attribute value.
      const attr = `${BASE_PATH}${href}`
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      return (
        `<a href="${attr}" class="hover:underline decoration-dotted underline-offset-2">` +
        openTag +
        name +
        closeTag +
        `</a>`
      );
    },
  );
}
