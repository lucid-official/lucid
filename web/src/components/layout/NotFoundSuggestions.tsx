"use client";

import * as React from "react";
import Link from "next/link";

interface NotFoundSuggestionsProps {
  /** Pre-built list of every routable slug on the docs site.  The
   *  ``not-found.tsx`` page passes this in from server-side so the
   *  client doesn't have to re-walk the JSON tree just to suggest
   *  alternatives. */
  slugs: string[];
}

/** Levenshtein-style edit distance — small enough that a hand-rolled
 *  matrix is fine for the (slug count) × (one input) work the 404
 *  page does on mount.  No external dep, no init cost. */
function _editDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  // Two-row rolling table — O(min(|a|, |b|)) space.
  let prev = new Array(b.length + 1);
  let curr = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    curr[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1, // deletion
        curr[j - 1] + 1, // insertion
        prev[j - 1] + cost, // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[b.length];
}

/** Best-N fuzzy matches against the slug list.  Ranks by edit distance
 *  with a small substring bonus so ``linear`` ranks ahead of ``linalg``
 *  when the user typed ``linaer``.  Caps at ``maxResults``. */
function _fuzzyMatches(target: string, slugs: string[], maxResults = 5): string[] {
  if (!target) return [];
  const normTarget = target.toLowerCase();
  const scored = slugs.map((slug) => {
    const last = slug.split(/[./]/).pop() ?? slug;
    const lower = last.toLowerCase();
    let score = _editDistance(normTarget, lower);
    // Substring bonus — strong signal the user wanted *this* family.
    if (lower.includes(normTarget) || normTarget.includes(lower)) score -= 3;
    // Prefer the bare basename match over the full slug match if both
    // candidates exist — most users type ``linear`` not ``lucid.nn.functional.linear``.
    if (lower === normTarget) score = -10;
    return { slug, score };
  });
  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, maxResults).map((s) => s.slug);
}

export function NotFoundSuggestions({ slugs }: NotFoundSuggestionsProps) {
  // Pull the slug from the current URL.  ``not-found.tsx`` doesn't
  // receive the missing slug as a prop in Next.js App Router, so we
  // read it from ``window.location.pathname`` on mount.
  const [target, setTarget] = React.useState<string>("");
  React.useEffect(() => {
    const path = window.location.pathname;
    // Strip leading ``/api/`` / ``/docs/`` so the fuzzy match operates
    // on the symbol name the user actually typed, not the framework prefix.
    const stripped = path.replace(/^\/(api|docs)\//, "").replace(/\/$/, "");
    setTarget(stripped);
  }, []);

  const matches = React.useMemo(
    () => _fuzzyMatches(target, slugs, 5),
    [target, slugs],
  );

  if (!target || matches.length === 0) return null;

  return (
    <div className="mt-8 max-w-md mx-auto text-left">
      <p className="text-[11px] font-semibold tracking-widest text-lucid-text-disabled uppercase mb-2">
        Did you mean…
      </p>
      <ul className="space-y-1.5 rounded-xl border border-lucid-border bg-lucid-surface/40 px-4 py-3">
        {matches.map((slug) => (
          <li key={slug}>
            <Link
              href={`/api/${encodeURIComponent(slug)}`}
              className="font-mono text-sm text-lucid-primary hover:text-lucid-primary-light hover:underline"
            >
              {slug}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
