"use client";

import * as React from "react";
// ``fuse.js`` is ~50 KB minified and only needed once the user opens
// the dialog — pull the *type* eagerly (cheap) and the *runtime* via
// dynamic import inside the open effect.  Cuts the layout shell's
// initial JS by the full Fuse module on first paint.
import type FuseModule from "fuse.js";
type FuseInstance<T> = InstanceType<typeof FuseModule<T>>;
import { Search, FileText, Box, Zap, BookOpen, Clock, ArrowRight } from "lucide-react";
import { getRecentPages, type RecentPage } from "@/lib/recent-pages";
import { useRouter } from "next/navigation";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { SearchEntry } from "@/lib/search-index";

const KIND_CONFIG = {
  "api-module": {
    icon: Box,
    color: "text-lucid-primary",
    bg: "bg-lucid-primary/10",
    label: "Module",
  },
  "api-class": {
    icon: Box,
    color: "text-lucid-warning",
    bg: "bg-lucid-warning/10",
    label: "Class",
  },
  "api-function": {
    icon: Zap,
    color: "text-lucid-blue",
    bg: "bg-lucid-blue/10",
    label: "Function",
  },
  doc: {
    icon: BookOpen,
    color: "text-lucid-success",
    bg: "bg-lucid-success/10",
    label: "Guide",
  },
} as const;

function KindIcon({ kind }: { kind: SearchEntry["kind"] }) {
  const { icon: Icon, color, bg } = KIND_CONFIG[kind];
  return (
    <span
      className={cn(
        "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg",
        bg,
      )}
    >
      <Icon className={cn("h-3.5 w-3.5", color)} />
    </span>
  );
}

interface SearchResultItemProps {
  entry: SearchEntry;
  active: boolean;
  onSelect: () => void;
}

function SearchResultItem({ entry, active, onSelect }: SearchResultItemProps) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition-colors",
        active ? "bg-lucid-elevated" : "hover:bg-lucid-elevated/60",
      )}
    >
      <KindIcon kind={entry.kind} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-lucid-text-high truncate">
            {entry.title}
          </span>
          {entry.badge && (
            <span className="shrink-0 text-[10px] text-lucid-text-disabled">
              {entry.badge}
            </span>
          )}
        </div>
        {entry.summary && (
          <p className="mt-0.5 text-xs text-lucid-text-low line-clamp-1">
            {/* A summary is markdown (`code` spans); one line of plain text here. */}
            {entry.summary.replace(/`/g, "")}
          </p>
        )}
      </div>
    </button>
  );
}

type KindFilter = "all" | SearchEntry["kind"];

const KIND_FILTERS: Array<{ value: KindFilter; label: string }> = [
  { value: "all",          label: "All" },
  { value: "api-class",    label: "Classes" },
  { value: "api-function", label: "Functions" },
  { value: "api-module",   label: "Modules" },
  { value: "doc",          label: "Guides" },
];

export function SearchDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const router = useRouter();
  const [query, setQuery] = React.useState("");
  // Unfiltered Fuse hits before per-kind narrowing — we keep BOTH so
  // switching filters never re-runs the (cheap, but cache-warm-only)
  // Fuse search.
  const [rawResults, setRawResults] = React.useState<SearchEntry[]>([]);
  // "Did you mean" suggestions surfaced when the primary search yields
  // nothing — built by re-running Fuse with a much looser threshold so
  // typos and partial-name recall still produce useful candidates.
  const [fuzzyHits, setFuzzyHits] = React.useState<SearchEntry[]>([]);
  const [kindFilter, setKindFilter] = React.useState<KindFilter>("all");
  const [activeIndex, setActiveIndex] = React.useState(0);
  const [index, setIndex] = React.useState<FuseInstance<SearchEntry> | null>(null);
  const [fuzzyIndex, setFuzzyIndex] = React.useState<FuseInstance<SearchEntry> | null>(null);
  // Recents are loaded each time the dialog opens — that way a page
  // visited *after* the previous open shows up without a refresh.
  // Stored separately from ``rawResults`` because the empty state
  // renders them in their own section with different ranking semantics.
  const [recents, setRecents] = React.useState<RecentPage[]>([]);
  // ``loadError`` tracks Fuse-module / search-data fetch failures.
  // Without it, the dialog would sit empty with no signal to the
  // user that something is wrong — and a retry would never fire
  // because ``index`` stays null but ``open`` doesn't change.
  const [loadError, setLoadError] = React.useState<string | null>(null);
  // ``loadAttempt`` lets the user click "Retry" without remounting
  // the dialog — bumping it forces the load effect to re-run.
  const [loadAttempt, setLoadAttempt] = React.useState(0);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Load search index once, lazy.  ``fuse.js`` is dynamic-imported
  // so it only enters the JS bundle on the first dialog open — users
  // who never press ``⌘K`` never pay its ~50 KB minified cost.
  React.useEffect(() => {
    if (!open) return;
    if (index) return;
    let cancelled = false;
    setLoadError(null);
    Promise.all([
      import("fuse.js"),
      fetch("/api/search-data").then((r) => {
        if (!r.ok) throw new Error(`search-data fetch failed: ${r.status}`);
        return r.json();
      }),
    ])
      .then(([fuseModule, data]) => {
        if (cancelled) return;
        // ``fuse.js`` ships as an ES module with the class as the
        // default export.  Some bundler configs unwrap it, others
        // don't — handle both shapes defensively so swapping out
        // Turbopack / Webpack / Vite later doesn't break us.
        const Fuse = (fuseModule.default ?? fuseModule) as typeof FuseModule;
        const entries = data as SearchEntry[];
        setIndex(
          new Fuse(entries, {
            keys: [
              { name: "title", weight: 3 },
              { name: "summary", weight: 1 },
              { name: "badge", weight: 1 },
            ],
            threshold: 0.35,
            includeScore: true,
          }),
        );
        // Looser sibling index used only when the strict one returns
        // nothing.  Title-only + ``threshold: 0.6`` catches single-
        // char typos and partial-name recall (``linaer`` → ``linear``)
        // without polluting the primary results list.
        setFuzzyIndex(
          new Fuse(entries, {
            keys: [{ name: "title", weight: 1 }],
            threshold: 0.6,
            distance: 50,
            includeScore: true,
          }),
        );
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        // Surface the failure to the user — common causes are a
        // bundle chunk that didn't deploy (CDN propagation lag) or
        // network drop mid-import.  A retry button lets the user
        // try again without remounting the whole dialog.
        const message = e instanceof Error ? e.message : String(e);
        setLoadError(message || "Failed to load search index");
      });
    return () => {
      cancelled = true;
    };
  }, [open, index, loadAttempt]);

  // Focus input on open + refresh the recents snapshot.  Reading
  // localStorage every open is cheap (one JSON.parse on a max-8-item
  // array) and guarantees a freshly-visited page shows up without
  // needing a page reload.
  React.useEffect(() => {
    if (open) {
      setQuery("");
      setKindFilter("all");
      setActiveIndex(0);
      setRecents(getRecentPages());
      // Clear on close so a rapidly-dismissed dialog doesn't focus a
      // stale input after unmount.
      const id = setTimeout(() => inputRef.current?.focus(), 50);
      return () => clearTimeout(id);
    }
  }, [open]);

  // Run search — fetch a deep set, then narrow per filter at render time.
  React.useEffect(() => {
    if (!index || !query.trim()) {
      setRawResults([]);
      setFuzzyHits([]);
      setActiveIndex(0);
      return;
    }
    const hits = index.search(query, { limit: 50 });
    setRawResults(hits.map((h) => h.item));
    setActiveIndex(0);
    // Compute "did you mean" candidates only when the strict search
    // returns nothing — saves the second Fuse pass on the common case
    // where the user typed something that matches.
    if (hits.length === 0 && fuzzyIndex) {
      const fuzzyResults = fuzzyIndex.search(query, { limit: 5 });
      setFuzzyHits(fuzzyResults.map((h) => h.item));
    } else {
      setFuzzyHits([]);
    }
  }, [query, index, fuzzyIndex]);

  // Reset highlight whenever the filter changes — the visible list shifts.
  React.useEffect(() => {
    setActiveIndex(0);
  }, [kindFilter]);

  // Counts per kind across the unfiltered hits — drives chip badges.
  const counts = React.useMemo(() => {
    const acc: Record<KindFilter, number> = {
      "all": rawResults.length,
      "api-class": 0,
      "api-function": 0,
      "api-module": 0,
      "doc": 0,
    };
    for (const r of rawResults) acc[r.kind]++;
    return acc;
  }, [rawResults]);

  // Final visible list — cap at 8 to keep the dialog scannable.
  const results = React.useMemo(() => {
    const filtered = kindFilter === "all"
      ? rawResults
      : rawResults.filter((r) => r.kind === kindFilter);
    return filtered.slice(0, 8);
  }, [rawResults, kindFilter]);

  const navigate = React.useCallback(
    (entry: SearchEntry) => {
      router.push(entry.href);
      onClose();
    },
    [router, onClose],
  );

  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      if (results.length === 0) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, results.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        navigate(results[activeIndex]);
      }
    },
    [results, activeIndex, navigate],
  );

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent>
        {/* Screen-reader-only title + description.  Radix Dialog
            requires a ``DialogTitle`` for assistive-tech announcement;
            the visible "Search" affordance is the input placeholder
            and the Search icon, both already clear to sighted users,
            so we mount the title via ``sr-only`` rather than adding a
            redundant visible heading. */}
        <DialogTitle className="sr-only">Search docs</DialogTitle>
        <DialogDescription className="sr-only">
          Search across Lucid API symbols, guides, and the Tensor surface.
          Use ↑↓ to navigate results, Enter to open, Esc to close.
        </DialogDescription>

        {/* Search input */}
        <div className="flex items-center gap-3 border-b border-lucid-border px-4 py-3.5">
          <Search className="h-4 w-4 shrink-0 text-lucid-text-low" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search docs and API…"
            className="flex-1 bg-transparent text-sm text-lucid-text-high placeholder:text-lucid-text-disabled outline-none"
          />
          <kbd className="hidden sm:inline-flex items-center gap-0.5 rounded border border-lucid-border px-1.5 py-0.5 text-[10px] text-lucid-text-disabled">
            esc
          </kbd>
        </div>

        {/* Kind filter chips — appear once there are results to filter. */}
        {rawResults.length > 0 && (
          <div className="flex flex-wrap gap-1.5 border-b border-lucid-border px-3 py-2">
            {KIND_FILTERS.map(({ value, label }) => {
              const count = counts[value];
              const active = kindFilter === value;
              const disabled = value !== "all" && count === 0;
              return (
                <button
                  key={value}
                  type="button"
                  onClick={() => setKindFilter(value)}
                  disabled={disabled}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium transition-colors",
                    active
                      ? "bg-lucid-primary/15 border-lucid-primary/40 text-lucid-primary"
                      : "border-lucid-border text-lucid-text-low hover:text-lucid-text-mid hover:bg-lucid-elevated",
                    disabled && "opacity-40 cursor-not-allowed hover:bg-transparent hover:text-lucid-text-low",
                  )}
                >
                  {label}
                  <span className={cn(
                    "font-mono text-[10px]",
                    active ? "text-lucid-primary/80" : "text-lucid-text-disabled",
                  )}>
                    {count}
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {/* Results */}
        <div className="max-h-[360px] overflow-y-auto p-2">
          {loadError && (
            <div className="flex flex-col items-center gap-3 py-10 text-center px-6">
              <FileText className="h-8 w-8 text-lucid-error/70" />
              <div>
                <p className="text-sm text-lucid-text-mid">
                  Search index didn&rsquo;t load
                </p>
                <p className="mt-1 text-[11px] font-mono text-lucid-text-disabled break-all">
                  {loadError}
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setLoadError(null);
                  setLoadAttempt((n) => n + 1);
                }}
                className="rounded-md border border-lucid-border bg-lucid-elevated px-3 py-1 text-xs text-lucid-text-mid hover:text-lucid-text-high transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {!loadError && query.trim() === "" && recents.length === 0 && (
            <div className="flex flex-col items-center gap-2 py-10">
              <FileText className="h-8 w-8 text-lucid-text-disabled" />
              <p className="text-sm text-lucid-text-low">
                Start typing to search…
              </p>
            </div>
          )}

          {query.trim() === "" && recents.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 px-3 pt-2 pb-1 text-[10px] font-semibold tracking-widest text-lucid-text-disabled uppercase">
                <Clock className="h-3 w-3" />
                Recent
              </div>
              {recents.map((page) => (
                <button
                  key={page.href}
                  type="button"
                  onClick={() => {
                    router.push(page.href);
                    onClose();
                  }}
                  className={cn(
                    "group flex w-full items-center justify-between gap-3",
                    "rounded-xl px-3 py-2 text-left transition-colors",
                    "hover:bg-lucid-elevated",
                  )}
                >
                  <span className="flex min-w-0 items-baseline gap-2">
                    <span className="truncate font-mono text-sm text-lucid-text-mid group-hover:text-lucid-text-high">
                      {page.title}
                    </span>
                    <span className="hidden sm:inline truncate text-[11px] text-lucid-text-disabled">
                      {page.href}
                    </span>
                  </span>
                  <ArrowRight className="h-3.5 w-3.5 shrink-0 text-lucid-text-disabled transition-colors group-hover:text-lucid-primary" />
                </button>
              ))}
            </div>
          )}

          {query.trim() !== "" && results.length === 0 && (
            <div className="py-10 text-center space-y-3">
              <p className="text-sm text-lucid-text-low">
                No results for{" "}
                <span className="font-medium text-lucid-text-mid">
                  &ldquo;{query}&rdquo;
                </span>
                {kindFilter !== "all" && (
                  <>
                    {" "}in <span className="font-mono text-lucid-text-mid">{kindFilter}</span>
                  </>
                )}.
              </p>
              {kindFilter !== "all" && rawResults.length > 0 && (
                <button
                  type="button"
                  onClick={() => setKindFilter("all")}
                  className="text-xs text-lucid-primary hover:text-lucid-primary-light"
                >
                  Show {rawResults.length} match{rawResults.length === 1 ? "" : "es"} in all kinds
                </button>
              )}
              {/* "Did you mean" — only renders when the user's strict
                  query genuinely returned nothing across every kind.
                  Hidden when ``rawResults`` has hits but the active
                  filter excluded them (we already offer the
                  ``Show … in all kinds`` button above for that). */}
              {kindFilter === "all" && fuzzyHits.length > 0 && (
                <div className="text-left max-w-sm mx-auto space-y-1.5 pt-1">
                  <p className="text-[11px] font-semibold tracking-widest text-lucid-text-disabled uppercase">
                    Did you mean…
                  </p>
                  <ul className="space-y-1">
                    {fuzzyHits.map((entry) => (
                      <li key={entry.id}>
                        <button
                          type="button"
                          onClick={() => navigate(entry)}
                          className="font-mono text-sm text-lucid-primary hover:text-lucid-primary-light hover:underline"
                        >
                          {entry.title}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {results.map((entry, i) => (
            <SearchResultItem
              key={entry.id}
              entry={entry}
              active={i === activeIndex}
              onSelect={() => navigate(entry)}
            />
          ))}
        </div>

        {/* Footer hint */}
        {results.length > 0 && (
          <div className="flex items-center gap-3 border-t border-lucid-border px-4 py-2.5 text-[11px] text-lucid-text-disabled">
            <span>
              <kbd className="rounded border border-lucid-border px-1 py-0.5">↑</kbd>
              <kbd className="ml-0.5 rounded border border-lucid-border px-1 py-0.5">↓</kbd>{" "}
              navigate
            </span>
            <span>
              <kbd className="rounded border border-lucid-border px-1 py-0.5">↵</kbd>{" "}
              open
            </span>
            <span>
              <kbd className="rounded border border-lucid-border px-1 py-0.5">esc</kbd>{" "}
              close
            </span>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
