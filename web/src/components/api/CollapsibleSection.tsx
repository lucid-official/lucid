"use client";

import * as React from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const STORAGE_KEY_PREFIX = "lucid-docs-section-collapsed:";

interface CollapsibleSectionProps {
  /** Stable identifier for this section — the localStorage key suffix
   *  that keeps the collapsed state across reloads.  It is also the
   *  anchor id, but the caller puts that on its own ``<h2>`` in
   *  ``header`` (the ToC reads headings); the ``<section>`` does not
   *  repeat it, since two elements with one id make ``#fragment``
   *  navigation and ``getElementById`` ambiguous. */
  id: string;
  /** Header content (label + any inline controls like ``AnchorLink``).
   *  Rendered next to the chevron button. */
  header: React.ReactNode;
  /** Section body — hidden when the user collapses the section.  We
   *  use ``hidden`` rather than unmounting so re-expanding is
   *  instant and the children's own state (form input, etc.) survives. */
  children: React.ReactNode;
  /** When ``true``, sections start collapsed on first visit.  Useful
   *  for huge pages (Tensor, nn.functional) where the user likely
   *  scans group headers first.  Default ``false``. */
  defaultCollapsed?: boolean;
  className?: string;
}

/** A ``<section>`` whose body the user can fold via a chevron click.
 *  State is persisted in localStorage per (page, section id) so a
 *  collapse on the Tensor page sticks across navigations.
 *
 *  Implementation note: we deliberately don't use the native
 *  ``<details>`` element here because we want to keep the chevron's
 *  rotation animation, the active-page ToC scroll-spy, and the
 *  per-section URL anchor — all of which fight with ``<details>``'s
 *  shadow-DOM semantics. */
export function CollapsibleSection({
  id,
  header,
  children,
  defaultCollapsed = false,
  className,
}: CollapsibleSectionProps) {
  // SSR-safe initial state: we always start with ``defaultCollapsed``
  // so the server-rendered HTML matches React's first client render.
  // After mount, ``useEffect`` reads the persisted value and updates
  // — any visible "flash" is only the very-first frame which is
  // overwhelmingly the right tradeoff vs. shipping a hydration
  // mismatch warning on every long page.
  const [collapsed, setCollapsed] = React.useState<boolean>(defaultCollapsed);

  React.useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY_PREFIX + id);
      if (stored === "1") setCollapsed(true);
      else if (stored === "0") setCollapsed(false);
    } catch {
      // Storage disabled — keep the default.
    }
  }, [id]);

  const toggle = React.useCallback(() => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(
          STORAGE_KEY_PREFIX + id,
          next ? "1" : "0",
        );
      } catch {
        // Ignore.
      }
      return next;
    });
  }, [id]);

  return (
    <section className={cn("group mb-10", className)}>
      <div className="flex items-center gap-2 mb-3">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={!collapsed}
          aria-label={collapsed ? "Expand section" : "Collapse section"}
          className={cn(
            "inline-flex h-5 w-5 items-center justify-center rounded",
            "text-lucid-text-disabled hover:text-lucid-text-mid transition-colors",
          )}
        >
          <ChevronDown
            className={cn(
              "h-3.5 w-3.5 transition-transform duration-150",
              collapsed && "-rotate-90",
            )}
          />
        </button>
        {header}
      </div>
      {/* ``hidden`` (display: none) over ``opacity-0`` so collapsed
          sections fully release vertical space — the point of the
          collapse is to shorten the page, not just hide pixels. */}
      <div className={cn(collapsed && "hidden")}>{children}</div>
    </section>
  );
}
