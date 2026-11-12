"use client";

import * as React from "react";
import { Link2, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface AnchorLinkProps {
  /** Anchor id of the heading this control belongs to.  The button
   *  copies ``<origin>/<pathname>#<id>`` to the clipboard. */
  id: string;
  className?: string;
}

/** Hover-reveal copy-anchor control rendered next to ``<h2>`` / ``<h3>``
 *  headings.  Standard docs UX pattern (GitHub, MDN, …) — click the
 *  ``#`` icon and the deep-link to that section is on the clipboard
 *  ready to paste into a chat / issue / commit message.
 *
 *  The button is invisible until the element that *hosts* it is hovered, or
 *  the button itself takes keyboard focus — at rest it doesn't visually
 *  compete with the heading text.  The host is whichever element carries
 *  ``group/anchor`` (the heading row, a collapsible group's header row, a
 *  method's card); that name — not Tailwind's bare ``group`` — is what the
 *  button listens to, because a bare ``group`` matches *every* ancestor that
 *  has it, so a section-wide one would reveal each icon in the section at once.
 *  Give each host the smallest element whose hover should show *its* icon, and
 *  never nest one ``group/anchor`` inside another that holds an ``AnchorLink``. */
export function AnchorLink({ id, className }: AnchorLinkProps) {
  const [copied, setCopied] = React.useState(false);

  const handleClick = React.useCallback(
    async (e: React.MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      const url =
        typeof window !== "undefined"
          ? `${window.location.origin}${window.location.pathname}#${id}`
          : `#${id}`;
      try {
        await navigator.clipboard.writeText(url);
        // Also update the address bar so the back button works as
        // expected after copy — small but appreciated.
        if (typeof window !== "undefined") {
          window.history.replaceState(null, "", `#${id}`);
        }
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      } catch {
        // Clipboard API blocked (insecure context) — fall back to
        // just updating the URL.  User can still copy from address bar.
        if (typeof window !== "undefined") {
          window.history.replaceState(null, "", `#${id}`);
        }
      }
    },
    [id],
  );

  // Visibility: ``sr-only`` at rest, ``not-sr-only`` when revealed.  Two
  // earlier recipes each broke one thing.  ``opacity-0`` reserved the icon's
  // 20 px box plus the 8 px flex gap on each side, so the trailing return type
  // looked orphaned from the function name at rest.  ``hidden`` (display: none)
  // released the slot but also took the button out of the tab order, so a
  // ``focus-within`` reveal could never fire and keyboard users could not reach
  // any copy-link icon.  ``sr-only`` is out of flow (absolute, 1 px, clipped):
  // no box, no flex gap, the heading is laid out as if the button were absent
  // — yet it stays focusable and in the accessibility tree.  It is revealed on
  // the host's hover and on the button's own ``focus-visible`` (not on
  // ``focus-within``, which would show it whenever any link in the host took
  // focus).  Revealing adds the slot back, a one-frame width shift small enough
  // to read as a reveal, not a jump.
  return (
    <button
      type="button"
      onClick={handleClick}
      aria-label={copied ? "Link copied" : "Copy link to section"}
      className={cn(
        "inline-flex items-center justify-center rounded transition-colors duration-150",
        copied
          ? "size-5 text-lucid-success"
          : cn(
              "sr-only",
              "group-hover/anchor:not-sr-only group-hover/anchor:size-5",
              "focus-visible:not-sr-only focus-visible:size-5",
              "text-lucid-text-disabled hover:text-lucid-primary focus-visible:text-lucid-primary",
            ),
        className,
      )}
    >
      {copied ? <Check className="h-3 w-3" /> : <Link2 className="h-3 w-3" />}
    </button>
  );
}
