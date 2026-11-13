"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, Search, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { springs } from "@/components/motion/springs";

export interface SidebarItem {
  title: string;
  href?: string;
  items?: SidebarItem[];
  /** "C" = class, "f" = function, or a plain count string like "22" */
  badge?: string;
  /** Optional secondary label rendered next to the title (e.g. the real
   *  Python slug "lucid.nn" next to the friendly alias "Neural Networks"). */
  tag?: string;
  /** When true, render as a horizontal divider — all other fields ignored.
   *  Used to visually break the top-level list before pinned-to-bottom
   *  entries like the C++ engine surface. */
  separator?: boolean;
}

// ---------------------------------------------------------------------------
// KindBadge — renders "C"/"f" with semantic api colors, or a plain count
// ---------------------------------------------------------------------------
function KindBadge({
  badge,
  depth,
  active,
}: {
  badge: string;
  depth: number;
  active: boolean;
}) {
  // Leaf-level kind indicators — single-letter pills coloured to match
  // the class-kind palette used by ClassBadge / ClassDoc.
  //   "C" = concrete class (violet)
  //   "D" = @dataclass     (cyan)
  //   "A" = abstract       (magenta, dashed)
  //   "P" = Protocol       (teal, dashed)
  //   "f" = function       (blue)
  if (badge === "C") {
    return (
      <span className="rounded px-1 py-px text-[8px] font-bold tracking-wide border bg-api-class/10 border-api-class/30 text-api-class">
        C
      </span>
    );
  }
  if (badge === "D") {
    return (
      <span className="rounded px-1 py-px text-[8px] font-bold tracking-wide border bg-api-class-dataclass/10 border-api-class-dataclass/35 text-api-class-dataclass">
        D
      </span>
    );
  }
  if (badge === "A") {
    return (
      <span className="rounded px-1 py-px text-[8px] font-bold tracking-wide border border-dashed bg-api-class-abstract/10 border-api-class-abstract/35 text-api-class-abstract">
        A
      </span>
    );
  }
  if (badge === "P") {
    return (
      <span className="rounded px-1 py-px text-[8px] font-bold tracking-wide border border-dashed bg-api-class-protocol/10 border-api-class-protocol/40 text-api-class-protocol">
        P
      </span>
    );
  }
  if (badge === "f") {
    return (
      <span className="rounded px-1 py-px text-[8px] font-bold tracking-wide border bg-api-fn/10 border-api-fn/30 text-api-fn">
        f
      </span>
    );
  }
  // Plain count badge (module level)
  return (
    <span
      className={cn(
        "rounded-sm px-1.5 py-0.5 text-[10px] font-semibold",
        active
          ? "bg-api-class/15 text-api-class"
          : "bg-lucid-surface text-lucid-text-disabled",
      )}
    >
      {badge}
    </span>
  );
}

/** Renders the item's title plus an optional dim mono-font tag.  Used to
 *  show the real Python slug ("lucid.nn") next to the friendly alias
 *  ("Neural Networks") in the sidebar. */
function TitleWithTag({ title, tag }: { title: string; tag?: string }) {
  if (!tag) return <span className="truncate">{title}</span>;
  return (
    <span className="flex items-baseline gap-1.5 min-w-0 flex-1">
      <span className="truncate">{title}</span>
      <span className="shrink-0 font-mono text-[10px] text-lucid-text-disabled">
        {tag}
      </span>
    </span>
  );
}

interface SidebarGroupProps {
  item: SidebarItem;
  depth?: number;
  /** When true, override the expand state to open regardless of the
   *  user's previous click history.  Threaded down from the sidebar's
   *  filter input so every match is visible while a query is active. */
  forceExpand?: boolean;
}

function SidebarSeparator() {
  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      className="my-3 mx-2 border-t border-lucid-border"
    />
  );
}

/** Count leaf entries (items with an ``href`` and no further nesting)
 *  reachable from the given tree.  Drives the "{N} of {M} match" hint
 *  shown above the filter input. */
function countLeaves(items: SidebarItem[]): number {
  let n = 0;
  for (const it of items) {
    if (it.separator) continue;
    if (it.href && (!it.items || it.items.length === 0)) {
      n += 1;
    } else if (it.items) {
      n += countLeaves(it.items);
    }
  }
  return n;
}

/** Recursive ancestor-expand probe: is any leaf in this subtree the
 *  current page?  Used by ``SidebarGroup`` to auto-open EVERY group on
 *  the path to the active leaf — without recursion, only the
 *  immediate-parent group expands, which leaves deeply-nested entries
 *  hidden from the user.  Honours both real hrefs and intermediate
 *  groups (which have no href of their own). */
function hasActiveDescendant(item: SidebarItem, pathname: string): boolean {
  if (item.href && (pathname === item.href || pathname.startsWith(`${item.href}/`))) {
    return true;
  }
  return item.items?.some((child) => hasActiveDescendant(child, pathname)) ?? false;
}

function SidebarGroup({ item, depth = 0, forceExpand = false }: SidebarGroupProps) {
  const pathname = usePathname();
  const isActive = item.href ? pathname === item.href || pathname.startsWith(`${item.href}/`) : false;
  const hasChildren = Boolean(item.items?.length);

  const isChildActive = React.useMemo(() => {
    if (!item.items) return false;
    return item.items.some((child) => hasActiveDescendant(child, pathname));
  }, [item.items, pathname]);

  const [userExpanded, setExpanded] = React.useState(isActive || isChildActive);
  const expanded = forceExpand || userExpanded;

  React.useEffect(() => {
    if (isChildActive) setExpanded(true);
  }, [isChildActive]);

  if (!hasChildren && item.href) {
    return (
      <Link
        href={item.href}
        className={cn(
          "group flex items-center justify-between rounded-md px-2.5 py-1.5 text-sm transition-colors",
          depth === 0 ? "font-medium" : "font-normal",
          isActive
            ? "bg-lucid-primary/10 text-lucid-primary"
            : "text-lucid-text-low hover:bg-lucid-surface hover:text-lucid-text-mid",
        )}
        aria-current={isActive ? "page" : undefined}
        style={{ paddingLeft: `${0.625 + depth * 0.625}rem` }}
      >
        <TitleWithTag title={item.title} tag={item.tag} />
        {item.badge && (
          <KindBadge badge={item.badge} depth={depth} active={isActive} />
        )}
      </Link>
    );
  }

  return (
    <div>
      {/* Row: navigable link + separate expand chevron */}
      <div
        className={cn(
          "group flex items-center rounded-md text-sm transition-colors",
          depth === 0 ? "font-medium" : "font-normal",
          isActive || isChildActive
            ? "text-lucid-text-high"
            : "text-lucid-text-low hover:text-lucid-text-mid",
        )}
      >
        {/* Title area — navigates if href exists */}
        {item.href ? (
          <Link
            href={item.href}
            onClick={() => setExpanded((v) => !v)}
            className={cn(
              "flex flex-1 items-center gap-1.5 rounded-md px-2.5 py-1.5 transition-colors min-w-0",
              isActive || isChildActive
                ? "hover:bg-lucid-primary/10"
                : "hover:bg-lucid-surface",
            )}
            style={{ paddingLeft: `${0.625 + depth * 0.625}rem` }}
            aria-current={isActive ? "page" : undefined}
          >
            <TitleWithTag title={item.title} tag={item.tag} />
            {item.badge && (
              <KindBadge badge={item.badge} depth={depth} active={isActive || isChildActive} />
            )}
          </Link>
        ) : (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className={cn(
              "flex flex-1 items-center gap-1.5 rounded-md px-2.5 py-1.5 transition-colors min-w-0",
              isActive || isChildActive ? "hover:bg-lucid-primary/10" : "hover:bg-lucid-surface",
            )}
            style={{ paddingLeft: `${0.625 + depth * 0.625}rem` }}
          >
            <TitleWithTag title={item.title} tag={item.tag} />
            {item.badge && (
              <KindBadge badge={item.badge} depth={depth} active={isActive || isChildActive} />
            )}
          </button>
        )}

        {/* Chevron — always toggles expansion */}
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="shrink-0 rounded-md p-1.5 hover:bg-lucid-surface transition-colors"
          aria-expanded={expanded}
          aria-label={expanded ? "Collapse" : "Expand"}
        >
          <motion.span
            animate={{ rotate: expanded ? 90 : 0 }}
            transition={springs.micro}
            className="block"
          >
            <ChevronRight className="h-3.5 w-3.5 text-lucid-text-disabled" />
          </motion.span>
        </button>
      </div>

      <AnimatePresence initial={false}>
        {expanded && item.items && (
          <motion.div
            key="children"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ ...springs.smooth, opacity: { duration: 0.15 } }}
            className="overflow-hidden"
          >
            <div className="mt-0.5 space-y-0.5 border-l border-lucid-border ml-3 pl-0">
              {item.items.map((child, idx) =>
                child.separator ? (
                  <SidebarSeparator key={`sep-${idx}`} />
                ) : (
                  <SidebarGroup
                    key={child.href ?? child.title}
                    item={child}
                    depth={depth + 1}
                    forceExpand={forceExpand}
                  />
                ),
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface SidebarProps {
  items: SidebarItem[];
  className?: string;
  /** When true, drop the desktop-only ``hidden lg:flex`` / fixed-width
   *  rules so the same tree can be embedded in a mobile drawer.  The
   *  caller is responsible for sizing the parent container. */
  variant?: "desktop" | "mobile";
}

/** Recursive substring filter — keep an item if its own title (or its
 *  ``tag``) matches, OR any descendant matches.  When a descendant
 *  matches but the parent doesn't, the parent is retained as a
 *  scaffold (so the tree remains navigable down to the hit).
 *
 *  Empty query → identity (return the full tree).  Match is a
 *  case-insensitive substring on title + tag so users can search by
 *  the friendly alias ("Tensor Operations") OR the real Python slug
 *  ("lucid.ops"). */
function filterSidebarItems(items: SidebarItem[], query: string): SidebarItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return items;

  function visit(item: SidebarItem): SidebarItem | null {
    if (item.separator) return null;
    const selfMatch =
      item.title.toLowerCase().includes(q) ||
      (item.tag?.toLowerCase().includes(q) ?? false) ||
      (item.href?.toLowerCase().includes(q) ?? false);
    const children = item.items?.map(visit).filter((c): c is SidebarItem => c !== null) ?? [];
    if (selfMatch || children.length > 0) {
      return { ...item, items: children.length > 0 ? children : item.items };
    }
    return null;
  }

  return items.map(visit).filter((i): i is SidebarItem => i !== null);
}

/** Center ``el`` in the sidebar's own scroll area — and nowhere else.
 *  ``el.scrollIntoView()`` scrolls *every* scrollable ancestor, the document
 *  included, so revealing the active entry also moved the page (about 48 px on
 *  every page open) and, run a frame after hydration, cancelled the scroll the
 *  browser had just started for a ``#fragment`` URL: a copied section link
 *  opened at the top of the page, 20 000 px short of its heading (LCD-1001). */
function centerInScrollArea(el: HTMLElement): void {
  const viewport = el.closest<HTMLElement>("[data-radix-scroll-area-viewport]");
  if (!viewport) return;
  const area = viewport.getBoundingClientRect();
  const item = el.getBoundingClientRect();
  viewport.scrollTop += item.top - area.top - (area.height - item.height) / 2;
}

export function Sidebar({ items, className, variant = "desktop" }: SidebarProps) {
  const isMobile = variant === "mobile";
  const [query, setQuery] = React.useState("");
  const filtered = React.useMemo(() => filterSidebarItems(items, query), [items, query]);
  const totalShown = countLeaves(filtered);
  const totalAll = countLeaves(items);
  const pathname = usePathname();
  const asideRef = React.useRef<HTMLElement>(null);

  // When the route changes (direct URL hit, search-dialog navigation,
  // breadcrumb-sibling jump, …) the active sidebar entry can end up
  // off-screen below the fold — especially in deep sub-trees like
  // ``lucid.nn / Modules / Conv2d``.  Scroll the active anchor into
  // the viewport's middle so users see where they are without manual
  // scrolling.  ``aria-current="page"`` is already stamped on every
  // active Link, which gives us a single selector that works for both
  // leaf items and group-with-href items.
  React.useEffect(() => {
    if (!asideRef.current) return;
    // Defer one frame so the freshly-mounted tree has its
    // ``aria-current`` attribute committed; otherwise the query runs
    // against last render's DOM.
    const id = requestAnimationFrame(() => {
      const active = asideRef.current?.querySelector<HTMLElement>(
        '[aria-current="page"]',
      );
      if (!active) return;
      // Not ``scrollIntoView`` — see ``centerInScrollArea``.
      centerInScrollArea(active);
    });
    return () => cancelAnimationFrame(id);
  }, [pathname, items]);

  return (
    <aside
      ref={asideRef}
      className={cn(
        isMobile ? "flex flex-col w-full" : "hidden lg:flex flex-col w-80 shrink-0",
        className,
      )}
      aria-label="Documentation navigation"
    >
      {/* Local filter — narrows the tree without making a remote call;
          backed by ``filterSidebarItems``.  Hidden on mobile where the
          drawer is already short and the keyboard would shrink the
          usable area further. */}
      <div className={cn(
        isMobile ? "hidden" : "flex items-center gap-2 pt-6 pr-4 pb-2",
      )}>
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-lucid-text-disabled" aria-hidden />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter sidebar…"
            aria-label="Filter sidebar items"
            // Suppress the browser's native ``×`` clear button on
            // ``type=search`` inputs — both WebKit and Edge render
            // their own circle-X that collided with our custom clear
            // button to the right.  Keeping ``type=search`` for its
            // semantic (Esc-to-clear behaviour, screen-reader role)
            // while hiding the native widget is the standard fix.
            className={cn(
              "w-full rounded-md border border-lucid-border bg-lucid-surface/40",
              "pl-8 pr-7 py-1.5 text-xs text-lucid-text-high placeholder:text-lucid-text-disabled",
              "focus:outline-none focus:border-lucid-primary/50 focus:bg-lucid-surface",
              "transition-colors",
              "[&::-webkit-search-cancel-button]:appearance-none",
              "[&::-webkit-search-decoration]:appearance-none",
            )}
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              aria-label="Clear filter"
              className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-lucid-text-disabled hover:text-lucid-text-mid"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
      {!isMobile && query && (
        <div className="px-1 pr-4 pb-2 text-[10px] font-mono text-lucid-text-disabled">
          {totalShown} of {totalAll} match
        </div>
      )}

      <ScrollArea className={cn("flex-1", isMobile ? "pt-2 pb-6 pr-2" : "pb-6 pr-4")}>
        <nav className="space-y-0.5">
          {filtered.map((item, idx) =>
            item.separator ? (
              <SidebarSeparator key={`sep-${idx}`} />
            ) : (
              <SidebarGroup key={item.href ?? item.title} item={item} forceExpand={!!query} />
            ),
          )}
        </nav>
        {!isMobile && query && filtered.length === 0 && (
          <p className="px-2 py-6 text-xs text-lucid-text-low">
            No sidebar entries match{" "}
            <code className="font-mono text-lucid-text-mid">
              &ldquo;{query}&rdquo;
            </code>.
          </p>
        )}
      </ScrollArea>
    </aside>
  );
}
