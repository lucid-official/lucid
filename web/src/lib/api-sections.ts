/**
 * Package sections — how a package with many sub-packages is grouped in the
 * sidebar and on its overview page.
 *
 * The grouping lives in ``web/data/api-sections.json`` (one source, read here,
 * by ``build-api-data.py`` which stamps each listed package's label and paper
 * onto its page data, and by the completeness test).  A package that is not
 * listed keeps the default: its sub-packages as one alphabetical run.
 *
 * Two kinds of section:
 *   - ``packages`` — a group of sub-package pages (``LoRA family`` → LoRA,
 *     AdaLoRA).  Each entry carries the label the site shows for the slug.
 *   - ``members`` — a slice of the package page's own classes and functions,
 *     either by source-file subcategory + kind, or ``rest`` (everything no
 *     other section claimed, so nothing the page documents can fall out of
 *     the sidebar when a name is added).
 */

import raw from "../../data/api-sections.json";

export interface SectionPackage {
  slug: string;
  label: string;
  /** Paper reference, shown on the package's page and card.  Optional —
   *  a method with no paper (trainable tokens) lists none. */
  citation?: string;
}

export interface MemberSelector {
  /** First path segment of a member's ``subcategory`` (``"config"``). */
  subcategories?: string[];
  /** Restrict to one member kind. */
  kind?: "class" | "function";
  /** Take every member no other section selected. */
  rest?: boolean;
  /** List the members as one flat run in the sidebar instead of bucketing
   *  them by source file. */
  flat?: boolean;
}

export interface ApiSection {
  id: string;
  title: string;
  /** One or two sentences under the section heading; inline markdown. */
  summary?: string;
  packages?: SectionPackage[];
  members?: MemberSelector;
}

export interface GuideLink {
  title: string;
  /** A docs-site path (``/docs/tutorials/lora-fine-tuning``). */
  href: string;
  summary?: string;
}

export interface PackageSections {
  /** The sub-package page that lists all of this package's sections as
   *  cards (``lucid.peft.tuners``).  The sidebar links it as "Overview". */
  overview?: string;
  /** Hand-written pages about the package, linked from its API page. */
  guides?: GuideLink[];
  sections: ApiSection[];
}

export const API_SECTIONS: Record<string, PackageSections> = raw as Record<
  string,
  PackageSections
>;

/** The ``packages`` entry for a slug, wherever it is listed. */
export function sectionPackage(slug: string): SectionPackage | undefined {
  for (const parent of Object.values(API_SECTIONS)) {
    for (const section of parent.sections) {
      const hit = section.packages?.find((p) => p.slug === slug);
      if (hit) return hit;
    }
  }
  return undefined;
}

/** The sections of whichever package lists ``slug`` as its overview page. */
export function sectionsForOverview(slug: string): ApiSection[] | undefined {
  for (const parent of Object.values(API_SECTIONS)) {
    if (parent.overview === slug) return parent.sections;
  }
  return undefined;
}

/** The guides of a sectioned package, from its own page or its overview page. */
export function guidesFor(slug: string): GuideLink[] {
  for (const [parent, plan] of Object.entries(API_SECTIONS)) {
    if (parent === slug || plan.overview === slug) return plan.guides ?? [];
  }
  return [];
}

/** All slugs listed under any section, in section order. */
export function sectionedSlugs(): string[] {
  return Object.values(API_SECTIONS).flatMap((parent) =>
    parent.sections.flatMap((s) => (s.packages ?? []).map((p) => p.slug)),
  );
}

/** The members a section selects, by section id.
 *
 *  Sections with an explicit selector claim first, in order (a member goes to
 *  the first section that selects it); ``rest`` sections take what is left.
 *  So every member lands in exactly one section — adding a public name can
 *  grow a section but never drop out of the sidebar or the overview. */
export function splitMembers<
  T extends { kind: string; subcategory?: string | null },
>(members: T[], sections: ApiSection[]): Map<string, T[]> {
  const claimed = new Set<T>();
  const picked = new Map<string, T[]>();
  for (const section of sections) {
    const sel = section.members;
    if (!sel || sel.rest) continue;
    const subs = new Set(sel.subcategories ?? []);
    const hit = members.filter(
      (m) =>
        !claimed.has(m) &&
        (subs.size === 0 || subs.has((m.subcategory ?? "").split("/")[0])) &&
        (!sel.kind || m.kind === sel.kind),
    );
    hit.forEach((m) => claimed.add(m));
    picked.set(section.id, hit);
  }
  for (const section of sections) {
    if (section.members?.rest) {
      picked.set(section.id, members.filter((m) => !claimed.has(m)));
    }
  }
  return picked;
}
