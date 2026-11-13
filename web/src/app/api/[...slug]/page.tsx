import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getAllModuleSlugs, loadApiData, findMember } from "@/lib/api-loader";
import { isApiClass, isApiFunction, isApiModule, isApiClassModule } from "@/lib/types";
import { ModuleOverview } from "@/components/api/ModuleOverview";
import { CppEngineOverview } from "@/components/api/CppEngineOverview";
import { ClassDoc } from "@/components/api/ClassDoc";
import { FunctionSignature } from "@/components/api/FunctionSignature";
import { PageTableOfContents } from "@/components/layout/TableOfContents";
import { BreadcrumbStep, type BreadcrumbSibling } from "@/components/api/BreadcrumbStep";
import { canonical } from "@/lib/site";

// ---------------------------------------------------------------------------
// Static generation
// ---------------------------------------------------------------------------

export async function generateStaticParams(): Promise<{ slug: string[] }[]> {
  const slugs = getAllModuleSlugs();
  const params: { slug: string[] }[] = [];

  // Module overview pages: /api/lucid.fft, /api/lucid.nn, …
  for (const slug of slugs) {
    params.push({ slug: [slug] });
  }

  // Member detail pages: /api/lucid.nn/Linear, /api/lucid.fft/fft, …
  for (const slug of slugs) {
    try {
      const data = loadApiData(slug);
      if (isApiModule(data)) {
        for (const member of data.members) {
          params.push({ slug: [slug, member.name] });
        }
      } else if (isApiClassModule(data)) {
        for (const method of data.methods) {
          params.push({ slug: [slug, method.name] });
        }
      }
    } catch {
      // skip missing data
    }
  }

  return params;
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const [moduleSlug, memberName] = slug;

  try {
    const data = loadApiData(moduleSlug);
    const alternates = canonical(`/api/${slug.join("/")}`);
    if (memberName) {
      return {
        title: `${memberName} — ${data.name}`,
        description: `API reference for ${memberName} in ${data.path}`,
        alternates,
      };
    }
    return {
      title: data.name,
      description: data.summary ?? `API reference for ${data.path}`,
      alternates,
    };
  } catch {
    return { title: "API Reference" };
  }
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default async function ApiSlugPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const [moduleSlug, memberName] = slug;

  let data;
  try {
    data = loadApiData(moduleSlug);
  } catch {
    notFound();
  }

  // Memoise once per request — both detail branches need the same set
  // to build their breadcrumbs.
  const validSlugs = new Set(getAllModuleSlugs());

  // ── Member detail page ────────────────────────────────────────────────────
  if (memberName) {
    const breadcrumbParts = buildBreadcrumb(moduleSlug, memberName, validSlugs);

    // Class-module (Tensor) method
    if (isApiClassModule(data)) {
      const method = data.methods.find((m) => m.name === memberName);
      if (!method) notFound();
      return (
        <div className="flex flex-col gap-10 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 max-w-4xl xl:flex-1">
            <Breadcrumb parts={breadcrumbParts} />
            <FunctionSignature
              fn={method}
              headingLevel="h2"
              className="mt-6"
              moduleSlug={moduleSlug}
            />
          </div>
          <PageTableOfContents />
        </div>
      );
    }

    // Regular module member
    if (isApiModule(data)) {
      const member = findMember(data, memberName);
      if (!member) notFound();

      return (
        <div className="flex flex-col gap-10 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 max-w-4xl xl:flex-1">
            <Breadcrumb parts={breadcrumbParts} />
            <div className="mt-6">
              {isApiClass(member) ? (
                <ClassDoc cls={member} moduleSlug={moduleSlug} />
              ) : isApiFunction(member) ? (
                <FunctionSignature fn={member} headingLevel="h2" moduleSlug={moduleSlug} />
              ) : null}
            </div>
          </div>
          <PageTableOfContents />
        </div>
      );
    }

    notFound();
  }

  // ── Module overview page ───────────────────────────────────────────────────
  // The C++ engine module gets a curated landing treatment — 813 member
  // cards would be unscannable, so ``CppEngineOverview`` surfaces only the
  // top-level surfaces (Core / Tensor / Backends / Ops / NN / Autograd / …)
  // and relies on the sidebar tree for the full drill-down.
  if (moduleSlug === "lucid._C.engine" && isApiModule(data)) {
    return (
      <div className="max-w-4xl">
        <CppEngineOverview data={data} />
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-10 xl:flex-row xl:items-start xl:justify-between">
      <div className="min-w-0 max-w-4xl xl:flex-1">
        <ModuleOverview data={data} />
      </div>
      <PageTableOfContents minEntries={2} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Breadcrumb
// ---------------------------------------------------------------------------

interface BreadcrumbPart {
  label: string;
  href?: string;
  /** Slugs at the same hierarchy level as this step — populated by
   *  ``buildBreadcrumb`` and rendered as a dropdown for fast lateral
   *  navigation.  Empty for the trailing member step (a function /
   *  class name has no siblings in this sense). */
  siblings: BreadcrumbSibling[];
  /** Slug for this step itself — used by the renderer to flag it as
   *  ``active`` inside the sibling list. */
  selfSlug?: string;
}

/** Friendly label overrides for slugs whose raw last-segment doesn't
 *  read well on its own (e.g. ``engine`` → ``C++ Engine``).  Only the
 *  step matching the FULL slug gets renamed; intermediate segments
 *  stay as-is.  Layered on top of the default ``segs[i]`` label in
 *  ``buildBreadcrumb``. */
const BREADCRUMB_LABEL_OVERRIDES: Record<string, string> = {
  "lucid._C.engine": "C++ Engine",
};

/** Build a multi-step breadcrumb from a slug like ``lucid.nn.functional``.
 *
 *  Each dot-separated prefix that has its own docs page becomes a
 *  clickable step ( ``lucid`` → ``lucid.nn`` → ``lucid.nn.functional`` );
 *  prefixes that aren't documented (e.g. the synthetic ``lucid._C``
 *  intermediate above ``lucid._C.engine``) are skipped to avoid dead
 *  links.  When the page renders a specific member (class / function),
 *  the member name is appended as the final non-link step. */
/** Siblings of ``prefix`` = every valid slug that shares its parent
 *  prefix and matches its depth.  ``lucid.nn`` is at depth 2 under
 *  ``lucid``, so siblings are every other depth-2 slug under
 *  ``lucid`` (``lucid.amp``, ``lucid.optim``, …).  The active prefix
 *  itself is included so the dropdown can highlight it. */
function _siblingsOf(
  prefix: string,
  validSlugs: Set<string>,
): BreadcrumbSibling[] {
  const segs = prefix.split(".");
  if (segs.length === 0) return [];
  const parent = segs.slice(0, -1).join(".");
  const targetDepth = segs.length;
  const out: BreadcrumbSibling[] = [];
  for (const candidate of validSlugs) {
    const cSegs = candidate.split(".");
    if (cSegs.length !== targetDepth) continue;
    const cParent = cSegs.slice(0, -1).join(".");
    if (cParent !== parent) continue;
    out.push({
      slug: candidate,
      label: BREADCRUMB_LABEL_OVERRIDES[candidate] ?? cSegs[cSegs.length - 1],
      active: candidate === prefix,
    });
  }
  out.sort((a, b) => a.slug.localeCompare(b.slug));
  return out;
}

function buildBreadcrumb(
  moduleSlug: string,
  memberName: string | undefined,
  validSlugs: Set<string>,
): BreadcrumbPart[] {
  const segs = moduleSlug.split(".");
  const parts: BreadcrumbPart[] = [];
  for (let i = 0; i < segs.length; i++) {
    const prefix = segs.slice(0, i + 1).join(".");
    if (!validSlugs.has(prefix)) continue;
    const isFinalSlugStep = i === segs.length - 1;
    const siblings = _siblingsOf(prefix, validSlugs);
    parts.push({
      label: BREADCRUMB_LABEL_OVERRIDES[prefix] ?? segs[i],
      // The terminal slug step links to itself only when there's a
      // deeper member step after it — otherwise it's the active page
      // and shouldn't be a link.
      href: !isFinalSlugStep || memberName ? `/api/${prefix}` : undefined,
      // Only surface a sibling dropdown when there's actually another
      // peer to jump to — single-entry dropdowns are noise.
      siblings: siblings.length > 1 ? siblings : [],
      selfSlug: prefix,
    });
  }
  if (memberName) {
    parts.push({ label: memberName, siblings: [] });
  }
  return parts;
}

interface BreadcrumbProps {
  parts: BreadcrumbPart[];
}

function Breadcrumb({ parts }: BreadcrumbProps) {
  return (
    <nav className="flex flex-wrap items-center gap-1.5 text-sm text-lucid-text-low mb-2" aria-label="Breadcrumb">
      {parts.map((part, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {/* Separators sit BETWEEN parts; the first step renders bare. */}
          {i > 0 && <span className="text-lucid-text-disabled">/</span>}
          <BreadcrumbStep
            label={part.label}
            href={part.href}
            siblings={part.siblings}
          />
        </span>
      ))}
    </nav>
  );
}
