import Link from "next/link";
import { SubsectionHeading } from "@/components/ui/SubsectionHeading";
import { ExternalLink, Pencil } from "lucide-react";
import { sourceToEditUrl } from "@/lib/github-edit";
import { highlight } from "@/lib/shiki";
import { TypeAnnotation } from "./TypeAnnotation";
import { ParameterTable, AttributeTable, RaisesTable } from "./ParameterTable";
import { ExampleBlock } from "./ExampleBlock";
import { SeeAlsoBlock } from "./SeeAlsoBlock";
import { AnchorLink } from "./AnchorLink";
import { CollapsibleSection } from "./CollapsibleSection";
import { UsedByBlock } from "./UsedByBlock";
import { FunctionSignature } from "./FunctionSignature";
import { ClassBadge, AutoKindBadge } from "./ApiKindBadge";
import { getClassNameColor, getClassHoverBorder } from "@/lib/api-kind-utils";
import { MathText } from "./MathText";
import { CrossLinkPanel } from "./CrossLinkPanel";
import { groupMethods } from "@/lib/method-groups";
import type { ApiClass, ApiClassKind } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ClassDocProps {
  cls: ApiClass;
  /** Module slug this class lives in.  Required at top level so the
   *  cross-link panel can resolve the right py↔cpp mapping; omitted on
   *  nested renders (e.g. methods inside another class) to suppress the
   *  panel where it isn't meaningful. */
  moduleSlug?: string;
  /** If provided, only render this method. Otherwise render the full class. */
  methodName?: string;
}

export async function ClassDoc({ cls, moduleSlug, methodName }: ClassDocProps) {
  if (methodName) {
    const method = cls.methods.find((m) => m.name === methodName);
    if (!method) return (
      <p className="text-lucid-text-low">Method <code>{methodName}</code> not found.</p>
    );
    return <FunctionSignature fn={method} headingLevel="h2" />;
  }

  const sigHtml = cls.signature ? await highlight(cls.signature, "python") : null;
  const clsKind: ApiClassKind = cls.class_kind ?? "regular";
  const nameColor = getClassNameColor(clsKind);

  // Bucketed method groups — handed to the renderer below.  Empty
  // groups are dropped by ``groupMethods``; the platform (Python vs
  // C++) is auto-detected from the labels.  Each group becomes its own
  // ``<h2>`` so the right-rail ToC surfaces them as scroll targets.
  const groups = groupMethods(cls.methods);
  const totalMethods = groups.reduce((sum, g) => sum + g.methods.length, 0);

  return (
    <article>
      {/* Class header */}
      <div className="mb-8 rounded-2xl border border-lucid-border bg-lucid-surface overflow-hidden">
        <div className="flex items-start justify-between gap-4 border-b border-lucid-border px-5 py-5">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <ClassBadge kind={clsKind} />
              <h1 className={cn("font-mono text-2xl font-bold", nameColor)}>
                {cls.name}
              </h1>
            </div>
            {/* Base classes */}
            {cls.bases && cls.bases.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <span className="text-[11px] text-lucid-text-disabled tracking-wide">
                  extends
                </span>
                {cls.bases.map((base) => (
                  <code
                    key={base}
                    className="text-[11px] font-mono text-lucid-text-low bg-lucid-elevated px-1.5 py-0.5 rounded border border-lucid-border"
                  >
                    {base}
                  </code>
                ))}
              </div>
            )}
            {sigHtml && (
              <div
                className={cn(
                  "text-xs rounded-lg border border-lucid-border bg-lucid-elevated",
                  "[&_pre]:px-3 [&_pre]:py-2.5 [&_pre]:overflow-x-auto [&_pre]:!bg-transparent",
                  "[&_code]:font-mono [&_code]:text-xs [&_code]:leading-relaxed",
                )}
                dangerouslySetInnerHTML={{ __html: sigHtml }}
              />
            )}
          </div>
          {cls.source && (
            <div className="shrink-0 flex items-center gap-1.5">
              <a
                href={cls.source}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-lucid-text-disabled hover:text-lucid-text-mid transition-colors border border-lucid-border hover:border-lucid-primary/40"
                aria-label="View source on GitHub"
              >
                <ExternalLink className="h-3 w-3" />
                <span>source</span>
              </a>
              {sourceToEditUrl(cls.source) && (
                <a
                  href={sourceToEditUrl(cls.source) ?? "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-lucid-text-disabled hover:text-lucid-primary transition-colors border border-lucid-border hover:border-lucid-primary/40"
                  aria-label="Edit this docstring on GitHub"
                  title="Edit on GitHub — proposes a PR after you save"
                >
                  <Pencil className="h-3 w-3" />
                  <span className="hidden sm:inline">edit</span>
                </a>
              )}
            </div>
          )}
        </div>

        <div className="px-5 py-5 space-y-6">
          {moduleSlug && (
            <CrossLinkPanel moduleSlug={moduleSlug} memberName={cls.name} />
          )}
          {cls.summary && (
            <MathText text={cls.summary} block className="text-sm text-lucid-text-mid leading-relaxed" />
          )}
          {cls.extended && (
            <MathText text={cls.extended} block className="text-sm text-lucid-text-mid leading-relaxed" />
          )}
          {cls.parameters.length > 0 && <ParameterTable parameters={cls.parameters} />}
          {cls.attributes.length > 0 && <AttributeTable attributes={cls.attributes} />}
          {cls.raises.length > 0 && <RaisesTable raises={cls.raises} />}
          {cls.notes.length > 0 && (
            <section className="space-y-2">
              <SubsectionHeading>
                Notes
              </SubsectionHeading>
              <div className="rounded-xl border-l-2 border-lucid-blue bg-lucid-blue/5 px-4 py-3 space-y-2">
                {cls.notes.map((note, i) => (
                  <MathText key={i} text={note} block className="text-sm text-lucid-text-mid leading-relaxed" />
                ))}
              </div>
            </section>
          )}
          {cls.examples.length > 0 && <ExampleBlock examples={cls.examples} />}
          {cls.see_also && cls.see_also.length > 0 && (
            <SeeAlsoBlock items={cls.see_also} />
          )}
          {/* See FunctionSignature for the placement rationale — Used-by
              sits at the end of the docstring body so it doesn't compete
              with the symbol's own narrative. */}
          <UsedByBlock path={cls.path} />
        </div>
      </div>

      {/* Methods — split into ``Constructors`` / ``Properties`` /
          ``Class methods`` / ``Instance methods`` / ``In-place ops`` /
          ``Dunder methods`` (Python) or ``Constructors`` / ``Static`` /
          ``Operators`` / ``Virtual`` / ``Methods`` (C++).
          Each group gets its own ``<h2 id>`` so the right-rail ToC
          surfaces them as scroll targets. */}
      {totalMethods > 0 && (
        <div className="space-y-10">
          {groups.map((group) => (
            <CollapsibleSection
              key={group.id}
              id={group.id}
              // Auto-collapse very-large default groups so the user
              // sees a structural overview first.  Five was picked
              // by feel — under it, expanded reads naturally; over
              // it, the scroll cost dominates the page.
              defaultCollapsed={group.methods.length > 20}
              header={
                <>
                  <h2
                    id={group.id}
                    className="scroll-mt-24 text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase"
                  >
                    {group.label}
                  </h2>
                  <span className="font-mono text-[10px] text-lucid-text-disabled">
                    {group.methods.length}
                  </span>
                  <AnchorLink id={group.id} />
                </>
              }
            >
              <div className="space-y-4">
                {group.methods.map((method) => (
                  <FunctionSignature
                    key={method.name}
                    fn={method}
                    headingLevel="h3"
                  />
                ))}
              </div>
            </CollapsibleSection>
          ))}
        </div>
      )}
    </article>
  );
}

/** Compact card for module overview */
interface ClassCardProps {
  cls: ApiClass;
  moduleSlug: string;
}

export function ClassCard({ cls, moduleSlug }: ClassCardProps) {
  const clsKind: ApiClassKind = cls.class_kind ?? "regular";
  const nameColor = getClassNameColor(clsKind);
  const hoverBorder = getClassHoverBorder(clsKind);

  return (
    <Link
      href={`/api/${moduleSlug}/${cls.name}`}
      // See FunctionCard for why the compact-mode hooks are plain
      // classes backed by globals.css rules instead of Tailwind
      // arbitrary variants with nested attribute selectors.
      className={cn(
        "compact-card group flex items-start justify-between gap-4",
        "rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3.5",
        "transition-all hover:bg-lucid-elevated",
        hoverBorder,
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="compact-card-row flex flex-wrap items-center gap-2 mb-1">
          <ClassBadge kind={clsKind} />
          <code className={cn("text-sm font-mono font-medium", nameColor)}>
            {cls.name}
          </code>
          {cls.methods.length > 0 && (
            <span className="text-[10px] text-lucid-text-disabled">
              {cls.methods.length} methods
            </span>
          )}
        </div>
        {cls.summary && (
          <p className="compact-card-summary text-xs text-lucid-text-low leading-relaxed line-clamp-2">
            <MathText text={cls.summary} className="text-xs" inLink />
          </p>
        )}
      </div>
    </Link>
  );
}
