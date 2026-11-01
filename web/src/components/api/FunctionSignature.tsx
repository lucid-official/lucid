import Link from "next/link";
import { SubsectionHeading } from "@/components/ui/SubsectionHeading";
import { ExternalLink, Pencil } from "lucide-react";
import { sourceToEditUrl } from "@/lib/github-edit";
import { highlight } from "@/lib/shiki";
import { TypeAnnotation } from "./TypeAnnotation";
import { ParameterTable, RaisesTable } from "./ParameterTable";
import { ExampleBlock } from "./ExampleBlock";
import { SeeAlsoBlock } from "./SeeAlsoBlock";
import { AnchorLink } from "./AnchorLink";
import { UsedByBlock } from "./UsedByBlock";
import { linkifyTypesInHtml } from "@/lib/type-links";
import { AutoKindBadge, AutoKindBadgeRow } from "./ApiKindBadge";
import { getMemberNameColor } from "@/lib/api-kind-utils";
import { MathText } from "./MathText";
import { CrossLinkPanel } from "./CrossLinkPanel";
import type { ApiFunction, ApiMethod } from "@/lib/types";
import { cn, formatCompactCount } from "@/lib/utils";
import { ModelSizeCard } from "./ModelSizeCard";


/** Inline pill rendering a factory function's paper-cited model size
 *  ("61.1M" / "1.4B" / "234K").  Shared by FunctionCard and
 *  FunctionSignature so the wording / colour stay in lockstep. */
function ParamCountPill({ count }: { count: number }) {
  return (
    <span
      title={`${count.toLocaleString()} parameters`}
      className={cn(
        "inline-flex items-center rounded-md border px-1.5 py-0.5",
        "text-[10px] font-mono font-medium leading-snug",
        "bg-lucid-text-low/10 text-lucid-text-mid border-lucid-text-low/30",
      )}
    >
      {formatCompactCount(count)}
    </span>
  );
}

interface FunctionSignatureProps {
  fn: ApiFunction | ApiMethod;
  headingLevel?: "h2" | "h3";
  className?: string;
  /** If true, show the full doc (params, examples). Otherwise show compact card. */
  expanded?: boolean;
  /** Module slug this function lives in.  Supplied at top-level detail
   *  pages so the cross-link panel can resolve the py↔cpp mapping;
   *  omitted for nested method renderings (where the panel isn't
   *  meaningful — methods don't have backward nodes of their own). */
  moduleSlug?: string;
}

export async function FunctionSignature({
  fn,
  headingLevel: Tag = "h3",
  className,
  expanded = true,
  moduleSlug,
}: FunctionSignatureProps) {
  const sigHtmlRaw = fn.signature ? await highlight(fn.signature, "python") : null;
  // Wrap recognised type names (``Tensor`` / ``Module`` / …) with
  // hyperlinks to their docs page so users can jump straight from a
  // signature to the underlying type without a sidebar detour.
  const sigHtml = sigHtmlRaw ? linkifyTypesInHtml(sigHtmlRaw) : null;
  const labels = fn.labels ?? [];
  const nameColor = getMemberNameColor(fn.name, labels);

  return (
    <article
      className={cn(
        "group rounded-2xl border border-lucid-border bg-lucid-surface",
        "overflow-hidden",
        className,
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-4 border-b border-lucid-border px-5 py-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <AutoKindBadgeRow labels={labels} name={fn.name} fallback="fn" />
            {/* Heading owns the anchor + scroll offset so #{name} links
                land below the sticky header AND the ToC scroll-spy can
                pick it up via the standard ``[id]`` query. */}
            <Tag
              id={fn.name}
              className={cn(
                "scroll-mt-24 font-mono text-base font-semibold",
                nameColor,
              )}
            >
              {fn.name}
            </Tag>
            <AnchorLink id={fn.name} />
            {fn.returns?.annotation && (
              <>
                <span className="text-lucid-text-disabled text-sm">→</span>
                <TypeAnnotation annotation={fn.returns.annotation} />
              </>
            )}
            {"param_count" in fn && fn.param_count !== undefined && (
              <ParamCountPill count={fn.param_count} />
            )}
            {"inherited_from" in fn && fn.inherited_from && (
              <span className="text-[11px] text-lucid-text-disabled">
                from{" "}
                <code className="font-mono text-lucid-text-low">{fn.inherited_from}</code>
              </span>
            )}
          </div>

          {/* Signature */}
          {sigHtml && (
            <div
              className={cn(
                "text-xs rounded-lg border border-lucid-border bg-lucid-elevated mt-2",
                "[&_pre]:px-3 [&_pre]:py-2 [&_pre]:overflow-x-auto [&_pre]:!bg-transparent",
                "[&_code]:font-mono [&_code]:text-xs [&_code]:leading-relaxed",
              )}
              dangerouslySetInnerHTML={{ __html: sigHtml }}
            />
          )}
        </div>

        {fn.source && (
          <div className="shrink-0 flex items-center gap-1.5">
            <a
              href={fn.source}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-lucid-text-disabled hover:text-lucid-text-mid transition-colors border border-lucid-border hover:border-lucid-primary/40"
              aria-label="View source on GitHub"
            >
              <ExternalLink className="h-3 w-3" />
              <span>source</span>
            </a>
            {sourceToEditUrl(fn.source) && (
              <a
                href={sourceToEditUrl(fn.source) ?? "#"}
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

      {/* Body */}
      {expanded && (
        <div className="px-5 py-5 space-y-6">
          {moduleSlug && (
            <CrossLinkPanel moduleSlug={moduleSlug} memberName={fn.name} />
          )}
          {fn.summary && (
            <MathText text={fn.summary} block className="text-sm text-lucid-text-mid leading-relaxed" />
          )}
          {fn.extended && (
            <MathText text={fn.extended} block className="text-sm text-lucid-text-mid leading-relaxed" />
          )}

          {fn.kind === "function" && (("param_count" in fn && fn.param_count !== undefined) || (("model_summary" in fn) && fn.model_summary)) && (
            <ModelSizeCard
              paramCount={(fn as ApiFunction).param_count}
              summary={(fn as ApiFunction).model_summary}
            />
          )}

          {fn.parameters.length > 0 && <ParameterTable parameters={fn.parameters} />}

          {fn.returns && fn.returns.description && (
            <section className="space-y-2">
              <SubsectionHeading>
                Returns
              </SubsectionHeading>
              <div className="rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3">
                <TypeAnnotation annotation={fn.returns.annotation} className="block mb-1" />
                <MathText text={fn.returns.description} block className="text-sm text-lucid-text-mid" />
              </div>
            </section>
          )}

          {fn.raises.length > 0 && <RaisesTable raises={fn.raises} />}

          {fn.notes.length > 0 && (
            <section className="space-y-2">
              <SubsectionHeading>
                Notes
              </SubsectionHeading>
              <div className="rounded-xl border-l-2 border-lucid-blue bg-lucid-blue/5 px-4 py-3">
                {fn.notes.map((note, i) => (
                  <MathText key={i} text={note} block className="text-sm text-lucid-text-mid leading-relaxed" />
                ))}
              </div>
            </section>
          )}

          {fn.examples.length > 0 && <ExampleBlock examples={fn.examples} />}

          {fn.see_also && fn.see_also.length > 0 && (
            <SeeAlsoBlock items={fn.see_also} />
          )}

          {/* "Used by" backlinks live last because they're orientational
              — readers usually want the symbol's own signature / examples
              first, and only after understanding the function does
              "where else is this called?" become a useful follow-up. */}
          <UsedByBlock path={fn.path} />
        </div>
      )}
    </article>
  );
}

/** Compact one-line card for module overview lists */
interface FunctionCardProps {
  fn: ApiFunction;
  moduleSlug: string;
}

export function FunctionCard({ fn, moduleSlug }: FunctionCardProps) {
  const labels = fn.labels ?? [];
  const nameColor = getMemberNameColor(fn.name, labels);

  return (
    <Link
      href={`/api/${moduleSlug}/${fn.name}`}
      // Compact-mode hooks are plain ``compact-card`` /
      // ``compact-card-row`` / ``compact-card-summary`` classes whose
      // CSS rules live in ``globals.css`` under the
      // ``article[data-view="compact"]`` descendant selector.  Tried
      // arbitrary Tailwind variants (``[article[data-view=compact]_&]:``)
      // first and the nested attribute brackets parsed ambiguously —
      // some rules leaked and shrank the page body width.  Explicit
      // CSS is the reliable fix.
      className={cn(
        "compact-card group flex items-start justify-between gap-4",
        "rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3.5",
        "transition-all hover:border-api-fn/40 hover:bg-lucid-elevated",
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="compact-card-row flex flex-wrap items-center gap-2 mb-1">
          <AutoKindBadge labels={labels} name={fn.name} fallback="fn" />
          <code className={cn("text-sm font-mono font-medium", nameColor)}>
            {fn.name}
          </code>
          {fn.returns?.annotation && (
            <span className="text-[11px] font-mono text-lucid-text-disabled">
              → {fn.returns.annotation}
            </span>
          )}
          {fn.param_count !== undefined && (
            <ParamCountPill count={fn.param_count} />
          )}
        </div>
        {fn.summary && (
          <p className="compact-card-summary text-xs text-lucid-text-low leading-relaxed line-clamp-2">
            <MathText text={fn.summary} className="text-xs" inLink />
          </p>
        )}
      </div>
    </Link>
  );
}
