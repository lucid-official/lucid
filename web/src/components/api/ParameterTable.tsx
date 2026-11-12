import * as React from "react";
import type { DocstringParameter, DocstringAttribute, DocstringRaise } from "@/lib/types";
import { SubsectionHeading } from "@/components/ui/SubsectionHeading";
import { TypeAnnotation } from "./TypeAnnotation";
import { MathText } from "./MathText";
import { cn } from "@/lib/utils";

/** The name column of a row: twelve rem wide beside the description, the
 *  whole row on a phone.  A name wider than the column
 *  (``TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING``) has to wrap,
 *  or it is drawn over the description next to it, or clipped by the table's
 *  rounded edge. */
const NAME_COLUMN = "shrink-0 sm:w-48 [overflow-wrap:anywhere]";

/** An identifier that may break after each underscore, where a reader would cut
 *  it, so a long constant wraps into words and not into letters. */
function BreakableName({ name }: { name: string }) {
  return name.split(/(?<=_)/).map((part, i) => (
    <React.Fragment key={i}>
      {i > 0 && <wbr />}
      {part}
    </React.Fragment>
  ));
}

interface ParameterTableProps {
  parameters: DocstringParameter[];
  className?: string;
}

export function ParameterTable({ parameters, className }: ParameterTableProps) {
  if (!parameters.length) return null;

  return (
    <section className={cn("space-y-2", className)}>
      <SubsectionHeading>
        Parameters
      </SubsectionHeading>
      <div className="divide-y divide-lucid-border rounded-xl border border-lucid-border overflow-hidden">
        {parameters.map((p) => (
          <div key={p.name} className="flex flex-col sm:flex-row gap-1 sm:gap-4 px-4 py-3 bg-lucid-surface hover:bg-lucid-elevated transition-colors">
            <div className={cn(NAME_COLUMN, "flex flex-col gap-0.5")}>
              <code className="text-xs font-mono font-semibold text-lucid-text-high">
                <BreakableName name={p.name} />
              </code>
              {p.annotation && (
                <TypeAnnotation annotation={p.annotation} />
              )}
              {p.default !== null && (
                <code className="text-[10px] font-mono text-lucid-text-disabled">
                  = {p.default}
                </code>
              )}
            </div>
            <div className="text-sm text-lucid-text-mid leading-relaxed flex-1">
              {p.description
                ? <MathText text={p.description} />
                : <span className="italic opacity-50">No description.</span>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

interface AttributeTableProps {
  attributes: DocstringAttribute[];
  className?: string;
  /** Draw the "Attributes" heading above the table.  A class card needs it; a
   *  module page puts the table under its own "Attributes" section title, where
   *  a second heading would say the same word twice. */
  heading?: boolean;
}

export function AttributeTable({ attributes, className, heading = true }: AttributeTableProps) {
  if (!attributes.length) return null;

  return (
    <section className={cn("space-y-2", className)}>
      {heading && (
        <SubsectionHeading>
          Attributes
        </SubsectionHeading>
      )}
      <div className="divide-y divide-lucid-border rounded-xl border border-lucid-border overflow-hidden">
        {attributes.map((a) => (
          <div key={a.name} className="flex flex-col sm:flex-row gap-1 sm:gap-4 px-4 py-3 bg-lucid-surface hover:bg-lucid-elevated transition-colors">
            <div className={cn(NAME_COLUMN, "flex flex-col gap-0.5")}>
              <code className="text-xs font-mono font-semibold text-lucid-text-high">
                <BreakableName name={a.name} />
              </code>
              {a.annotation && <TypeAnnotation annotation={a.annotation} />}
            </div>
            <div className="text-sm text-lucid-text-mid leading-relaxed flex-1">
              <MathText text={a.description} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

interface RaisesTableProps {
  raises: DocstringRaise[];
  className?: string;
}

export function RaisesTable({ raises, className }: RaisesTableProps) {
  if (!raises.length) return null;

  return (
    <section className={cn("space-y-2", className)}>
      <SubsectionHeading>
        Raises
      </SubsectionHeading>
      <div className="divide-y divide-lucid-border rounded-xl border border-lucid-border overflow-hidden">
        {raises.map((r, i) => (
          <div key={i} className="flex flex-col sm:flex-row gap-1 sm:gap-4 px-4 py-3 bg-lucid-surface">
            <div className={NAME_COLUMN}>
              {r.annotation && (
                <code className="text-xs font-mono font-semibold text-lucid-error">
                  {r.annotation}
                </code>
              )}
            </div>
            <div className="text-sm text-lucid-text-mid leading-relaxed flex-1">
              <MathText text={r.description} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
