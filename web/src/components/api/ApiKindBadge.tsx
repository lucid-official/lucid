"use client";

/**
 * ApiKindBadge — visual kind indicators for API reference entries.
 *
 * Color source of truth: globals.css @theme block (--color-api-*)
 * To retheme, edit globals.css only — nothing here needs to change.
 */

import type { ApiLabel, ApiClassKind } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  getMemberNameColor as _getMemberNameColor,
  getClassNameColor as _getClassNameColor,
  getClassHoverBorder as _getClassHoverBorder,
} from "@/lib/api-kind-utils";

// Re-export so existing imports from this file still work
export { _getMemberNameColor as getMemberNameColor, _getClassNameColor as getClassNameColor, _getClassHoverBorder as getClassHoverBorder };

// ---------------------------------------------------------------------------
// Pill primitive
// ---------------------------------------------------------------------------

function Pill({ label, className }: { label: string; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 select-none items-center",
        "rounded px-1.5 py-0.5",
        "text-[9px] font-bold tracking-widest uppercase",
        "border",
        className,
      )}
    >
      {label}
    </span>
  );
}

// Color helpers are defined in @/lib/api-kind-utils (server-safe, no "use client")
// and re-exported above from this file for backwards compatibility.

// ---------------------------------------------------------------------------
// Named kind badges
// ---------------------------------------------------------------------------

/** Concrete class */
export function ClassBadge({
  kind = "regular",
  className,
}: {
  kind?: ApiClassKind;
  className?: string;
}) {
  if (kind === "abstract") {
    return (
      <Pill
        label="abstract"
        className={cn(
          "border-dashed bg-api-class-abstract/10 border-api-class-abstract/35 text-api-class-abstract",
          className,
        )}
      />
    );
  }
  if (kind === "dataclass") {
    return (
      <Pill
        label="data"
        className={cn(
          "bg-api-class-dataclass/10 border-api-class-dataclass/35 text-api-class-dataclass",
          className,
        )}
      />
    );
  }
  if (kind === "protocol") {
    return (
      <Pill
        label="proto"
        className={cn(
          // Dashed border echoes the "abstract" pill — Protocol is a
          // structural contract, not a concrete implementation — but
          // tinted with its own teal accent so the two are easy to
          // tell apart at a glance.
          "border-dashed bg-api-class-protocol/10 border-api-class-protocol/40 text-api-class-protocol",
          className,
        )}
      />
    );
  }
  return (
    <Pill
      label="class"
      className={cn(
        "bg-api-class/10 border-api-class/30 text-api-class",
        className,
      )}
    />
  );
}

/** Regular function / method */
export function FnBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="fn"
      className={cn("bg-api-fn/10 border-api-fn/30 text-api-fn", className)}
    />
  );
}

/** @property */
export function PropBadge({
  writable = false,
  className,
}: {
  writable?: boolean;
  className?: string;
}) {
  return (
    <Pill
      label={writable ? "prop rw" : "prop"}
      className={cn(
        "bg-api-prop/10 border-api-prop/30 text-api-prop",
        className,
      )}
    />
  );
}

/** A documented plain attribute (not a property) */
export function AttrBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="attr"
      className={cn(
        "bg-api-prop/10 border-api-prop/30 text-api-prop",
        className,
      )}
    />
  );
}

/** @classmethod */
export function ClsMethodBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="cls"
      className={cn("bg-api-cls/10 border-api-cls/30 text-api-cls", className)}
    />
  );
}

/** @staticmethod */
export function StaticBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="static"
      className={cn(
        "bg-api-static/10 border-api-static/30 text-api-static",
        className,
      )}
    />
  );
}

/** @abstractmethod */
export function AbstractBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="abstract"
      className={cn(
        "border-dashed bg-api-abstract/8 border-api-abstract/30 text-api-abstract",
        className,
      )}
    />
  );
}

/** __dunder__ methods */
export function DunderBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="dunder"
      className={cn(
        "bg-api-dunder/10 border-api-dunder/30 text-api-dunder",
        className,
      )}
    />
  );
}

// ---------------------------------------------------------------------------
// C++ method kind badges
// ---------------------------------------------------------------------------

/** C++ constructor (``ClassName(args...)``). */
export function CppCtorBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="ctor"
      className={cn(
        "bg-api-cpp-ctor/10 border-api-cpp-ctor/35 text-api-cpp-ctor",
        className,
      )}
    />
  );
}

/** C++ destructor (``~ClassName()``). */
export function CppDtorBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="dtor"
      className={cn(
        "bg-api-cpp-dtor/10 border-api-cpp-dtor/35 text-api-cpp-dtor",
        className,
      )}
    />
  );
}

/** C++ operator overload (``operator+`` / ``operator()`` / ``operator[]`` …). */
export function CppOperatorBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="op"
      className={cn(
        "bg-api-cpp-operator/10 border-api-cpp-operator/35 text-api-cpp-operator",
        className,
      )}
    />
  );
}

/** C++ ``virtual`` (non-pure) method.  Pure-virtual uses a dashed pill
 *  to echo Python's ``abstract`` badge convention. */
export function CppVirtualBadge({
  pure = false,
  className,
}: {
  pure?: boolean;
  className?: string;
}) {
  return (
    <Pill
      label={pure ? "= 0" : "virt"}
      className={cn(
        pure
          ? "border-dashed bg-api-cpp-pure-virtual/10 border-api-cpp-pure-virtual/40 text-api-cpp-pure-virtual"
          : "bg-api-cpp-virtual/10 border-api-cpp-virtual/35 text-api-cpp-virtual",
        className,
      )}
    />
  );
}

/** C++ function template (``template<typename T> ...``). */
export function CppTemplateBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="tmpl"
      className={cn(
        "bg-api-cpp-template/10 border-api-cpp-template/35 text-api-cpp-template",
        className,
      )}
    />
  );
}

/** C++ ``const`` method.  Standalone qualifier — rendered alongside the
 *  primary kind badge when the method is also virtual / template / etc. */
export function CppConstBadge({ className }: { className?: string }) {
  return (
    <Pill
      label="const"
      className={cn(
        "bg-api-cpp-const/10 border-api-cpp-const/35 text-api-cpp-const",
        className,
      )}
    />
  );
}

// ---------------------------------------------------------------------------
// AutoKindBadge — picks the correct badge automatically
//
// Priority:
//   Python kinds (property > classmethod > staticmethod > abstractmethod)
//   C++ kinds    (dtor > ctor > operator > pure-virtual > virtual > template > static)
//   Dunder
//   Plain fn / class fallback
//
// Secondary C++ qualifiers (``const``, plain ``static`` on non-template
// methods) are surfaced through ``AutoKindBadgeRow`` below which composes
// the primary badge with subordinate ones — call AutoKindBadgeRow when
// you want the full set, AutoKindBadge for just the dominant one.
// ---------------------------------------------------------------------------

interface AutoKindBadgeProps {
  labels: ApiLabel[];
  /** Name is used to detect __dunder__ methods */
  name?: string;
  fallback?: "fn" | "class";
  classKind?: ApiClassKind;
  className?: string;
}

export function AutoKindBadge({
  labels,
  name,
  fallback = "fn",
  classKind = "regular",
  className,
}: AutoKindBadgeProps) {
  const isWritable = labels.includes("writable");

  // Python kinds (these are mutually exclusive with C++ kinds in practice
  // because the build pipelines emit one or the other, never both).
  if (labels.includes("property"))       return <PropBadge writable={isWritable} className={className} />;
  if (labels.includes("attribute"))      return <AttrBadge className={className} />;
  if (labels.includes("classmethod"))    return <ClsMethodBadge className={className} />;
  if (labels.includes("staticmethod"))   return <StaticBadge className={className} />;
  if (labels.includes("abstractmethod")) return <AbstractBadge className={className} />;

  // C++ kinds — most-specific first.  Dtor before ctor so ``~Foo`` wins
  // when libclang emits both labels (it shouldn't, but defensively).
  if (labels.includes("cpp-dtor"))         return <CppDtorBadge className={className} />;
  if (labels.includes("cpp-ctor"))         return <CppCtorBadge className={className} />;
  if (labels.includes("cpp-operator"))     return <CppOperatorBadge className={className} />;
  if (labels.includes("cpp-pure-virtual")) return <CppVirtualBadge pure className={className} />;
  if (labels.includes("cpp-virtual"))      return <CppVirtualBadge className={className} />;
  if (labels.includes("cpp-template"))     return <CppTemplateBadge className={className} />;
  if (labels.includes("cpp-static"))       return <StaticBadge className={className} />;

  if (name && name.startsWith("__") && name.endsWith("__")) {
    return <DunderBadge className={className} />;
  }

  if (fallback === "class") return <ClassBadge kind={classKind} className={className} />;
  return <FnBadge className={className} />;
}

/** Compose the dominant kind badge with C++ qualifier badges (``const``,
 *  ``static`` when subordinate) — used by FunctionSignature so a method
 *  declared as ``virtual void foo() const`` shows both ``virt`` and
 *  ``const`` pills. */
export function AutoKindBadgeRow({
  labels,
  name,
  fallback = "fn",
  classKind = "regular",
  className,
}: AutoKindBadgeProps) {
  // The primary badge already absorbs cpp-static when no template/virtual
  // outranks it.  Const is always a sibling — emit when present.
  const isConst = labels.includes("cpp-const");
  // Static is a sibling only when there's a more specific primary
  // (template/virtual/const).  Otherwise the primary IS the static pill.
  const showStaticSibling =
    labels.includes("cpp-static") &&
    (labels.includes("cpp-virtual") ||
     labels.includes("cpp-pure-virtual") ||
     labels.includes("cpp-template"));
  return (
    <span className="inline-flex items-center gap-1 shrink-0">
      <AutoKindBadge
        labels={labels}
        name={name}
        fallback={fallback}
        classKind={classKind}
        className={className}
      />
      {showStaticSibling && <StaticBadge />}
      {isConst && <CppConstBadge />}
    </span>
  );
}
