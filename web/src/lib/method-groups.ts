import type { ApiMethod } from "@/lib/types";

export interface MethodGroup {
  /** Display label rendered as the section header.  Picked from a small
   *  fixed enumeration per platform (Python vs C++) so the user sees a
   *  predictable taxonomy across every class. */
  label: string;
  /** Anchor slug for the section header (and the TOC link).  Derived
   *  from ``label`` by lowercasing + hyphenating; stable so external
   *  links to ``#instance-methods`` keep working. */
  id: string;
  methods: ApiMethod[];
}

const PY_LIFECYCLE = new Set([
  "__init__",
  "__new__",
  "__call__",
  "__del__",
]);

/** Convert ``"In-place mutations"`` to ``"in-place-mutations"`` for use
 *  as an HTML id / anchor.  ASCII-only — group labels are authored here,
 *  so non-ASCII isn't a concern. */
function slugify(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function sortByName<T extends { name: string }>(items: T[]): T[] {
  return [...items].sort((a, b) => a.name.localeCompare(b.name));
}

/** Determine whether the method list came from the C++ engine pipeline.
 *  C++ classes always emit at least one ``cpp-*`` label across their
 *  methods (every member is one of ctor / dtor / static / virtual /
 *  ...), so a single sighting flips the whole class to C++ grouping. */
function isCppMethods(methods: ApiMethod[]): boolean {
  return methods.some((m) =>
    (m.labels ?? []).some((l) => l.startsWith("cpp-")),
  );
}

// ---------------------------------------------------------------------------
// Python grouping
//
// Order chosen to match how a user typically scans a class page:
// constructors first (how do I make one), then read-only access (props),
// then class / static utilities, then mutating instance methods, then
// in-place variants, with dunders folded last because most readers
// don't need them on a first pass.
// ---------------------------------------------------------------------------

function groupPython(methods: ApiMethod[]): MethodGroup[] {
  const buckets: Record<string, ApiMethod[]> = {
    "Constructors":     [],
    "Properties":       [],
    "Attributes":       [],
    "Class methods":    [],
    "Static methods":   [],
    "Instance methods": [],
    "In-place ops":     [],
    "Dunder methods":   [],
  };

  for (const m of methods) {
    const labels = m.labels ?? [];
    if (PY_LIFECYCLE.has(m.name)) {
      buckets["Constructors"].push(m);
    } else if (labels.includes("property")) {
      buckets["Properties"].push(m);
    } else if (labels.includes("attribute")) {
      buckets["Attributes"].push(m);
    } else if (labels.includes("classmethod")) {
      buckets["Class methods"].push(m);
    } else if (labels.includes("staticmethod")) {
      buckets["Static methods"].push(m);
    } else if (m.name.startsWith("__") && m.name.endsWith("__")) {
      buckets["Dunder methods"].push(m);
    } else if (
      // ``tensor.add_(x)`` / ``parameter.zero_()`` convention — the
      // trailing-underscore method is the in-place sibling of the same
      // non-suffix call.  Single-underscore prefix (``_internal``) is
      // skipped by the upstream filter and never reaches us here.
      m.name.endsWith("_") &&
      !m.name.startsWith("_") &&
      m.name.length > 1
    ) {
      buckets["In-place ops"].push(m);
    } else {
      buckets["Instance methods"].push(m);
    }
  }

  // Constructors: keep lifecycle order; everything else alphabetises.
  const ctorOrder = ["__init__", "__new__", "__call__", "__del__"];
  buckets["Constructors"] = buckets["Constructors"].sort((a, b) => {
    const ia = ctorOrder.indexOf(a.name);
    const ib = ctorOrder.indexOf(b.name);
    if (ia !== -1 || ib !== -1) {
      return (ia === -1 ? Infinity : ia) - (ib === -1 ? Infinity : ib);
    }
    return a.name.localeCompare(b.name);
  });
  for (const k of ["Properties", "Attributes", "Class methods", "Static methods", "Instance methods", "In-place ops", "Dunder methods"]) {
    buckets[k] = sortByName(buckets[k]);
  }

  const order = [
    "Constructors",
    "Properties",
    "Attributes",
    "Class methods",
    "Static methods",
    "Instance methods",
    "In-place ops",
    "Dunder methods",
  ];
  return order
    .map((label) => ({ label, id: slugify(label), methods: buckets[label] }))
    .filter((g) => g.methods.length > 0);
}

// ---------------------------------------------------------------------------
// C++ grouping
//
// Mirrors what a header skim looks like: ctors / dtor at the top, then
// static factories, operators, virtual hooks, and finally the regular
// const-or-not member methods.
// ---------------------------------------------------------------------------

function groupCpp(methods: ApiMethod[]): MethodGroup[] {
  const buckets: Record<string, ApiMethod[]> = {
    "Constructors":    [],
    "Destructor":      [],
    "Static methods":  [],
    "Operators":       [],
    "Virtual methods": [],
    "Methods":         [],
  };

  for (const m of methods) {
    const labels = m.labels ?? [];
    if (labels.includes("cpp-ctor")) {
      buckets["Constructors"].push(m);
    } else if (labels.includes("cpp-dtor")) {
      buckets["Destructor"].push(m);
    } else if (labels.includes("cpp-operator")) {
      buckets["Operators"].push(m);
    } else if (labels.includes("cpp-pure-virtual") || labels.includes("cpp-virtual")) {
      buckets["Virtual methods"].push(m);
    } else if (labels.includes("cpp-static")) {
      buckets["Static methods"].push(m);
    } else {
      buckets["Methods"].push(m);
    }
  }

  for (const k of Object.keys(buckets)) {
    buckets[k] = sortByName(buckets[k]);
  }

  const order = [
    "Constructors",
    "Destructor",
    "Static methods",
    "Operators",
    "Virtual methods",
    "Methods",
  ];
  return order
    .map((label) => ({ label, id: slugify(label), methods: buckets[label] }))
    .filter((g) => g.methods.length > 0);
}

/** Top-level entry — groups methods into platform-appropriate sections.
 *
 *  Used by ``ClassDoc`` to break long class pages (309-method
 *  ``Tensor``, 80-method ``Module``, dozens of ``Backward`` nodes) into
 *  scannable subsections.  Each returned group gets its own ``<h2>`` on
 *  the page, so the table-of-contents rail picks them up as top-level
 *  jumps.
 */
export function groupMethods(methods: ApiMethod[]): MethodGroup[] {
  return isCppMethods(methods) ? groupCpp(methods) : groupPython(methods);
}
