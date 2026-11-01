import type { ApiLabel, ApiClassKind } from "@/lib/types";

/** Tailwind text-color class for a method/function based on its labels + name.
 *
 *  Precedence (highest → lowest):
 *    Python kinds — property / classmethod / staticmethod / abstractmethod
 *    C++ kinds    — ctor / dtor / operator / pure-virtual / virtual / static / const / template
 *    Dunder fallback (``__init__`` etc.)
 *    Plain function default.
 *
 *  A symbol can carry multiple labels (e.g. ``virtual ... const``) — we
 *  pick the most specific one for the name colour and let the kind
 *  badge surface secondary attributes. */
export function getMemberNameColor(name: string, labels: ApiLabel[]): string {
  // Python kinds first — they take precedence on Python-side members.
  if (labels.includes("property"))       return "text-api-prop";
  if (labels.includes("attribute"))      return "text-api-prop";
  if (labels.includes("classmethod"))    return "text-api-cls";
  if (labels.includes("staticmethod"))   return "text-api-static";
  if (labels.includes("abstractmethod")) return "text-api-abstract";
  // C++ kinds — ordered by specificity.
  if (labels.includes("cpp-dtor"))         return "text-api-cpp-dtor";
  if (labels.includes("cpp-ctor"))         return "text-api-cpp-ctor";
  if (labels.includes("cpp-operator"))     return "text-api-cpp-operator";
  if (labels.includes("cpp-pure-virtual")) return "text-api-cpp-pure-virtual";
  if (labels.includes("cpp-virtual"))      return "text-api-cpp-virtual";
  if (labels.includes("cpp-static"))       return "text-api-static";
  if (labels.includes("cpp-template"))     return "text-api-cpp-template";
  if (name.startsWith("__") && name.endsWith("__")) return "text-api-dunder";
  return "text-api-fn";
}

/** Tailwind text-color class for a class name based on its kind. */
export function getClassNameColor(classKind: ApiClassKind): string {
  if (classKind === "abstract")  return "text-api-class-abstract";
  if (classKind === "dataclass") return "text-api-class-dataclass";
  if (classKind === "protocol")  return "text-api-class-protocol";
  return "text-api-class";
}

/** Tailwind hover-border class for a class card based on its kind. */
export function getClassHoverBorder(classKind: ApiClassKind): string {
  if (classKind === "abstract")  return "hover:border-api-class-abstract/40";
  if (classKind === "dataclass") return "hover:border-api-class-dataclass/40";
  if (classKind === "protocol")  return "hover:border-api-class-protocol/40";
  return "hover:border-api-class/40";
}
