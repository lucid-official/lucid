import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar, type SidebarItem } from "@/components/layout/Sidebar";
import { MobileSidebarProvider } from "@/components/layout/MobileSidebarContext";
import { loadApiData, getAllModuleSlugs } from "@/lib/api-loader";
import { isApiModule, isApiClassModule, isApiClass } from "@/lib/types";
import { packageLabel, segmentLabel, titleize } from "@/lib/labels";
import { API_SECTIONS, splitMembers, type PackageSections } from "@/lib/api-sections";
import type { ApiMember, ApiModule } from "@/lib/types";

// ---------------------------------------------------------------------------
// Display labels (slug aliases + subcategory/segment labels + acronym-aware
// titleization) all live in ``@/lib/labels`` — the single source of truth.
// This file owns only the *navigation-specific* policy: ordering, tags, and
// the synthetic-parent / sub-package nesting that shapes the sidebar tree.
// ---------------------------------------------------------------------------

// The eight task bases from ``lucid/models/_tasks.py``.  A class that
// inherits one of them is a head-bearing wrapper; one that inherits
// ``PretrainedModel`` directly is the family's backbone.  Kept here
// rather than derived because the sidebar is built from the emitted JSON,
// which carries base *names* and not the classes themselves.
const TASK_BASES = new Set<string>([
  "ImageClassificationModel",
  "ImageGenerationModel",
  "ObjectDetectionModel",
  "SemanticSegmentationModel",
  "SequenceClassificationModel",
  "TokenClassificationModel",
  "LanguageModelingModel",
  "WorldModelingModel",
]);

// Explicit display order for the `lucid` top-level (mirrors lucid/ on disk).
const LUCID_ORDER: string[] = [
  "tensor", "types", "dtypes", "device",
  "factories", "ops", "composite",
  "autograd", "predicates", "serialization",
  "globals", "threads", "dispatch", "vmap",
];

// Real Python path tag shown next to the alias label.
//
// Default: the slug itself (real public package, e.g. ``lucid.nn``).
// Overrides:
//   - ``lucid.tensor`` → ``lucid.Tensor`` (class, not a module).
//   - Synthesised slugs (``lucid.creation``, ``lucid.ops``,
//     ``lucid.ops.composite``) → no tag.  The underlying source
//     (``_factories/``, ``_ops/``, ``_ops/composite/``) is intentionally
//     private and the alias is the only stable label, so showing a tag
//     would be misleading either way.
//   - Synthetic hierarchy headers (``lucid.utils``) → no tag.
const PACKAGE_TAG_OVERRIDES: Record<string, string | null> = {
  "lucid.tensor":        "lucid.Tensor",
  "lucid.creation":      null,
  "lucid.ops":           null,
  "lucid.ops.composite": null,
  // Model Zoo family roots — the label ("Vision Models" etc.) already says
  // everything; the real path tag is redundant noise in the sidebar.
  "lucid.models.vision":     null,
  "lucid.models.text":       null,
  "lucid.models.generative": null,
  "lucid.models.multimodal": null,
  // C++ engine — pinned to the bottom under its own divider; the path tag
  // ("lucid._C.engine") just duplicates what the divider already conveys.
  "lucid._C.engine":         null,
};

function packageTag(slug: string, isSynthetic: boolean): string | undefined {
  if (isSynthetic) return undefined;
  if (slug in PACKAGE_TAG_OVERRIDES) {
    return PACKAGE_TAG_OVERRIDES[slug] ?? undefined;
  }
  return slug;
}

// Top-level slugs that are pinned to the head of the sidebar in this exact
// order — the canonical Tensor entry plus its closest sibling concepts
// (creation, operations).  Everything else sorts alphabetically by alias
// after these.
const TOP_PINNED: string[] = [
  "lucid.tensor",
  "lucid.creation",
  "lucid.ops",
];

// Top-level slugs pinned to the **bottom** of the sidebar, in order.
// Used for low-level / engine-internal surfaces that we want present but
// not alphabetised into the middle of the user-facing Python namespaces.
const BOTTOM_PINNED: string[] = [
  "lucid._C.engine",
];

// ---------------------------------------------------------------------------
// Build sidebar tree — server-side, reads public/api-data/*.json
// ---------------------------------------------------------------------------

/** The members of a page that the sidebar draws a leaf for: classes and
 *  functions.  One definition, so a page cannot list a member the tree lacks. */
function listedMembers(data: ApiModule): ApiMember[] {
  return data.members.filter((m) => isApiClass(m) || m.kind === "function");
}

function memberLeaf(slug: string, m: ApiMember): SidebarItem {
  // Encode the class-kind into the badge code so the sidebar pill picks
  // up the matching colour token.  Keep ``"C"`` for plain classes so
  // existing call sites don't shift colour; add ``"D" / "A" / "P"`` for
  // dataclass / abstract / protocol.  Functions stay ``"f"``.
  let badge: string;
  if (isApiClass(m)) {
    switch (m.class_kind) {
      case "dataclass": badge = "D"; break;
      case "abstract":  badge = "A"; break;
      case "protocol":  badge = "P"; break;
      default:          badge = "C";
    }
  } else {
    badge = "f";
  }
  return {
    title: m.name,
    href: `/api/${slug}/${m.name}`,
    badge,
  };
}

/** Group module members into subcategory sub-items mirroring the source tree.
 *
 *  ``subcategory`` is a slash-separated path (e.g. ``"modules/conv"``).  The
 *  renderer recursively buckets by the leading segment, so a module like
 *  ``lucid.nn`` (with members from ``lucid/nn/modules/*.py`` plus a few from
 *  ``lucid/nn/{hooks,parameter,module}.py``) produces a two-level tree —
 *  ``Modules → Convolution → Conv2d`` — that mirrors ``ls lucid/nn/``.
 *
 *  Flattening rules:
 *  - All members share the same subcategory → flat list.
 *  - Bucket has exactly 1 member → direct link (no "Foo (1) → Foo" wrapper).
 *  - A subdirectory bucket whose children have ≤ 1 unique subgroup → flat
 *    inside that bucket (no degenerate "Other" sub-tree).
 */
function groupMembers(
  slug: string,
  members: ApiMember[],
  explicitOrder?: string[],
): SidebarItem[] {
  // Bucket by the FIRST segment of subcategory ("modules/conv" → "modules").
  const buckets = new Map<string | null, ApiMember[]>();
  for (const m of members) {
    const first = m.subcategory?.split("/")[0] ?? null;
    if (!buckets.has(first)) buckets.set(first, []);
    buckets.get(first)!.push(m);
  }

  // Only one bucket total — nesting adds no value.
  if (buckets.size === 1) {
    return members.map((m) => memberLeaf(slug, m));
  }

  /** Recurse into a bucket: strip the leading segment from each member's
   *  subcategory, then group again.  Returns the list of child SidebarItems. */
  function buildSubtree(group: ApiMember[]): SidebarItem[] {
    const stripped: ApiMember[] = group.map((m) => {
      const rest = (m.subcategory ?? "").split("/").slice(1).join("/");
      return { ...m, subcategory: rest || null };
    });
    return groupMembers(slug, stripped);
  }

  /** Normalise a label/name for comparison: lowercase + strip spaces and
   *  underscores.  Used by the exact-match singleton flatten rule. */
  function norm(s: string): string {
    return s.toLowerCase().replace(/[\s_]+/g, "");
  }

  /** Push a bucket to the result, applying singleton flattening + recursive
   *  subgrouping when the bucket's members carry deeper subcategory paths.
   *
   *  Singleton flatten rule (anti-redundancy):
   *  - If a bucket has exactly 1 member AND the member's name normalises to
   *    the same string as the bucket's label, render it as a direct link
   *    (no "Foo → Foo" wrapper).  Examples that flatten: ``Module → Module``,
   *    ``Parameter → Parameter``, ``device → device``.
   *  - Otherwise keep the bucket — preserves semantic context like
   *    ``Attention (1) → scaled_dot_product_attention`` where the bucket
   *    label adds info the leaf name doesn't carry. */
  function pushBucket(label: string, group: ApiMember[]) {
    if (group.length === 1 && norm(group[0].name) === norm(label)) {
      result.push(memberLeaf(slug, group[0]));
      return;
    }
    const anyDeeper = group.some((m) => (m.subcategory ?? "").includes("/"));
    const items = anyDeeper
      ? buildSubtree(group)
      : group.map((m) => memberLeaf(slug, m));
    result.push({
      title: label,
      badge: `${group.length}`,
      items,
    });
  }

  const result: SidebarItem[] = [];
  const seen = new Set<string>();

  // 1. Explicit order first (only buckets that actually have members)
  if (explicitOrder) {
    for (const cat of explicitOrder) {
      const group = buckets.get(cat);
      if (!group?.length) continue;
      seen.add(cat);
      pushBucket(segmentLabel(cat), group);
    }
  }

  // 2. Remaining named buckets — alphabetical by display label
  const remaining: Array<{ cat: string; label: string }> = [];
  for (const cat of buckets.keys()) {
    if (cat === null || seen.has(cat)) continue;
    remaining.push({ cat, label: segmentLabel(cat) });
  }
  remaining.sort((a, b) => a.label.localeCompare(b.label));
  for (const { cat, label } of remaining) {
    pushBucket(label, buckets.get(cat)!);
  }

  // 3. Null-bucket members (defined directly in the module's __init__.py)
  //    are surfaced as direct links rather than wrapped in an "Other" group.
  //    This matches user expectation: a tensor-creation factory at
  //    `lucid/nn/__init__.py` (if any) should sit at the module's top level.
  const nullGroup = buckets.get(null);
  if (nullGroup?.length) {
    for (const m of nullGroup) result.push(memberLeaf(slug, m));
  }

  return result;
}

// True for slugs of the form ``lucid.models.{vision|text|generative}.<family>`` —
// the leaf level under Model Zoo where each model family lives.  Used to swap
// the default subcategory-bucketed sidebar tree for the strict 4-slot layout
// (Config / Direct Model / Task Wrappers / Pretrained).
function isFamilyLeaf(slug: string): boolean {
  const parts = slug.split(".");
  if (parts.length !== 4) return false;
  if (parts[0] !== "lucid" || parts[1] !== "models") return false;
  return (
    parts[2] === "vision" ||
    parts[2] === "text" ||
    parts[2] === "generative" ||
    parts[2] === "multimodal"
  );
}

/** Build the 4-slot family-leaf sidebar item:
 *
 *   <CanonicalName>
 *   ├── <FamilyConfig>           (slot 1, direct leaf)
 *   ├── <DirectModelClass>       (slot 2, direct leaf)
 *   ├── Task Wrappers ──┐        (slot 3, sub-group: every *For* class)
 *   │   └── ...
 *   └── Pretrained ─────┘        (slot 4, sub-group: every factory function)
 *       └── ...
 *
 *  ``*Output`` dataclasses are hidden — they're forward-return types, not
 *  user-facing entry points.
 */
function buildFamilyLeafItem(slug: string, data: ApiModule): SidebarItem {
  const dirName = slug.split(".").pop() ?? slug;
  const title = data.canonical_name && data.canonical_name.length > 0
    ? data.canonical_name
    : titleize(dirName);

  const allMembers = listedMembers(data);

  // Partition into the family slots.  Membership is decided by what a
  // class *inherits*, not by where it happens to sit in the member list:
  // the previous version took the first leftover class as the direct
  // model, so Stable Diffusion — whose members sort ``autoencoder``
  // before ``model`` — advertised ``AutoencoderKL`` as the model and
  // dropped ``StableDiffusionModel`` entirely.
  const configClasses = allMembers.filter(
    (m) => isApiClass(m) && m.name.endsWith("Config"),
  );
  const configClass = configClasses[0];
  // A head-bearing wrapper subclasses one of the eight task bases
  // (``lucid/models/_tasks.py``).  Naming alone missed the families that
  // predate the ``<Family>For<Task>`` convention — ``GPTLMHeadModel`` and
  // ``GPT2DoubleHeadsModel`` carry heads but no "For" in the name, and so
  // were competing for the direct-model slot instead of filling this one.
  const taskWrappers = allMembers.filter(
    (m) =>
      isApiClass(m) &&
      !m.name.endsWith("Output") &&
      !m.name.endsWith("Config") &&
      // ``For`` must start a task name: ``RoFormerModel`` and
      // ``MaskFormer`` contain the substring and are backbones, not
      // wrappers, so a bare ``includes`` files them under the wrong slot.
      ((m.bases ?? []).some((b) => TASK_BASES.has(b)) || /For[A-Z]/.test(m.name)),
  );
  // *Output classes are dataclasses for forward return types — not entry
  // points; suppress from the sidebar to keep the family tree clean.
  const outputClasses = allMembers.filter(
    (m) => isApiClass(m) && m.name.endsWith("Output"),
  );
  // Pretrained-weight enums (``<Family>Weights``).  The model-zoo-wide
  // ``Weights`` slot — every family owns its weight classes directly (the
  // ``lucid.models.weights`` aggregator is not documented as a package), so
  // they surface here next to the model rather than in a separate tree.
  const weightClasses = allMembers.filter(
    (m) => isApiClass(m) && m.name.endsWith("Weights"),
  );
  // The backbone is the class that subclasses ``PretrainedModel``
  // directly — the one slot the whole family tree hangs off.
  const slotted = new Set<ApiMember>([
    ...configClasses, ...taskWrappers, ...outputClasses, ...weightClasses,
  ]);
  const candidates = allMembers.filter((m) => isApiClass(m) && !slotted.has(m));
  // Strictly ``PretrainedModel``: twelve detection and segmentation
  // families define no separate backbone at all (the task wrapper *is*
  // the model), and promoting whatever else they export would put a
  // detection utility like ``MultiScaleResolution`` in the model slot.
  const directModel = candidates.find(
    (m) => isApiClass(m) && (m.bases ?? []).includes("PretrainedModel"),
  );
  // Whatever is left is a composition primitive the family exports on
  // purpose — Stable Diffusion's VAE, U-Net and schedulers, a text
  // family's tokenizers, Score-SDE's SDE variants.  They used to fall out
  // of the tree silently; twenty-one entries across eight families were
  // reachable on the page and absent from the sidebar.
  const components = candidates.filter((m) => m !== directModel);
  const pretrained = allMembers.filter((m) => m.kind === "function");

  const items: SidebarItem[] = [];
  // YOLO ships one Config per paper version; ``.find`` kept only the
  // first and the other three never appeared.
  if (configClasses.length === 1) {
    items.push(memberLeaf(slug, configClasses[0]));
  } else if (configClasses.length > 1) {
    items.push({
      title: "Configs",
      badge: `${configClasses.length}`,
      items: configClasses.map((m) => memberLeaf(slug, m)),
    });
  }
  if (directModel) items.push(memberLeaf(slug, directModel));
  if (taskWrappers.length > 0) {
    items.push({
      title: "Task Wrappers",
      badge: `${taskWrappers.length}`,
      items: taskWrappers.map((m) => memberLeaf(slug, m)),
    });
  }
  if (pretrained.length > 0) {
    items.push({
      title: "Pretrained",
      badge: `${pretrained.length}`,
      items: pretrained.map((m) => memberLeaf(slug, m)),
    });
  }
  if (components.length > 0) {
    items.push({
      title: "Components",
      badge: `${components.length}`,
      items: components.map((m) => memberLeaf(slug, m)),
    });
  }
  if (weightClasses.length > 0) {
    items.push({
      title: "Weights",
      badge: `${weightClasses.length}`,
      items: weightClasses.map((m) => memberLeaf(slug, m)),
    });
  }

  const badgeCount =
    configClasses.length +
    (directModel ? 1 : 0) +
    taskWrappers.length +
    components.length +
    pretrained.length +
    weightClasses.length;

  return {
    title,
    // No tag on family-leaf entries — the canonical name (ResNet, BERT, …)
    // is the user-facing identifier; the raw ``lucid.models.vision.resnet``
    // path is just noise here.  Same rationale as the family-root tag
    // suppression in PACKAGE_TAG_OVERRIDES.
    href: `/api/${slug}`,
    badge: `${badgeCount}`,
    items,
  };
}

/** A tuner package (``lucid.peft.tuners.lora``) as a sidebar entry.
 *
 *  Same idea as the model-family leaf: the config and the model first, where
 *  a reader starts, then whatever else the method defines — its layer
 *  classes, the encoder of a prompt method, an init enum — under one
 *  "Components" group.  The default bucketing by source file would put a
 *  one-class "Configuration" group beside a one-class "Tuner Model" group. */
function buildTunerItem(slug: string, data: ApiModule): SidebarItem {
  const members = listedMembers(data);
  const configs = members.filter((m) => isApiClass(m) && m.name.endsWith("Config"));
  const models = members.filter((m) => isApiClass(m) && m.name.endsWith("Model"));
  const taken = new Set<ApiMember>([...configs, ...models]);
  const components = members.filter((m) => !taken.has(m));

  const items: SidebarItem[] = [
    ...configs.map((m) => memberLeaf(slug, m)),
    ...models.map((m) => memberLeaf(slug, m)),
  ];
  if (components.length > 0) {
    items.push({
      title: "Components",
      badge: `${components.length}`,
      items: components.map((m) => memberLeaf(slug, m)),
    });
  }
  return {
    title: data.canonical_name && data.canonical_name.length > 0
      ? data.canonical_name
      : packageLabel(slug),
    href: `/api/${slug}`,
    badge: `${members.length}`,
    items,
  };
}

/** The class and function members of an overview page; none when the page has
 *  no data yet (the JSON is not built — the link degrades to a plain one). */
function overviewMembers(slug: string): ApiMember[] {
  try {
    const data = loadApiData(slug);
    return isApiModule(data) ? listedMembers(data) : [];
  } catch {
    return [];
  }
}

function buildModuleItem(slug: string): SidebarItem {
  let memberItems: SidebarItem[] | undefined;
  let badge: string | undefined;

  try {
    const data = loadApiData(slug);

    // Family-leaf pages get the strict 4-slot tree (Config / Direct /
    // Task Wrappers / Pretrained).  Bypass the default subcategory
    // bucketing entirely.
    if (isApiModule(data) && isFamilyLeaf(slug)) {
      return buildFamilyLeafItem(slug, data);
    }

    if (isApiModule(data)) {
      const members = listedMembers(data);

      // Family-root pages (lucid.models.vision/text/generative) — and the
      // top-level lucid.models — carry a ``family_groups`` array instead of
      // a flat member list.  Show that count for the badge so users see
      // "Vision (45)" not "Vision (0)".
      if (data.family_groups && data.family_groups.length > 0 && members.length === 0) {
        badge = `${data.family_groups.length}`;
      } else if (members.length > 0) {
        // For the `lucid` top-level, skip the Tensor class itself — it has
        // its own dedicated entry at `lucid.tensor` where all of its
        // methods are surfaced.  Listing it twice in the sidebar would
        // just create a redundant single-item "Tensor" group.
        const visible =
          slug === "lucid"
            ? members.filter((m) => m.subcategory !== "tensor")
            : members;
        badge = `${visible.length}`;
        // Path-based subcategory grouping applies to every module.  The
        // lucid top-level uses an explicit order; others fall back to
        // alphabetical-by-display-label inside groupMembers.
        const order = slug === "lucid" ? LUCID_ORDER : undefined;
        memberItems = groupMembers(slug, visible, order);
      }
    } else if (isApiClassModule(data)) {
      // lucid.tensor — the module IS the class; its methods are not listed
      // here per the user's request ("클래스 내 메소드는 제외").
      badge = `${data.methods.length}`;
      memberItems = undefined;
    }
  } catch {
    // JSON not yet built — gracefully degrade to a plain link
  }

  return {
    title: packageLabel(slug),
    tag: packageTag(slug, false),
    href: `/api/${slug}`,
    badge,
    items: memberItems,
  };
}

/** Find a slug's documented parent (the longest documented prefix), excluding
 *  the bare ``lucid`` root.  Depth-1 (``lucid``) and depth-2 (``lucid.nn``,
 *  ``lucid.optim`` …) slugs are always top-level — they sit as siblings to
 *  mirror how the reference framework and NumPy docs lay out their first-level
 *  subpackages.  Depth-3+ slugs (``lucid.nn.functional`` etc.) nest under
 *  whichever shorter prefix is also in the slug set. */
function findParent(slug: string, slugSet: Set<string>): string | null {
  const parts = slug.split(".");
  if (parts.length <= 2) return null;
  for (let i = parts.length - 1; i >= 2; i--) {
    const candidate = parts.slice(0, i).join(".");
    if (slugSet.has(candidate)) return candidate;
  }
  return null;
}

function buildApiSidebar(): SidebarItem[] {
  const docSlugs = new Set(getAllModuleSlugs());

  // Synthetic parents: any depth-3+ slug whose intermediate ancestors aren't
  // in the doc set gets a non-clickable placeholder so the Python hierarchy
  // is preserved.  Example: ``lucid.utils.data`` exists but ``lucid.utils``
  // is `_EXCLUDED_SELF` — we still inject a ``lucid.utils`` sidebar header
  // (no link, just a toggle) so ``Data`` shows up under it instead of at
  // the top level.
  //
  // Private-segment skip: when a slug routes through a ``_``-prefixed
  // namespace (e.g. ``lucid._C.engine``), don't manufacture a synthetic
  // parent for it — ``lucid._C`` has no public-API meaning, and exposing
  // it as a sidebar header would only confuse readers.  Such slugs are
  // treated as top-level entries instead.
  const synthetic = new Set<string>();
  for (const slug of docSlugs) {
    const parts = slug.split(".");
    if (parts.length < 3) continue;
    // Skip synthetic-parent generation for slugs routed through a
    // ``_``-prefixed (private) namespace — they bubble to top-level
    // instead of producing a meaningless header like ``lucid._C``.
    if (parts.slice(1, -1).some((p) => p.startsWith("_"))) continue;
    let hasAncestor = false;
    for (let i = parts.length - 1; i >= 2; i--) {
      if (docSlugs.has(parts.slice(0, i).join("."))) {
        hasAncestor = true;
        break;
      }
    }
    if (!hasAncestor) synthetic.add(parts.slice(0, 2).join("."));
  }

  const allSlugs = new Set([...docSlugs, ...synthetic]);

  // Bucket each slug under its parent (or null for top-level).
  const childrenOf = new Map<string | null, string[]>();
  for (const slug of allSlugs) {
    const parent = findParent(slug, allSlugs);
    if (!childrenOf.has(parent)) childrenOf.set(parent, []);
    childrenOf.get(parent)!.push(slug);
  }

  // Top-level sort: TOP_PINNED first in declared order, then everything
  // else alphabetically by alias, then BOTTOM_PINNED last in declared
  // order (used for the C++ engine surface).  Nested levels stay
  // alphabetical.
  for (const [parent, kids] of childrenOf) {
    if (parent === null) {
      const rank = (s: string): number => {
        const top = TOP_PINNED.indexOf(s);
        if (top !== -1) return top;                          // 0..N-1
        const bot = BOTTOM_PINNED.indexOf(s);
        if (bot !== -1) return 1_000_000 + bot;              // sink
        return TOP_PINNED.length;                            // alphabetic middle
      };
      kids.sort((a, b) => {
        const rd = rank(a) - rank(b);
        if (rd !== 0) return rd;
        return packageLabel(a).localeCompare(packageLabel(b));
      });
    } else {
      kids.sort((a, b) => packageLabel(a).localeCompare(packageLabel(b)));
    }
  }

  // Slugs whose sub-packages are the *primary* content rather than
  // auxiliary — render them FIRST in the items list so the family-root
  // entries (e.g. Vision / Text / Generative under Model Zoo) sit above
  // the slug's own bucketed members instead of being buried at the bottom.
  const SUBPKGS_FIRST = new Set<string>(["lucid.models"]);

  /** A package that groups its sub-packages in ``api-sections.json``
   *  (``lucid.peft``): its children are the sections, not the alphabetical
   *  run of sub-packages.
   *
   *  A sub-package section holds the tuner pages; a member section holds a
   *  slice of the package's own classes and functions (``splitMembers`` — the
   *  ``rest`` section takes whatever no other claimed).  The overview page
   *  (``lucid.peft.tuners``) is one link at the top, opening onto the classes
   *  and functions that page itself defines.  A documented
   *  sub-package no section lists still gets an entry, under "More", so a
   *  tuner added without touching the file shows up (the completeness test is
   *  what insists it be listed). */
  function buildSectionedItem(slug: string, plan: PackageSections): SidebarItem {
    const base = buildModuleItem(slug);
    const data = (() => {
      try { return loadApiData(slug); } catch { return null; }
    })();
    const members = data && isApiModule(data) ? listedMembers(data) : [];
    const picked = splitMembers(members, plan.sections);

    const children: SidebarItem[] = [];
    const listed = new Set<string>();
    if (plan.overview && docSlugs.has(plan.overview)) {
      listed.add(plan.overview);
      // The overview page lists its own classes and functions after the
      // sub-package sections (``ModuleOverview``), and they belong to no
      // section of ``plan`` — so each gets a leaf under the link, or a page
      // member would be reachable from the page alone (``MixedModel`` on
      // ``lucid.peft.tuners``).
      const own = overviewMembers(plan.overview);
      children.push({
        title: `All ${packageLabel(plan.overview)}`,
        href: `/api/${plan.overview}`,
        items: own.length > 0 ? groupMembers(plan.overview, own) : undefined,
      });
    }

    for (const section of plan.sections) {
      if (section.packages) {
        const entries: SidebarItem[] = [];
        for (const pkg of section.packages) {
          if (!docSlugs.has(pkg.slug)) continue;
          listed.add(pkg.slug);
          let entry: SidebarItem;
          try {
            const pkgData = loadApiData(pkg.slug);
            entry = isApiModule(pkgData)
              ? buildTunerItem(pkg.slug, pkgData)
              : buildModuleItem(pkg.slug);
          } catch {
            entry = { title: pkg.label, href: `/api/${pkg.slug}` };
          }
          entries.push(entry);
        }
        if (entries.length > 0) {
          children.push({ title: section.title, badge: `${entries.length}`, items: entries });
        }
      }
      const own = picked.get(section.id) ?? [];
      if (section.members && own.length > 0) {
        children.push({
          title: section.title,
          badge: `${own.length}`,
          items: section.members.flat
            ? own.map((m) => memberLeaf(slug, m))
            : groupMembers(slug, own),
        });
      }
    }

    const unlisted = [...docSlugs]
      .filter((s) => s.startsWith(`${slug}.`) && !listed.has(s))
      .sort((a, b) => packageLabel(a).localeCompare(packageLabel(b)));
    if (unlisted.length > 0) {
      children.push({
        title: "More",
        badge: `${unlisted.length}`,
        items: unlisted.map((s) => buildModuleItem(s)),
      });
    }
    return { ...base, items: children };
  }

  /** Build a SidebarItem for a slug, recursively attaching any documented
   *  sub-packages as nested children. */
  function buildItem(slug: string, parent: string | null): SidebarItem {
    const plan = API_SECTIONS[slug];
    if (plan) return buildSectionedItem(slug, plan);
    const isSynthetic = synthetic.has(slug);
    const base: SidebarItem = isSynthetic
      ? { title: packageLabel(slug), tag: packageTag(slug, true) }
      : buildModuleItem(slug);
    // ``buildModuleItem`` already runs the slug through ``packageLabel`` so
    // titles are aliases at every depth (e.g. ``Functional`` for
    // ``lucid.nn.functional``).  No further prefix-stripping needed.
    const subPkgs = childrenOf.get(slug) ?? [];
    if (subPkgs.length === 0) return base;
    const subPkgItems = subPkgs.map((s) => buildItem(s, slug));
    const ownItems = base.items ?? [];
    return {
      ...base,
      items: SUBPKGS_FIRST.has(slug)
        ? [...subPkgItems, ...ownItems]
        : [...ownItems, ...subPkgItems],
    };
  }

  const topLevel = childrenOf.get(null) ?? [];
  const items = topLevel.map((s) => buildItem(s, null));

  // Inject a visual divider directly before the first BOTTOM_PINNED slug so
  // the engine-internal surface (C++ Engine) reads as its own section rather
  // than the alphabetic tail of the Python namespaces.
  const firstBottomIdx = topLevel.findIndex((s) => BOTTOM_PINNED.includes(s));
  if (firstBottomIdx > 0) {
    items.splice(firstBottomIdx, 0, { title: "__bottom-pinned__", separator: true });
  }
  return items;
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export default function ApiLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const sidebar = buildApiSidebar();

  return (
    <MobileSidebarProvider items={sidebar}>
      <div className="flex min-h-dvh flex-col">
        <Header />
        <div className="mx-auto flex w-full max-w-screen-2xl flex-1 gap-0 px-4 sm:px-6 pt-14">
          <Sidebar
            items={sidebar}
            className="sticky top-14 h-[calc(100dvh-3.5rem)]"
          />
          <main
            id="main-content"
            tabIndex={-1}
            className="flex-1 min-w-0 pt-10 pb-12 lg:px-8 focus:outline-none"
          >
            {children}
          </main>
        </div>
        <Footer />
      </div>
    </MobileSidebarProvider>
  );
}
