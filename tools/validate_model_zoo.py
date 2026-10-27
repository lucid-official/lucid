"""Layer-3 contract validator for the Lucid model zoo.

Walks ``lucid/models/<domain>/<family>/`` and checks each family against the
contract documented in ``obsidian/architecture/arch-models-family-contract.md``.
Reports violations as ``path:line: <code> <message>`` so editors / CI can
jump straight to the offending site.

Layer 1 (``ModelConfig.__init_subclass__``) and Layer 2
(``@model_family_meta`` decorator) cover *runtime* mistakes that surface
the moment a malformed Config is imported.  This validator covers the
rest: directory layout, decorator presence, naming patterns of model
classes and factory functions — things that ``ast.parse`` can verify
without executing user code.

Usage
-----
::

    python -m tools.validate_model_zoo                       # full sweep
    python -m tools.validate_model_zoo --family resnet       # one family
    python -m tools.validate_model_zoo --domain text         # one domain
    python -m tools.validate_model_zoo --strict              # warnings → errors

Exit code is 0 on success and 1 on any violation (errors only; warnings
do not affect exit code unless ``--strict`` is passed).
"""

import argparse
import ast
import difflib
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Task-wrapper class names follow ``<Family>For<TaskCamelCase>`` — the ``For``
# must be a word boundary into a CapWord task name.  This regex extracts the
# trailing ``For<...>`` segment only when it ends a CapWord run, so ``RoFormerModel``
# / ``EfficientFormer`` / ``Former`` (mere substrings of ``For``) don't match.
_TASK_SUFFIX_RE = re.compile(r"For[A-Z][A-Za-z]*$")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "lucid" / "models"
DOMAINS = ("vision", "text", "generative", "multimodal")

# Task suffixes mirror ``web/scripts/build-api-data.py`` _TASK_SUFFIX_MAP.
# Kept duplicated here (small, ~17 entries) instead of importing the
# build-script module so the validator stays standalone.
KNOWN_TASK_SUFFIXES: set[str] = {
    "ForImageClassification",
    "ForObjectDetection",
    "ForInstanceSegmentation",
    "ForSemanticSegmentation",
    "ForPanopticSegmentation",
    "ForImageGeneration",
    "ForImageToImage",
    "ForMaskedImageModeling",
    "ForMaskedLM",
    "ForCausalLM",
    "ForSeq2SeqLM",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForNextSentencePrediction",
    "ForMultipleChoice",
    "ForPreTraining",
    "ForWorldModeling",
    # A clip, not an image, but the task tag stays "image-classification":
    # the taxonomy is deliberately coarse and the name is where the
    # specificity lives (see V-JEPA, and CLIP's zero-shot classifier).
    "ForVideoClassification",
    "ForZeroShotImageClassification",
    # Alternative head-style naming (GPT/GPT-2 use reference-framework names).
    "LMHeadModel",
    "DoubleHeadsModel",
}


@dataclass
class Issue:
    """A single violation report."""

    severity: str  # "error" | "warning"
    code: str  # short identifier, e.g. "MZ001"
    path: Path
    line: int
    message: str

    def format(self) -> str:
        rel = self.path.relative_to(REPO_ROOT) if self.path.is_absolute() else self.path
        return f"{rel}:{self.line}: {self.severity} {self.code} {self.message}"


# ---------------------------------------------------------------------------
# Per-file AST helpers
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None


def _classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [n for n in tree.body if isinstance(n, ast.ClassDef)]


def _functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)]


def _base_names(cls: ast.ClassDef) -> list[str]:
    out: list[str] = []
    for b in cls.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def _decorator_call_names(cls: ast.ClassDef) -> list[str]:
    """Return the bare names of ``@call(...)`` style decorators only."""
    out: list[str] = []
    for d in cls.decorator_list:
        if isinstance(d, ast.Call):
            func = d.func
            if isinstance(func, ast.Name):
                out.append(func.id)
            elif isinstance(func, ast.Attribute):
                out.append(func.attr)
    return out


def _decorator_args_dict(cls: ast.ClassDef, name: str) -> dict[str, ast.expr] | None:
    """Find a ``@<name>(...)`` decorator and return its kwargs (literal-friendly)."""
    for d in cls.decorator_list:
        if not isinstance(d, ast.Call):
            continue
        func = d.func
        fn = (
            func.id
            if isinstance(func, ast.Name)
            else (func.attr if isinstance(func, ast.Attribute) else None)
        )
        if fn != name:
            continue
        return {kw.arg: kw.value for kw in d.keywords if kw.arg is not None}
    return None


def _class_var(cls: ast.ClassDef, name: str) -> ast.expr | None:
    """Return the RHS expression of a ``name: ... = value`` class attribute."""
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == name and stmt.value is not None:
                return stmt.value
    return None


def _literal_str(node: ast.expr) -> str | None:
    try:
        v = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return v if isinstance(v, str) else None


# ---------------------------------------------------------------------------
# Family-level checks
# ---------------------------------------------------------------------------


def _has_split_layout(family_dir: Path) -> bool:
    """YOLO-style: ``_v1.py``, ``_v2.py``, ... instead of the standard trio."""
    return (
        any(family_dir.glob("_v[0-9]*.py"))
        and not (family_dir / "_config.py").is_file()
    )


def _check_layout(
    family_dir: Path, issues: list[Issue]
) -> tuple[list[Path], list[Path]]:
    """Verify directory layout.  Returns (config_files, model_files).

    Standard layout: ``_config.py``, ``_model.py``, ``_pretrained.py``.
    Split layout:    ``_v1.py``, ``_v2.py``, … each containing everything.
    """
    cfg_files: list[Path] = []
    model_files: list[Path] = []
    if _has_split_layout(family_dir):
        for vp in sorted(family_dir.glob("_v[0-9]*.py")):
            cfg_files.append(vp)
            model_files.append(vp)
        return cfg_files, model_files
    expected = {"_config.py", "_model.py", "_pretrained.py", "__init__.py"}
    present = {
        p.name for p in family_dir.iterdir() if p.is_file() and p.suffix == ".py"
    }
    missing = expected - present
    if missing:
        issues.append(
            Issue(
                "error",
                "MZ001",
                family_dir,
                1,
                f"family directory missing required files: {sorted(missing)}",
            )
        )
    if (family_dir / "_config.py").is_file():
        cfg_files.append(family_dir / "_config.py")
    if (family_dir / "_model.py").is_file():
        model_files.append(family_dir / "_model.py")
    return cfg_files, model_files


def _check_configs(
    cfg_files: list[Path], family_dir: Path, issues: list[Issue]
) -> set[str]:
    """Check every ``<Family>Config`` class in the family.  Returns the
    set of recognised Config class names (used by model-class checks)."""
    config_names: set[str] = set()
    found_any = False
    for fp in cfg_files:
        tree = _parse(fp)
        if tree is None:
            continue
        for cls in _classes(tree):
            if not cls.name.endswith("Config"):
                continue
            if cls.name.startswith("_"):
                continue
            found_any = True
            config_names.add(cls.name)

            # MZ010: must inherit (directly or via a known intermediate) from
            # something that ends with ``Config``.  We accept any base name
            # ending with ``Config`` — covers ``ModelConfig`` /
            # ``LanguageModelConfig`` / ``DiffusionModelConfig`` / etc.
            bases = _base_names(cls)
            if not any(b.endswith("Config") for b in bases):
                issues.append(
                    Issue(
                        "error",
                        "MZ010",
                        fp,
                        cls.lineno,
                        f"{cls.name}: must inherit from ModelConfig (or a subclass).  "
                        f"Got bases={bases}.",
                    )
                )

            # MZ011: must be decorated with @dataclass(frozen=True).
            dataclass_args = _decorator_args_dict(cls, "dataclass")
            if "dataclass" not in _decorator_call_names(cls):
                # Bare ``@dataclass`` (no call) is not detected by our helper —
                # check the raw decorator list for the name too.
                bare_names = [
                    (
                        d.id
                        if isinstance(d, ast.Name)
                        else (d.attr if isinstance(d, ast.Attribute) else None)
                    )
                    for d in cls.decorator_list
                ]
                if "dataclass" not in bare_names:
                    issues.append(
                        Issue(
                            "error",
                            "MZ011",
                            fp,
                            cls.lineno,
                            f"{cls.name}: must be a @dataclass (frozen=True).",
                        )
                    )
            elif dataclass_args is not None:
                frozen = dataclass_args.get("frozen")
                if (
                    frozen is None
                    or _literal_str(frozen) == "False"
                    or (isinstance(frozen, ast.Constant) and frozen.value is False)
                ):
                    issues.append(
                        Issue(
                            "error",
                            "MZ012",
                            fp,
                            cls.lineno,
                            f"{cls.name}: @dataclass must use frozen=True.",
                        )
                    )

            # MZ020: @model_family_meta required, with all 3 string-literal args.
            meta_kwargs = _decorator_args_dict(cls, "model_family_meta")
            if meta_kwargs is None:
                issues.append(
                    Issue(
                        "error",
                        "MZ020",
                        fp,
                        cls.lineno,
                        f"{cls.name}: missing @model_family_meta(...) decorator.",
                    )
                )
            else:
                for key in ("canonical_name", "citation", "theory"):
                    val = meta_kwargs.get(key)
                    if val is None or _literal_str(val) in (None, ""):
                        issues.append(
                            Issue(
                                "error",
                                "MZ021",
                                fp,
                                cls.lineno,
                                f"{cls.name}: @model_family_meta is missing or empty "
                                f"'{key}=' string literal.",
                            )
                        )

            # MZ030: model_type ClassVar must be set to a non-empty, non-base id.
            mt = _class_var(cls, "model_type")
            mt_val = _literal_str(mt) if mt is not None else None
            if mt is None:
                issues.append(
                    Issue(
                        "error",
                        "MZ030",
                        fp,
                        cls.lineno,
                        f"{cls.name}: missing 'model_type: ClassVar[str] = \"...\"'.",
                    )
                )
            elif mt_val in (None, "", "base"):
                issues.append(
                    Issue(
                        "error",
                        "MZ031",
                        fp,
                        cls.lineno,
                        f"{cls.name}: model_type must be a unique family identifier "
                        f"(got {mt_val!r}).",
                    )
                )

    if not found_any:
        issues.append(
            Issue(
                "error",
                "MZ002",
                family_dir,
                1,
                f"family {family_dir.name}: no <Family>Config class found.",
            )
        )
    return config_names


def _check_models(
    model_files: list[Path],
    family_dir: Path,
    config_names: set[str],
    issues: list[Issue],
) -> None:
    """Check direct-model + task-wrapper + output dataclasses in _model.py
    (or each _v*.py for split families)."""
    direct_models_found = False
    task_wrappers_found = False
    for fp in model_files:
        tree = _parse(fp)
        if tree is None:
            continue
        for cls in _classes(tree):
            if cls.name.startswith("_"):
                continue
            bases = _base_names(cls)

            # Output dataclasses — name ends with Output. Skip further checks
            # (light contract: just the name pattern).
            if cls.name.endswith("Output"):
                continue
            # Config classes already checked in _check_configs (in split layout
            # they live in the same file).
            if cls.name.endswith("Config"):
                continue

            # MZ040: every non-output, non-config public class in _model.py
            # should inherit from PretrainedModel — directly for a bare
            # backbone, or through one of the eight task bases for a
            # head-bearing wrapper.  Checking the name alone flagged all
            # eighty-two wrappers the moment the task bases were
            # introduced, because AST cannot follow the indirection.
            if not ({"PretrainedModel"} | _TASK_BASES) & set(bases):
                issues.append(
                    Issue(
                        "warning",
                        "MZ040",
                        fp,
                        cls.lineno,
                        f"{cls.name}: public class in {fp.name} should inherit from "
                        f"PretrainedModel or a task base.  Bases={bases}.",
                    )
                )

            # Task wrapper detection — registered suffix wins first.  Then
            # try the ``For<CapWord>$`` regex to catch unknown task names.
            # ``Former`` substring false-positives (RoFormer / EfficientFormer)
            # are excluded by the regex's CapWord-tail anchor.
            known_match = next(
                (s for s in KNOWN_TASK_SUFFIXES if cls.name.endswith(s)),
                None,
            )
            if known_match is not None:
                task_wrappers_found = True
            else:
                m = _TASK_SUFFIX_RE.search(cls.name)
                if m is not None:
                    task_wrappers_found = True
                    issues.append(
                        Issue(
                            "warning",
                            "MZ050",
                            fp,
                            cls.lineno,
                            f"{cls.name}: '{m.group(0)}' suffix not in known task "
                            f"list — either rename to a registered task or extend "
                            f"_TASK_SUFFIX_MAP in build-api-data.py.",
                        )
                    )
                else:
                    # Direct model class.
                    direct_models_found = True

            # MZ060: must declare ``config_class`` ClassVar (any concrete model).
            cc = _class_var(cls, "config_class")
            if cc is None:
                issues.append(
                    Issue(
                        "warning",
                        "MZ060",
                        fp,
                        cls.lineno,
                        f"{cls.name}: missing 'config_class: ClassVar[...]' — "
                        f"required by PretrainedModel.from_pretrained().",
                    )
                )
    # MZ003: only flag when the family has *no* public model classes at all.
    # Detection / segmentation families intentionally ship only task-wrapper
    # classes (e.g. ``DETR`` only has ``DETRForObjectDetection``) — that is a
    # legitimate layout and shouldn't be warned about.
    if not direct_models_found and not task_wrappers_found:
        issues.append(
            Issue(
                "warning",
                "MZ003",
                family_dir,
                1,
                f"family {family_dir.name}: no public model class found.",
            )
        )


def _check_pretrained(family_dir: Path, issues: list[Issue]) -> None:
    """Check factory functions in _pretrained.py (or in _v*.py for split)."""
    if _has_split_layout(family_dir):
        targets = list(family_dir.glob("_v[0-9]*.py"))
    else:
        targets = [family_dir / "_pretrained.py"]
        if not targets[0].is_file():
            return
    any_factories = False
    for fp in targets:
        tree = _parse(fp)
        if tree is None:
            continue
        for fn in _functions(tree):
            if fn.name.startswith("_"):
                continue
            any_factories = True
            args = fn.args
            # MZ070: first arg must be ``pretrained: bool = False`` (after self
            # if any — but module-level functions have no self).
            first = (
                (args.posonlyargs + args.args)[0]
                if (args.posonlyargs or args.args)
                else None
            )
            if first is None or first.arg != "pretrained":
                issues.append(
                    Issue(
                        "warning",
                        "MZ070",
                        fp,
                        fn.lineno,
                        f"{fn.name}: factory should take 'pretrained: bool = False' "
                        f"as first parameter.",
                    )
                )
            # MZ071: must have a docstring.
            if not ast.get_docstring(fn):
                issues.append(
                    Issue(
                        "warning",
                        "MZ071",
                        fp,
                        fn.lineno,
                        f"{fn.name}: factory function is missing a docstring.",
                    )
                )
    if not any_factories:
        issues.append(
            Issue(
                "warning",
                "MZ004",
                family_dir,
                1,
                f"family {family_dir.name}: no public factory functions found.",
            )
        )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def iter_families(
    domain_filter: str | None, family_filter: str | None
) -> Iterable[Path]:
    domains = (domain_filter,) if domain_filter else DOMAINS
    for d in domains:
        dom_dir = MODELS_DIR / d
        if not dom_dir.is_dir():
            continue
        for fam in sorted(dom_dir.iterdir()):
            if not fam.is_dir() or fam.name.startswith("_"):
                continue
            if family_filter and fam.name != family_filter:
                continue
            yield fam


#: The task bases from ``lucid.models._tasks``.  A wrapper reaches
#: ``PretrainedModel`` through one of these, so MZ040 accepts them as
#: evidence of the contract just as it accepts the direct base.
_TASK_BASES = {
    "TaskModel",
    "ImageClassificationModel",
    "ImageGenerationModel",
    "ObjectDetectionModel",
    "SemanticSegmentationModel",
    "SequenceClassificationModel",
    "TokenClassificationModel",
    "LanguageModelingModel",
    "WorldModelingModel",
}


def _all_family_names() -> list[str]:
    """Every family directory name, for suggesting a near miss on a typo."""
    return [fam.name for fam in iter_families(None, None)]


def _check_protocols(fam: Path, issues: list[Issue]) -> None:
    """Layer-3.5 structural check via ``isinstance(cls, Protocol)``.

    Imports the family's `_config.py` module and verifies every
    ``<Family>Config`` class satisfies :class:`ModelConfigProtocol`
    *structurally* — independent of how it inherits.  Catches third-party
    or refactor mistakes where the AST passes but the runtime contract
    breaks (e.g. ``@model_family_meta`` swallowed by another decorator
    that strips ``__model_family_meta__``).
    """
    from lucid.models._protocols import ModelConfigProtocol

    domain, family_name = fam.parent.name, fam.name
    mod_path = f"lucid.models.{domain}.{family_name}"
    try:
        mod = importlib.import_module(mod_path)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            Issue(
                "error",
                "MZ100",
                fam,
                1,
                f"failed to import {mod_path}: {type(exc).__name__}: {exc}",
            )
        )
        return

    found_any = False
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if not isinstance(obj, type):
            continue
        if not name.endswith("Config"):
            continue
        if name in (
            "ModelConfig",
            "LanguageModelConfig",
            "DiffusionModelConfig",
            "GenerativeModelConfig",
        ):
            continue
        found_any = True
        if not isinstance(obj, ModelConfigProtocol):
            issues.append(
                Issue(
                    "error",
                    "MZ101",
                    fam,
                    1,
                    f"{name}: does not satisfy ModelConfigProtocol "
                    f"(missing one of: model_type, __model_family_meta__, "
                    f"__dataclass_fields__).",
                )
            )
    if not found_any:
        issues.append(
            Issue(
                "warning",
                "MZ102",
                fam,
                1,
                f"family {family_name}: no <Family>Config exported from __init__.py "
                f"— protocol check skipped.",
            )
        )


def validate(
    domain: str | None, family: str | None, strict: bool, runtime: bool = False
) -> int:
    issues: list[Issue] = []
    n_families = 0
    for fam in iter_families(domain, family):
        n_families += 1
        cfg_files, model_files = _check_layout(fam, issues)
        config_names = _check_configs(cfg_files, fam, issues)
        _check_models(model_files, fam, config_names, issues)
        _check_pretrained(fam, issues)
        if runtime:
            _check_protocols(fam, issues)

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    for i in issues:
        print(i.format())

    print(
        f"\n{n_families} families checked: "
        f"{len(errors)} error(s), {len(warnings)} warning(s)"
    )
    # A filter that matches nothing checked nothing, and "0 error(s)" over
    # zero families reads as a pass — which is how ``--family vq_vae``
    # (the directory is ``vqvae``) came back green.  A gate that turns
    # green on a typo is worse than no gate, so this is an error.
    if n_families == 0:
        asked = ", ".join(
            f"{k}={v!r}"
            for k, v in (("domain", domain), ("family", family))
            if v is not None
        )
        print(
            f"error: no family matched ({asked or 'no filter'}) — nothing was "
            f"validated.",
            file=sys.stderr,
        )
        if family is not None:
            near = difflib.get_close_matches(family, _all_family_names(), n=3)
            if near:
                print(f"       did you mean: {', '.join(near)}?", file=sys.stderr)
        return 1
    if errors or (strict and warnings):
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate Lucid model-zoo family structures."
    )
    p.add_argument("--domain", choices=DOMAINS, help="Limit to one domain.")
    p.add_argument("--family", help="Limit to one family directory name.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit 1 on any).",
    )
    p.add_argument(
        "--runtime",
        action="store_true",
        help="Additionally import each family module and verify "
        "every Config satisfies ModelConfigProtocol "
        "(structural isinstance check). Slower; catches "
        "decorator stripping / runtime-only issues.",
    )
    args = p.parse_args()
    return validate(args.domain, args.family, args.strict, args.runtime)


if __name__ == "__main__":
    sys.exit(main())
