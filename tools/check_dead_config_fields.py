#!/usr/bin/env python3
"""tools/check_dead_config_fields.py — a Config field the model never reads.

A model family's ``_config.py`` is a frozen dataclass, and nothing
checks that the fields it declares reach the forward pass. They
routinely do not: the config is usually written before the model, and a
field that never gets wired changes no shape, raises nothing and is
invisible to every other gate.

V-JEPA 2 shipped with four of them. ``drop_path_rate`` was the clearest:

    rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
    blocks = [_Block(...) for _ in rates]     # rates counted, never used

The per-depth rates were computed and thrown away, ``_Block`` had no
DropPath at all, and the validator, the family contract test, ``mypy
--strict``, ruff and the summary builder all passed. A user setting
``drop_path_rate=0.3`` got a model identical to ``0.0``.

How a field is credited
-----------------------
This is a *name* check, not a dataflow one, and it credits a field
three ways. The first version had only the first of them and
over-reported by more than half — ten of the seventeen fields it
flagged were reached by one of the other two:

``own``
    The name appears somewhere in the family's own sources outside the
    config that declares it. The ordinary case.

``derived``
    The name appears in ``_config.py`` itself, inside a method other
    than ``__post_init__`` or a ``_validate*`` helper — a derived
    property the model reads. ``flow_matching.exact_trace_max_dim``
    only ever reaches the model through ``resolved_trace_method``.
    Validators are excluded deliberately: a field that is only
    range-checked still does nothing.

``config-read``
    Some other file in the zoo reads it *off a config* —
    ``config.x`` / ``cfg.x`` / ``self._cfg.x`` / ``getattr(cfg, "x")``.
    Mask R-CNN's ``rpn_*`` thresholds are consumed by Faster R-CNN's
    shared proposal layer, and the text families' ``eos_token_id`` by
    ``GenerationMixin``; neither name occurs in the owning family at
    all. The receiver is required — matching a bare ``.x`` credited
    ``ddpm.clip_denoised`` to the *scheduler's* own attribute of that
    name, which is a different variable that happens to agree.

Two things it still cannot see, both worth knowing:

* A name mentioned but overridden at the call site, as
  ``uniform_power`` was by ``_sincos_3d(dim, grid, False)``.
* Whose config a ``config-read`` belongs to. A field name shared
  across a domain's configs is credited domain-wide, so
  ``LanguageModelConfig``'s token ids count as read for every text
  family once one of them reads them.

Read the call sites when adding a family; this catches the half a
reader would otherwise have to hold in their head. ``--list`` prints
the rule that credited each field so the verdict can be audited rather
than trusted.

Everything left over is in ``_ALLOWED`` with a reason. There is no
untriaged baseline: a field that is neither read nor answered for
fails the gate.

Run::

    python tools/check_dead_config_fields.py
    python tools/check_dead_config_fields.py --list   # every field, with its verdict

Exit codes
----------
0 — every declared field is read by one of the three rules, or allowed.
1 — at least one is neither.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "lucid" / "models"

#: Fields that are declared and legitimately unread, each with the
#: reason.  Keep every entry answerable: an unexplained one is how a
#: real dead field hides.
_ALLOWED: dict[tuple[str, str], str] = {
    ("*", "model_type"): ("ClassVar the registry reads, not a family-local parameter"),
    ("vjepa", "sampling_rate"): (
        "records the released clip's frame stride; the model is handed an " "already-sampled clip"
    ),
    ("diamond", "burn_in"): (
        "steps of real experience an imagination loop replays to warm the "
        "LSTM state before rollout; the family ships the denoiser and its "
        "heads, not the loop, so the paper's value is recorded for whoever "
        "writes one"
    ),
    ("genie", "action_encoder_head_dim"): (
        "read as getattr(self, f'{stack}_head_dim') by "
        "GenieConfig.attention_head_dim, which _model.py calls per stack — "
        "the literal name occurs nowhere for a name search to find"
    ),
    ("genie", "action_decoder_head_dim"): (
        "read as getattr(self, f'{stack}_head_dim') by "
        "GenieConfig.attention_head_dim, which _model.py calls per stack — "
        "the literal name occurs nowhere for a name search to find"
    ),
    ("bert", "position_embedding_type"): (
        "single-valued Literal['absolute'] recording which positional "
        "scheme the family implements; the type admits no other value, so "
        "it documents rather than selects"
    ),
    ("roformer", "position_embedding_type"): (
        "single-valued Literal['rotary'] recording which positional scheme "
        "the family implements; the type admits no other value, so it "
        "documents rather than selects"
    ),
}

#: ``__post_init__`` and ``_validate*`` in a config are excluded from
#: the ``derived`` rule: range-checking a field is not using it.
_VALIDATORS = ("__post_init__",)


def _declared_fields(config_path: Path) -> list[str]:
    """Public annotated names on every class in a family's ``_config.py``."""
    tree = ast.parse(config_path.read_text())
    names: list[str] = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in cls.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.append(node.target.id)
    return names


def _derived_source(config_path: Path) -> str:
    """The config's own methods, minus the ones that only validate."""
    text = config_path.read_text()
    lines = text.splitlines()
    chunks: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in _VALIDATORS or node.name.startswith("_validate"):
            continue
        chunks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n".join(chunks)


def _config_read(field: str) -> re.Pattern[str]:
    """``config.field`` / ``cfg.field`` / ``getattr(cfg, "field")``.

    The receiver is part of the pattern on purpose.  A bare ``.field``
    matches any object's attribute, and the zoo has several that share
    a config field's name without being it.
    """
    name = re.escape(field)
    holder = r"(?:config|cfg|_cfg|_config)"
    return re.compile(
        rf"(?:\b{holder}\.{name}\b)" rf"|(?:getattr\(\s*(?:self\.)?{holder}\s*,\s*[\"']{name}[\"'])"
    )


def _families() -> list[tuple[str, Path]]:
    """Every family directory holding both a config and a model."""
    found = []
    for config in sorted(MODELS_ROOT.rglob("_config.py")):
        model = config.with_name("_model.py")
        if model.is_file():
            found.append((config.parent.name, config.parent))
    return found


def _allowed(family: str, field: str) -> str | None:
    return _ALLOWED.get((family, field)) or _ALLOWED.get(("*", field))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--list", action="store_true", help="print every field and its verdict")
    parser.add_argument("--family", help="only this family")
    args = parser.parse_args()

    # Read the zoo once — the config-read rule scans all of it per field.
    zoo = {p: p.read_text() for p in MODELS_ROOT.rglob("*.py")}

    dead: list[tuple[str, str]] = []
    checked = 0
    for family, family_dir in _families():
        if args.family and family != args.family:
            continue
        # Every .py in the family except the config itself: a factory
        # that selects a variant reads the field and the model never
        # sees it, which is not the failure this looks for.
        own = [f.read_text() for f in sorted(family_dir.rglob("*.py")) if f.name != "_config.py"]
        derived = _derived_source(family_dir / "_config.py")
        outside = {p: t for p, t in zoo.items() if family_dir not in p.parents}

        for field in _declared_fields(family_dir / "_config.py"):
            checked += 1
            if any(field in text for text in own):
                if args.list:
                    print(f"  own         {family}.{field}")
                continue
            if field in derived:
                if args.list:
                    print(f"  derived     {family}.{field}")
                continue
            pattern = _config_read(field)
            reader = next((p for p, t in outside.items() if pattern.search(t)), None)
            if reader is not None:
                if args.list:
                    where = reader.relative_to(MODELS_ROOT)
                    print(f"  config-read {family}.{field} — {where}")
                continue
            reason = _allowed(family, field)
            if reason is not None:
                if args.list:
                    print(f"  allowed     {family}.{field} — {reason}")
                continue
            dead.append((family, field))
            if args.list:
                print(f"  DEAD        {family}.{field}")

    if dead:
        print(
            f"[check_dead_config_fields] {len(dead)} field(s) declared but " f"read by nothing:",
            file=sys.stderr,
        )
        for family, field in dead:
            print(f"  {family}.{field}", file=sys.stderr)
        print(
            "\n  Either wire the field into the forward pass, or delete it. If it "
            "is genuinely decorative, add it to _ALLOWED in this file with the "
            "reason — an entry nobody can answer is how a real one hides. Run "
            "with --list to see which rule credited every other field.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[check_dead_config_fields] OK — {checked} field(s) across "
        f"{len(_families())} families, every one read or answered for."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
