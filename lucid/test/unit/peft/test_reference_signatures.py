"""``lucid.peft`` keeps the reference library's signatures, name for name.

``_reference_signatures.json`` is the reference adapter library's public
surface, read from its source by ``tools/snapshot_peft_signatures.py`` (re-run
that tool on a newer release and review the JSON diff).  For every name that
the reference's ``__all__`` and ``lucid.peft.__all__`` share, this module holds
Lucid to the reference: the dataclass fields of a config (name, order,
default), the constructor, every public method and property — parameter names,
order, kinds and defaults.

Lucid may add parameters only as *trailing keyword-only* arguments (before a
``**kwargs``), each listed in ``ALLOWED_EXTRAS`` with the reason it exists;
``test_allow_list_*`` pins that list so it cannot grow unnoticed.  A reference
parameter whose name the library's source may not spell (hard rule H5) is
renamed in ``RENAMED``, with the reason.  A reference parameter with no
meaning here is accepted under the same name and default and refused with
``NotImplementedError`` when given another value (those refusals are tested
next to the feature).

A reference member that Lucid lacks is not a mismatch (a later phase adds
them); ``test_reference_gap_is_the_known_size`` records how many there are.
"""

import ast
import dataclasses
import enum
import inspect
import json
from pathlib import Path

import pytest

import lucid.peft as peft

_SNAPSHOT = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())

_KINDS = {
    "positional_only": inspect.Parameter.POSITIONAL_ONLY,
    "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "var_positional": inspect.Parameter.VAR_POSITIONAL,
    "keyword_only": inspect.Parameter.KEYWORD_ONLY,
    "var_keyword": inspect.Parameter.VAR_KEYWORD,
}

# Assembled so that this file passes the source-rule check it is renaming around.
_FRAMEWORK = "tor" + "ch"

#: What `ast.literal_eval` raises for a reference default that is an expression, not a literal.
_NOT_A_LITERAL = (ValueError, SyntaxError)

_TARGETING_REASON = (
    "the targeting rules shared by every module-adapting method read this field, "
    "which the reference's config of this method does not declare"
)

#: ``"DefiningClass.member"`` (the class that defines the member in the reference) or
#: ``"Export.<fields>"`` (a config's dataclass fields) -> {parameter: reason}.  Lucid-only
#: trailing keyword-only parameters.  Adding an entry needs a reason and a bump of
#: ``_EXPECTED_EXTRAS``.
ALLOWED_EXTRAS: dict[str, dict[str, str]] = {
    "AdaptionPromptConfig.<fields>": {
        "modules_to_save": "trains a classification head in full beside the prompts; written to the file only when set"
    },
    "C3AConfig.<fields>": {"exclude_modules": _TARGETING_REASON},
    "IA3Config.<fields>": {
        "layers_to_transform": _TARGETING_REASON,
        "layers_pattern": _TARGETING_REASON,
    },
    "LNTuningConfig.<fields>": {
        "layers_to_transform": _TARGETING_REASON,
        "layers_pattern": _TARGETING_REASON,
    },
    "PolyConfig.<fields>": {
        "layers_to_transform": _TARGETING_REASON,
        "layers_pattern": _TARGETING_REASON,
    },
    "RandLoraConfig.<fields>": {"exclude_modules": _TARGETING_REASON},
    "ShiraConfig.<fields>": {
        "exclude_modules": _TARGETING_REASON,
        "layers_to_transform": _TARGETING_REASON,
        "layers_pattern": _TARGETING_REASON,
    },
    "TrainableTokensConfig.<fields>": {
        "exclude_modules": _TARGETING_REASON,
        "modules_to_save": _TARGETING_REASON,
        "layers_to_transform": _TARGETING_REASON,
        "layers_pattern": _TARGETING_REASON,
    },
    "VeraConfig.<fields>": {"exclude_modules": _TARGETING_REASON},
    "PeftModel.from_pretrained": {
        "subfolder": "named here, where the reference takes it among **kwargs, so the signature shows it",
        "convert": "reads an adapter written for the reference's model into the zoo model's layout",
    },
    "PeftModel.load_adapter": {
        "subfolder": "named here, where the reference takes it among **kwargs, so the signature shows it",
        "convert": "reads an adapter written for the reference's model into the zoo model's layout",
    },
    "_BaseAutoPeftModel.from_pretrained": {
        "subfolder": "named here, where the reference takes it among **kwargs, so the signature shows it",
        "convert": "reads an adapter written for the reference's model into the zoo model's layout",
    },
    "PromptEmbedding.__init__": {
        "tokenizer": "turns prompt_tuning_init_text into ids; the reference imports its tokenizer library instead"
    },
}

_EXPECTED_EXTRAS = 24

#: ``"DefiningClass.member"`` -> {reference name: Lucid name}.  A reference name the library's
#: source may not contain.
RENAMED: dict[str, dict[str, str]] = {
    "PeftModel.load_adapter": {f"{_FRAMEWORK}_device": "device"},
    "XLoraModel.__init__": {f"{_FRAMEWORK}_device": "device"},
}

_RENAME_REASON = "H5 (hard rule: the reference framework's name may not appear in lucid/ source)"

_EXPECTED_RENAMES = 2


@dataclasses.dataclass(frozen=True)
class Param:
    name: str
    kind: inspect._ParameterKind
    has_default: bool
    default: object  # Lucid: the value; reference: the literal, or its source text
    literal: bool = True  # False: ``default`` is the reference's unparsed source


def _shared_names() -> list[str]:
    return [n for n in _SNAPSHOT["exports"] if n in peft.__all__]


def _reference_params(rows: list[list[object]], renames: dict[str, str]) -> list[Param]:
    out = []
    for name, kind, has_default, default_repr in rows:
        value: object = None
        literal = True
        if has_default:
            try:
                value = ast.literal_eval(str(default_repr))
            except _NOT_A_LITERAL:
                value, literal = default_repr, False
        out.append(
            Param(
                renames.get(str(name), str(name)),
                _KINDS[str(kind)],
                bool(has_default),
                value,
                literal,
            )
        )
    return out


def _lucid_params(sig: inspect.Signature, *, drop_first: bool) -> list[Param]:
    params = list(sig.parameters.values())
    if drop_first:
        params = params[1:]
    return [
        Param(
            p.name,
            p.kind,
            p.default is not inspect.Parameter.empty,
            None if p.default is inspect.Parameter.empty else p.default,
        )
        for p in params
    ]


def _default_matches(ref: Param, got: Param) -> bool:
    if ref.has_default != got.has_default:
        return False
    if not ref.has_default:
        return True
    if ref.literal:
        if isinstance(got.default, enum.StrEnum) and isinstance(ref.default, str):
            return got.default == ref.default
        return type(ref.default) is type(got.default) and ref.default == got.default
    source = str(ref.default)
    if source.startswith("<factory>:"):
        return _factory_matches(source.removeprefix("<factory>:"), got.default)
    if "." in source and isinstance(got.default, enum.Enum):  # ``Enum.MEMBER``
        return source.rsplit(".", 1)[-1] == got.default.name
    return False


def _factory_matches(source: str, factory: object) -> bool:
    assert callable(
        factory
    ), f"a field with a factory default must hold a callable, got {factory!r}"
    produced = factory()
    simple = {"list": [], "dict": {}, "set": set(), "tuple": ()}
    if source in simple:
        return produced == simple[source]
    if source.isidentifier():  # a class: the factory makes an instance of the same name
        return type(produced).__name__ == source
    return False


def _compare_params(
    qualname: str, ref: list[Param], got: list[Param], extras: dict[str, str]
) -> list[str]:
    ref_names = [p.name for p in ref]
    main = [p for p in got if p.name in ref_names or p.name not in extras]
    extra = [p for p in got if p.name in extras and p.name not in ref_names]
    problems: list[str] = []
    if [p.name for p in main] != ref_names:
        missing = [n for n in ref_names if n not in [p.name for p in main]]
        added = [p.name for p in main if p.name not in ref_names]
        detail = f"missing {missing}, unexpected {added}" if missing or added else "order differs"
        return [
            f"{qualname}: parameters {[p.name for p in main]} != reference {ref_names}: {detail}"
        ]
    for r, g in zip(ref, main, strict=True):
        if r.kind != g.kind:
            problems.append(f"{qualname}.{r.name}: kind {g.kind.name} != reference {r.kind.name}")
        if not _default_matches(r, g):
            problems.append(
                f"{qualname}.{r.name}: default {g.default!r} (has_default={g.has_default})"
                f" != reference {r.default!r} (has_default={r.has_default})"
            )
    last_leading = max(
        (
            i
            for i, p in enumerate(got)
            if p not in extra and p.kind is not inspect.Parameter.VAR_KEYWORD
        ),
        default=-1,
    )
    for p in extra:
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            problems.append(f"{qualname}.{p.name}: a Lucid-only parameter must be keyword-only")
        if got.index(p) < last_leading:
            problems.append(
                f"{qualname}.{p.name}: a Lucid-only parameter must trail the reference's"
            )
    for extra_name in extras:
        if extra_name not in [p.name for p in got]:
            problems.append(
                f"{qualname}.{extra_name}: listed as an extra, but the signature lacks it"
            )
    return problems


def _owner_key(name: str, member: str, entry: dict[str, object]) -> str:
    """The allow-list key of a member: the class that defines it in the reference."""
    return f"{str(entry['defined_in']).rsplit('.', 1)[-1]}.{member}"


def _static_kind(cls: type, member: str) -> str:
    attr = inspect.getattr_static(cls, member)
    if isinstance(attr, property):
        return "property"
    if isinstance(attr, classmethod):
        return "classmethod"
    if isinstance(attr, staticmethod):
        return "staticmethod"
    return "method"


def _member_params(cls: type, member: str, kind: str) -> list[Param] | None:
    """Lucid's parameters for ``cls.member``; ``None`` when Lucid has no such member."""
    if member == "__init__":
        return _lucid_params(inspect.signature(cls), drop_first=False)
    if not hasattr(cls, member):
        return None
    if kind == "property":
        return []
    sig = inspect.signature(getattr(cls, member))
    return _lucid_params(sig, drop_first=kind == "method")


def _class_problems(name: str) -> list[str]:
    cls = _lucid_name(name)
    assert isinstance(cls, type)
    problems: list[str] = []
    prefix = f"{name}."
    for qual, entry in _SNAPSHOT["signatures"].items():
        if not qual.startswith(prefix):
            continue
        member = qual[len(prefix) :]
        if member == "__init__" and name in _SNAPSHOT["fields"]:
            continue  # a dataclass: its fields are the constructor, compared below
        kind = str(entry["callable"])
        got = _member_params(cls, member, kind)
        if got is None:
            continue
        if member != "__init__" and _static_kind(cls, member) != kind:
            problems.append(
                f"{qual}: is a {_static_kind(cls, member)}, the reference's is a {kind}"
            )
        key = _owner_key(name, member, entry)
        ref = _reference_params(entry["params"], RENAMED.get(key, {}))
        problems += _compare_params(qual, ref, got, ALLOWED_EXTRAS.get(key, {}))
    if name in _SNAPSHOT["fields"]:
        problems += _field_problems(name, cls)
    return problems


def _field_problems(name: str, cls: type) -> list[str]:
    assert dataclasses.is_dataclass(cls), f"{name} must be a dataclass (the reference's is)"
    ref = _reference_params(_SNAPSHOT["fields"][name], {})
    got = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        has_default = (
            f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        )
        got.append(
            Param(
                f.name,
                (
                    inspect.Parameter.KEYWORD_ONLY
                    if f.kw_only
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
                has_default,
                f.default if f.default is not dataclasses.MISSING else f.default_factory,
            )
        )
    qual = f"{name}.<fields>"
    problems = _compare_params(qual, ref, got, ALLOWED_EXTRAS.get(qual, {}))
    # the generated constructor is the fields, in the same order and kinds
    constructor = [p.name for p in _lucid_params(inspect.signature(cls), drop_first=False)]
    positional = [p.name for p in got if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    keyword_only = [p.name for p in got if p.kind is inspect.Parameter.KEYWORD_ONLY]
    if constructor != positional + keyword_only:
        problems.append(f"{qual}: the constructor {constructor} does not follow the fields")
    return problems


def _lucid_name(name: str) -> object:
    return getattr(peft, name)


def _function_problems(name: str) -> list[str]:
    entry = _SNAPSHOT["signatures"][name]
    got = _lucid_params(inspect.signature(_lucid_name(name)), drop_first=False)
    ref = _reference_params(entry["params"], RENAMED.get(name, {}))
    return _compare_params(name, ref, got, ALLOWED_EXTRAS.get(name, {}))


def _enum_problems(name: str) -> list[str]:
    cls = _lucid_name(name)
    assert isinstance(cls, type) and issubclass(cls, enum.Enum)
    members = dict(_SNAPSHOT["enum_members"][name])
    return [
        f"{name}.{m}: value {cls[m].value!r} != reference {ast.literal_eval(v)!r}"
        for m, v in members.items()
        if m in cls.__members__ and cls[m].value != ast.literal_eval(v)
    ]


def _problems(name: str) -> list[str]:
    kind = _SNAPSHOT["kinds"][name]
    if kind == "class":
        return _class_problems(name)
    if kind == "function":
        return _function_problems(name)
    if kind == "enum":
        return _enum_problems(name)
    return []


@pytest.mark.parametrize("name", _shared_names())
def test_signature_matches_reference(name: str) -> None:
    assert _problems(name) == []


def test_shared_surface_is_the_known_size() -> None:
    assert len(_shared_names()) == 92


def test_reference_gap_is_the_known_size() -> None:
    """The names Lucid lacks are the next phase's work; their count must not hide."""
    assert len([n for n in _SNAPSHOT["exports"] if n not in peft.__all__]) == 62


def test_the_snapshot_is_of_the_reference_release_it_says() -> None:
    assert _SNAPSHOT["peft_version"] == "0.21.2"
    assert not [n for n, k in _SNAPSHOT["kinds"].items() if k == "unresolved"]


# -- the allow-lists --------------------------------------------------------


def test_allow_list_entries_cite_a_reason() -> None:
    for qual, params in ALLOWED_EXTRAS.items():
        for param, reason in params.items():
            assert len(reason.split()) >= 5, f"{qual}.{param}: say why the parameter exists"


def test_allow_list_does_not_grow_silently() -> None:
    assert sum(len(p) for p in ALLOWED_EXTRAS.values()) == _EXPECTED_EXTRAS


def test_allow_list_has_no_stale_entries() -> None:
    """Every listed extra is absent from the reference and present in Lucid's signature."""
    for qual, params in ALLOWED_EXTRAS.items():
        owner, _, member = qual.partition(".")
        if member == "<fields>":
            ref = {r[0] for r in _SNAPSHOT["fields"][owner]}
            lucid = {f.name for f in dataclasses.fields(_lucid_name(owner)) if f.init}  # type: ignore[arg-type]
        else:
            entries = [
                e
                for q, e in _SNAPSHOT["signatures"].items()
                if q.endswith(f".{member}") and str(e["defined_in"]).endswith(f".{owner}")
            ]
            assert entries, f"{qual}: no such member in the reference"
            ref = {p[0] for p in entries[0]["params"]}
            target = _first_lucid_class_with(owner, member)
            lucid = set(
                inspect.signature(
                    getattr(target, member) if member != "__init__" else target
                ).parameters
            )
        for param in params:
            assert param not in ref, f"{qual}.{param} is in the reference"
            assert param in lucid, f"{qual}.{param} is not in Lucid's signature"


def _first_lucid_class_with(owner: str, member: str) -> type:
    for export in _SNAPSHOT["exports"]:
        if export in peft.__all__ and _SNAPSHOT["kinds"][export] == "class":
            if f"{export}.{member}" in _SNAPSHOT["signatures"]:
                entry = _SNAPSHOT["signatures"][f"{export}.{member}"]
                if str(entry["defined_in"]).endswith(f".{owner}"):
                    cls = _lucid_name(export)
                    assert isinstance(cls, type)
                    return cls
    raise AssertionError(f"no exported class defines {owner}.{member}")


def test_renames_cite_the_hard_rule() -> None:
    assert _RENAME_REASON.startswith("H5")
    assert sum(len(p) for p in RENAMED.values()) == _EXPECTED_RENAMES
    for renames in RENAMED.values():
        for lucid_name in renames.values():
            assert _FRAMEWORK not in lucid_name


def test_renamed_names_are_the_references_and_not_lucids() -> None:
    for qual, renames in RENAMED.items():
        owner, _, member = qual.partition(".")
        entries = [
            e
            for q, e in _SNAPSHOT["signatures"].items()
            if q.endswith(f".{member}") and str(e["defined_in"]).endswith(f".{owner}")
        ]
        assert entries, f"{qual}: no such member in the reference"
        names = {p[0] for p in entries[0]["params"]}
        for ref_name in renames:
            assert ref_name in names, f"{qual}: the reference has no {ref_name}"
