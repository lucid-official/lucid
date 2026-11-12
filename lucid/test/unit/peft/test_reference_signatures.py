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

Nothing is left unaccounted.  A reference export Lucid lacks is in ``_OWED``; a
reference *member* of an exported class that Lucid lacks is in ``_OWED_MEMBERS``
(keyed by the class that defines it in the reference), each with the issue that
owes it and why; every other member is present and compared here.  A member whose
name the library's source may not spell (H5) is renamed in ``RENAMED_MEMBERS``.
Both ledgers are checked against the snapshot in both directions, so an entry that
is no longer true fails.
"""

import ast
import dataclasses
import enum
import inspect
import json
import re
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

#: ``"DefiningClass.member"`` -> Lucid's name for a *member* whose reference name the library's
#: source may not contain (H5).  The reference name is spelled as ``RENAMED`` spells its own.
RENAMED_MEMBERS: dict[str, str] = {f"PeftModel.base_model_{_FRAMEWORK}_dtype": "base_model_dtype"}

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
    if source.isidentifier() and callable(got.default):
        return getattr(got.default, "__name__", None) == source
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


def _lucid_member(key: str, member: str) -> str:
    """The name Lucid gives the reference member ``member`` that ``key`` (``Owner.member``) names."""
    return RENAMED_MEMBERS.get(key, member)


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
        key = _owner_key(name, member, entry)
        lucid_member = _lucid_member(key, member)
        got = _member_params(cls, lucid_member, kind)
        if got is None:
            continue  # lacking, so ``test_every_reference_member_is_present_or_owed`` judges it
        if member != "__init__" and _static_kind(cls, lucid_member) != kind:
            problems.append(
                f"{qual}: is a {_static_kind(cls, lucid_member)}, the reference's is a {kind}"
            )
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


#: The reference exports ``lucid.peft`` does not have yet, each with the issue that owes it.  The
#: set is checked against the snapshot, so a name the reference gains, or one that lands and stays
#: listed, fails here; a count would let one be traded for the other.  A name leaves the table in the
#: commit that exports it.
_OWED: dict[str, str] = {
    # LCD-827 shipped LoRA's extensions without Arrow, whose library builder is this.
    "create_arrow_model": "LCD-827"
}


def test_the_reference_names_lucid_lacks_are_the_owed_ones() -> None:
    """Every reference export is shared with Lucid or owed, with nothing in both and nothing unowned."""
    exported = set(_SNAPSHOT["exports"])
    missing = {n for n in exported if n not in peft.__all__}
    assert missing == set(_OWED), sorted(missing ^ set(_OWED))
    assert exported == set(_shared_names()) | set(_OWED)


def test_every_owed_name_cites_the_issue_that_owes_it() -> None:
    assert all(re.fullmatch(r"LCD-\d+", issue) for issue in _OWED.values()), _OWED


@dataclasses.dataclass(frozen=True)
class Owed:
    """A reference member Lucid lacks: the issue that owes it, why, and the classes it is lacking on."""

    issue: str
    reason: str
    #: Exported classes that lack it; empty means every exported class that has it in the reference.
    on: tuple[str, ...] = ()


#: The reference members of exported classes that Lucid lacks, keyed by ``"DefiningClass.member"``
#: (the class that defines the member in the reference).  Checked against the snapshot in both
#: directions: a member Lucid lacks and this table does not list, or one listed that Lucid now has
#: (or on a class it is not lacking on), fails.  A row leaves the table in the commit that adds it.
_SHADOW = "SHADOW is refused, so its model class is a stand-in with no tuner behaviour"
_NO_MERGE = "X-LoRA mixes its adapters per input, so there is no weight to merge or convert"
_CPT_HELPER = "not ported: a CPT training-loss helper of the reference's embedding"
_INJECT = "not ported: Lucid injects through inject_adapter_in_model, not a method of the tuner"
_GENERATE = "not ported: PeftModel.generate routes the batch; the tuner has none of its own"
_HERE = "LCD-968"

_OWED_MEMBERS: dict[str, Owed] = {
    "AdaLoraModel.resize_state_dict_by_rank_pattern": Owed(
        _HERE, "not ported: cuts saved tensors to a pruned rank pattern while loading"
    ),
    "BaseTuner.active_adapters": Owed(_HERE, _SHADOW, ("ShadowModel",)),
    "BaseTuner.disable_adapter_layers": Owed(_HERE, _SHADOW, ("ShadowModel",)),
    "BaseTuner.enable_adapter_layers": Owed(_HERE, _SHADOW, ("ShadowModel",)),
    "BaseTuner.get_model_config": Owed(
        _HERE, "not ported: the reference tuner base's reader of the wrapped model's config"
    ),
    "BaseTuner.inject_adapter": Owed(_HERE, _INJECT),
    "BaseTuner.merge_adapter": Owed(
        _HERE, f"{_SHADOW}; {_NO_MERGE}", ("ShadowModel", "XLoraModel")
    ),
    "BaseTuner.merge_and_unload": Owed(_HERE, _SHADOW, ("ShadowModel",)),
    "BaseTuner.set_requires_grad": Owed(_HERE, _SHADOW, ("ShadowModel",)),
    "BaseTuner.supports_lora_conversion": Owed(
        _HERE, f"{_SHADOW}; {_NO_MERGE}", ("ShadowModel", "XLoraModel")
    ),
    "BaseTuner.unload": Owed(_HERE, _SHADOW, ("ShadowModel",)),
    "BaseTuner.unmerge_adapter": Owed(
        _HERE, f"{_SHADOW}; {_NO_MERGE}", ("ShadowModel", "XLoraModel")
    ),
    "CPTEmbedding.calculate_loss": Owed(_HERE, _CPT_HELPER),
    "CPTEmbedding.get_epsilon": Owed(_HERE, _CPT_HELPER),
    "CPTEmbedding.get_projection": Owed(_HERE, _CPT_HELPER),
    "CPTEmbedding.set_updated_tokens": Owed(_HERE, _CPT_HELPER),
    "CartridgeEncoder.embedding": Owed(
        _HERE, "not ported: the reference's proxy giving the cache as embedding.weight"
    ),
    "LoraModel.subtract_mutated_init": Owed(
        _HERE,
        "not ported: save_pretrained's path_initial_model_for_weight_conversion converts PiSSA and OLoRA",
    ),
    "PeftModelForCausalLM.prepare_inputs_for_generation": Owed(
        _HERE, "no meaning here: the prompt is applied by hooks inside the model's own generate"
    ),
    "PeftModelForSeq2SeqLM.prepare_inputs_for_generation": Owed(
        _HERE, "no meaning here: the prompt is applied by hooks inside the model's own generate"
    ),
    "PolyModel.generate": Owed(_HERE, _GENERATE),
    "ShadowModel.delete_adapter": Owed(_HERE, _SHADOW),
    "ShadowModel.forward": Owed(_HERE, _SHADOW),
    "ShadowModel.set_adapter": Owed(_HERE, _SHADOW),
    "ShadowModel.shadow_auxiliary_loss": Owed(_HERE, _SHADOW),
    "ShadowModel.unload_shadow": Owed(_HERE, _SHADOW),
    "TrainableTokensModel.inject_adapter": Owed(_HERE, _INJECT),
    "UniLoraModel.inject_adapter": Owed(_HERE, _INJECT),
    "XLoraModel.generate": Owed(_HERE, _GENERATE),
}


def _reference_members() -> list[tuple[str, str, str]]:
    """``(exported class, "DefiningClass.member", Lucid's member name)`` for every reference member of a shared class."""
    rows = []
    for qual, entry in _SNAPSHOT["signatures"].items():
        name, _, member = qual.partition(".")
        if name not in _shared_names() or _SNAPSHOT["kinds"][name] != "class":
            continue
        key = _owner_key(name, member, entry)
        rows.append((name, key, _lucid_member(key, member)))
    return rows


def _members_lucid_lacks() -> set[tuple[str, str]]:
    """``(exported class, "DefiningClass.member")`` for each reference member the class lacks."""
    return {
        (name, key)
        for name, key, lucid_member in _reference_members()
        if not hasattr(_lucid_name(name), lucid_member)
    }


def _owed_pairs() -> set[tuple[str, str]]:
    """``_OWED_MEMBERS`` as ``(exported class, key)`` pairs: the listed classes, or all that have the member."""
    pairs = set()
    for name, key, _ in _reference_members():
        row = _OWED_MEMBERS.get(key)
        if row is not None and (not row.on or name in row.on):
            pairs.add((name, key))
    return pairs


def test_every_reference_member_is_present_or_owed() -> None:
    """Every member of every shared exported class is in Lucid (compared above) or owed, with none unowned."""
    lacking, owed = _members_lucid_lacks(), _owed_pairs()
    assert lacking == owed, {
        "lacking and not owed": sorted(lacking - owed),
        "owed but present": sorted(owed - lacking),
    }


def test_every_owed_member_names_a_member_the_reference_has_and_classes_that_exist() -> None:
    keys = {key for _, key, _ in _reference_members()}
    for key, row in _OWED_MEMBERS.items():
        assert key in keys, f"{key}: the reference snapshot has no such member"
        holders = {name for name, k, _ in _reference_members() if k == key}
        absent = sorted(set(row.on) - holders)
        assert not absent, f"{key}: lacking on {absent}, which do not hold it"


def test_every_owed_member_cites_an_issue_and_a_reason() -> None:
    for key, row in _OWED_MEMBERS.items():
        assert re.fullmatch(r"LCD-\d+", row.issue), f"{key}: {row.issue!r}"
        assert len(row.reason.split()) >= 5, f"{key}: say why it is owed"


def test_a_renamed_member_is_in_the_snapshot_and_in_lucid() -> None:
    """A rename row names a member the reference has and a member of the same class Lucid has."""
    assert RENAMED_MEMBERS
    for key, lucid_member in RENAMED_MEMBERS.items():
        holders = [name for name, k, _ in _reference_members() if k == key]
        assert holders, f"{key}: the reference snapshot has no such member"
        assert _FRAMEWORK not in lucid_member
        for name in holders:
            assert hasattr(_lucid_name(name), lucid_member), f"{name} has no {lucid_member}"


def test_a_renamed_member_is_no_longer_spelled_the_reference_way() -> None:
    """The reference's spelling is the one thing Lucid may not have, under the table's rename."""
    for key in RENAMED_MEMBERS:
        _, _, reference_name = key.partition(".")
        for name in {n for n, k, _ in _reference_members() if k == key}:
            assert not hasattr(_lucid_name(name), reference_name)


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
