"""The module names a user brings from the reference library select modules of the zoo's classes.

Two ways in, both held here.  A config that leaves ``target_modules`` out takes a default per
``model_type``, and the reference's default tables (read from its source by
``tools/snapshot_peft_targets.py`` into ``_reference_targets.json``, release 0.21.2) are the
yardstick: for every method this library has and every model type the zoo shares with a row,
the reference's names — through the aliases — are the zoo's default.  A config that names modules
(``["query", "value"]`` from a ViT recipe, ``["q_proj", "v_proj"]`` from a CLIP one) must
reach modules too: the text models are named alike on both sides (GPT-2, BERT, RoFormer, T5)
and need nothing, the vision and multimodal families have a per-type alias in
``_target_aliases.py``.

Models are built under ``shadow_alloc`` at their paper size, for every factory of a type, so a
name that holds for the tiny factory but not a sibling fails too.
"""

import functools
import json
import types
import warnings
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.models._registry import _REGISTRY
from lucid.nn._shadow import shadow_alloc
from lucid.peft import (
    AdaLoraConfig,
    LNTuningConfig,
    LoraConfig,
    LoraModel,
    NoMatchingPeftModuleError,
    PeftModel,
    PeftType,
    get_peft_model,
)
from lucid.peft._config import _TargetedConfig
from lucid.peft._target_aliases import REFERENCE_TARGET_ALIASES, _zoo_target_names
from lucid.peft._targeting import _module_names_matching, _selects
from lucid.peft.tuners._base import _PENDING_TUNERS, _REFUSED_TUNERS, _TUNERS
from lucid.peft.tuners.ia3._defaults import IA3_DEFAULT_FEEDFORWARD_MODULES
from lucid.test.unit.models._peft_matrix import _refusal
from lucid.test.unit.models._peft_rows import ROWS

_SNAPSHOT = json.loads((Path(__file__).parent / "_reference_targets.json").read_text())
_SIGNATURES = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())

#: The reference's table key where it is not the method's ``PeftType`` name.
_PEFT_TYPE_OF = {"LNTUNING": "LN_TUNING"}
_FEEDFORWARD = "IA3_FEEDFORWARD"
_ALIAS_WARNING = "reference library's names"

def _rows(table: str) -> dict[str, list[str] | str]:
    """The table ``table`` as the reference leaves it: its base's rows, edited by its own."""
    entry = _SNAPSHOT["tables"][table]
    rows = {} if entry["base"] is None else _rows(entry["base"])
    rows = {k: v for k, v in rows.items() if k not in entry["removed"]}
    return rows | entry["rows"]


def _peft_type(table: str) -> PeftType | None:
    name = _PEFT_TYPE_OF.get(table, table.removesuffix("_FEEDFORWARD"))
    return PeftType[name] if name in PeftType.__members__ else None


def _lucid_default(table: str, model_type: str) -> frozenset[str] | None:
    """The names this library's tuner for ``table`` takes by default for ``model_type``."""
    if table == _FEEDFORWARD:
        return IA3_DEFAULT_FEEDFORWARD_MODULES.get(model_type)
    peft_type = _peft_type(table)
    tuner = None if peft_type is None else _TUNERS.get(peft_type)
    return None if tuner is None else tuner._default_targets.get(model_type)


def _has_tuner(table: str) -> bool:
    peft_type = _peft_type(table)
    return peft_type is not None and peft_type in _TUNERS


@functools.cache
def _factories(model_type: str) -> list[str]:
    return sorted(name for name, entry in _REGISTRY.items() if entry.model_type == model_type)


@functools.cache
def _shadow(factory: str) -> nn.Module:
    with shadow_alloc():
        return _REGISTRY[factory].factory()


def _zoo_types() -> frozenset[str]:
    return frozenset(entry.model_type for entry in _REGISTRY.values())


def _shared() -> list[tuple[str, str]]:
    """Every (table, model type) the reference has a list row for, a zoo type, and a tuner here."""
    return [
        (table, model_type)
        for table in _SNAPSHOT["tables"]
        if _has_tuner(table)
        for model_type, row in _rows(table).items()
        if model_type in _zoo_types() and isinstance(row, list)
    ]


# ── the snapshot ─────────────────────────────────────────────────────────────


def test_the_snapshot_is_of_the_release_the_signatures_are() -> None:
    assert _SNAPSHOT["peft_version"] == _SIGNATURES["peft_version"] == "0.21.2"


def test_a_copied_table_is_its_base_with_its_own_rows() -> None:
    """The tool stores how a table differs from the one it copied; the lookup restores it."""
    assert _rows("C3A")["gpt2"] == ["mlp.c_proj"]
    assert _rows("C3A")["bert"] == _rows("LORA")["bert"] == ["query", "value"]
    assert _rows("FROD")["vit"] == ["query", "value"]


def test_the_methods_without_a_tuner_are_the_pending_and_refused_ones() -> None:
    """A table has no tuner here exactly when its method's tuner is a stand-in.

    A method whose tuner lands leaves ``_PENDING_TUNERS`` and its table joins the parametrized test
    below, which holds its defaults to the reference's; nothing in this file changes.
    """
    without = {table for table in _SNAPSHOT["tables"] if not _has_tuner(table)}
    withheld = {str(peft_type) for peft_type in (*_PENDING_TUNERS, *_REFUSED_TUNERS)}
    assert without == withheld & set(_SNAPSHOT["tables"]), sorted(without ^ withheld)


# ── the defaults ─────────────────────────────────────────────────────────────


def test_the_shared_rows_are_the_text_models_and_nothing_is_skipped_silently() -> None:
    shared = _shared()
    assert {model_type for _, model_type in shared} == {"bert", "gpt2", "t5"}
    # one row per method with a tuner here, for each of the three (IA3 and its feed-forward table too).
    assert len(shared) >= 3 * 13


@pytest.mark.parametrize(
    ("table", "model_type"), _shared(), ids=[f"{table}-{mt}" for table, mt in _shared()]
)
def test_the_default_is_the_reference_s_in_the_zoo_s_names(table: str, model_type: str) -> None:
    reference = _rows(table)[model_type]
    assert isinstance(reference, list)
    model = _shadow(_factories(model_type)[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{_ALIAS_WARNING}.*")
        resolved = _zoo_target_names(model, reference)
    assert resolved == _lucid_default(table, model_type)


# ── names from a recipe ──────────────────────────────────────────────────────

#: (model_type, the names a reference recipe writes, what the config stores).  An entry whose
#: stored names equal the written ones is a model named alike; otherwise it goes through an alias.
_RECIPES: list[tuple[str, list[str], set[str]]] = [
    ("gpt2", ["c_attn", "c_proj"], {"c_attn", "c_proj"}),
    ("gpt2", ["c_attn", "mlp.c_proj"], {"c_attn", "mlp.c_proj"}),
    ("gpt2", ["c_attn", "c_proj", "c_fc"], {"c_attn", "c_proj", "c_fc"}),
    ("gpt", ["c_attn", "c_proj"], {"c_attn", "c_proj"}),
    ("bert", ["query", "key", "value"], {"query", "key", "value"}),
    ("bert", ["query", "key", "value", "dense"], {"query", "key", "value", "dense"}),
    ("roformer", ["query", "key", "value"], {"query", "key", "value"}),
    ("t5", ["q", "k", "v", "o"], {"q", "k", "v", "o"}),
    ("t5", ["q", "v", "wi", "wo"], {"q", "v", "wi", "wo"}),
    ("vit", ["qkv", "proj"], {"qkv", "proj"}),
    ("vit", ["query", "value"], {"qkv"}),
    ("vit", ["query", "key", "value"], {"qkv"}),
    ("vit", ["query", "key", "value", "attention.output.dense"], {"qkv", "attn.proj"}),
    ("vit", ["dense"], {"attn.proj", "mlp.fc1", "mlp.fc2"}),
    ("vit", ["query", "qkv"], {"qkv"}),
    ("vit", ["q_proj", "v_proj"], {"qkv"}),
    (
        "vit",
        ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        {"qkv", "attn.proj", "fc1", "fc2"},
    ),
    ("swin", ["query", "key", "value"], {"qkv"}),
    ("swin", ["q_proj", "v_proj", "o_proj", "fc1", "fc2"], {"qkv", "attn.proj", "mlp.0", "mlp.3"}),
    ("swin", ["query", "value", "dense"], {"qkv", "attn.proj", "mlp.0", "mlp.3"}),
    ("clip", ["q_proj", "v_proj"], {"attn"}),
    ("clip", ["q_proj", "k_proj", "v_proj", "out_proj"], {"attn"}),
    ("clip", ["fc1", "fc2"], {"mlp.0", "mlp.2"}),
    ("detr", ["q_proj", "v_proj"], {"self_attn", "multihead_attn"}),
    ("detr", ["q_proj", "o_proj"], {"self_attn", "multihead_attn"}),
    (
        "detr",
        ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        {"self_attn", "multihead_attn", "linear1", "linear2"},
    ),
    (
        "maskformer",
        ["q_proj", "k_proj", "v_proj", "out_proj"],
        {"q_proj", "k_proj", "v_proj", "o_proj"},
    ),
    ("mask2former", ["q_proj", "k_proj", "v_proj"], {"q_proj", "k_proj", "v_proj"}),
    ("stable_diffusion", ["to_q", "to_k", "to_v", "to_out.0"], {"attn1", "attn2"}),
    ("dit", ["to_q", "to_k", "to_v", "to_out.0"], {"attn"}),
]


def _stored(
    model: nn.Module, names: list[str]
) -> tuple[_TargetedConfig, list[warnings.WarningMessage]]:
    """The config a default adapter stores for ``names`` on ``model``, and the alias warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = LoraModel._prepare_config(LoraConfig(target_modules=names), model)
    return config, [w for w in caught if _ALIAS_WARNING in str(w.message)]


@pytest.mark.parametrize(
    ("model_type", "names", "expected"),
    _RECIPES,
    ids=[f"{mt}-{'+'.join(names)}" for mt, names, _ in _RECIPES],
)
def test_a_recipe_s_names_reach_modules_the_adapter_can_take(
    model_type: str, names: list[str], expected: set[str]
) -> None:
    for factory in _factories(model_type):
        model = _shadow(factory)
        config, warned = _stored(model, names)
        stored = config.target_modules
        assert stored == expected, factory
        assert bool(warned) == (stored != set(names)), factory
        reached = [(n, m) for n, m in model.named_modules() if n and _selects(config, n)]
        assert reached, factory
        assert not [n for n, m in reached if _refusal(m)], factory
        for name in expected:
            assert _module_names_matching(model, [name]), (factory, name)


def test_every_alias_reaches_a_module_in_every_factory_of_its_type() -> None:
    for model_type, table in REFERENCE_TARGET_ALIASES.items():
        assert _factories(model_type), f"no factory has model type {model_type!r}"
        for factory in _factories(model_type):
            model = _shadow(factory)
            for name, zoo_names in table.items():
                for zoo in zoo_names:
                    assert _module_names_matching(model, [zoo]), (factory, name, zoo)


def test_an_alias_never_replaces_a_name_the_model_has() -> None:
    """ViT's ``qkv`` is a zoo name; and a model that has a module called ``query`` keeps it."""
    model = _shadow(_factories("vit")[0])
    config, warned = _stored(model, ["qkv"])
    assert config.target_modules == {"qkv"} and not warned

    class _Query(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(model_type="vit")
            self.query = nn.Linear(4, 4)
            self.qkv = nn.Linear(4, 12)

    assert _zoo_target_names(_Query(), ["query"]) == {"query"}


def test_a_name_that_selects_nothing_and_has_no_alias_is_left_for_the_error() -> None:
    model = ROWS["vit"].build()
    with pytest.raises(NoMatchingPeftModuleError, match="'nonexistent'.*not found") as raised:
        get_peft_model(model, LoraConfig(target_modules=["nonexistent"]))
    assert isinstance(raised.value, ValueError)


def test_the_aliased_adapter_trains_saves_and_loads_under_the_zoo_s_names(tmp_path: Path) -> None:
    model = ROWS["vit"].build()
    args, kwargs = ROWS["vit"].inputs()
    with pytest.warns(UserWarning, match=_ALIAS_WARNING):
        peft = get_peft_model(
            model, LoraConfig(r=2, target_modules=["query", "value"], init_lora_weights=False)
        )
    assert peft.peft_config["default"].target_modules == {"qkv"}
    peft.eval()
    with lucid.no_grad():
        before = peft(*args, **kwargs)
    peft.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved["target_modules"] == ["qkv"]
    again = PeftModel.from_pretrained(ROWS["vit"].build(), tmp_path).eval()
    with lucid.no_grad():
        assert bool(
            lucid.allclose(again(*args, **kwargs).logits, before.logits, atol=1e-5, rtol=1e-4)
        )


def test_layer_norm_tuning_takes_the_reference_s_t5_names() -> None:
    model = ROWS["t5"].build()
    with pytest.warns(UserWarning, match=_ALIAS_WARNING):
        peft = get_peft_model(
            model, LNTuningConfig(target_modules=["layer_norm", "final_layer_norm"])
        )
    default = get_peft_model(ROWS["t5"].build(), LNTuningConfig())
    assert (
        peft.peft_config["default"].target_modules == default.peft_config["default"].target_modules
    )


def test_adalora_takes_its_own_t5_default_not_loras() -> None:
    peft = get_peft_model(ROWS["t5"].build(), AdaLoraConfig(init_r=4, target_r=2, total_step=10))
    assert peft.peft_config["default"].target_modules == {"q", "k", "v", "o", "wi", "wo"}
    assert _rows("LORA")["t5"] == ["q", "v"]
    assert _rows("ADALORA")["t5"] == ["q", "k", "v", "o", "wi", "wo"]
