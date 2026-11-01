"""One rule decides which ``QConfig`` a module gets — in every entry point.

``prepare`` / ``prepare_qat`` / ``quantize_dynamic`` used to each decide on
their own, and only a ``QConfigMapping`` was ever read: ``child.qconfig =
None`` did not exclude a layer, a model-level ``qconfig`` could not pick the
bit width, and ``quantize_dynamic`` rejected a set of names or a dict spec
(LCD-723).  They now resolve through ``QConfigResolver`` with the reference
framework's ``propagate_qconfig_`` semantics: a module's own ``.qconfig``
(``None`` excludes its subtree) beats a per-name / per-type rule, which beats
what the parent resolved to.

The propagation table below is run through both frameworks and the sets of
observed / quantized modules must agree; the rest is Lucid-only surface
(mapping form, dtype-valued specs, bit widths, input validation).
"""

import contextlib
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.nn.quantized.dynamic as nnqd
import lucid.quantization as Q
from lucid.quantization._propagate import QConfigResolver, dynamic_qconfig
from lucid.quantization._qscheme import per_channel_symmetric, qint4, qint8
from lucid.quantization.observer import PerChannelMinMaxObserver


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    """Pin the dequant path so a Linear swap is the plain quantized Linear."""
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


@contextlib.contextmanager
def _ref_qengine(ref: Any) -> Iterator[None]:
    prev = ref.backends.quantized.engine
    ref.backends.quantized.engine = "qnnpack"
    try:
        yield
    finally:
        if prev in ref.backends.quantized.supported_engines:
            ref.backends.quantized.engine = prev


def _build(lib: Any) -> Any:
    """``0`` Linear, ``1`` ReLU, ``2`` Sequential holding Linears ``2.0`` and ``2.1``."""
    return lib.nn.Sequential(
        lib.nn.Linear(64, 8),
        lib.nn.ReLU(),
        lib.nn.Sequential(lib.nn.Linear(8, 8), lib.nn.Linear(8, 8)),
    )


def _module(model: Any, path: str) -> Any:
    out = model
    for part in path.split("."):
        out = out[int(part)]
    return out


# ── the propagation table ────────────────────────────────────────────────────
# path -> True (set the framework's default recipe) / None (exclude).  "" is the
# root.  ``observed`` is the set of paths the reference attaches an observer to.
_ATTRIBUTE_CASES: dict[str, dict[str, bool | None]] = {
    "root_only": {"": True},
    "none_at_depth_1_leaf": {"": True, "0": None},
    "none_at_depth_2_leaf": {"": True, "2.1": None},
    "none_container": {"": True, "2": None},
    "none_container_child_overrides": {"": True, "2": None, "2.0": True},
    "none_container_deep_child_overrides": {"": True, "2": None, "2.1": True},
    "none_root": {"": None},
    "none_root_child_overrides": {"": None, "0": True},
    "none_root_container_child_overrides": {"": None, "2": True},
    "none_root_and_container_leaf_overrides": {"": None, "2": None, "2.1": True},
    "only_a_leaf": {"2.0": True},
    "only_a_container": {"2": True},
    "several_none": {"": True, "0": None, "2.1": None},
}


def _apply(model: Any, setup: dict[str, bool | None], on: Any) -> None:
    for path, value in setup.items():
        target = model if path == "" else _module(model, path)
        target.qconfig = on if value else None


def _observed(model: Any) -> set[str]:
    return {n for n, m in model.named_modules() if hasattr(m, "activation_post_process")}


class TestPropagationAgainstTheReference:
    @pytest.mark.parity
    @pytest.mark.parametrize("case", list(_ATTRIBUTE_CASES))
    def test_prepare_observes_the_same_modules(self, case: str, ref: Any) -> None:
        setup = _ATTRIBUTE_CASES[case]
        theirs = _build(ref)
        _apply(theirs, setup, ref.ao.quantization.get_default_qconfig("qnnpack"))
        with _ref_qengine(ref), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = _observed(ref.ao.quantization.prepare(theirs))
        ours = _build(lucid)
        _apply(ours, setup, Q.get_default_qconfig())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _observed(Q.prepare(ours))
        assert got == want

    @pytest.mark.parametrize("case", list(_ATTRIBUTE_CASES))
    def test_convert_quantizes_what_was_observed(self, case: str) -> None:
        setup = _ATTRIBUTE_CASES[case]
        model = _build(lucid)
        _apply(model, setup, Q.get_default_qconfig())
        with warnings.catch_warnings(), _reference_engine():
            warnings.simplefilter("ignore")
            prepared = Q.prepare(model)
            prepared(lucid.randn(4, 64))
            converted = Q.convert(prepared)
        quantized = {n for n, m in converted.named_modules() if isinstance(m, nnq.Linear)}
        observed_linears = {
            n for n in _observed(prepared) if isinstance(_module(prepared, n), nn.Linear)
        }
        assert quantized == observed_linears


# quantize_dynamic: spec forms the reference accepts, keyed by name / type.  A
# callable is resolved against the framework's ``nn`` / default recipe.
_DYNAMIC_CASES: dict[str, Callable[[Any, Any], Any]] = {
    "set_of_a_name": lambda lib, cfg: {"0"},
    "set_of_a_container_name": lambda lib, cfg: {"2"},
    "set_of_a_dotted_name": lambda lib, cfg: {"2.1"},
    "set_of_a_type": lambda lib, cfg: {lib.nn.Linear},
    "set_of_a_name_and_a_type": lambda lib, cfg: {"0", lib.nn.Linear},
    "set_of_a_missing_name": lambda lib, cfg: {"zz"},
    "dict_type_to_config": lambda lib, cfg: {lib.nn.Linear: cfg},
    "dict_type_and_name_none": lambda lib, cfg: {lib.nn.Linear: cfg, "0": None},
    "dict_dotted_name_none": lambda lib, cfg: {lib.nn.Linear: cfg, "2.1": None},
    "dict_container_none": lambda lib, cfg: {lib.nn.Linear: cfg, "2": None},
    "dict_only_none": lambda lib, cfg: {"0": None},
    "dict_type_none": lambda lib, cfg: {lib.nn.Linear: None},
    "dict_name_to_config": lambda lib, cfg: {"2": cfg},
}

# The same, with a ``.qconfig`` attribute set on the model first.
_DYNAMIC_ATTRIBUTE_CASES: dict[str, tuple[dict[str, bool | None], bool]] = {
    "leaf_none_with_type_spec": ({"2.0": None}, True),
    "container_none_with_type_spec": ({"2": None}, True),
    "leaf_set_with_name_spec": ({"0": True}, False),
}


def _dynamic_quantized(model: Any, is_dynamic: Callable[[Any], bool]) -> set[str]:
    return {n for n, m in model.named_modules() if n and is_dynamic(m)}


def _is_ref_dynamic(ref: Any) -> Callable[[Any], bool]:
    prefix = ref.__name__ + ".ao.nn.quantized.dynamic"
    return lambda m: type(m).__module__.startswith(prefix)


class TestDynamicSpecAgainstTheReference:
    @pytest.mark.parity
    @pytest.mark.parametrize("case", list(_DYNAMIC_CASES))
    def test_spec_form(self, case: str, ref: Any) -> None:
        make = _DYNAMIC_CASES[case]
        with _ref_qengine(ref), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = ref.ao.quantization.quantize_dynamic(
                _build(ref), make(ref, ref.ao.quantization.default_dynamic_qconfig)
            )
        want = _dynamic_quantized(theirs, _is_ref_dynamic(ref))
        with warnings.catch_warnings(), _reference_engine():
            warnings.simplefilter("ignore")
            ours = Q.quantize_dynamic(_build(lucid), make(lucid, dynamic_qconfig(qint8)))
        got = _dynamic_quantized(ours, lambda m: isinstance(m, nnqd.Linear))
        assert got == want

    @pytest.mark.parity
    @pytest.mark.parametrize("case", list(_DYNAMIC_ATTRIBUTE_CASES))
    def test_attribute_with_a_spec(self, case: str, ref: Any) -> None:
        setup, with_type_spec = _DYNAMIC_ATTRIBUTE_CASES[case]

        def spec(lib: Any) -> Any:
            return {lib.nn.Linear} if with_type_spec else {"2.1"}

        theirs = _build(ref)
        _apply(theirs, setup, ref.ao.quantization.default_dynamic_qconfig)
        with _ref_qengine(ref), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qtheirs = ref.ao.quantization.quantize_dynamic(theirs, spec(ref))
        want = _dynamic_quantized(qtheirs, _is_ref_dynamic(ref))
        ours = _build(lucid)
        _apply(ours, setup, dynamic_qconfig(qint8))
        with warnings.catch_warnings(), _reference_engine():
            warnings.simplefilter("ignore")
            qours = Q.quantize_dynamic(ours, spec(lucid))
        assert _dynamic_quantized(qours, lambda m: isinstance(m, nnqd.Linear)) == want


# ── Lucid-only surface ───────────────────────────────────────────────────────


def _qint4_config() -> Q.QConfig:
    return Q.QConfig(
        activation=Q.get_default_qconfig().activation,
        weight=PerChannelMinMaxObserver.with_args(
            ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint4
        ),
    )


def _max_code(model: nn.Module, path: str) -> int:
    codes = _module(model, path).weight_int8.numpy()
    return int(abs(codes.astype("int32")).max())


def _calibrated(model: nn.Module, qconfig: Q.QConfig | None = None) -> nn.Module:
    with _reference_engine(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = Q.prepare(model, qconfig)
        prepared(lucid.randn(8, 64))
        return Q.convert(prepared)


class TestMappingForm:
    def test_module_name_none_excludes_one_module(self) -> None:
        mapping = Q.get_default_qconfig_mapping().set_module_name("2.1", None)
        assert _observed(Q.prepare(_build(lucid), mapping)) == {"0", "2.0"}

    def test_object_type_none_excludes_every_module_of_that_type(self) -> None:
        mapping = Q.get_default_qconfig_mapping().set_object_type(nn.Linear, None)
        with pytest.warns(UserWarning, match="matched no quantizable"):
            assert _observed(Q.prepare(_build(lucid), mapping)) == set()

    def test_a_module_attribute_wins_over_a_mapping_rule(self) -> None:
        model = _build(lucid)
        model[0].qconfig = None
        mapping = Q.QConfigMapping().set_global(Q.get_default_qconfig())
        mapping.set_module_name("0", Q.get_default_qconfig())
        assert _observed(Q.prepare(model, mapping)) == {"2.0", "2.1"}

    def test_a_mapping_rule_applies_without_an_attribute(self) -> None:
        mapping = Q.QConfigMapping().set_module_name("2.0", Q.get_default_qconfig())
        assert _observed(Q.prepare(_build(lucid), mapping)) == {"2.0"}

    def test_a_bare_qconfig_argument_seeds_the_root(self) -> None:
        model = _build(lucid)
        model[2].qconfig = None
        assert _observed(Q.prepare(model, Q.get_default_qconfig())) == {"0"}

    def test_no_attribute_and_no_argument_keeps_the_default_recipe(self) -> None:
        assert _observed(Q.prepare(_build(lucid))) == {"0", "2.0", "2.1"}

    def test_a_malformed_qconfig_attribute_is_rejected(self) -> None:
        model = _build(lucid)
        model[0].qconfig = "int8"
        with pytest.raises(TypeError, match=r"0\.qconfig must be a QConfig or None"):
            Q.prepare(model)

    def test_a_malformed_argument_is_rejected(self) -> None:
        bad: Any = {"0": None}
        with pytest.raises(TypeError, match="QConfigMapping or None"):
            QConfigResolver.from_argument(bad, Q.get_default_qconfig_mapping)


class TestBitWidthIsHonoured:
    def test_model_attribute_equals_the_argument(self) -> None:
        by_attribute = _build(lucid)
        by_attribute.qconfig = _qint4_config()
        by_argument = _build(lucid)
        a = _calibrated(by_attribute)
        b = _calibrated(by_argument, _qint4_config())
        assert _max_code(a, "0") <= 8
        assert _max_code(b, "0") <= 8

    def test_default_recipe_uses_the_full_eight_bits(self) -> None:
        model = _build(lucid)
        model.qconfig = Q.get_default_qconfig()
        assert _max_code(_calibrated(model), "0") > 8

    def test_a_child_attribute_overrides_the_inherited_width(self) -> None:
        model = _build(lucid)
        model.qconfig = Q.get_default_qconfig()
        model[2][0].qconfig = _qint4_config()
        quantized = _calibrated(model)
        assert _max_code(quantized, "0") > 8
        assert _max_code(quantized, "2.0") <= 8

    def test_prepare_qat_follows_the_model_attribute(self) -> None:
        model = _build(lucid)
        qat_cfg = Q.get_default_qat_qconfig()
        wide = Q.QConfig(
            activation=qat_cfg.activation,
            weight=Q.FakeQuantize.with_args(
                observer=PerChannelMinMaxObserver,
                ch_axis=0,
                qscheme=per_channel_symmetric,
                qdtype=qint4,
            ),
        )
        model.qconfig = wide
        model[2].qconfig = None
        prepared = Q.prepare_qat(model)
        assert prepared[0].weight_fake_quant.qdtype.bits == 4
        assert isinstance(prepared[2][0], nn.Linear)
        assert not hasattr(prepared[2][0], "weight_fake_quant")

    def test_prepare_qat_attribute_none_excludes_and_child_re_enters(self) -> None:
        model = _build(lucid)
        model.qconfig = Q.get_default_qat_qconfig()
        model[2].qconfig = None
        model[2][1].qconfig = Q.get_default_qat_qconfig()
        prepared = Q.prepare_qat(model)
        swapped = {n for n, m in prepared.named_modules() if hasattr(m, "weight_fake_quant")}
        assert swapped == {"0", "2.1"}


class TestDynamicSpecLucid:
    def test_dtype_valued_dict_picks_the_bit_width_per_entry(self) -> None:
        with _reference_engine():
            quantized = Q.quantize_dynamic(_build(lucid), {"0": qint4, "2": qint8})
        assert _max_code(quantized, "0") <= 8
        assert _max_code(quantized, "2.0") > 8

    def test_qconfig_valued_dict_takes_the_bit_width_from_the_recipe(self) -> None:
        with _reference_engine():
            quantized = Q.quantize_dynamic(_build(lucid), {nn.Linear: _qint4_config()})
        assert _max_code(quantized, "2.1") <= 8

    def test_dtype_argument_applies_to_set_entries(self) -> None:
        with _reference_engine():
            quantized = Q.quantize_dynamic(_build(lucid), {"0"}, dtype=qint4)
        assert _max_code(quantized, "0") <= 8
        assert type(quantized[2][0]) is nn.Linear

    def test_dtype_valued_none_skips_and_default_still_applies(self) -> None:
        with _reference_engine():
            quantized = Q.quantize_dynamic(_build(lucid), {nn.Linear: qint8, "2.0": None})
        assert type(quantized[2][0]) is nn.Linear
        assert isinstance(quantized[2][1], nnqd.Linear)

    def test_a_bare_linear_with_a_name_spec(self) -> None:
        with _reference_engine():
            assert isinstance(Q.quantize_dynamic(nn.Linear(64, 4), {nn.Linear}), nnqd.Linear)
            with pytest.warns(UserWarning, match="no module of a targeted type"):
                assert type(Q.quantize_dynamic(nn.Linear(64, 4), {"0"})) is nn.Linear

    def test_a_missing_name_warns_that_nothing_was_targeted(self) -> None:
        with pytest.warns(UserWarning, match="no module of a targeted type"):
            Q.quantize_dynamic(_build(lucid), {"zz"})

    def test_a_named_module_without_a_dynamic_form_warns(self) -> None:
        model = nn.Sequential(nn.Conv2d(1, 1, 1))
        with pytest.warns(UserWarning, match="Conv2d has no dynamically quantized form"):
            Q.quantize_dynamic(model, {"0"})

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("0", "set of module types"),
            (nn.Linear, "set of module types"),
            ({3}, "keys must be nn.Module subclasses or module names"),
            ({int}, "keys must be nn.Module subclasses or module names"),
            ({"0": 8}, r"must be a QConfig, a QDtype or None"),
        ],
    )
    def test_bad_specs_are_rejected_at_the_boundary(self, spec: Any, message: str) -> None:
        model = _build(lucid)
        before = list(model.state_dict())
        with pytest.raises(TypeError, match=message):
            Q.quantize_dynamic(model, spec, inplace=True)
        assert list(model.state_dict()) == before
        assert type(model[0]) is nn.Linear
