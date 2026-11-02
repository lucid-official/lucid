"""``convert`` leaves no ``.qconfig`` on the converted model (LCD-810).

``prepare`` / ``prepare_qat`` stamp the resolved recipe on every module they walk
(LCD-809) — containers and activations included.  The attribute is a
``prepare``-time channel; the reference framework's ``convert`` removes it from
every module once swapping is done, and ``QConfigResolver.strip`` is the
counterpart here.  Stripping must come *after* the swap, because ``from_float``
and the excluded-Embedding test still read it.
"""

import contextlib
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.quantization as Q
from lucid.quantization._propagate import dynamic_qconfig
from lucid.quantization._qscheme import per_channel_symmetric, qint4, qint8
from lucid.quantization.observer import PerChannelMinMaxObserver


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
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
    return lib.nn.Sequential(
        lib.nn.Linear(8, 8), lib.nn.ReLU(), lib.nn.Sequential(lib.nn.Linear(8, 4), lib.nn.Tanh())
    )


def _with_qconfig(model: Any) -> list[str]:
    return [n for n, m in model.named_modules() if hasattr(m, "qconfig")]


def _observed(model: Any) -> set[str]:
    return {n for n, m in model.named_modules() if hasattr(m, "activation_post_process")}


def _ours(qat: bool = False, *, inplace: bool = False) -> tuple[nn.Module, nn.Module]:
    """``(prepared, converted)`` of the sample model through Lucid."""
    model = _build(lucid)
    if qat:
        model.qconfig = Q.get_default_qat_qconfig()
        prepared = Q.prepare_qat(model).eval()
    else:
        model.qconfig = Q.get_default_qconfig()
        prepared = Q.prepare(model)
    prepared(lucid.randn(2, 8))
    with warnings.catch_warnings(), _reference_engine():
        warnings.simplefilter("ignore")
        return prepared, Q.convert(prepared, inplace=inplace)


class TestConvertStripsQConfig:
    @pytest.mark.parametrize("qat", [False, True], ids=["ptq", "qat"])
    @pytest.mark.parametrize("inplace", [False, True], ids=["copy", "inplace"])
    def test_no_module_keeps_a_qconfig(self, qat: bool, inplace: bool) -> None:
        prepared, converted = _ours(qat, inplace=inplace)
        assert _with_qconfig(converted) == []
        assert (converted is prepared) is inplace

    def test_copy_leaves_the_prepared_model_configured(self) -> None:
        prepared, _ = _ours()
        assert "" in _with_qconfig(prepared)

    def test_convert_fx_strips_too(self) -> None:
        model = _build(lucid)
        model.qconfig = Q.get_default_qconfig()
        prepared = Q.prepare_fx(model)
        prepared(lucid.randn(2, 8))
        with warnings.catch_warnings(), _reference_engine():
            warnings.simplefilter("ignore")
            assert _with_qconfig(Q.convert_fx(prepared)) == []

    @pytest.mark.parity
    @pytest.mark.parametrize("qat", [False, True], ids=["ptq", "qat"])
    def test_matches_the_reference(self, qat: bool, ref: Any) -> None:
        ao = ref.ao.quantization
        theirs = _build(ref)
        with _ref_qengine(ref), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if qat:
                theirs.qconfig = ao.get_default_qat_qconfig("qnnpack")
                prepared = ao.prepare_qat(theirs).eval()
            else:
                theirs.qconfig = ao.get_default_qconfig("qnnpack")
                prepared = ao.prepare(theirs)
            prepared(ref.randn(2, 8))
            want = ao.convert(prepared)
        _, got = _ours(qat)
        assert _with_qconfig(want) == []
        assert _with_qconfig(got) == _with_qconfig(want)

    @pytest.mark.parity
    def test_reprepare_a_converted_model_like_the_reference(self, ref: Any) -> None:
        ao = ref.ao.quantization
        theirs = _build(ref)
        with _ref_qengine(ref), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs.qconfig = ao.get_default_qconfig("qnnpack")
            prepared = ao.prepare(theirs)
            prepared(ref.randn(2, 8))
            want = _observed(ao.prepare(ao.convert(prepared)))
        _, converted = _ours()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _observed(Q.prepare(converted))
        assert got == want


class TestStripComesAfterTheSwap:
    """LCD-809's readers of ``.qconfig`` still see it during ``convert``."""

    def test_excluded_embedding_stays_float_and_loses_the_marker(self) -> None:
        model = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 4))
        model.qconfig = Q.get_default_qconfig()
        model[0].qconfig = None
        prepared = Q.prepare(model)
        prepared(lucid.tensor([1, 2]))
        with warnings.catch_warnings(), _reference_engine():
            warnings.simplefilter("ignore")
            converted = Q.convert(prepared)
        assert type(converted[0]) is nn.Embedding
        assert _with_qconfig(converted) == []

    def test_qint4_recipe_reaches_from_float(self) -> None:
        model = nn.Sequential(nn.Embedding(10, 4))
        model.qconfig = Q.QConfig(
            activation=Q.get_default_qconfig().activation,
            weight=PerChannelMinMaxObserver.with_args(
                ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint4
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # prepare(): no observed module (weight-only)
            converted = Q.convert(Q.prepare(model))
        emb = converted[0]
        assert isinstance(emb, nnq.Embedding)
        codes = emb.weight_int8
        assert int(codes.min().item()) >= -8 and int(codes.max().item()) <= 7
        assert _with_qconfig(converted) == []


class TestDynamicStrips:
    """``quantize_dynamic`` converts internally in the reference, so a user's ``.qconfig`` goes too."""

    def _configured(self, lib: Any, qconfig: Any) -> Any:
        model = _build(lib)
        model.qconfig = qconfig
        model[1].qconfig = qconfig
        return model

    def test_unconfigured_model_has_none(self) -> None:
        assert _with_qconfig(Q.quantize_dynamic(_build(lucid))) == []

    @pytest.mark.parametrize("inplace", [False, True], ids=["copy", "inplace"])
    def test_user_qconfig_is_removed(self, inplace: bool) -> None:
        model = self._configured(lucid, dynamic_qconfig(qint8))
        out = Q.quantize_dynamic(model, inplace=inplace)
        assert _with_qconfig(out) == []
        assert (out is model) is inplace

    def test_bare_target_root_is_stripped(self) -> None:
        lin = nn.Linear(4, 4)
        lin.qconfig = dynamic_qconfig(qint8)
        assert _with_qconfig(Q.quantize_dynamic(lin)) == []

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: Any) -> None:
        ao = ref.ao.quantization
        theirs = self._configured(ref, ao.default_dynamic_qconfig)
        with _ref_qengine(ref), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = _with_qconfig(ao.quantize_dynamic(theirs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _with_qconfig(Q.quantize_dynamic(self._configured(lucid, dynamic_qconfig(qint8))))
        assert got == want
