"""A model-level ``QConfig`` reaches every quantized module's ``from_float`` (LCD-809).

``prepare`` resolves one recipe per module but used to write it back only onto
the types it observes.  A weight-only layer (``Embedding``, ``EmbeddingBag``)
is never observed, so its ``from_float`` saw no ``.qconfig`` and silently fell
back to a hard-coded per-row ``qint8`` observer — a ``qint4`` model recipe
produced ``-128..127`` codes.  ``QConfigResolver`` now stamps the resolved
recipe (``None`` included) on every module it walks, and ``convert`` keeps an
excluded weight-only layer float.

The guard walks ``lucid.nn.quantized`` for every class that has a
``from_float``, finds the float module(s) ``convert`` maps onto it, and runs
both invariants per class — so a new quantized layer cannot ship without a
sample (or an explained exemption) here.
"""

import contextlib
import inspect
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.intrinsic as nni
import lucid.nn.quantized as nnq
import lucid.quantization as Q
from lucid import Tensor
from lucid.quantization._qscheme import per_channel_symmetric, qint4
from lucid.quantization._quantize import _qat_mapping, _quant_mapping
from lucid.quantization.observer import PerChannelMinMaxObserver

Sample = Callable[[], tuple[nn.Module, Tensor]]

# Float module -> (module, calibration input).  Keyed by the float type that
# ``convert`` / ``prepare_qat`` maps, so the table cannot drift from the mapping.
_SAMPLES: dict[type, Sample] = {
    nn.Linear: lambda: (nn.Linear(8, 4), lucid.randn(2, 8)),
    nn.Conv1d: lambda: (nn.Conv1d(2, 3, 3), lucid.randn(2, 2, 8)),
    nn.Conv2d: lambda: (nn.Conv2d(2, 3, 3), lucid.randn(2, 2, 6, 6)),
    nn.Conv3d: lambda: (nn.Conv3d(2, 3, 3), lucid.randn(1, 2, 5, 5, 5)),
    nn.ConvTranspose1d: lambda: (nn.ConvTranspose1d(2, 3, 3), lucid.randn(2, 2, 8)),
    nn.ConvTranspose2d: lambda: (nn.ConvTranspose2d(2, 3, 3), lucid.randn(2, 2, 6, 6)),
    nn.ConvTranspose3d: lambda: (nn.ConvTranspose3d(2, 3, 3), lucid.randn(1, 2, 5, 5, 5)),
    nn.Embedding: lambda: (nn.Embedding(10, 4), lucid.tensor([1, 2, 9])),
    nn.EmbeddingBag: lambda: (nn.EmbeddingBag(10, 4), lucid.tensor([[1, 2], [3, 9]])),
    nn.Sigmoid: lambda: (nn.Sigmoid(), lucid.randn(2, 4)),
    nn.Hardswish: lambda: (nn.Hardswish(), lucid.randn(2, 4)),
    nn.Hardsigmoid: lambda: (nn.Hardsigmoid(), lucid.randn(2, 4)),
    nn.Tanh: lambda: (nn.Tanh(), lucid.randn(2, 4)),
    nn.ELU: lambda: (nn.ELU(), lucid.randn(2, 4)),
    nn.LeakyReLU: lambda: (nn.LeakyReLU(), lucid.randn(2, 4)),
    nni.ConvReLU1d: lambda: (nni.ConvReLU1d(nn.Conv1d(2, 3, 3), nn.ReLU()), lucid.randn(2, 2, 8)),
    nni.ConvReLU2d: lambda: (
        nni.ConvReLU2d(nn.Conv2d(2, 3, 3), nn.ReLU()),
        lucid.randn(2, 2, 6, 6),
    ),
    nni.ConvReLU3d: lambda: (
        nni.ConvReLU3d(nn.Conv3d(2, 3, 3), nn.ReLU()),
        lucid.randn(1, 2, 5, 5, 5),
    ),
    nni.LinearReLU: lambda: (nni.LinearReLU(nn.Linear(8, 4), nn.ReLU()), lucid.randn(2, 8)),
    nnq.QuantStub: lambda: (nnq.QuantStub(), lucid.randn(2, 4)),
    nnq.DeQuantStub: lambda: (nnq.DeQuantStub(), lucid.randn(2, 4)),
}

# Quantized classes the guard cannot drive from a float module through a model —
# each one says why.
_EXEMPT: dict[str, str] = {
    "QuantizedLinearMLX": "installed by the Linear entry's MLX builder, not a mapping entry",
    "QFunctional": "FloatFunctional has no forward, so it cannot be calibrated through a model",
}


def _is_qat_key(ft: type) -> bool:
    return ft.__module__.startswith(("lucid.nn.qat", "lucid.nn.intrinsic.qat"))


def _quantized_classes() -> list[type]:
    """Every class under ``lucid.nn.quantized`` that defines its own ``from_float``."""
    return sorted(
        (
            cls
            for _, cls in inspect.getmembers(nnq, inspect.isclass)
            if issubclass(cls, nn.Module) and "from_float" in vars(cls)
        ),
        key=lambda c: c.__name__,
    )


def _float_sources(cls: type) -> list[type]:
    """The float module types ``convert`` builds ``cls`` from."""
    return [
        ft
        for ft, build in _quant_mapping().items()
        if getattr(build, "__self__", None) is cls and not _is_qat_key(ft)
    ]


def _label(ft: type) -> str:
    return f"{ft.__module__.split('.')[2]}.{ft.__name__}"


_PAIRS = [
    (cls, ft)
    for cls in _quantized_classes()
    if cls.__name__ not in _EXEMPT
    for ft in _float_sources(cls)
]
_PAIR_IDS = [f"{cls.__name__}<-{_label(ft)}" for cls, ft in _PAIRS]
_QAT_PAIRS = [(cls, ft) for cls, ft in _PAIRS if ft in _qat_mapping()]
_QAT_IDS = [f"{cls.__name__}<-{_label(ft)}" for cls, ft in _QAT_PAIRS]


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    """Pin the dequant path so a Linear swap is the plain quantized Linear."""
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


def _qint4_ptq() -> Q.QConfig:
    return Q.QConfig(
        activation=Q.get_default_qconfig().activation,
        weight=PerChannelMinMaxObserver.with_args(
            ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint4
        ),
    )


def _qint4_qat() -> Q.QConfig:
    return Q.QConfig(
        activation=Q.get_default_qat_qconfig().activation,
        weight=Q.FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            ch_axis=0,
            qscheme=per_channel_symmetric,
            qdtype=qint4,
        ),
    )


def _convert(ft: type, qconfig: Q.QConfig, *, qat: bool, exclude_module: bool = False) -> nn.Module:
    """Run ``prepare`` (or ``prepare_qat``) -> one forward -> ``convert`` on a one-layer model."""
    module, x = _SAMPLES[ft]()
    model = nn.Sequential(module)
    model.qconfig = qconfig
    if exclude_module:
        model[0].qconfig = None
    with _reference_engine(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = Q.prepare_qat(model) if qat else Q.prepare(model)
        prepared(x)
        return Q.convert(prepared)


def _codes(module: nn.Module) -> Any:
    return module.weight_int8.numpy().astype("int32")


class TestEveryQuantizedClassIsCovered:
    def test_every_class_has_a_sample_or_an_exemption(self) -> None:
        for cls in _quantized_classes():
            if cls.__name__ in _EXEMPT:
                continue
            sources = _float_sources(cls)
            assert sources, (
                f"nn.quantized.{cls.__name__}.from_float is reachable from no float module "
                "in convert()'s mapping — add it to _EXEMPT with the reason"
            )
            for ft in sources:
                assert ft in _SAMPLES, f"add a _SAMPLES entry for {ft.__name__}"

    def test_the_exemptions_are_still_needed(self) -> None:
        names = {c.__name__ for c in _quantized_classes()}
        for name in _EXEMPT:
            assert name in names, f"_EXEMPT lists {name}, which no longer has a from_float"


class TestAModelLevelQint4RecipeReachesFromFloat:
    @pytest.mark.parametrize(("cls", "ft"), _PAIRS, ids=_PAIR_IDS)
    def test_ptq(self, cls: type, ft: type) -> None:
        wide = _convert(ft, Q.get_default_qconfig(), qat=False)[0]
        narrow = _convert(ft, _qint4_ptq(), qat=False)[0]
        assert isinstance(narrow, cls)
        if not hasattr(narrow, "weight_int8"):
            return  # no weight, so no bit width to honour
        assert int(_codes(wide).max()) > 7, "the contrast case must use the full qint8 grid"
        assert -8 <= int(_codes(narrow).min()) and int(_codes(narrow).max()) <= 7

    @pytest.mark.parametrize(("cls", "ft"), _QAT_PAIRS, ids=_QAT_IDS)
    def test_qat(self, cls: type, ft: type) -> None:
        converted = _convert(ft, _qint4_qat(), qat=True)[0]
        assert isinstance(converted, cls)
        if hasattr(converted, "weight_int8"):
            assert -8 <= int(_codes(converted).min()) and int(_codes(converted).max()) <= 7


class TestAModuleWithQconfigNoneStaysFloat:
    @pytest.mark.parametrize(("cls", "ft"), _PAIRS, ids=_PAIR_IDS)
    def test_ptq(self, cls: type, ft: type) -> None:
        converted = _convert(ft, Q.get_default_qconfig(), qat=False, exclude_module=True)[0]
        assert type(converted) is ft

    @pytest.mark.parametrize(("cls", "ft"), _QAT_PAIRS, ids=_QAT_IDS)
    def test_qat(self, cls: type, ft: type) -> None:
        converted = _convert(ft, Q.get_default_qat_qconfig(), qat=True, exclude_module=True)[0]
        assert type(converted) is ft

    def test_prepare_records_the_exclusion_on_every_module(self) -> None:
        model = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 4))
        model.qconfig = Q.get_default_qconfig()
        model[0].qconfig = None
        prepared = Q.prepare(model)
        assert prepared[0].qconfig is None
        assert prepared[1].qconfig is not None
        assert prepared.qconfig is not None

    def test_a_module_prepare_never_saw_still_converts(self) -> None:
        # No ``prepare`` -> no attribute -> the historical "always convert" path.
        converted = Q.convert(nn.Sequential(nn.Embedding(10, 4)))
        assert isinstance(converted[0], nnq.Embedding)


class TestExcludedWeightOnlyLayerAgainstTheReference:
    @pytest.mark.parity
    @pytest.mark.parametrize("name", ["Embedding", "EmbeddingBag"])
    @pytest.mark.parametrize("excluded", [False, True])
    def test_convert_swaps_exactly_what_the_reference_swaps(
        self, name: str, excluded: bool, ref: Any
    ) -> None:
        ref_q = ref.ao.quantization
        prev = ref.backends.quantized.engine
        ref.backends.quantized.engine = "qnnpack"
        try:
            model = ref.nn.Sequential(getattr(ref.nn, name)(10, 4))
            model.qconfig = ref_q.float_qparams_weight_only_qconfig
            if excluded:
                model[0].qconfig = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prepared = ref_q.prepare(model)
                prepared(ref.tensor([[1, 2]]) if name == "EmbeddingBag" else ref.tensor([1, 2]))
                want = "quantized" in type(ref_q.convert(prepared)[0]).__module__
        finally:
            if prev in ref.backends.quantized.supported_engines:
                ref.backends.quantized.engine = prev
        ours = nn.Sequential(getattr(nn, name)(10, 4))
        ours.qconfig = Q.get_default_qconfig()
        if excluded:
            ours[0].qconfig = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = "quantized" in type(Q.convert(Q.prepare(ours))[0]).__module__
        assert got == want == (not excluded)
