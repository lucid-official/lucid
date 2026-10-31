"""Every module that owns parameters can put them back to their initial state (LCD-749).

The reference gives every parametrised layer a ``reset_parameters()``; Lucid
had it on a handful and each of the rest drew its values inline in
``__init__``, where nothing could call it again.  The class-wide invariants:

* a layer that owns a parameter has ``reset_parameters``;
* perturbing the parameters and calling it, from the same seed, lands on
  exactly the values a fresh construction has -- so the constructor and the
  method cannot disagree about the initialisation;
* the draw has the reference's distribution.
"""

import inspect
import importlib
import math
import pkgutil
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.init as init
import lucid.nn.modules as nn_modules
from lucid.nn.module import Module
from lucid.test._fixtures.devices import metal_available

DEVICES: list[str] = ["cpu", "metal"] if metal_available() else ["cpu"]


@dataclass(frozen=True)
class _Case:
    """One layer to build: ``cls(*args, **kwargs)``, then a forward on ``probe`` if lazy."""

    cls: str
    args: tuple[object, ...] = ()
    kwargs: tuple[tuple[str, object], ...] = ()
    probe: tuple[int, ...] | None = None
    leaf: bool = True

    def build(self, device: str) -> Module:
        module: Module = getattr(nn, self.cls)(*self.args, device=device, **dict(self.kwargs))
        if self.probe is not None:
            module(lucid.zeros(*self.probe, device=device))
        return module


def _c(
    cls: str,
    *args: object,
    probe: tuple[int, ...] | None = None,
    leaf: bool = True,
    **kwargs: object,
) -> _Case:
    return _Case(cls, args, tuple(kwargs.items()), probe, leaf)


_CASES: dict[str, _Case] = {
    "Linear": _c("Linear", 5, 4),
    "Linear_nobias": _c("Linear", 5, 4, bias=False),
    "FusedLinear": _c("FusedLinear", 5, 4),
    "Bilinear": _c("Bilinear", 3, 4, 5),
    "NoisyLinear": _c("NoisyLinear", 5, 4),
    "LazyLinear": _c("LazyLinear", 4, probe=(2, 5)),
    "Conv1d": _c("Conv1d", 3, 4, 3),
    "Conv2d": _c("Conv2d", 4, 6, 3, groups=2),
    "Conv2d_nobias": _c("Conv2d", 3, 4, 3, bias=False),
    "Conv3d": _c("Conv3d", 2, 4, 3),
    "ConvTranspose1d": _c("ConvTranspose1d", 3, 4, 3),
    "ConvTranspose2d": _c("ConvTranspose2d", 4, 6, 3, groups=2),
    "ConvTranspose3d": _c("ConvTranspose3d", 2, 4, 3),
    "LazyConv1d": _c("LazyConv1d", 4, 3, probe=(1, 3, 8)),
    "LazyConv2d": _c("LazyConv2d", 4, 3, probe=(1, 3, 8, 8)),
    "LazyConv3d": _c("LazyConv3d", 4, 3, probe=(1, 2, 5, 5, 5)),
    "LazyConvTranspose1d": _c("LazyConvTranspose1d", 4, 3, probe=(1, 3, 8)),
    "LazyConvTranspose2d": _c("LazyConvTranspose2d", 4, 3, probe=(1, 3, 8, 8)),
    "LazyConvTranspose3d": _c("LazyConvTranspose3d", 4, 3, probe=(1, 2, 5, 5, 5)),
    "Embedding": _c("Embedding", 10, 4),
    "Embedding_pad": _c("Embedding", 10, 4, padding_idx=2),
    "Embedding_negpad": _c("Embedding", 10, 4, padding_idx=-1),
    "EmbeddingBag": _c("EmbeddingBag", 10, 4),
    "EmbeddingBag_pad": _c("EmbeddingBag", 10, 4, padding_idx=1),
    "VectorQuantizer": _c("VectorQuantizer", 8, 4),
    "RNN": _c("RNN", 3, 4, num_layers=2, bidirectional=True),
    "GRU": _c("GRU", 3, 4, num_layers=2),
    "LSTM": _c("LSTM", 3, 4, num_layers=2, bidirectional=True),
    "LSTM_proj": _c("LSTM", 3, 4, proj_size=2),
    "RNNCell": _c("RNNCell", 3, 4),
    "LSTMCell": _c("LSTMCell", 3, 4),
    "RNNBase": _c("RNNBase", "GRU", 3, 4, num_layers=2),
    "RNNCellBase": _c("RNNCellBase", 3, 4, True, 3),
    "GRUCell": _c("GRUCell", 3, 4),
    "LayerNorm": _c("LayerNorm", 4),
    "GroupNorm": _c("GroupNorm", 2, 4),
    "RMSNorm": _c("RMSNorm", 4),
    "PReLU": _c("PReLU", 3, init=0.1),
    "BatchNorm1d": _c("BatchNorm1d", 4),
    "InstanceNorm1d": _c("InstanceNorm1d", 4, affine=True),
    "LazyBatchNorm1d": _c("LazyBatchNorm1d", probe=(3, 4)),
    "LazyInstanceNorm1d": _c("LazyInstanceNorm1d", affine=True, probe=(2, 4, 5)),
    "MultiheadAttention": _c("MultiheadAttention", 8, 2),
    "MultiheadAttention_kv": _c("MultiheadAttention", 8, 2, add_bias_kv=True, kdim=6, vdim=5),
    "TransformerEncoderLayer": _c("TransformerEncoderLayer", 8, 2, 16, leaf=False),
    "Transformer": _c("Transformer", 8, 2, 1, 1, 16, leaf=False),
}

_LEAF = [name for name, case in _CASES.items() if case.leaf]

# Classes that mention ``Parameter(`` yet own no layer-initialised parameter:
# the containers hold parameters the caller made, and ``_LazyModule`` is the
# base the lazy layers register placeholders through.
_NO_DEFAULT_INIT = frozenset({"ParameterList", "ParameterDict", "_LazyModule"})


def _reset(module: Module) -> None:
    getattr(module, "reset_parameters")()


def _get(module: Module, name: str) -> lucid.Tensor:
    return getattr(module, name)


def _own(module: Module) -> dict[str, lucid.Tensor]:
    return dict(module.named_parameters(recurse=False))


def _params(module: Module) -> dict[str, list[float]]:
    return {name: p.reshape(-1).tolist() for name, p in module.named_parameters()}


class TestEveryOwnerHasReset:
    @pytest.mark.parametrize("name", list(_CASES))
    def test_each_parameter_owner_exposes_reset_parameters(self, name: str) -> None:
        model = _CASES[name].build("cpu")
        owners = [(path, m) for path, m in model.named_modules() if _own(m)]
        assert owners, f"{name} owns no parameters; drop it from the table"
        for path, owner in owners:
            assert callable(getattr(owner, "reset_parameters", None)), (
                f"{name}.{path or '<root>'} ({type(owner).__name__}) owns "
                f"{sorted(_own(owner))} but has no reset_parameters()"
            )

    def test_table_covers_every_class_that_creates_parameters(self) -> None:
        """A new parametrised layer must be added to ``_CASES`` (or exempted with a reason)."""
        covered = {case.cls for case in _CASES.values()}
        missing: list[str] = []
        for info in pkgutil.iter_modules(nn_modules.__path__):
            mod = importlib.import_module(f"lucid.nn.modules.{info.name}")
            for cls_name, cls in vars(mod).items():
                if not (inspect.isclass(cls) and issubclass(cls, Module)):
                    continue
                if cls.__module__ != mod.__name__ or cls_name.startswith("_"):
                    continue
                if cls_name in covered or cls_name in _NO_DEFAULT_INIT:
                    continue
                source = inspect.getsource(cls)
                if "Parameter(" in source or "register_parameter(" in source:
                    missing.append(f"{cls.__module__}.{cls_name}")
        assert not missing, f"add these to _CASES: {missing}"


class TestResetRestoresInitialState:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("name", _LEAF)
    def test_reset_after_perturbation_equals_fresh_seeded_construction(
        self, name: str, device: str
    ) -> None:
        case = _CASES[name]
        lucid.manual_seed(1)
        used = case.build(device)
        for p in used.parameters():
            init.constant_(p, 9.0)

        lucid.manual_seed(0)
        _reset(used)
        lucid.manual_seed(0)
        fresh = case.build(device)

        assert _params(used) == _params(fresh)

    @pytest.mark.parametrize("name", ["Embedding_pad", "Embedding_negpad", "EmbeddingBag_pad"])
    def test_padding_row_is_zero_after_construction_and_after_reset(self, name: str) -> None:
        module = _CASES[name].build("cpu")
        row = getattr(module, "padding_idx")
        for stage in ("constructed", "reset"):
            weight = _get(module, "weight")
            assert weight[row].tolist() == [0.0] * weight.shape[1], stage
            others = [r for r in range(weight.shape[0]) if r != row]
            assert any(v != 0.0 for r in others for v in weight[r].tolist()), stage
            init.constant_(weight, 5.0)
            _reset(module)

    def test_embedding_bag_with_a_given_table_keeps_it(self) -> None:
        table = lucid.ones(5, 3)
        bag = nn.EmbeddingBag(5, 3, padding_idx=1, _weight=table)
        assert bag.weight.tolist() == table.tolist()

    def test_reset_keeps_the_parameter_objects(self) -> None:
        conv = nn.Conv2d(3, 4, 3)
        weight, bias = conv.weight, conv.bias
        conv.reset_parameters()
        assert conv.weight is weight and conv.bias is bias
        assert weight.requires_grad and weight.is_leaf

    def test_lazy_layer_reset_before_sizing_is_a_no_op(self) -> None:
        for lazy in (nn.LazyLinear(4), nn.LazyConv2d(4, 3), nn.LazyConvTranspose1d(4, 3)):
            lazy.reset_parameters()
            assert lazy.has_uninitialized_parameters()

    def test_lazy_linear_initialize_parameters_sizes_from_an_input(self) -> None:
        lazy = nn.LazyLinear(4)
        lazy.initialize_parameters(lucid.zeros(2, 7))
        assert lazy.in_features == 7 and tuple(lazy.weight.shape) == (4, 7)
        lazy.initialize_parameters(lucid.zeros(2, 9))  # already sized: unchanged
        assert tuple(lazy.weight.shape) == (4, 7)


class TestDistributions:
    def test_conv_and_linear_are_bounded_by_the_fan_in(self) -> None:
        for layer, fan_in in (
            (nn.Conv2d(16, 32, 3), 16 * 9),
            (nn.ConvTranspose2d(16, 32, 3), 32 * 9),
            (nn.Linear(144, 64), 144),
        ):
            bound = 1.0 / math.sqrt(fan_in)
            for p in (layer.weight, layer.bias):
                assert float(p.abs().max()) <= bound
            # U(-b, b) has std b / sqrt(3).
            assert float(layer.weight.std()) == pytest.approx(bound / math.sqrt(3), rel=0.15)

    def test_embedding_is_standard_normal(self) -> None:
        w = nn.Embedding(400, 64).weight
        assert float(w.mean()) == pytest.approx(0.0, abs=0.05)
        assert float(w.std()) == pytest.approx(1.0, rel=0.05)

    @pytest.mark.parametrize("name", ["RNN", "GRU", "LSTM", "LSTM_proj", "RNNCell", "GRUCell"])
    def test_recurrent_layers_draw_from_the_hidden_size_bound(self, name: str) -> None:
        model = _CASES[name].build("cpu")
        bound = 1.0 / math.sqrt(4)
        for p in model.parameters():
            assert float(p.abs().max()) <= bound

    @pytest.mark.parametrize("name", ["LayerNorm", "GroupNorm", "RMSNorm"])
    def test_norm_scale_is_ones_and_shift_is_zeros(self, name: str) -> None:
        model = _CASES[name].build("cpu")
        for p in model.parameters():
            init.constant_(p, 3.0)
        _reset(model)
        assert set(_get(model, "weight").tolist()) == {1.0}
        if "bias" in dict(model.named_parameters()):
            assert set(_get(model, "bias").tolist()) == {0.0}

    def test_prelu_returns_to_its_init_value(self) -> None:
        m = nn.PReLU(3, init=0.1)
        init.constant_(m.weight, 7.0)
        m.reset_parameters()
        assert m.init == 0.1
        assert m.weight.tolist() == pytest.approx([0.1, 0.1, 0.1])

    def test_multihead_attention_resets_under_both_reference_names(self) -> None:
        m = nn.MultiheadAttention(8, 2)
        init.constant_(m.out_proj_weight, 4.0)
        m._reset_parameters()
        assert float(m.out_proj_weight.abs().max()) < 4.0
        init.constant_(m.out_proj_weight, 4.0)
        m.reset_parameters()
        assert float(m.out_proj_weight.abs().max()) < 4.0

    def test_transformer_reset_xavier_initialises_every_matrix(self) -> None:
        m = nn.Transformer(8, 2, 1, 1, 16)
        for p in m.parameters():
            init.constant_(p, 4.0)
        m._reset_parameters()
        for name, p in m.named_parameters():
            if p.ndim > 1:
                assert float(p.abs().max()) < 4.0, name
            else:
                assert set(p.tolist()) == {4.0}, name


@pytest.mark.parity
class TestAgainstTheReference:
    @pytest.mark.parametrize("name", sorted({case.cls for case in _CASES.values()}))
    def test_reset_parameters_is_present_wherever_the_reference_has_it(self, ref, name) -> None:
        ref_cls = getattr(ref.nn, name, None)
        if ref_cls is None:
            pytest.skip(f"{name} has no reference counterpart")
        if not hasattr(ref_cls, "reset_parameters"):
            pytest.skip(f"the reference's {name} has no reset_parameters")
        assert hasattr(getattr(nn, name), "reset_parameters")

    def test_private_reset_names_the_reference_has(self, ref) -> None:
        assert hasattr(ref.nn.MultiheadAttention, "_reset_parameters")
        assert hasattr(ref.nn.Transformer, "_reset_parameters")
        assert hasattr(nn.MultiheadAttention, "_reset_parameters")
        assert hasattr(nn.Transformer, "_reset_parameters")
        assert hasattr(ref.nn.Embedding, "_fill_padding_idx_with_zero")
        assert hasattr(nn.Embedding, "_fill_padding_idx_with_zero")
        assert hasattr(ref.nn.LazyLinear, "initialize_parameters")
        assert hasattr(nn.LazyLinear, "initialize_parameters")

    @pytest.mark.parametrize(
        "make",
        [
            lambda lib: lib.nn.Conv2d(16, 32, 3),
            lambda lib: lib.nn.Linear(144, 64),
            lambda lib: lib.nn.GRU(3, 64),
            lambda lib: lib.nn.Embedding(400, 64),
        ],
        ids=["Conv2d", "Linear", "GRU", "Embedding"],
    )
    def test_initial_spread_matches_the_reference(self, ref, make) -> None:
        ours = make(lucid)
        theirs = make(ref)
        mine = {n: float(p.std()) for n, p in ours.named_parameters()}
        for n, p in theirs.named_parameters():
            assert mine[n] == pytest.approx(float(p.detach().std()), rel=0.2), n
