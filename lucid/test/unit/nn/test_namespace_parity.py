"""The ``nn`` names the reference has and Lucid lacked (LCD-751).

``Embedding.from_pretrained`` and the ``_weight`` / ``_freeze`` constructor
arguments, ``Buffer``, ``AdaptiveLogSoftmaxWithLoss``, ``RNNBase`` /
``RNNCellBase``, ``CrossMapLRN2d``, the ``*_with_indices`` pools, ``celu_``,
the deprecated ``upsample*`` spellings and ``clip_grads_with_norm_``.

Each is held to the reference: values, gradients and — for every new name —
the parameters its signature takes.
"""

import inspect
import warnings
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.utils import clip_grads_with_norm_, get_total_norm

pytestmark = pytest.mark.parity


def _data(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _close(got: lucid.Tensor, want: object, atol: float = 1e-4) -> None:
    want_np = want.detach().numpy()
    assert tuple(got.shape) == tuple(want_np.shape)
    np.testing.assert_allclose(got.numpy(), want_np, rtol=1e-4, atol=atol)


def _parameters(fn: Callable[..., object]) -> dict[str, tuple[object, object]]:
    """``name -> (kind, default)`` of a callable, annotations left out."""
    sig = inspect.signature(fn)
    return {name: (p.kind, p.default) for name, p in sig.parameters.items()}


def _same_parameters(ours: Callable[..., object], theirs: Callable[..., object]) -> None:
    """The reference's parameters, in its order, with its kinds and defaults."""
    assert list(_parameters(ours).items()) == list(_parameters(theirs).items())


# ── signatures ───────────────────────────────────────────────────────────────

#: (name on ``nn`` / ``nn.functional`` / ``nn.utils``, namespace)
_SIGNATURES = [
    ("Embedding.from_pretrained", "nn"),
    ("EmbeddingBag.from_pretrained", "nn"),
    ("EmbeddingBag.__init__", "nn"),
    ("Buffer", "nn"),
    ("AdaptiveLogSoftmaxWithLoss.__init__", "nn"),
    ("CrossMapLRN2d.__init__", "nn"),
    ("RNNBase.__init__", "nn"),
    ("RNNCellBase.__init__", "nn"),
    ("max_pool1d_with_indices", "F"),
    ("max_pool2d_with_indices", "F"),
    ("max_pool3d_with_indices", "F"),
    ("adaptive_max_pool1d_with_indices", "F"),
    ("adaptive_max_pool2d_with_indices", "F"),
    ("adaptive_max_pool3d_with_indices", "F"),
    ("fractional_max_pool2d_with_indices", "F"),
    ("fractional_max_pool3d_with_indices", "F"),
    ("upsample", "F"),
    ("upsample_nearest", "F"),
    ("upsample_bilinear", "F"),
    ("clip_grads_with_norm_", "utils"),
]


def _resolve(root: object, dotted: str) -> Callable[..., object]:
    for part in dotted.split("."):
        root = getattr(root, part)
    assert callable(root)
    return root


@pytest.mark.parametrize("name,space", _SIGNATURES)
def test_signature_is_the_references(ref: ModuleType, name: str, space: str) -> None:
    roots = {"nn": (nn, ref.nn), "F": (F, ref.nn.functional), "utils": (nn.utils, ref.nn.utils)}[
        space
    ]
    _same_parameters(_resolve(roots[0], name), _resolve(roots[1], name))


def test_embedding_init_takes_the_references_arguments(ref: ModuleType) -> None:
    """``_weight`` and ``_freeze`` precede ``device`` and ``dtype``, as in the reference."""
    _same_parameters(nn.Embedding.__init__, ref.nn.Embedding.__init__)


@pytest.mark.parametrize(
    "name", ["Buffer", "AdaptiveLogSoftmaxWithLoss", "RNNBase", "RNNCellBase", "CrossMapLRN2d"]
)
def test_new_classes_are_where_the_reference_has_them(ref: ModuleType, name: str) -> None:
    assert hasattr(nn, name) and hasattr(ref.nn, name)
    assert name in nn.__all__


# ── Embedding / EmbeddingBag ─────────────────────────────────────────────────


@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("padding_idx", [None, 1, -1])
def test_embedding_from_pretrained(ref: ModuleType, device: str, freeze: bool, padding_idx) -> None:
    table = _data(5, 3)
    ours = nn.Embedding.from_pretrained(
        lucid.tensor(table, device=device), freeze=freeze, padding_idx=padding_idx
    )
    theirs = ref.nn.Embedding.from_pretrained(
        ref.from_numpy(table.copy()), freeze=freeze, padding_idx=padding_idx
    )
    assert ours.weight.requires_grad is theirs.weight.requires_grad is (not freeze)
    assert ours.padding_idx == theirs.padding_idx
    assert (ours.num_embeddings, ours.embedding_dim) == (5, 3)
    # the table is the caller's, not redrawn and not zeroed at padding_idx
    _close(ours.weight, theirs.weight)
    idx = np.array([[0, 1, 4], [4, 1, 2]])
    got = ours(lucid.tensor(idx, dtype=lucid.int64, device=device))
    want = theirs(ref.from_numpy(idx))
    _close(got, want)
    if not freeze:
        got.sum().backward()
        want.sum().backward()
        _close(ours.weight.grad, theirs.weight.grad)


def test_embedding_from_pretrained_binds_the_table_without_copying(device: str) -> None:
    table = lucid.zeros(3, 2, device=device)
    emb = nn.Embedding.from_pretrained(table)
    with lucid.no_grad():
        table += 1.0
    assert float(emb.weight.sum()) == 6.0


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("freeze", [True, False])
def test_embedding_bag_from_pretrained(
    ref: ModuleType, device: str, mode: str, freeze: bool
) -> None:
    table = _data(6, 4)
    ours = nn.EmbeddingBag.from_pretrained(lucid.tensor(table, device=device), freeze, mode=mode)
    theirs = ref.nn.EmbeddingBag.from_pretrained(ref.from_numpy(table.copy()), freeze, mode=mode)
    assert ours.weight.requires_grad is theirs.weight.requires_grad is (not freeze)
    idx = np.array([[0, 5, 2], [1, 1, 3]])
    got = ours(lucid.tensor(idx, dtype=lucid.int64, device=device))
    want = theirs(ref.from_numpy(idx))
    _close(got, want)
    if not freeze:
        got.sum().backward()
        want.sum().backward()
        _close(ours.weight.grad, theirs.weight.grad)


def test_embedding_weight_argument_is_adopted_and_checked(ref: ModuleType, device: str) -> None:
    table = _data(4, 3)
    ours = nn.Embedding(
        4, 3, padding_idx=2, _weight=lucid.tensor(table, device=device), _freeze=True
    )
    theirs = ref.nn.Embedding(
        4, 3, padding_idx=2, _weight=ref.from_numpy(table.copy()), _freeze=True
    )
    assert ours.weight.requires_grad is theirs.weight.requires_grad is False
    _close(ours.weight, theirs.weight)
    thawed = nn.Embedding(4, 3, _weight=lucid.tensor(table, device=device))
    assert thawed.weight.requires_grad
    with pytest.raises(AssertionError):
        ref.nn.Embedding(5, 3, _weight=ref.from_numpy(table.copy()))
    for build in (
        lambda: nn.Embedding(5, 3, _weight=lucid.tensor(table)),
        lambda: nn.EmbeddingBag(4, 5, _weight=lucid.tensor(table)),
    ):
        with pytest.raises(ValueError, match="does not match"):
            build()


@pytest.mark.parametrize("cls", ["Embedding", "EmbeddingBag"])
def test_from_pretrained_refuses_what_the_reference_refuses(ref: ModuleType, cls: str) -> None:
    with pytest.raises(AssertionError):
        getattr(ref.nn, cls).from_pretrained(ref.zeros(3))
    with pytest.raises(ValueError, match="2-dimensional"):
        getattr(nn, cls).from_pretrained(lucid.zeros(3))


def test_sparse_embedding_stays_unsupported() -> None:
    emb = nn.Embedding.from_pretrained(lucid.ones(3, 2), sparse=True)
    with pytest.raises(NotImplementedError):
        emb(lucid.tensor([0], dtype=lucid.int64))


def test_from_pretrained_builds_the_subclass() -> None:
    class Mine(nn.Embedding):
        pass

    assert type(Mine.from_pretrained(lucid.ones(2, 2))) is Mine


# ── Buffer ───────────────────────────────────────────────────────────────────


def _buffered(space: ModuleType, make: Callable[[float], object]):
    class Holder(space.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kept = space.Buffer(make(1.0))
            self.scratch = space.Buffer(make(2.0), persistent=False)
            self.plain = make(3.0)
            self.w = space.Parameter(make(4.0))

    return Holder()


def test_buffer_registers_like_the_reference(ref: ModuleType, device: str) -> None:
    ours = _buffered(nn, lambda v: lucid.full((2,), v, device=device))
    theirs = _buffered(ref.nn, lambda v: ref.full((2,), v))
    assert [n for n, _ in ours.named_buffers()] == [n for n, _ in theirs.named_buffers()]
    assert list(ours.state_dict()) == list(theirs.state_dict())
    assert [n for n, _ in ours.named_parameters()] == [n for n, _ in theirs.named_parameters()]
    assert "plain" not in dict(ours.named_buffers())
    assert isinstance(ours.kept, nn.Buffer) and isinstance(ours.kept, lucid.Tensor)
    assert ours.kept.persistent and not ours.scratch.persistent


def test_buffer_follows_the_module_across_dtype_and_device(device: str) -> None:
    m = _buffered(nn, lambda v: lucid.full((2,), v, device=device)).half()
    assert m.kept.dtype == lucid.float16 and m.scratch.dtype == lucid.float16
    assert m.state_dict()["kept"].dtype == lucid.float16


def test_buffer_assignment_replaces_in_place_and_keeps_its_flag() -> None:
    m = _buffered(nn, lambda v: lucid.full((2,), v))
    m.kept = nn.Buffer(lucid.zeros(2), persistent=False)
    assert "kept" not in m.state_dict()
    m.scratch = lucid.ones(2)  # a bare tensor keeps the registered flag
    assert "scratch" not in m.state_dict()
    assert list(dict(m.named_buffers())) == ["kept", "scratch"]


def test_buffer_defaults_and_aliasing() -> None:
    assert nn.Buffer().shape == (0,)
    data = lucid.zeros(2)
    buf = nn.Buffer(data)
    with lucid.no_grad():
        data += 1.0
    assert float(buf.sum()) == 2.0
    assert nn.Buffer(lucid.zeros(2, requires_grad=True)).requires_grad
    assert not nn.Buffer(lucid.zeros(2)).requires_grad


def test_buffer_before_module_init_is_refused() -> None:
    class Early(nn.Module):
        def __init__(self) -> None:
            self.b = nn.Buffer(lucid.zeros(1))

    with pytest.raises(AttributeError, match="before Module.__init__"):
        Early()


# ── AdaptiveLogSoftmaxWithLoss ───────────────────────────────────────────────

_ASM = dict(in_features=16, n_classes=50, cutoffs=[5, 20], div_value=2.0)
_TARGETS = np.array([0, 4, 5, 19, 20, 49, 7, 30, 1])


def _asm_pair(ref: ModuleType, device: str, head_bias: bool):
    theirs = ref.nn.AdaptiveLogSoftmaxWithLoss(**_ASM, head_bias=head_bias)
    ours = nn.AdaptiveLogSoftmaxWithLoss(**_ASM, head_bias=head_bias, device=device)
    assert list(ours.state_dict()) == list(theirs.state_dict())
    ours.load_state_dict(
        {k: lucid.tensor(v.detach().numpy(), device=device) for k, v in theirs.state_dict().items()}
    )
    return ours, theirs


@pytest.mark.parametrize("head_bias", [False, True])
def test_adaptive_softmax_forward_and_gradients(
    ref: ModuleType, device: str, head_bias: bool
) -> None:
    ours, theirs = _asm_pair(ref, device, head_bias)
    x = _data(9, 16)
    xl = lucid.tensor(x, device=device, requires_grad=True)
    xt = ref.tensor(x, requires_grad=True)
    got = ours(xl, lucid.tensor(_TARGETS, dtype=lucid.int64, device=device))
    want = theirs(xt, ref.tensor(_TARGETS))
    _close(got.output, want.output)
    _close(got.loss, want.loss)
    got.loss.backward()
    want.loss.backward()
    _close(xl.grad, xt.grad, atol=1e-5)
    for (name, a), (_, b) in zip(ours.named_parameters(), theirs.named_parameters(), strict=True):
        _close(a.grad, b.grad, atol=1e-5)


def test_adaptive_softmax_log_prob_predict(ref: ModuleType, device: str) -> None:
    ours, theirs = _asm_pair(ref, device, False)
    # a scale at which both the shortlist-only and the mixed predict paths run
    for scale in (0.1, 1.0, 6.0):
        x = _data(9, 16, seed=3) * scale
        _close(ours.log_prob(lucid.tensor(x, device=device)), theirs.log_prob(ref.tensor(x)), 1e-4)
        np.testing.assert_array_equal(
            ours.predict(lucid.tensor(x, device=device)).numpy(),
            theirs.predict(ref.tensor(x)).numpy(),
        )
    probs = ours.log_prob(lucid.tensor(_data(4, 16), device=device)).exp().sum(dim=1)
    np.testing.assert_allclose(probs.numpy(), 1.0, atol=1e-5)


def test_adaptive_softmax_unbatched_and_single_row(ref: ModuleType, device: str) -> None:
    ours, theirs = _asm_pair(ref, device, False)
    x = _data(9, 16, seed=1)
    for row, cls in ((3, 20), (0, 0), (5, 49)):
        got = ours(
            lucid.tensor(x[row], device=device), lucid.tensor(cls, dtype=lucid.int64, device=device)
        )
        want = theirs(ref.tensor(x[row]), ref.tensor(cls))
        assert got.output.shape == tuple(want.output.shape) == ()
        _close(got.output, want.output)
        _close(got.loss, want.loss)
    got = ours(
        lucid.tensor(x[:1], device=device), lucid.tensor([30], dtype=lucid.int64, device=device)
    )
    want = theirs(ref.tensor(x[:1]), ref.tensor([30]))
    _close(got.output, want.output)


def test_adaptive_softmax_refusals_match_the_references(ref: ModuleType) -> None:
    ours, theirs = _asm_pair(ref, "cpu", False)
    x = _data(4, 16)
    cases = [
        (x, np.array([0, 1, 2])),  # batch mismatch
        (x[0], np.array([0, 1])),  # multi-target
        (x, np.array([[0], [1], [2], [3]])),  # 2-D target
        (x, np.array([0, 1, 2, 50])),  # out of range
        (x, np.array([0, 1, 2, -1])),
    ]
    for features, target in cases:
        with pytest.raises(RuntimeError):
            theirs(ref.tensor(features), ref.tensor(target))
        # ValueError for a malformed call, IndexError for a class out of range;
        # both are RuntimeErrors too where they come from the engine.
        with pytest.raises((ValueError, IndexError)):
            ours(lucid.tensor(features), lucid.tensor(target, dtype=lucid.int64))
    # the 1-D target against a 3-D input
    with pytest.raises(ValueError):
        ours(lucid.zeros(4, 1, 16), lucid.zeros(4, dtype=lucid.int64))


@pytest.mark.parametrize(
    "cutoffs",
    [[], [0, 5], [5, 5], [20, 5], [5, 50], [2.5, 10]],
    ids=["empty", "zero", "repeat", "unsorted", "too_large", "fractional"],
)
def test_adaptive_softmax_cutoff_validation(ref: ModuleType, cutoffs: list) -> None:
    with pytest.raises(ValueError):
        ref.nn.AdaptiveLogSoftmaxWithLoss(16, 50, cutoffs)
    with pytest.raises(ValueError):
        nn.AdaptiveLogSoftmaxWithLoss(16, 50, cutoffs)


def test_adaptive_softmax_cluster_sizes_match_the_references(ref: ModuleType) -> None:
    ours = nn.AdaptiveLogSoftmaxWithLoss(32, 1000, [10, 100, 500], div_value=4.0)
    theirs = ref.nn.AdaptiveLogSoftmaxWithLoss(32, 1000, [10, 100, 500], div_value=4.0)
    assert {k: tuple(v.shape) for k, v in ours.state_dict().items()} == {
        k: tuple(v.shape) for k, v in theirs.state_dict().items()
    }
    assert ours.cutoffs == theirs.cutoffs == [10, 100, 500, 1000]
    assert (ours.shortlist_size, ours.n_clusters, ours.head_size) == (10, 3, 13)


def test_adaptive_softmax_reset_parameters_redraws() -> None:
    m = nn.AdaptiveLogSoftmaxWithLoss(8, 30, [4, 12])
    before = m.head.weight.numpy().copy()
    m.reset_parameters()
    assert not np.array_equal(before, m.head.weight.numpy())


# ── RNNBase / RNNCellBase ────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", ["RNN", "LSTM", "GRU"])
def test_recurrent_layers_derive_from_rnn_base(cls: str) -> None:
    assert issubclass(getattr(nn, cls), nn.RNNBase)


@pytest.mark.parametrize("cls", ["RNNCell", "LSTMCell", "GRUCell"])
def test_recurrent_cells_derive_from_rnn_cell_base(cls: str) -> None:
    assert issubclass(getattr(nn, cls), nn.RNNCellBase)


@pytest.mark.parametrize("mode", ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_rnn_base_parameters_match_the_references(
    ref: ModuleType, mode: str, bias: bool, bidirectional: bool
) -> None:
    kwargs = dict(
        num_layers=2, bias=bias, bidirectional=bidirectional, batch_first=True, dropout=0.25
    )
    ours = nn.RNNBase(mode, 4, 5, **kwargs)
    theirs = ref.nn.RNNBase(mode, 4, 5, **kwargs)
    assert {k: tuple(v.shape) for k, v in ours.named_parameters()} == {
        k: tuple(v.shape) for k, v in theirs.named_parameters()
    }
    assert list(dict(ours.named_parameters())) == list(dict(theirs.named_parameters()))
    for attr in ("mode", "input_size", "hidden_size", "num_layers", "bias", "batch_first"):
        assert getattr(ours, attr) == getattr(theirs, attr)
    assert ours.dropout == theirs.dropout == 0.25
    assert ours.bidirectional is theirs.bidirectional
    assert len(ours.all_weights) == len(theirs.all_weights)
    assert [[tuple(w.shape) for w in g] for g in ours.all_weights] == [
        [tuple(w.shape) for w in g] for g in theirs.all_weights
    ]


def test_rnn_base_projection_matches_the_reference(ref: ModuleType) -> None:
    ours = nn.RNNBase("LSTM", 4, 6, num_layers=2, bidirectional=True, proj_size=3)
    theirs = ref.nn.RNNBase("LSTM", 4, 6, num_layers=2, bidirectional=True, proj_size=3)
    assert {k: tuple(v.shape) for k, v in ours.named_parameters()} == {
        k: tuple(v.shape) for k, v in theirs.named_parameters()
    }
    assert list(dict(ours.named_parameters())) == list(dict(theirs.named_parameters()))


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (("BOGUS", 4, 5), {}),
        (("LSTM", 4, 5), {"proj_size": 5}),
        (("LSTM", 4, 5), {"proj_size": -1}),
        (("LSTM", 4, 0), {}),
        (("LSTM", 4, 5), {"dropout": 1.5}),
    ],
)
def test_rnn_base_refuses_what_the_reference_refuses(
    ref: ModuleType, args: tuple, kwargs: dict
) -> None:
    with pytest.raises(ValueError):
        ref.nn.RNNBase(*args, **kwargs)
    with pytest.raises(ValueError):
        nn.RNNBase(*args, **kwargs)


def test_rnn_base_refuses_a_projection_for_modes_without_one(ref: ModuleType) -> None:
    """Stricter than the reference, which builds a ``weight_hr`` for a GRU too.

    No layer class takes ``proj_size`` for a GRU or RNN; a base that did would
    hold a parameter nothing reads.
    """
    assert "weight_hr_l0" in dict(ref.nn.RNNBase("GRU", 4, 5, proj_size=2).named_parameters())
    for mode in ("GRU", "RNN_TANH", "RNN_RELU"):
        with pytest.raises(ValueError, match="only supported for LSTM"):
            nn.RNNBase(mode, 4, 5, proj_size=2)


@pytest.mark.parametrize("chunks", [1, 3, 4])
@pytest.mark.parametrize("bias", [True, False])
def test_rnn_cell_base_parameters_match_the_references(
    ref: ModuleType, chunks: int, bias: bool
) -> None:
    ours, theirs = nn.RNNCellBase(4, 5, bias, chunks), ref.nn.RNNCellBase(4, 5, bias, chunks)
    assert {k: tuple(v.shape) for k, v in ours.named_parameters()} == {
        k: tuple(v.shape) for k, v in theirs.named_parameters()
    }
    assert list(dict(ours.named_parameters())) == list(dict(theirs.named_parameters()))
    assert (ours.input_size, ours.hidden_size, ours.bias) == (4, 5, bias)


def test_rnn_base_layers_keep_their_parameter_order(ref: ModuleType) -> None:
    for cls, kw in (("LSTM", {"proj_size": 2}), ("GRU", {}), ("RNN", {"nonlinearity": "relu"})):
        ours = getattr(nn, cls)(3, 4, num_layers=2, bidirectional=True, **kw)
        theirs = getattr(ref.nn, cls)(3, 4, num_layers=2, bidirectional=True, **kw)
        assert list(dict(ours.named_parameters())) == list(dict(theirs.named_parameters()))
        assert ours.dropout == theirs.dropout == 0.0


# ── CrossMapLRN2d ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("channels", [1, 2, 3, 5, 9])
def test_cross_map_lrn2d_matches_the_reference(
    ref: ModuleType, device: str, size: int, channels: int
) -> None:
    x = _data(2, channels, 3, 4, seed=size * 10 + channels)
    g = _data(2, channels, 3, 4, seed=99)
    kwargs = dict(alpha=0.5, beta=0.75, k=1.5)
    xl = lucid.tensor(x, device=device, requires_grad=True)
    xt = ref.tensor(x, requires_grad=True)
    got = nn.CrossMapLRN2d(size, **kwargs)(xl)
    want = ref.nn.CrossMapLRN2d(size, **kwargs)(xt)
    _close(got, want, atol=1e-5)
    got.backward(lucid.tensor(g, device=device))
    want.backward(ref.from_numpy(g))
    _close(xl.grad, xt.grad, atol=1e-4)


def test_cross_map_lrn2d_defaults_and_errors(ref: ModuleType) -> None:
    ours, theirs = nn.CrossMapLRN2d(3), ref.nn.CrossMapLRN2d(3)
    assert (ours.alpha, ours.beta, ours.k) == (theirs.alpha, theirs.beta, theirs.k)
    assert repr(ours) == repr(theirs) == "CrossMapLRN2d(3, alpha=0.0001, beta=0.75, k=1)"
    with pytest.raises(ValueError):
        theirs(ref.zeros(2, 3, 4))
    with pytest.raises(ValueError):
        ours(lucid.zeros(2, 3, 4))


def test_cross_map_lrn2d_size_one_cannot_backpropagate_as_in_the_reference(ref: ModuleType) -> None:
    xt = ref.randn(1, 4, 2, 2, requires_grad=True)
    out = ref.nn.CrossMapLRN2d(1)(xt)
    with pytest.raises(RuntimeError):
        out.sum().backward()
    xl = lucid.randn(1, 4, 2, 2, requires_grad=True)
    out = nn.CrossMapLRN2d(1)(xl)
    with pytest.raises(RuntimeError):
        out.sum().backward()


# ── *_with_indices ───────────────────────────────────────────────────────────

_WINDOWED = [
    (1, (2, 3, 9), dict(kernel_size=3, stride=2, padding=1)),
    (2, (2, 3, 8, 7), dict(kernel_size=(3, 2), stride=2, ceil_mode=True)),
    (3, (1, 2, 6, 6, 5), dict(kernel_size=2, stride=1)),
    (2, (2, 3, 9, 9), dict(kernel_size=3, stride=2, padding=1, dilation=2)),
]


@pytest.mark.parametrize("rank,shape,kwargs", _WINDOWED)
@pytest.mark.parametrize("flag", [False, True], ids=["flag_off", "flag_on"])
def test_max_pool_with_indices(
    ref: ModuleType, device: str, rank, shape, kwargs, flag: bool
) -> None:
    name = f"max_pool{rank}d_with_indices"
    x = _data(*shape)
    xl = lucid.tensor(x, device=device, requires_grad=True)
    xt = ref.tensor(x, requires_grad=True)
    got = getattr(F, name)(xl, return_indices=flag, **kwargs)
    want = getattr(ref.nn.functional, name)(xt, return_indices=flag, **kwargs)
    assert isinstance(got, tuple) and len(got) == 2
    _close(got[0], want[0])
    np.testing.assert_array_equal(got[1].numpy(), want[1].numpy())
    got[0].sum().backward()
    want[0].sum().backward()
    _close(xl.grad, xt.grad)


@pytest.mark.parametrize(
    "rank,shape,size", [(1, (2, 3, 10), 4), (2, (2, 3, 9, 7), (4, 3)), (3, (1, 2, 6, 5, 7), 3)]
)
@pytest.mark.parametrize("flag", [False, True], ids=["flag_off", "flag_on"])
def test_adaptive_max_pool_with_indices(
    ref: ModuleType, device: str, rank, shape, size, flag: bool
) -> None:
    name = f"adaptive_max_pool{rank}d_with_indices"
    x = _data(*shape)
    got = getattr(F, name)(lucid.tensor(x, device=device), size, flag)
    want = getattr(ref.nn.functional, name)(ref.tensor(x), size, flag)
    _close(got[0], want[0])
    np.testing.assert_array_equal(got[1].numpy(), want[1].numpy())


@pytest.mark.parametrize(
    "rank,shape,kernel,out", [(2, (2, 3, 12, 10), 2, (6, 5)), (3, (1, 2, 8, 8, 8), 2, (4, 4, 4))]
)
@pytest.mark.parametrize("flag", [False, True], ids=["flag_off", "flag_on"])
def test_fractional_max_pool_with_indices(
    ref: ModuleType, device: str, rank, shape, kernel, out, flag: bool
) -> None:
    name = f"fractional_max_pool{rank}d_with_indices"
    x = _data(*shape)
    samples = np.random.default_rng(5).random((shape[0], shape[1], rank)).astype(np.float32)
    xl = lucid.tensor(x, device=device, requires_grad=True)
    xt = ref.tensor(x, requires_grad=True)
    got = getattr(F, name)(
        xl, kernel, out, return_indices=flag, _random_samples=lucid.tensor(samples, device=device)
    )
    want = getattr(ref.nn.functional, name)(
        xt, kernel, out, return_indices=flag, _random_samples=ref.from_numpy(samples)
    )
    _close(got[0], want[0])
    np.testing.assert_array_equal(got[1].numpy(), want[1].numpy())
    got[0].sum().backward()
    want[0].sum().backward()
    _close(xl.grad, xt.grad)


def test_with_indices_functions_report_their_own_names() -> None:
    names = [
        f"{prefix}{rank}d_with_indices"
        for prefix in ("max_pool", "adaptive_max_pool")
        for rank in (1, 2, 3)
    ] + [f"fractional_max_pool{rank}d_with_indices" for rank in (2, 3)]
    for name in names:
        fn = getattr(F, name)
        assert fn.__name__ == fn.__qualname__ == name
        assert name in F.__all__
        assert name.removesuffix("_with_indices") in (fn.__doc__ or "")


# ── celu_ and the upsample aliases ───────────────────────────────────────────


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.5])
def test_celu_in_place(ref: ModuleType, device: str, alpha: float) -> None:
    x = _data(3, 4)
    ours = lucid.tensor(x, device=device)
    theirs = ref.tensor(x)
    out = F.celu_(ours, alpha)
    ref.nn.functional.celu_(theirs, alpha)
    assert out is ours
    _close(ours, theirs)
    default = lucid.tensor(x, device=device)
    F.celu_(default)
    _close(default, ref.nn.functional.celu(ref.tensor(x)))


def test_celu_in_place_extends_the_graph(ref: ModuleType) -> None:
    x = _data(6)
    leaf_l = lucid.tensor(x, requires_grad=True)
    leaf_t = ref.tensor(x, requires_grad=True)
    F.celu_(leaf_l * 1.0, 0.7).sum().backward()
    ref.nn.functional.celu_(leaf_t * 1.0, 0.7).sum().backward()
    _close(leaf_l.grad, leaf_t.grad)


@pytest.mark.parametrize(
    "name,shape,kwargs",
    [
        ("upsample", (1, 2, 5), dict(scale_factor=2, mode="linear")),
        ("upsample", (1, 2, 4, 5), dict(size=(7, 9), mode="bilinear", align_corners=True)),
        ("upsample", (1, 2, 4, 5), dict(scale_factor=2)),
        ("upsample", (1, 2, 3, 4, 5), dict(scale_factor=2, mode="trilinear", align_corners=False)),
        ("upsample_nearest", (1, 2, 4, 5), dict(scale_factor=3)),
        ("upsample_nearest", (1, 2, 4, 5), dict(size=(6, 7))),
        ("upsample_bilinear", (1, 2, 4, 5), dict(scale_factor=2)),
        ("upsample_bilinear", (1, 2, 4, 5), dict(size=(7, 9))),
    ],
)
def test_upsample_aliases_match_the_reference_and_warn(
    ref: ModuleType, device: str, name, shape, kwargs
) -> None:
    x = _data(*shape)
    xl = lucid.tensor(x, device=device, requires_grad=True)
    xt = ref.tensor(x, requires_grad=True)
    with pytest.warns(UserWarning, match="deprecated"):
        got = getattr(F, name)(xl, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = getattr(ref.nn.functional, name)(xt, **kwargs)
    _close(got, want)
    got.sum().backward()
    want.sum().backward()
    _close(xl.grad, xt.grad)


def test_upsample_alias_warning_points_at_the_caller() -> None:
    with pytest.warns(UserWarning) as record:
        F.upsample_nearest(lucid.zeros(1, 1, 2, 2), scale_factor=2)
    assert record[0].filename == __file__


# ── clip_grads_with_norm_ ────────────────────────────────────────────────────


def _grads(space: object, device: str | None, shapes: list[tuple[int, ...]], scale: float):
    params = []
    for i, shape in enumerate(shapes):
        data = _data(*shape, seed=i)
        if device is None:
            p = space.nn.Parameter(space.ones(*shape))
            p.grad = space.tensor(data * scale)
        else:
            p = nn.Parameter(lucid.ones(*shape, device=device))
            p.grad = lucid.tensor(data * scale, device=device)
        params.append(p)
    return params


@pytest.mark.parametrize("scale", [0.01, 1.0, 50.0], ids=["within", "near", "over"])
@pytest.mark.parametrize("max_norm", [0.5, 3.0])
def test_clip_grads_with_norm_matches_the_reference(
    ref: ModuleType, device: str, scale: float, max_norm: float
) -> None:
    shapes = [(3, 4), (5,), (2, 2, 2)]
    ours = _grads(None, device, shapes, scale)
    theirs = _grads(ref, None, shapes, scale)
    norm_l = get_total_norm([p.grad for p in ours])
    norm_t = ref.nn.utils.get_total_norm([p.grad for p in theirs])
    assert clip_grads_with_norm_(ours, max_norm, norm_l) is None
    ref.nn.utils.clip_grads_with_norm_(theirs, max_norm, norm_t)
    for a, b in zip(ours, theirs, strict=True):
        _close(a.grad, b.grad, atol=1e-6)
    clipped = get_total_norm([p.grad for p in ours])
    assert float(clipped) <= max_norm + 1e-4


def test_clip_grads_with_norm_agrees_with_clip_grad_norm_(device: str) -> None:
    a = _grads(None, device, [(3, 4), (5,)], 10.0)
    b = _grads(None, device, [(3, 4), (5,)], 10.0)
    total = nn.utils.clip_grad_norm_(a, 1.0)
    clip_grads_with_norm_(b, 1.0, total)
    for p, q in zip(a, b, strict=True):
        np.testing.assert_array_equal(p.grad.numpy(), q.grad.numpy())


def test_clip_grads_with_norm_edge_cases(ref: ModuleType) -> None:
    # a single parameter, a missing gradient, a non-finite norm
    p = _grads(None, "cpu", [(3,)], 10.0)[0]
    q = nn.Parameter(lucid.ones(2))
    clip_grads_with_norm_(p, 1.0, get_total_norm([p.grad]))
    clip_grads_with_norm_([q], 1.0, lucid.tensor(5.0))
    assert q.grad is None
    assert abs(float(get_total_norm([p.grad])) - 1.0) < 1e-3
    for arg in (float("nan"), float("inf")):
        r = _grads(None, "cpu", [(3,)], 1.0)[0]
        t = _grads(ref, None, [(3,)], 1.0)[0]
        clip_grads_with_norm_(r, 1.0, lucid.tensor(arg))
        ref.nn.utils.clip_grads_with_norm_(t, 1.0, ref.tensor(arg))
        np.testing.assert_array_equal(r.grad.numpy(), t.grad.numpy())


def test_clip_grads_with_norm_validates_the_norm(ref: ModuleType) -> None:
    p = _grads(None, "cpu", [(3,)], 1.0)
    with pytest.raises(TypeError, match="total_norm"):
        clip_grads_with_norm_(p, 1.0, 5.0)
    with pytest.raises(ValueError, match="one element"):
        clip_grads_with_norm_(p, 1.0, lucid.tensor([1.0, 2.0]))


@pytest.mark.parametrize("dtype", [lucid.int8, lucid.int16, lucid.int32])
@pytest.mark.parametrize("dilation", [1, 2])
def test_max_pool_indices_of_narrow_integers(
    ref: ModuleType, device: str, dtype, dilation: int
) -> None:
    """The CPU ``pad`` has no kernel for ``int8`` / ``int16``; the indices path
    padded with it, so only Metal answered (and the reference answers both)."""
    x = (np.arange(2 * 3 * 6 * 6).reshape(2, 3, 6, 6) * 7 % 11 - 5).astype(np.int32)
    kwargs = dict(kernel_size=2, stride=1, dilation=dilation)
    got = F.max_pool2d(lucid.tensor(x, dtype=dtype, device=device), return_indices=True, **kwargs)
    want = ref.nn.functional.max_pool2d_with_indices(
        ref.from_numpy(x).to(getattr(ref, str(dtype).split(".")[-1])), **kwargs
    )
    assert got[0].dtype == dtype
    np.testing.assert_array_equal(got[0].numpy(), want[0].numpy())
    np.testing.assert_array_equal(got[1].numpy(), want[1].numpy())
