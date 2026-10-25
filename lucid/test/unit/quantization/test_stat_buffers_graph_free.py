"""Observer and fake-quantize statistics never hold the autograd graph (LCD-188).

An observer folded each batch into ``min_val`` / ``max_val`` with ops that
recorded a graph, and a :class:`FakeQuantize` derived ``scale`` /
``zero_point`` from those buffers the same way.  The buffers then required
grad and carried a ``grad_fn``; the next step's update chained onto them, so
every training step's activations stayed reachable and QAT ran out of memory.

Statistics are written in one place, ``observer._store_running_stat``, under
``no_grad``.  These tests hold that for every stateful-statistics module in
the subsystem, on both devices:

* after 20 forward + backward steps in train mode, no buffer requires grad
  or carries a ``grad_fn``;
* the steps leave nothing alive — the engine's live byte count is flat.  A
  Python ``weakref`` cannot tell: the wrapper of a retained activation dies
  while the C++ graph still owns its storage, so the leak passed that check;
* under ``-m parity``, the buffers follow the reference through the same
  input sequence and the straight-through gradient reaching ``x`` matches.
"""

import gc
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.intrinsic.qat as nniqat
import lucid.quantization as Q
from lucid.test._fixtures.memory import live_bytes
from lucid.test._helpers.compare import assert_close

_STEPS = 20
_SHAPE = (64, 256)


def _fq(observer: type[Q.ObserverBase], **kwargs: object) -> Callable[[], nn.Module]:
    return lambda: Q.FakeQuantize(observer=observer, **kwargs)


_PER_CHANNEL = {"ch_axis": 1, "qscheme": Q.per_channel_symmetric, "qdtype": Q.qint8}

# Each factory builds a module whose forward updates statistics buffers.
_STATEFUL: dict[str, Callable[[], nn.Module]] = {
    "MinMaxObserver": Q.MinMaxObserver,
    "MovingAverageMinMaxObserver": Q.MovingAverageMinMaxObserver,
    "PerChannelMinMaxObserver": lambda: Q.PerChannelMinMaxObserver(ch_axis=1),
    "MovingAveragePerChannelMinMaxObserver": lambda: (
        Q.MovingAveragePerChannelMinMaxObserver(ch_axis=1)
    ),
    "HistogramObserver": lambda: Q.HistogramObserver(bins=64),
    "FakeQuantize[MovingAverage]": Q.FakeQuantize,
    "FakeQuantize[MinMax]": _fq(Q.MinMaxObserver),
    "FakeQuantize[PerChannel]": _fq(Q.PerChannelMinMaxObserver, **_PER_CHANNEL),
    "FakeQuantize[MovingAveragePerChannel]": _fq(
        Q.MovingAveragePerChannelMinMaxObserver, **_PER_CHANNEL
    ),
    "FakeQuantize[Histogram]": _fq(Q.HistogramObserver, bins=64),
    "FakeQuantize[FixedQParams]": _fq(
        Q.FixedQParamsObserver, scale=1.0 / 256.0, zero_point=0
    ),
}


def _qat_linear() -> nn.Module:
    model = nn.Sequential(nn.Linear(_SHAPE[1], 32), nn.ReLU())
    return Q.prepare_qat(model, Q.get_default_qat_qconfig_mapping())


def _qat_conv_bn() -> nn.Module:
    conv = nn.Conv1d(_SHAPE[0], 8, 3, padding=1, bias=False)
    return nniqat.ConvBnReLU1d(
        conv, nn.BatchNorm1d(8), qconfig=Q.get_default_qat_qconfig()
    )


_MODELS: dict[str, Callable[[], nn.Module]] = {
    **_STATEFUL,
    "prepare_qat[Linear+ReLU]": _qat_linear,
    "ConvBnReLU1d": _qat_conv_bn,
}


def _input(name: str, device: str) -> lucid.Tensor:
    shape = (1,) + _SHAPE if name == "ConvBnReLU1d" else _SHAPE
    return lucid.randn(*shape, device=device, requires_grad=True)


def _step(module: nn.Module, name: str, device: str) -> None:
    x = _input(name, device)
    hidden = x * 2.0
    # ``hidden`` is the activation a captured graph would keep alive.
    (module(hidden) * hidden.mean()).sum().backward()


@pytest.mark.parametrize("name", list(_MODELS))
def test_no_buffer_carries_a_graph_after_training(name: str, device: str) -> None:
    lucid.manual_seed(0)
    module = _MODELS[name]().to(device)
    module.train()
    for _ in range(_STEPS):
        _step(module, name, device)
    buffers = dict(module.named_buffers())
    assert buffers, f"{name} registers no buffers — the case tests nothing"
    holding = {
        key: buf.grad_fn
        for key, buf in buffers.items()
        if buf.requires_grad or buf.grad_fn is not None
    }
    assert not holding, f"{name}: buffers hold the autograd graph: {holding}"


@pytest.mark.parametrize("name", list(_MODELS))
def test_training_steps_keep_no_activation_alive(name: str, device: str) -> None:
    lucid.manual_seed(0)
    module = _MODELS[name]().to(device)
    module.train()
    _step(module, name, device)  # first call settles shapes and devices
    gc.collect()
    before = live_bytes(device)
    for _ in range(_STEPS):
        _step(module, name, device)
    gc.collect()
    grown = live_bytes(device) - before
    one_activation = 4 * _SHAPE[0] * _SHAPE[1]
    assert grown < one_activation, (
        f"{name}: {_STEPS} steps left {grown} bytes alive "
        f"(one activation is {one_activation})"
    )


# ── parity: the same statistics and the same straight-through gradient ──


def _ref_cases(ref: Any) -> dict[str, tuple[Callable[[], nn.Module], Any]]:
    obs = ref.ao.quantization.observer
    fq = ref.ao.quantization.FakeQuantize
    per_tensor = {"dtype": ref.quint8, "qscheme": ref.per_tensor_affine}
    per_channel = {
        "ch_axis": 1,
        "dtype": ref.qint8,
        "qscheme": ref.per_channel_symmetric,
    }
    return {
        "MinMaxObserver": (Q.MinMaxObserver, lambda: obs.MinMaxObserver(**per_tensor)),
        "MovingAverageMinMaxObserver": (
            lambda: Q.MovingAverageMinMaxObserver(averaging_constant=0.1),
            lambda: obs.MovingAverageMinMaxObserver(
                averaging_constant=0.1, **per_tensor
            ),
        ),
        "PerChannelMinMaxObserver": (
            lambda: Q.PerChannelMinMaxObserver(ch_axis=1),
            lambda: obs.PerChannelMinMaxObserver(**per_channel),
        ),
        "MovingAveragePerChannelMinMaxObserver": (
            lambda: Q.MovingAveragePerChannelMinMaxObserver(
                ch_axis=1, averaging_constant=0.1
            ),
            lambda: obs.MovingAveragePerChannelMinMaxObserver(
                averaging_constant=0.1, **per_channel
            ),
        ),
        "FakeQuantize[MovingAverage]": (
            Q.FakeQuantize,
            lambda: fq(
                observer=obs.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                **per_tensor,
            ),
        ),
        "FakeQuantize[PerChannel]": (
            _fq(Q.PerChannelMinMaxObserver, **_PER_CHANNEL),
            lambda: fq(
                observer=obs.PerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                **per_channel,
            ),
        ),
        "FakeQuantize[MovingAveragePerChannel]": (
            _fq(Q.MovingAveragePerChannelMinMaxObserver, **_PER_CHANNEL),
            lambda: fq(
                observer=obs.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                **per_channel,
            ),
        ),
    }


_PARITY = [
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "FakeQuantize[MovingAverage]",
    "FakeQuantize[PerChannel]",
    "FakeQuantize[MovingAveragePerChannel]",
]

_STE_PARITY = [name for name in _PARITY if name.startswith("FakeQuantize")]


def _observer_of(module: Any) -> Any:
    return getattr(module, "activation_post_process", module)


def _train_side_by_side(
    name: str, device: str, ref: Any
) -> tuple[nn.Module, Any, list[tuple[Any, Any]]]:
    """Feed both frameworks one input sequence; return them and each x.grad pair."""
    make_lucid, make_ref = _ref_cases(ref)[name]
    lm = make_lucid().to(device)
    rm = make_ref()
    lm.train()
    rm.train()
    grads = []
    rng = np.random.default_rng(188)
    for step in range(5):
        batch = (rng.standard_normal(_SHAPE) * (1.0 + step)).astype(np.float32)
        weight = rng.standard_normal(_SHAPE).astype(np.float32)
        lx = lucid.tensor(batch, device=device, requires_grad=True)
        rx = ref.tensor(batch, requires_grad=True)
        (lm(lx) * lucid.tensor(weight, device=device)).sum().backward()
        (rm(rx) * ref.tensor(weight)).sum().backward()
        grads.append((lx.grad, rx.grad))
    return lm, rm, grads


@pytest.mark.parity
@pytest.mark.parametrize("name", _PARITY)
def test_statistics_follow_the_reference(name: str, device: str, ref: Any) -> None:
    lm, rm, _ = _train_side_by_side(name, device, ref)
    lo, ro = _observer_of(lm), _observer_of(rm)
    for stat in ("min_val", "max_val"):
        assert_close(getattr(lo, stat), getattr(ro, stat), atol=1e-6, msg=stat)
    if lm is not lo:
        assert_close(lm.scale, rm.scale.reshape(lm.scale.shape), atol=1e-7)
        zp = rm.zero_point.reshape(lm.zero_point.shape).to(ref.float32)
        assert_close(lm.zero_point, zp, atol=0.0, msg="zero_point")


@pytest.mark.parity
@pytest.mark.parametrize("name", _STE_PARITY)
def test_ste_gradient_follows_the_reference(name: str, device: str, ref: Any) -> None:
    _, _, grads = _train_side_by_side(name, device, ref)
    for step, (lucid_grad, ref_grad) in enumerate(grads):
        assert_close(lucid_grad, ref_grad, atol=1e-6, msg=f"step {step} x.grad")
