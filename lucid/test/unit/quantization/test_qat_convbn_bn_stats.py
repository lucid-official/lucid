"""QAT ``ConvBn*`` trains its BatchNorm statistics (LCD-394).

The fused QAT conv folded BN's *running* statistics into the weight and never
called the wrapped ``bn``, so training never used batch statistics and the
running mean / variance stayed at their initial values — ``convert`` then
baked an un-normalised conv.  Each fused module now runs its conv output
through ``bn`` (batch statistics and the momentum update in training, running
statistics in eval or once frozen), as the reference does.

These tests hold that for every fused QAT module in ``lucid.nn.intrinsic.qat``:

* training moves the running statistics and ``num_batches_tracked``;
  ``freeze_bn_stats`` stops it and survives ``train()``; eval output is the
  folded conv;
* under ``-m parity``, five training steps follow the reference — outputs,
  running statistics and every gradient at every step — then frozen steps,
  the eval output, and the converted int8 module's output.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.intrinsic.qat as nniqat
import lucid.quantization as Q
from lucid.test._helpers.compare import assert_close

_IN, _OUT, _K = 3, 8, 3
_STEPS = 5

_LUCID_CLS = {
    ("ConvBn", 1): nniqat.ConvBn1d,
    ("ConvBn", 2): nniqat.ConvBn2d,
    ("ConvBn", 3): nniqat.ConvBn3d,
    ("ConvBnReLU", 1): nniqat.ConvBnReLU1d,
    ("ConvBnReLU", 2): nniqat.ConvBnReLU2d,
    ("ConvBnReLU", 3): nniqat.ConvBnReLU3d,
}
_ALL = [f"{kind}{rank}d" for kind, rank in _LUCID_CLS]
# The card's guard set plus the variants that take other BN paths: a conv with
# its own bias, and ``momentum=None`` (cumulative average, ``num_batches_tracked``).
_PARITY = [
    ("ConvBn1d", False, 0.1),
    ("ConvBn2d", False, 0.1),
    ("ConvBn3d", False, 0.1),
    ("ConvBnReLU2d", False, 0.1),
    ("ConvBnReLU1d", False, 0.1),
    ("ConvBnReLU3d", False, 0.1),
    ("ConvBn2d", True, 0.1),
    ("ConvBn2d", False, None),
]


def _split(name: str) -> tuple[str, int]:
    return name[:-2], int(name[-2])


def _params(rank: int, conv_bias: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(394 + rank)
    params = {
        "weight": rng.standard_normal((_OUT, _IN) + (_K,) * rank) * 0.3,
        "gamma": rng.uniform(0.5, 1.5, _OUT),
        "beta": rng.standard_normal(_OUT) * 0.1,
    }
    if conv_bias:
        params["bias"] = rng.standard_normal(_OUT) * 0.2
    return {k: v.astype(np.float32) for k, v in params.items()}


def _lucid_module(
    name: str, device: str, conv_bias: bool = False, momentum: float | None = 0.1
) -> nn.Module:
    kind, rank = _split(name)
    conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[rank - 1](_IN, _OUT, _K, padding=1, bias=conv_bias)
    bn = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[rank - 1](_OUT, momentum=momentum)
    p = _params(rank, conv_bias)
    with lucid.no_grad():
        conv.weight.copy_(lucid.tensor(p["weight"]))
        bn.weight.copy_(lucid.tensor(p["gamma"]))
        bn.bias.copy_(lucid.tensor(p["beta"]))
        if conv_bias:
            conv.bias.copy_(lucid.tensor(p["bias"]))
    module = _LUCID_CLS[(kind, rank)](conv, bn, qconfig=Q.get_default_qat_qconfig())
    return module.to(device)


def _batch(rng: np.random.Generator, rank: int) -> np.ndarray:
    # Off-centre and wide, so batch and running statistics differ visibly.
    shape = (4, _IN) + (6,) * rank
    return (rng.standard_normal(shape) * 3.0 + 5.0).astype(np.float32)


def _train_steps(module: nn.Module, rank: int, device: str, steps: int) -> None:
    rng = np.random.default_rng(7)
    for _ in range(steps):
        x = lucid.tensor(_batch(rng, rank), device=device)
        module(x).sum().backward()


def _bn_state(module: nn.Module) -> tuple[np.ndarray, np.ndarray, int]:
    bn = module.bn
    return (
        bn.running_mean.cpu().numpy().copy(),
        bn.running_var.cpu().numpy().copy(),
        int(bn.num_batches_tracked.item()),
    )


# ── behaviour (no reference needed) ─────────────────────────────────────────


@pytest.mark.parametrize("name", _ALL)
def test_training_updates_running_statistics(name: str, device: str) -> None:
    module = _lucid_module(name, device)
    module.train()
    _train_steps(module, _split(name)[1], device, _STEPS)
    mean, var, tracked = _bn_state(module)
    assert tracked == _STEPS
    # Inputs centre near 5 → conv outputs far from the initial (0, 1).
    assert np.abs(mean).max() > 0.1
    assert np.abs(var - 1.0).max() > 0.1


@pytest.mark.parametrize("name", _ALL)
def test_freeze_bn_stats_stops_updates_and_survives_train(name: str, device: str) -> None:
    module = _lucid_module(name, device)
    module.train()
    _train_steps(module, _split(name)[1], device, 2)
    before = _bn_state(module)
    assert module.freeze_bn_stats() is module
    module.train()  # a frozen BN must stay frozen under train()
    assert module.training and not module.bn.training
    _train_steps(module, _split(name)[1], device, 2)
    after = _bn_state(module)
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2] == before[2]
    module.update_bn_stats()
    _train_steps(module, _split(name)[1], device, 1)
    assert _bn_state(module)[2] == before[2] + 1


@pytest.mark.parametrize("name", _ALL)
def test_eval_output_is_the_folded_conv(name: str, device: str) -> None:
    kind, rank = _split(name)
    module = _lucid_module(name, device)
    module.train()
    _train_steps(module, rank, device, 3)
    module.eval()
    assert not module.bn.training
    rng = np.random.default_rng(11)
    x = lucid.tensor(_batch(rng, rank), device=device)
    module.activation_post_process.disable_fake_quant()
    module.weight_fake_quant.disable_fake_quant()
    weight, bias = module._fold()
    expected = module._conv(x, weight)
    expected = expected + bias.reshape((1, _OUT) + (1,) * rank)
    if kind == "ConvBnReLU":
        expected = lucid.nn.functional.relu(expected)
    assert_close(module(x), expected, atol=1e-4, rtol=1e-4)


def test_bn_without_running_statistics_is_refused() -> None:
    bn = nn.BatchNorm2d(_OUT, track_running_stats=False)
    with pytest.raises(ValueError, match="track_running_stats"):
        nniqat.ConvBn2d(nn.Conv2d(_IN, _OUT, _K), bn, qconfig=Q.get_default_qat_qconfig())


# ── parity ──────────────────────────────────────────────────────────────────


def _ref_qconfig(ref: Any) -> Any:
    obs = ref.ao.quantization.observer
    fq = ref.ao.quantization.FakeQuantize
    return ref.ao.quantization.QConfig(
        activation=fq.with_args(
            observer=obs.MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=ref.quint8,
            qscheme=ref.per_tensor_affine,
        ),
        weight=fq.with_args(
            observer=obs.PerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=ref.qint8,
            qscheme=ref.per_channel_symmetric,
            ch_axis=0,
        ),
    )


def _ref_module(ref: Any, name: str, conv_bias: bool, momentum: float | None) -> tuple[Any, Any]:
    """The reference QAT module and its output fake-quant (added by prepare)."""
    kind, rank = _split(name)
    conv = (ref.nn.Conv1d, ref.nn.Conv2d, ref.nn.Conv3d)[rank - 1](
        _IN, _OUT, _K, padding=1, bias=conv_bias
    )
    bn = (ref.nn.BatchNorm1d, ref.nn.BatchNorm2d, ref.nn.BatchNorm3d)[rank - 1](
        _OUT, momentum=momentum
    )
    p = _params(rank, conv_bias)
    with ref.no_grad():
        conv.weight.copy_(ref.tensor(p["weight"]))
        bn.weight.copy_(ref.tensor(p["gamma"]))
        bn.bias.copy_(ref.tensor(p["beta"]))
        if conv_bias:
            conv.bias.copy_(ref.tensor(p["bias"]))
    nni = ref.ao.nn.intrinsic
    fused = (
        getattr(nni, name)(conv, bn, ref.nn.ReLU())
        if kind == "ConvBnReLU"
        else getattr(nni, name)(conv, bn)
    )
    qconfig = _ref_qconfig(ref)
    fused.qconfig = qconfig
    module = getattr(nni.qat, name).from_float(fused)
    return module, qconfig.activation()


def _assert_step(
    step: str,
    lm: nn.Module,
    rm: Any,
    out: tuple[Any, Any],
    x_grad: tuple[Any, Any],
    conv_bias: bool,
) -> None:
    assert_close(out[0], out[1].detach(), atol=1e-5, msg=f"{step} output")
    assert_close(x_grad[0], x_grad[1], atol=1e-5, msg=f"{step} x.grad")
    assert_close(lm.bn.running_mean, rm.bn.running_mean, atol=1e-5, msg=f"{step} mean")
    assert_close(lm.bn.running_var, rm.bn.running_var, atol=1e-5, msg=f"{step} var")
    assert int(lm.bn.num_batches_tracked.item()) == int(rm.bn.num_batches_tracked)
    grads = [
        (lm.conv.weight.grad, rm.weight.grad, "conv.weight"),
        (lm.bn.weight.grad, rm.bn.weight.grad, "bn.weight"),
        (lm.bn.bias.grad, rm.bn.bias.grad, "bn.bias"),
    ]
    if conv_bias:
        grads.append((lm.conv.bias.grad, rm.bias.grad, "conv.bias"))
    for lucid_grad, ref_grad, what in grads:
        assert_close(lucid_grad, ref_grad, atol=1e-4, msg=f"{step} {what}.grad")


def _side_by_side(
    lm: nn.Module,
    rm: Any,
    r_afq: Any,
    ref: Any,
    rank: int,
    device: str,
    conv_bias: bool,
    tag: str,
    rng: np.random.Generator,
) -> None:
    for step in range(_STEPS):
        batch = _batch(rng, rank)
        upstream = rng.standard_normal((4, _OUT) + (6,) * rank).astype(np.float32)
        lx = lucid.tensor(batch, device=device, requires_grad=True)
        rx = ref.tensor(batch, requires_grad=True)
        ly, ry = lm(lx), r_afq(rm(rx))
        (ly * lucid.tensor(upstream, device=device)).sum().backward()
        (ry * ref.tensor(upstream)).sum().backward()
        _assert_step(f"{tag} step {step}", lm, rm, (ly, ry), (lx.grad, rx.grad), conv_bias)


@pytest.mark.parity
@pytest.mark.parametrize(("name", "conv_bias", "momentum"), _PARITY)
def test_training_follows_the_reference(
    name: str, conv_bias: bool, momentum: float | None, device: str, ref: Any
) -> None:
    rank = _split(name)[1]
    lm = _lucid_module(name, device, conv_bias, momentum)
    rm, r_afq = _ref_module(ref, name, conv_bias, momentum)
    lm.train()
    rm.train()
    rng = np.random.default_rng(394)
    _side_by_side(lm, rm, r_afq, ref, rank, device, conv_bias, "train", rng)

    lm.freeze_bn_stats()
    rm.freeze_bn_stats()
    lm.train()
    rm.train()
    frozen = _bn_state(lm)
    _side_by_side(lm, rm, r_afq, ref, rank, device, conv_bias, "frozen", rng)
    np.testing.assert_array_equal(_bn_state(lm)[0], frozen[0])
    np.testing.assert_array_equal(_bn_state(lm)[1], frozen[1])

    lm.eval()
    rm.eval()
    batch = _batch(rng, rank)
    with lucid.no_grad(), ref.no_grad():
        ly = lm(lucid.tensor(batch, device=device))
        ry = r_afq(rm(ref.tensor(batch)))
    assert_close(ly, ry, atol=1e-5, msg="eval output")


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n, bias, mom in _PARITY if not bias and mom])
def test_converted_module_follows_the_reference(name: str, device: str, ref: Any) -> None:
    kind, rank = _split(name)
    lm = _lucid_module(name, device)
    rm, r_afq = _ref_module(ref, name, False, 0.1)
    lm.train()
    rm.train()
    rng = np.random.default_rng(1394)
    for _ in range(_STEPS):
        batch = _batch(rng, rank)
        lm(lucid.tensor(batch, device=device)).sum().backward()
        r_afq(rm(ref.tensor(batch))).sum().backward()
    lm.eval()
    rm.eval()

    converted = Q.convert(nn.Sequential(lm))[0]
    rm.activation_post_process = r_afq
    quantized = ref.ao.nn.intrinsic.quantized if kind == "ConvBnReLU" else None
    ref_cls = (
        getattr(quantized, f"ConvReLU{rank}d")
        if quantized is not None
        else getattr(ref.ao.nn.quantized, f"Conv{rank}d")
    )
    previous_engine = ref.backends.quantized.engine
    ref.backends.quantized.engine = "qnnpack"
    try:
        ref_q = ref_cls.from_float(rm)
        # ``from_float`` replaces the module's weight / bias with the fold of the
        # final running statistics — the values the int8 conv must carry.
        assert_close(converted.bias, rm.bias.detach(), atol=1e-5, msg="folded bias")
        w_scale, _ = rm.weight_fake_quant.calculate_qparams()
        assert_close(converted.weight_scale, w_scale, atol=1e-7, msg="weight scale")
        assert_close(converted.scale, ref_q.scale, atol=1e-7, msg="output scale")
        assert float(converted.zero_point.item()) == ref_q.zero_point

        scale, zero_point = 0.1, 40
        batch = _batch(rng, rank)
        x_q = ref.quantize_per_tensor(ref.tensor(batch), scale, zero_point, ref.quint8)
        ref_out = ref_q(x_q).dequantize().numpy()
    finally:
        # "none" (no engine chosen yet) cannot be assigned back.
        if previous_engine != "none":
            ref.backends.quantized.engine = previous_engine
    lx = Q.fake_quantize(
        lucid.tensor(batch, device=device),
        lucid.tensor(scale, device=device),
        lucid.tensor(float(zero_point), device=device),
        0,
        255,
    )
    out = converted.to(device)(lx).cpu().numpy()
    # The reference requantizes through an int32 bias and fixed-point
    # multipliers, so a value may land on the neighbouring grid point.
    step = ref_q.scale
    diff = np.abs(out - ref_out)
    assert diff.max() <= step * 1.001 + 1e-6, f"off by {diff.max() / step:.2f} steps"
    assert diff.mean() <= 0.25 * step
