"""``GradScaler.unscale_`` keeps every gradient's dtype (CHA-72).

``unscale_`` divided each gradient on a float32 copy and wrote that float32
result back through ``set_grad``.  ``set_grad`` replaces the gradient's
storage without checking the dtype, so a float16 / bfloat16 slot then held
float32 bytes read as half-precision bit patterns: ``[4, 8, 16] / 4`` came
back as ``[0, 1.875, 1.875]`` on the CPU and ``[0, 1.875, 0]`` on Metal.

Now, as in the reference framework:

- a float16 gradient is refused with ``ValueError`` before any gradient is
  touched — loss scaling assumes float32 master weights;
- a bfloat16 gradient is unscaled in float32 and rounded back once, and
  float32 / float64 in their own dtype, so every gradient keeps its dtype,
  inside an autocast scope too;
- complex gradients are refused with ``NotImplementedError``.
"""

import contextlib
from collections.abc import Callable

import pytest

import lucid
from lucid.amp import GradScaler
from lucid.test._fixtures.devices import device_dtype_params

INF = float("inf")
NAN = float("nan")

#: The gradient dtypes ``unscale_`` divides.
_UNSCALED = [lucid.bfloat16, lucid.float32, lucid.float64]


class _PlainSGD:
    """What ``GradScaler`` reads of an optimizer, with a plain SGD step.

    The engine optimizers update float32 / float64 parameters only; this
    one updates any dtype, so the scaler can be tested on bfloat16.  ``lr``
    is a power of two so ``grad * lr`` is exact in every dtype and the
    reference framework's scalar promotion does not enter the comparison.
    """

    def __init__(
        self,
        params: list[object],
        no_grad: Callable[[], contextlib.AbstractContextManager[object]],
        lr: float = 0.25,
    ) -> None:
        self.param_groups = [{"params": params}]
        self._no_grad = no_grad
        self._lr = lr
        self.steps = 0

    def zero_grad(self) -> None:
        for p in self.param_groups[0]["params"]:
            p.grad = None  # type: ignore[attr-defined]

    def step(self) -> None:
        self.steps += 1
        with self._no_grad():
            for p in self.param_groups[0]["params"]:
                p.sub_(p.grad * self._lr)  # type: ignore[attr-defined]


def _param(values: list[float], device: str, dtype: lucid.dtype) -> lucid.nn.Parameter:
    return lucid.nn.Parameter(lucid.tensor(values, device=device, dtype=dtype))


def _backward(p: lucid.Tensor, weights: list[float], scale: float = 1.0) -> None:
    """Backward of ``scale * sum(p * weights)``, so ``p.grad == scale * weights``."""
    w = lucid.tensor(weights, device=p.device, dtype=p.dtype)
    ((p * w).sum() * scale).backward()


# ── the reported repro ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device,dtype", device_dtype_params(_UNSCALED))
def test_unscale_keeps_dtype_and_value(device: str, dtype: lucid.dtype) -> None:
    p = _param([1.0, 1.0, 1.0], device, dtype)
    _backward(p, [4.0, 8.0, 16.0])
    GradScaler(init_scale=4.0).unscale_(_PlainSGD([p], lucid.no_grad))
    assert p.grad.dtype == dtype
    assert p.grad.tolist() == [1.0, 2.0, 4.0]  # f16 was [0, 1.875, 1.875]


@pytest.mark.parametrize("device,dtype", device_dtype_params(_UNSCALED))
def test_unscale_at_the_default_scale(device: str, dtype: lucid.dtype) -> None:
    # 2**-16 is a normal number in bfloat16, and the float32 work copy keeps
    # gradients far below it exact.
    p = _param([1.0, 1.0, 1.0], device, dtype)
    _backward(p, [2.0**-20, 0.75, 3.0e4], scale=2.0**16)
    GradScaler().unscale_(_PlainSGD([p], lucid.no_grad))
    assert p.grad.dtype == dtype
    assert p.grad.tolist() == lucid.tensor([2.0**-20, 0.75, 3.0e4], dtype=dtype).tolist()


@pytest.mark.parametrize("device,dtype", device_dtype_params(_UNSCALED))
def test_unscale_inside_autocast_keeps_dtype(device: str, dtype: lucid.dtype) -> None:
    # ``mul`` is autocast-eligible: without the guard a float16 scope would
    # cast a bfloat16 gradient's work copy to float16, and a float32 one would
    # cast a float64 gradient down.  1 + 2**-40 is not a float32 number.
    p = _param([1.0], device, dtype)
    _backward(p, [1.0 + 2.0**-40], scale=4.0)
    want = lucid.tensor([1.0 + 2.0**-40], dtype=dtype).tolist()
    scope_dtype = lucid.float16 if device == "metal" else lucid.bfloat16
    with lucid.amp.autocast(device_type=device, dtype=scope_dtype):
        GradScaler(init_scale=4.0).unscale_(_PlainSGD([p], lucid.no_grad))
        assert lucid.amp.autocast.get_autocast_dtype() == scope_dtype
    assert p.grad.dtype == dtype
    assert p.grad.tolist() == want


# ── float16 and complex gradients are refused ───────────────────────────────


@pytest.mark.parametrize("device,dtype", device_dtype_params([lucid.float16]))
def test_float16_gradients_are_refused(device: str, dtype: lucid.dtype) -> None:
    p = _param([1.0, 1.0, 1.0], device, dtype)
    _backward(p, [4.0, 8.0, 16.0])
    opt = _PlainSGD([p], lucid.no_grad)
    scaler = GradScaler(init_scale=4.0)
    with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients."):
        scaler.unscale_(opt)
    # Nothing was touched, and the optimizer is still ready: step() tries the
    # unscale again and refuses again, without stepping.
    assert p.grad.dtype == lucid.float16
    assert p.grad.tolist() == [4.0, 8.0, 16.0]
    with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients."):
        scaler.step(opt)
    assert opt.steps == 0
    scaler.update()
    assert scaler.get_scale() == 4.0


def test_float16_refusal_leaves_the_other_gradients_alone(device: str) -> None:
    # The dtypes are checked before anything is divided, so a float32 gradient
    # ahead of the float16 one is not unscaled by the refused call.
    p32 = _param([1.0, 1.0], device, lucid.float32)
    p16 = _param([1.0, 1.0], device, lucid.float16)
    _backward(p32, [4.0, 8.0])
    _backward(p16, [4.0, 8.0])
    with pytest.raises(ValueError, match="FP16"):
        GradScaler(init_scale=4.0).unscale_(_PlainSGD([p32, p16], lucid.no_grad))
    assert p32.grad.tolist() == [4.0, 8.0]


def test_disabled_scaler_does_not_check_dtypes(device: str) -> None:
    p = _param([1.0, 1.0], device, lucid.float16)
    _backward(p, [4.0, 8.0])
    opt = _PlainSGD([p], lucid.no_grad)
    scaler = GradScaler(enabled=False)
    scaler.unscale_(opt)
    scaler.step(opt)
    assert opt.steps == 1
    assert p.grad.tolist() == [4.0, 8.0]


def test_complex_gradients_are_refused(device: str) -> None:
    p = lucid.nn.Parameter(lucid.tensor([1 + 0j, 1 + 0j], device=device))
    lucid.real((p * lucid.tensor([4 + 4j, 8 + 0j], device=device)).sum()).backward()
    before = p.grad.tolist()
    with pytest.raises(NotImplementedError, match="complex64"):
        GradScaler(init_scale=4.0).unscale_(_PlainSGD([p], lucid.no_grad))
    assert p.grad.dtype == lucid.complex64
    assert p.grad.tolist() == before


# ── overflow ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device,dtype", device_dtype_params(_UNSCALED))
@pytest.mark.parametrize("bad", [INF, NAN], ids=["inf", "nan"])
def test_overflow_skips_the_step(device: str, dtype: lucid.dtype, bad: float) -> None:
    p = _param([1.0, 2.0], device, dtype)
    _backward(p, [1.0, bad], scale=4.0)
    opt = _PlainSGD([p], lucid.no_grad)
    scaler = GradScaler(init_scale=4.0, backoff_factor=0.5)
    scaler.unscale_(opt)
    assert p.grad.dtype == dtype
    assert scaler.step(opt) is None
    scaler.update()
    assert opt.steps == 0
    assert p.tolist() == [1.0, 2.0]
    assert scaler.get_scale() == 2.0


@pytest.mark.parametrize("device,dtype", device_dtype_params(_UNSCALED))
def test_unscale_then_step_updates_in_the_parameter_dtype(device: str, dtype: lucid.dtype) -> None:
    p = _param([1.0, -2.0], device, dtype)
    _backward(p, [0.5, 3.0], scale=8.0)
    opt = _PlainSGD([p], lucid.no_grad)
    scaler = GradScaler(init_scale=8.0)
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    assert opt.steps == 1
    assert p.dtype == dtype
    assert p.tolist() == [0.875, -2.75]  # p - 0.25 * [0.5, 3.0]


# ── against the reference ──────────────────────────────────────────────────

# Iteration 2 overflows; growth_interval=2 makes the scale grow on the clean
# ones, so the trajectory covers growth, backoff and a skipped step.
_SCHEDULE = [
    [0.3, -1.7, 2.9],
    [3.1, 0.25, -0.55],
    [1.0, INF, 2.0],
    [1.1, 0.9, -0.3],
    [-2.2, 0.7, 0.45],
]


def _trajectory(lib, scaler, dtype, **device_kw) -> list[tuple[list[float], list[float], float]]:
    """``(unscaled grad, params, scale)`` after each iteration of the schedule."""
    p = lib.nn.Parameter(lib.tensor([1.0, -2.0, 0.5], dtype=dtype, **device_kw))
    opt = _PlainSGD([p], lib.no_grad)
    log = []
    for weights in _SCHEDULE:
        opt.zero_grad()
        scaler.scale((p * lib.tensor(weights, dtype=dtype, **device_kw)).sum()).backward()
        scaler.unscale_(opt)
        grad = p.grad.tolist()
        scaler.step(opt)
        scaler.update()
        log.append((grad, p.tolist(), scaler.get_scale()))
    return log


@pytest.mark.parity
@pytest.mark.parametrize("device,dtype", device_dtype_params([lucid.bfloat16, lucid.float32]))
def test_trajectory_matches_the_reference(device: str, dtype: lucid.dtype, ref) -> None:
    got = _trajectory(
        lucid, GradScaler(init_scale=2.0**10, growth_interval=2), dtype, device=device
    )
    want = _trajectory(
        ref,
        ref.amp.GradScaler("cpu", init_scale=2.0**10, growth_interval=2),
        getattr(ref, str(dtype).split(".")[-1]),
    )
    for it, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"iteration {it}: (grad, params, scale) {g} != {w}"


@pytest.mark.parity
def test_float16_refusal_matches_the_reference(device: str, ref) -> None:
    rp = ref.nn.Parameter(ref.ones(2, dtype=ref.float16))
    rs = ref.amp.GradScaler("cpu", init_scale=4.0)
    rs.scale((rp * ref.tensor([4.0, 8.0], dtype=ref.float16)).sum()).backward()
    with pytest.raises(ValueError) as want:
        rs.unscale_(_PlainSGD([rp], ref.no_grad))
    p = _param([1.0, 1.0], device, lucid.float16)
    _backward(p, [4.0, 8.0], scale=4.0)
    with pytest.raises(ValueError) as got:
        GradScaler(init_scale=4.0).unscale_(_PlainSGD([p], lucid.no_grad))
    assert str(got.value) == str(want.value)
