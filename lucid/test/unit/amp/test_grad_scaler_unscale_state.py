"""``GradScaler`` keeps a per-optimizer stage, so gradients are unscaled once.

The AMP gradient-clipping recipe calls ``unscale_(opt)`` to clip the real
gradients and then ``step(opt)``.  ``step`` used to unscale again, dividing
the gradients by ``scale ** 2`` (CHA-46: ``p = [1.0]``, ``lr = 1``,
``init_scale = 4`` stepped to ``0.75`` instead of ``0.0``).  Each optimizer
now moves through *ready → unscaled → stepped* once per iteration,
``update()`` returns it to *ready*, and an overflow is recorded per
optimizer so it skips only that optimizer's step.

The finiteness check of a bfloat16 gradient runs on a float32 copy, so it
does not go through the half-precision ``isfinite`` kernel (CHA-34).
float16 gradients are refused (CHA-72, test_grad_scaler_half_grads.py).
"""

import math

import pytest

import lucid
import lucid.optim as optim
from lucid.amp import GradScaler
from lucid.nn.utils import clip_grad_norm_

INF = float("inf")
NAN = float("nan")


def _param(
    values: list[float], device: str, dtype: lucid.dtype = lucid.float32
) -> lucid.nn.Parameter:
    return lucid.nn.Parameter(lucid.tensor(values, device=device, dtype=dtype))


def _backward(scaler: GradScaler, p: lucid.Tensor, weights: list[float]) -> None:
    """Scaled backward of ``sum(p * weights)``, so ``p.grad == scale * weights``."""
    w = lucid.tensor(weights, device=p.device, dtype=p.dtype)
    scaler.scale((p * w).sum()).backward()


def _same(got: list[float], want: list[float]) -> bool:
    """Equal up to float32 rounding, with ``inf`` / ``nan`` matched by kind."""
    if len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if math.isnan(w) or math.isinf(w):
            if not (math.isnan(g) == math.isnan(w) and math.isinf(g) == math.isinf(w)):
                return False
            if math.isinf(w) and (g > 0) != (w > 0):
                return False
        elif abs(g - w) > 1e-5 * max(1.0, abs(w)):
            return False
    return True


# ── the reported repro ──────────────────────────────────────────────────────


def test_unscale_then_step_divides_once(device: str) -> None:
    p = _param([1.0], device)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(init_scale=4.0)
    _backward(scaler, p, [1.0])
    scaler.unscale_(opt)
    assert p.grad.tolist() == [1.0]
    scaler.step(opt)
    scaler.update()
    assert p.tolist() == [0.0]  # was 0.75: the gradient was divided by 4 twice


def test_step_alone_still_unscales(device: str) -> None:
    p = _param([1.0], device)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(init_scale=4.0)
    _backward(scaler, p, [1.0])
    scaler.step(opt)
    scaler.update()
    assert p.tolist() == [0.0]


# ── the stage machine ───────────────────────────────────────────────────────


def _ready(device: str) -> tuple[lucid.nn.Parameter, optim.SGD, GradScaler]:
    p = _param([1.0], device)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(init_scale=4.0)
    _backward(scaler, p, [1.0])
    return p, opt, scaler


def test_unscale_twice_raises(device: str) -> None:
    _, opt, scaler = _ready(device)
    scaler.unscale_(opt)
    with pytest.raises(RuntimeError, match="already been called on this optimizer"):
        scaler.unscale_(opt)


def test_unscale_after_step_raises(device: str) -> None:
    _, opt, scaler = _ready(device)
    scaler.step(opt)
    with pytest.raises(RuntimeError, match="after step"):
        scaler.unscale_(opt)


def test_step_twice_raises(device: str) -> None:
    p, opt, scaler = _ready(device)
    scaler.step(opt)
    with pytest.raises(RuntimeError, match="step\\(\\) has already been called"):
        scaler.step(opt)
    assert p.tolist() == [0.0]  # the refused second step did not move it


def test_closure_is_refused_while_enabled(device: str) -> None:
    _, opt, scaler = _ready(device)
    with pytest.raises(RuntimeError, match="Closure"):
        scaler.step(opt, closure=lambda: None)


def test_update_starts_the_next_iteration(device: str) -> None:
    p, opt, scaler = _ready(device)
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    _backward(scaler, p, [1.0])
    scaler.unscale_(opt)  # allowed again after update()
    scaler.step(opt)
    scaler.update()
    assert p.tolist() == [-1.0]


def test_update_with_new_scale_also_resets(device: str) -> None:
    p, opt, scaler = _ready(device)
    scaler.step(opt)
    scaler.update(new_scale=8.0)
    assert scaler.get_scale() == 8.0
    opt.zero_grad()
    _backward(scaler, p, [1.0])
    scaler.step(opt)
    assert p.tolist() == [-1.0]


def test_disabled_scaler_has_no_stages(device: str) -> None:
    p = _param([1.0], device)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(enabled=False)
    (p * 1.0).sum().backward()
    scaler.unscale_(opt)
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.step(opt)
    scaler.update()
    assert p.tolist() == [-1.0]


# ── overflow ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [INF, NAN], ids=["inf", "nan"])
def test_overflow_skips_the_step_and_backs_off(device: str, bad: float) -> None:
    p = _param([1.0, 2.0], device)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(init_scale=4.0, backoff_factor=0.5)
    _backward(scaler, p, [1.0, bad])
    assert scaler.step(opt) is None
    scaler.update()
    assert p.tolist() == [1.0, 2.0]
    assert scaler.get_scale() == 2.0


def test_unscale_divides_every_entry_even_on_overflow(device: str) -> None:
    p = _param([1.0, 2.0], device)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(init_scale=4.0)
    _backward(scaler, p, [1.0, INF])
    scaler.unscale_(opt)
    assert _same(p.grad.tolist(), [1.0, INF])


def test_overflow_in_bfloat16_gradients_is_caught(device: str) -> None:
    # The check reads a float32 copy, so it does not depend on the half
    # ``isfinite`` kernels (CHA-34).  float16 gradients are refused outright
    # (CHA-72) — see test_grad_scaler_half_grads.py.
    p = _param([1.0, 2.0], device, lucid.bfloat16)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(init_scale=4.0)
    _backward(scaler, p, [1.0, INF])
    scaler.step(opt)
    scaler.update()
    assert p.tolist() == [1.0, 2.0]
    assert scaler.get_scale() == 2.0


def test_overflow_skips_only_the_optimizer_that_overflowed(device: str) -> None:
    p1, p2 = _param([1.0], device), _param([1.0], device)
    o1, o2 = optim.SGD([p1], lr=1.0), optim.SGD([p2], lr=1.0)
    scaler = GradScaler(init_scale=4.0)
    w = lucid.tensor([INF], device=device)
    scaler.scale((p1 * 1.0 + p2 * w).sum()).backward()
    scaler.step(o1)
    scaler.step(o2)
    scaler.update()
    assert p1.tolist() == [0.0]
    assert p2.tolist() == [1.0]
    assert scaler.get_scale() == 2.0  # any overflow backs the scale off


def test_unscale_inside_autocast_stays_float32(device_gpu_only: str) -> None:
    # ``mul`` is autocast-eligible: inside a float16 scope it would cast
    # ``65536 * (1/65536)`` back to float16 (65536 itself overflows there).
    p = _param([1.0], device_gpu_only)
    opt = optim.SGD([p], lr=1.0)
    scaler = GradScaler(init_scale=2.0**16)
    _backward(scaler, p, [1.0])
    with lucid.amp.autocast(device_type="metal", dtype=lucid.float16):
        scaler.unscale_(opt)
        assert lucid.amp.autocast.get_autocast_dtype() == lucid.float16
    assert p.grad.dtype == lucid.float32
    assert p.grad.tolist() == [1.0]


def test_fused_step_found_inf_still_drives_update() -> None:
    # ``lucid.compile``'s fused step unscales inside its own graph and
    # hands the scaler only the bit; update() must still honour it.
    scaler = GradScaler(init_scale=4.0, growth_interval=1)
    scaler._found_inf = True
    scaler.update()
    assert scaler.get_scale() == 2.0
    scaler.update()
    assert scaler.get_scale() == 4.0


# ── against the reference ───────────────────────────────────────────────────

# Per iteration: the input weights, and whether to clip.  Iteration 2
# overflows; growth_interval=2 makes the scale grow on the clean ones.
_SCHEDULE = [
    ([0.5, -1.0, 2.0], True),
    ([3.0, 0.25, -0.5], True),
    ([1.0, INF, 2.0], True),
    ([-2.0, 1.5, 0.5], False),
    ([0.75, -0.25, 4.0], True),
    ([1.0, 1.0, 1.0], True),
]


class _Side:
    """One framework's view of the recipe, so both run the same steps."""

    def __init__(self, tensor, parameter, sgd, scaler, clip) -> None:
        self.tensor, self.parameter, self.sgd = tensor, parameter, sgd
        self.scaler_cls, self.clip = scaler, clip

    def run(self, n_opts: int, unscale_first: bool) -> list[tuple[list[list[float]], float]]:
        params = [self.parameter([1.0, -2.0, 0.5]) for _ in range(n_opts)]
        opts = [self.sgd([p], lr=0.1) for p in params]
        scaler = self.scaler_cls(init_scale=2.0**10, growth_interval=2)
        log = []
        for weights, clip in _SCHEDULE:
            for o in opts:
                o.zero_grad()
            # With two optimizers only the second sees the overflow.
            loss = None
            for i, p in enumerate(params):
                w = (
                    weights
                    if (i == n_opts - 1)
                    else [abs(x) if math.isfinite(x) else 1.0 for x in weights]
                )
                term = (p * self.tensor(w)).sum()
                loss = term if loss is None else loss + term
            scaler.scale(loss).backward()
            for o, p in zip(opts, params):
                if unscale_first:
                    scaler.unscale_(o)
                    if clip:
                        self.clip([p], 1.0)
                scaler.step(o)
            scaler.update()
            log.append(([p.tolist() for p in params], scaler.get_scale()))
        return log


def _lucid_side(device: str) -> _Side:
    return _Side(
        lambda v: lucid.tensor(v, device=device),
        lambda v: lucid.nn.Parameter(lucid.tensor(v, device=device)),
        optim.SGD,
        GradScaler,
        lambda ps, m: clip_grad_norm_(ps, max_norm=m),
    )


def _ref_side(ref) -> _Side:
    return _Side(
        lambda v: ref.tensor(v),
        lambda v: ref.nn.Parameter(ref.tensor(v)),
        ref.optim.SGD,
        lambda **kw: ref.amp.GradScaler("cpu", **kw),
        lambda ps, m: ref.nn.utils.clip_grad_norm_(ps, max_norm=m),
    )


@pytest.mark.parity
@pytest.mark.parametrize("n_opts", [1, 2], ids=["one-optimizer", "two-optimizers"])
@pytest.mark.parametrize("unscale_first", [True, False], ids=["unscale-clip-step", "step"])
def test_trajectory_matches_the_reference(
    device: str, ref, n_opts: int, unscale_first: bool
) -> None:
    got = _lucid_side(device).run(n_opts, unscale_first)
    want = _ref_side(ref).run(n_opts, unscale_first)
    for it, ((gp, gs), (wp, ws)) in enumerate(zip(got, want)):
        assert gs == ws, f"iteration {it}: scale {gs} != {ws}"
        for g, w in zip(gp, wp):
            assert _same(g, w), f"iteration {it}: params {gp} != {wp}"
