"""Mixed-precision training soak: ``autocast`` + ``GradScaler`` over whole runs.

The unit tests around ``GradScaler`` poke one call at a time.  This module
drives whole training runs — a small MLP and a small conv net under
``autocast(float16 | bfloat16)`` with a ``GradScaler`` — across

    device {cpu, metal} × autocast dtype × optimizer {SGD+momentum, AdamW,
    RMSprop} × path {eager, ``lucid.compile.fused_step``}

(``fused_step`` is Metal-only; the CPU row is eager only.  Lion is not a Lucid
optimizer.)  Each cell checks the run as a user would see it:

* the loss goes down and neither the parameters nor the scale go non-finite;
* an overflow step is skipped — parameters and optimizer state untouched, the
  scale backed off — and the next steps resume and grow it on schedule;
* ``unscale_`` + ``clip_grad_norm_`` + ``step`` tracks the reference's scale
  trajectory, skip decisions and parameters;
* a mid-run checkpoint of model + optimizer + scaler resumes identically;
* a long run holds the engine's live bytes constant.

Value comparisons that cross the GPU use a tolerance: CI's Apple GPU is not
the one the numbers were first seen on.  Skip / no-change decisions are exact.
"""

import copy
import gc
import math
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.amp import GradScaler, autocast
from lucid.compile import fused_step
from lucid.nn.utils import clip_grad_norm_
from lucid.test._fixtures.devices import metal_available
from lucid.test._fixtures.memory import live_bytes

INF = float("inf")
SEED = 1234

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]
_DTYPES = [lucid.float16, lucid.bfloat16]
_DTYPE_IDS = ["fp16", "bf16"]

# name -> (class name, kwargs).  Resolved on ``lucid.optim`` and, for the
# reference comparison, on the reference's ``optim`` — same names, same kwargs.
_OPTIMIZERS: dict[str, tuple[str, dict[str, float]]] = {
    "sgd_momentum": ("SGD", {"lr": 0.05, "momentum": 0.9}),
    "adamw": ("AdamW", {"lr": 1e-2}),
    "rmsprop": ("RMSprop", {"lr": 3e-3}),
}


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).relu())


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(F.max_pool2d(self.conv(x).relu(), 2).flatten(1))


_MODELS: dict[str, tuple[Callable[[], nn.Module], tuple[int, ...]]] = {
    "mlp": (_MLP, (16, 8)),
    "conv": (_ConvNet, (8, 1, 8, 8)),
}


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    """MSE in float32 — module-level so ``fused_step`` keys its cache on it."""
    return F.mse_loss(out.float(), target)


def _cells(*, models: tuple[str, ...] = ("mlp",)) -> list:
    """Every (model, device, dtype, optimizer, path) cell; compiled only on Metal."""
    cells = []
    for model in models:
        for device in _DEVICES:
            for dtype, dtype_id in zip(_DTYPES, _DTYPE_IDS, strict=True):
                for opt in _OPTIMIZERS:
                    for path in ("eager", "compiled"):
                        if path == "compiled" and device != "metal":
                            continue
                        cells.append(
                            pytest.param(
                                model, device, dtype, opt, path,
                                id=f"{model}-{device}-{dtype_id}-{opt}-{path}",
                            )
                        )  # fmt: skip
    return cells


def _flat(obj: object) -> list[np.ndarray]:
    """Every tensor / number in a nested state, as numpy arrays, in order."""
    if isinstance(obj, lucid.Tensor):
        return [obj.numpy().copy()]
    if isinstance(obj, dict):
        return [a for v in obj.values() for a in _flat(v)]
    if isinstance(obj, (list, tuple)):
        return [a for v in obj for a in _flat(v)]
    if isinstance(obj, (int, float)):
        return [np.asarray(obj)]
    return []


def _clone(obj: object) -> object:
    """Deep copy of a nested state with tensors cloned, so it never aliases live buffers."""
    if isinstance(obj, lucid.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone(v) for v in obj)
    return copy.deepcopy(obj)


def _same(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


class _Trainer:
    """One training run: model + optimizer + scaler on an eager or fused path."""

    def __init__(
        self,
        model: str,
        device: str,
        dtype: lucid.dtype,
        opt: str,
        path: str,
        *,
        init_scale: float = 2.0**8,
        growth_interval: int = 3,
        n_batches: int = 1,
        init_seed: int = SEED,
    ) -> None:
        make, x_shape = _MODELS[model]
        self.device, self.dtype, self.path = device, dtype, path
        lucid.manual_seed(init_seed)
        self.model = make().to(device)
        cls, kwargs = _OPTIMIZERS[opt]
        self.opt = getattr(optim, cls)(self.model.parameters(), **kwargs)
        self.scaler = GradScaler(init_scale=init_scale, growth_interval=growth_interval)
        lucid.manual_seed(SEED + 1)
        self.xs = [lucid.randn(*x_shape, device=device) for _ in range(n_batches)]
        self.ys = [lucid.randn(x_shape[0], 4, device=device) for _ in range(n_batches)]
        self._fused = (
            fused_step(self.model, _loss, self.opt, grad_scaler=self.scaler)
            if path == "compiled"
            else None
        )

    def step(self, i: int, *, bad: bool = False) -> float:
        """Run iteration ``i``; ``bad`` feeds an infinite input, so every gradient overflows."""
        x, y = self.xs[i % len(self.xs)], self.ys[i % len(self.ys)]
        if bad:
            x = x * INF
        with autocast(self.device, dtype=self.dtype):
            if self._fused is not None:
                return float(self._fused(x, y).item())
            self.opt.zero_grad()
            loss = _loss(self.model(x), y)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.item())

    def accumulate_step(self, i: int, *, bad_micro: int | None = None) -> float:
        """One eager step over every batch as micro-batches (loss / n), one ``step`` for all."""
        n = len(self.xs)
        self.opt.zero_grad()
        total = 0.0
        for k in range(n):
            x = self.xs[k] * INF if k == bad_micro else self.xs[k]
            with autocast(self.device, dtype=self.dtype):
                loss = _loss(self.model(x), self.ys[k]) / n
            self.scaler.scale(loss).backward()
            total += float(loss.item())
        self.scaler.step(self.opt)
        self.scaler.update()
        return total

    def params(self) -> list[np.ndarray]:
        return [p.numpy().copy() for p in self.model.parameters()]

    def opt_state(self) -> list[np.ndarray]:
        return _flat(self.opt.state_dict())

    def checkpoint(self) -> dict[str, object]:
        return {
            "model": _clone(self.model.state_dict()),
            "opt": _clone(self.opt.state_dict()),
            "scaler": self.scaler.state_dict(),
        }

    def restore(self, ck: dict[str, object]) -> None:
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.scaler.load_state_dict(ck["scaler"])


def _expected_scales(
    init: float,
    bad_flags: list[bool],
    growth_interval: int,
    growth: float = 2.0,
    backoff: float = 0.5,
) -> list[float]:
    """The documented growth / backoff schedule, one scale per iteration."""
    scale, tracker, out = init, 0, []
    for bad in bad_flags:
        if bad:
            scale, tracker = scale * backoff, 0
        else:
            tracker += 1
            if tracker >= growth_interval:
                scale, tracker = scale * growth, 0
        out.append(scale)
    return out


# ── training converges ───────────────────────────────────────────────────────


@pytest.mark.parametrize("model, device, dtype, opt, path", _cells(models=("mlp", "conv")))
def test_trains_under_autocast_and_scaler(
    model: str, device: str, dtype: lucid.dtype, opt: str, path: str
) -> None:
    t = _Trainer(model, device, dtype, opt, path)
    losses = [t.step(i) for i in range(30)]

    assert all(math.isfinite(v) for v in losses), f"non-finite loss: {losses}"
    assert losses[-1] < 0.85 * losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(np.isfinite(p).all() for p in t.params()), "a parameter went non-finite"
    assert math.isfinite(t.scaler.get_scale()) and t.scaler.get_scale() > 0
    # 30 clean steps at growth_interval=3 grow the scale on every third step.
    assert t.scaler.get_scale() == 2.0**8 * 2.0**10


# ── skip on overflow ─────────────────────────────────────────────────────────

# Two isolated overflows, then two back to back; clean runs of length 3 between
# them are exactly one growth interval.
_BAD = [False, False, True, False, False, False, False, True, True, False, False, False]


@pytest.mark.parametrize("model, device, dtype, opt, path", _cells(models=("mlp", "conv")))
def test_overflow_step_is_skipped_and_the_run_resumes(
    model: str, device: str, dtype: lucid.dtype, opt: str, path: str
) -> None:
    t = _Trainer(model, device, dtype, opt, path)
    want = _expected_scales(2.0**8, _BAD, growth_interval=3)
    losses = []
    for i, bad in enumerate(_BAD):
        before_params, before_state = t.params(), t.opt_state()
        loss = t.step(i, bad=bad)
        if bad:
            assert _same(t.params(), before_params), f"step {i}: a skipped step moved the params"
            assert _same(t.opt_state(), before_state), f"step {i}: a skipped step moved opt state"
        else:
            assert not _same(t.params(), before_params), f"step {i}: a clean step did not step"
            assert math.isfinite(loss), f"step {i}: clean step has loss {loss}"
            losses.append(loss)
        assert (
            t.scaler.get_scale() == want[i]
        ), f"step {i}: scale {t.scaler.get_scale()} != {want[i]}"

    assert all(np.isfinite(p).all() for p in t.params())
    assert losses[-1] < losses[0], "training did not resume after the overflows"
    assert want[2] == want[1] * 0.5, "schedule sanity: backoff_factor"


# ── scaling is invisible when nothing overflows ──────────────────────────────


def _lossless_tol(device: str, dtype: lucid.dtype) -> dict[str, float]:
    """CPU and bfloat16 match to round-off.  Metal float16 flushes tiny unscaled
    products to zero, which scaling exists to prevent — so the two runs may
    differ by that much, and only that much."""
    if device == "metal" and dtype == lucid.float16:
        return {"rtol": 2e-3, "atol": 2e-4}
    return {"rtol": 1e-5, "atol": 1e-6}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("opt", list(_OPTIMIZERS))
def test_power_of_two_scaling_is_lossless(device: str, dtype: lucid.dtype, opt: str) -> None:
    # A power-of-two scale only shifts exponents, so a run that never overflows
    # must equal the same run with the scaler disabled (CPU: bit for bit).
    on = _Trainer("mlp", device, dtype, opt, "eager", n_batches=2)
    off = _Trainer("mlp", device, dtype, opt, "eager", n_batches=2)
    off.scaler = GradScaler(enabled=False)
    losses_on = [on.step(i) for i in range(15)]
    losses_off = [off.step(i) for i in range(15)]

    tol = _lossless_tol(device, dtype)
    np.testing.assert_allclose(losses_on, losses_off, **tol)
    for a, b in zip(on.params(), off.params(), strict=True):
        np.testing.assert_allclose(a, b, **tol)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("opt", list(_OPTIMIZERS))
def test_gradient_accumulation_with_the_scaler(device: str, dtype: lucid.dtype, opt: str) -> None:
    on = _Trainer("mlp", device, dtype, opt, "eager", n_batches=2)
    off = _Trainer("mlp", device, dtype, opt, "eager", n_batches=2)
    off.scaler = GradScaler(enabled=False)
    for i in range(6):
        on.accumulate_step(i)
        off.accumulate_step(i)
    for a, b in zip(on.params(), off.params(), strict=True):
        np.testing.assert_allclose(a, b, **_lossless_tol(device, dtype))

    # One overflowing micro-batch poisons the accumulated gradient: the whole
    # step is skipped, not just that micro-batch's share.
    before, state, scale = on.params(), on.opt_state(), on.scaler.get_scale()
    on.accumulate_step(6, bad_micro=1)
    assert _same(on.params(), before) and _same(on.opt_state(), state)
    assert on.scaler.get_scale() == scale * 0.5
    on.accumulate_step(7)
    assert not _same(on.params(), before), "training did not resume"


# ── eager and fused paths agree ──────────────────────────────────────────────


@pytest.mark.skipif(not metal_available(), reason="fused_step is Metal-only")
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("opt", list(_OPTIMIZERS))
def test_fused_step_tracks_the_eager_run(dtype: lucid.dtype, opt: str) -> None:
    eager = _Trainer("mlp", "metal", dtype, opt, "eager", n_batches=2)
    fused = _Trainer("mlp", "metal", dtype, opt, "compiled", n_batches=2)
    bad = {5, 6}
    le = [eager.step(i, bad=i in bad) for i in range(15)]
    lf = [fused.step(i, bad=i in bad) for i in range(15)]

    assert eager.scaler.state_dict() == fused.scaler.state_dict(), "scale schedules diverged"
    ok = [i for i in range(15) if i not in bad]
    # Half-precision rounding differs between the two executors; the trajectory
    # must still agree to a couple of percent.
    np.testing.assert_allclose([le[i] for i in ok], [lf[i] for i in ok], rtol=3e-2, atol=1e-3)


# ── unscale_ + clip against the reference ────────────────────────────────────

_REF_BAD = {3, 8}
_REF_ITERS = 12


class _Side:
    """One framework's view of the same recipe, so both take the same steps."""

    def __init__(
        self,
        fw: ModuleType,
        tensor: Callable[[np.ndarray], object],
        parameter: Callable[[np.ndarray], object],
        scaler: Callable[..., object],
        autocast_ctx: Callable[[], object],
        clip: Callable[[list, float], object],
    ) -> None:
        self.fw, self.tensor, self.parameter = fw, tensor, parameter
        self.scaler, self.autocast_ctx, self.clip = scaler, autocast_ctx, clip

    def run(
        self, opt: str, unscale_first: bool
    ) -> list[tuple[list[np.ndarray], float, float, bool]]:
        rng = np.random.default_rng(7)
        w1 = self.parameter((rng.standard_normal((6, 5)) * 0.5).astype(np.float32))
        w2 = self.parameter((rng.standard_normal((5, 3)) * 0.5).astype(np.float32))
        xs = [self.tensor(rng.standard_normal((8, 6)).astype(np.float32)) for _ in range(2)]
        ys = [self.tensor(rng.standard_normal((8, 3)).astype(np.float32)) for _ in range(2)]
        cls, kwargs = _OPTIMIZERS[opt]
        optimizer = getattr(self.fw.optim, cls)([w1, w2], **kwargs)
        scaler = self.scaler(init_scale=2.0**8, growth_interval=3)

        log = []
        for i in range(_REF_ITERS):
            x = xs[i % 2] * INF if i in _REF_BAD else xs[i % 2]
            before = [np.array(p.tolist()) for p in (w1, w2)]
            optimizer.zero_grad()
            with self.autocast_ctx():
                out = (x @ w1).tanh() @ w2
            loss = ((out.float() - ys[i % 2]) ** 2).mean()
            scaler.scale(loss).backward()
            norm = math.nan
            if unscale_first:
                scaler.unscale_(optimizer)
                norm = float(self.clip([w1, w2], 0.5))
            scaler.step(optimizer)
            scaler.update()
            after = [np.array(p.tolist()) for p in (w1, w2)]
            moved = any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))
            log.append((after, float(scaler.get_scale()), norm, moved))
        return log


def _lucid_side(device: str, dtype: lucid.dtype | None) -> _Side:
    def autocast_ctx() -> object:
        return autocast(device, dtype=dtype, enabled=dtype is not None)

    return _Side(
        lucid,
        lambda a: lucid.tensor(a, device=device),
        lambda a: nn.Parameter(lucid.tensor(a, device=device)),
        GradScaler,
        autocast_ctx,
        lambda ps, m: clip_grad_norm_(ps, max_norm=m),
    )


def _ref_side(ref: ModuleType, dtype_name: str | None) -> _Side:
    def autocast_ctx() -> object:
        if dtype_name is None:
            return ref.autocast("cpu", enabled=False)
        return ref.autocast("cpu", dtype=getattr(ref, dtype_name))

    return _Side(
        ref,
        lambda a: ref.tensor(a),
        lambda a: ref.nn.Parameter(ref.tensor(a)),
        lambda **kw: ref.amp.GradScaler("cpu", **kw),
        autocast_ctx,
        lambda ps, m: ref.nn.utils.clip_grad_norm_(ps, max_norm=m),
    )


@pytest.mark.parity
@pytest.mark.parametrize("opt", list(_OPTIMIZERS))
@pytest.mark.parametrize("unscale_first", [True, False], ids=["unscale-clip-step", "step"])
@pytest.mark.parametrize("dtype_name", [None, "bfloat16", "float16"], ids=["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("device", _DEVICES)
def test_unscale_clip_step_matches_the_reference(
    device: str, dtype_name: str | None, unscale_first: bool, opt: str, ref: ModuleType
) -> None:
    dtype = None if dtype_name is None else getattr(lucid, dtype_name)
    got = _lucid_side(device, dtype).run(opt, unscale_first)
    try:
        want = _ref_side(ref, dtype_name).run(opt, unscale_first)
    except (RuntimeError, TypeError) as e:  # the reference lacks this CPU autocast dtype
        pytest.skip(f"reference cannot run {dtype_name} CPU autocast here: {e}")

    # fp32 is a pure round-off comparison; half precision rounds the forward
    # differently on each side, so only decisions are exact there.
    rtol, atol = (1e-4, 1e-5) if dtype_name is None else (5e-2, 2e-2)
    for it, ((gp, gs, gn, gm), (wp, ws, wn, wm)) in enumerate(zip(got, want, strict=True)):
        assert gs == ws, f"iter {it}: scale {gs} != reference {ws}"
        assert gm == wm == (it not in _REF_BAD), f"iter {it}: stepped {gm}, reference {wm}"
        if math.isfinite(wn):
            assert gn == pytest.approx(wn, rel=rtol, abs=atol), f"iter {it}: norm {gn} != {wn}"
        else:
            assert not math.isfinite(gn), f"iter {it}: norm {gn}, reference {wn}"
        for a, b in zip(gp, wp, strict=True):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"iter {it}: params")


# ── checkpoint / resume ──────────────────────────────────────────────────────


@pytest.mark.parametrize("model, device, dtype, opt, path", _cells(models=("mlp", "conv")))
def test_checkpoint_resume_is_identical(
    model: str, device: str, dtype: lucid.dtype, opt: str, path: str
) -> None:
    # Three batches in rotation, so the resumed run cannot match by repeating one step.
    a = _Trainer(model, device, dtype, opt, path, n_batches=3)
    for i in range(5):
        a.step(i, bad=(i == 2))
    ck = a.checkpoint()
    assert ck["scaler"]["growth_tracker"] != 0, "checkpoint should land mid growth interval"
    tail_a = [a.step(i) for i in range(5, 12)]

    b = _Trainer(model, device, dtype, opt, path, n_batches=3, init_seed=SEED + 99)
    b.restore(ck)
    assert b.scaler.state_dict() == ck["scaler"]
    tail_b = [b.step(i) for i in range(5, 12)]

    np.testing.assert_allclose(tail_a, tail_b, rtol=1e-6, atol=1e-7, err_msg="loss tail")
    assert b.scaler.get_scale() == a.scaler.get_scale()
    assert b.scaler.state_dict() == a.scaler.state_dict()
    for pa, pb in zip(a.params(), b.params(), strict=True):
        np.testing.assert_allclose(pa, pb, rtol=1e-6, atol=1e-7, err_msg="params")
    for sa, sb in zip(a.opt_state(), b.opt_state(), strict=True):
        np.testing.assert_allclose(sa, sb, rtol=1e-6, atol=1e-7, err_msg="optimizer state")


# ── no leak across steps ─────────────────────────────────────────────────────


@pytest.mark.parametrize("model, device, dtype, opt, path", _cells(models=("mlp", "conv")))
def test_repeated_steps_do_not_grow_live_memory(
    model: str, device: str, dtype: lucid.dtype, opt: str, path: str
) -> None:
    t = _Trainer(model, device, dtype, opt, path, growth_interval=5)
    for i in range(6):  # optimizer state, grad buffers and executables allocate lazily
        t.step(i, bad=(i == 3))
    gc.collect()
    steady = live_bytes(device)

    for i in range(40):
        t.step(i, bad=(i % 13 == 7))
    gc.collect()
    grown = live_bytes(device) - steady

    assert grown <= 0, f"{grown} bytes of engine memory accumulated over 40 steps"
