"""layer_norm / rms_norm / 3-D pooling / group_norm take the rank-canonical policy (LCD-927).

MPSGraph runs a kernel that reads or writes a tensor of rank 5+ as a slice loop over the dims
above the last four, and the ``RankCanon`` pass cannot see a tensor an emitter builds inside
itself.  Three emitters built or took rank 5+ tensors:

* ``layer_norm`` / ``rms_norm`` normalised the trailing axes of a rank-5+ input in place
  (``(4096, 4, 8, 16, 16)``: 93 ms against 12 ms at rank 2, backward 169 against 17);
* ``max_pool3d`` / ``avg_pool3d`` lifted the volume to rank 6 for the SDK's 4-D pool
  (``avg_pool3d`` k=(1,2,2) on ``(64, 16, 8, 32, 32)``: 8.2 ms against 1.5, the max pool's
  backward 168 ms against 9);
* ``group_norm`` ignored ``rank_canon_wants``, so ``LUCID_COMPILE_RANK_CANON_MIN_LEAD=-1``
  could not turn its rank-2 form off.

Each now works at rank <= 4 when ``rank_canon_wants`` says the lead is worth a reshape, and
keeps its native form otherwise.  The graph cannot be inspected from Python, so the guards are

* value tables (forward, loss and every gradient of a compiled step against eager) with the
  policy forced on (``MIN_LEAD=0``), at its default, and off (``-1``), so both forms are
  checked wherever a shape could land;
* NaN / inf rows;
* timing, against the same elements viewed at rank 2 / 4 inside the graph (what the pass
  itself would make) and against the policy off.  The executable cache key ignores the
  environment (LCD-922), so the cache is cleared around every change of it.
"""

import time
from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile import make_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, to_metal

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]

_ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"


def _set_policy(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, value)


@pytest.fixture(params=["0", None, "-1"], ids=["canon-always", "canon-default", "canon-off"])
def policy(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str | None]:
    """The rank-canon threshold: always on, the default (64), and off."""
    _set_policy(monkeypatch, request.param)
    yield request.param


@pytest.fixture
def canon_always(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _set_policy(monkeypatch, "0")
    yield


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _input(
    shape: tuple[int, ...], dtype: lucid.dtype, *, poison: float | None = None
) -> lucid.Tensor:
    n = _numel(shape)
    x = (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 1.3) * 2.0 + 0.4).reshape(*shape)
    if poison is not None:
        flat = x.reshape(-1)
        flat[n // 3] = poison
        x = flat.reshape(*shape)
    return x.to(dtype).to(COMPILE_DEVICE)


def _weights(n: int, dtype: lucid.dtype) -> lucid.Tensor:
    return lucid.linspace(0.5, 1.5, n, device=COMPILE_DEVICE, dtype=dtype)


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    """Squared, so half-precision rows do not cancel to a loss of rounding noise."""
    return (y * y * _weights(y.numel(), y.dtype).reshape(*y.shape)).sum()


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str, tol: float) -> None:
    """Equal to ``tol`` relative to the largest finite value of ``want``; NaN / inf cells exact."""
    g, w = got.float().reshape(-1).tolist(), want.float().reshape(-1).tolist()
    assert len(g) == len(w), why
    scale = max([abs(b) for b in w if b == b and abs(b) != float("inf")] + [1.0])
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        if b != b:  # NaN must stay NaN
            assert a != a, f"{why}[{i}]: {a} != nan"
        elif b in (float("inf"), float("-inf")):
            assert a == b, f"{why}[{i}]: {a} != {b}"
        else:
            assert abs(a - b) <= tol * scale, f"{why}[{i}]: {a} != {b} (scale {scale})"


def _grads(m: nn.Module) -> dict[str, lucid.Tensor]:
    out = {}
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        out[name] = p.grad
    return out


def _check_module(
    make: Callable[[], nn.Module], x: lucid.Tensor, tol: float, *, forward: bool = True
) -> None:
    """A compiled step (loss + every gradient) and forward against an eager twin of ``make()``."""
    eager = make()
    e_loss = _loss(eager(x))
    e_loss.backward()
    comp = make()
    c_loss = make_step(comp, _loss)(x)
    c_loss.backward()
    _same(c_loss, e_loss, "loss", tol)
    e_g, c_g = _grads(eager), _grads(comp)
    assert e_g.keys() == c_g.keys()
    for name in e_g:
        _same(c_g[name], e_g[name], f"grad {name}", tol)
    if forward:
        with lucid.no_grad():
            _same(lucid.compile(comp.eval())(x), eager.eval()(x), "forward", tol)


# ── layer_norm / rms_norm ────────────────────────────────────────────────────────────

#: ``(input shape, normalized axes)``: rank 5 and 6, one to three axes.  Dim 0 is 64 on the
#: rank-5 rows, so the default policy (lead >= 64) takes the canonical form there and the
#: rank-6 rows (lead 6) keep the native one; ``MIN_LEAD=0`` forces it everywhere.
_NORM_CASES: list[tuple[tuple[int, ...], int]] = [
    ((64, 2, 2, 2, 6), 1),
    ((64, 2, 2, 2, 6), 2),
    ((64, 2, 2, 2, 6), 3),
    ((2, 3, 2, 2, 2, 5), 1),
    ((2, 3, 2, 2, 2, 5), 2),
    ((2, 2, 2, 2, 5), 1),
    ((2, 2, 2, 2, 5), 5),
]


class _NormNet(nn.Module):
    """``norm(x * w)`` with a full-shape ``w`` so the gradient reaches the input."""

    def __init__(
        self, kind: str, shape: tuple[int, ...], k: int, affine: bool, dtype: lucid.dtype
    ) -> None:
        super().__init__()
        self.kind = kind
        self.norm_shape = shape[len(shape) - k :]
        self.w = nn.Parameter(lucid.linspace(0.6, 1.4, _numel(shape)).reshape(*shape))
        n = _numel(self.norm_shape)
        self.gamma = (
            nn.Parameter(lucid.linspace(0.5, 1.5, n).reshape(*self.norm_shape)) if affine else None
        )
        self.beta = (
            nn.Parameter(lucid.linspace(-0.3, 0.3, n).reshape(*self.norm_shape))
            if affine and kind == "layer_norm"
            else None
        )
        to_metal(self)
        self.to(dtype)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = x * self.w
        if self.kind == "layer_norm":
            return F.layer_norm(y, self.norm_shape, self.gamma, self.beta)
        return F.rms_norm(y, self.norm_shape, self.gamma)


def _norm_check(
    kind: str,
    case: tuple[tuple[int, ...], int],
    affine: bool,
    dtype: lucid.dtype,
    tol: float,
    *,
    poison: float | None = None,
) -> None:
    shape, k = case
    x = _input(shape, dtype, poison=poison)
    _check_module(lambda: _NormNet(kind, shape, k, affine, dtype), x, tol)


def _case_id(case: tuple[tuple[int, ...], int]) -> str:
    shape, k = case
    return "x".join(map(str, shape)) + f"-k{k}"


@pytest.mark.usefixtures("policy")
@pytest.mark.parametrize("affine", [True, False], ids=["affine", "plain"])
@pytest.mark.parametrize("case", _NORM_CASES, ids=_case_id)
@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
def test_a_compiled_norm_matches_eager(
    kind: str, case: tuple[tuple[int, ...], int], affine: bool
) -> None:
    _norm_check(kind, case, affine, lucid.float32, 5e-4)


@pytest.mark.usefixtures("canon_always")
@pytest.mark.parametrize(
    "dtype,tol", [(lucid.float16, 4e-2), (lucid.bfloat16, 1e-1)], ids=["float16", "bfloat16"]
)
@pytest.mark.parametrize("case", [_NORM_CASES[0], _NORM_CASES[1], _NORM_CASES[3]], ids=_case_id)
@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
def test_a_half_precision_norm_matches_eager(
    kind: str, case: tuple[tuple[int, ...], int], dtype: lucid.dtype, tol: float
) -> None:
    _norm_check(kind, case, True, dtype, tol)


@pytest.mark.usefixtures("canon_always")
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("case", [_NORM_CASES[0], _NORM_CASES[1], _NORM_CASES[4]], ids=_case_id)
@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
def test_a_poisoned_row_stays_poisoned_like_eager(
    kind: str, case: tuple[tuple[int, ...], int], poison: float
) -> None:
    _norm_check(kind, case, True, lucid.float32, 5e-4, poison=poison)


# ── 3-D pooling ──────────────────────────────────────────────────────────────────────

#: ``(label, builder, input shape)``.  Kernel / stride / padding / ceil-mode combinations the
#: emitter lowers, plus a dilated window, which composes other ops instead of the pool kernel.
_POOL_CASES: list[tuple[str, Callable[[lucid.Tensor], lucid.Tensor], tuple[int, ...]]] = [
    ("avg-k122", lambda x: F.avg_pool3d(x, (1, 2, 2)), (2, 3, 4, 6, 6)),
    ("avg-k222", lambda x: F.avg_pool3d(x, 2), (2, 3, 4, 6, 6)),
    ("avg-k333-s2-p1", lambda x: F.avg_pool3d(x, 3, stride=2, padding=1), (2, 3, 5, 7, 7)),
    (
        "avg-nopad-count",
        lambda x: F.avg_pool3d(x, 3, stride=2, padding=1, count_include_pad=False),
        (2, 3, 5, 7, 7),
    ),
    ("avg-overlap", lambda x: F.avg_pool3d(x, 2, stride=1), (2, 2, 4, 5, 5)),
    ("avg-ceil", lambda x: F.avg_pool3d(x, 2, stride=2, ceil_mode=True), (2, 2, 5, 5, 7)),
    ("max-k122", lambda x: F.max_pool3d(x, (1, 2, 2)), (2, 3, 4, 6, 6)),
    ("max-k222", lambda x: F.max_pool3d(x, 2), (2, 3, 4, 6, 6)),
    ("max-k333-s2-p1", lambda x: F.max_pool3d(x, 3, stride=2, padding=1), (2, 3, 5, 7, 7)),
    ("max-overlap", lambda x: F.max_pool3d(x, 2, stride=1), (2, 2, 4, 5, 5)),
    ("max-ceil", lambda x: F.max_pool3d(x, 2, stride=2, ceil_mode=True), (2, 2, 5, 5, 7)),
    ("max-asym", lambda x: F.max_pool3d(x, (2, 3, 1), stride=(1, 2, 1)), (2, 2, 4, 7, 5)),
    ("max-dilated", lambda x: F.max_pool3d(x, 2, dilation=2), (2, 2, 5, 6, 6)),
]


class _PoolNet(nn.Module):
    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor], shape: tuple[int, ...]) -> None:
        super().__init__()
        self.fn = fn
        self.w = nn.Parameter(lucid.linspace(0.6, 1.4, _numel(shape)).reshape(*shape))
        to_metal(self)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x * self.w)


def _pool_ids(case: tuple[str, object, tuple[int, ...]]) -> str:
    return case[0]


@pytest.mark.usefixtures("policy")
@pytest.mark.parametrize("case", _POOL_CASES, ids=_pool_ids)
def test_a_compiled_pool3d_matches_eager(
    case: tuple[str, Callable[[lucid.Tensor], lucid.Tensor], tuple[int, ...]],
) -> None:
    _, fn, shape = case
    # N * C = 4..6 is under the default threshold, so ``canon-default`` keeps the rank-6 form
    # and ``canon-always`` the merged one.
    _check_module(lambda: _PoolNet(fn, shape), _input(shape, lucid.float32), 5e-4)


@pytest.mark.usefixtures("canon_always")
@pytest.mark.parametrize("dtype,tol", [(lucid.float16, 4e-2), (lucid.bfloat16, 1e-1)])
@pytest.mark.parametrize(
    "case", [_POOL_CASES[0], _POOL_CASES[2], _POOL_CASES[6], _POOL_CASES[8]], ids=_pool_ids
)
def test_a_half_precision_pool3d_matches_eager(
    case: tuple[str, Callable[[lucid.Tensor], lucid.Tensor], tuple[int, ...]],
    dtype: lucid.dtype,
    tol: float,
) -> None:
    _, fn, shape = case
    _check_module(lambda: _PoolNet(fn, shape).to(dtype), _input(shape, dtype), tol)


@pytest.mark.usefixtures("canon_always")
@pytest.mark.parametrize("case", [_POOL_CASES[6], _POOL_CASES[7], _POOL_CASES[8]], ids=_pool_ids)
def test_a_nan_window_pools_to_nan_like_eager(
    case: tuple[str, Callable[[lucid.Tensor], lucid.Tensor], tuple[int, ...]],
) -> None:
    """A window holding a NaN answers NaN and its gradient lands where eager puts it."""
    _, fn, shape = case
    _check_module(
        lambda: _PoolNet(fn, shape), _input(shape, lucid.float32, poison=float("nan")), 5e-4
    )


@pytest.mark.usefixtures("canon_always")
def test_an_all_neg_inf_window_stays_neg_inf() -> None:
    shape = (2, 2, 4, 4, 4)
    x = _input(shape, lucid.float32)
    flat = x.reshape(-1)
    for i in range(0, 16):  # the first (2, 2, 2) window's worth, along depth 0 and 1
        flat[i] = float("-inf")
    x = flat.reshape(*shape)

    def fwd(t: lucid.Tensor) -> lucid.Tensor:
        return F.max_pool3d(t, 2)

    with lucid.no_grad():
        _same(lucid.compile(fwd)(x), fwd(x), "forward", 0.0)


# ── group_norm obeys the same switch ─────────────────────────────────────────────────


class _GroupNet(nn.Module):
    def __init__(self, shape: tuple[int, ...], groups: int) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.linspace(0.6, 1.4, _numel(shape)).reshape(*shape))
        self.gn = nn.GroupNorm(groups, shape[1])
        assert self.gn.weight is not None and self.gn.bias is not None
        self.gn.weight.data = lucid.linspace(0.5, 1.5, shape[1])
        self.gn.bias.data = lucid.linspace(-0.3, 0.3, shape[1])
        to_metal(self)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.gn(x * self.w)


@pytest.mark.usefixtures("policy")
@pytest.mark.parametrize("shape,groups", [((64, 8, 3, 2), 2), ((4, 8, 3, 2, 3), 4)])
def test_group_norm_matches_eager_under_every_policy(shape: tuple[int, ...], groups: int) -> None:
    """``-1`` is the legacy grouped layout; ``0`` the rank-2 one; both are the same function."""
    _check_module(lambda: _GroupNet(shape, groups), _input(shape, lucid.float32), 5e-4)


def test_group_norm_layout_follows_the_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy grouped layout and the rank-2 one agree to rounding, whichever is chosen."""
    shape, groups = (64, 8, 3, 2), 2
    x = _input(shape, lucid.float32)
    outs = []
    for value in ("-1", "0"):
        _set_policy(monkeypatch, value)
        m = _GroupNet(shape, groups).eval()
        with lucid.no_grad():
            outs.append(lucid.compile(m)(x))
    _same(outs[0], outs[1], "legacy vs canonical layout", 5e-4)


# ── timing ───────────────────────────────────────────────────────────────────────────


def _best_ms(
    calls: dict[str, Callable[[], lucid.Tensor]], rounds: int = 6, reps: int = 5
) -> dict[str, float]:
    """Min over interleaved rounds per variant: a load spike hits every variant alike."""
    for call in calls.values():
        for _ in range(3):
            float(call().sum().item())
    best = dict.fromkeys(calls, float("inf"))
    for _ in range(rounds):
        for name, call in calls.items():
            for _ in range(reps):
                t0 = time.perf_counter()
                float(call().sum().item())
                best[name] = min(best[name], time.perf_counter() - t0)
    return {k: v * 1e3 for k, v in best.items()}


def _fwd_call(
    fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor
) -> Callable[[], lucid.Tensor]:
    cm = lucid.compile(fn)

    def call() -> lucid.Tensor:
        with lucid.no_grad():
            return cm(x)

    return call


def _step_call(
    fn: Callable[[lucid.Tensor], lucid.Tensor], shape: tuple[int, ...], x: lucid.Tensor
) -> Callable[[], lucid.Tensor]:
    step = make_step(_PoolNet(fn, shape), lambda y: (y * y).sum())

    def call() -> lucid.Tensor:
        loss = step(x)
        loss.backward()
        return loss

    return call


def _ratio_within(
    build: Callable[[], dict[str, Callable[[], lucid.Tensor]]],
    num: str,
    den: str,
    bound: float,
    attempts: int = 3,
) -> None:
    """``num / den <= bound`` on any of ``attempts`` fresh measurements (a loaded machine
    can spoil one)."""
    seen = []
    for _ in range(attempts):
        _C_engine.compile.session_cache_clear()
        t = _best_ms(build())
        seen.append(t)
        if t[num] <= bound * t[den]:
            return
    pytest.fail(f"{num} / {den} > {bound} on every attempt: {seen}")


def _hand_view_norm(kind: str, k: int) -> Callable[[lucid.Tensor], lucid.Tensor]:
    def fn(x: lucid.Tensor) -> lucid.Tensor:
        flat = x.reshape(-1, k)
        y = F.layer_norm(flat, (k,)) if kind == "layer_norm" else F.rms_norm(flat, (k,))
        return y.reshape(*x.shape)

    return fn


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("k", [96, 16])
def test_a_rank_5_norm_is_not_slower_than_the_same_elements_at_rank_2(kind: str, k: int) -> None:
    """Producer -> norm -> consumer on a rank-5 value, against the rank-2 view of it (the
    layout the emitter now builds).  The native form ran 2-19x that."""
    shape = (64, 8, 16, 16, k) if k == 96 else (1024, 4, 8, 16, k)
    x = _input(shape, lucid.float32)

    def native(t: lucid.Tensor) -> lucid.Tensor:
        n = F.layer_norm(t * 1.5, (k,)) if kind == "layer_norm" else F.rms_norm(t * 1.5, (k,))
        return n * 2.0

    twin = _hand_view_norm(kind, k)

    def view(t: lucid.Tensor) -> lucid.Tensor:
        return twin(t * 1.5) * 2.0

    _ratio_within(
        lambda: {"native": _fwd_call(native, x), "view": _fwd_call(view, x)}, "native", "view", 1.3
    )


@pytest.mark.parametrize("kind", ["layer_norm", "rms_norm"])
def test_a_rank_5_norm_step_is_not_slower_than_the_same_elements_at_rank_2(kind: str) -> None:
    shape = (1024, 4, 8, 16, 16)
    x = _input(shape, lucid.float32)
    norm = (
        (lambda t: F.layer_norm(t, (16,)))
        if kind == "layer_norm"
        else (lambda t: F.rms_norm(t, (16,)))
    )
    twin = _hand_view_norm(kind, 16)
    _ratio_within(
        lambda: {"native": _step_call(norm, shape, x), "view": _step_call(twin, shape, x)},
        "native",
        "view",
        1.3,
    )


def test_the_canonical_norm_beats_the_native_rank_5_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same graph with the policy off (the old rank-5 spelling) against on."""
    shape = (1024, 4, 8, 16, 16)
    x = _input(shape, lucid.float32)

    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return F.layer_norm(t * 1.5, (16,)) * 2.0

    for attempt in range(3):
        times = {}
        for label, value in (("off", "-1"), ("on", None)):
            _set_policy(monkeypatch, value)
            times[label] = _best_ms({label: _fwd_call(fn, x)})[label]
        if times["on"] <= 0.6 * times["off"]:
            break
    else:
        pytest.fail(f"canonical layer_norm is not at least 1.7x the native form: {times}")


def test_a_rank_5_avg_pool3d_is_not_slower_than_its_rank_4_view() -> None:
    """A (1, 2, 2) window is a 2-D pool of ``(N*C, D, H, W)``; the rank-6 lift ran 5x that."""
    shape = (64, 16, 8, 32, 32)
    x = _input(shape, lucid.float32)

    def pool(t: lucid.Tensor) -> lucid.Tensor:
        return F.avg_pool3d(t * 1.5, (1, 2, 2)) * 2.0

    def view(t: lucid.Tensor) -> lucid.Tensor:
        y = F.avg_pool2d((t * 1.5).reshape(-1, *t.shape[2:]), 2)
        return y.reshape(t.shape[0], t.shape[1], *y.shape[1:]) * 2.0

    _ratio_within(
        lambda: {"pool3d": _fwd_call(pool, x), "view": _fwd_call(view, x)}, "pool3d", "view", 1.5
    )


def test_a_rank_5_max_pool3d_step_is_not_slower_than_its_rank_4_view() -> None:
    """The max pool's backward slices its source per window offset: at rank 6 it ran 18x."""
    shape = (64, 16, 8, 32, 32)
    x = _input(shape, lucid.float32)

    def pool(t: lucid.Tensor) -> lucid.Tensor:
        return F.max_pool3d(t, (1, 2, 2))

    def view(t: lucid.Tensor) -> lucid.Tensor:
        y = F.max_pool2d(t.reshape(-1, *t.shape[2:]), 2)
        return y.reshape(t.shape[0], t.shape[1], *y.shape[1:])

    _ratio_within(
        lambda: {"pool3d": _step_call(pool, shape, x), "view": _step_call(view, shape, x)},
        "pool3d",
        "view",
        1.5,
    )
