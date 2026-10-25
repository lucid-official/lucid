"""Graphs eager answers and the compile path used to refuse, fall back on, or die in.

Found by compiling each op over float32 / int64 / bool and replaying on a
second input.  None of these was a wrong answer; each was an error, an
eager fallback, or an abort where eager simply works:

* the session executable cache keyed on op outputs only, so ``isnan(x)``
  traced on float32 and then on int64 — same op outputs — reused the
  float executable and raised "feed slot 0 expects dtype float32";
* ``isnan`` / ``isinf`` / ``isfinite`` / ``square`` handed an integer
  tensor to an MPSGraph primitive that takes floating point only, which
  surfaced as "Caught an unknown exception!" — and an int64 ``x * x``
  still does on MPSGraph's side, so a first run that fails now falls back;
* ``triu`` / ``tril`` declined every ``k != 0`` — ``triu(ones, 1)`` is the
  usual way to spell a causal mask;
* sorting a bool tensor on Metal died in eager already: MLX has no bool
  sort kernel;
* ``tensordot`` under ``dynamic=True`` flattened a symbolic batch axis
  into a negative reshape size.
"""

import numpy as np
import pytest

import lucid

from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


def _x(dtype: object, seed: int = 0, shape: tuple[int, ...] = (4, 6)) -> lucid.Tensor:
    lucid.manual_seed(seed)
    if dtype is lucid.bool_:
        return (lucid.randint(0, 2, shape) == 1).to(COMPILE_DEVICE)
    return lucid.randint(-9, 9, shape).to(dtype).to(COMPILE_DEVICE)  # type: ignore[arg-type]


def _compiled_matches(fn: object, *args: lucid.Tensor) -> None:
    compiled = lucid.compile(fn)  # type: ignore[arg-type]
    got = compiled(*args).to("cpu").numpy()
    want = fn(*(a.to("cpu") for a in args)).numpy()  # type: ignore[operator]
    info = compiled.cache_info()
    assert info["entries"] == 1 and len(info["eager_only"]) == 0, info
    assert got.dtype == want.dtype
    assert np.array_equal(got, want), f"got {got.tolist()}, want {want.tolist()}"


def test_cache_does_not_share_an_executable_across_feed_dtypes() -> None:
    _C_engine.compile.session_cache_clear()
    fn = lambda t: lucid.isnan(t).to(lucid.int64) + 1  # noqa: E731
    _compiled_matches(fn, _x(lucid.float32))
    _compiled_matches(fn, _x(lucid.int64))


def test_cache_does_not_share_an_executable_across_feed_shapes() -> None:
    """``sum`` of (4, 6) and of (6, 4) have the same op outputs."""
    _C_engine.compile.session_cache_clear()
    fn = lambda t: t.sum() + 1  # noqa: E731
    _compiled_matches(fn, _x(lucid.float32, shape=(4, 6)))
    _compiled_matches(fn, _x(lucid.float32, shape=(6, 4)))


@pytest.mark.parametrize("op", ["isnan", "isinf", "isfinite"])
@pytest.mark.parametrize("dtype", [lucid.int64, lucid.int32, lucid.bool_])
def test_float_predicates_on_non_float_inputs(op: str, dtype: object) -> None:
    fn = lambda t: getattr(lucid, op)(t)  # noqa: E731
    _compiled_matches(fn, _x(dtype))


@pytest.mark.parametrize("dtype", [lucid.int64, lucid.int32, lucid.float32])
def test_square_on_integers(dtype: object) -> None:
    _compiled_matches(lambda t: lucid.square(t), _x(dtype))


@pytest.mark.parametrize("spelling", ["square", "self_mul", "deduplicated"])
def test_int64_square_compiles_without_falling_back(spelling: str) -> None:
    """MPSGraph has no ``square_i64`` kernel, so 64-bit self-products shield
    their operands with ``maximum(a, a) * minimum(b, b)`` to stay on ``mul_i64``.

    Every spelling below — ``square(t)``, ``t * t``, and ``abs(t) * abs(t)`` —
    must compile and execute on the GPU with zero eager fallback.
    """
    fns = {
        "square": lambda t: lucid.square(t),
        "self_mul": lambda t: t * t,
        "deduplicated": lambda t: lucid.abs(t) * lucid.abs(t),
    }
    fn = fns[spelling]
    x = _x(lucid.int64)
    compiled = lucid.compile(fn)
    for _ in range(2):
        got = compiled(x).to("cpu").numpy()
        assert np.array_equal(got, fn(x.to("cpu")).numpy())
    info = compiled.cache_info()
    assert info["entries"] == 1 and len(info["eager_only"]) == 0, info


@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("op", ["triu", "tril"])
@pytest.mark.parametrize("shape", [(5, 5), (4, 6), (2, 3, 4)])
def test_triangular_at_any_offset(op: str, k: int, shape: tuple[int, ...]) -> None:
    fn = lambda t: getattr(lucid, op)(t, k) + 1  # noqa: E731
    _compiled_matches(fn, _x(lucid.float32, shape=shape))


def test_causal_mask_spelled_with_triu() -> None:
    def fn(t: lucid.Tensor) -> lucid.Tensor:
        return t.masked_fill(lucid.triu(lucid.ones_like(t), 1) > 0, 0.0)

    _compiled_matches(fn, _x(lucid.float32, shape=(5, 5)))


def test_bool_sort_on_metal() -> None:
    x = _x(lucid.bool_)
    got = lucid.sort(x, dim=-1).values.to("cpu").numpy()
    assert np.array_equal(got, np.sort(x.to("cpu").numpy(), axis=-1))
    got_desc = lucid.topk(x.to(lucid.int32), 2, dim=-1)[0].to("cpu").numpy()
    assert np.array_equal(
        got_desc, -np.sort(-x.to("cpu").numpy().astype(np.int32))[:, :2]
    )


def test_tensordot_under_symbolic_batch() -> None:
    """The batch axis is one of the kept axes; the result must follow the batch."""
    lucid.manual_seed(0)
    w = lucid.randn(6, 3).to(COMPILE_DEVICE)
    fn = lambda t: lucid.tensordot(t, w, dims=1) + 1  # noqa: E731
    compiled = lucid.compile(fn, dynamic=True)
    for batch in (4, 7):
        x = lucid.randn(batch, 5, 6).to(COMPILE_DEVICE)
        got = compiled(x).to("cpu").numpy()
        want = fn(x.to("cpu").to(COMPILE_DEVICE)).to("cpu").numpy()
        assert got.shape == (batch, 5, 3)
        assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
