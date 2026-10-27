"""Ops that skip ``wire_autograd`` must record their trace I/O by hand.

``wire_autograd`` is also what records a traced op's operands, so an op —
or one dtype branch of an op — that has no derivative and skips it arrives
in the trace with no inputs.  What ``lucid.compile`` does next depends on
the op's shape:

* single output (``sort`` on integers, ``nextafter``, ``inner`` without
  grad) — the builder refuses the graph and the model runs eager, correct
  and silently uncompiled;
* ``topk`` on integers — the follow-up registration of the indices becomes
  the op's *first* output, the values are never traced, and the tensor that
  reads them becomes an external feed.  The graph compiles, and every
  replay returns the values of the call that was traced.

Each op's emitter had never run for the same reason, and two were wrong:
``nextafter`` stepped by a fixed 1.19e-7 instead of one ULP, and ``inner``
summed an element-wise product, which is ``inner`` only for two vectors.
The cases below are the ones that tell the old emitters from the new.
"""

import numpy as np
import pytest

import lucid

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


def _ints(seed: int) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randint(-9, 9, (4, 6)).to(lucid.int64).to(COMPILE_DEVICE)


def _captured(compiled: object) -> None:
    info = compiled.cache_info()  # type: ignore[attr-defined]
    assert info["entries"] == 1 and len(info["eager_only"]) == 0, (
        "the op did not reach the graph — most likely it is not recording "
        f"its trace I/O again, which routes the whole model to eager: {info}"
    )


def _replays(fn: object, x1: lucid.Tensor, x2: lucid.Tensor) -> None:
    """Trace on ``x1``, replay on ``x2``: a frozen feed answers for ``x1``."""
    compiled = lucid.compile(fn)  # type: ignore[arg-type]
    compiled(x1)
    got = compiled(x2).numpy()
    want = fn(x2).numpy()  # type: ignore[operator]
    _captured(compiled)
    assert np.array_equal(got, want), (
        f"compiled replay does not follow its input: got {got.tolist()}, " f"want {want.tolist()}"
    )


def test_integer_topk_values_follow_the_input() -> None:
    """Compiled and answered for the traced call — the case that was wrong."""
    _replays(lambda x: lucid.topk(x, 2, dim=-1)[0] + 1, _ints(1), _ints(2))


def test_integer_topk_indices_follow_the_input() -> None:
    """These rows have ties: the compiled topk names them lowest first, as
    eager does (LCD-425)."""
    _replays(lambda x: lucid.topk(x, 2, dim=-1)[1] + 1, _ints(1), _ints(2))


def test_integer_sort_is_captured() -> None:
    _replays(lambda x: lucid.sort(x, dim=-1)[0] + 1, _ints(1), _ints(2))


def test_nextafter_steps_one_ulp() -> None:
    """Magnitudes where a fixed epsilon is too small, too large, or zero."""
    pairs = [
        (100.0, 200.0),
        (-100.0, 200.0),
        (100.0, -200.0),
        (1e-10, 1.0),
        (3.0e38, float("inf")),
        (float("inf"), 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (-0.0, 1.0),
        (2.5, 2.5),
        (float("nan"), 1.0),
        (1.0, float("nan")),
    ]
    a = lucid.tensor([p[0] for p in pairs]).to(COMPILE_DEVICE)
    b = lucid.tensor([p[1] for p in pairs]).to(COMPILE_DEVICE)
    want = np.nextafter(
        np.array([p[0] for p in pairs], dtype=np.float32),
        np.array([p[1] for p in pairs], dtype=np.float32),
    )

    # No ``* 1`` after the op: compiled ``x * 1`` flushes a subnormal to zero
    # as eager Metal does (LCD-452), and the steps off zero are subnormal.
    fn = lambda x, y: lucid.nextafter(x, y)  # noqa: E731
    compiled = lucid.compile(fn)
    got = compiled(a, b).numpy()
    _captured(compiled)
    assert np.array_equal(
        got.view(np.int32), want.view(np.int32)
    ), f"compiled nextafter is not one ULP: got {got.tolist()}, want {want.tolist()}"
    eager = fn(a, b).numpy()
    assert np.array_equal(got.view(np.int32), eager.view(np.int32))


@pytest.mark.parametrize(
    ("sa", "sb"), [((6,), (6,)), ((4, 6), (4, 6)), ((4, 6), (3, 6)), ((2, 3, 6), (5, 6))]
)
def test_inner_contracts_the_last_axis(sa: tuple[int, ...], sb: tuple[int, ...]) -> None:
    """[M, K] · [M, K] is [M, M]; the old emitter gave [M]."""
    lucid.manual_seed(0)
    a = lucid.randn(*sa).to(COMPILE_DEVICE)
    b = lucid.randn(*sb).to(COMPILE_DEVICE)
    fn = lambda x, y: lucid.linalg.inner(x, y) * 1  # noqa: E731

    compiled = lucid.compile(fn)
    got = compiled(a, b).numpy()
    want = np.inner(a.numpy(), b.numpy())
    _captured(compiled)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_tensordot_binds_a_non_scalar_result() -> None:
    """Only the full contraction used to be bound; the consumer declined."""
    lucid.manual_seed(0)
    a = lucid.randn(4, 6).to(COMPILE_DEVICE)
    b = lucid.randn(6, 3).to(COMPILE_DEVICE)
    fn = lambda x, y: lucid.tensordot(x, y, dims=1) + 1  # noqa: E731

    compiled = lucid.compile(fn)
    got = compiled(a, b).numpy()
    _captured(compiled)
    assert np.allclose(got, a.numpy() @ b.numpy() + 1, rtol=1e-5, atol=1e-5)


def test_integer_matrix_power_declines_instead_of_aborting() -> None:
    """MPSGraph aborts the process on an integer matmul; the emitter must decline."""
    x = _ints(1)[:, :4]
    fn = lambda m: lucid.linalg.matrix_power(m, 2) + 1  # noqa: E731

    compiled = lucid.compile(fn)
    got = compiled(x).numpy()
    assert np.array_equal(got, fn(x).numpy())
    assert len(compiled.cache_info()["eager_only"]) == 1
