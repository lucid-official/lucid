"""Every norm of a complex input reduces through ``|x|`` and is real (CHA-231).

**Defect class.**  A norm is a function of the magnitudes of the entries
(or of the singular values, which are real).  Squaring, summing or comparing
the complex entries themselves gives a number that is not a norm.
``vector_norm`` stopped doing that in CHA-147, but ``matrix_norm(·, "fro")``
kept its own copy of the step, ``sum(x · x)``.  So did ``norm(x, "fro")`` and
``norm(x, dim=(0, 1))``, which dispatch to it.  For one input that gave
``(3.37-0.59j)`` on Metal where the norm is a real 4.123, and an error on the
CPU.

**Owner.**  ``vector_norm`` is the one place a norm takes ``|x|``.
``matrix_norm`` reduces its entry-wise orders (``"fro"``, ±1, ±inf) through
it, ``norm`` dispatches to the two of them, and ``cond`` reaches the
entry-wise orders through ``matrix_norm``.

**Guard.**  The test sweeps every public norm in linalg ({``vector_norm``,
``matrix_norm``, ``norm``, ``cond``}) × every order × {real, complex} ×
every device.  Each case either returns a real dtype equal to the
reference's, or is a pinned, typed refusal.  Those refusals are the orders
that need a kernel linalg has only for real input: the singular values
(``"nuc"``, ±2) and, for ``cond``, the inverse.  A refused case that starts
to work fails ``test_complex_norm_without_a_kernel_is_refused``.  That is
the cue to move the case into the supported set, not to delete the check.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA

_INF = float("inf")

_VECTOR_ORDS: tuple[float | None, ...] = (None, 0, 1, 2, 3, 0.5, _INF, -_INF, -1, -2)
_MATRIX_ORDS: tuple[int | float | str | None, ...] = (
    None,
    "fro",
    "nuc",
    1,
    -1,
    2,
    -2,
    _INF,
    -_INF,
)
#: Matrix orders that are read off the singular values.
_SPECTRAL: tuple[int | str, ...] = ("nuc", 2, -2)


def _call_kwargs(ord: object, **rest: Any) -> dict[str, Any]:
    """``ord`` as a keyword, left out when it is ``None`` (each function's
    default) so the reference and Lucid both take their own default."""
    return rest if ord is None else {"ord": ord, **rest}


_Case = tuple[str, tuple[int, ...], dict[str, Any], str | None]


def _cases() -> list[_Case]:
    """``(fn, shape, kwargs, refused)`` for every norm call the sweep makes.

    ``refused`` is the reason a *complex* input is turned away (a regex for
    the message), or ``None`` when a complex input must work.  Real input
    must work in every case.
    """
    cases: list[_Case] = []

    def add(fn: str, shape: tuple[int, ...], kw: dict[str, Any], refused: str | None) -> None:
        cases.append((fn, shape, kw, refused))

    for o in _VECTOR_ORDS:
        if o is None:
            continue
        add("vector_norm", (3, 4), {"ord": o, "dim": -1}, None)
        add("vector_norm", (2, 3, 4), {"ord": o}, None)
        add("vector_norm", (2, 3, 4), {"ord": o, "dim": (0, 2), "keepdim": True}, None)

    svd_refusal = r"matrix_norm: ord=.* svd has no complex kernel"
    for o in _MATRIX_ORDS:
        if o is None:
            continue
        refused = svd_refusal if o in _SPECTRAL else None
        add("matrix_norm", (2, 3, 4), {"ord": o}, refused)
        add(
            "matrix_norm",
            (3, 2, 4),
            {"ord": o, "dim": (0, 2), "keepdim": True},
            refused,
        )

    # ``norm`` with a matrix order: without ``dim`` on a matrix, and with an
    # explicit pair of axes on a batch.  ``ord=None`` is the flat 2-norm in
    # the first and the Frobenius norm in the second.
    for o in _MATRIX_ORDS:
        refused = svd_refusal if o in _SPECTRAL else None
        add("norm", (3, 4), _call_kwargs(o), refused)
        add("norm", (2, 3, 4), _call_kwargs(o, dim=(1, 2)), refused)
    # ... and with a vector order: on a vector, and along one axis.
    for o in _VECTOR_ORDS:
        add("norm", (5,), _call_kwargs(o), None)
        add("norm", (3, 4), _call_kwargs(o, dim=1, keepdim=True), None)

    for o in _MATRIX_ORDS:
        needs = "SVD" if o is None or o in _SPECTRAL else "inverse"
        add(
            "cond",
            (2, 3, 3),
            {} if o is None else {"p": o},
            rf"cond\(p=.*\): a complex A needs a complex {needs}",
        )
    return cases


def _id(case: _Case) -> str:
    fn, shape, kw, _ = case
    tag = ",".join(f"{k}={v}" for k, v in kw.items())
    return f"{fn}-{'x'.join(map(str, shape))}-{tag}"


_CASES = _cases()
#: Real input, every case; complex input, the cases that must work.
_PARITY = [
    pytest.param(*c[:3], kind, id=f"{kind}-{_id(c)}")
    for kind in ("real", "complex")
    for c in _CASES
    if kind == "real" or c[3] is None
]
_COMPLEX_SUPPORTED = [pytest.param(*c[:3], id=_id(c)) for c in _CASES if c[3] is None]
_COMPLEX_REFUSED = [pytest.param(*c, id=_id(c)) for c in _CASES if c[3] is not None]


def _data(shape: tuple[int, ...], complex_: bool, fn: str) -> np.ndarray:
    rng = np.random.default_rng(sum(shape) + 7 * len(shape))
    a = rng.standard_normal(shape)
    if complex_:
        a = a + 1j * rng.standard_normal(shape)
    if fn == "cond":
        # Well conditioned, so the reference and Lucid's inverses agree to
        # the tolerance below.
        a = a + 3 * np.eye(shape[-1])
    return a.astype(np.complex64 if complex_ else np.float32)


def _np(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.to("cpu").numpy())


def _tol(device: str) -> dict[str, float]:
    return {"rtol": 1e-3, "atol": 1e-4} if device == "metal" else {"rtol": 1e-4, "atol": 1e-5}


def _lucid_call(fn: str, a: np.ndarray, kw: dict[str, Any], device: str) -> Any:
    return getattr(LA, fn)(lucid.tensor(a, device=device), **kw)


# ── the class, through the reference ──────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize(("fn", "shape", "kw", "kind"), _PARITY)
def test_norm_is_real_and_matches_the_reference(
    fn: str,
    shape: tuple[int, ...],
    kw: dict[str, Any],
    kind: str,
    device: str,
    ref: Any,
) -> None:
    a = _data(shape, kind == "complex", fn)
    got = _lucid_call(fn, a, kw, device)
    want = getattr(ref.linalg, fn)(ref.tensor(a), **kw)

    assert not got.is_complex()
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(_np(got), want.numpy(), **_tol(device))


# ── the class, without the reference ─────────────────────────────────────────


@pytest.mark.parametrize(("fn", "shape", "kw"), _COMPLEX_SUPPORTED)
def test_complex_norm_is_real(
    fn: str, shape: tuple[int, ...], kw: dict[str, Any], device: str
) -> None:
    """A complex input gives the real dtype of the same precision, and the
    value the same call gives on ``|x|``.

    The reference is not needed for this half, so it runs everywhere.  Every
    order that accepts a complex input is entry-wise, so ``|x|`` in place of
    ``x`` is its definition, not an approximation of it.
    """
    a = _data(shape, True, fn)
    got = _lucid_call(fn, a, kw, device)
    assert not got.is_complex()
    assert got.dtype == lucid.float32
    mags = _lucid_call(fn, np.abs(a).astype(np.float32), kw, device)
    np.testing.assert_allclose(_np(got), _np(mags), **_tol(device))


def test_complex128_norm_is_float64() -> None:
    """The real counterpart keeps the precision: complex128 gives float64."""
    a = _data((3, 4), True, "norm").astype(np.complex128)
    x = lucid.tensor(a)
    for out in (
        LA.vector_norm(x),
        LA.matrix_norm(x, "fro"),
        LA.matrix_norm(x, 1),
        LA.matrix_norm(x, -_INF),
        LA.norm(x),
        LA.norm(x, "fro"),
        LA.norm(x, dim=(0, 1)),
    ):
        assert out.dtype == lucid.float64


@pytest.mark.parametrize(("fn", "shape", "kw", "refused"), _COMPLEX_REFUSED)
def test_complex_norm_without_a_kernel_is_refused(
    fn: str, shape: tuple[int, ...], kw: dict[str, Any], refused: str, device: str
) -> None:
    """The orders with no complex kernel refuse by name, on every device.

    The refusal is a typed ``NotImplementedError`` from the norm itself —
    not whichever of ``svd`` / ``inv_ex`` fails first with a message that
    does not say which norm was asked for.
    """
    a = _data(shape, True, fn)
    with pytest.raises(NotImplementedError, match=refused):
        _lucid_call(fn, a, kw, device)


def test_the_issue_repro(device: str) -> None:
    """The numbers from CHA-231."""
    x = lucid.tensor([[1 + 1j, 2], [1j, 3 - 1j]], device=device)
    want = float(np.sqrt(17.0))
    for out in (
        LA.matrix_norm(x, "fro"),
        LA.norm(x, "fro"),
        LA.norm(x, dim=(0, 1)),
    ):
        assert not out.is_complex()
        assert float(out.item()) == pytest.approx(want, rel=1e-6)


# ── gradients of the entry-wise matrix orders ────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("ord", ["fro", 1, -1, _INF, -_INF])
def test_complex_matrix_norm_gradient_matches_the_reference(
    ord: int | float | str, ref: Any
) -> None:
    rng = np.random.default_rng(11)
    a = rng.standard_normal((2, 3, 4)) + 1j * rng.standard_normal((2, 3, 4))
    t = lucid.tensor(a, requires_grad=True)
    LA.matrix_norm(t, ord).sum().backward()
    r = ref.tensor(a, requires_grad=True)
    ref.linalg.matrix_norm(r, ord).sum().backward()
    assert t.grad is not None
    got = _np(lucid.real(t.grad)) + 1j * _np(lucid.imag(t.grad))
    np.testing.assert_allclose(got, r.grad.numpy(), rtol=1e-10, atol=1e-12)
