"""The norm and variance reductions own their numerics (LCD-38, LCD-39, LCD-114).

Six entry points — ``linalg.vector_norm``, ``linalg.norm``, ``var``,
``std``, ``var_mean`` and ``std_mean`` — end in the same three places, and
each used to get one of them wrong:

* **zero-safe roots** — the gradient of a zero norm, or of the ``std`` of a
  constant slice, was ``nan`` (the root's infinite slope times a zero);
  the reference gives 0;
* **the divisor** ``max(N - correction, 0)`` — too few elements answered a
  variance that silently skipped its correction (``var([5.])`` was 0, the
  reference's is ``nan``), and an empty one answered 0 on the CPU;
* **the accumulator** — a half-precision norm summed its squares in half
  precision and overflowed: the norm of 65 536 ones was ``inf``, not 256.

``cdist`` and ``nn.functional.pairwise_distance`` take the same roots, and
had the same ``nan`` at coincident points.

Alongside: every reduction refuses a dim named twice, or a dim that is not
an integer, in its own name; and the four variance functions share one
reading of their arguments — a positional bool means what it means to the
reference (``var(x, False)`` is ``unbiased=False``, and ``var(x, dim, b)``
reads ``b`` as ``unbiased``; it was ``keepdim`` until LCD-410).

The parity tests compare values and gradients against the reference on
both devices.  A value or a gradient the reference gives as finite must be
finite and equal here; a value it gives as ``nan``/``inf`` must be the same
here.
"""

import math
import warnings
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.linalg as LA

#: Named inputs, each a 2-D float list: a zero slice, a constant slice, a
#: mix of the two with a non-trivial row, and a column (one element per
#: row, so ``N = 1`` along dim 1).
_INPUTS: dict[str, list[list[float]]] = {
    "zero": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    "constant": [[2.0, 2.0, 2.0, 2.0], [-3.0, -3.0, -3.0, -3.0]],
    "mixed": [[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 3.0, 0.5]],
    "column": [[1.0], [2.0], [3.0]],
}
_DIMS: list[int | tuple[int, ...] | None] = [None, 1, -1, (0, 1)]

#: The norm orders whose gradient goes through a root, an ``abs`` or a
#: comparison — all of them see a zero norm.
_ORDS = [2, 1, 3, 0.5, 1.5, -1, math.inf, -math.inf]
_CORRECTIONS = [0, 1, 3, 5]

_VARIANCE_FNS = ["var", "std", "var_mean", "std_mean"]


def _outputs(result: object) -> list[object]:
    return list(result) if isinstance(result, tuple) else [result]


def _lucid_np(t: lucid.Tensor) -> np.ndarray:
    return t.to(lucid.float32).numpy().astype(np.float64)


def _ref_np(t: object) -> np.ndarray:
    return t.detach().float().numpy().astype(np.float64)


def _assert_matches(
    ours: np.ndarray,
    theirs: np.ndarray,
    what: str,
    rtol: float = 1e-5,
    *,
    exact_nonfinite: bool = True,
) -> None:
    """``ours`` equals ``theirs`` wherever the reference is finite.

    A value must also be ``nan`` / ``inf`` exactly where the reference's is.
    A gradient need not (``exact_nonfinite=False``): where the reference's is
    non-finite it is its own ``inf * 0``, and not even consistent — with
    ``N < correction`` it is finite for an explicit ``dim`` and ``nan`` for
    ``dim=None``.  Lucid gives the finite slope in both.
    """
    assert ours.shape == theirs.shape, f"{what}: shape {ours.shape} vs {theirs.shape}"
    finite = np.isfinite(theirs)
    assert np.isfinite(
        ours[finite]
    ).all(), f"{what}: nan/inf where the reference is finite\n{ours}\nvs\n{theirs}"
    np.testing.assert_allclose(ours[finite], theirs[finite], rtol=rtol, atol=rtol, err_msg=what)
    if exact_nonfinite:
        np.testing.assert_array_equal(ours[~finite], theirs[~finite], err_msg=what)


def _compare(
    ref: ModuleType,
    device: str,
    values: list[list[float]],
    ours_fn: Callable[[lucid.Tensor], object],
    ref_fn: Callable[[object], object],
    what: str,
) -> None:
    x = lucid.tensor(values, device=device, requires_grad=True)
    rx = ref.tensor(values, requires_grad=True)
    ours, theirs = _outputs(ours_fn(x)), _outputs(ref_fn(rx))
    assert len(ours) == len(theirs)
    # Without a gradient to protect the reductions take a shorter path,
    # which must answer the same values.
    plain = _outputs(ours_fn(lucid.tensor(values, device=device)))
    for i, (o, p, t) in enumerate(zip(ours, plain, theirs, strict=True)):
        _assert_matches(_lucid_np(o), _ref_np(t), f"{what} output {i}")
        _assert_matches(_lucid_np(p), _ref_np(t), f"{what} output {i}, no grad")
    sum(o.sum() for o in ours).backward()
    sum(t.sum() for t in theirs).backward()
    assert x.grad is not None
    _assert_matches(_lucid_np(x.grad), _ref_np(rx.grad), f"{what} grad", exact_nonfinite=False)


# ── Norms: zero-safe gradient ────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("dim", _DIMS, ids=str)
@pytest.mark.parametrize("ord_", _ORDS, ids=str)
@pytest.mark.parametrize("name", list(_INPUTS))
def test_vector_norm_value_and_gradient_match(
    ref: ModuleType, device: str, name: str, ord_: float, dim: object
) -> None:
    _compare(
        ref,
        device,
        _INPUTS[name],
        lambda t: LA.vector_norm(t, ord_, dim),
        lambda t: ref.linalg.vector_norm(t, ord_, dim),
        f"vector_norm(ord={ord_}, dim={dim}) of {name}",
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    "ord_, dim",
    [(None, None), ("fro", None), (2, 1), (3, 1), (None, (0, 1)), (1, None)],
    ids=str,
)
@pytest.mark.parametrize("name", list(_INPUTS))
def test_norm_value_and_gradient_match(
    ref: ModuleType, device: str, name: str, ord_: object, dim: object
) -> None:
    _compare(
        ref,
        device,
        _INPUTS[name],
        lambda t: LA.norm(t, ord_, dim),
        lambda t: ref.linalg.norm(t, ord_, dim),
        f"norm(ord={ord_}, dim={dim}) of {name}",
    )


# ── Variances: the divisor and the std's root ────────────────────────────────


def _variance_call(fn: str, dim: object, correction: int) -> Callable[..., object]:
    def call(module: object, t: object) -> object:
        return getattr(module, fn)(t, dim=dim, correction=correction)

    return call


@pytest.mark.parity
@pytest.mark.parametrize("correction", _CORRECTIONS)
@pytest.mark.parametrize("dim", _DIMS, ids=str)
@pytest.mark.parametrize("name", list(_INPUTS))
@pytest.mark.parametrize("fn", _VARIANCE_FNS)
def test_variance_value_and_gradient_match(
    ref: ModuleType, device: str, fn: str, name: str, dim: object, correction: int
) -> None:
    call = _variance_call(fn, dim, correction)
    with warnings.catch_warnings():
        # The reference warns that N <= correction; Lucid answers silently.
        warnings.simplefilter("ignore", UserWarning)
        _compare(
            ref,
            device,
            _INPUTS[name],
            lambda t: call(lucid, t),
            lambda t: call(ref, t),
            f"{fn}(dim={dim}, correction={correction}) of {name}",
        )


@pytest.mark.parametrize("fn", _VARIANCE_FNS)
def test_a_constant_slice_has_a_zero_gradient(device: str, fn: str) -> None:
    x = lucid.full((3, 4), 2.0, device=device, requires_grad=True)
    out = _outputs(getattr(lucid, fn)(x, 1))[0]
    out.sum().backward()
    assert x.grad is not None
    assert (_lucid_np(x.grad) == 0).all()


@pytest.mark.parametrize("fn", _VARIANCE_FNS)
@pytest.mark.parametrize("shape, dim", [((0,), None), ((2, 0), 1), ((0, 3), 0)])
def test_an_empty_reduction_is_nan(
    device: str, fn: str, shape: tuple[int, ...], dim: int | None
) -> None:
    # The CPU kernel used to answer 0 here and Metal nan.
    out = _outputs(getattr(lucid, fn)(lucid.zeros(*shape, device=device), dim))[0]
    assert np.isnan(_lucid_np(out)).all()


def test_a_single_element_has_no_sample_variance(device: str) -> None:
    v, m = lucid.var_mean(lucid.tensor([5.0], device=device))
    assert math.isnan(v.item()) and m.item() == 5.0
    assert math.isinf(lucid.var(lucid.tensor([1.0, 2.0, 3.0], device=device), correction=3).item())


@pytest.mark.parametrize("fn", ["var_mean", "std_mean"])
def test_the_pair_honours_correction_and_keepdim_over_every_axis(device: str, fn: str) -> None:
    # ``dim=None`` used to drop both and answer the default.
    x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    spread, mean = getattr(lucid, fn)(x, None, 0, True)
    single = getattr(lucid, fn.removesuffix("_mean"))(x, correction=0, keepdim=True)
    assert spread.shape == mean.shape == (1, 1)
    np.testing.assert_allclose(_lucid_np(spread), _lucid_np(single))


# ── Half precision accumulates in float32 ────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "fn",
    [
        "vector_norm-2",
        "vector_norm-3",
        "vector_norm-1.5",
        "norm",
        "var",
        "std",
    ],
)
def test_half_precision_does_not_overflow(
    ref: ModuleType, device: str, dtype: str, fn: str
) -> None:
    # 65 536 elements of ~1: every sum of squares is past float16's range,
    # while every answer is inside it.
    base = np.ones((64, 1024), dtype=np.float32)
    base[:, ::2] += 1.0
    x = lucid.tensor(base, device=device).to(getattr(lucid, dtype))
    rx = ref.tensor(base).to(getattr(ref, dtype))
    if fn.startswith("vector_norm"):
        p = float(fn.split("-")[1])
        ours, theirs = LA.vector_norm(x, p), ref.linalg.vector_norm(rx, p)
    elif fn == "norm":
        ours, theirs = LA.norm(x), ref.linalg.norm(rx)
    else:
        ours, theirs = getattr(lucid, fn)(x), getattr(ref, fn)(rx)
    assert ours.dtype == getattr(lucid, dtype)
    assert math.isfinite(ours.item())
    np.testing.assert_allclose(ours.item(), theirs.item(), rtol=1e-2)


# ── Distances: the same zero-root class ──────────────────────────────────────

#: Rows 0 and 2 of ``_A`` coincide with row 0 of ``_B``: a zero distance.
#: No difference vector holds a tie in magnitude, so the ``inf`` order's
#: subgradient is unambiguous.
_A = [[0.0, 1.0], [2.0, 4.5], [0.0, 1.0]]
_B = [[0.0, 1.0], [5.0, 5.5]]


@pytest.mark.parity
@pytest.mark.parametrize("p", [2.0, 1.0, 3.0, 1.5, 0.5, math.inf], ids=str)
def test_cdist_at_coincident_points_matches(ref: ModuleType, device: str, p: float) -> None:
    y, ry = lucid.tensor(_B, device=device), ref.tensor(_B)
    _compare(
        ref,
        device,
        _A,
        lambda t: lucid.cdist(t, y, p),
        lambda t: ref.cdist(t, ry, p),
        f"cdist(p={p})",
    )


@pytest.mark.parity
@pytest.mark.parametrize("eps", [0.0, 1e-6], ids=str)
@pytest.mark.parametrize("p", [2.0, 1.0, 3.0, 1.5, 0.5], ids=str)
def test_pairwise_distance_at_coincident_points_matches(
    ref: ModuleType, device: str, p: float, eps: float
) -> None:
    import lucid.nn.functional as F

    other = [[0.0, 1.0], [1.0, 2.0], [0.0, 1.0]]
    y, ry = lucid.tensor(other, device=device), ref.tensor(other)
    _compare(
        ref,
        device,
        _A,
        lambda t: F.pairwise_distance(t, y, p, eps),
        lambda t: ref.nn.functional.pairwise_distance(t, ry, p, eps),
        f"pairwise_distance(p={p}, eps={eps})",
    )


# ── Axes ─────────────────────────────────────────────────────────────────────

_ALL_FNS: dict[str, Callable[[lucid.Tensor, object], object]] = {
    "vector_norm": lambda t, d: LA.vector_norm(t, 2, d),
    "norm": lambda t, d: LA.norm(t, None, d),
    "var": lambda t, d: lucid.var(t, d),
    "std": lambda t, d: lucid.std(t, d),
    "var_mean": lambda t, d: lucid.var_mean(t, d),
    "std_mean": lambda t, d: lucid.std_mean(t, d),
    "sum": lambda t, d: lucid.sum(t, d),
    "mean": lambda t, d: lucid.mean(t, d),
    "prod": lambda t, d: lucid.prod(t, d),
    "max": lambda t, d: lucid.max(t, d),
    "min": lambda t, d: lucid.min(t, d),
    "amax": lambda t, d: lucid.amax(t, d),
    "amin": lambda t, d: lucid.amin(t, d),
    "nansum": lambda t, d: lucid.nansum(t, d),
    "nanmean": lambda t, d: lucid.nanmean(t, d),
    "count_nonzero": lambda t, d: lucid.count_nonzero(t, d),
    "logsumexp": lambda t, d: lucid.logsumexp(t, d),
}


@pytest.mark.parametrize("dim", [(1, 1), (1, -1), (0, 1, 0)], ids=str)
@pytest.mark.parametrize("fn", list(_ALL_FNS))
def test_an_axis_named_twice_is_refused_in_the_ops_own_name(
    device: str, fn: str, dim: object
) -> None:
    # ``var`` counted the axis into N twice and ``logsumexp`` failed in
    # ``squeeze``; the reference refuses the call.  The message names the op
    # that was called, not the one it is built on (``amax`` is not ``max``).
    # ``norm`` hands two axes to ``matrix_norm``, which words it its own way.
    x = lucid.ones(2, 3, device=device)
    expected = "matrix_norm|norm" if fn == "norm" else rf"^{fn}\(\): .*multiple times"
    with pytest.raises(ValueError, match=expected):
        _ALL_FNS[fn](x, dim)


@pytest.mark.parametrize("dim", [1.0, (0, 1.5), "1"], ids=str)
@pytest.mark.parametrize("fn", [f for f in _ALL_FNS if f != "norm"])
def test_a_non_integer_dim_is_a_type_error_naming_dim(fn: str, dim: object) -> None:
    # A float used to truncate, or fail as "'float' object is not iterable".
    with pytest.raises(TypeError, match=rf"{fn}\(\): dim must be an int"):
        _ALL_FNS[fn](lucid.ones(2, 3), dim)


# ── The positional contract — one reading for all four ───────────────────────

_X = [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]

#: Every spelling the reference accepts.  ``(dim, bool)`` is the one that
#: differs: ``var`` / ``std`` still read it as ``(dim, keepdim)`` until 3.18.
_SPELLINGS: list[tuple[tuple[object, ...], dict[str, object]]] = [
    ((), {}),
    ((1,), {}),
    ((False,), {}),
    ((True,), {}),
    ((1, False), {}),
    ((1, True), {}),
    ((1, True, True), {}),
    ((1, False, False), {}),
    (((0, 1), False, True), {}),
    ((1,), {"keepdim": True}),
    ((), {"unbiased": False}),
    ((), {"dim": 1, "unbiased": False}),
    ((1,), {"unbiased": True, "keepdim": True}),
    ((), {"dim": 1, "correction": 0, "keepdim": True}),
    ((), {"correction": True}),
    ((), {"correction": False}),
    ((), {"correction": 0.5, "dim": 1}),
    ((0,), {"correction": 2}),
    ((), {"dim": (0, 1), "correction": 0}),
    ((1, False), {"keepdim": True}),
]


def _callers(fn: str, x: lucid.Tensor) -> list[Callable[..., object]]:
    free = getattr(lucid, fn)
    if fn in ("var", "std"):
        return [
            lambda *a, **k: free(x, *a, **k),
            lambda *a, **k: getattr(x, fn)(*a, **k),
        ]
    return [lambda *a, **k: free(x, *a, **k)]


@pytest.mark.parity
@pytest.mark.parametrize("args, kwargs", _SPELLINGS, ids=[f"{a}-{k}" for a, k in _SPELLINGS])
@pytest.mark.parametrize("fn", _VARIANCE_FNS)
def test_every_spelling_matches_the_reference(
    ref: ModuleType,
    device: str,
    fn: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    x = lucid.tensor(_X, device=device)
    # ``var(x, dim, b)`` reads ``b`` as ``unbiased``, as the reference does;
    # it was ``keepdim`` until 3.16, then a deprecation until LCD-410.
    for call in _callers(fn, x):
        with warnings.catch_warnings(record=True) as ours_said:
            warnings.simplefilter("always")
            ours = _outputs(call(*args, **kwargs))
        with warnings.catch_warnings(record=True) as theirs_said:
            warnings.simplefilter("always")
            theirs = _outputs(getattr(ref, fn)(ref.tensor(_X), *args, **kwargs))
        # Both warn that N <= correction, and Lucid of nothing else.
        assert all(_dof(w) for w in ours_said)
        assert any(map(_dof, ours_said)) == any(map(_dof, theirs_said))
        for o, t in zip(ours, theirs, strict=True):
            _assert_matches(_lucid_np(o), _ref_np(t), f"{fn}{args}{kwargs}")


def _dof(caught: warnings.WarningMessage) -> bool:
    return "degrees of freedom is <= 0" in str(caught.message)


@pytest.mark.parametrize("fn", ["var_mean", "std_mean"])
def test_the_pairs_keep_their_released_positional_correction(device: str, fn: str) -> None:
    # ``var_mean(x, dim, correction, keepdim)`` is how the pair was spelled
    # before the shared reading; an int in that place stays a correction.
    x = lucid.tensor(_X, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = getattr(lucid, fn)(x, 1, 2, True)
        by_name = getattr(lucid, fn)(input=x, dim=1, correction=2, keepdim=True)
    for g, b in zip(got, by_name, strict=True):
        assert g.shape == (2, 1)
        np.testing.assert_allclose(_lucid_np(g), _lucid_np(b))


@pytest.mark.parametrize("fn", ["var", "std"])
def test_input_by_keyword_stays_silent(device: str, fn: str) -> None:
    x = lucid.tensor(_X, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(lucid, fn)(input=x, dim=1)


@pytest.mark.parametrize("fn", _VARIANCE_FNS)
@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((False, 1), {}),
        ((1, True, True, 0), {}),
        ((1,), {"dim": 1}),
        ((1, True, False), {"keepdim": True}),
        ((), {"axis": 1}),
        ((), {"correction": "1"}),
        ((1.5,), {}),
    ],
    ids=str,
)
def test_malformed_calls_are_refused(
    fn: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    x = lucid.tensor(_X)
    with pytest.raises(TypeError):
        getattr(lucid, fn)(x, *args, **kwargs)


@pytest.mark.parametrize("fn", ["amax", "amin", "nansum", "nanmean"])
def test_dim_none_honours_keepdim(device: str, fn: str) -> None:
    x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    free = getattr(lucid, fn)(x, dim=None, keepdim=True)
    assert free.shape == (1, 1)


@pytest.mark.parity
@pytest.mark.parametrize("ord", ["fro", "nuc", 1, 2, float("inf")])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (2, 0, 3)], ids=str)
@pytest.mark.parametrize("keepdim", [False, True])
def test_matrix_norm_empty_axes_return_zero(
    ref: ModuleType,
    device: str,
    ord: str | int | float,
    shape: tuple[int, ...],
    keepdim: bool,
) -> None:
    x = lucid.zeros(*shape, device=device)
    rx = ref.zeros(*shape)
    ours = LA.matrix_norm(x, ord=ord, keepdim=keepdim)
    theirs = ref.linalg.matrix_norm(rx, ord=ord, keepdim=keepdim)
    assert ours.shape == tuple(theirs.shape)
    np.testing.assert_array_equal(_lucid_np(ours), _ref_np(theirs))


@pytest.mark.parity
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("fn", ["var", "std", "var_mean", "std_mean"])
def test_half_std_rounds_once_after_sqrt_and_compiled_matches_eager(
    ref: ModuleType, device: str, dtype_name: str, correction: int, fn: str
) -> None:
    # Small-magnitude inputs where rounding var to half before sqrt loses 15%
    # (bfloat16 [0, 0.01]: var=5e-5 rounds to 4.98e-5, sqrt 0.00704956 vs
    # float32 sqrt(5e-5)=0.00707107 -> bfloat16 0.00708008).
    dt = getattr(lucid, dtype_name)
    rdt = getattr(ref, dtype_name)
    vals = [[0.0, 0.01, -0.01, 0.02], [1e-3, -1e-3, 2e-3, -2e-3]]
    x = lucid.tensor(vals, dtype=dt, device=device)
    rx = ref.tensor(vals, dtype=rdt)
    ours = _outputs(getattr(lucid, fn)(x, dim=-1, correction=correction))
    theirs = _outputs(getattr(ref, fn)(rx, dim=-1, correction=correction))
    for o, t in zip(ours, theirs, strict=True):
        assert o.dtype == dt
        np.testing.assert_array_equal(_lucid_np(o), _ref_np(t), err_msg=f"{fn} {dtype_name}")
    if device == "metal" and fn in ("var", "std"):
        c_fn = lucid.compile.compile(
            lambda t_in: getattr(lucid, fn)(t_in, dim=-1, correction=correction)
            + (t_in[:, :1].squeeze(-1) * 0)
        )
        c_out = c_fn(x)
        assert c_out.dtype == dt
        np.testing.assert_array_equal(_lucid_np(c_out), _lucid_np(ours[0]))


@pytest.mark.parity
def test_int64_prod_preserves_bits_above_2_pow_53_and_wraps_modulo_2_pow_64(
    ref: ModuleType, device: str
) -> None:
    for vals in ([2**53 + 1, 1], [2**62, 4], [2**32 + 3, 2**31 + 5]):
        x = lucid.tensor(vals, dtype=lucid.int64, device=device)
        rx = ref.tensor(vals, dtype=ref.int64)
        assert x.prod().item() == rx.prod().item()
        assert lucid.prod(x, dim=0).item() == ref.prod(rx, dim=0).item()
