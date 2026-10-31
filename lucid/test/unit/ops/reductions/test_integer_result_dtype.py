"""Integer and bool results keep the reference's dtype (LCD-668).

``sum`` has always widened ``bool`` / ``int8`` / ``int16`` / ``int32`` to ``int64``
before adding, but ``trace`` kept the narrow input dtype (so ``int8`` overflowed
where the reference does not), and ``square`` of a ``bool`` answered ``bool`` where
the reference answers the ``int64`` of ``x ** 2``.  The matrix below holds every
op that adds up or multiplies integer elements to the reference's result dtype
and values, on both streams.
"""

from collections.abc import Callable
from typing import Any
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params, metal_available

_DTYPES = [lucid.bool, lucid.int8, lucid.int16, lucid.int32, lucid.int64]
_INT_DTYPES = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


def _matrix(dtype: lucid.dtype) -> list[list[int]]:
    return [[1, 0], [1, 1]] if dtype == lucid.bool else [[1, 2], [3, 4]]


_OPS: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], Callable[[object], object]]] = {
    "sum": (lambda x: lucid.sum(x), lambda x: x.sum()),
    "sum_dim": (lambda x: lucid.sum(x, 0, keepdim=True), lambda x: x.sum(0, keepdim=True)),
    "prod": (lambda x: lucid.prod(x), lambda x: x.prod()),
    "cumsum": (lambda x: lucid.cumsum(x, 0), lambda x: x.cumsum(0)),
    "cumprod": (lambda x: lucid.cumprod(x, 1), lambda x: x.cumprod(1)),
    "nansum": (lambda x: lucid.nansum(x), lambda x: x.nansum()),
    "trace": (lambda x: lucid.trace(x), lambda x: x.trace()),
    "square": (lambda x: lucid.square(x), lambda x: x.square()),
}


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
@pytest.mark.parametrize("name", list(_OPS))
def test_integer_result_dtype_and_values_match_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype, name: str
) -> None:
    """One result-dtype rule for every op that accumulates integer elements."""
    lucid_fn, ref_fn = _OPS[name]
    rx = ref.tensor(_matrix(dtype), dtype=_ref_dtype(ref, dtype))
    x = lucid.tensor(_matrix(dtype), dtype=dtype, device=device)
    try:
        want = ref_fn(rx)
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            lucid_fn(x)
        return
    got = lucid_fn(x)
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES[:3]))
def test_trace_adds_up_in_int64(device: str, dtype: lucid.dtype) -> None:
    """The sum of the diagonal has int64 headroom, as ``sum`` does."""
    limit = {lucid.int8: 100, lucid.int16: 30000, lucid.int32: 2_000_000_000}[dtype]
    x = lucid.tensor([[limit, 0], [0, limit]], dtype=dtype, device=device)
    out = lucid.trace(x)
    assert out.dtype == lucid.int64
    assert out.item() == 2 * limit
    assert out.item() == lucid.sum(lucid.diagonal(x)).item()


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES))
def test_batched_trace_follows_the_two_dimensional_rule(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    """A trace over leading matrix dimensions widens like the 2-D one."""
    vals = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    got = lucid.trace(lucid.tensor(vals, dtype=dtype, device=device))
    want = ref.tensor(vals, dtype=_ref_dtype(ref, dtype)).diagonal(0, 0, 1).sum(-1)
    assert got.dtype == lucid.int64
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_trace_of_bool_is_refused_as_in_the_reference(ref: ModuleType, device: str) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    with pytest.raises(NotImplementedError):
        ref.tensor([[True, False], [True, True]]).trace()
    with pytest.raises(NotImplementedError):
        lucid.trace(lucid.tensor([[True, False], [True, True]], device=device))


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_square_of_bool_is_an_int64_power(ref: ModuleType, device: str) -> None:
    """``square`` is ``x ** 2`` (int64 for a bool), not the logical and of ``x * x``."""
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    vals = [True, False, True, True]
    b = lucid.tensor(vals, device=device)
    rb = ref.tensor(vals)
    for got, want in ((lucid.square(b), rb.square()), (b.square(), rb.square())):
        assert got.dtype == lucid.int64
        assert got.tolist() == want.tolist()
    assert (b * b).dtype == lucid.bool
    assert (rb * rb).dtype == ref.bool

    # In place the int64 answer cannot be written back into a bool, for ``cube_``
    # (``x ** 3``) as for ``square_``.
    for method in ("square_", "cube_"):
        with pytest.raises(RuntimeError, match="can't be cast"):
            getattr(b.clone(), method)()
    with pytest.raises(RuntimeError, match="can't be cast"):
        rb.clone().square_()
    with pytest.raises(RuntimeError, match="can't be cast"):
        rb.clone().pow_(3)


# ── contractions (LCD-768, LCD-728) ──────────────────────────────────────────
#
# A product summed over a shared axis keeps its operands' dtype; a sum inside
# one operand widens to int64; bool takes no contraction.  ``einsum`` and the
# BLAS composites decide this through ``lucid._contraction``.


def _operands(mod: ModuleType, dtype: object, device: str | None = None) -> dict[str, Any]:
    """Small operands in ``dtype``: ``a`` (2, 3), ``b`` (3, 2), ``m`` (2, 2), ``v`` (3,)."""
    kw: dict[str, object] = {"dtype": dtype}
    if device is not None:
        kw["device"] = device
    a = mod.tensor([[1, 0, 1], [1, 1, 0]], **kw)
    b = mod.tensor([[1, 1], [0, 1], [1, 0]], **kw)
    return {
        "a": a,
        "b": b,
        "m": mod.tensor([[1, 0], [1, 1]], **kw),
        "v": mod.tensor([1, 1, 0], **kw),
        "w": mod.tensor([[1, 0, 1, 1], [0, 1, 1, 0]], **kw),
        "ab": a[None],
        "bb": b[None],
        "mb": mod.tensor([[[1, 0], [1, 1]]], **kw),
        "vec3": mod.tensor([1, 0, 1], **kw),
        "sq": mod.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]], **kw),
    }


_CONTRACTIONS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "einsum_matmul": lambda m, o: m.einsum("ij,jk->ik", o["a"], o["b"]),
    "einsum_batched": lambda m, o: m.einsum("bij,bjk->bik", o["ab"], o["bb"]),
    "einsum_dot_rows": lambda m, o: m.einsum("ij,ij->i", o["a"], o["a"]),
    "einsum_chain": lambda m, o: m.einsum("ij,jk,kl->il", o["a"], o["b"], o["a"]),
    "einsum_list": lambda m, o: m.einsum("ij,jk->ik", [o["a"], o["b"]]),
    "einsum_outer": lambda m, o: m.einsum("i,j->ij", o["v"], o["vec3"]),
    "einsum_hadamard": lambda m, o: m.einsum("ij,ij->ij", o["a"], o["a"]),
    "einsum_sum": lambda m, o: m.einsum("ij->j", o["a"]),
    "einsum_trace": lambda m, o: m.einsum("ii->", o["m"]),
    "einsum_diagonal": lambda m, o: m.einsum("ii->i", o["m"]),
    "einsum_loner": lambda m, o: m.einsum("ij,kl->ik", o["a"], o["w"]),
    "einsum_contract_all": lambda m, o: m.einsum("ij,jk->", o["a"], o["b"]),
    "mv": lambda m, o: m.mv(o["a"], o["vec3"]),
    "vdot": lambda m, o: m.vdot(o["v"], o["vec3"]),
    "addmm": lambda m, o: m.addmm(o["m"], o["a"], o["b"]),
    "addmm_scaled": lambda m, o: m.addmm(o["m"], o["a"], o["b"], beta=3, alpha=2),
    "addmm_beta_zero": lambda m, o: m.addmm(o["m"], o["a"], o["b"], beta=0),
    "addmv": lambda m, o: m.addmv(o["m"][0], o["a"], o["vec3"]),
    "baddbmm": lambda m, o: m.baddbmm(o["mb"], o["ab"], o["bb"]),
    "baddbmm_scaled": lambda m, o: m.baddbmm(o["mb"], o["ab"], o["bb"], beta=2, alpha=3),
    "addbmm": lambda m, o: m.addbmm(o["m"], o["ab"], o["bb"]),
    "addbmm_scaled": lambda m, o: m.addbmm(o["m"], o["ab"], o["bb"], beta=2, alpha=3),
    "addr": lambda m, o: m.addr(o["sq"], o["vec3"], o["vec3"]),
    "addr_scaled": lambda m, o: m.addr(o["sq"], o["vec3"], o["vec3"], beta=2, alpha=3),
    "addcmul": lambda m, o: m.addcmul(o["v"], o["v"], o["vec3"]),
    "addcmul_scaled": lambda m, o: m.addcmul(o["v"], o["v"], o["vec3"], value=3),
    # A float scalar is cast to an integer dtype by the GEMM-backed ops and
    # refused by ``addr``; the reference does both.
    "addmm_float_scales": lambda m, o: m.addmm(o["m"], o["a"], o["b"], beta=2.7, alpha=2.0),
    "addmv_float_scales": lambda m, o: m.addmv(o["m"][0], o["a"], o["vec3"], beta=0.5, alpha=3.0),
    "baddbmm_float_scales": lambda m, o: m.baddbmm(o["mb"], o["ab"], o["bb"], beta=2.0, alpha=0.5),
    "addbmm_float_scales": lambda m, o: m.addbmm(o["m"], o["ab"], o["bb"], beta=2.0, alpha=2.9),
    "addcmul_float_value": lambda m, o: m.addcmul(o["v"], o["v"], o["vec3"], value=2.5),
    "addr_float_scales": lambda m, o: m.addr(o["sq"], o["vec3"], o["vec3"], beta=2.0),
}


class _LucidEinsum:
    """``lucid`` with ``einsum`` where the reference has it: under ``lucid.einops``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(lucid.einops if name == "einsum" else lucid, name)


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_DTYPES))
@pytest.mark.parametrize("name", list(_CONTRACTIONS))
def test_contraction_result_dtype_and_values_match_reference(
    ref: ModuleType, device: str, dtype: lucid.dtype, name: str
) -> None:
    """One result-dtype rule for every contraction, on both streams."""
    build = _CONTRACTIONS[name]
    theirs = _operands(ref, _ref_dtype(ref, dtype))
    ours = _operands(lucid, dtype, device)
    try:
        want = build(ref, theirs)
    except (NotImplementedError, RuntimeError) as refused:
        with pytest.raises(type(refused)):
            build(_LucidEinsum(), ours)
        return
    got = build(_LucidEinsum(), ours)
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params(_INT_DTYPES[:3]))
def test_an_integer_contraction_wraps_at_the_operand_width(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    """The narrowed sum is the one an operand-width accumulator gives."""
    big = {lucid.int8: 100, lucid.int16: 30000, lucid.int32: 2_000_000_000}[dtype]
    vals = [[big, big], [big, 3]]
    x = lucid.tensor(vals, dtype=dtype, device=device)
    out = lucid.einops.einsum("ij,jk->ik", x, x)
    rx = ref.tensor(vals, dtype=_ref_dtype(ref, dtype))
    assert out.tolist() == (rx @ rx).tolist()


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("scale", ["beta", "alpha"])
def test_addr_refuses_a_float_scale_of_an_integer_input(
    ref: ModuleType, device: str, scale: str
) -> None:
    """``addr`` names the scalar it refuses, as the reference does."""
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    message = f"argument {scale} must not be a floating"
    for mod, dev in ((ref, None), (lucid, device)):
        o = _operands(mod, mod.int32, dev)
        with pytest.raises(RuntimeError, match=message):
            mod.addr(o["sq"], o["vec3"], o["vec3"], **{scale: 0.5})


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_a_float_blas_op_keeps_its_float_scales(device: str) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    m = lucid.ones(2, 2, device=device)
    out = lucid.addmm(m, m, m, beta=0.5, alpha=2)
    assert out.dtype == lucid.float32
    assert out.tolist() == [[4.5, 4.5], [4.5, 4.5]]


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_addbmm_of_integers_is_not_widened(device: str) -> None:
    """The batch sum inside ``addbmm`` is part of the contraction (LCD-728)."""
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    for dtype in _INT_DTYPES:
        x = lucid.ones(3, 2, 2, dtype=dtype, device=device)
        out = lucid.addbmm(lucid.zeros(2, 2, dtype=dtype, device=device), x, x)
        assert out.dtype == dtype
        assert out.tolist() == [[6, 6], [6, 6]]


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_integer_matmul_family_keeps_the_operand_dtype(device: str) -> None:
    """The engine's own contractions are the rule the composites follow."""
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")
    for dtype in _INT_DTYPES:
        a = lucid.ones(2, 2, dtype=dtype, device=device)
        assert lucid.matmul(a, a).dtype == dtype
        assert lucid.bmm(a[None], a[None]).dtype == dtype
        assert lucid.mv(a, a[0]).dtype == dtype
