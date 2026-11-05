"""A traced custom VJP hands ``backward`` the cotangents eager hands it (LCD-863).

A ``Function`` that does not materialize grads (``_materialize_grads = False``,
or ``ctx.set_materialize_grads(False)`` in ``forward``) receives ``None`` in
eager for an output no gradient reached.  The compiled region instead traces
``backward`` once, at ``apply``, with a cotangent tensor for every
differentiable output (``custom_backward._build_cotangents``), and the backward
walker binds zeros to the ones the loss leaves out (``VjpEmitter.mm``
``bind_vjp_region_cotangents``).  So a term eager skips runs on a zero
cotangent.  ``_SVD`` divides by ``s_j² - s_i²`` and, off the square, by ``s``;
``_Eigh`` by ``w_j - w_i``.  On a repeated or zero spectrum — an identity or
orthogonal initial weight — ``0 · inf`` makes the whole compiled gradient NaN
while eager's is finite.

The affected cases are ``xfail(strict=True, raises=AssertionError)``: they
fail on the NaN assertion, and anything else (an exception, a fallback to
eager under ``CompileFallbackWarning``) fails the test.  The controls — a
distinct spectrum, and ``qr(mode="r")``, whose zero-cotangent term is
``solve_triangular(R, 0) = 0`` for a nonsingular ``R`` — must stay right.
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.autograd import Function
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning

_LA = lucid.linalg

_OPS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "svd": lambda a: _LA.svd(a)[1],
    "svdvals": lambda a: _LA.svdvals(a),
    "matrix_norm_nuc": lambda a: _LA.matrix_norm(a, "nuc"),
    "matrix_norm_2": lambda a: _LA.matrix_norm(a, 2),
    "eigh_values": lambda a: _LA.eigh(a)[0],
    "eigvalsh": lambda a: _LA.eigvalsh(a),
    "qr_mode_r": lambda a: _LA.qr(a, mode="r")[1],
}

_MATRICES: dict[str, list[list[float]]] = {
    "eye": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "diag_3_3_1": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
    "zero": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    # The largest singular value is unique, the other two repeat.
    "diag_3_1_1": [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    # Distinct singular values, one of them zero: ``_SVD``'s ``1 / s`` term.
    "tall_rank_deficient": [[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
    "distinct": [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]],
}

_REASON = (
    "LCD-863: the compiled VJP region hands an output no gradient reached a zero "
    "cotangent where eager hands None; 0 * inf in the skipped term makes it NaN"
)
# ``matrix_norm(a, 2)`` is a ``max`` over the singular values: on a tied
# largest one the compiled ``max`` VJP sends the full gradient to every tie.
_REASON_TIED_MAX = _REASON + "; then LCD-865: compiled max VJP on a tie"

_AFFECTED = [
    pytest.param(op, mat, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=why))
    for op, mat, why in [
        *(
            (op, mat, _REASON)
            for op in ("svd", "svdvals", "matrix_norm_nuc", "eigh_values", "eigvalsh")
            for mat in ("eye", "diag_3_3_1", "zero")
        ),
        *(("matrix_norm_2", mat, _REASON_TIED_MAX) for mat in ("eye", "diag_3_3_1", "zero")),
        ("matrix_norm_2", "diag_3_1_1", _REASON),
        ("svdvals", "tall_rank_deficient", _REASON),
        ("matrix_norm_nuc", "tall_rank_deficient", _REASON),
    ]
]

_CONTROLS = [
    *((op, "distinct") for op in _OPS),
    ("qr_mode_r", "eye"),
    ("qr_mode_r", "diag_3_3_1"),
]


class _OfMatrix(nn.Module):
    def __init__(self, a: lucid.Tensor, op: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.a = nn.Parameter(a)
        self.op = op

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return (self.op(self.a) * x).sum().reshape(1)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


def _compiled_and_eager(
    build: Callable[[], nn.Module], param: Callable[[nn.Module], lucid.Tensor], device: str
) -> tuple[list[lucid.Tensor], lucid.Tensor]:
    """The parameter's gradient from two compiled steps, and from one eager step."""
    x = lucid.ones(1, device=device)
    t = lucid.ones(1, device=device)
    model = build()
    step = make_step(model, _loss)
    got: list[lucid.Tensor] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        for _ in range(2):
            param(model).grad = None
            step(x, t).backward()
            grad = param(model).grad
            assert grad is not None
            got.append(grad.cpu())
    assert not step.eager_only
    eager = build()
    _loss(eager(x), t).backward()
    want = param(eager).grad
    assert want is not None
    return got, want.cpu()


def _assert_matches_eager(got: list[lucid.Tensor], want: lucid.Tensor) -> None:
    for k, grad in enumerate(got):
        assert bool(lucid.isfinite(grad).all().item()), (
            f"step {k}: compiled gradient {grad.reshape(-1).tolist()} is not finite, "
            f"eager's is {want.reshape(-1).tolist()}"
        )
        assert bool(lucid.allclose(grad, want, rtol=1e-4, atol=1e-5)), (
            f"step {k}: compiled {grad.reshape(-1).tolist()} != eager {want.reshape(-1).tolist()}"
        )


def _linalg_case(op: str, mat: str, device: str) -> tuple[list[lucid.Tensor], lucid.Tensor]:
    def build() -> nn.Module:
        return _OfMatrix(lucid.tensor(_MATRICES[mat], device=device), _OPS[op])

    return _compiled_and_eager(build, lambda m: m.a, device)


@pytest.mark.parametrize(("op", "mat"), [p.values for p in _AFFECTED])
def test_eager_gradient_on_the_spectrum_is_finite(op: str, mat: str, device_gpu_only: str) -> None:
    """The premise of the xfails below: eager answers these with a finite gradient."""
    model = _OfMatrix(lucid.tensor(_MATRICES[mat], device=device_gpu_only), _OPS[op])
    x = lucid.ones(1, device=device_gpu_only)
    _loss(model(x), lucid.ones(1, device=device_gpu_only)).backward()
    assert model.a.grad is not None
    assert bool(lucid.isfinite(model.a.grad).all().item())


@pytest.mark.parametrize(("op", "mat"), _AFFECTED)
def test_compiled_gradient_matches_eager_on_a_degenerate_spectrum(
    op: str, mat: str, device_gpu_only: str
) -> None:
    got, want = _linalg_case(op, mat, device_gpu_only)
    _assert_matches_eager(got, want)


@pytest.mark.parametrize(("op", "mat"), _CONTROLS)
def test_compiled_gradient_matches_eager_on_a_regular_spectrum(
    op: str, mat: str, device_gpu_only: str
) -> None:
    got, want = _linalg_case(op, mat, device_gpu_only)
    _assert_matches_eager(got, want)


class _ClassFlag(Function):
    """Does not materialize grads by class attribute; the second output goes unused."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        (a,) = ctx.saved_tensors
        grad = g_first * 2.0
        if g_second is not None:
            # Finite only while no gradient reaches the second output.
            grad = grad + g_second / (a - a)
        return grad


class _CtxFlagUnused(Function):
    """Does not materialize grads by ``ctx``; the second output goes unused."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        (a,) = ctx.saved_tensors
        grad = g_first * 2.0
        if g_second is not None:
            grad = grad + g_second / (a - a)
        return grad


class _CtxFlagMarked(Function):
    """Does not materialize grads by ``ctx``; the second output is non-differentiable.

    ``_run_vjp_backward`` reads only the class attribute, so it turns the
    marked output's ``None`` into zeros where eager keeps ``None``.
    """

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(a)
        second = a * 3.0
        ctx.mark_non_differentiable(second)
        return a * 2.0, second

    @staticmethod
    def backward(ctx, g_first, g_second):
        (a,) = ctx.saved_tensors
        grad = g_first * 2.0
        if g_second is not None:
            grad = grad + g_second / (a - a)
        return grad


class _FirstOutputOnly(nn.Module):
    def __init__(self, fn: type[Function], device: str) -> None:
        super().__init__()
        self.p = nn.Parameter(lucid.ones(1, device=device))
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        first, _ = self.fn.apply(self.p * x)
        return first


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_REASON)
@pytest.mark.parametrize("fn", [_ClassFlag, _CtxFlagUnused, _CtxFlagMarked])
def test_unreached_output_of_a_non_materializing_function_gets_none(
    fn: type[Function], device_gpu_only: str
) -> None:
    got, want = _compiled_and_eager(
        lambda: _FirstOutputOnly(fn, device_gpu_only), lambda m: m.p, device_gpu_only
    )
    assert bool(lucid.isfinite(want).all().item())
    _assert_matches_eager(got, want)
