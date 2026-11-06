"""A traced custom VJP hands ``backward`` the cotangents eager hands it (LCD-863).

A ``Function`` that does not materialize grads (``_materialize_grads = False``,
or ``ctx.set_materialize_grads(False)`` in ``forward``) receives ``None`` in
eager for an output no gradient reached.  The compiled region used to trace
``backward`` once, at ``apply``, with a cotangent tensor for every
differentiable output, and the backward walker bound zeros to the ones the
loss left out.  So a term eager skips ran on a zero cotangent.  ``_SVD``
divides by ``s_j² - s_i²`` and, off the square, by ``s``; ``_Eigh`` by
``w_j - w_i``.  On a repeated or zero spectrum — an identity or orthogonal
initial weight — ``0 · inf`` made the whole compiled gradient NaN while
eager's was finite.

Such a call now commits one region per subset of its outputs a gradient may
reach (``custom_backward._trace_demand_variants``), and the walker emits the
one for the outputs that do (``VjpEmitter.mm`` ``emit_vjp_region_group``).
Every case here must compile — a fallback to eager under
``CompileFallbackWarning`` fails it — and match eager.  The controls — a
distinct spectrum, and ``qr(mode="r")``, whose zero-cotangent term is
``solve_triangular(R, 0) = 0`` for a nonsingular ``R`` — were right before.
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.autograd import Function
from lucid.compile import make_step
from lucid.compile._core import custom_backward
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.compile._core.trace_context import recording

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

# ``matrix_norm(a, 2)`` is a ``max`` over the singular values: on a tied
# largest one the compiled ``max`` VJP sends the full gradient to every tie.
_TIED_MAX = pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="LCD-865: compiled max VJP on a tie"
)

_AFFECTED = [
    *(
        pytest.param(op, mat)
        for op in ("svd", "svdvals", "matrix_norm_nuc", "eigh_values", "eigvalsh")
        for mat in ("eye", "diag_3_3_1", "zero")
    ),
    *(pytest.param("matrix_norm_2", mat, marks=_TIED_MAX) for mat in ("eye", "diag_3_3_1", "zero")),
    pytest.param("matrix_norm_2", "diag_3_1_1"),
    pytest.param("svdvals", "tall_rank_deficient"),
    pytest.param("matrix_norm_nuc", "tall_rank_deficient"),
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

    ``_run_vjp_backward`` used to read only the class attribute, so it turned
    the marked output's ``None`` into zeros where eager keeps ``None``.
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


def _finite_while_only_the_first_is_reached(ctx, *grads):
    (a,) = ctx.saved_tensors
    grad = grads[0] * 2.0
    for g in grads[1:]:
        if g is not None:
            grad = grad + g / (a - a)
    return grad


class _FourOutputs(Function):
    """As many differentiable outputs as a call is traced per demand for."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0, a * 4.0, a * 5.0

    backward = staticmethod(_finite_while_only_the_first_is_reached)


class _FiveOutputs(Function):
    """One output past the limit: 31 variants are not traced, the step runs eagerly."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0, a * 4.0, a * 5.0, a * 6.0

    backward = staticmethod(_finite_while_only_the_first_is_reached)


class _Materializing(Function):
    """Materializes grads: the unused output gets zeros, in eager and compiled alike."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        return g_first * 2.0 + g_second * 3.0


class _FirstOutputOnly(nn.Module):
    def __init__(self, fn: type[Function], device: str) -> None:
        super().__init__()
        self.p = nn.Parameter(lucid.ones(1, device=device))
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn.apply(self.p * x)[0]


@pytest.mark.parametrize("fn", [_ClassFlag, _CtxFlagUnused, _CtxFlagMarked, _FourOutputs])
def test_unreached_output_of_a_non_materializing_function_gets_none(
    fn: type[Function], device_gpu_only: str
) -> None:
    got, want = _compiled_and_eager(
        lambda: _FirstOutputOnly(fn, device_gpu_only), lambda m: m.p, device_gpu_only
    )
    assert bool(lucid.isfinite(want).all().item())
    _assert_matches_eager(got, want)


def _present_sets(call: Callable[[lucid.Tensor], object], device: str) -> list[tuple[int, ...]]:
    """Per region ``call`` commits, the outputs it was traced with a cotangent for."""
    a = lucid.tensor(_MATRICES["distinct"], device=device, requires_grad=True)
    with recording(watch=(a,)) as tracer:
        call(a)
    regions = tracer.graph.vjp_regions
    assert len({(int(r.fwd_begin), int(r.fwd_end)) for r in regions}) == 1
    # Every variant answers for the input: the full one, last, covers them all.
    assert all(int(r.input_grad_ids[0]) >= 0 for r in regions)
    return [tuple(j for j, cid in enumerate(r.cotangent_ids) if int(cid) >= 0) for r in regions]


@pytest.mark.parametrize(
    ("call", "present"),
    [
        (_LA.svd, [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]),
        (_LA.qr, [(0,), (1,), (0, 1)]),
        (_LA.eigh, [(0,), (1,), (0, 1)]),
        (_CtxFlagUnused.apply, [(0,), (1,), (0, 1)]),
        # One differentiable output, or zeros for the rest: one region.
        (_CtxFlagMarked.apply, [(0,)]),
        (_Materializing.apply, [(0, 1)]),
    ],
    ids=["svd", "qr", "eigh", "ctx_flag", "ctx_flag_marked", "materializing"],
)
def test_a_call_commits_one_region_per_demand_the_full_set_last(
    call: Callable[[lucid.Tensor], object], present: list[tuple[int, ...]], device_gpu_only: str
) -> None:
    assert _present_sets(call, device_gpu_only) == present


def test_a_function_with_too_many_outputs_to_trace_runs_eagerly(device_gpu_only: str) -> None:
    x = lucid.ones(1, device=device_gpu_only)
    t = lucid.ones(1, device=device_gpu_only)
    model = _FirstOutputOnly(_FiveOutputs, device_gpu_only)
    step = make_step(model, _loss)
    _NOTICED.discard("custom-backward")
    with pytest.warns(CompileFallbackWarning, match="_FiveOutputs"):
        step(x, t).backward()
    assert len(step.eager_only) == 1
    eager = _FirstOutputOnly(_FiveOutputs, device_gpu_only)
    _loss(eager(x), t).backward()
    assert model.p.grad is not None and eager.p.grad is not None
    _assert_matches_eager([model.p.grad.cpu()], eager.p.grad.cpu())


def test_a_demand_no_region_was_traced_for_is_refused_loudly(
    monkeypatch: pytest.MonkeyPatch, device_gpu_only: str
) -> None:
    """The walker never stands in another variant: it fails and the step runs eagerly."""
    every = custom_backward._demand_subsets
    monkeypatch.setattr(
        custom_backward, "_demand_subsets", lambda n: [s for s in every(n) if s != (1,)]
    )
    x = lucid.ones(1, device=device_gpu_only)
    t = lucid.ones(1, device=device_gpu_only)

    def build() -> _OfMatrix:
        return _OfMatrix(lucid.tensor(_MATRICES["eye"], device=device_gpu_only), _OPS["svdvals"])

    model = build()
    step = make_step(model, _loss)
    _NOTICED.discard("lowering")
    match = (
        r"no traced VJP variant of the custom Function at ops \[\d+, \d+\) for the outputs "
        r"that receive a gradient \{1\} \(traced: \{0\} \{2\} \{0, 1\} \{0, 2\} \{1, 2\} "
        r"\{0, 1, 2\}\)"
    )
    with pytest.warns(CompileFallbackWarning, match=match):
        step(x, t).backward()
    assert len(step.eager_only) == 1
    eager = build()
    _loss(eager(x), t).backward()
    assert model.a.grad is not None and eager.a.grad is not None
    _assert_matches_eager([model.a.grad.cpu()], eager.a.grad.cpu())
