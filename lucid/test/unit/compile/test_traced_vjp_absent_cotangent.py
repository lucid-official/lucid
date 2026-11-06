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
The spectrum and demand cases must compile — a fallback to eager under
``CompileFallbackWarning`` fails them — and match eager.  The controls — a
distinct spectrum, and ``qr(mode="r")``, whose zero-cotangent term is
``solve_triangular(R, 0) = 0`` for a nonsingular ``R`` — were right before.
A call that cannot be traced for the demands it may meet runs eagerly
instead, with the notice saying why.
"""

import itertools
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


@pytest.fixture(autouse=True)
def _clear_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``LUCID_COMPILE_VERBOSE`` left by earlier matrix runs does not suppress warnings."""
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)


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
# largest one it also holds the compiled ``max`` VJP to eager's even split
# among the ties (LCD-865, ``test_extremum_tie_gradient.py``).
_AFFECTED = [
    *(
        pytest.param(op, mat)
        for op in ("svd", "svdvals", "matrix_norm_nuc", "matrix_norm_2", "eigh_values", "eigvalsh")
        for mat in ("eye", "diag_3_3_1", "zero")
    ),
    pytest.param("matrix_norm_2", "diag_3_1_1"),
    pytest.param("svdvals", "tall_rank_deficient"),
    pytest.param("matrix_norm_nuc", "tall_rank_deficient"),
]

_CONTROLS = [*((op, "distinct") for op in _OPS), ("qr_mode_r", "eye"), ("qr_mode_r", "diag_3_3_1")]


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
        assert bool(
            lucid.allclose(grad, want, rtol=1e-4, atol=1e-5)
        ), f"step {k}: compiled {grad.reshape(-1).tolist()} != eager {want.reshape(-1).tolist()}"


def _linalg_case(op: str, mat: str, device: str) -> tuple[list[lucid.Tensor], lucid.Tensor]:
    def build() -> nn.Module:
        return _OfMatrix(lucid.tensor(_MATRICES[mat], device=device), _OPS[op])

    return _compiled_and_eager(build, lambda m: m.a, device)


@pytest.mark.parametrize(("op", "mat"), [p.values for p in _AFFECTED])
def test_eager_gradient_on_the_spectrum_is_finite(op: str, mat: str, device_gpu_only: str) -> None:
    """The premise of the cases below: eager answers these with a finite gradient."""
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


class _AssertsFirst(Function):
    """Asserts on the gradient it is written for: the demand without it is not traced."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        assert g_first is not None
        return _finite_while_only_the_first_is_reached(ctx, g_first, g_second)


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


class _OffMetalUnlessEveryOutput(Function):
    """Traces only for a gradient on every output: its path for ``None`` leaves Metal."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        (a,) = ctx.saved_tensors
        if g_first is None or g_second is None:
            g = g_first if g_first is not None else g_second
            return (g.cpu() * 2.0).to(a.device)
        return g_first * 2.0 + g_second / (a - a)


class _EveryOutputOrRaise(Function):
    """Written for a gradient on every output: eager raises on any other demand."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        return g_first * 2.0 + g_second * 3.0


def _off_metal(a, g):
    return (g.cpu() * 2.0).to(a.device)


class _NeverOnMetal(Function):
    """Does not materialize grads; ``backward`` leaves Metal for every demand."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        (a,) = ctx.saved_tensors
        return _off_metal(a, g_first if g_first is not None else g_second)


class _MaterializingOffMetal(Function):
    """Materializes grads; its one region leaves Metal."""

    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        (a,) = ctx.saved_tensors
        return _off_metal(a, g_first) + g_second * 0.0


class _GradOnlyWithoutTheSecond(Function):
    """Gives its input a gradient only when the second output gets none.

    The full variant then gives none, and the walker's dependency walk,
    which reads it, would never look at the group: no gradient, no error.
    """

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        return g_first * 2.0 if g_second is None else None


class _OffMetalOnlyWithoutTheSecond(Function):
    """As above, but the demand that gives the gradient is the one that cannot trace."""

    _materialize_grads = False
    _traced_backward = True

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a * 2.0, a * 3.0

    @staticmethod
    def backward(ctx, g_first, g_second):
        (a,) = ctx.saved_tensors
        if g_first is not None and g_second is None:
            return _off_metal(a, g_first)
        return None


class _FirstOutputOnly(nn.Module):
    def __init__(self, fn: type[Function], device: str) -> None:
        super().__init__()
        self.p = nn.Parameter(lucid.ones(1, device=device))
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn.apply(self.p * x)[0]


@pytest.mark.parametrize(
    "fn", [_ClassFlag, _CtxFlagUnused, _CtxFlagMarked, _FourOutputs, _AssertsFirst]
)
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
    assert len({(int(r.fwd_begin), int(r.fwd_end)) for r in regions}) <= 1
    # Every variant answers for the input: the full one, last, covers them all.
    assert all(int(r.input_grad_ids[0]) >= 0 for r in regions)
    return [tuple(j for j, cid in enumerate(r.cotangent_ids) if int(cid) >= 0) for r in regions]


@pytest.mark.parametrize(
    ("call", "present"),
    [
        (_LA.svd, [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]),
        (_LA.qr, [(0,), (1,), (0, 1)]),
        (_LA.eigh, [(0,), (1,), (0, 1)]),
        # ``backward`` cannot take ``None`` for the first output: the demand
        # without it is not traced, and no region of the group matches it.
        (_CtxFlagUnused.apply, [(0,), (0, 1)]),
        (_AssertsFirst.apply, [(0,), (0, 1)]),
        # No demand short of every output traces.  The full region alone
        # would read as a call that materializes grads, so nothing is kept.
        (_OffMetalUnlessEveryOutput.apply, []),
        (_EveryOutputOrRaise.apply, []),
        # One differentiable output, or zeros for the rest: one region.
        (_CtxFlagMarked.apply, [(0,)]),
        (_Materializing.apply, [(0, 1)]),
    ],
    ids=[
        "svd",
        "qr",
        "eigh",
        "ctx_flag",
        "asserts",
        "off_metal",
        "every_or_raise",
        "ctx_flag_marked",
        "materializing",
    ],
)
def test_a_call_commits_one_region_per_demand_the_full_set_last(
    call: Callable[[lucid.Tensor], object], present: list[tuple[int, ...]], device_gpu_only: str
) -> None:
    assert _present_sets(call, device_gpu_only) == present


def test_a_function_with_too_many_outputs_to_trace_runs_eagerly(device_gpu_only: str) -> None:
    """Past four outputs (31 variants and more) the call is not traced."""
    _assert_runs_eagerly_as_eager(_FiveOutputs, "has 5 differentiable outputs", device_gpu_only)


@pytest.mark.parametrize(
    ("fn", "why"),
    [
        # A lone full region would get zeros for the unreached output: 0 / 0.
        (_OffMetalUnlessEveryOutput, "only for a gradient on every output"),
        (_NeverOnMetal, "tracing it for every output failed: it recorded an op"),
        (_MaterializingOffMetal, "tracing it failed: it recorded an op"),
        # The walker would not see the demand that gives the gradient.
        (_GradOnlyWithoutTheSecond, "for a gradient on some of its outputs it gives an input"),
        (_OffMetalOnlyWithoutTheSecond, "for a gradient on every output it gives an input none"),
    ],
    ids=["every_output_only", "never", "materializing", "grad_only_partial", "left_out_grad"],
)
def test_an_untraced_call_runs_eagerly_and_says_why(
    fn: type[Function], why: str, device_gpu_only: str
) -> None:
    _assert_runs_eagerly_as_eager(fn, why, device_gpu_only)


def _assert_runs_eagerly_as_eager(fn: type[Function], why: str, device: str) -> None:
    x = lucid.ones(1, device=device)
    t = lucid.ones(1, device=device)
    model = _FirstOutputOnly(fn, device)
    step = make_step(model, _loss)
    _NOTICED.discard("custom-backward")
    match = rf"{fn.__name__}'s backward could not be traced into the compiled step: .*{why}"
    with pytest.warns(CompileFallbackWarning, match=match):
        step(x, t).backward()
    assert len(step.eager_only) == 1
    eager = _FirstOutputOnly(fn, device)
    _loss(eager(x), t).backward()
    assert model.p.grad is not None and eager.p.grad is not None
    assert bool(lucid.isfinite(eager.p.grad).all().item())
    _assert_matches_eager([model.p.grad.cpu()], eager.p.grad.cpu())


def test_a_demand_eager_refuses_is_not_compiled(device_gpu_only: str) -> None:
    """No compiled gradient where eager's ``backward`` raises: the step runs it eagerly."""
    x = lucid.ones(1, device=device_gpu_only)
    t = lucid.ones(1, device=device_gpu_only)
    eager = _FirstOutputOnly(_EveryOutputOrRaise, device_gpu_only)
    with pytest.raises(TypeError):
        _loss(eager(x), t).backward()
    model = _FirstOutputOnly(_EveryOutputOrRaise, device_gpu_only)
    step = make_step(model, _loss)
    _NOTICED.discard("custom-backward")
    with pytest.warns(CompileFallbackWarning, match="_EveryOutputOrRaise's backward"):
        loss = step(x, t)
    with pytest.raises(TypeError):
        loss.backward()
    assert len(step.eager_only) == 1


def test_a_backward_that_fails_to_trace_is_narrated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], device_gpu_only: str
) -> None:
    """Why a demand variant was left out shows under ``LUCID_COMPILE_VERBOSE=1``."""
    monkeypatch.setenv("LUCID_COMPILE_VERBOSE", "1")
    assert _present_sets(_AssertsFirst.apply, device_gpu_only) == [(0,), (0, 1)]
    assert (
        "tracing _AssertsFirst.backward(None, a cotangent) failed: it raised AssertionError"
        in capsys.readouterr().err
    )


_W_MATRIX = [[1.0, 2.0, 3.0], [0.5, -1.0, 2.5], [1.5, 0.25, -0.5]]
_W_VECTOR = [1.0, -2.0, 0.5]

_FACTORIZATIONS: dict[str, tuple[Callable[[lucid.Tensor], tuple[lucid.Tensor, ...]], int]] = {
    "svd": (lambda a: tuple(_LA.svd(a)), 3),
    "qr": (lambda a: tuple(_LA.qr(a)), 2),
    "eigh": (lambda a: tuple(_LA.eigh(a)), 2),
}


class _OfDemand(nn.Module):
    """A weighted sum of squares of the factors in ``demand``, blind to their column signs."""

    def __init__(
        self,
        a: lucid.Tensor,
        factorize: Callable[[lucid.Tensor], tuple[lucid.Tensor, ...]],
        demand: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.a = nn.Parameter(a)
        self.factorize = factorize
        self.demand = demand
        self.register_buffer("w_matrix", lucid.tensor(_W_MATRIX, device=a.device))
        self.register_buffer("w_vector", lucid.tensor(_W_VECTOR, device=a.device))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        factors = self.factorize(self.a)
        terms = [
            (factors[j] ** 2 * (self.w_matrix if factors[j].ndim == 2 else self.w_vector)).sum()
            for j in self.demand
        ]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return (total * x).sum().reshape(1)


@pytest.mark.parametrize(
    ("name", "demand"),
    [
        pytest.param(name, demand, id=f"{name}-{''.join(map(str, demand))}")
        for name, (_, n) in _FACTORIZATIONS.items()
        for size in range(1, n + 1)
        for demand in itertools.combinations(range(n), size)
    ],
)
def test_every_demand_gets_its_own_variant(
    name: str, demand: tuple[int, ...], device_gpu_only: str
) -> None:
    """A variant for another demand of the same size would drop or add a factor's term."""
    factorize, _ = _FACTORIZATIONS[name]

    def build() -> nn.Module:
        return _OfDemand(
            lucid.tensor(_MATRICES["distinct"], device=device_gpu_only), factorize, demand
        )

    got, want = _compiled_and_eager(build, lambda m: m.a, device_gpu_only)
    _assert_matches_eager(got, want)


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
