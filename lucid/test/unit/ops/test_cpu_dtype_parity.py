"""Regression tests: the CPU backend refused dtypes Metal accepted.

Found 2026-08-02 by the audit's ``dtype`` axis, which runs every op over
every dtype on both devices and compares the two sets.

Three separate gaps, and one of them was not a refusal at all:

**Unary on int64 returned its own input.**  ``unary_op``'s I64 branch read

    op[i] = (std::int64_t)(double)(double)ip[i];

which is an identity — the function was never applied.  int64 is Lucid's
default integer dtype, so ``lucid.exp(lucid.tensor([2, 3]))`` answered
``[2, 3]`` instead of ``[7, 20]``, silently, on the CPU only, while Metal
was correct.  Of the defects this audit has found, this is the one that
produced a plausible wrong number rather than an error.

**Unary refused bool / int8 / int16 / float16.**  Accelerate has no
vector maths below 32 bits and no half accumulator, so those dtypes need
the widen-reuse-narrow shape ``binary_op`` already used.

**Reductions and convolutions refused the same set.**  ``reduce_one_axis``
handled F32 and F64 only; ``conv_nd_forward`` handled F32 and F64 only.

The tests below assert cpu/metal agreement rather than hand-written
values wherever both devices can run the op, because a hand-written
expected value encodes a convention rather than checking one.  float64 is
excluded from the comparison: it does not exist on Metal, and the engine
documents the downcast.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg
import lucid.nn.functional as F

_DEVICES = ["cpu", "metal"]

#: Every dtype that exists on both devices.
_SHARED = [
    ("bool", np.bool_),
    ("int8", np.int8),
    ("int16", np.int16),
    ("int32", np.int32),
    ("int64", np.int64),
    ("float16", np.float16),
    ("float32", np.float32),
]

#: Ops whose answer is a real number whatever the input was, so an
#: integer input is promoted rather than truncated.
_REAL_VALUED = [
    "exp",
    "log",
    "sqrt",
    "tanh",
    "sigmoid",
    "reciprocal",
    "rsqrt",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "erf",
]

_UNARY = [
    ("exp", lucid.exp),
    ("tanh", lucid.tanh),
    ("sin", lucid.sin),
    ("abs", lucid.abs),
    ("sqrt", lucid.sqrt),
]


def _tensor(values, dtype_name, np_dtype, device="cpu"):
    return lucid.tensor(
        np.ascontiguousarray(np.asarray(values).astype(np_dtype)),
        dtype=getattr(lucid, dtype_name),
        device=device,
    )


# ── the silent wrong answer ──────────────────────────────────────────────────


def test_unary_on_int64_actually_applies_the_function() -> None:
    """The defect: every unary op returned its own input on int64.

    The assertion used to read ``exp([2, 3]) == [7, 20]`` — the function
    was applied and then truncated back to int64.  That was enough to
    catch the identity bug it was written for, and wrong about the answer:
    the value of ``exp(2)`` is not an integer, and the reference framework
    promotes an integer input to float32 rather than rounding it.  See
    :func:`test_real_valued_ops_promote_integers`.
    """
    x = lucid.tensor(np.array([2, 3]))  # int64 by default
    assert str(x.dtype).endswith("int64")
    got = lucid.exp(x).numpy()
    assert not np.array_equal(got, [2, 3]), "exp(int64) returned its input"
    assert np.allclose(got, [7.389056, 20.085537], rtol=1e-5)


@pytest.mark.parametrize("name", _REAL_VALUED)
@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_real_valued_ops_promote_integers(name: str, dtype_name: str) -> None:
    """``exp(1)`` is 2.718, not 2, and not an error.

    Lucid had this split two ways: ``exp``/``log``/``sqrt``/``tanh``
    computed in the integer type and truncated, while
    ``sigmoid``/``reciprocal``/``rsqrt``/``erfinv`` raised
    ``NotImplementedError`` for want of an integer kernel.  Neither
    matches the reference, which promotes to float32 for all of them.

    Carried by ``OpSchema::real_valued`` rather than by the AMP policy:
    ``AmpPolicy::Promote`` is also worn by ``matmul``, whose integer
    answer really is an integer.
    """
    x = lucid.tensor(np.array([[1, 2], [3, 4]]), dtype=getattr(lucid, dtype_name))
    out = getattr(lucid, name)(x)
    assert str(out.dtype).endswith("float32"), f"{name} did not promote"
    assert np.isfinite(out.numpy()).all()


@pytest.mark.parametrize("name", _REAL_VALUED)
def test_promotion_agrees_with_the_float_answer(name: str) -> None:
    """Guard the instrument: promoting has to compute, not just re-label."""
    values = np.array([[1, 2], [3, 4]])
    from_int = getattr(lucid, name)(lucid.tensor(values, dtype=lucid.int32)).numpy()
    from_float = getattr(lucid, name)(
        lucid.tensor(values.astype(np.float32), dtype=lucid.float32)
    ).numpy()
    assert np.allclose(from_int, from_float, rtol=1e-6), name


def test_matmul_keeps_its_integers() -> None:
    """The reason ``real_valued`` is not inferred from the AMP policy.

    ``matmul`` is ``AmpPolicy::Promote`` too, and the product of two
    integer matrices is an integer matrix.
    """
    a = lucid.tensor(np.array([[1, 2], [3, 4]]), dtype=lucid.int32)
    out = lucid.matmul(a, a)
    assert str(out.dtype).endswith("int32"), "matmul promoted when it should not"
    assert np.array_equal(out.numpy(), [[7, 10], [15, 22]])


@pytest.mark.parametrize("name,fn", _UNARY)
def test_int64_matches_int32(name: str, fn) -> None:
    """Guard the instrument.

    The I32 path was always correct, so if I64 now disagrees with it the
    fix is wrong in a different way.  Both truncate towards zero, so the
    two must land on the same integers.
    """
    values = [1, 2, 3, 4]
    got64 = fn(_tensor(values, "int64", np.int64)).numpy()
    got32 = fn(_tensor(values, "int32", np.int32)).numpy()
    assert np.array_equal(got64, got32), name


# ── dtype parity ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype_name,np_dtype", _SHARED)
@pytest.mark.parametrize("name,fn", _UNARY)
def test_unary_accepts_every_shared_dtype_on_the_cpu(
    name: str, fn, dtype_name: str, np_dtype
) -> None:
    """Accepted, but not necessarily in the dtype it was handed.

    ``abs`` keeps an integer integer; ``exp`` does not, because its answer
    is not one.  The assertion here is that every dtype is *accepted* —
    which is what the CPU used to fail at — with the promotion rule
    checked separately in :func:`test_real_valued_ops_promote_integers`.
    """
    out = fn(_tensor(np.ones((2, 3)), dtype_name, np_dtype))
    if name in _REAL_VALUED and dtype_name.startswith(("int", "bool")):
        assert str(out.dtype).endswith("float32"), f"{name} did not promote"
    else:
        assert str(out.dtype).endswith(dtype_name), f"{name} changed dtype"


@pytest.mark.parametrize("dtype_name,np_dtype", _SHARED)
def test_unary_agrees_across_devices(dtype_name: str, np_dtype) -> None:
    """The comparison the old sweeps could not make, because the CPU raised."""
    values = np.ones((2, 3)) * 2
    cpu_out = lucid.tanh(_tensor(values, dtype_name, np_dtype, "cpu"))
    metal_out = lucid.tanh(_tensor(values, dtype_name, np_dtype, "metal"))
    assert str(cpu_out.device) == "device('cpu')"
    assert str(metal_out.device) == "device('metal')"
    assert np.allclose(
        cpu_out.numpy().astype(np.float64),
        metal_out.numpy().astype(np.float64),
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
    ), dtype_name


@pytest.mark.parametrize("dtype_name,np_dtype", _SHARED)
def test_prod_accepts_every_shared_dtype(dtype_name: str, np_dtype) -> None:
    """``reduce_one_axis`` handled F32 and F64 only."""
    out = lucid.prod(_tensor(np.ones((2, 3)) * 2, dtype_name, np_dtype))
    value = float(np.asarray(out.numpy()).reshape(-1)[0])
    assert value == (1.0 if dtype_name == "bool" else 64.0), dtype_name


@pytest.mark.parametrize("device", _DEVICES)
def test_conv2d_accepts_float16(device: str) -> None:
    """``conv_nd_forward`` handled F32 and F64 only; Metal took F16."""
    x = _tensor(np.random.default_rng(0).random((1, 2, 4, 4)), "float16", np.float16, device)
    w = _tensor(np.random.default_rng(1).random((2, 2, 3, 3)), "float16", np.float16, device)
    out = F.conv2d(x, w, None, padding=1)
    assert out.shape == (1, 2, 4, 4)
    assert str(out.dtype).endswith("float16")


def test_conv2d_float16_matches_float32_closely() -> None:
    """Guard the instrument: accepting F16 is not enough, it has to compute.

    The half path widens to float, convolves, and rounds once — so it must
    land near the float32 answer rather than merely returning a
    correctly-shaped buffer.
    """
    rng = np.random.default_rng(3)
    x_values = rng.random((1, 2, 4, 4)) - 0.5
    w_values = rng.random((2, 2, 3, 3)) - 0.5
    half = F.conv2d(
        _tensor(x_values, "float16", np.float16),
        _tensor(w_values, "float16", np.float16),
        None,
        padding=1,
    ).numpy()
    single = F.conv2d(
        _tensor(x_values, "float32", np.float32),
        _tensor(w_values, "float32", np.float32),
        None,
        padding=1,
    ).numpy()
    assert np.allclose(half.astype(np.float64), single.astype(np.float64), atol=5e-3)
    assert np.abs(half).max() > 0.0, "the half path returned an empty buffer"


def test_half_conversion_has_one_home() -> None:
    """The helpers were duplicated to reach the reduction kernels; they are not.

    Two copies of IEEE rounding code is how two copies quietly disagree.
    """
    from pathlib import Path

    root = Path(lucid.__file__).parent / "_C"
    definitions = [
        path for path in root.rglob("*.h") if "inline float half_bits_to_float" in path.read_text()
    ]
    assert len(definitions) == 1, [str(p) for p in definitions]
    assert definitions[0].name == "Half.h"


# ── memory safety ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_scatter_add_reads_its_index_at_the_right_width(dtype_name: str) -> None:
    """The worst of the set: a bus error reachable from Python.

    The index buffer was read as ``int32`` whatever its dtype.  An int16
    index had two of its values read as one — a silently wrong result —
    and an int8 index had four read as one, producing an index far outside
    the base and an out-of-bounds write.  ``pytest -m audit`` took the
    whole process down on ``dtype-lucid.scatter_add``.  An int8 / int16
    index is now refused at the op's door, as ``gather`` and ``scatter``
    refuse it (LCD-288); int32 and int64 are read at their own width.
    """
    base = lucid.tensor(np.zeros((2, 4), dtype=np.float32))
    index = lucid.tensor(np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), dtype=getattr(lucid, dtype_name))
    source = lucid.tensor(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], np.float32))
    if dtype_name in ("int8", "int16"):
        with pytest.raises((TypeError, IndexError), match="index"):
            lucid.scatter_add(base, 1, index, source)
        return
    got = lucid.scatter_add(base, 1, index, source).numpy()
    assert np.allclose(got, [[1.0, 2.0, 3.0, 4.0], [8.0, 7.0, 6.0, 5.0]]), dtype_name


def test_scatter_add_refuses_an_out_of_range_index() -> None:
    """It used to write outside the buffer instead."""
    base = lucid.tensor(np.zeros((2, 4), dtype=np.float32))
    index = lucid.tensor(np.array([[99, 0, 0, 0], [0, 0, 0, 0]]), dtype=lucid.int64)
    source = lucid.tensor(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(Exception, match="out of range"):
        lucid.scatter_add(base, 1, index, source)


def test_scatter_add_still_scatters() -> None:
    """Guard the instrument: the bounds check must not have disabled the op."""
    base = lucid.tensor(np.zeros((1, 3), dtype=np.float32))
    index = lucid.tensor(np.array([[1, 1, 1]]), dtype=lucid.int64)
    source = lucid.tensor(np.array([[1.0, 2.0, 3.0]], np.float32))
    got = lucid.scatter_add(base, 1, index, source).numpy()
    assert np.allclose(got, [[0.0, 6.0, 0.0]])


# ── lu_solve pivots ──────────────────────────────────────────────────────────
# Found 2026-08-03: the full audit sweep did not finish, it died. Exit 138 is
# SIGBUS, and the last symbol it printed was the one before this op.


def _pivoting_system() -> tuple[lucid.Tensor, lucid.Tensor]:
    """A system whose factorization genuinely permutes rows.

    A near-identity matrix pivots trivially, and every wrong pivot width
    happened to give the right answer on one — which is how this looked
    fine until the probe used a matrix with a tiny first row.
    """
    rng = np.random.default_rng(0)
    values = rng.random((5, 5))
    values[0] *= 1e-6
    return (
        lucid.tensor(values, dtype=lucid.float64),
        lucid.tensor(rng.random((5, 1)), dtype=lucid.float64),
    )


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_lu_solve_agrees_with_solve_at_every_pivot_width(dtype_name: str) -> None:
    """LAPACK reads ``ipiv`` as ``const int*``; nothing converted to it.

    int8 and bool were read past the end of their own buffer and took the
    process down with SIGBUS.  int16 and int64 were long enough to
    survive the over-read and *returned*: int16 answered ``5.9e+133`` and
    int64 answered a plausible vector that was not the solution.  Only
    int32 — the width ``lu_factor`` happens to emit — was correct.
    """
    a, b = _pivoting_system()
    reference = lucid.linalg.solve(a, b).numpy().ravel()
    lu, pivots = lucid.linalg.lu_factor(a)
    got = lucid.linalg.lu_solve(lu, pivots.to(getattr(lucid, dtype_name)), b).numpy().ravel()
    assert np.allclose(got, reference, rtol=1e-9), dtype_name


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "bool"])
def test_lu_solve_refuses_non_integer_pivots(dtype_name: str) -> None:
    """A float pivot vector had its mantissa bits read as row indices."""
    a, b = _pivoting_system()
    lu, pivots = lucid.linalg.lu_factor(a)
    with pytest.raises(TypeError, match="must be an integer tensor"):
        lucid.linalg.lu_solve(lu, pivots.to(getattr(lucid, dtype_name)), b)


def test_lu_solve_refuses_an_out_of_range_pivot() -> None:
    """A pivot of 99 on a 4x4 was accepted, and answered."""
    a = lucid.tensor(np.eye(4) + 0.1, dtype=lucid.float64)
    b = lucid.tensor(np.ones((4, 1)), dtype=lucid.float64)
    bad = lucid.tensor(np.array([99, 2, 3, 4]), dtype=lucid.int32)
    with pytest.raises(IndexError, match="out of range"):
        lucid.linalg.lu_solve(a, bad, b)


def test_gather_refuses_an_index_outside_the_axis() -> None:
    """The read-side twin of the scatter_add defect.

    ``gather`` computed a flat offset from the index and ``memcpy``'d from
    it with nothing in between.  An index of 99 into a 2x3 returned
    ``0.0`` read from past the allocation — a wrong number, silently —
    and gathering along an axis of extent zero segfaulted.  The audit's
    ``edge`` axis substitutes an empty tensor for the operand while
    leaving the index alone, which is exactly that second case, and the
    sweep died on ``edge-Tensor.gather`` every run.
    """
    base = lucid.tensor(np.ones((2, 3)), dtype=lucid.float64)
    index = lucid.tensor(np.full((2, 3), 99), dtype=lucid.int64)
    with pytest.raises(Exception, match="out of range"):
        base.gather(0, index)


def test_gather_refuses_an_empty_axis() -> None:
    """Gathering from nothing is not zero — there is no row to read."""
    empty = lucid.tensor(np.zeros((0, 5)), dtype=lucid.float64)
    index = lucid.tensor(np.zeros((4, 5), dtype=np.int64), dtype=lucid.int64)
    with pytest.raises(Exception, match="out of range"):
        empty.gather(0, index)


def test_gather_still_gathers() -> None:
    """Guard the instrument: the bounds check must not have disabled the op."""
    base = lucid.tensor(np.arange(6.0).reshape(2, 3), dtype=lucid.float64)
    index = lucid.tensor(np.array([[1, 0, 1], [0, 1, 0]]), dtype=lucid.int64)
    got = base.gather(0, index).numpy()
    assert np.allclose(got, [[3.0, 1.0, 5.0], [0.0, 4.0, 2.0]])


def test_gather_backward_refuses_an_index_outside_the_axis() -> None:
    """The backward pass *writes* at the index, so this one corrupts memory.

    Reached through the graph rather than called directly: the check has
    to hold on the path autograd actually takes.
    """
    base = lucid.tensor(np.ones((2, 3)), dtype=lucid.float64, requires_grad=True)
    index = lucid.tensor(np.full((2, 3), 99), dtype=lucid.int64)
    with pytest.raises(Exception, match="out of range"):
        base.gather(0, index).sum().backward()


def test_lu_solve_still_solves() -> None:
    """Guard the instrument: the checks must not have disabled the op."""
    a = lucid.tensor(np.array([[3.0, 1.0], [1.0, 2.0]]), dtype=lucid.float64)
    b = lucid.tensor(np.array([[9.0], [8.0]]), dtype=lucid.float64)
    lu, pivots = lucid.linalg.lu_factor(a)
    assert np.allclose(lucid.linalg.lu_solve(lu, pivots, b).numpy(), [[2.0], [3.0]])


# ── NaN through the order-based ops ──────────────────────────────────────────
# Found 2026-08-03 once the device axis stopped refusing to compare over
# float64: cpu and metal disagreed on every one of these, and neither
# matched the reference.

_NAN_VECTOR = np.array([1.0, np.nan, 3.0, -1.0], dtype=np.float32)


def _both_devices(name, values=_NAN_VECTOR):
    out = []
    for device in _DEVICES:
        got = getattr(lucid, name)(lucid.tensor(values, dtype=lucid.float32, device=device))
        got = got[0] if isinstance(got, tuple) else got
        out.append(np.asarray(got.numpy()).ravel())
    return out


@pytest.mark.parametrize("name", ["max", "min"])
def test_max_and_min_propagate_nan(name: str) -> None:
    """They answered with a plausible number instead.

    ``a > b ? a : b`` drops a NaN, because every comparison against one is
    false — so ``max`` of a poisoned batch came back 3.0 and looked
    healthy, on the CPU only, while Metal said nan.
    """
    cpu, metal = _both_devices(name)
    assert np.isnan(cpu).all(), f"{name} on the CPU swallowed the NaN"
    assert np.isnan(metal).all()


@pytest.mark.parametrize("name", ["argmax", "argmin"])
def test_arg_reduce_points_at_the_nan(name: str) -> None:
    """``x[argmax(x)]`` has to be ``max(x)``.

    ``max`` reports nan and the arg-reduce reported the index of an
    ordinary element, so the two disagreed with each other.  Neither
    device did this correctly: the CPU comparison skipped the NaN and
    MLX's own argmax does too.
    """
    cpu, metal = _both_devices(name)
    assert cpu.tolist() == [1], f"{name} on the CPU missed the NaN at index 1"
    assert metal.tolist() == [1], f"{name} on Metal missed the NaN at index 1"


def test_sort_orders_nan_last() -> None:
    """The comparator was not a strict weak ordering, which is UB.

    ``lv < rv`` is false in both directions when either side is NaN, so
    ``std::sort`` was free to do anything — and did: sorting
    ``[1, nan, 3, -1]`` returned ``[1, nan, -1, 3]``, which is not sorted.
    """
    cpu, metal = _both_devices("sort")
    assert np.allclose(cpu[:3], [-1.0, 1.0, 3.0]) and np.isnan(cpu[3])
    assert np.allclose(metal[:3], [-1.0, 1.0, 3.0]) and np.isnan(metal[3])


def test_argsort_agrees_with_sort() -> None:
    cpu, metal = _both_devices("argsort")
    assert cpu.tolist() == [3, 0, 2, 1]
    assert metal.tolist() == [3, 0, 2, 1]


@pytest.mark.parametrize("name", ["max", "min", "argmax", "argmin", "sort", "argsort"])
def test_the_nan_free_answer_is_unchanged(name: str) -> None:
    """Guard the instrument: NaN handling must not disturb ordinary data."""
    clean = np.array([1.0, 5.0, 3.0, -1.0], dtype=np.float32)
    cpu, metal = _both_devices(name, clean)
    assert np.array_equal(cpu, metal), name


# ── in-place ops and the graph ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["relu_", "exp_", "sin_", "cos_", "tanh_", "sigmoid_", "square_", "log_"]
)
def test_inplace_ops_extend_the_graph(name: str) -> None:
    """They computed the right value and the wrong derivative.

    ``inplace_unary`` ran the differentiable out-of-place op and then took
    only its storage, discarding the autograd node — so the tensor kept
    the new numbers while still sitting at its old position in the graph,
    and backward reported the gradient of whatever produced it:

        y = x * 1.0; y.exp_(); y.sum().backward()   ->  dx = 1

    Twenty-four ops, silent, and exactly the kind a training run would
    never notice.
    """
    # A negative entry is in here on purpose.  ``relu_`` has derivative
    # exactly 1 wherever its input is positive, so an all-positive probe
    # cannot tell the fixed op from the broken one — the first version of
    # this test asserted "not all ones" and failed on relu_ for that
    # reason rather than because anything was wrong.
    values = np.array([[0.5, -0.3], [1.2, 0.8]], dtype=np.float64)
    x = lucid.tensor(values, dtype=lucid.float64, requires_grad=True)
    y = x * 1.0
    getattr(y, name)()
    y.sum().backward()
    assert x.grad is not None
    got = x.grad.numpy()
    assert not np.allclose(got, 1.0), f"{name} did not extend the graph"


def test_inplace_still_writes_in_place() -> None:
    """Guard the instrument: the graph fix must not have made it a copy."""
    x = lucid.tensor(np.array([-1.0, 2.0]), dtype=lucid.float64)
    same = x
    x.relu_()
    assert np.allclose(same.numpy(), [0.0, 2.0])


def test_cross_entropy_does_not_gather_at_the_ignore_index() -> None:
    """It gathered at -100 and masked the garbage afterwards.

    The loss came out right because the out-of-range read was multiplied
    by the keep-mask, so nothing pointed at the fact that every ignored
    token in every masked-LM step read past the end of the logits.
    """
    rng = np.random.default_rng(0)
    logits = lucid.tensor(rng.random((2, 5, 3)).astype(np.float32), dtype=lucid.float32)
    target = lucid.tensor(np.array([[0, 1, -100], [2, -100, -100]]), dtype=lucid.int64)
    loss = F.cross_entropy(logits, target, ignore_index=-100)
    assert np.isfinite(float(loss))


@pytest.mark.parametrize("name", ["exp_", "sin_", "cos_", "tanh_", "sigmoid_", "log_"])
def test_inplace_second_route_matches_the_first(name: str) -> None:
    """``backward()`` and ``grad(create_graph=True)`` must agree.

    They did not, and only in graph mode.  A node that saves its input
    keeps a handle on the tensor it was handed; eager backward saves a
    Storage by value at forward time and never notices the later write,
    while graph-mode backward re-reads the handle — so after ``y.sin_()``
    overwrote it, the second route computed ``cos(sin(x))``.

    The op now runs against a snapshot that shares the buffer and inherits
    ``y``'s position in the graph, so the saved input stays put.
    """
    values = np.array([[0.5, 0.3], [1.2, 0.8]], dtype=np.float64)

    x = lucid.tensor(values, dtype=lucid.float64, requires_grad=True)
    y = x * 1.0
    getattr(y, name)()
    y.sum().backward()
    eager = x.grad.numpy()

    x2 = lucid.tensor(values, dtype=lucid.float64, requires_grad=True)
    y2 = x2 * 1.0
    getattr(y2, name)()
    (graph,) = lucid.autograd.grad(y2.sum(), [x2], create_graph=True)

    assert np.allclose(eager, graph.numpy(), rtol=1e-10), name


# ── casting and the graph ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["bool", "int", "long"])
def test_casting_to_a_discrete_dtype_ends_the_graph(name: str) -> None:
    """The derivative of rounding is zero almost everywhere.

    Lucid wired a backward through it anyway, and it did not produce a
    wrong gradient so much as fall over: ``x.long().sum().backward()``
    raised ``NotImplementedError: cpu_backend::broadcast_back_for_reduce``
    from several layers below anything the caller could act on.  The
    reference refuses the backward outright, which is the same statement
    made legibly.
    """
    x = lucid.tensor(np.array([[0.5, 0.3], [1.2, 0.8]]), dtype=lucid.float64)
    x.requires_grad_(True)
    out = getattr(x, name)()
    assert not out.requires_grad, f"{name}() stayed in the graph"
    with pytest.raises(RuntimeError, match="does not require grad"):
        out.sum().backward()
    assert x.grad is None


@pytest.mark.parametrize("name", ["float", "half"])
def test_casting_between_float_dtypes_stays_differentiable(name: str) -> None:
    """Guard the instrument: only the *discrete* casts detach.

    ``logits.float()`` mid-graph under autocast has to keep carrying
    gradients, which is the case the surrounding code was written for.
    """
    x = lucid.tensor(np.array([[0.5, 0.3], [1.2, 0.8]]), dtype=lucid.float64)
    x.requires_grad_(True)
    out = getattr(x, name)()
    assert out.requires_grad
    out.sum().backward()
    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), 1.0)


# ── overflow-safe hypot, NaN through the cumulative scans ────────────────────


def test_hypot_does_not_overflow_on_the_way() -> None:
    """``hypot(1e200, 1e200)`` is 1.41e200, an ordinary double.

    The naive ``sqrt(a² + b²)`` squares first and overflowed to inf.
    Avoiding exactly that is what distinguishes ``hypot`` from writing the
    formula out by hand.
    """
    big = lucid.tensor(np.array([1e200]), dtype=lucid.float64)
    got = lucid.hypot(big, big).numpy()
    assert np.isfinite(got).all(), "hypot overflowed"
    assert np.allclose(got, np.hypot(1e200, 1e200), rtol=1e-12)


@pytest.mark.parametrize(
    "pair", [(3.0, 4.0), (0.0, 0.0), (1e-200, 1e-200), (5.0, 0.0), (1e200, 1e200)]
)
def test_hypot_matches_across_magnitudes(pair) -> None:
    """Guard the instrument: the scaling must not disturb ordinary values."""
    a = lucid.tensor(np.array([pair[0]]), dtype=lucid.float64)
    b = lucid.tensor(np.array([pair[1]]), dtype=lucid.float64)
    assert np.allclose(lucid.hypot(a, b).numpy(), np.hypot(*pair), rtol=1e-12)


@pytest.mark.parametrize("name", ["cummax", "cummin"])
@pytest.mark.parametrize("device", _DEVICES)
def test_cumulative_scans_poison_after_a_nan(name: str, device: str) -> None:
    """Once a NaN is seen it is the running extreme for the rest of the scan.

    ``v > running`` is false against a NaN, so the CPU skipped it and
    carried a plausible running maximum; MLX's own cumulative scans skip
    it too, so both devices were wrong and in different ways.
    """
    values = np.array([1.0, np.nan, 3.0, -1.0], dtype=np.float32)
    out = getattr(lucid, name)(lucid.tensor(values, dtype=lucid.float32, device=device), 0)
    out = out[0] if isinstance(out, tuple) else out
    got = np.asarray(out.numpy()).ravel()
    assert got[0] == 1.0
    assert np.isnan(got[1:]).all(), f"{name} on {device} did not carry the NaN"


@pytest.mark.parametrize("name", ["cummax", "cummin"])
@pytest.mark.parametrize("device", _DEVICES)
def test_cumulative_scans_unchanged_without_a_nan(name: str, device: str) -> None:
    """Guard the instrument."""
    values = np.array([1.0, 5.0, 3.0, -1.0], dtype=np.float32)
    out = getattr(lucid, name)(lucid.tensor(values, dtype=lucid.float32, device=device), 0)
    out = out[0] if isinstance(out, tuple) else out
    expected = [1.0, 5.0, 5.0, 5.0] if name == "cummax" else [1.0, 1.0, 1.0, -1.0]
    assert np.allclose(np.asarray(out.numpy()).ravel(), expected)


# ── bfloat16 ─────────────────────────────────────────────────────────────────
# It was an alias for float16 — `lucid.tensor(..., dtype=lucid.bfloat16).dtype`
# answered `lucid.float16` — which the audit's `constant` axis reported.


def test_bfloat16_is_its_own_dtype() -> None:
    x = lucid.tensor(np.array([[1.0, 2.5]]), dtype=lucid.bfloat16)
    assert str(x.dtype).endswith("bfloat16"), "bfloat16 is still aliased"


def test_bfloat16_keeps_float32s_exponent_range() -> None:
    """The entire reason to choose the format over float16.

    Aliasing the two meant code that picked bfloat16 specifically to avoid
    overflow got the format that overflows: 1e30 is representable in
    bfloat16 and is an infinity in float16.
    """
    big = np.array([1e30])
    as_bf = float(lucid.tensor(big, dtype=lucid.bfloat16).numpy()[0])
    as_half = float(lucid.tensor(big, dtype=lucid.float16).numpy()[0])
    assert np.isfinite(as_bf) and np.isclose(as_bf, 1e30, rtol=1e-2)
    assert np.isinf(as_half), "float16 is expected to overflow here"

    tiny = np.array([1e-30])
    assert float(lucid.tensor(tiny, dtype=lucid.bfloat16).numpy()[0]) > 0.0
    assert float(lucid.tensor(tiny, dtype=lucid.float16).numpy()[0]) == 0.0


def test_bfloat16_numpy_export_converts_rather_than_relabels() -> None:
    """NumPy has no bfloat16, so the bridge widens — it must not re-describe.

    A view handed NumPy a 2-byte buffer as 4-byte elements and read across
    the boundaries: 1e30 came back as 4e-41, from the wrong position.
    """
    values = np.array([[1.0, 2.0], [4.0, 8.0]])
    got = lucid.tensor(values, dtype=lucid.bfloat16).numpy()
    assert got.dtype == np.float32
    assert np.allclose(got, values)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("name", ["relu", "exp", "sum", "softmax", "matmul"])
def test_bfloat16_computes_on_both_devices(name: str, device: str) -> None:
    """Accelerate has no bf16 kernels, so the CPU widens through float.

    Metal carries the format natively, so the two arriving at the same
    answer is the check worth making.
    """
    values = np.array([[1.0, -2.0], [3.0, 4.0]])
    x = lucid.tensor(values, dtype=lucid.bfloat16, device=device)
    ops = {
        "relu": lambda t: F.relu(t),
        "exp": lambda t: lucid.exp(t),
        "sum": lambda t: t.sum(),
        "softmax": lambda t: F.softmax(t, dim=-1),
        "matmul": lambda t: lucid.matmul(t, t),
    }
    out = ops[name](x)
    assert str(out.dtype).endswith("bfloat16"), f"{name} changed dtype"
    assert np.isfinite(np.asarray(out.numpy())).all()


@pytest.mark.parametrize("name", ["relu", "exp", "softmax", "matmul"])
def test_bfloat16_agrees_across_devices(name: str) -> None:
    values = np.array([[1.0, -2.0], [3.0, 4.0]])
    ops = {
        "relu": lambda t: F.relu(t),
        "exp": lambda t: lucid.exp(t),
        "softmax": lambda t: F.softmax(t, dim=-1),
        "matmul": lambda t: lucid.matmul(t, t),
    }
    outs = [
        np.asarray(ops[name](lucid.tensor(values, dtype=lucid.bfloat16, device=d)).numpy())
        for d in _DEVICES
    ]
    assert np.allclose(outs[0], outs[1], rtol=2e-2), name


# ── integer matmul and the ordering family ───────────────────────────────────


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("device", _DEVICES)
def test_integer_matmul_agrees_everywhere(dtype_name: str, device: str) -> None:
    """Neither backend had an integer GEMM, for different reasons.

    BLAS has none, and MLX refuses outright — "[matmul] Only inexact types
    are supported" — while the product of two integer matrices is an
    integer matrix. Both widen now, and both land on the exact answer.
    """
    values = np.array([[1, 2], [3, 4]])
    x = lucid.tensor(values, dtype=getattr(lucid, dtype_name), device=device)
    assert np.array_equal(lucid.matmul(x, x).numpy(), values @ values)


@pytest.mark.parametrize("dtype_name", ["int8", "int16"])
@pytest.mark.parametrize("name", ["sort", "argsort", "argmax", "kthvalue", "cummax", "cummin"])
def test_ordering_ops_take_narrow_integers(name: str, dtype_name: str) -> None:
    """Ordering is exact through int64, so the narrow widths widen.

    Every int8 and int16 value fits in an int64 and comparing the widened
    values gives the same order, so this is the answer a native kernel
    would have produced rather than an approximation of it. Metal answered
    all six already.
    """
    values = np.array([[3, 1], [2, 4]])
    ops = {
        "sort": lambda t: lucid.sort(t),
        "argsort": lambda t: lucid.argsort(t),
        "argmax": lambda t: lucid.argmax(t),
        "kthvalue": lambda t: lucid.kthvalue(t, 1),
        "cummax": lambda t: lucid.cummax(t, 0),
        "cummin": lambda t: lucid.cummin(t, 0),
    }
    outs = []
    for device in _DEVICES:
        got = ops[name](lucid.tensor(values, dtype=getattr(lucid, dtype_name), device=device))
        got = got[0] if isinstance(got, tuple) else got
        outs.append(np.asarray(got.numpy()))
    assert np.array_equal(outs[0], outs[1]), f"{name} disagrees across devices"


def test_sort_narrow_integer_keeps_its_dtype() -> None:
    """The values narrow back to the caller's width; the indices do not.

    An index tensor's dtype is the framework's own choice and is checked
    across devices rather than against a literal — asserting int64 here
    was asserting my assumption, and Lucid answers int32.
    """
    values = np.array([[3, 1], [2, 4]])
    for device in _DEVICES:
        x = lucid.tensor(values, dtype=lucid.int8, device=device)
        sorted_values = lucid.sort(x)
        sorted_values = sorted_values[0] if isinstance(sorted_values, tuple) else sorted_values
        assert str(sorted_values.dtype).endswith("int8"), device

    index_dtypes = {
        str(lucid.argsort(lucid.tensor(values, dtype=lucid.int8, device=d)).dtype) for d in _DEVICES
    }
    assert len(index_dtypes) == 1, f"index dtype differs across devices: {index_dtypes}"


# ── half and integer coverage in the layer ops ───────────────────────────────


_POOLING = {
    "max_pool1d": (lambda t: F.max_pool1d(t, 2), (1, 2, 8)),
    "max_pool2d": (lambda t: F.max_pool2d(t, 2), (1, 2, 8, 8)),
    "avg_pool1d": (lambda t: F.avg_pool1d(t, 2), (1, 2, 8)),
    "avg_pool2d": (lambda t: F.avg_pool2d(t, 2), (1, 2, 8, 8)),
    "adaptive_avg_pool2d": (lambda t: F.adaptive_avg_pool2d(t, 2), (1, 2, 8, 8)),
    "adaptive_max_pool2d": (lambda t: F.adaptive_max_pool2d(t, 2), (1, 2, 8, 8)),
}


@pytest.mark.parametrize("name", sorted(_POOLING))
def test_pooling_takes_half_on_both_devices(name: str) -> None:
    """Accelerate has no half kernel; the entry point widens and rounds.

    Refusing half here made ``model.half()`` a device-dependent
    proposition — the same network ran on Metal and raised on the CPU.
    """
    call, shape = _POOLING[name]
    outs = []
    for device in _DEVICES:
        x = np.ascontiguousarray(np.random.default_rng(0).random(shape).astype(np.float16))
        got = call(lucid.tensor(x, dtype=lucid.float16, device=device))
        got = got[0] if isinstance(got, tuple) else got
        assert str(got.dtype).endswith("float16"), f"{name} on {device}"
        outs.append(np.asarray(got.numpy(), dtype=np.float64))
    assert np.allclose(outs[0], outs[1], atol=1e-3), f"{name} disagrees across devices"


@pytest.mark.parametrize("name", sorted(_POOLING))
@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_pooling_takes_integers_on_both_devices(name: str, dtype_name: str) -> None:
    """Pooling an integer image answers in that integer type.

    ``max`` selects one of its inputs and ``avg`` casts a fractional mean,
    so both are defined without a float kernel.  The mean truncates toward
    zero, which is what the framework's own float-to-integer cast does —
    rounding to nearest instead put the CPU one apart from Metal and from
    the reference on every half-way window.
    """
    call, shape = _POOLING[name]
    values = (np.random.default_rng(1).integers(-9, 10, shape)).astype(dtype_name)
    outs = []
    for device in _DEVICES:
        got = call(
            lucid.tensor(
                np.ascontiguousarray(values),
                dtype=getattr(lucid, dtype_name),
                device=device,
            )
        )
        got = got[0] if isinstance(got, tuple) else got
        assert str(got.dtype).endswith(dtype_name), f"{name} on {device}"
        outs.append(np.asarray(got.numpy()))
    assert np.array_equal(outs[0], outs[1]), f"{name} disagrees across devices"


def test_integer_avg_pool_truncates_toward_zero() -> None:
    """The half-way case, pinned in both signs.

    4.5 becomes 4 and -4.5 becomes -4.  Round-to-nearest would answer 5
    and -5 (or -4, depending on the tie rule), and the disagreement is
    invisible on any window whose mean is already an integer.
    """
    positive = np.array([[[[3, 1], [5, 9]]]], dtype=np.int32)  # mean 4.5
    negative = -positive  # mean -4.5
    for device in _DEVICES:
        for values, expected in ((positive, 4), (negative, -4)):
            got = F.avg_pool2d(
                lucid.tensor(np.ascontiguousarray(values), dtype=lucid.int32, device=device),
                2,
            )
            assert int(np.asarray(got.numpy()).ravel()[0]) == expected, device


_PROMOTING_LAYER_OPS = {
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "softmin": F.softmin,
    "silu": F.silu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
    "hardsigmoid": F.hardsigmoid,
    "hardswish": F.hardswish,
    # ``normalize`` is not here: an integer norm is refused, as the
    # reference refuses it (LCD-384).
    "var": lucid.var,
    "std": lucid.std,
}


@pytest.mark.parametrize("name", sorted(_PROMOTING_LAYER_OPS))
def test_real_valued_layer_ops_promote_integers(name: str) -> None:
    """An integer input gets a float answer, on both devices.

    Each of these builds its own forward rather than going through a
    kernel template, and so none of them consulted the schema that
    already said the result was float.  Metal computed in the integer
    type — ``softmax([1, 2, 3])`` came back ``[0, 0, 0]`` — while the CPU
    raised NotImplementedError for the same call.
    """
    values = np.array([[1, 2, 3]], dtype=np.int32)
    outs = []
    for device in _DEVICES:
        got = _PROMOTING_LAYER_OPS[name](lucid.tensor(values, dtype=lucid.int32, device=device))
        assert lucid.is_floating_point(got), f"{name} on {device} stayed integral"
        outs.append(np.asarray(got.numpy(), dtype=np.float64))
    assert np.allclose(outs[0], outs[1], atol=1e-5), f"{name} disagrees across devices"


def test_softmax_of_integers_is_a_distribution() -> None:
    """The values, not only the dtype.

    Promoting could have been done by relabelling the output, which would
    keep the dtype assertion above green while returning the same wrong
    numbers.  These are the softmax of [1, 2, 3] and they sum to one.
    """
    for device in _DEVICES:
        got = F.softmax(lucid.tensor([[1, 2, 3]], dtype=lucid.int32, device=device))
        row = np.asarray(got.numpy(), dtype=np.float64).ravel()
        assert abs(row.sum() - 1.0) < 1e-6, device
        assert np.allclose(row, [0.09003057, 0.24472847, 0.66524096], atol=1e-6), device


@pytest.mark.parametrize("name", ["rand_like", "randn_like", "dropout", "uniform", "normal"])
def test_random_generators_draw_half(name: str) -> None:
    """No input to widen, so the draw is made wide and rounded.

    Sixteen public symbols sat behind three switches in the CPU
    generators — every ``nn.init`` routine, both ``*_like`` factories and
    the whole dropout family refused float16 while Metal ran them.
    """
    x = np.ascontiguousarray(np.ones((4, 4), dtype=np.float16))
    for device in _DEVICES:
        t = lucid.tensor(x, dtype=lucid.float16, device=device)
        if name in ("rand_like", "randn_like"):
            got = getattr(lucid, name)(t)
        elif name == "dropout":
            got = F.dropout(t, p=0.5)
        else:
            got = getattr(lucid.nn.init, name)(t)
        assert str(got.dtype).endswith("float16"), f"{name} on {device}"


def test_metal_storage_dtype_matches_its_bytes() -> None:
    """MLX promotes; the wrapper must convert rather than relabel.

    ``mean`` of an int32 array comes back float32.  Recording the
    caller's dtype next to the array's own byte count made the storage
    claim int32 while holding float32 bits, and ``avg_pool2d`` answered
    1083179008 — 0x40900000, the float 4.5 read as an integer — for a
    window whose mean is 4.5.  Nothing raised.
    """
    values = np.array([[[[3, 1], [5, 9]]]], dtype=np.int32)
    got = F.avg_pool2d(
        lucid.tensor(np.ascontiguousarray(values), dtype=lucid.int32, device="metal"), 2
    )
    assert abs(int(np.asarray(got.numpy()).ravel()[0])) < 100, "bytes were reinterpreted"


def test_bool_has_no_negation_on_either_device() -> None:
    """The CPU answered, and its answer was the input.

    Bool went through the integer path as uint8, where ``-1`` wraps to
    255 and 255 reads back as true — so ``-tensor([True, False])`` came
    back ``[True, False]``, unchanged, with nothing raised.  Metal and
    the reference both refuse.
    """
    values = np.array([[True, False], [False, True]])
    for device in _DEVICES:
        with pytest.raises(Exception):  # noqa: B017 - the two backends raise differently
            lucid.neg(lucid.tensor(values, dtype=lucid.bool, device=device))


@pytest.mark.parametrize("name", ["argmax", "argmin", "trace"])
def test_ordering_of_bools_is_refused_on_both_devices(name: str) -> None:
    """Metal answered these and its answers were not useful.

    ``trace`` summed a bool diagonal as bool, so a matrix with two trues
    on it traced to ``true`` — saturating rather than counting.  The
    reference refuses all three.
    """
    values = np.array([[True, False], [False, True]])
    for device in _DEVICES:
        with pytest.raises(Exception):  # noqa: B017 - the two backends raise differently
            getattr(lucid, name)(lucid.tensor(values, dtype=lucid.bool, device=device))


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_scatter_add_and_movement_keep_integers(dtype_name: str) -> None:
    """scatter_add, unfold and fold, at every integer width.

    int64 needed its own kernel rather than a widening: `as_i64` passes
    an int64 storage through untouched, so a widen-and-recurse guard that
    matched int64 called itself forever and the process died on a stack
    overflow.  int64 lands natively; the narrow widths widen onto it.
    """
    base = np.arange(4).reshape(2, 2).astype(dtype_name)
    index = np.array([[0, 1], [1, 0]])
    outs: dict[str, list] = {}
    for device in _DEVICES:
        x = lucid.tensor(
            np.ascontiguousarray(base), dtype=getattr(lucid, dtype_name), device=device
        )
        idx = lucid.tensor(index, dtype=lucid.int64, device=device)
        scattered = lucid.scatter_add(x, 0, idx, x)
        unfolded = F.unfold(x.reshape(1, 1, 2, 2), 2)
        assert str(scattered.dtype).endswith(dtype_name), device
        assert str(unfolded.dtype).endswith(dtype_name), device
        outs[device] = [
            np.asarray(scattered.numpy()).tolist(),
            np.asarray(unfolded.numpy()).tolist(),
        ]
    assert outs["cpu"] == outs["metal"]


def test_int64_scatter_survives_metal_without_precision_loss() -> None:
    """MLX's Metal scatter has no int64 kernel, so this one round-trips.

    The values are chosen above 2^24 so a float32 detour — the tempting
    on-device workaround — would be visible.
    """
    base = np.array([[2**40, 2**41], [2**42, 2**43]], dtype=np.int64)
    index = np.array([[0, 1], [1, 0]])
    outs = []
    for device in _DEVICES:
        x = lucid.tensor(np.ascontiguousarray(base), dtype=lucid.int64, device=device)
        idx = lucid.tensor(index, dtype=lucid.int64, device=device)
        got = lucid.scatter_add(x, 0, idx, x)
        assert str(got.dtype).endswith("int64"), device
        outs.append(np.asarray(got.numpy()).tolist())
    assert outs[0] == outs[1]
    assert 2**41 + 2**43 in [v for row in outs[0] for v in row]


def test_eye_builds_an_int64_identity_on_both_devices() -> None:
    """`mlx.core.eye` scatters, and Metal's scatter has no int64 kernel.

    Its entries are 0 and 1, so it builds at int32 and widens.
    """
    for device in _DEVICES:
        got = lucid.nn.init.eye(lucid.zeros((3, 3), dtype=lucid.int64, device=device))
        assert str(got.dtype).endswith("int64"), device
        assert np.array_equal(np.asarray(got.numpy()), np.eye(3, dtype=np.int64)), device


def test_local_response_norm_promotes_rather_than_degenerating() -> None:
    """A composite has no schema, so its promotion has to be written down.

    ``alpha`` is 1e-4.  In an integer dtype it truncates to zero, every
    scale collapses to ``k`` and the whole op silently becomes the
    identity — which is why this checks a value and not just a dtype.
    """
    values = np.arange(1, 5, dtype=np.int64).reshape(1, 4, 1, 1)
    for device in _DEVICES:
        x = lucid.tensor(np.ascontiguousarray(values), dtype=lucid.int64, device=device)
        got = F.local_response_norm(x, 2)
        assert lucid.is_floating_point(got), device
        flat = np.asarray(got.numpy(), dtype=np.float64).ravel()
        assert not np.array_equal(flat, values.ravel()), f"{device}: became the identity"
        # Against the float path rather than a literal — the numbers are
        # the composition's, not mine, and writing them out by hand was
        # how this assertion first failed.
        as_float = F.local_response_norm(
            lucid.tensor(
                np.ascontiguousarray(values.astype(np.float32)),
                dtype=lucid.float32,
                device=device,
            ),
            2,
        )
        assert np.allclose(
            flat, np.asarray(as_float.numpy(), dtype=np.float64).ravel(), atol=1e-5
        ), device


# ── NaN, and the gradient of an op that has none ─────────────────────────────


@pytest.mark.parametrize("dtype_name,np_dtype", [("float32", np.float32), ("float64", np.float64)])
@pytest.mark.parametrize("name", ["minimum", "maximum"])
def test_minimum_and_maximum_propagate_nan_at_both_widths(
    name: str, dtype_name: str, np_dtype
) -> None:
    """``vDSP_vmaxD`` does not propagate NaN; ``vDSP_vmax`` does.

    Same primitive family, opposite answer, nothing in the documentation
    distinguishing them — so whether an op saw a NaN depended on the
    dtype it was called at.  Both widths are pinned here, including the
    float one that is still on vDSP, so a change in Accelerate is a test
    failure rather than a silent one.
    """
    a = np.array([np.nan, 1.0, 3.0], dtype=np_dtype)
    b = np.array([2.0, np.nan, 2.0], dtype=np_dtype)
    for device in _DEVICES:
        if dtype_name == "float64" and device == "metal":
            continue  # Metal has no float64
        got = getattr(lucid, name)(
            lucid.tensor(np.ascontiguousarray(a), dtype=getattr(lucid, dtype_name), device=device),
            lucid.tensor(np.ascontiguousarray(b), dtype=getattr(lucid, dtype_name), device=device),
        )
        flat = np.asarray(got.numpy(), dtype=np.float64)
        assert np.isnan(flat[0]), f"{name}/{dtype_name}/{device}: lost NaN from the left"
        assert np.isnan(flat[1]), f"{name}/{dtype_name}/{device}: lost NaN from the right"
        assert flat[2] == (2.0 if name == "minimum" else 3.0)


@pytest.mark.parametrize("name", ["ceil_", "floor_", "round_", "sign_"])
def test_non_differentiable_inplace_ops_end_the_graph(name: str) -> None:
    """They returned the gradient of whatever produced the tensor.

    ``y = x * 1.0; y.ceil_(); y.sum().backward()`` gave ``dx = 1``.  The
    in-place wrapper adopts the forward's graph position, and these ops
    built no node to adopt — so the old one stayed and the gradient
    flowed through as if the call had not happened.  They now build a node
    whose gradient is zero, in place as out of place, and the reference
    answers 0 too.
    """
    x = lucid.tensor(np.array([1.3, 2.7, 3.5]), requires_grad=True)
    y = x * 1.0
    getattr(y, name)()
    y.sum().backward()
    assert x.grad is not None, f"{name} left the graph"
    np.testing.assert_array_equal(x.grad.numpy(), [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "name,expected",
    [
        ("exp_", np.exp(np.array([1.3, 2.7, 3.5]))),
        ("sin_", np.cos(np.array([1.3, 2.7, 3.5]))),
        ("sqrt_", 0.5 / np.sqrt(np.array([1.3, 2.7, 3.5]))),
    ],
)
def test_differentiable_inplace_ops_still_carry_their_gradient(
    name: str, expected: np.ndarray
) -> None:
    """Guard the instrument: the cut must only apply where there is no node."""
    x = lucid.tensor(np.array([1.3, 2.7, 3.5]), requires_grad=True)
    y = x * 1.0
    getattr(y, name)()
    y.sum().backward()
    assert np.allclose(np.asarray(x.grad.numpy()).ravel(), expected), name


def test_inplace_under_no_grad_keeps_the_chain() -> None:
    """Inside ``no_grad`` the write is an ordinary mutation.

    Severing there would lose a chain the caller means to keep; the
    version counter is what guards that case.
    """
    x = lucid.tensor(np.array([1.3, 2.7]), requires_grad=True)
    y = x * 2.0
    with lucid.no_grad():
        y.ceil_()
    assert y.requires_grad, "no_grad in-place cut a graph it should have left alone"


def test_a_leaf_requiring_grad_refuses_inplace_mutation() -> None:
    """The other half of the same defect, and the one with no answer.

    A leaf is where gradient accumulates.  Writing to it in place moves
    it and leaves nothing behind that says so, so ``x.ceil_()`` answered
    ``dx = 1`` — a leaf has no grad_fn to cut, and ``ceil`` builds a node
    whose gradient is empty, so nothing was adopted either.  After the
    write ``x`` *is* the result, and "the gradient with respect to x"
    names two different tensors depending on when it is asked.  The
    reference refuses this setup; so does Lucid now.

    ``no_grad`` is how a caller says they mean to overwrite — which is
    what every optimiser step already does.
    """
    p = lucid.nn.Parameter(lucid.tensor(np.array([1.3, 2.7])))
    with pytest.raises(Exception, match="leaf tensor that requires grad"):
        p.round_()

    with lucid.no_grad():
        p.round_()
    assert p.requires_grad
    assert np.array_equal(np.asarray(p.numpy()).ravel(), [1.0, 3.0])


@pytest.mark.parametrize("name", ["cummax", "cummin"])
def test_running_extremes_accept_bool(name: str) -> None:
    """A running maximum of booleans is a running OR.

    The widen-onto-int64 path already covered the narrow integers and
    excluded bool by name, so Metal answered a bool scan and the CPU
    refused one.  0 and 1 survive the widening unchanged, which is the
    whole reason the widening is exact.
    """
    values = np.array([[True, False], [False, True]])
    outs = []
    for device in _DEVICES:
        got = getattr(lucid, name)(lucid.tensor(values, dtype=lucid.bool, device=device), 0)
        got = got[0] if isinstance(got, tuple) else got
        assert str(got.dtype).endswith("bool"), f"{name} on {device}"
        outs.append(np.asarray(got.numpy()).tolist())
    assert outs[0] == outs[1]


@pytest.mark.parametrize("dtype_name", ["int8", "int16", "int32", "int64"])
def test_cube_keeps_its_integer_width(dtype_name: str) -> None:
    """``x·x·x`` is arithmetic, so it goes through int64, not double."""
    values = np.array([[1, 2], [3, 4]]).astype(dtype_name)
    outs = []
    for device in _DEVICES:
        x = lucid.tensor(
            np.ascontiguousarray(values),
            dtype=getattr(lucid, dtype_name),
            device=device,
        )
        x.cube_()
        assert str(x.dtype).endswith(dtype_name), device
        outs.append(np.asarray(x.numpy()).tolist())
    assert outs[0] == outs[1]


def test_int64_cube_is_exact_past_doubles_range() -> None:
    """1e6 cubed is 1e18, which a double cannot hold exactly.

    The widen-through-double route that works for a *selection* would
    round here — the cube of an int64 leaves double's exact range at
    |x| > 2^17 — so this one is written out at its own width.
    """
    x = lucid.tensor(np.array([1_000_000], dtype=np.int64), dtype=lucid.int64)
    x.cube_()
    assert int(np.asarray(x.numpy()).ravel()[0]) == 10**18


def test_dropout_refuses_integers_on_both_devices() -> None:
    """The mask is 1/(1-p) and 0, and neither survives an integer dtype.

    The scale truncates to 1, so the op silently becomes a random zeroing
    with no compensation.  The CPU has no integer Bernoulli and said so;
    Metal cast one into existence, so the same call ran on one device and
    raised on the other.  The reference refuses it too.
    """
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    for device in _DEVICES:
        with pytest.raises(Exception):  # noqa: B017 - the two backends raise differently
            F.dropout(
                lucid.tensor(np.ascontiguousarray(values), dtype=lucid.int32, device=device),
                p=0.5,
            )


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_dropout_still_takes_every_float(dtype_name: str) -> None:
    """Guard the instrument: the gate must not have closed on half."""
    values = np.ones((4, 4)).astype(dtype_name)
    for device in _DEVICES:
        got = F.dropout(
            lucid.tensor(
                np.ascontiguousarray(values),
                dtype=getattr(lucid, dtype_name),
                device=device,
            ),
            p=0.5,
        )
        assert str(got.dtype).endswith(dtype_name), device


# ── the in-place binary family and its graph position ────────────────────────


@pytest.mark.parametrize(
    "name,expected_dx,expected_dy",
    [
        ("add_", [1.0, 1.0], [1.0, 1.0]),
        ("sub_", [1.0, 1.0], [-1.0, -1.0]),
        ("mul_", [2.0, 0.5], [1.5, 2.5]),  # d/dx = y, d/dy = x
        ("div_", [0.5, 2.0], [-0.375, -10.0]),
    ],
)
def test_inplace_binary_ops_carry_both_gradients(name: str, expected_dx, expected_dy) -> None:
    """``z.mul_(y)`` gave ``dx = 1`` and ``dy = None``.

    The binary in-place helper took the forward's storage and left its
    grad_fn behind, so ``z`` stayed where it sat before the call and
    reported the gradient of whatever produced it — while ``y`` was never
    in the graph at all.  Seven ops, and ``add_`` hid it: its derivative
    with respect to the first operand really is 1, so the wrong answer
    and the right one coincided there.
    """
    x = lucid.tensor(np.array([1.5, 2.5]), requires_grad=True)
    y = lucid.tensor(np.array([2.0, 0.5]), requires_grad=True)
    z = x * 1.0
    getattr(z, name)(y)
    z.sum().backward()
    assert x.grad is not None and y.grad is not None, f"{name}: a gradient is missing"
    assert np.allclose(np.asarray(x.grad.numpy()).ravel(), expected_dx), name
    assert np.allclose(np.asarray(y.grad.numpy()).ravel(), expected_dy), name


def test_inplace_binary_op_on_a_leaf_is_refused() -> None:
    """Same rule as the unary family — one header, one decision."""
    p = lucid.nn.Parameter(lucid.tensor(np.array([1.0, 2.0])))
    with pytest.raises(Exception, match="leaf tensor that requires grad"):
        p.mul_(lucid.tensor(np.array([2.0, 2.0])))

    with lucid.no_grad():
        p.mul_(lucid.tensor(np.array([2.0, 2.0])))
    assert p.requires_grad
    assert np.array_equal(np.asarray(p.numpy()).ravel(), [2.0, 4.0])


def test_inplace_binary_survives_create_graph() -> None:
    """The forward runs against a snapshot, so the saved input is intact.

    Without it the node re-reads a buffer the assignment has already
    overwritten, and only the graph-mode route notices — eager
    ``backward()`` saves a Storage by value at forward time.
    """
    x = lucid.tensor(np.array([1.5, 2.5]), requires_grad=True)
    y = lucid.tensor(np.array([2.0, 3.0]), requires_grad=True)
    z = x * 1.0
    z.mul_(y)
    (first,) = lucid.autograd.grad(z.sum(), x, create_graph=True)
    assert np.allclose(np.asarray(first.numpy()).ravel(), [2.0, 3.0])
    # d²/dxdy of (x·y) is 1
    (second,) = lucid.autograd.grad(first.sum(), y)
    assert np.allclose(np.asarray(second.numpy()).ravel(), [1.0, 1.0])


def test_tensordot_takes_half_on_both_devices() -> None:
    """Accelerate has no half GEMM and the loop is written out per float type.

    Metal ran a half-precision contraction and the CPU did not, so
    ``tensordot`` was a device-dependent proposition under ``model.half()``.
    """
    a = np.arange(6, dtype=np.float16).reshape(2, 3)
    b = np.arange(12, dtype=np.float16).reshape(3, 4)
    outs = []
    for device in _DEVICES:
        got = lucid.tensordot(
            lucid.tensor(np.ascontiguousarray(a), dtype=lucid.float16, device=device),
            lucid.tensor(np.ascontiguousarray(b), dtype=lucid.float16, device=device),
            1,
        )
        assert str(got.dtype).endswith("float16"), device
        outs.append(np.asarray(got.numpy(), dtype=np.float64))
    assert np.allclose(outs[0], outs[1], rtol=1e-2)
    assert np.allclose(outs[0], a.astype(np.float64) @ b.astype(np.float64), rtol=1e-2)


@pytest.mark.parametrize("dtype_name", ["int32", "float32"])
def test_grid_sample_promotes_both_of_its_operands(dtype_name: str) -> None:
    """Resampling interpolates *between* samples, so the answer is real.

    The forward is assembled by hand and read ``input->dtype()`` directly
    rather than the schema that already said so, so an integer image ran
    on Metal and raised on the CPU.  Both operands promote: the grid
    carries fractional coordinates and is no more integral than the image.
    """
    image = np.arange(16).reshape(1, 1, 4, 4).astype(dtype_name)
    grid = np.zeros((1, 2, 2, 2)).astype(dtype_name)
    outs = []
    for device in _DEVICES:
        got = F.grid_sample(
            lucid.tensor(
                np.ascontiguousarray(image),
                dtype=getattr(lucid, dtype_name),
                device=device,
            ),
            lucid.tensor(
                np.ascontiguousarray(grid),
                dtype=getattr(lucid, dtype_name),
                device=device,
            ),
        )
        assert lucid.is_floating_point(got), device
        outs.append(np.asarray(got.numpy(), dtype=np.float64))
    assert np.allclose(outs[0], outs[1])
    # the centre of a 4x4 ramp
    assert np.allclose(outs[0].ravel()[:2], [7.5, 7.5])


@pytest.mark.parametrize("buffer_dtype", ["int32", "float32", "float64"])
def test_batch_norm_updates_its_statistics_in_a_float(buffer_dtype: str) -> None:
    """A running mean and variance are real numbers.

    The update ran at the *buffers'* dtype, so integer buffers put every
    step of it at that width and ``reciprocal`` refused on the CPU while
    Metal answered.  It runs at the dtype the forward computed in now and
    casts back on the write, so the caller's storage keeps its own dtype.
    """
    if buffer_dtype == "float64":
        devices = ["cpu"]  # Metal has no float64
    else:
        devices = _DEVICES
    values = np.random.default_rng(0).random((2, 3, 2, 2)) * 10
    outs = []
    for device in devices:
        dt = getattr(lucid, buffer_dtype)

        def make(array, d=dt, dev=device):
            return lucid.tensor(
                np.ascontiguousarray(np.asarray(array).astype(buffer_dtype)),
                dtype=d,
                device=dev,
            )

        running_mean, running_var = make(np.zeros(3)), make(np.ones(3))
        got = F.batch_norm(
            make(values),
            running_mean,
            running_var,
            make(np.ones(3)),
            make(np.zeros(3)),
            True,
        )
        assert lucid.is_floating_point(got), (buffer_dtype, device)
        assert str(running_mean.dtype).endswith(buffer_dtype), "buffer changed dtype"
        outs.append(np.asarray(got.numpy(), dtype=np.float64))
    if len(outs) == 2:
        assert np.allclose(outs[0], outs[1], rtol=1e-4)


def test_batch_norm_keeps_float32_statistics_under_a_half_input() -> None:
    """The AMP shape, and the one the relabel would have corrupted.

    ``saved_rstd`` is at the forward's dtype and was handed to ``square``
    labelled as the *buffers'* — reading one dtype's bytes as another's
    whenever the two differ, which under autocast they do.
    """
    values = (np.random.default_rng(0).random((2, 3, 2, 2)) * 10).astype(np.float16)
    means = []
    for device in _DEVICES:
        x = lucid.tensor(np.ascontiguousarray(values), dtype=lucid.float16, device=device)
        f32 = lambda a: lucid.tensor(  # noqa: E731
            np.ascontiguousarray(np.asarray(a, dtype=np.float32)),
            dtype=lucid.float32,
            device=device,
        )
        running_mean, running_var = f32(np.zeros(3)), f32(np.ones(3))
        got = F.batch_norm(x, running_mean, running_var, f32(np.ones(3)), f32(np.zeros(3)), True)
        assert str(got.dtype).endswith("float16"), device
        assert str(running_mean.dtype).endswith("float32"), device
        means.append(np.asarray(running_mean.numpy(), dtype=np.float64))
    assert np.allclose(means[0], means[1], atol=1e-3)
    assert np.isfinite(means[0]).all()


@pytest.mark.parametrize("dtype_name", ["int32", "float32"])
def test_embedding_bag_promotes_an_integer_table(dtype_name: str) -> None:
    """Embeddings are real numbers; pooling them sums or averages.

    The CPU kernel is written for floats only, so Metal took an integer
    table and the CPU refused one.  The indices and offsets beside it stay
    integral — they are positions, not values.
    """
    weight = np.arange(15).reshape(5, 3).astype(dtype_name)
    index = np.array([0, 2, 4, 1, 3], dtype=np.int64)
    offsets = np.array([0, 3], dtype=np.int64)
    outs = []
    for device in _DEVICES:
        got = F.embedding_bag(
            lucid.tensor(index, dtype=lucid.int64, device=device),
            lucid.tensor(
                np.ascontiguousarray(weight),
                dtype=getattr(lucid, dtype_name),
                device=device,
            ),
            lucid.tensor(offsets, dtype=lucid.int64, device=device),
            mode="sum",
        )
        assert lucid.is_floating_point(got), device
        outs.append(np.asarray(got.numpy(), dtype=np.float64))
    assert np.allclose(outs[0], outs[1])
    assert np.allclose(outs[0].ravel()[:3], [18.0, 21.0, 24.0])


@pytest.mark.parametrize("dtype_name", ["int32", "int64", "float32"])
def test_fold_keeps_its_dtype_on_both_devices(dtype_name: str) -> None:
    """Metal's fold accumulates with a scatter, which has no int64 kernel.

    Same route the scatter ops themselves take: there is no exact way
    around it on-device, and refusing a dtype the CPU folds is the
    device-dependent behaviour the backend pair exists to avoid.
    """
    values = np.arange(1, 5).reshape(1, 4, 1).astype(dtype_name)
    outs = []
    for device in _DEVICES:
        got = F.fold(
            lucid.tensor(
                np.ascontiguousarray(values),
                dtype=getattr(lucid, dtype_name),
                device=device,
            ),
            (2, 2),
            2,
        )
        assert str(got.dtype).endswith(dtype_name), device
        outs.append(np.asarray(got.numpy()).tolist())
    assert outs[0] == outs[1]


def test_fold_masks_its_padding_by_selecting() -> None:
    """``0 * NaN`` is not nothing — the same defect ``unfold`` had."""
    values = np.array([[[np.nan], [1.0], [2.0], [3.0]]], dtype=np.float32)
    outs = []
    for device in _DEVICES:
        got = F.fold(
            lucid.tensor(np.ascontiguousarray(values), dtype=lucid.float32, device=device),
            (2, 2),
            2,
        )
        outs.append(np.asarray(got.numpy(), dtype=np.float64).ravel())
    assert np.array_equal(outs[0], outs[1], equal_nan=True)
    assert np.isnan(outs[0][0]) and np.array_equal(outs[0][1:], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("supplied", ["partial", "none"])
def test_multi_head_attention_forward_runs_on_metal(supplied: str) -> None:
    """It built its temporary module on the *default* device.

    Only the weights the caller supplied were bound over those
    parameters, so with Metal inputs and a partial set the rest stayed on
    the CPU and the first projection raised a device mismatch — the entry
    point worked on Metal only if you passed every weight there is.
    """
    rng = np.random.default_rng(0)
    embed, heads, length, batch = 8, 2, 4, 2
    q = rng.random((length, batch, embed)).astype(np.float32)
    for device in _DEVICES:

        def make(array, dev=device):
            return lucid.tensor(np.ascontiguousarray(array), dtype=lucid.float32, device=dev)

        extra = (
            {"in_proj_weight": make(rng.random((3 * embed, embed)).astype(np.float32))}
            if supplied == "partial"
            else {}
        )
        out, _ = F.multi_head_attention_forward(make(q), make(q), make(q), embed, heads, **extra)
        assert tuple(out.shape) == (length, batch, embed), device
        assert device in str(out.device), device


@pytest.mark.parametrize("name", ["jacobian", "vjp", "jvp", "hessian"])
def test_autograd_functionals_refuse_a_discrete_input(name: str) -> None:
    """A derivative with respect to an integer is not a number.

    These accepted one and produced whatever the ops underneath happened
    to do with it, which differed by device — ``jacobian`` ran on Metal
    for int8 and raised on the CPU, because the two backends refuse
    different narrow widths for unrelated reasons.  The reference states
    it as "only Tensors of floating point dtype can require gradients".
    """
    fn = getattr(lucid.autograd, name)
    for device in _DEVICES:
        x = lucid.tensor(np.array([1, 2, 3], dtype=np.int32), dtype=lucid.int32, device=device)
        rest = (x.to(lucid.float32),) if name in ("vjp", "jvp") else ()
        with pytest.raises(TypeError, match="floating-point input"):
            fn(lambda a: a * a, x, *rest)


# ── the overwriting assignments ──────────────────────────────────────────────


def test_fill_and_zero_end_the_graph() -> None:
    """They wrote through the raw buffer copy, so the graph stayed put.

    ``y = x * 1.0; y.fill_(2.0); y.sum().backward()`` gave ``dx = 1`` —
    the gradient ``y`` had before the fill, for a tensor whose value no
    longer depends on ``x`` at all.
    """
    for name, call in (
        ("fill_", lambda y: y.fill_(2.0)),
        ("zero_", lambda y: y.zero_()),
    ):
        x = lucid.tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = x * 1.0
        call(y)
        y.sum().backward()
        grad = None if x.grad is None else np.asarray(x.grad.numpy()).ravel()
        assert grad is None or np.allclose(grad, 0.0), f"{name}: {grad}"


def test_copy_routes_the_gradient_to_its_source() -> None:
    """Which is the whole point of copying a differentiable tensor in."""
    x = lucid.tensor(np.array([1.0, 2.0]), requires_grad=True)
    z = lucid.tensor(np.array([3.0, 4.0]), requires_grad=True)
    y = x * 1.0
    y.copy_(z * 2.0)
    y.sum().backward()
    assert z.grad is not None and np.allclose(np.asarray(z.grad.numpy()).ravel(), 2.0)
    assert x.grad is None or np.allclose(np.asarray(x.grad.numpy()).ravel(), 0.0)


@pytest.mark.parametrize("name", ["fill_", "zero_", "copy_"])
def test_assignment_to_a_leaf_is_refused(name: str) -> None:
    """The reference refuses all three on a leaf that requires grad."""
    p = lucid.nn.Parameter(lucid.tensor(np.array([1.0, 2.0])))
    call = {
        "fill_": lambda: p.fill_(3.0),
        "zero_": lambda: p.zero_(),
        "copy_": lambda: p.copy_(lucid.tensor(np.array([3.0, 4.0]))),
    }[name]
    with pytest.raises(Exception, match="leaf tensor that requires grad"):
        call()
    with lucid.no_grad():
        call()
    assert p.requires_grad


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_initialisers_keep_the_parameter_they_were_given(dtype_name: str) -> None:
    """``_fill_from_impl`` replaced the impl rather than writing through it.

    That silently detached the tensor from any graph it was in — no write
    means no version bump, so a backward built from overwritten values
    said nothing — and it adopted the source's dtype wholesale, so
    ``orthogonal_`` turned a float64 parameter into a float32 one.
    """
    for name in ("eye", "constant", "xavier_uniform", "orthogonal", "zeros_", "ones_"):
        fn = getattr(lucid.nn.init, name)
        p = lucid.nn.Parameter(lucid.tensor(np.zeros((4, 4)), dtype=getattr(lucid, dtype_name)))
        before = id(p._impl)
        fn(p, 1.0) if name == "constant" else fn(p)
        assert str(p.dtype).endswith(dtype_name), f"{name} changed the dtype"
        assert p.requires_grad, f"{name} dropped requires_grad"
        assert id(p._impl) == before, f"{name} replaced the impl instead of writing"
