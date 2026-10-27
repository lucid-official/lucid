"""A custom ``Function``'s gradients are held to its inputs.

The engine reads each gradient ``backward`` returns as a buffer of the
matching input's shape, dtype and device — the tensor's own metadata is
not carried across.  Nothing checked the two agreed, so:

* a gradient of the wrong shape was read short or past its end — a
  ``(3,)`` gradient for a ``(2, 5)`` input gave ``[[1, 1, 1, 2, 2], ...]``,
  the tail being whatever the allocator held;
* a gradient of another dtype was reinterpreted bit for bit wherever the
  input was not a leaf (a leaf's accumulator casts; nothing else did);
* a gradient on the wrong device became a ``.grad`` whose every read
  raised ``bad_variant_access``.

The same silence let the FFT backward return gradients at the transform
size rather than the input's for every ``n`` / ``s`` — wrong values, not
an error.  Each gradient is now checked as the reference checks a node's
results: the input's shape passes, a shape the input broadcasts to is
summed back down, anything else is refused naming the ``Function`` and
the index; a dtype is cast; a device mismatch is refused, except for a
0-d gradient, which is moved.

``backward`` may also answer with one gradient per positional argument of
``forward`` — ``None`` for a non-tensor — as documented; that form used
to fail with an engine-internal size mismatch.
"""

import pytest

import lucid
from lucid.autograd import Function


def _returning(grad_fn):
    """A ``Function`` doubling its input whose backward returns ``grad_fn(g)``."""

    class Returning(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        def backward(ctx, g):
            return grad_fn(g)

    return Returning


# ── Shape ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("shape", "grad_shape"),
    [((2, 5), (3,)), ((4,), (2, 3)), ((2, 5), (1, 5)), ((2, 5), (5, 2)), ((3,), ())],
)
def test_a_gradient_of_an_incompatible_shape_is_refused(shape, grad_shape, device: str) -> None:
    fn = _returning(lambda g: lucid.ones(grad_shape, device=device))
    x = lucid.ones(shape, device=device, requires_grad=True)
    expected = (
        f"Function ReturningBackward returned an invalid gradient at index 0 - "
        f"got {list(grad_shape)} but expected shape compatible with {list(shape)}"
    )
    with pytest.raises(RuntimeError) as info:
        fn.apply(x).sum().backward()
    assert str(info.value) == expected
    assert x.grad is None


@pytest.mark.parametrize(
    ("shape", "grad_shape", "expected"),
    [
        ((2, 5), (3, 2, 5), 3.0),  # leading axes summed away
        ((2, 1), (2, 5), 5.0),  # a size-1 axis summed, kept
        ((1, 3), (4, 2, 3), 8.0),  # both at once
        ((), (1,), 1.0),  # a 0-d input
    ],
)
def test_a_gradient_the_input_broadcasts_to_is_summed(
    shape, grad_shape, expected, device: str
) -> None:
    fn = _returning(lambda g: lucid.ones(grad_shape, device=device))
    x = lucid.ones(shape, device=device, requires_grad=True)
    fn.apply(x).sum().backward()
    assert tuple(x.grad.shape) == shape
    assert lucid.all(x.grad == expected).item()


def test_a_correct_gradient_is_untouched(device: str) -> None:
    fn = _returning(lambda g: g * 2)
    x = lucid.ones(2, 3, device=device, requires_grad=True)
    fn.apply(x).sum().backward()
    assert x.grad.tolist() == [[2.0] * 3] * 2


def test_the_index_is_the_tensor_input_s(device: str) -> None:
    class Pair(Function):
        @staticmethod
        def forward(ctx, a, b):
            return a * b

        @staticmethod
        def backward(ctx, g):
            return g, lucid.ones(7, device=device)

    a = lucid.ones(2, 3, device=device, requires_grad=True)
    b = lucid.ones(3, device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r"PairBackward returned an invalid gradient at index 1 - got \[7\]",
    ):
        Pair.apply(a, b).sum().backward()


def test_an_input_needing_no_gradient_is_not_checked(device: str) -> None:
    """The reference skips an input with no edge — its result goes nowhere."""

    class Pair(Function):
        @staticmethod
        def forward(ctx, a, b):
            return a * b

        @staticmethod
        def backward(ctx, g):
            return g, lucid.ones(7, device=device)

    a = lucid.ones(2, 3, device=device, requires_grad=True)
    b = lucid.ones(2, 3, device=device)
    Pair.apply(a, b).sum().backward()
    assert a.grad.tolist() == [[1.0] * 3] * 2


def test_a_wrong_shape_is_refused_under_create_graph(device: str) -> None:
    fn = _returning(lambda g: lucid.ones(3, device=device))
    x = lucid.ones(2, 5, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="invalid gradient at index 0"):
        lucid.autograd.grad(fn.apply(x).sum(), x, create_graph=True)


def test_a_broadcast_gradient_keeps_its_graph_under_create_graph(device: str) -> None:
    """The sum that brings it back to shape is recorded, so a second
    derivative still flows through it."""

    class Cube(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x * x

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            # (4, 3) for a (3,) input — summed back down by the check.
            return (3 * x * x * g).expand(4, 3) / 4

    x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    (g,) = lucid.autograd.grad(Cube.apply(x).sum(), x, create_graph=True)
    assert tuple(g.shape) == (3,)
    assert g.tolist() == pytest.approx([3.0, 12.0, 27.0])
    (gg,) = lucid.autograd.grad(g.sum(), x)
    assert gg.tolist() == pytest.approx([6.0, 12.0, 18.0])


def test_a_wrong_shape_is_refused_for_a_several_output_function(device: str) -> None:
    class Split(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 1, x * 2

        @staticmethod
        def backward(ctx, g1, g2):
            return lucid.ones(4, device=device)

    x = lucid.ones(2, 3, device=device, requires_grad=True)
    a, b = Split.apply(x)
    with pytest.raises(RuntimeError, match="SplitBackward returned an invalid gradient"):
        (a + b).sum().backward()


# ── Dtype ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.int32])
def test_another_dtype_is_cast_for_an_input_that_is_not_a_leaf(dtype, device: str) -> None:
    """A leaf's accumulator always cast; an intermediate input read the
    buffer bit for bit — 32.06 for 3, or zeros, depending on the dtype."""
    fn = _returning(lambda g: (g * 3).to(dtype))
    x = lucid.ones(2, 3, device=device, requires_grad=True)
    fn.apply(x * 1.0).sum().backward()
    assert x.grad.dtype == lucid.float32
    assert x.grad.tolist() == [[3.0] * 3] * 2


def test_a_complex_gradient_for_a_real_input_keeps_its_real_part(device: str) -> None:
    fn = _returning(lambda g: lucid.complex(g * 3, g))
    x = lucid.ones(2, 3, device=device, requires_grad=True)
    fn.apply(x * 1.0).sum().backward()
    assert x.grad.dtype == lucid.float32
    assert x.grad.tolist() == [[3.0] * 3] * 2


# ── Device ───────────────────────────────────────────────────────────────────


def test_a_gradient_on_another_device_is_refused(device_gpu_only: str) -> None:
    fn = _returning(lambda g: g.to("cpu"))
    x = lucid.ones(2, 3, device="metal", requires_grad=True)
    with pytest.raises(
        RuntimeError,
        match="ReturningBackward returned an invalid gradient at index 0 - "
        "expected device metal but got cpu",
    ):
        fn.apply(x).sum().backward()
    assert x.grad is None


def test_a_zero_dim_gradient_is_moved_to_the_input_s_device(
    device_gpu_only: str,
) -> None:
    fn = _returning(lambda g: lucid.tensor(5.0))
    x = lucid.tensor(1.0, device="metal", requires_grad=True)
    fn.apply(x).backward()
    assert x.grad.device == lucid.device("metal")
    assert x.grad.item() == 5.0


# ── How many gradients ───────────────────────────────────────────────────────


class _ScaleBy(Function):
    """``x * k`` with ``k`` a plain number, in either argument order."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.x_first = isinstance(a, lucid.Tensor)
        return a * b

    @staticmethod
    def backward(ctx, g):
        return (g * 3, None) if ctx.x_first else (None, g * 3)


@pytest.mark.parametrize("x_first", [True, False], ids=["x-then-k", "k-then-x"])
def test_one_gradient_per_positional_argument_is_accepted(x_first: bool, device: str) -> None:
    x = lucid.ones(2, 3, device=device, requires_grad=True)
    out = _ScaleBy.apply(x, 3) if x_first else _ScaleBy.apply(3, x)
    out.sum().backward()
    assert x.grad.tolist() == [[3.0] * 3] * 2


def test_trailing_nones_past_the_arguments_are_ignored(device: str) -> None:
    fn = _returning(lambda g: (g, None, None))
    x = lucid.ones(3, device=device, requires_grad=True)
    fn.apply(x).sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 1.0]


def test_a_gradient_for_a_non_tensor_argument_is_refused(device: str) -> None:
    class Bad(Function):
        @staticmethod
        def forward(ctx, x, k):
            return x * k

        @staticmethod
        def backward(ctx, g):
            return g, g

    x = lucid.ones(3, device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError,
        match="function BadBackward returned a gradient different than None at "
        "position 2, but the corresponding forward input was not a Tensor",
    ):
        Bad.apply(x, 2).sum().backward()


def test_the_wrong_number_of_gradients_is_refused(device: str) -> None:
    fn = _returning(lambda g: (g, g))
    x = lucid.ones(3, device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r"function ReturningBackward returned an incorrect number of "
        r"gradients \(expected 1, got 2\)",
    ):
        fn.apply(x).sum().backward()


# ── Against the reference ────────────────────────────────────────────────────

#: (input shape, gradient shape) — accepted, summed, or refused.
SHAPE_CASES = {
    "same": ((2, 5), (2, 5)),
    "short": ((2, 5), (3,)),
    "long": ((4,), (2, 3)),
    "leading": ((2, 5), (3, 2, 5)),
    "size-1 axis": ((2, 1), (2, 5)),
    "input bigger": ((2, 5), (1, 5)),
    "0-d input": ((), (1,)),
}


def _outcome(make_ones, apply_twice, shape, grad_shape):
    """``x.grad`` as a list, or the refusal message."""
    x = make_ones(shape, True)
    try:
        apply_twice(x, lambda g: make_ones(grad_shape, False)).sum().backward()
    except RuntimeError as e:
        return str(e)
    return x.grad.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("case", list(SHAPE_CASES), ids=list(SHAPE_CASES))
def test_shapes_are_judged_as_the_reference_judges_them(case: str, device: str, ref) -> None:
    shape, grad_shape = SHAPE_CASES[case]

    def lucid_apply(x, grad_fn):
        return _returning(grad_fn).apply(x)

    def ref_apply(x, grad_fn):
        class Returning(ref.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return grad_fn(g)

        return Returning.apply(x)

    got = _outcome(
        lambda s, rg: lucid.ones(s, device=device, requires_grad=rg),
        lucid_apply,
        shape,
        grad_shape,
    )
    want = _outcome(lambda s, rg: ref.ones(s, requires_grad=rg), ref_apply, shape, grad_shape)
    if isinstance(want, str):
        # The reference's wording, from "returned an invalid gradient" on.
        assert isinstance(got, str), got
        assert got[got.index("returned") :] == want[want.index("returned") :]
    else:
        assert got == want


# ── What backward returns must be a tensor or None (LCD-247) ────────────────
#
# Anything else used to become "no gradient": a ``backward`` returning ``3``
# finished without error and left ``x.grad`` at ``None``.  The reference
# raises TypeError, and so does ``_validate`` now — in the eager and the
# ``create_graph`` paths alike, since both call the same ``backward_fn``.

_RETURNS = {
    "int": lambda g: 3,
    "float": lambda g: 1.5,
    # A list is read as the sequence of gradients, one per input — so a
    # one-element list holding a plain list of numbers.
    "list": lambda g: [[1.0, 1.0]],
    "ndarray": lambda g: g.detach().cpu().numpy(),
    "tensor": lambda g: g * 3,
    "None": lambda g: None,
}


@pytest.mark.parametrize("create_graph", [False, True], ids=["eager", "create-graph"])
@pytest.mark.parametrize("kind", list(_RETURNS))
def test_a_result_that_is_not_a_tensor_is_a_type_error(
    kind: str, create_graph: bool, device: str
) -> None:
    fn = _returning(_RETURNS[kind])
    x = lucid.ones(2, device=device, requires_grad=True)
    loss = fn.apply(x).sum()
    if kind == "tensor":
        loss.backward(create_graph=create_graph)
        assert x.grad.tolist() == [3.0, 3.0]
    elif kind == "None":
        loss.backward(create_graph=create_graph)
        assert x.grad is None
    else:
        with pytest.raises(TypeError, match="ReturningBackward.*index 0"):
            loss.backward(create_graph=create_graph)
        assert x.grad is None


def test_a_non_tensor_is_refused_even_for_an_input_needing_no_gradient(
    device: str,
) -> None:
    class Two(Function):
        @staticmethod
        def forward(ctx, a, b):
            return a * b

        @staticmethod
        def backward(ctx, g):
            return g, 3

    x = lucid.ones(2, device=device, requires_grad=True)
    k = lucid.ones(2, device=device)
    with pytest.raises(TypeError, match="index 1"):
        Two.apply(x, k).sum().backward()


def test_mark_non_differentiable_detaches_marked_outputs(device: str) -> None:
    class ValueAndIndex(Function):
        @staticmethod
        def forward(ctx, x):
            idx = x * 0.0
            ctx.mark_non_differentiable(idx)
            return x * 2.0, idx

        @staticmethod
        def backward(ctx, grad_val, grad_idx):
            del ctx, grad_idx
            return grad_val * 2.0

    x = lucid.ones(3, device=device, requires_grad=True)
    val, idx = ValueAndIndex.apply(x)
    assert val.requires_grad
    assert not idx.requires_grad
    assert idx.grad_fn is None
    val.sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [2.0, 2.0, 2.0]
