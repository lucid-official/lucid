"""``p.grad = g`` refuses a gradient that is not of ``p``'s kind (CHA-72).

A gradient lives in its tensor as a bare buffer, and ``.grad``, the
optimizers and the next backward read that buffer with the tensor's own
dtype, shape and device.  The assignment used to put ``g``'s buffer there
as it was: a float32 gradient on a float16 parameter came back as float16
bits (``[1, 2, 4]`` read ``[0, 1.875, 1.875]``, CPU and Metal alike), a
shorter gradient was read past its end, and a Metal array sat in a CPU
tensor's slot.  The reference refuses all of these with a ``RuntimeError``
naming what differs, and so does Lucid now, leaving the old gradient as it
was.  Every internal write-back of a whole gradient goes the same way
(tensor hooks, ``clip_grad_*``, ``GradScaler``), so those are checked too.
"""

from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.nn.utils import clip_grad_norm_, clip_grad_value_
from lucid.test._fixtures.devices import metal_available

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")


def _param(
    values: list[float], device: str, dtype: lucid.dtype = lucid.float32
) -> lucid.Tensor:
    return lucid.nn.Parameter(lucid.tensor(values, dtype=dtype, device=device))


# ── the reported case ──────────────────────────────────────────────────────


def test_float32_gradient_on_a_float16_parameter_is_refused(device: str) -> None:
    p = _param([1.0, 1.0, 1.0], device, lucid.float16)
    with pytest.raises(RuntimeError, match="dtype"):
        p.grad = lucid.tensor([1.0, 2.0, 4.0], device=device)
    assert p.grad is None


@pytest.mark.parametrize(
    ("param_dtype", "grad_dtype"),
    [
        (lucid.float16, lucid.float32),
        (lucid.float32, lucid.float16),
        (lucid.float32, lucid.bfloat16),
        (lucid.bfloat16, lucid.float16),
        (lucid.float32, lucid.int32),
    ],
    ids=["f32-on-f16", "f16-on-f32", "bf16-on-f32", "f16-on-bf16", "i32-on-f32"],
)
def test_every_dtype_mismatch_is_refused(
    device: str, param_dtype: lucid.dtype, grad_dtype: lucid.dtype
) -> None:
    p = lucid.ones(3, dtype=param_dtype, device=device)
    with pytest.raises(_C_engine.DtypeMismatch):
        p.grad = lucid.ones(3, dtype=grad_dtype, device=device)


@pytest.mark.parametrize(
    "grad_dtype", [lucid.float64, lucid.complex64], ids=["f64", "c64"]
)
def test_wider_dtypes_are_refused_on_the_cpu(grad_dtype: lucid.dtype) -> None:
    # Metal holds no float64, so these two pairs only exist on the CPU.
    p = lucid.ones(2, requires_grad=True)
    with pytest.raises(_C_engine.DtypeMismatch):
        p.grad = lucid.ones(2, dtype=grad_dtype)


# ── shape and device ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("param_shape", "grad_shape"),
    [((3,), (3, 1)), ((3,), (4,)), ((3,), (2,)), ((), (1,)), ((0, 3), (3, 0))],
    ids=["extra-axis", "longer", "shorter", "0d-vs-1", "empty-transposed"],
)
def test_every_shape_mismatch_is_refused(
    device: str, param_shape: tuple[int, ...], grad_shape: tuple[int, ...]
) -> None:
    # Same element count is not enough: ``[3, 1]`` and ``[3]`` hold the same
    # bytes, and the reference still refuses.  A shorter gradient was read
    # past its end.
    p = lucid.nn.Parameter(lucid.ones(*param_shape, device=device))
    with pytest.raises(_C_engine.ShapeMismatch):
        p.grad = lucid.ones(*grad_shape, device=device)
    assert p.grad is None


@_needs_metal
@pytest.mark.parametrize(
    ("param_device", "grad_device"), [("cpu", "metal"), ("metal", "cpu")], ids=str
)
def test_a_gradient_on_another_device_is_refused(
    param_device: str, grad_device: str
) -> None:
    p = _param([1.0, 2.0], param_device)
    with pytest.raises(_C_engine.DeviceMismatch):
        p.grad = lucid.tensor([3.0, 4.0], device=grad_device)
    assert p.grad is None


def test_a_tensor_is_not_its_own_gradient(device: str) -> None:
    p = _param([1.0, 2.0], device)
    with pytest.raises(RuntimeError, match="own gradient"):
        p.grad = p
    # A different tensor over the same values is fine, as in the reference.
    p.grad = p.detach()
    assert p.grad.tolist() == [1.0, 2.0]
    p.grad = p.data
    assert p.grad.tolist() == [1.0, 2.0]


def test_a_refused_assignment_leaves_the_old_gradient(device: str) -> None:
    p = _param([1.0, 2.0], device, lucid.float16)
    p.grad = lucid.tensor([7.0, 8.0], dtype=lucid.float16, device=device)
    for bad in (
        lucid.tensor([1.0, 2.0], device=device),
        lucid.ones(3, dtype=lucid.float16, device=device),
    ):
        with pytest.raises(RuntimeError):
            p.grad = bad
        assert p.grad.dtype == lucid.float16
        assert p.grad.tolist() == [7.0, 8.0]


def test_a_non_tensor_is_a_type_error(device: str) -> None:
    p = _param([1.0, 2.0], device)
    with pytest.raises(TypeError):
        p.grad = [1.0, 2.0]  # type: ignore[assignment]


def test_the_engine_binding_refuses_none() -> None:
    # ``.grad = None`` clears through zero_grad; the binding itself takes a
    # tensor and must not dereference a null one.
    p = _param([1.0, 2.0], "cpu")
    with pytest.raises(ValueError, match="zero_grad"):
        p._impl.set_grad(None)  # type: ignore[arg-type]


# ── what is still accepted, with the values it holds ───────────────────────


@pytest.mark.parametrize(
    "dtype",
    [lucid.float32, lucid.float16, lucid.bfloat16],
    ids=["f32", "f16", "bf16"],
)
def test_a_matching_gradient_is_installed(device: str, dtype: lucid.dtype) -> None:
    p = _param([1.0, 1.0, 1.0], device, dtype)
    p.grad = lucid.tensor([1.0, 2.0, 4.0], dtype=dtype, device=device)
    assert p.grad.dtype == dtype
    assert p.grad.device.type == device
    assert p.grad.tolist() == [1.0, 2.0, 4.0]


def test_integer_tensors_take_integer_gradients(device: str) -> None:
    t = lucid.ones(2, dtype=lucid.int32, device=device)
    t.grad = lucid.full((2,), 3, dtype=lucid.int32, device=device)
    assert t.grad.tolist() == [3, 3]


@pytest.mark.parametrize(
    "make",
    [
        lambda d: lucid.arange(6.0, device=d)[1:4].reshape(3),
        lambda d: lucid.arange(6.0, device=d)[::2],
    ],
    ids=["offset-view", "strided-view"],
)
def test_a_view_gives_its_own_elements(
    device: str, make: Callable[[str], lucid.Tensor]
) -> None:
    # The slot is read from its first byte in row-major order, so a view's
    # values must arrive packed, not as the bytes at the front of its base.
    p = _param([0.0, 0.0, 0.0], device)
    g = make(device)
    p.grad = g
    assert p.grad.tolist() == g.tolist()


def test_a_transposed_or_expanded_gradient_keeps_its_layout(device: str) -> None:
    p = lucid.nn.Parameter(lucid.zeros(2, 3, device=device))
    p.grad = lucid.arange(6.0, device=device).reshape(3, 2).T
    assert p.grad.tolist() == [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]
    p.grad = lucid.arange(3.0, device=device).expand(2, 3)
    assert p.grad.tolist() == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]


def test_zero_d_and_empty_gradients(device: str) -> None:
    s = lucid.nn.Parameter(lucid.tensor(1.0, device=device))
    s.grad = lucid.tensor(5.0, device=device)
    assert s.grad.shape == ()
    assert s.grad.item() == 5.0
    e = lucid.nn.Parameter(lucid.ones(0, 3, device=device))
    e.grad = lucid.zeros(0, 3, device=device)
    assert e.grad.shape == (0, 3)


def test_none_clears_the_gradient(device: str) -> None:
    p = _param([1.0, 2.0], device)
    p.grad = lucid.tensor([3.0, 4.0], device=device)
    p.grad = None
    assert p.grad is None


def test_an_assigned_gradient_is_what_the_optimizer_steps_with(device: str) -> None:
    p = _param([1.0, 2.0], device)
    opt = lucid.optim.SGD([p], lr=1.0)
    p.grad = lucid.tensor([0.5, 0.25], device=device)
    opt.step()
    assert p.tolist() == [0.5, 1.75]


# ── the internal write-backs ───────────────────────────────────────────────


def test_a_hook_that_changes_the_dtype_is_refused(device: str) -> None:
    # The reference refuses a hook that changes its gradient's type, too.
    x = lucid.ones(3, device=device, requires_grad=True)
    x.register_hook(lambda g: g.to(lucid.float16))
    with pytest.raises(_C_engine.DtypeMismatch):
        (x * 2.0).sum().backward()


def test_a_hook_that_changes_the_shape_is_refused(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    x.register_hook(lambda g: g[:2])
    with pytest.raises(_C_engine.ShapeMismatch):
        (x * 2.0).sum().backward()


def test_a_hook_replacement_of_the_same_kind_is_installed(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    x.register_hook(lambda g: g * 3.0)
    (x * 2.0).sum().backward()
    assert x.grad.tolist() == [6.0, 6.0, 6.0]


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["f16", "bf16"])
def test_clipping_half_precision_gradients_keeps_them(
    device: str, dtype: lucid.dtype
) -> None:
    p = _param([1.0, 1.0], device, dtype)
    p.grad = lucid.tensor([3.0, -4.0], dtype=dtype, device=device)
    clip_grad_value_([p], clip_value=1.0)
    assert p.grad.dtype == dtype
    assert p.grad.tolist() == [1.0, -1.0]
    p.grad = lucid.tensor([3.0, 4.0], dtype=dtype, device=device)
    clip_grad_norm_([p], max_norm=1.0)
    assert p.grad.dtype == dtype
    assert p.grad.tolist() == pytest.approx([0.6, 0.8], abs=1e-2)


def test_the_engine_binding_is_checked_like_the_setter(device: str) -> None:
    p = _param([1.0, 1.0, 1.0], device, lucid.float16)
    with pytest.raises(_C_engine.DtypeMismatch):
        p._impl.set_grad(lucid.tensor([1.0, 2.0, 4.0], device=device)._impl)
    assert p.grad is None


# ── against the reference ──────────────────────────────────────────────────

# (param maker, grad maker) on the CPU, with what the refusal is about —
# the word the reference's message and Lucid's both name.
_CASES: dict[
    str, tuple[Callable[[object], object], Callable[[object], object], str | None]
] = {
    "dtype": (
        lambda m: m.ones(3, dtype=m.float16),
        lambda m: m.ones(3, dtype=m.float32),
        "dtype",
    ),
    "dtype-f64": (
        lambda m: m.ones(3, dtype=m.float32),
        lambda m: m.ones(3, dtype=m.float64),
        "dtype",
    ),
    "same-numel": (lambda m: m.ones(3), lambda m: m.ones(3, 1), "size"),
    "longer": (lambda m: m.ones(3), lambda m: m.ones(4), "size"),
    "zero-d": (lambda m: m.ones(()), lambda m: m.ones(1), "size"),
    "match": (lambda m: m.ones(2, 3), lambda m: m.ones(2, 3), None),
    "transposed": (lambda m: m.ones(2, 3), lambda m: m.ones(3, 2).T, None),
    "int": (
        lambda m: m.ones(3, dtype=m.int32),
        lambda m: m.ones(3, dtype=m.int32),
        None,
    ),
}

_LUCID_WORD = {"dtype": "dtype", "size": "shape"}


def _outcome(module: object, case: str) -> str | None:
    make_param, make_grad, _ = _CASES[case]
    p = make_param(module)
    try:
        p.grad = make_grad(module)  # type: ignore[attr-defined]
    except RuntimeError as e:
        return str(e)
    return None


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_refuses_what_the_reference_refuses(case: str, ref: object) -> None:
    word = _CASES[case][2]
    want = _outcome(ref, case)
    got = _outcome(lucid, case)
    if word is None:
        assert want is None and got is None
    else:
        assert want is not None and word in want
        assert got is not None and _LUCID_WORD[word] in got


@pytest.mark.parity
@_needs_metal
def test_refuses_a_device_mismatch_like_the_reference(ref: object) -> None:
    rp = ref.ones(3, device="mps")  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="device"):
        rp.grad = ref.ones(3)  # type: ignore[attr-defined]
    lp = lucid.ones(3, device="metal")
    with pytest.raises(RuntimeError, match="device"):
        lp.grad = lucid.ones(3)


@pytest.mark.parity
def test_refuses_self_assignment_like_the_reference(ref: object) -> None:
    rp = ref.nn.Parameter(ref.ones(3))  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        rp.grad = rp
    lp = lucid.nn.Parameter(lucid.ones(3))
    with pytest.raises(RuntimeError):
        lp.grad = lp


# ── after backward(create_graph=True) (LCD-81) ─────────────────────────────


def test_set_grad_after_create_graph_replaces_graph_grad_with_plain_tensor(
    device: str,
) -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    (w * w).sum().backward(create_graph=True)
    assert w.grad is not None and w.grad.requires_grad
    assert w.grad.tolist() == [2.0, 4.0, 6.0]

    w.grad = lucid.zeros(3, device=device)
    assert w.grad is not None
    assert w.grad.tolist() == [0.0, 0.0, 0.0]
    assert not w.grad.requires_grad
    assert w.grad.grad_fn is None


def test_set_grad_after_create_graph_preserves_graph_when_assigned_grad_tracks_autograd(
    device: str,
) -> None:
    w = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    (w * w).sum().backward(create_graph=True)
    w.grad = w * 5.0
    assert w.grad is not None
    assert w.grad.tolist() == [5.0, 10.0, 15.0]
    assert w.grad.requires_grad
    (g2,) = lucid.autograd.grad(w.grad.sum(), [w])
    assert g2.tolist() == [5.0, 5.0, 5.0]


def test_clip_grad_norm_and_value_after_create_graph(device: str) -> None:
    w = _param([3.0, 4.0], device)
    (w * w).sum().backward(create_graph=True)  # grad = [6.0, 8.0], norm = 10.0
    clip_grad_norm_([w], max_norm=1.0)
    assert w.grad is not None
    assert w.grad.tolist() == pytest.approx([0.6, 0.8], abs=1e-4)
    assert w.grad.requires_grad

    clip_grad_value_([w], clip_value=0.7)
    assert w.grad is not None
    assert w.grad.tolist() == pytest.approx([0.6, 0.7], abs=1e-4)

    # Also verify clipping under no_grad() replaces the graph-mode grad cleanly.
    w2 = _param([3.0, 4.0], device)
    (w2 * w2).sum().backward(create_graph=True)
    with lucid.no_grad():
        clip_grad_norm_([w2], max_norm=1.0)
    assert w2.grad is not None
    assert w2.grad.tolist() == pytest.approx([0.6, 0.8], abs=1e-4)
    assert not w2.grad.requires_grad


def test_mixed_create_graph_and_eager_backward_accumulates(device: str) -> None:
    # create_graph=True followed by create_graph=False
    w = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    (w * w).sum().backward(create_graph=True)  # + [2.0, 4.0] (graph)
    (w * 3.0).sum().backward(create_graph=False)  # + [3.0, 3.0] (eager)
    assert w.grad is not None
    assert w.grad.tolist() == [5.0, 7.0]
    assert w.grad.requires_grad

    # create_graph=False followed by create_graph=True
    w2 = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    (w2 * 3.0).sum().backward(create_graph=False)  # + [3.0, 3.0] (eager)
    (w2 * w2).sum().backward(create_graph=True)  # + [2.0, 4.0] (graph)
    assert w2.grad is not None
    assert w2.grad.tolist() == [5.0, 7.0]
    assert w2.grad.requires_grad


@pytest.mark.parity
def test_create_graph_grad_assignment_and_clipping_parity(ref: object) -> None:
    rw = ref.nn.Parameter(ref.tensor([3.0, 4.0]))  # type: ignore[attr-defined]
    (rw * rw).sum().backward(create_graph=True)
    ref.nn.utils.clip_grad_norm_([rw], max_norm=1.0)  # type: ignore[attr-defined]

    lw = lucid.nn.Parameter(lucid.tensor([3.0, 4.0]))
    (lw * lw).sum().backward(create_graph=True)
    clip_grad_norm_([lw], max_norm=1.0)

    assert lw.grad is not None
    assert lw.grad.tolist() == pytest.approx(rw.grad.tolist(), abs=1e-5)
    assert lw.grad.requires_grad == rw.grad.requires_grad

    rw.grad = ref.zeros(2)  # type: ignore[attr-defined]
    lw.grad = lucid.zeros(2)
    assert lw.grad.tolist() == rw.grad.tolist()
    assert lw.grad.requires_grad == rw.grad.requires_grad
