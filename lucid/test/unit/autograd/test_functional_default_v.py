"""``vjp`` / ``jvp`` take ``v=None`` where the reference does (LCD-391).

``vjp(lambda x: (x * x).sum(), x)`` raised "missing 1 required positional
argument: 'v'".  The reference's ``v`` defaults to ``None``, which it
reads as a seed of 1 when the side ``v`` would cover — the outputs for
``vjp``, the inputs for ``jvp`` — is one single-element tensor, and
refuses otherwise.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.autograd import jvp, vjp

_DATA = [1.0, -2.0, 3.0]


@pytest.mark.parametrize("keep", [False, True], ids=["0-d", "shape-1"])
def test_vjp_seeds_a_single_element_output(ref: ModuleType, device: str, keep: bool) -> None:
    def cube_sum(x: lucid.Tensor) -> lucid.Tensor:
        s = (x * x * x).sum()
        return s.reshape(1) if keep else s

    out, grad = vjp(cube_sum, lucid.tensor(_DATA, device=device))
    want_out, want_grad = ref.autograd.functional.vjp(cube_sum, ref.tensor(_DATA))
    assert out.shape == tuple(want_out.shape)
    assert out.tolist() == pytest.approx(want_out.tolist())
    assert grad.tolist() == pytest.approx(want_grad.tolist())


@pytest.mark.parametrize("shape", [(), (1,)], ids=["0-d", "shape-1"])
def test_jvp_seeds_a_single_element_input(
    ref: ModuleType, device: str, shape: tuple[int, ...]
) -> None:
    x = lucid.full(shape, 3.0, device=device)
    out, tangent = jvp(lambda t: t * t * t, x)
    want_out, want_tangent = ref.autograd.functional.jvp(lambda t: t * t * t, ref.full(shape, 3.0))
    assert tangent.shape == tuple(want_tangent.shape)
    assert tangent.tolist() == pytest.approx(want_tangent.tolist())
    assert out.tolist() == pytest.approx(want_out.tolist())


@pytest.mark.parametrize(
    "call",
    [
        lambda m, F: F.vjp(lambda x: x * x, m.tensor(_DATA)),
        lambda m, F: F.vjp(lambda x: (x.sum(), x.sum()), m.tensor(_DATA)),
        lambda m, F: F.jvp(lambda x: x * x, m.tensor(_DATA)),
        lambda m, F: F.jvp(lambda a, b: a * b, (m.tensor(3.0), m.tensor(2.0))),
    ],
    ids=["vjp-vector-out", "vjp-two-outs", "jvp-vector-in", "jvp-two-ins"],
)
def test_v_left_out_elsewhere_is_refused(
    ref: ModuleType, call: Callable[[ModuleType, object], object]
) -> None:
    with pytest.raises(RuntimeError, match="v can only be None"):
        call(ref, ref.autograd.functional)
    with pytest.raises(RuntimeError, match="v can only be None"):
        call(lucid, lucid.autograd)


def test_the_default_seed_under_create_graph(device: str) -> None:
    x = lucid.tensor(_DATA, device=device, requires_grad=True)
    _, grad = vjp(lambda t: (t * t * t).sum(), x, create_graph=True)
    (second,) = lucid.autograd.grad(grad.sum(), x)
    assert second.tolist() == pytest.approx([6.0 * v for v in _DATA])


def test_one_tensor_in_a_tuple_is_seeded_too(ref: ModuleType, device: str) -> None:
    # ``(loss,)`` and ``(x,)`` cover one single-element tensor as a bare one
    # does; the reference seeds both.
    data = [2.0]
    x = lucid.tensor(data, device=device)
    _, grad = vjp(lambda a: ((a * a).sum(),), x)
    _, want = ref.autograd.functional.vjp(lambda a: ((a * a).sum(),), ref.tensor(data))
    assert grad.tolist() == want.tolist()
    _, tangent = jvp(lambda a: a * 3, (x,))
    _, want_t = ref.autograd.functional.jvp(lambda a: a * 3, (ref.tensor(data),))
    assert tangent.tolist() == want_t.tolist()
