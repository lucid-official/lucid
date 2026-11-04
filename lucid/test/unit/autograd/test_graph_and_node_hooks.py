"""Unit and reference-parity tests for LCD-785:
- ``Tensor.register_post_accumulate_grad_hook``
- ``Node.next_functions``, ``Node.register_hook``, ``Node.register_prehook``,
  ``AccumulateGrad.variable``
- ``saved_tensors_hooks`` and ``save_on_cpu`` on built-in C++ ops
"""

from types import ModuleType
import pytest

import lucid
from lucid.autograd.graph import save_on_cpu, saved_tensors_hooks


def _vec(lib: ModuleType, device: str, *vals: float, requires_grad: bool = True):
    return lib.tensor(list(vals), dtype=lib.float32, device=device, requires_grad=requires_grad)


# ============================================================================
# 1. Tensor.register_post_accumulate_grad_hook
# ============================================================================


def test_post_accumulate_grad_hook_requires_leaf_and_grad(device: str) -> None:
    x_no_grad = lucid.tensor([1.0, 2.0], device=device, requires_grad=False)
    with pytest.raises(RuntimeError, match="require"):
        x_no_grad.register_post_accumulate_grad_hook(lambda t: None)

    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    y = x * 2.0
    with pytest.raises(RuntimeError, match="leaf"):
        y.register_post_accumulate_grad_hook(lambda t: None)


@pytest.mark.parity
def test_post_accumulate_grad_hook_basic_and_in_place_mutation(
    ref: ModuleType, device: str
) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[list[list[float]], list[float] | None]:
        seen: list[list[float]] = []
        x = _vec(lib, dev, 2.0, 3.0)

        def hook(param) -> None:
            assert param is x
            assert param.grad is not None
            seen.append(param.grad.detach().cpu().tolist())
            param.grad.mul_(10.0)

        x.register_post_accumulate_grad_hook(hook)
        (x * x).sum().backward()
        return seen, x.grad.detach().cpu().tolist()

    assert run(lucid, device) == run(ref, "cpu")


@pytest.mark.parity
def test_post_accumulate_grad_hook_can_clear_grad_like_fused_optimizer(
    ref: ModuleType, device: str
) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[list[float], bool]:
        x = _vec(lib, dev, 4.0, 5.0)

        def fused_step(param) -> None:
            with lib.no_grad():
                param.sub_(0.5 * param.grad)
            param.grad = None

        x.register_post_accumulate_grad_hook(fused_step)
        (x * x).sum().backward()
        return x.detach().cpu().tolist(), x.grad is None

    assert run(lucid, device) == run(ref, "cpu")


def test_post_accumulate_grad_hook_rejects_non_none_return(device: str) -> None:
    x = _vec(lucid, device, 1.0, 2.0)
    x.register_post_accumulate_grad_hook(lambda t: t)  # type: ignore[arg-type,return-value]
    with pytest.raises(RuntimeError, match="post accumulate grad hooks should return None"):
        (x * 2.0).sum().backward()


@pytest.mark.parity
def test_post_accumulate_grad_hook_chaining_and_removal(ref: ModuleType, device: str) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[list[str], list[float]]:
        events: list[str] = []
        x = _vec(lib, dev, 1.0, 2.0)
        h1 = x.register_post_accumulate_grad_hook(
            lambda p: (events.append("h1"), p.grad.add_(1.0))[0] and None
        )
        x.register_post_accumulate_grad_hook(
            lambda p: (events.append("h2"), p.grad.mul_(2.0))[0] and None
        )
        (x * 3.0).sum().backward()
        h1.remove()
        (x * 3.0).sum().backward()
        return events, x.grad.detach().cpu().tolist()

    assert run(lucid, device) == run(ref, "cpu")


@pytest.mark.parity
def test_post_accumulate_grad_hook_does_not_fire_on_autograd_grad(
    ref: ModuleType, device: str
) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[int, list[float]]:
        calls = 0
        x = _vec(lib, dev, 2.0, 3.0)

        def hook(_p) -> None:
            nonlocal calls
            calls += 1

        x.register_post_accumulate_grad_hook(hook)
        (gx,) = lib.autograd.grad((x * x).sum(), (x,))
        return calls, gx.detach().cpu().tolist()

    assert run(lucid, device) == run(ref, "cpu")


# ============================================================================
# 2. grad_fn.next_functions, AccumulateGrad.variable, Node hooks & order
# ============================================================================


def test_next_functions_and_accumulate_grad_variable(device: str) -> None:
    x = _vec(lucid, device, 1.0, 2.0)
    c = _vec(lucid, device, 3.0, 4.0, requires_grad=False)
    u = x * c
    y = u.sin()

    assert y.grad_fn is not None
    nf_y = y.grad_fn.next_functions
    assert len(nf_y) == 1
    mul_node, nr0 = nf_y[0]
    assert mul_node is u.grad_fn
    assert nr0 == 0

    assert mul_node is not None
    nf_u = mul_node.next_functions
    assert len(nf_u) == 2
    acc_node, acc_nr = nf_u[0]
    const_node, const_nr = nf_u[1]
    assert acc_node is not None
    assert acc_nr == 0
    assert hasattr(acc_node, "variable")
    assert acc_node.variable is not None
    assert acc_node.variable._impl is x._impl
    assert acc_node.variable.detach().cpu().tolist() == [1.0, 2.0]
    assert acc_node.next_functions == ()
    assert const_node is None
    assert const_nr == 0


@pytest.mark.parity
def test_node_prehook_and_posthook_modify_gradients(ref: ModuleType, device: str) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[list[float], list[float]]:
        a = _vec(lib, dev, 2.0, 3.0)
        b = _vec(lib, dev, 4.0, 5.0)
        y = a * b
        assert y.grad_fn is not None

        # Prehook scales incoming grad_outputs by 2
        y.grad_fn.register_prehook(lambda grad_outputs: (grad_outputs[0] * 2.0,))
        # Posthook scales grad_input[0] by 3 and leaves grad_input[1] as-is
        y.grad_fn.register_hook(
            lambda grad_inputs, _grad_outputs: (grad_inputs[0] * 3.0, grad_inputs[1])
        )

        y.sum().backward()
        return a.grad.detach().cpu().tolist(), b.grad.detach().cpu().tolist()

    assert run(lucid, device) == run(ref, "cpu")


@pytest.mark.parity
def test_node_posthook_can_drop_branch_with_none(ref: ModuleType, device: str) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[bool, list[float]]:
        a = _vec(lib, dev, 2.0, 3.0)
        b = _vec(lib, dev, 4.0, 5.0)
        y = a * b
        assert y.grad_fn is not None
        y.grad_fn.register_hook(lambda grad_inputs, _grad_outputs: (None, grad_inputs[1]))
        y.sum().backward()
        return a.grad is None, b.grad.detach().cpu().tolist()

    assert run(lucid, device) == run(ref, "cpu")


@pytest.mark.parity
def test_accumulate_grad_five_stage_hook_execution_order(ref: ModuleType, device: str) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[list[str], list[float]]:
        order: list[str] = []
        x = _vec(lib, dev, 2.0, 3.0)
        y = x * x
        acc_node = y.grad_fn.next_functions[0][0]
        assert acc_node is not None

        x.register_hook(lambda g: (order.append("1:tensor_hook"), g + 1.0)[1])
        acc_node.register_prehook(lambda go: (order.append("2:acc_prehook"), (go[0] * 2.0,))[1])
        x.register_post_accumulate_grad_hook(
            lambda p: (
                order.append(f"3:post_acc(grad={p.grad.detach().cpu().tolist()})"),
                p.grad.add_(5.0),
            )[0]
            and None
        )
        acc_node.register_hook(
            lambda gi, go: (
                order.append(f"4:acc_posthook(grad={x.grad.detach().cpu().tolist()})"),
                None,
            )[1]
        )

        y.sum().backward()
        return order, x.grad.detach().cpu().tolist()

    assert run(lucid, device) == run(ref, "cpu")


@pytest.mark.parity
def test_node_hooks_under_create_graph_second_derivative(ref: ModuleType, device: str) -> None:
    def run(lib: ModuleType, dev: str) -> list[float]:
        x = _vec(lib, dev, 2.0, 3.0)
        y = x * x * x
        assert y.grad_fn is not None
        y.grad_fn.register_prehook(lambda go: (go[0] * 2.0,))
        (gx,) = lib.autograd.grad(y.sum(), (x,), create_graph=True)
        gx.sum().backward()
        return x.grad.detach().cpu().tolist()

    assert run(lucid, device) == run(ref, "cpu")


# ============================================================================
# 3. saved_tensors_hooks and save_on_cpu on built-in C++ ops
# ============================================================================


@pytest.mark.parity
def test_saved_tensors_hooks_built_in_input_and_output_saving_ops(
    ref: ModuleType, device: str
) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[int, int, list[float]]:
        packs = 0
        unpacks = 0

        def pack(t):
            nonlocal packs
            packs += 1
            return t.detach().clone()

        def unpack(p):
            nonlocal unpacks
            unpacks += 1
            return p

        x = _vec(lib, dev, 0.5, 1.2)
        with lib.autograd.graph.saved_tensors_hooks(pack, unpack):
            h = (x * x).sin().exp().sigmoid().tanh()
            y = lib.nn.functional.softmax(h, dim=0)
        y.sum().backward()
        return packs, unpacks, x.grad.detach().cpu().tolist()

    lp, lu, lg = run(lucid, device)
    rp, ru, rg = run(ref, "cpu")
    assert (lp, lu) == (rp, ru)
    assert pytest.approx(lg, rel=1e-5, abs=1e-5) == rg


@pytest.mark.parity
def test_save_on_cpu_built_in_mlp_offloads_to_host_and_matches_reference(
    ref: ModuleType, device: str
) -> None:
    held_devices: list[str] = []
    restored_devices: list[str] = []
    ctx = save_on_cpu()
    orig_pack, orig_unpack = ctx.pack_hook, ctx.unpack_hook

    def spy_pack(t):
        packed = orig_pack(t)
        held_devices.append(packed.host.device.type)
        return packed

    def spy_unpack(p):
        t = orig_unpack(p)
        restored_devices.append(t.device.type)
        return t

    ctx.pack_hook, ctx.unpack_hook = spy_pack, spy_unpack

    x = lucid.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=lucid.float32, device=device, requires_grad=True)
    w1 = lucid.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=lucid.float32, device=device, requires_grad=True)
    w2 = lucid.tensor([[-0.5, 0.7], [0.3, -0.2]], dtype=lucid.float32, device=device, requires_grad=True)

    with ctx:
        h = (x @ w1).sin()
        out = (h @ w2).exp()
    out.sum().backward()

    assert len(held_devices) == 6
    assert set(held_devices) == {"cpu"}
    assert set(restored_devices) == {x.device.type}

    rx = ref.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=ref.float32, device="cpu", requires_grad=True)
    rw1 = ref.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=ref.float32, device="cpu", requires_grad=True)
    rw2 = ref.tensor([[-0.5, 0.7], [0.3, -0.2]], dtype=ref.float32, device="cpu", requires_grad=True)
    with ref.autograd.graph.save_on_cpu():
        rh = (rx @ rw1).sin()
        rout = (rh @ rw2).exp()
    rout.sum().backward()

    assert pytest.approx(x.grad.detach().cpu().reshape((-1,)).tolist(), rel=1e-5, abs=1e-5) == (
        rx.grad.detach().cpu().reshape((-1,)).tolist()
    )
    assert pytest.approx(w1.grad.detach().cpu().reshape((-1,)).tolist(), rel=1e-5, abs=1e-5) == (
        rw1.grad.detach().cpu().reshape((-1,)).tolist()
    )
    assert pytest.approx(w2.grad.detach().cpu().reshape((-1,)).tolist(), rel=1e-5, abs=1e-5) == (
        rw2.grad.detach().cpu().reshape((-1,)).tolist()
    )


@pytest.mark.parity
def test_saved_tensors_hooks_built_in_retain_graph_and_create_graph(
    ref: ModuleType, device: str
) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[int, list[float], list[float]]:
        unpacks = 0

        def pack(t):
            return t.detach().clone()

        def unpack(p):
            nonlocal unpacks
            unpacks += 1
            return p

        x = _vec(lib, dev, 1.0, 2.0)
        with lib.autograd.graph.saved_tensors_hooks(pack, unpack):
            y = (x * x).exp()

        (gx,) = lib.autograd.grad(y.sum(), (x,), create_graph=True, retain_graph=True)
        gx.sum().backward(retain_graph=True)
        second_deriv = x.grad.detach().cpu().tolist()

        x.grad = None
        y.sum().backward()
        first_deriv = x.grad.detach().cpu().tolist()
        return unpacks, first_deriv, second_deriv

    lu, lf, ls = run(lucid, device)
    ru, rf, rs = run(ref, "cpu")
    assert lu == ru
    assert pytest.approx(lf, rel=1e-5, abs=1e-5) == rf
    assert pytest.approx(ls, rel=1e-5, abs=1e-5) == rs


def test_saved_tensors_hooks_built_in_skips_version_check_when_copied(device: str) -> None:
    x = _vec(lucid, device, 1.0, 2.0)
    w = x * 1.0
    with saved_tensors_hooks(lambda t: t.detach().clone(), lambda p: p):
        y = w * w
    w.add_(10.0)
    y.sum().backward()
    assert x.grad is not None and x.grad.detach().cpu().tolist() == [2.0, 4.0]
