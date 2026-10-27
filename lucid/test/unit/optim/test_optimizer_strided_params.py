"""An optimizer step writes a parameter's own elements, wherever they sit.

The engine optimizers stepped every parameter as though its elements were
the first ``numel`` of its buffer, in row-major order.  A parameter that is
a view at an offset or through other strides — ``Parameter(x[:, 2])``,
``Parameter(x.t())``, two parameters over one tensor — had the wrong
elements written: the column view stepped the row at the buffer's start and
corrupted its neighbours, and a transpose paired each gradient with another
element.  Metal refused a view of part of a tensor outright (LCD-476).

Every gradient here differs element by element.  A gradient that is the
same everywhere hid the transpose: whichever element it was paired with,
the step was the same.

Metal views of part of a tensor are built with the engine's ``_make_view``
hook, the same family member an engine view op makes.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import device_dtype_params

_STEPS = 3
_BASE_SHAPE = (4, 6)
_TOL = 1e-5

# Every optimizer, with the flags that take another path through its update.
_OPTIMIZERS: list[tuple[str, dict[str, Any]]] = [
    ("SGD", {"lr": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9, "dampening": 0.3, "weight_decay": 0.1}),
    ("SGD", {"lr": 0.1, "momentum": 0.9, "nesterov": True}),
    ("Adam", {"lr": 0.05}),
    ("Adam", {"lr": 0.05, "weight_decay": 0.1, "amsgrad": True}),
    ("AdamW", {"lr": 0.05, "weight_decay": 0.1}),
    ("AdamW", {"lr": 0.05, "amsgrad": True}),
    ("Adamax", {"lr": 0.05, "weight_decay": 0.1}),
    ("NAdam", {"lr": 0.05, "weight_decay": 0.1}),
    ("RAdam", {"lr": 0.05, "betas": (0.7, 0.5)}),
    ("RMSprop", {"lr": 0.01}),
    ("RMSprop", {"lr": 0.01, "centered": True, "momentum": 0.9, "weight_decay": 0.1}),
    ("Rprop", {"lr": 0.05}),
    ("Adagrad", {"lr": 0.1, "lr_decay": 0.05, "initial_accumulator_value": 0.1}),
    ("Adadelta", {"lr": 0.5, "weight_decay": 0.1}),
    ("ASGD", {"lr": 0.05, "t0": 0.0, "lambd": 0.1}),
    ("SparseAdam", {"lr": 0.05}),
]

_OPTIMIZER_PARAMS = [
    pytest.param(c, id=c[0] + "".join(f"-{k}={v}" for k, v in c[1].items())) for c in _OPTIMIZERS
]

# Each parameter's (shape, stride, offset) over the 4 x 6 base, in elements.
_LAYOUTS: dict[str, list[tuple[tuple[int, ...], tuple[int, ...], int]]] = {
    "column": [((4,), (6,), 2)],
    "offset_block": [((2, 3), (6, 1), 7)],
    "transpose": [((6, 4), (1, 6), 0)],
    # A column and a transposed 2 x 2 block, disjoint, over one tensor.
    "two_params_one_base": [((4,), (6,), 1), ((2, 2), (1, 6), 15)],
}


def _covered(layout: Sequence[tuple[tuple[int, ...], tuple[int, ...], int]]) -> np.ndarray:
    """Which elements of the base the layout's parameters read."""
    mask = np.zeros(int(np.prod(_BASE_SHAPE)), dtype=bool)
    for shape, stride, offset in layout:
        for index in np.ndindex(*shape):
            mask[offset + sum(i * s for i, s in zip(index, stride, strict=True))] = True
    return mask.reshape(_BASE_SHAPE)


def _view(
    base: lucid.Tensor, shape: tuple[int, ...], stride: tuple[int, ...], offset: int
) -> lucid.Tensor:
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, list(shape), list(stride), offset))


def _grads(step: int, shapes: Sequence[tuple[int, ...]]) -> list[np.ndarray]:
    # Distinct values per element: a gradient paired with the wrong element
    # moves it by the wrong amount.
    rng = np.random.default_rng(100 + step)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def _ref_grad(ref: Any, g: np.ndarray, name: str) -> Any:
    t = ref.from_numpy(g.copy())
    return t.to_sparse(1) if name == "SparseAdam" else t


def _flat(t: lucid.Tensor) -> np.ndarray:
    # Half precision widens exactly; NumPy has no bfloat16.
    if t.dtype in (lucid.float16, lucid.bfloat16):
        t = t.float()
    return np.asarray(t.detach().numpy(), dtype=np.float64)


@pytest.mark.parametrize("case", _OPTIMIZER_PARAMS)
@pytest.mark.parametrize("layout_name", list(_LAYOUTS))
def test_a_step_writes_only_the_parameters_elements(
    case: tuple[str, dict[str, Any]], layout_name: str, device: str, ref: Any
) -> None:
    name, kw = case
    layout = _LAYOUTS[layout_name]
    init = np.random.default_rng(0).standard_normal(_BASE_SHAPE).astype(np.float32)
    base = lucid.tensor(init.copy()).to(device)
    params = [nn.Parameter(_view(base, *geometry)) for geometry in layout]
    opt = getattr(optim, name)(params, **kw)

    ref_base = ref.from_numpy(init.copy())
    ref_params = [
        ref.nn.Parameter(ref_base.as_strided(shape, stride, offset))
        for shape, stride, offset in layout
    ]
    ref_opt = getattr(ref.optim, name)(ref_params, **kw)

    for step in range(_STEPS):
        grads = _grads(step, [shape for shape, _, _ in layout])
        for p, rp, g in zip(params, ref_params, grads, strict=True):
            p.grad = lucid.tensor(g.copy()).to(device)
            rp.grad = _ref_grad(ref, g, name)
        opt.step()
        ref_opt.step()
        for i, (p, rp) in enumerate(zip(params, ref_params, strict=True)):
            np.testing.assert_allclose(
                _flat(p),
                rp.detach().numpy(),
                rtol=_TOL,
                atol=_TOL,
                err_msg=f"param {i} after step {step + 1}",
            )

    after = _flat(base)
    np.testing.assert_allclose(after, ref_base.numpy(), rtol=_TOL, atol=_TOL)
    untouched = ~_covered(layout)
    np.testing.assert_array_equal(after[untouched], init[untouched])


@pytest.mark.parametrize("case", _OPTIMIZER_PARAMS)
@pytest.mark.parametrize(
    "device, dtype",
    device_dtype_params([lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64]),
)
def test_a_strided_parameter_steps_as_a_dense_one(
    case: tuple[str, dict[str, Any]], device: str, dtype: lucid.dtype
) -> None:
    # Bit for bit: the same update over the same values, only the layout
    # differs.
    name, kw = case
    init = np.random.default_rng(1).standard_normal(_BASE_SHAPE).astype(np.float32)
    base = lucid.tensor(init.copy(), dtype=dtype).to(device)
    strided = nn.Parameter(base.t())
    dense = nn.Parameter(lucid.tensor(init.T.copy(), dtype=dtype).to(device))
    opt_strided = getattr(optim, name)([strided], **kw)
    opt_dense = getattr(optim, name)([dense], **kw)
    for step in range(_STEPS):
        (g,) = _grads(step, [(6, 4)])
        strided.grad = lucid.tensor(g.copy(), dtype=dtype).to(device)
        dense.grad = lucid.tensor(g.copy(), dtype=dtype).to(device)
        opt_strided.step()
        opt_dense.step()
    np.testing.assert_array_equal(_flat(strided), _flat(dense))
    np.testing.assert_array_equal(_flat(base).T, _flat(dense))


def _natural_views(x: lucid.Tensor) -> dict[str, list[lucid.Tensor]]:
    return {
        "column": [x[:, 2]],
        "offset_block": [x[1:3, 1:4]],
        "transpose": [x.t()],
        "two_params_one_base": [x[:, 1], x[2:4, 3:5].t()],
    }


def _ref_natural_views(x: Any) -> dict[str, list[Any]]:
    return {
        "column": [x[:, 2]],
        "offset_block": [x[1:3, 1:4]],
        "transpose": [x.t()],
        "two_params_one_base": [x[:, 1], x[2:4, 3:5].t()],
    }


@pytest.mark.parametrize("case", _OPTIMIZER_PARAMS)
@pytest.mark.parametrize("layout_name", list(_LAYOUTS))
def test_a_parameter_made_from_an_indexed_view_steps_into_its_base(
    case: tuple[str, dict[str, Any]], layout_name: str, device: str, ref: Any
) -> None:
    # What a user writes: view ops on either CPU or Metal, as in LCD-476/LCD-501.
    name, kw = case
    init = np.random.default_rng(2).standard_normal(_BASE_SHAPE).astype(np.float32)
    x = lucid.tensor(init.copy()).to(device)
    params = [nn.Parameter(v) for v in _natural_views(x)[layout_name]]
    opt = getattr(optim, name)(params, **kw)
    ref_x = ref.from_numpy(init.copy())
    ref_params = [ref.nn.Parameter(v) for v in _ref_natural_views(ref_x)[layout_name]]
    ref_opt = getattr(ref.optim, name)(ref_params, **kw)
    for step in range(_STEPS):
        grads = _grads(step, [tuple(p.shape) for p in params])
        for p, rp, g in zip(params, ref_params, grads, strict=True):
            p.grad = lucid.tensor(g.copy()).to(device)
            rp.grad = _ref_grad(ref, g, name)
        opt.step()
        ref_opt.step()
    after = _flat(x)
    np.testing.assert_allclose(after, ref_x.numpy(), rtol=_TOL, atol=_TOL)
    untouched = ~_covered(_LAYOUTS[layout_name])
    np.testing.assert_array_equal(after[untouched], init[untouched])


def test_lbfgs_steps_strided_parameters_into_their_base(device: str, ref: Any) -> None:
    # The reference flattens each gradient with ``view(-1)``, which a
    # gradient laid out as a full transpose refuses: the two views here are
    # not dense, so their gradients are.
    layout = _LAYOUTS["two_params_one_base"]
    init = np.random.default_rng(3).standard_normal(_BASE_SHAPE).astype(np.float32)
    rng = np.random.default_rng(4)
    weights = [rng.uniform(0.5, 2.0, shape).astype(np.float32) for shape, _, _ in layout]
    base = lucid.tensor(init.copy()).to(device)
    params = [nn.Parameter(_view(base, *geometry)) for geometry in layout]
    ws = [lucid.tensor(w.copy()).to(device) for w in weights]
    opt = optim.LBFGS(params, lr=0.5, max_iter=4)
    ref_base = ref.from_numpy(init.copy())
    ref_params = [
        ref.nn.Parameter(ref_base.as_strided(shape, stride, offset))
        for shape, stride, offset in layout
    ]
    ref_ws = [ref.from_numpy(w.copy()) for w in weights]
    ref_opt = ref.optim.LBFGS(ref_params, lr=0.5, max_iter=4)

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = sum((w * (p - 1.0) ** 2).sum() for p, w in zip(params, ws, strict=True))
        assert isinstance(loss, lucid.Tensor)
        loss.backward()
        return loss

    def ref_closure() -> Any:
        ref_opt.zero_grad()
        loss = sum((w * (p - 1.0) ** 2).sum() for p, w in zip(ref_params, ref_ws, strict=True))
        loss.backward()
        return loss

    opt.step(closure)
    ref_opt.step(ref_closure)
    after = _flat(base)
    np.testing.assert_allclose(after, ref_base.detach().numpy(), rtol=1e-4, atol=1e-4)
    untouched = ~_covered(layout)
    np.testing.assert_array_equal(after[untouched], init[untouched])


@pytest.mark.parametrize(
    "make",
    [lambda ps: optim.SGD(ps, lr=0.1, momentum=0.9), lambda ps: optim.Adam(ps, lr=0.1)],
    ids=["SGD", "Adam"],
)
def test_a_parameter_whose_elements_overlap_is_refused(
    make: Callable[[list[nn.Parameter]], optim.Optimizer], device: str
) -> None:
    # Every element of the expanded view is the base's one element: there is
    # no single value to write.  The step is refused atomically before any
    # earlier parameter in the group or any optimizer state slot is modified.
    dense = nn.Parameter(lucid.tensor([2.0, -3.0]).to(device))
    base = lucid.zeros(1, device=device)
    p = nn.Parameter(_view(base, (4,), (0,), 0))
    opt = make([dense, p])

    # Step ``dense`` once while ``p`` has no gradient so ``dense`` has live
    # optimizer state; ``p`` remains uninitialized.
    dense.grad = lucid.tensor([0.5, -1.5]).to(device)
    opt.step()
    dense_before = _flat(dense).copy()
    state_before = {k: np.array(v, copy=True) for k, v in opt.state_dict()["state"][0].items()}
    assert 1 not in opt.state_dict()["state"]

    dense.grad = lucid.tensor([4.0, 5.0]).to(device)
    p.grad = lucid.tensor([1.0, 2.0, 3.0, 4.0]).to(device)
    with pytest.raises((_C_engine.InvalidArgument, RuntimeError), match=r"parameter 1.*overlap"):
        opt.step()

    np.testing.assert_array_equal(_flat(dense), dense_before)
    assert _flat(base).tolist() == [0.0]
    state_after = opt.state_dict()["state"]
    assert 1 not in state_after
    for k, expected in state_before.items():
        np.testing.assert_array_equal(np.asarray(state_after[0][k]), expected)
