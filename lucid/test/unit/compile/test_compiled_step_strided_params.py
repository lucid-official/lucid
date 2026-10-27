"""A compiled training step writes a parameter's own elements, wherever they sit.

``run_executable_inplace`` — the write of ``fused_step`` and
``compile_optimizer`` — installed each new value as its target's whole
buffer, and so refused a Metal parameter that is a view of part of a tensor
(``NotImplementedError: ... part of a tensor``) where the eager optimizer step
writes it into its part of the base (LCD-476).  It now writes through the same
owner as the eager step, ``core/ElementWrite.h`` (LCD-497).

Every gradient differs element by element (a per-element weight in the
loss), so a value paired with the wrong element shows.  Metal views of part
of a tensor are built with the engine's ``_make_view`` hook, as in
``optim/test_optimizer_strided_params.py``.  The compiled step is Metal only:
on the CPU ``make_step`` runs eager, which holds that side to the same rule.
"""

from collections.abc import Callable, Sequence
from typing import override

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.compile import _tracing, compile_optimizer, fused_step, make_step
from lucid.compile._entry.fused_step import _FusedStep
from lucid.test._fixtures.devices import metal_available

_BASE_SHAPE = (4, 6)
_STEPS = 3
_TOL = 1e-5

# Each parameter's (shape, stride, offset) over the 4 x 6 base, in elements.
_LAYOUTS: dict[str, list[tuple[tuple[int, ...], tuple[int, ...], int]]] = {
    "column": [((4,), (6,), 2)],
    "offset_block": [((2, 3), (6, 1), 7)],
    "transpose": [((6, 4), (1, 6), 0)],
    # A column and a transposed 2 x 2 block, disjoint, over one tensor.
    "two_params_one_base": [((4,), (6,), 1), ((2, 2), (1, 6), 15)],
}

_OPTIMIZERS: dict[str, Callable[[list[nn.Parameter]], optim.Optimizer]] = {
    "SGD": lambda ps: optim.SGD(ps, lr=0.1),
    "SGD-momentum": lambda ps: optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.1),
    "Adam": lambda ps: optim.Adam(ps, lr=0.05),
}

_metal = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


class _Fit(nn.Module):
    """Pulls each parameter towards ``1 / x``, weighted element by element."""

    def __init__(self, params: Sequence[nn.Parameter], device: str) -> None:
        super().__init__()
        self.ps = nn.ParameterList(list(params))
        rng = np.random.default_rng(7)
        for k, p in enumerate(params):
            w = rng.uniform(0.5, 2.0, tuple(p.shape)).astype(np.float32)
            self.register_buffer(f"w{k}", lucid.tensor(w).to(device))

    @override
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:  # type: ignore[override]
        total = None
        for k, p in enumerate(self.ps):
            term = ((p * x - 1.0) ** 2 * getattr(self, f"w{k}")).sum()
            total = term if total is None else total + term
        assert total is not None
        return total


def _view(
    base: lucid.Tensor, shape: tuple[int, ...], stride: tuple[int, ...], offset: int
) -> lucid.Tensor:
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, list(shape), list(stride), offset))


def _flat(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.detach().numpy(), dtype=np.float64)


def _covered(
    layout: Sequence[tuple[tuple[int, ...], tuple[int, ...], int]],
) -> np.ndarray:
    """Which elements of the base the layout's parameters read."""
    mask = np.zeros(int(np.prod(_BASE_SHAPE)), dtype=bool)
    for shape, stride, offset in layout:
        for index in np.ndindex(*shape):
            mask[offset + sum(i * s for i, s in zip(index, stride, strict=True))] = True
    return mask.reshape(_BASE_SHAPE)


def _setup(
    layout_name: str, opt_name: str, device: str, init: np.ndarray
) -> tuple[lucid.Tensor, _Fit, optim.Optimizer]:
    base = lucid.tensor(init.copy()).to(device)
    params = [nn.Parameter(_view(base, *g)) for g in _LAYOUTS[layout_name]]
    model = _Fit(params, device)
    return base, model, _OPTIMIZERS[opt_name](params)


def _loss(out: lucid.Tensor) -> lucid.Tensor:
    return out


def _run_compiled(
    entry: str, model: _Fit, opt: optim.Optimizer, x: lucid.Tensor
) -> Callable[[], bool]:
    """Take the steps through ``entry``; returns whether it stayed compiled."""
    if entry == "fused":
        fused = fused_step(model, _loss, opt)
        assert isinstance(fused, _FusedStep)
        for _ in range(_STEPS):
            fused(x)
        info = fused.cache_info()
        return lambda: info["entries"] == 1 and not info["eager_only"]
    if entry == "compile_optimizer":
        copt = compile_optimizer(opt)
        for _ in range(_STEPS):
            copt.zero_grad()
            model(x).backward()
            copt.step()
        return lambda: len(copt._plans) == 1
    step = make_step(model, _loss)
    for _ in range(_STEPS):
        opt.zero_grad()
        step(x).backward()
        opt.step()
    cache: dict[object, object] = step.cache  # type: ignore[attr-defined]
    return lambda: len(cache) == 1


def _run_eager(model: _Fit, opt: optim.Optimizer, x: lucid.Tensor) -> None:
    for _ in range(_STEPS):
        opt.zero_grad()
        model(x).backward()
        opt.step()


_CASES = [
    pytest.param(entry, device, marks=_metal if device == "metal" else ())
    for entry, device in [
        ("fused", "metal"),
        ("compile_optimizer", "metal"),
        ("make", "metal"),
        ("make", "cpu"),
    ]
]


@pytest.mark.filterwarnings("ignore::lucid.compile._core.fallback.CompileFallbackWarning")
@pytest.mark.parametrize("opt_name", list(_OPTIMIZERS))
@pytest.mark.parametrize("layout_name", list(_LAYOUTS))
@pytest.mark.parametrize("entry, device", _CASES)
def test_a_compiled_step_writes_only_the_parameters_elements(
    entry: str, device: str, layout_name: str, opt_name: str
) -> None:
    init = np.random.default_rng(0).standard_normal(_BASE_SHAPE).astype(np.float32)
    x = lucid.tensor([0.5]).to(device)
    base, model, opt = _setup(layout_name, opt_name, device, init)
    compiled = _run_compiled(entry, model, opt, x)
    ref_base, ref_model, ref_opt = _setup(layout_name, opt_name, device, init)
    _run_eager(ref_model, ref_opt, x)

    # On Metal the step ran the executable rather than falling back to eager.
    assert compiled() == (device == "metal")
    after = _flat(base)
    np.testing.assert_allclose(after, _flat(ref_base), rtol=_TOL, atol=_TOL)
    assert not np.allclose(after, init)
    untouched = ~_covered(_LAYOUTS[layout_name])
    np.testing.assert_array_equal(after[untouched], init[untouched])
    for p, rp in zip(model.ps, ref_model.ps, strict=True):
        np.testing.assert_allclose(_flat(p), _flat(rp), rtol=_TOL, atol=_TOL)


# ── run_executable_inplace itself ───────────────────────────────────────────


def _increment_exe(
    samples: Sequence[lucid.Tensor],
) -> tuple[object, list[_C_engine.TensorImpl]]:
    """An executable giving ``s + k + 1`` for the ``k``-th sample, and its feeds."""
    with lucid.no_grad(), _tracing() as tracer:
        outs = [s + float(k + 1) for k, s in enumerate(samples)]
    ids = [int(tracer.lookup_id(_unwrap(o))) for o in outs]
    ext = tracer.external_feeds
    exe = _C_engine.compile.compile_or_cached(tracer.graph, ext, False, [], ids)
    assert exe is not None
    return exe, [ext[tid] for tid in exe.input_ids]


@_metal
def test_a_view_of_part_of_a_tensor_takes_its_output_in_its_part_of_the_base() -> None:
    # Refused as "part of a tensor" until LCD-497; the dense target beside it
    # is written as before.
    a = lucid.ones(3, device="metal")
    base = lucid.zeros(6, device="metal")
    b = _view(base, (3,), (1,), 2)
    exe, feeds = _increment_exe([lucid.ones(3, device="metal")] * 2)
    versions = (a._impl.version, b._impl.version)
    _C_engine.compile.run_executable_inplace(exe, feeds, [a._impl, b._impl])
    assert _flat(a).tolist() == [2.0] * 3
    assert _flat(b).tolist() == [3.0] * 3
    assert _flat(base).tolist() == [0.0, 0.0, 3.0, 3.0, 3.0, 0.0]
    assert a._impl.version > versions[0]
    assert b._impl.version > versions[1]


@_metal
def test_a_refused_target_leaves_every_target_as_it_was() -> None:
    # Shape, dtype and device are checked for every target before the run.
    a = lucid.ones(3, device="metal")
    base = lucid.zeros(6, device="metal")
    wrong = _view(base, (2,), (1,), 2)
    exe, feeds = _increment_exe([lucid.ones(3, device="metal")] * 2)
    version = a._impl.version
    with pytest.raises(ValueError, match="shape mismatch at slot 1"):
        _C_engine.compile.run_executable_inplace(exe, feeds, [a._impl, wrong._impl])
    assert _flat(a).tolist() == [1.0] * 3
    assert a._impl.version == version
    assert _flat(base).tolist() == [0.0] * 6


@_metal
def test_a_target_whose_elements_overlap_is_refused() -> None:
    # Every element of the expanded view is the base's one element: there is
    # no single value to write, as in the eager step.
    base = lucid.zeros(1, device="metal")
    expanded = _view(base, (3,), (0,), 0)
    exe, feeds = _increment_exe([lucid.ones(3, device="metal")])
    with pytest.raises(RuntimeError, match="overlap"):
        _C_engine.compile.run_executable_inplace(exe, feeds, [expanded._impl])
    assert _flat(base).tolist() == [0.0]
