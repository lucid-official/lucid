"""A view of a Metal tensor aliases it in the engine, as a CPU view does.

On the CPU a view and its base read one buffer (``ViewFamily``).  On Metal an
MLX array is a value, so a family keeps the buffer's current value as its root
instead: every member reads its elements out of the root through its own
geometry, and a write through any member replaces the root.  So a whole-buffer
Metal view — the reshape family, ``.data``, ``detach``, ``Parameter(x)`` — and
its base see each other's writes in both directions, through every writer: the
in-place ops, ``copy_from`` / ``assign_from`` (``nn.init``, ``load_state_dict``,
``nn.utils``) and the C++ optimizers.  Each case is held to what the CPU does.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.einops
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.nn.utils import copy_parameters_and_buffers, vector_to_parameters
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

DEVICES = ["cpu", "metal"]


def _flat(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _view(base: lucid.Tensor, shape: list[int], stride: list[int], offset: int) -> lucid.Tensor:
    """A member of ``base``'s family with any geometry, in elements."""
    return _wrap(_C_engine.TensorImpl._make_view(base._impl, shape, stride, offset))


# ── both directions, whole-buffer views ─────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_a_view_sees_a_later_write_to_its_base(device: str) -> None:
    x = lucid.zeros(3, 4, device=device)
    v = x.view(-1)
    x.add_(1.0)
    assert _flat(v) == [1.0] * 12


@pytest.mark.parametrize("device", DEVICES)
def test_a_write_through_a_view_reaches_its_base(device: str) -> None:
    x = lucid.zeros(3, 4, device=device)
    v = x.view(-1)
    v.add_(1.0)
    assert _flat(x) == [1.0] * 12
    v.mul_(3.0)
    x.sub_(1.0)
    assert _flat(v) == [2.0] * 12


@pytest.mark.parametrize("device", DEVICES)
def test_views_of_views_share_one_value(device: str) -> None:
    x = lucid.arange(12.0).to(device)
    a = x.reshape(3, 4)
    b = a.unsqueeze(0).squeeze(0).view(4, 3)
    b.mul_(2.0)
    assert _flat(x) == [2.0 * i for i in range(12)]
    assert _flat(a) == _flat(x)
    assert x.data_ptr() == b.data_ptr()


@pytest.mark.parametrize("device", DEVICES)
def test_parameter_of_a_live_tensor_writes_into_it(device: str) -> None:
    x = lucid.zeros(2, 3, device=device)
    p = nn.Parameter(x)
    with lucid.no_grad():
        p.add_(2.0)
    assert _flat(x) == [2.0] * 6
    # p was made from x, so x is its base, not a view of it: a recorded write
    # to x goes through, as the reference's does.  One through a view of p is
    # what is refused.
    x.mul_(5.0)
    assert _flat(p) == [10.0] * 6
    with pytest.raises(RuntimeError, match="view of a leaf"):
        p[0].mul_(5.0)
    assert _flat(p) == [10.0] * 6


@pytest.mark.parametrize("device", DEVICES)
def test_data_and_detach_alias_both_ways(device: str) -> None:
    p = nn.Parameter(lucid.ones(4, device=device))
    d = p.data
    t = p.detach()
    d.add_(1.0)
    assert _flat(p) == [2.0] * 4
    assert _flat(t) == [2.0] * 4
    with lucid.no_grad():
        p.mul_(2.0)
    assert _flat(d) == [4.0] * 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "make",
    [
        lambda x: nn.Flatten(0)(x),
        lambda x: lucid.einops.rearrange(x, "a b -> (a b)"),
        lambda x: lucid.Tensor(x),
        lambda x: x.flatten(),
    ],
    ids=["nn.Flatten", "einops.rearrange", "Tensor(x)", "flatten"],
)
def test_engine_made_views_write_into_their_base(
    device: str, make: Callable[[lucid.Tensor], lucid.Tensor]
) -> None:
    x = lucid.zeros(2, 3, device=device)
    make(x).add_(1.0)
    assert _flat(x) == [1.0] * 6


# ── raw writers: copy_from / assign_from and the C++ optimizers ─────────────


@pytest.mark.parametrize("device", DEVICES)
def test_nn_init_reaches_a_live_whole_view(device: str) -> None:
    w = lucid.zeros(2, 3, device=device)
    v = w.view(-1)
    nn.init.constant_(w, 4.0)
    assert _flat(v) == [4.0] * 6


@pytest.mark.parametrize("device", DEVICES)
def test_load_state_dict_reaches_a_live_whole_view(device: str) -> None:
    m = nn.Linear(3, 2).to(device)
    alias = m.weight.data.view(-1)
    m.load_state_dict({"weight": lucid.full((2, 3), 7.0), "bias": lucid.zeros(2)})
    assert _flat(alias) == [7.0] * 6


@pytest.mark.parametrize("device", DEVICES)
def test_nn_utils_copies_reach_a_live_whole_view(device: str) -> None:
    src = nn.Linear(3, 2).to(device)
    dst = nn.Linear(3, 2).to(device)
    alias = dst.weight.data
    copy_parameters_and_buffers(src, dst)
    assert _flat(alias) == _flat(src.weight)
    vector_to_parameters(lucid.full((8,), -1.0, device=device), dst.parameters())
    assert _flat(alias) == [-1.0] * 6


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("make", [optim.SGD, optim.Adam], ids=["SGD", "Adam"])
def test_an_optimizer_step_reaches_a_live_data_alias(
    device: str, make: Callable[..., optim.Optimizer]
) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    alias = p.data
    view = p.data.view(3, 1)
    opt = make([p], lr=0.5)
    (p * p).sum().backward()
    opt.step()
    assert _flat(alias) == _flat(p)
    assert _flat(view) == _flat(p)
    assert _flat(p) != [1.0] * 3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("make", [optim.SGD, optim.Adam], ids=["SGD", "Adam"])
def test_an_engine_step_writes_a_partial_view_into_its_part_of_the_base(
    device: str, make: Callable[..., optim.Optimizer]
) -> None:
    # A view of part of a tensor is stepped on a dense copy and written back
    # through its geometry, on Metal into the family's root, while the
    # parameter ahead of it steps in place.  Metal refused it until LCD-476;
    # every layout and optimizer: optim/test_optimizer_strided_params.py.
    a = nn.Parameter(lucid.ones(3, device=device))
    base = lucid.arange(6.0).to(device)
    b = nn.Parameter(_view(base, [3], [1], 2))
    dense = nn.Parameter(lucid.tensor([2.0, 3.0, 4.0], device=device))
    opt = make([a, b], lr=0.5)
    opt_dense = make([dense], lr=0.5)
    ((a * a).sum() + (b * b * b).sum()).backward()
    (dense * dense * dense).sum().backward()
    opt.step()
    opt_dense.step()
    assert _flat(b) == _flat(dense) != [2.0, 3.0, 4.0]
    assert _flat(base) == [0.0, 1.0, *_flat(dense), 5.0]
    assert _flat(a) != [1.0] * 3


@pytest.mark.parametrize("device", DEVICES)
def test_a_running_stat_write_lands_in_its_part_of_the_base(device: str) -> None:
    # A running statistic that is a view of part of a tensor is updated on a
    # dense copy and written back through its geometry, on Metal into the
    # family's root, while the mean beside it is updated in place.  Metal
    # refused it until LCD-491; every layout and BatchNorm:
    # nn/test_running_stats_layout.py.
    bn = nn.BatchNorm1d(3).to(device)
    dense = nn.BatchNorm1d(3).to(device)
    base = lucid.ones(6, device=device)
    bn.running_var = _view(base, [3], [1], 3)
    x = lucid.randn(4, 3, device=device)
    bn(x)
    dense(x)
    assert _flat(bn.running_var) == _flat(dense.running_var) != [1.0] * 3
    assert _flat(bn.running_mean) == _flat(dense.running_mean)
    assert _flat(base) == [1.0] * 3 + _flat(dense.running_var)
    assert bn.num_batches_tracked.item() == 1


# A compiled step's write into a view of part of a tensor (LCD-497):
# compile/test_compiled_step_strided_params.py.


# ── autograd: versions, refusals, re-derived views ──────────────────────────


def test_a_write_to_the_base_invalidates_a_saved_metal_view() -> None:
    x = lucid.ones(4, device="metal", requires_grad=True)
    y = x * 1.0
    v = y.view(2, 2)
    z = (v * v).sum()
    with lucid.no_grad():
        y.add_(1.0)
    with pytest.raises(_C_engine.VersionMismatch):
        z.backward()


def test_a_write_to_the_base_of_a_saved_cpu_view_is_refused() -> None:
    # The CPU refuses earlier: the node holds the buffer the write would
    # change.  A Metal node holds a value no write changes, so the version
    # is what refuses there.
    x = lucid.ones(4, requires_grad=True)
    y = x * 1.0
    v = y.view(2, 2)
    z = (v * v).sum()
    with (
        lucid.no_grad(),
        pytest.raises(NotImplementedError, match="saved for backward"),
    ):
        y.add_(1.0)
    assert z.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_a_view_of_a_leaf_is_refused_while_recording(device: str) -> None:
    x = lucid.zeros(4, device=device, requires_grad=True)
    v = x.view(2, 2)
    with pytest.raises(RuntimeError, match="view of a leaf"):
        v.add_(1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_a_dtype_change_through_a_live_view_is_refused(device: str) -> None:
    x = lucid.arange(4, device=device)
    v = x.view(2, 2)
    with pytest.raises(RuntimeError):
        x.exp_()
    assert _flat(v) == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_lone_member_keeps_its_dtype_as_any_tensor_does(device: str) -> None:
    # No in-place op changes its target's dtype (LCD-474).  The float result
    # of exp_ cannot be cast to int64, so the write is refused before the
    # family takes part, as the reference refuses it.
    x = lucid.arange(4, device=device).view(2, 2)
    with pytest.raises(RuntimeError, match="can't be cast"):
        x.exp_()
    assert x.dtype == lucid.int64
    assert _flat(x) == [0.0, 1.0, 2.0, 3.0]
    y = lucid.arange(4.0, device=device).view(2, 2)
    y.exp_()
    assert y.dtype == lucid.float32


@pytest.mark.parametrize("device", DEVICES)
def test_a_write_to_the_base_re_derives_a_live_view(device: str) -> None:
    x = lucid.arange(1.0, 5.0).to(device).requires_grad_()
    y = x * 1.0
    v = y.view(-1)
    y.mul_(2.0)
    v.sum().backward()
    assert _flat(x.grad) == [2.0] * 4


@pytest.mark.parametrize("device", DEVICES)
def test_a_write_through_a_view_re_derives_its_base(device: str) -> None:
    x = lucid.arange(1.0, 5.0).to(device).requires_grad_()
    y = x * 2.0
    # An engine view with no Python link: nothing but the engine carries
    # the write to y.  It has no graph of its own, so its values enter the
    # product as a constant: y = (2x)·x with only the second factor tracked.
    v = _view(y, [2, 2], [2, 1], 0)
    v.mul_(x.view(2, 2))
    (y * y).sum().backward()
    expected = [8.0 * a**3 for a in [1.0, 2.0, 3.0, 4.0]]
    assert _flat(x.grad) == pytest.approx(expected, rel=1e-5)


def _recorded_family_write(x: lucid.Tensor) -> lucid.Tensor:
    y = x * 1.0
    v = y.view(-1)
    y.mul_(x)
    return (v * v).sum() + y.sum()


def test_gradcheck_of_a_recorded_write_with_a_live_view_cpu() -> None:
    x = lucid.tensor([0.3, -1.2, 0.7, 2.0], dtype=lucid.float64).reshape(2, 2)
    assert lucid.autograd.gradcheck(_recorded_family_write, [x.requires_grad_()])


def test_a_recorded_write_with_a_live_view_matches_finite_differences_on_metal() -> None:
    # Metal has no float64, so the central difference runs in float32.
    values = [0.3, -1.2, 0.7, 2.0]
    x = lucid.tensor(values).reshape(2, 2).to("metal").requires_grad_()
    _recorded_family_write(x).backward()
    eps = 1e-2
    numeric = []
    for k in range(4):
        hi = list(values)
        lo = list(values)
        hi[k] += eps
        lo[k] -= eps
        f_hi = _recorded_family_write(lucid.tensor(hi).reshape(2, 2).to("metal")).item()
        f_lo = _recorded_family_write(lucid.tensor(lo).reshape(2, 2).to("metal")).item()
        numeric.append((f_hi - f_lo) / (2 * eps))
    analytic = [(2 * a * a + 1) * 2 * a for a in values]
    assert _flat(x.grad) == pytest.approx(analytic, rel=1e-4)
    assert numeric == pytest.approx(analytic, rel=1e-2, abs=1e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_a_recorded_write_into_part_of_a_family_splices_the_base(device: str) -> None:
    # A run of y (no grad_fn of its own) written with w: the base's gradient
    # skips the written elements and w receives theirs (CopySlices, runs).
    x = lucid.arange(1.0, 9.0).to(device).requires_grad_()
    y = x * 1.0
    s = _view(y, [3], [1], 2)
    w = lucid.full((3,), 3.0, device=device, requires_grad=True)
    s.mul_(w)
    c = lucid.arange(1.0, 9.0).to(device)
    (y * c).sum().backward()
    assert _flat(y) == [1.0, 2.0, 9.0, 12.0, 15.0, 6.0, 7.0, 8.0]
    assert _flat(x.grad) == [1.0, 2.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0]
    assert _flat(w.grad) == [9.0, 16.0, 25.0]


def test_a_recorded_write_next_to_a_strided_metal_view_splices_its_graph() -> None:
    def run(dev: str) -> tuple[list[float], list[float]]:
        x = lucid.arange(6.0, device=dev, requires_grad=True)
        y = x * 1.0
        t = _view(y, [3, 2], [1, 3], 0)
        y.mul_(2.0)
        c = lucid.arange(1.0, 7.0, device=dev).reshape(3, 2)
        (t * c).sum().backward()
        assert x.grad is not None
        return _flat(t), _flat(x.grad)

    assert run("metal") == run("cpu")


# ── any geometry under no_grad: runs, strided slices, scatters ──────────────


_GEOMETRIES = [
    ([4], [1], 3),  # a run at an offset
    ([2, 2], [4, 1], 5),  # a strided slice: rows of a 4-wide matrix
    ([3, 2], [1, 4], 0),  # a transposed slice
    ([2, 3], [6, 2], 1),  # steps along the inner axis
    ([2, 2], [5, 2], 0),  # strides no root shape slices: written by position
    ([], [], 7),  # one element
    ([0, 3], [3, 1], 0),  # empty
]


@pytest.mark.parametrize("geometry", _GEOMETRIES, ids=lambda g: f"{g[0]}-{g[1]}-{g[2]}")
def test_any_geometry_reads_and_writes_as_on_the_cpu(
    geometry: tuple[list[int], list[int], int],
) -> None:
    shape, stride, offset = geometry
    seen = {}
    for device in DEVICES:
        base = lucid.arange(12.0).to(device)
        v = _view(base, shape, stride, offset)
        before = _flat(v) if v.numel() else []
        v.mul_(-1.0)
        v.add_(0.5)
        seen[device] = (before, _flat(v) if v.numel() else [], _flat(base))
    assert seen["metal"] == seen["cpu"]


def test_an_overlapping_metal_view_refuses_a_write() -> None:
    base = lucid.arange(6.0).to("metal")
    v = _view(base, [3, 2], [1, 1], 0)
    with pytest.raises(RuntimeError, match="overlap"):
        v.add_(1.0)


# ── strict view (D10c) ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_view_of_a_transpose_is_refused(device: str) -> None:
    # A Metal transpose is still a dense copy (LCD-18), but the view links
    # track its logical strides (LCD-116), so view() refuses it as the CPU
    # and the reference do.
    x = lucid.zeros(3, 4, device=device)
    with pytest.raises(RuntimeError, match="use .reshape"):
        x.t().view(-1)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_view_refuses_a_strided_view_and_reshape_copies(device: str) -> None:
    # On Metal the view links read the strided member's own geometry from
    # the engine (LCD-513), so the reshape is a copy there too (LCD-116).
    x = lucid.zeros(12, device=device)
    t = _view(x, [3, 4], [1, 3], 0)
    with pytest.raises(RuntimeError, match="use .reshape"):
        t.view(-1)
    r = t.reshape(-1)
    r.add_(1.0)
    assert _flat(x) == [0.0] * 12
    assert _flat(r) == [1.0] * 12


@pytest.mark.parametrize("device", DEVICES)
def test_a_strided_view_takes_and_drops_unit_axes_as_a_view(device: str) -> None:
    x = lucid.arange(12.0).to(device)
    t = _view(x, [4, 3], [1, 4], 0)
    u = t.view(4, 1, 3).squeeze(1).unsqueeze(0)
    with lucid.no_grad():
        u.mul_(-1.0)
    assert _flat(x) == [-float(i) for i in range(12)]


@pytest.mark.parametrize("device", DEVICES)
def test_view_and_reshape_agree_on_viewable_input(device: str) -> None:
    x = lucid.arange(6.0).to(device)
    assert _flat(x.view(2, 3)) == _flat(x.reshape(2, 3))
    assert x.view(2, -1).shape == (2, 3)


# ── the Metal shared buffer stays coherent ──────────────────────────────────


def test_a_view_write_reaches_a_metal_shared_buffer_and_its_cpu_alias() -> None:
    from lucid import metal

    s = metal.shared_tensor((4,))
    g = s.to("metal")
    v = g.view(2, 2)
    v.add_(3.0)
    assert _flat(s) == [3.0] * 4
    assert g.is_shared
    s.mul_(2.0)
    assert _flat(v) == [6.0] * 4


def test_a_hooked_output_off_a_replaced_shared_root_is_its_own_value() -> None:
    # A module output aliased for a full backward hook is a value of its own,
    # not a family member.  Once a recorded write has replaced the root, a
    # write into the shared buffer the family began on cannot reach it, so it
    # must not move its version either: backward through it stays valid.
    from lucid import metal

    class _Flat(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x.view(2, 2)

    s = metal.shared_tensor((4,))
    g = s.to("metal")
    v = g.view(4)
    v.mul_(lucid.ones(4, device="metal", requires_grad=True))  # replaces the root
    # Run it: a lazy result still reads the buffer when it runs.
    assert _flat(v) == [0.0] * 4
    m = _Flat()
    m.register_full_backward_hook(lambda mod, gi, go: None)
    out = m(g)
    w = lucid.ones(2, 2, device="metal", requires_grad=True)
    z = (out * w).sum()
    with lucid.no_grad():
        s.add_(1.0)
    z.backward()
    assert w.grad is not None and _flat(w.grad) == [0.0] * 4
