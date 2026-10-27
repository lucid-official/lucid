"""``shadow_alloc`` — building a model without allocating its weights.

``_shadow.py`` sat at 18.4%.  It hot-patches the engine's tensor-creation
entry points so a module's ``__init__`` runs end to end while the impls
it ends up holding carry only shape, dtype and device — which is how the
docs pipeline reads a layer tree for a model too large to instantiate.

The properties worth pinning are the ones that make it safe: the patch
has to be undone on the way out *including when the body raises*, the
phantom has to carry enough metadata for ``named_parameters`` and
``.shape`` to work, and a real allocation afterwards has to be real.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn._shadow import is_active, is_shadow, shadow_alloc

# ── the context ───────────────────────────────────────────────────────────────


def test_it_is_inactive_outside_and_active_inside():
    assert not is_active()
    with shadow_alloc():
        assert is_active()
    assert not is_active()


def test_a_phantom_arange_has_the_real_length_dtype_and_device():
    """The engine takes ``arange(start, end, step, dtype, device)``.

    The phantom read dtype and device two places too far along, so every
    one claimed float32 on the CPU; and its length formula was only right
    for whole steps.
    """
    with shadow_alloc():
        quarters = lucid.arange(0, 1, 0.25)
        ids = lucid.arange(5, device="metal")
        empty = lucid.arange(3, 0)
    assert quarters.shape == (4,)
    assert ids.shape == (5,)
    assert ids.dtype == lucid.int64
    assert ids.device.type == "metal"
    assert empty.shape == (0,)


def test_the_patch_is_undone_even_when_the_body_raises():
    """A leaked patch would make every later allocation phantom, and the
    failure would surface far from here."""
    with pytest.raises(RuntimeError):
        with shadow_alloc():
            raise RuntimeError("boom")
    assert not is_active()
    real = lucid.zeros(4)
    assert np.allclose(np.asarray(real.numpy()), 0.0)


def test_nesting_leaves_it_active_until_the_outermost_exit():
    with shadow_alloc():
        with shadow_alloc():
            assert is_active()
        assert is_active()
    assert not is_active()


# ── what a shadow-built module knows about itself ─────────────────────────────


def test_a_layer_built_in_shadow_reports_its_shapes():
    with shadow_alloc():
        layer = nn.Linear(128, 64)
    assert tuple(layer.weight.shape) == (64, 128)
    assert tuple(layer.bias.shape) == (64,)


def test_named_parameters_still_walks_the_tree():
    with shadow_alloc():
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    names = [name for name, _ in model.named_parameters()]
    assert "0.weight" in names and "2.weight" in names


def test_the_parameter_count_is_readable_without_allocating():
    """The number the docs pipeline exists to produce."""
    with shadow_alloc():
        model = nn.Linear(1000, 1000)
    total = sum(int(np.prod([int(d) for d in p.shape])) for _, p in model.named_parameters())
    assert total == 1000 * 1000 + 1000


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda: nn.Conv2d(3, 16, kernel_size=3), (16, 3, 3, 3)),
        (lambda: nn.Embedding(1000, 64), (1000, 64)),
        (lambda: nn.LayerNorm(32), (32,)),
    ],
)
def test_other_layer_kinds_report_their_shapes(build, expected):
    with shadow_alloc():
        layer = build()
    first = next(iter(layer.parameters()))
    assert tuple(first.shape) == expected


def test_a_phantom_carries_dtype_and_device():
    with shadow_alloc():
        layer = nn.Linear(4, 4)
    assert layer.weight.dtype is not None
    assert layer.weight.device is not None


def test_the_impl_really_is_a_phantom():
    with shadow_alloc():
        layer = nn.Linear(4, 4)
    assert is_shadow(layer.weight)


# ── and it does not leak into ordinary use ────────────────────────────────────


def test_a_module_built_outside_is_ordinary():
    layer = nn.Linear(4, 4)
    assert not is_shadow(layer.weight)
    out = layer(lucid.tensor(np.ones((2, 4), dtype=np.float32)))
    assert np.isfinite(np.asarray(out.numpy())).all()


def test_a_module_built_after_a_shadow_block_is_ordinary():
    with shadow_alloc():
        nn.Linear(64, 64)
    layer = nn.Linear(4, 4)
    assert not is_shadow(layer.weight)
    layer(lucid.tensor(np.ones((2, 4), dtype=np.float32))).sum().backward()
    assert layer.weight.grad is not None


@pytest.mark.parametrize(
    "build",
    [
        lambda: lucid.zeros(4),
        lambda: lucid.ones(4),
        lambda: lucid.full((2, 2), 3.0),
        lambda: lucid.eye(3),
        lambda: lucid.arange(0, 4, 1),
        lambda: lucid.randn(2, 2),
    ],
)
def test_every_patched_factory_is_restored(build):
    """Each entry point the context replaces has to come back."""
    with shadow_alloc():
        build()
    out = np.asarray(build().numpy())
    assert out.dtype != object
    assert out.size > 0
