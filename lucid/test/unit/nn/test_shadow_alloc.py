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


def test_a_layer_built_in_shadow_reports_its_shapes(device):
    with shadow_alloc():
        layer = nn.Linear(128, 64, device=device)
    assert tuple(layer.weight.shape) == (64, 128)
    assert tuple(layer.bias.shape) == (64,)


def test_named_parameters_still_walks_the_tree(device):
    with shadow_alloc():
        model = nn.Sequential(
            nn.Linear(8, 4, device=device), nn.ReLU(), nn.Linear(4, 2, device=device)
        )
    names = [name for name, _ in model.named_parameters()]
    assert "0.weight" in names and "2.weight" in names


def test_the_parameter_count_is_readable_without_allocating(device):
    """The number the docs pipeline exists to produce."""
    with shadow_alloc():
        model = nn.Linear(1000, 1000, device=device)
    total = sum(int(np.prod([int(d) for d in p.shape])) for _, p in model.named_parameters())
    assert total == 1000 * 1000 + 1000


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda d: nn.Conv2d(3, 16, kernel_size=3, device=d), (16, 3, 3, 3)),
        (lambda d: nn.Embedding(1000, 64, device=d), (1000, 64)),
        (lambda d: nn.LayerNorm(32, device=d), (32,)),
    ],
)
def test_other_layer_kinds_report_their_shapes(build, expected, device):
    with shadow_alloc():
        layer = build(device)
    first = next(iter(layer.parameters()))
    assert tuple(first.shape) == expected
    assert first.device == lucid.device(device)


def test_an_embedding_with_a_padding_index_is_a_phantom_on_any_device(device):
    """Zeroing the padding row runs ``index_fill``, then ``where``, under the
    mode; on Metal ``where`` read its result's shape off an MLX array the phantom
    backend's answer does not have, and raised bad_variant_access (LCD-850)."""
    with shadow_alloc():
        layer = nn.Embedding(4, 2, padding_idx=0, device=device)
    assert is_shadow(layer.weight)
    assert tuple(layer.weight.shape) == (4, 2)
    assert layer.weight.dtype == lucid.float32
    assert layer.weight.device == lucid.device(device)


def test_a_phantom_carries_dtype_and_device(device):
    with shadow_alloc():
        layer = nn.Linear(4, 4, device=device)
    assert layer.weight.dtype == lucid.float32
    assert layer.weight.device == lucid.device(device)


def test_the_impl_really_is_a_phantom(device):
    with shadow_alloc():
        layer = nn.Linear(4, 4, device=device)
    assert is_shadow(layer.weight)


# ── and it does not leak into ordinary use ────────────────────────────────────


def test_a_module_built_outside_is_ordinary(device):
    layer = nn.Linear(4, 4, device=device)
    assert not is_shadow(layer.weight)
    out = layer(lucid.tensor(np.ones((2, 4), dtype=np.float32), device=device))
    assert np.isfinite(np.asarray(out.numpy())).all()


def test_a_module_built_after_a_shadow_block_is_ordinary(device):
    with shadow_alloc():
        nn.Linear(64, 64, device=device)
    layer = nn.Linear(4, 4, device=device)
    assert not is_shadow(layer.weight)
    layer(lucid.tensor(np.ones((2, 4), dtype=np.float32), device=device)).sum().backward()
    assert layer.weight.grad is not None


@pytest.mark.parametrize(
    "build",
    [
        lambda d: lucid.zeros(4, device=d),
        lambda d: lucid.ones(4, device=d),
        lambda d: lucid.full((2, 2), 3.0, device=d),
        lambda d: lucid.eye(3, device=d),
        lambda d: lucid.arange(0, 4, 1, device=d),
        lambda d: lucid.randn(2, 2, device=d),
    ],
)
def test_every_patched_factory_is_restored(build, device):
    """Each entry point the context replaces has to come back."""
    with shadow_alloc():
        build(device)
    out = np.asarray(build(device).numpy())
    assert out.dtype != object
    assert out.size > 0
