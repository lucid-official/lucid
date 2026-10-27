"""Compile over a module with a registered parametrization (LCD-624, LCD-647).

Before LCD-624, ``nn.utils.parametrize`` stored its chains in a plain
``dict`` named ``parametrizations`` and reparametrised ``module.<name>``
through a forward pre-hook with a ``_hook_handle``.
``lucid/compile/_core/signature.py`` had two helpers that walked that exact
shape (``_parametrization_modules``, the ``parametrizations`` branch of
``_internal_pre_hook_keys``).  Since LCD-624, ``module.parametrizations`` is
a registered :class:`~lucid.nn.ModuleDict` of
:class:`~lucid.nn.utils.parametrize.ParametrizationList` — an ordinary
submodule that ``Module.modules()`` already walks into — and ``module.<name>``
is a property on a generated ``Parametrized<Cls>`` subclass, with no
pre-hook at all.  Both helpers were therefore always empty / dead and have
been removed (LCD-647); this file is the correctness proof that compiling a
parametrized module still works with them gone — through the ordinary
module-tree walk, with no special-casing.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.nn.utils import parametrize
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.filterwarnings(
    "ignore::lucid.compile._core.fallback.CompileFallbackWarning"
)

_TOL = 1e-5
_METAL = pytest.mark.skipif(not metal_available(), reason="metal unavailable")
_DEVICES = ["cpu", pytest.param("metal", marks=_METAL)]


class _Symmetric(nn.Module):
    """``f(X) = 0.5 * (X + X.mT)`` — the parametrize.py docstring's example."""

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return 0.5 * (x + x.mT)


def _model(device: str) -> nn.Linear:
    lucid.manual_seed(0)
    m = nn.Linear(4, 4).to(device)
    parametrize.register_parametrization(m, "weight", _Symmetric())
    return m


def _x(device: str) -> lucid.Tensor:
    rng = np.random.default_rng(1)
    return lucid.tensor(rng.standard_normal((3, 4)).astype(np.float32)).to(device)


def _close(a: lucid.Tensor, b: lucid.Tensor) -> None:
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=_TOL, atol=_TOL)


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("device", _DEVICES)
def test_compiled_forward_matches_eager(device: str) -> None:
    """A compiled parametrized module reads ``module.weight`` the same way
    eager does — the chain is folded into the trace like any other forward
    code, since the originals sit in the module tree the walk already
    covers."""
    m = _model(device)
    x = _x(device)
    compiled = lucid.compile(m)

    _close(compiled(x), m(x))
    y1, y2 = compiled(x).numpy(), compiled(x).numpy()
    assert np.array_equal(y1, y2)
    if device == "metal":
        info = compiled.cache_info()
        assert info["entries"] == 1 and not info["eager_only"], info


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("device", _DEVICES)
def test_in_place_write_to_original_is_seen_without_retrace(device: str) -> None:
    """``original``'s *values* are not part of the cache key (the module
    docstring at the top of signature.py) — a compiled call re-reads them
    from the feed on every call, so an in-place write needs no new trace."""
    m = _model(device)
    x = _x(device)
    compiled = lucid.compile(m)
    _close(compiled(x), m(x))

    original = m.parametrizations.weight.original
    with lucid.no_grad():
        original.copy_(original + 0.3)
    _close(compiled(x), m(x))
    if device == "metal":
        info = compiled.cache_info()
        assert info["entries"] == 1 and not info["eager_only"], info


@pytest.mark.parametrize("device", _DEVICES)
def test_training_step_updates_original_and_matches_eager(device: str) -> None:
    """A training step through the compiled module reaches ``original``:
    gradients flow back through the parametrization chain, and the
    optimizer's in-place update to it is read by the next compiled call —
    held to an identical eager loop over the whole run."""
    compiled_model, eager_model = _model(device), _model(device)
    x = _x(device)
    step = lucid.compile.make_step(compiled_model, lambda y: (y * y).sum())
    c_opt = optim.SGD(compiled_model.parameters(), lr=0.1)
    e_opt = optim.SGD(eager_model.parameters(), lr=0.1)

    for _ in range(3):
        c_opt.zero_grad()
        c_loss = step(x)
        c_loss.backward()
        c_opt.step()

        e_opt.zero_grad()
        e_loss = (eager_model(x) ** 2).sum()
        e_loss.backward()
        e_opt.step()

        assert np.isclose(float(c_loss.item()), float(e_loss.item()), rtol=_TOL, atol=_TOL)

    c_original = compiled_model.parametrizations.weight.original
    e_original = eager_model.parametrizations.weight.original
    assert c_original.grad is not None and e_original.grad is not None
    _close(c_original, e_original)
    _close(c_original.grad, e_original.grad)
    if device == "metal":
        assert len(step.cache) == 1 and not step.eager_only, (step.cache, step.eager_only)
