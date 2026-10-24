"""A compiled module assigned onto a parent stays in the parent's tree.

``lucid.compile`` returns a wrapper that is not a ``Module``, and assigned
as ``parent.denoiser = lucid.compile(parent.denoiser)`` it used to land as
a plain attribute: the parent's ``parameters()`` lost the whole subtree
(an optimizer built from it never trained those weights) and its
``state_dict`` lost their keys.  A world-model port had to route around
it with a hand-written shim.  The parent now registers the wrapped model
under the attribute's name and the attribute answers with the wrapper.

The executables pin the tensors they were traced with, so anything that
replaces them — ``.to()`` or an ``assign=True`` load, through the parent
or on the model itself — has to reach the wrapper's cache as well: a
cached call went on answering with the old weights.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.compile._entry.module import CompiledModule
from lucid.test._fixtures.devices import metal_available

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.b(self.a(x).relu())


class _Parent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(8, 8)
        self.body = _Net()

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.body(self.stem(x))


def _names(module: nn.Module) -> list[str]:
    return [name for name, _ in module.named_parameters()]


def test_the_parent_keeps_the_compiled_subtree() -> None:
    plain = _Parent()
    parent = _Parent()
    body = parent.body
    parent.body = lucid.compile(parent.body)

    assert _names(parent) == _names(plain)
    assert list(parent.state_dict()) == list(plain.state_dict())
    assert sum(p.numel() for p in parent.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )
    # The attribute answers with the wrapper; the tree holds the model.
    assert isinstance(parent.body, CompiledModule)
    assert dict(parent.named_children())["body"] is body

    parent.eval()
    assert body.training is False and parent.body.training is False


def test_removing_or_replacing_the_attribute_clears_both() -> None:
    parent = _Parent()
    parent.body = lucid.compile(parent.body)
    del parent.body
    assert "body" not in dict(parent.named_children())
    assert not hasattr(parent, "body")

    parent.body = lucid.compile(_Net())
    plain = _Net()
    parent.body = plain
    assert parent.body is plain
    assert dict(parent.named_children())["body"] is plain


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("through_parent", [True, False])
def test_an_assign_load_reaches_the_cache(through_parent: bool) -> None:
    lucid.manual_seed(0)
    parent = _Parent().to("metal").eval()
    body = parent.body
    parent.body = lucid.compile(body)
    x = lucid.randn(2, 8, device="metal")
    before = parent(x).numpy()
    assert parent.body.cache_info()["entries"] == 1

    target = parent if through_parent else body
    fresh = {
        key: lucid.randn(*value.shape, device="metal")
        for key, value in target.state_dict().items()
    }
    target.load_state_dict(fresh, assign=True)

    compiled = parent(x).numpy()
    eager = body(parent.stem(x)).numpy()
    assert not np.allclose(compiled, before)
    np.testing.assert_allclose(compiled, eager, atol=1e-5)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_a_copy_load_keeps_the_cache_and_is_seen() -> None:
    # A copy writes into the tensors the executable already holds.
    lucid.manual_seed(0)
    model = _Net().to("metal").eval()
    compiled = lucid.compile(model)
    x = lucid.randn(2, 8, device="metal")
    compiled(x)
    fresh = {
        key: lucid.randn(*value.shape, device="metal")
        for key, value in model.state_dict().items()
    }
    model.load_state_dict(fresh)
    np.testing.assert_allclose(compiled(x).numpy(), model(x).numpy(), atol=1e-5)
    assert compiled.cache_info()["entries"] == 1


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_moving_the_parent_drops_the_cache() -> None:
    parent = _Parent().to("metal").eval()
    parent.body = lucid.compile(parent.body)
    parent(lucid.randn(2, 8, device="metal"))
    assert parent.body.cache_info()["entries"] == 1
    # Where the parent already is, a move replaces no tensor (LCD-601).
    parent.to("metal")
    assert parent.body.cache_info()["entries"] == 1
    parent.to("cpu")
    assert parent.body.cache_info()["entries"] == 0
