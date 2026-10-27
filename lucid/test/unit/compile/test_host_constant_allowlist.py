"""The builder drops a node only when dropping it is safe.

A node with no recorded inputs and no live outputs looks the same
whether it is a factory whose value the arguments fix, or an op that
forgot to record its trace I/O and whose result is about to be frozen
into the graph as a feed. The builder used to drop both; the second
kind is how a compiled model comes to answer every input with the
first one's result.

It now drops such a node only for a named list of host-constant
producers and refuses otherwise, so a new op that forgets its wiring
costs an eager fallback rather than a wrong answer.

That makes the list load-bearing in the other direction: a producer
missing from it stops compiling, quietly, with only a slowdown to show
for it. These tests are what notices.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable"),
    pytest.mark.usefixtures("inference"),
]


class _Apply(nn.Module):
    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


def _device_of(t: lucid.Tensor) -> object:
    return t.device


# Everything the builder is allowed to drop. A factory built inside
# ``forward`` has to name the device, or it lands on the CPU and the op
# that consumes it mismatches before compilation is even reached.
PRODUCERS = [
    ("zeros", lambda t: t + lucid.zeros(2, 4, device=_device_of(t))),
    ("ones", lambda t: t * lucid.ones(2, 4, device=_device_of(t))),
    ("full", lambda t: t + lucid.full((2, 4), 3.0, device=_device_of(t))),
    ("arange", lambda t: t + lucid.arange(4, dtype=lucid.float32, device=_device_of(t))),
    ("eye", lambda t: t[:, :2] + lucid.eye(2, device=_device_of(t))),
    ("linspace", lambda t: t + lucid.linspace(0, 1, 4, device=_device_of(t))),
    ("logspace", lambda t: t + lucid.logspace(0, 1, 4, device=_device_of(t))),
    ("tril", lambda t: lucid.tril(t[:, :2])),
    ("triu", lambda t: lucid.triu(t[:, :2])),
    # rearrange/reduce/repeat lower in Python to reshape/permute/reduce/broadcast
    # primitives that record their trace I/O (not via result_is_a_host_constant).
    ("einops_rearrange", lambda t: lucid.einops.rearrange(t, "a b -> b a")),
    ("einops_reduce", lambda t: lucid.einops.reduce(t, "a b -> a", "sum")),
    ("einops_repeat", lambda t: lucid.einops.repeat(t, "a b -> a b c", c=2)),
    ("einsum", lambda t: lucid.einops.einsum("i j, i j -> i", t, t)),
]


@pytest.mark.parametrize(("name", "fn"), PRODUCERS, ids=[c[0] for c in PRODUCERS])
def test_a_host_constant_producer_still_compiles(name, fn):
    lucid.manual_seed(0)
    model = _Apply(fn).eval()
    x = lucid.randn(2, 4)
    reference = model(x)

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")

    assert _C_engine.compile.session_cache_size() > 0, (
        f"{name} stopped compiling — if it is a genuine host constant it is "
        "missing from the builder's allowlist; if it is not, it should be "
        "recording its trace I/O instead"
    )
    scale = max(float(reference.abs().max().item()), 1e-30)
    assert float((got - reference).abs().max().item()) / scale < 1e-5


def test_an_ordinary_model_is_untouched():
    """The guard reads the zero-input case only.

    An op that recorded its inputs is dropped as before when nothing
    reads it — that is real dead-code elimination and cannot freeze
    anything, because nothing is reading the result either.
    """

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3)
            self.fc = nn.Linear(8, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            h = self.conv(x).relu().mean(dim=(2, 3))
            return self.fc(lucid.concat([h, h], dim=1))

    lucid.manual_seed(0)
    model = Net().eval()
    x = lucid.randn(2, 3, 8, 8)
    reference = model(x)

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")

    assert _C_engine.compile.session_cache_size() > 0
    scale = max(float(reference.abs().max().item()), 1e-30)
    assert float((got - reference).abs().max().item()) / scale < 1e-5
