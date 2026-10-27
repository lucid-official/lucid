"""Three-dimensional pooling, through the SDK's four-dimensional one.

MPSGraph ships 2-D and 4-D pooling and nothing between, and Lucid's 3-D
pool is a rank-5 tensor, so this was a stub with the reason written down:
"the SDK only ships 2D variants (or 4D-spatial via rank-6 pooling, not
the rank-5 form Lucid uses)".

The rank is the whole of the gap. A length-1 spatial axis in front makes
the volume rank 6, the 4-D operation pools it with a kernel of 1 on that
axis, and the axis comes back out — so the reason was accurate about the
shapes and wrong about the conclusion.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


class Pool(nn.Module):
    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


def _run(fn: object, shape: tuple[int, ...]) -> tuple[bool, float, tuple[int, ...]]:
    model = Pool(fn).eval()
    x = lucid.randn(*shape)
    reference = model(x)
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")
    return (
        _C_engine.compile.session_cache_size() > 0,
        float((got - reference).abs().max().item()),
        tuple(got.shape),
    )


class TestPool3dCompiles:
    """Exact agreement, not approximate: this is the same pooling.

    A decomposition that changed the arithmetic would show here, and one
    that got the axis order wrong would show as a shape.
    """

    @pytest.mark.parametrize(
        ("name", "fn", "shape"),
        [
            ("max plain", lambda x: F.max_pool3d(x, 2), (1, 4, 6, 6, 6)),
            ("avg plain", lambda x: F.avg_pool3d(x, 2), (1, 4, 6, 6, 6)),
            (
                "max strided and padded",
                lambda x: F.max_pool3d(x, 3, stride=2, padding=1),
                (2, 3, 7, 7, 7),
            ),
            (
                "avg overlapping",
                lambda x: F.avg_pool3d(x, 2, stride=1),
                (1, 2, 5, 5, 5),
            ),
            (
                "asymmetric volume",
                lambda x: F.max_pool3d(x, 2),
                (1, 2, 4, 6, 8),
            ),
        ],
        ids=["max", "avg", "max-strided", "avg-overlap", "asymmetric"],
    )
    def test_it_compiles_and_matches_exactly(
        self, name: str, fn: object, shape: tuple[int, ...]
    ) -> None:
        compiled, difference, produced = _run(fn, shape)
        assert compiled, f"{name} fell back to eager"
        assert difference == 0.0
        assert len(produced) == 5

    def test_they_are_no_longer_stubs(self) -> None:
        """Registration never changed; compiling did."""
        for name in ("max_pool3d", "avg_pool3d"):
            assert _C_engine.compile.emitter_registered(name)
        compiled, _difference, _shape = _run(lambda x: F.max_pool3d(x, 2), (1, 2, 4, 4, 4))
        assert compiled
