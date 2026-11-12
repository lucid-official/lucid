"""The MobileNet and ViT parity comparisons can fail (LCD-960).

Six of them -- MobileNet-v2 (1.0, 0.75), MobileNet-v3 (large, small) and ViT-B/16, B/32 --
compared a reference answer that was zero to within the tolerance (1e-9 behind norm buffers
that were randomised and then overwritten by the transfer, exactly 0 behind a
zero-initialised head), so an all-zeros model passed, and they were later reduced to
skips.  Each now has to reject a model with one deliberate defect of each kind: noise of
1e-3 added to one weight tensor, a norm epsilon from the other convention, one activation
swapped.  If a change to the harness ever makes a comparison blind again, these fail first.
"""

import pytest
import lucid.models as M
from lucid.test.parity.models._mutations import MutationKind, Where, at
from lucid.test.parity.models._registry import SPECS, ParitySpec
from lucid.test.parity.models._utils import _run_parity, _spec_param, requires_timm

_FACTORIES = frozenset(
    {
        M.mobilenet_v2_cls,
        M.mobilenet_v2_075_cls,
        M.mobilenet_v3_large_cls,
        M.mobilenet_v3_small_cls,
        M.vit_base_16_cls,
        M.vit_base_32_cls,
    }
)
_SPECS = [s for s in SPECS if s.lucid_factory in _FACTORIES]

# Where a defect sits matters.  A norm epsilon changes the answer only where the layer's
# variance is small, so it goes in the first norm layer (a ViT LayerNorm is insensitive to it
# at 22 of its 25 sites); a weight or an activation is caught at every position tried.
_CASES: list[tuple[MutationKind, Where]] = [
    ("weight", "first"),
    ("weight", "middle"),
    ("weight", "last"),
    ("activation", "first"),
    ("activation", "middle"),
    ("activation", "last"),
    ("norm_eps", "first"),
]


@requires_timm
@pytest.mark.parametrize(("kind", "where"), _CASES, ids=[f"{k}-{w}" for k, w in _CASES])
@pytest.mark.parametrize("spec", [_spec_param(s) for s in _SPECS])
def test_a_wrong_model_fails_the_comparison(
    spec: ParitySpec, kind: MutationKind, where: Where
) -> None:
    with pytest.raises(AssertionError, match="parity failed"):
        _run_parity(spec, mutate=at(kind, where))
