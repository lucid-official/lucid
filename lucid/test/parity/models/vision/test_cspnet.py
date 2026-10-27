"""CSPNet parity tests.

All three registered variants have a reference counterpart.  Only
cspresnet_50 was listed here for a long time, and the two that were not
had both drifted: cspdarknet_53 activated at darknet's leaky slope of
0.1 where its checkpoint was trained at 0.01, and cspresnext_50 split
two cross-stages with an activation the checkpoint leaves linear.
Neither could be seen from a load — an activation has no weights — and
neither was ever compared."""

import pytest
import lucid.models as M
from lucid.test.parity.models._registry import SPECS, ParitySpec
from lucid.test.parity.models._utils import (
    requires_timm,
    _run_parity,
    _run_self_consistency,
    _spec_param,
)

_FACTORIES = frozenset({M.cspresnet_50_cls, M.cspresnext_50_cls, M.cspdarknet_53_cls})
_SPECS = [s for s in SPECS if s.lucid_factory in _FACTORIES]
_TIMM = [s for s in _SPECS if s.timm_name is not None]
_SC = [s for s in _SPECS if s.timm_name is None]

if _TIMM:

    @requires_timm
    @pytest.mark.parametrize("spec", [_spec_param(s) for s in _TIMM])
    def test_parity(spec: ParitySpec) -> None:
        """Numeric logit parity against timm reference."""
        _run_parity(spec)


if _SC:

    @pytest.mark.parametrize("spec", [_spec_param(s) for s in _SC])
    def test_self_consistency(spec: ParitySpec) -> None:
        """Deterministic forward pass (no external reference)."""
        _run_self_consistency(spec)
