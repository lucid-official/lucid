"""MobileNet v2 parity tests (1.0 / 0.75 width vs timm mobilenetv2_*).

Both widths align positionally; the 0.75 shape mismatch this file used to
record is gone (mobilenet_v2_075 builds 1,355,424 parameters and its
classifier 2,636,424, both matching the reference layer for layer)."""

import pytest
import lucid.models as M
from lucid.test.parity.models._registry import SPECS, ParitySpec
from lucid.test.parity.models._utils import (
    requires_timm,
    _run_parity,
    _run_self_consistency,
    _spec_param,
)

_FACTORIES = frozenset({M.mobilenet_v2_cls, M.mobilenet_v2_075_cls})
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
