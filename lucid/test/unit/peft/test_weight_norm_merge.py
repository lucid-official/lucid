"""LoRA merges into a weight-normalised layer by writing through its ``(g, v)`` pair.

Under ``nn.utils.weight_norm`` the layer has no ``weight`` parameter: its weight
is ``g v / ||v||``, recomputed by a hook every forward.  A merge must therefore
change ``g`` and ``v`` so the derived weight is the merged one, and an unmerge
must give back the very ``g`` and ``v`` the merge started from.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import LoraConfig, PeftModel, get_peft_model
from lucid.peft.tuners.lora._weight_site import _Normalised, _Stored, _WeightSite

_MAKERS: dict[str, tuple[Callable[[], nn.Module], tuple[int, ...]]] = {
    "linear": (lambda: nn.Linear(6, 5), (3, 6)),
    "conv2d": (lambda: nn.Conv2d(3, 4, 3, padding=1), (2, 3, 5, 5)),
    "conv1d": (lambda: nn.Conv1d(3, 4, 3, padding=1), (2, 3, 7)),
}


def _distance(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


@pytest.mark.parametrize("dora", [False, True], ids=["lora", "dora"])
@pytest.mark.parametrize("kind", sorted(_MAKERS))
class TestMergeThroughWeightNorm:
    def _adapted(self, kind: str, dora: bool) -> tuple[nn.Module, lucid.Tensor, PeftModel]:
        make, shape = _MAKERS[kind]
        lucid.manual_seed(0)
        layer = nn.utils.weight_norm(make())
        base = nn.Sequential(layer)
        config = LoraConfig(r=2, target_modules=["0"], init_lora_weights=False, use_dora=dora)
        return layer, lucid.randn(*shape), get_peft_model(base, config).eval()

    def test_merge_keeps_the_output(self, kind: str, dora: bool) -> None:
        _base, x, peft = self._adapted(kind, dora)
        want = peft(x)
        peft.merge_adapter()
        assert _distance(peft(x), want) < 1e-4

    def test_unmerge_restores_g_and_v(self, kind: str, dora: bool) -> None:
        layer, x, peft = self._adapted(kind, dora)
        g, v = layer.weight_g.clone(), layer.weight_v.clone()
        want = peft(x)
        peft.merge_adapter()
        assert _distance(layer.weight_g, g) > 0  # the merge wrote through g
        peft.unmerge_adapter()
        assert _distance(layer.weight_g, g) < 1e-5 and _distance(layer.weight_v, v) < 1e-5
        assert _distance(peft(x), want) < 1e-4

    def test_merge_and_unload_is_the_plain_model(self, kind: str, dora: bool) -> None:
        _base, x, peft = self._adapted(kind, dora)
        want = peft(x)
        merged = peft.merge_and_unload()
        assert _distance(merged(x), want) < 1e-4
        assert [n for n, _ in merged.named_parameters()] == ["0.bias", "0.weight_g", "0.weight_v"]


def test_a_site_exists_for_each_way_a_weight_is_stored() -> None:
    plain = nn.Linear(3, 2)
    normed = nn.utils.weight_norm(nn.Linear(3, 2))
    assert type(_WeightSite.of(plain)) is _Stored
    assert type(_WeightSite.of(normed)) is _Normalised
    with pytest.raises(AttributeError, match="no weight parameter"):
        _WeightSite.of(nn.ReLU())


def test_writing_a_zero_slice_leaves_the_layer_finite() -> None:
    layer = nn.utils.weight_norm(nn.Linear(3, 2))
    site = _WeightSite.of(layer)
    value = site.read().detach().clone()
    value[0] = 0.0
    with lucid.no_grad():
        site.write(value)
    assert bool(layer(lucid.randn(4, 3)).isfinite().all().item())
    assert _distance(site.read(), value) < 1e-6
