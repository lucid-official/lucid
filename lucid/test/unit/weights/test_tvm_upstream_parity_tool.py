"""The key map ``tools.check_tvm_upstream_parity`` moves random weights through."""

import pytest

from tools.check_tvm_upstream_parity import _lucid_key


def test_the_key_map_refuses_what_the_network_does_not_hold() -> None:
    assert _lucid_key("model.blocks.3.attn.qkv.weight") == "blocks.3.attn.qkv.weight"
    assert _lucid_key("model.final_layer.linear.bias") == "final.proj.bias"
    assert _lucid_key("model.pos_embed") is None
    for key in (
        "model.blocks.3.attn.extra.weight",
        "model.blocks.x.attn.qkv.weight",
        "model.unknown.weight",
        "sigma_data",
    ):
        with pytest.raises(KeyError):
            _lucid_key(key)
