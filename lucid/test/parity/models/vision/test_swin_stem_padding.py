"""Swin's patch-embedding stem against the reference, off the patch grid.

A strided patch convolution drops the border of an image whose side is not a
multiple of ``patch_size``.  The reference (``SwinPatchEmbeddings.maybe_pad``)
zero-pads the right and bottom first and hands the padded grid to the window
stages, so for such an input the final hidden states, the pooled feature and
the logits only agree if the stem pads too.  Both are compared here with
Lucid's weights copied into the reference, at on-grid and off-grid sizes
(square and non-square) that keep every stage at least one window wide.
"""

import re
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.models as M
from lucid.test._helpers.compare import assert_close

_SIZES: dict[str, Any] = {
    "image_size": 32,
    "patch_size": 4,
    "embed_dim": 16,
    "depths": (1, 1, 1, 1),
    "num_heads": (1, 2, 2, 2),
    "window_size": 2,
    "num_classes": 3,
}

#: ``(height, width)`` — 36 is on the grid (a guard for the weight copy), the
#: rest are not: 35 is the case the card names, 33x37 is not square, 45x46 pads
#: a different amount on each side.  Every one keeps the last stage's map at
#: least one window wide, which is as small as the reference's own windows
#: allow (it raises on a map narrower than the window).
_SHAPES = [(36, 36), (35, 35), (33, 37), (45, 46)]


def _reference_name(name: str) -> str:
    """The ``SwinForImageClassification`` parameter a Lucid one maps onto."""
    if name.startswith("classifier."):
        return name
    name = re.sub(r"^patch_embed\.proj\.", "embeddings.patch_embeddings.projection.", name)
    name = re.sub(r"^patch_embed\.norm\.", "embeddings.norm.", name)
    name = re.sub(r"^norm\.", "layernorm.", name)
    name = re.sub(r"^stages\.", "encoder.layers.", name)
    name = name.replace(
        ".attn.rel_pos_bias", ".attention.relative_position_bias.relative_position_bias_table"
    )
    name = name.replace(".attn.proj.", ".attention.o_proj.")
    name = name.replace(".norm1.", ".layernorm_before.").replace(".norm2.", ".layernorm_after.")
    name = name.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.3.", ".mlp.fc2.")
    return "swin." + name.replace(".downsample.proj.", ".downsample.reduction.")


def _reference_state(model: Any) -> dict[str, np.ndarray]:
    """Lucid's weights under the reference's names; the fused ``qkv`` is split in three."""
    state: dict[str, np.ndarray] = {}
    for name, value in model.state_dict().items():
        array = value.numpy()
        if ".attn.qkv." in name:
            prefix, kind = name.split(".attn.qkv.")
            for part, rows in zip(("q", "k", "v"), np.split(array, 3, axis=0), strict=True):
                state[f"{_reference_name(prefix)}.attention.{part}_proj.{kind}"] = rows
        else:
            state[_reference_name(name)] = array
    return state


def _models(ref: Any) -> tuple[Any, Any]:
    """The Lucid classifier and the reference ``SwinForImageClassification`` with its weights."""
    transformers = pytest.importorskip("transformers")
    lucid.manual_seed(0)
    ours = M.create_model("swin_tiny_cls", **_SIZES).eval()
    # Distinct, non-trivial values everywhere, so a mis-copied tensor cannot hide.
    with lucid.no_grad():
        for parameter in ours.parameters():
            parameter.copy_(lucid.randn(*parameter.shape) * 0.3)
    config = transformers.SwinConfig(
        image_size=32,
        patch_size=4,
        num_channels=3,
        embed_dim=16,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 2, 2],
        window_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        num_labels=3,
    )
    theirs = transformers.SwinForImageClassification(config).eval()
    # strict: a Lucid tensor with no home, or a reference one left at its init, fails here.
    theirs.load_state_dict({k: ref.tensor(v) for k, v in _reference_state(ours).items()})
    return ours, theirs


def _tokens(model: Any, image: Any) -> Any:
    """The final ``(B, L, C)`` hidden states, before pooling."""
    x = model.patch_embed(image)
    for stage in model.stages:
        x = stage(x)
    x = model.norm(x)
    batch, height, width, channels = (int(d) for d in x.shape)
    return x.reshape(batch, height * width, channels)


@pytest.mark.parity
@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for h, w in _SHAPES])
def test_final_hidden_states_pooled_feature_and_logits_match_the_reference(
    shape: tuple[int, int], ref: Any
) -> None:
    ours, reference = _models(ref)
    pixels = np.random.default_rng(0).standard_normal((2, 3, *shape)).astype("float32")

    with ref.no_grad():
        theirs = reference.swin(ref.tensor(pixels))
        want_logits = reference(ref.tensor(pixels)).logits
    image = lucid.tensor(pixels)
    with lucid.no_grad():
        got_tokens = _tokens(ours, image)
        got_logits = ours(image).logits

    assert_close(got_tokens, theirs.last_hidden_state, atol=2e-5, rtol=1e-4)
    assert_close(got_tokens.mean(dim=1), theirs.pooler_output, atol=2e-5, rtol=1e-4)
    assert_close(got_logits, want_logits, atol=2e-5, rtol=1e-4)
