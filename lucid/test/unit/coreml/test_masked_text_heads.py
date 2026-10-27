"""A masked text head answers a fresh mask after export (LCD-363).

``gpt_cls``, ``gpt2_small_cls`` and ``roformer_multiple_choice`` pool the
last token their padding mask keeps.  That position used to be read on
the host, so an exported package replayed the example's mask: on a new
one the Core ML scout measured relative errors of 1.38 and 2.40 for the
GPT heads, and 0.454 against outputs of 0.635 for RoFormer.  Since
LCD-337 the export refuses such a trace instead.

Each head is exported on an all-ones mask and asked about masks it has
never seen — left padding, right padding, a single token — and must agree
with eager on each.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

TINY = {
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "intermediate_size": 64,
}

L = 8

#: Two rows each, so a batch carries two different selections at once.
FRESH_MASKS = {
    "left_padding": [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1]],
    "right_padding": [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
    "single_token": [[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
    "mixed": [[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]],
}


def _feed(name: str, rows: list[list[int]]) -> tuple[Tensor, Tensor]:
    lucid.manual_seed(1)
    mask = lucid.tensor(rows).long()
    if name == "roformer_multiple_choice":
        # (N=1, C=2, L): the two rows are two choices of one question.
        return lucid.randint(0, 100, (1, 2, L)), mask.reshape(1, 2, L)
    return lucid.randint(0, 100, (2, L)), mask


@pytest.mark.parametrize("name", ["gpt_cls", "gpt2_small_cls", "roformer_multiple_choice"])
def test_an_exported_head_follows_a_fresh_mask(name: str, tmp_path: object) -> None:
    lucid.manual_seed(0)
    model = M.create_model(name, **TINY).eval()
    ids, first = _feed(name, [[1] * L, [1] * L])

    exported = cml.export(model, (ids, first), f"{tmp_path}/{name}.mlpackage")
    try:
        for label, rows in FRESH_MASKS.items():
            _ids, mask = _feed(name, rows)
            want = model(ids, mask).logits
            got = exported.predict((ids, mask))
            assert isinstance(got, Tensor)
            scale = float(want.abs().max().item())
            assert scale > 1e-6, f"{name}: logits too small to compare"
            err = float((got - want).abs().max().item()) / scale
            assert err <= 1e-4, f"{name} on {label}: relative error {err:.3g}"
    finally:
        exported.close()
