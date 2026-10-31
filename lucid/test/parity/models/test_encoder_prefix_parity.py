"""Parity: a key/value prefix in front of a bidirectional encoder, against the reference library.

The reference numbers the input after the cache (``past_key_values_length``): BERT looks up its
position embeddings from there, RoFormer rotates the input from that phase on; the cached keys and
values themselves are attended to as they are.  An adapter trained with a prefix on one is only
interchangeable with the other if this holds.
"""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.models import create_model
from lucid.test._helpers.compare import assert_close
from lucid.utils.cache import DynamicCache

_SIZES: dict[str, object] = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 50,
    "max_position_embeddings": 32,
}
_PREFIX, _LENGTH, _BATCH, _HEADS, _HEAD_DIM = 3, 5, 2, 2, 16


def _reference_model(family: str, ref: Any) -> Any:
    transformers = pytest.importorskip("transformers")
    ref.manual_seed(0)
    if family == "bert_base":
        config = transformers.BertConfig(
            **_SIZES, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0
        )
        return transformers.BertModel(config).eval()
    config = transformers.RoFormerConfig(
        **_SIZES, embedding_size=32, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0
    )
    return transformers.RoFormerModel(config).eval()


@pytest.mark.parity
class TestEncoderPrefixParity:
    @pytest.mark.parametrize("family", ["bert_base", "roformer"])
    def test_a_prefix_matches_the_reference(self, family: str, ref: Any) -> None:
        transformers = pytest.importorskip("transformers")
        reference = _reference_model(family, ref)
        model = create_model(family, hidden_dropout=0.0, attention_dropout=0.0, **_SIZES).eval()
        weights = {k: v.detach().numpy() for k, v in reference.state_dict().items()}
        model.load_state_dict(
            {k: lucid.tensor(weights[k]) for k in model.state_dict() if k in weights}, strict=False
        )

        rng = np.random.default_rng(0)
        ids = rng.integers(2, 50, (_BATCH, _LENGTH))
        shape = (_BATCH, _HEADS, _PREFIX, _HEAD_DIM)
        pairs = [
            (
                rng.standard_normal(shape).astype("float32"),
                rng.standard_normal(shape).astype("float32"),
            )
            for _ in range(2)
        ]
        ours, theirs = DynamicCache(), transformers.DynamicCache()
        for layer, (k, v) in enumerate(pairs):
            ours.update(lucid.tensor(k), lucid.tensor(v), layer)
            theirs.update(ref.tensor(k), ref.tensor(v), layer)

        with ref.no_grad():
            want = reference(
                ref.tensor(ids),
                attention_mask=ref.ones(_BATCH, _PREFIX + _LENGTH, dtype=ref.long),
                past_key_values=theirs,
                use_cache=False,
            ).last_hidden_state
        got = model(
            lucid.tensor(ids).long(),
            attention_mask=lucid.ones(_BATCH, _PREFIX + _LENGTH).long(),
            past_key_values=ours,
        ).last_hidden_state
        assert_close(got, want, atol=1e-5)
