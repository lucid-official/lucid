"""LCD-709: RoFormer's classification head is the reference's nested ``classifier.{dense,out_proj}``.

The head used to be a top-level ``dense`` plus ``classifier`` (so a fine-tuning
wrapper could not address it without matching the encoder's ``*.dense`` too),
and the trunk built a pooler no head read.  These pin the new layout, that the
numerics did not move, and that checkpoints written in the old layout load.
"""

import pytest

import lucid
from lucid.models import (
    RoFormerConfig,
    RoFormerForMaskedLM,
    RoFormerForMultipleChoice,
    RoFormerForQuestionAnswering,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerModel,
)
from lucid.models._utils._text import text_activation
from lucid.models.text.roformer._model import _RoFormerEncoderOnly

_HIDDEN = 32


def _config() -> RoFormerConfig:
    return RoFormerConfig(
        vocab_size=64,
        hidden_size=_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        num_labels=3,
    )


def _ids() -> lucid.Tensor:
    return lucid.tensor([[1, 2, 3, 4, 5]]).long()


def test_the_head_is_nested_and_no_pooler_is_built() -> None:
    names = set(RoFormerForSequenceClassification(_config()).state_dict())
    assert {
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    } <= names
    assert "dense.weight" not in names and "classifier.weight" not in names
    assert not any("pooler" in n for n in names)


@pytest.mark.parametrize(
    "head",
    [RoFormerForTokenClassification, RoFormerForMultipleChoice, RoFormerForQuestionAnswering],
)
def test_the_other_heads_build_no_dead_pooler(head: type) -> None:
    assert not any("pooler" in n for n in head(_config()).state_dict())


def test_the_bare_trunk_and_masked_lm_keep_the_pooler_the_shipped_weights_carry() -> None:
    assert any(n.startswith("pooler.") for n in RoFormerModel(_config()).state_dict())
    assert any(
        n.startswith("roformer.pooler.") for n in RoFormerForMaskedLM(_config()).state_dict()
    )


def test_the_encoder_only_trunk_returns_a_plain_encoder_output() -> None:
    out = _RoFormerEncoderOnly(_config()).eval()(_ids())
    assert tuple(out.last_hidden_state.shape) == (1, 5, _HIDDEN)
    assert not hasattr(out, "pooler_output")


def test_the_nested_head_computes_what_the_flat_one_did() -> None:
    # The old head: dropout -> dense -> activation -> dropout -> classifier on
    # the raw first-token state.  Same weights, new layout, same bits (eval).
    cfg = _config()
    model = RoFormerForSequenceClassification(cfg).eval()
    with lucid.no_grad():
        first = model.roformer(_ids()).last_hidden_state[:, 0, :]
        flat = model.classifier.out_proj(
            text_activation(cfg.hidden_act, model.classifier.dense(first))
        )
        assert bool(lucid.equal(model(_ids()).logits, flat))


def test_an_old_layout_checkpoint_still_loads() -> None:
    lucid.manual_seed(0)
    src = RoFormerForSequenceClassification(_config()).eval()
    old = {
        key.replace("classifier.dense.", "dense.").replace("classifier.out_proj.", "classifier."): v
        for key, v in src.state_dict().items()
    }
    # The old model also carried the trunk's never-called pooler.
    for key, value in RoFormerModel(_config()).state_dict().items():
        if key.startswith("pooler."):
            old[f"roformer.{key}"] = value
    assert "dense.weight" in old and "classifier.weight" in old

    lucid.manual_seed(1)
    dst = RoFormerForSequenceClassification(_config()).eval()
    result = dst.load_state_dict(old)
    assert not result.missing_keys and not result.unexpected_keys
    with lucid.no_grad():
        assert bool(lucid.equal(dst(_ids()).logits, src(_ids()).logits))


def test_a_new_layout_checkpoint_is_left_alone() -> None:
    src = RoFormerForSequenceClassification(_config()).eval()
    dst = RoFormerForSequenceClassification(_config()).eval()
    result = dst.load_state_dict(src.state_dict())
    assert not result.missing_keys and not result.unexpected_keys
    with lucid.no_grad():
        assert bool(lucid.equal(dst(_ids()).logits, src(_ids()).logits))


def test_a_pooler_less_trunk_loads_encoder_weights_that_include_a_pooler() -> None:
    bare = _RoFormerEncoderOnly(_config())
    result = bare.load_state_dict(RoFormerModel(_config()).state_dict())
    assert not result.missing_keys and not result.unexpected_keys
