"""``generate()`` on an empty batch gives one shape, with an EOS id or without (LCD-1111).

Both decode loops stop early once every row has emitted EOS, and "every row" of zero rows is
vacuously true: an empty batch stopped after its first step when an EOS id was set (config or
call), and ran to the cap when none was, so the same call returned ``(0, 6)`` one way and
``(0, 8)`` the other.  The rule is owned by ``_SamplingParams.stops_early``, which both loops
(``CausalLMMixin.generate`` and ``seq2seq_generate``) ask; an empty batch has nothing to wait
for and never stops early.
"""

import pytest

import lucid
from lucid.models import create_model
from lucid.models._sampling import _SamplingParams

_V = 40
_TINY = dict(
    vocab_size=_V,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=16,
    pad_token_id=0,
)
_SEQ2SEQ = {
    "t5_small_seq2seq": dict(_TINY, num_decoder_layers=1, head_dim=8),
    "transformer_base_seq2seq": dict(_TINY, num_decoder_layers=1),
}
_EOS = 1


def _sampling(eos: int | None) -> _SamplingParams:
    return _SamplingParams(
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        eos_token_id=eos,
        pad_token_id=0,
        dev="cpu",
        cache_implementation="dynamic",
    )


def _empty(width: int) -> lucid.Tensor:
    return lucid.zeros((0, width)).long()


def test_stops_early_is_false_for_no_rows() -> None:
    assert _sampling(_EOS).stops_early(lucid.zeros((0,)).bool()) is False


def test_stops_early_needs_an_eos_id_and_every_row_finished() -> None:
    done = lucid.tensor([True, True])
    assert _sampling(None).stops_early(done) is False
    assert _sampling(_EOS).stops_early(done) is True
    assert _sampling(_EOS).stops_early(lucid.tensor([True, False])) is False


@pytest.mark.parametrize("cache", ["dynamic", "static"])
@pytest.mark.parametrize("use_cache", [True, False])
def test_a_causal_model_gives_an_empty_batch_one_shape(use_cache: bool, cache: str) -> None:
    model = create_model("gpt_lm", **_TINY).eval()
    kwargs = dict(max_new_tokens=5, use_cache=use_cache, cache_implementation=cache)
    plain = model.generate(_empty(3), eos_token_id=None, **kwargs)
    with_eos = model.generate(_empty(3), eos_token_id=_EOS, **kwargs)
    assert tuple(plain.shape) == tuple(with_eos.shape) == (0, 8)


def test_an_eos_id_from_the_config_does_not_change_the_shape_either() -> None:
    model = create_model("gpt_lm", **_TINY, eos_token_id=_EOS).eval()
    assert tuple(model.generate(_empty(3), max_new_tokens=5).shape) == (0, 8)


@pytest.mark.parametrize("name", sorted(_SEQ2SEQ))
def test_a_seq2seq_model_gives_an_empty_batch_one_shape(name: str) -> None:
    model = create_model(name, **_SEQ2SEQ[name]).eval()
    plain = model.generate(_empty(4), max_new_tokens=5, eos_token_id=None)
    with_eos = model.generate(_empty(4), max_new_tokens=5, eos_token_id=_EOS)
    assert tuple(plain.shape) == tuple(with_eos.shape)
    assert int(plain.shape[0]) == 0 and int(plain.shape[1]) > 1


def test_a_batch_that_has_rows_still_stops_when_they_all_finish() -> None:
    model = create_model("gpt_lm", **_TINY).eval()
    prompt = lucid.tensor([[3, 4, 5], [3, 4, 5]]).long()
    free = model.generate(prompt, max_new_tokens=6, eos_token_id=None)
    first = int(free[0, 3].item())
    stopped = model.generate(prompt, max_new_tokens=6, eos_token_id=first)
    assert tuple(free.shape) == (2, 9)
    assert tuple(stopped.shape) == (2, 4)  # the prompt and the one token that was EOS
