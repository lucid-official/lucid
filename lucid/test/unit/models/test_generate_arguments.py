"""``generate()`` takes what ``forward`` and the reference already have (LCD-1110).

* Decoder-only ``generate`` takes ``attention_mask``: a batch of prompts
  left-padded to one width decodes each row as that prompt would alone, with
  the pad masked at every step and the positions counted from the first real
  token.  A mask is read as ``forward`` reads one (above 0.5 is a real token),
  whatever its dtype.  A host whose ``forward`` cannot take the mask is refused
  (``TypeError``; the reference raises ``ValueError`` -- a recorded deviation),
  not ignored.
* The encoder-decoder ``generate`` (T5, Transformer) takes ``max_new_tokens``,
  counted after the start token, and it wins over ``max_length``.

The families come through ``create_model`` on the real factories with small
config overrides -- there is no ``_tiny`` factory to reach for (H11).
"""

import inspect
from collections.abc import Callable
from typing import Any

import pytest

import lucid
import lucid.nn as nn
import lucid.models as M
from lucid.models import CausalLMOutput
from lucid.models._mixins import CausalLMMixin
from lucid.models.text.t5._model import T5ForSeq2SeqLM
from lucid.models.text.transformer._model import TransformerForSeq2SeqLM
from lucid.test._helpers.compare import assert_close
from lucid.utils.cache import StaticCache

# ``initializer_range`` is widened so the greedy tokens actually vary with the
# context: at the default a tiny random model emits one token forever, and a
# padding bug could not show.  ``eos_token_id=None`` keeps every row running to
# the cap.
_DECODER_ONLY: dict[str, dict[str, object]] = {"gpt_lm": {}, "gpt2_small_lm": {}}
_DECODER_SIZE: dict[str, object] = {
    "vocab_size": 50,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
    "eos_token_id": None,
    "initializer_range": 1.0,
}
_SEQ2SEQ: dict[str, dict[str, object]] = {
    "t5_small_seq2seq": {
        "vocab_size": 50,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "head_dim": 8,
        "num_hidden_layers": 2,
        "num_decoder_layers": 2,
        "intermediate_size": 64,
        "eos_token_id": None,
    },
    "transformer_base_seq2seq": {
        "vocab_size": 50,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_decoder_layers": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 32,
        "eos_token_id": None,
    },
}

#: The attribute each encoder-decoder family keeps its ``encode`` trunk under.
_SEQ2SEQ_TRUNK = {"t5_small_seq2seq": "t5", "transformer_base_seq2seq": "transformer"}

_PROMPTS = [[3, 7, 1, 9, 2], [4, 5, 6], [8, 2]]
_NEW_TOKENS = 6
#: ``(use_cache, cache_implementation)`` -- the cached paths and the re-encoding one.
_CACHES = [(True, "dynamic"), (True, "static"), (False, "dynamic")]
_CACHE_IDS = ["dynamic-cache", "static-cache", "no-cache"]


def _decoder(name: str, device: str) -> nn.Module:
    lucid.manual_seed(0)
    model = M.create_model(name, **_DECODER_SIZE)
    return model.to(device).eval()


def _left_padded(
    prompts: list[list[int]], device: str, pad: int = 0
) -> tuple[lucid.Tensor, lucid.Tensor]:
    """``(ids, mask)`` with each prompt right-aligned in one width, the pad on the left."""
    width = max(len(p) for p in prompts)
    ids = [[pad] * (width - len(p)) + p for p in prompts]
    mask = [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
    return (lucid.tensor(ids, device=device).long(), lucid.tensor(mask, device=device).long())


def _spy_static_cache(monkeypatch: pytest.MonkeyPatch, module: str) -> list[int]:
    """Record the ``max_cache_len`` of every ``StaticCache`` that ``module`` builds."""
    sizes: list[int] = []

    def spy(*, max_cache_len: int) -> StaticCache:
        sizes.append(max_cache_len)
        return StaticCache(max_cache_len=max_cache_len)

    monkeypatch.setattr(f"{module}.StaticCache", spy)
    return sizes


def _alone(model: nn.Module, prompt: list[int], device: str, **kwargs: object) -> list[int]:
    """What ``prompt`` generates with no padding and no mask, past the prompt itself."""
    out = model.generate(lucid.tensor([prompt], device=device).long(), **kwargs)
    return [int(t) for t in out[0, len(prompt) :].tolist()]


# ─────────────────────────────────────────────────────────────────────────────
# Decoder-only: attention_mask
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(_DECODER_ONLY))
class TestLeftPaddedBatch:
    @pytest.mark.parametrize(("use_cache", "cache_implementation"), _CACHES, ids=_CACHE_IDS)
    def test_each_row_equals_its_prompt_run_alone(
        self, name: str, device: str, use_cache: bool, cache_implementation: str
    ) -> None:
        model = _decoder(name, device)
        options = {
            "max_new_tokens": _NEW_TOKENS,
            "use_cache": use_cache,
            "cache_implementation": cache_implementation,
        }
        ids, mask = _left_padded(_PROMPTS, device)
        out = model.generate(ids, attention_mask=mask, **options)

        width = int(ids.shape[1])
        assert tuple(out.shape) == (len(_PROMPTS), width + _NEW_TOKENS)
        assert_close(out[:, :width], ids, atol=0.0)  # the prompt comes back as given
        for row, prompt in enumerate(_PROMPTS):
            assert [int(t) for t in out[row, width:].tolist()] == _alone(
                model, prompt, device, **options
            ), f"row {row} (length {len(prompt)}) drifted from its unpadded run"

    def test_without_the_mask_the_pad_changes_the_answer(self, name: str, device: str) -> None:
        # The guard that gives the test above its teeth: were the pad harmless on
        # these models, the equality would hold with no mask at all.
        model = _decoder(name, device)
        ids, _ = _left_padded(_PROMPTS, device)
        out = model.generate(ids, max_new_tokens=_NEW_TOKENS)
        width = int(ids.shape[1])
        drifted = [
            [int(t) for t in out[row, width:].tolist()]
            != _alone(model, prompt, device, max_new_tokens=_NEW_TOKENS)
            for row, prompt in enumerate(_PROMPTS)
            if len(prompt) < width
        ]
        assert all(drifted)

    def test_forward_positions_follow_the_mask(self, name: str, device: str) -> None:
        # What ``generate`` hands ``forward``: the mask, and positions counted from
        # the first real token.  The real slots must match the prompt run alone.
        model = _decoder(name, device)
        ids, mask = _left_padded([[3, 7, 1, 9, 2], [4, 5, 6]], device)
        positions = lucid.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 2]], device=device).long()
        alone = model(lucid.tensor([[4, 5, 6]], device=device).long()).logits
        right = model(ids, attention_mask=mask, position_ids=positions).logits
        assert_close(right[1, 2:], alone[0], atol=1e-4)

        # Mask alone is not enough: the cache-offset positions put the row's
        # first token at slot 2, not 0.
        wrong = model(ids, attention_mask=mask).logits
        assert float((wrong[1, 2:] - alone[0]).abs().max().item()) > 1e-3

    @pytest.mark.parametrize("dtype", [lucid.bool_, lucid.float32, lucid.float16])
    def test_a_mask_of_any_dtype_is_the_same_as_an_integer_one(
        self, name: str, device: str, dtype: lucid.dtype
    ) -> None:
        # ``forward`` and the reference both read a float 0/1 mask, so ``generate`` does.
        model = _decoder(name, device)
        ids, mask = _left_padded(_PROMPTS, device)
        as_int = model.generate(ids, attention_mask=mask, max_new_tokens=_NEW_TOKENS)
        other = model.generate(ids, attention_mask=mask.to(dtype), max_new_tokens=_NEW_TOKENS)
        assert_close(other, as_int, atol=0.0)

    def test_a_mask_of_ones_is_no_mask(self, name: str, device: str) -> None:
        model = _decoder(name, device)
        ids = lucid.tensor([[3, 7, 1, 9, 2], [4, 5, 6, 8, 2]], device=device).long()
        masked = model.generate(
            ids, attention_mask=lucid.ones_like(ids), max_new_tokens=_NEW_TOKENS
        )
        plain = model.generate(ids, max_new_tokens=_NEW_TOKENS)
        assert_close(masked, plain, atol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# The mask a host is handed, step by step
# ─────────────────────────────────────────────────────────────────────────────


class _RecordingLM(CausalLMMixin, nn.Module):
    """A host that always predicts token 1 and notes what ``forward`` was given."""

    _VOCAB = 5

    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "C", (), {"eos_token_id": None, "pad_token_id": 0, "vocab_size": self._VOCAB}
        )()
        self.calls: list[dict[str, object]] = []

    def _record(
        self,
        input_ids: lucid.Tensor,
        attention_mask: lucid.Tensor | None = None,
        position_ids: lucid.Tensor | None = None,
    ) -> CausalLMOutput:
        self.calls.append(
            {
                "width": int(input_ids.shape[1]),
                "mask": None if attention_mask is None else attention_mask.tolist(),
                "positions": None if position_ids is None else position_ids.tolist(),
            }
        )
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        logits = lucid.zeros((B, T, self._VOCAB))
        logits[:, :, 1] = 5.0
        return CausalLMOutput(logits=logits, loss=None)


class _MaskAndPositionsLM(_RecordingLM):
    def forward(
        self,
        input_ids: lucid.Tensor,
        attention_mask: lucid.Tensor | None = None,
        *,
        position_ids: lucid.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: object = None,
    ) -> CausalLMOutput:
        return self._record(input_ids, attention_mask, position_ids)


class _MaskOnlyLM(_RecordingLM):
    def forward(
        self,
        input_ids: lucid.Tensor,
        attention_mask: lucid.Tensor | None = None,
        *,
        use_cache: bool = False,
        past_key_values: object = None,
    ) -> CausalLMOutput:
        return self._record(input_ids, attention_mask)


class _NoMaskLM(_RecordingLM):
    def forward(
        self, input_ids: lucid.Tensor, *, use_cache: bool = False, past_key_values: object = None
    ) -> CausalLMOutput:
        return self._record(input_ids)


_LEFT_PADDED_IDS = [[0, 0, 4, 5], [3, 7, 1, 9]]
_LEFT_PADDED_MASK = [[0, 0, 1, 1], [1, 1, 1, 1]]


def _generate_recording(host: _RecordingLM, **kwargs: object) -> list[dict[str, object]]:
    host.generate(
        lucid.tensor(_LEFT_PADDED_IDS).long(),
        attention_mask=lucid.tensor(_LEFT_PADDED_MASK).long(),
        max_new_tokens=2,
        **kwargs,
    )
    return host.calls


class TestMaskReachesForward:
    def test_the_mask_grows_a_kept_column_per_token_and_positions_follow(self) -> None:
        calls = _generate_recording(_MaskAndPositionsLM())
        # With a cache the prompt goes in whole, then one token per step -- and
        # the mask covers the cached history as well as the new token.
        assert [c["width"] for c in calls] == [4, 1]
        assert calls[0]["mask"] == _LEFT_PADDED_MASK
        assert calls[1]["mask"] == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
        # cumsum(mask) - 1, a pad slot at 0 and a new token placed
        # after the row's last real position, not after the pad.
        assert calls[0]["positions"] == [[0, 0, 0, 1], [0, 1, 2, 3]]
        assert calls[1]["positions"] == [[2], [4]]

    def test_without_a_cache_the_whole_prefix_and_its_mask_go_in_each_step(self) -> None:
        calls = _generate_recording(_MaskAndPositionsLM(), use_cache=False)
        assert [c["width"] for c in calls] == [4, 5]
        assert calls[1]["mask"] == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
        assert calls[1]["positions"] == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]

    def test_a_host_with_no_position_ids_is_given_the_mask_only(self) -> None:
        calls = _generate_recording(_MaskOnlyLM())
        # No TypeError fall-back to re-encoding either: the cached feed stays 4, 1.
        assert [c["width"] for c in calls] == [4, 1]
        assert all(c["positions"] is None for c in calls)
        assert calls[1]["mask"] == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]

    def test_a_float_mask_is_read_above_one_half(self) -> None:
        host = _MaskAndPositionsLM()
        ids = lucid.tensor(_LEFT_PADDED_IDS).long()
        mask = lucid.tensor([[0.0, 0.2, 0.9, 1.0], [1.0, 1.0, 1.0, 1.0]])
        host.generate(ids, attention_mask=mask, max_new_tokens=2)
        assert host.calls[0]["mask"] == _LEFT_PADDED_MASK
        assert host.calls[0]["positions"] == [[0, 0, 0, 1], [0, 1, 2, 3]]

    def test_a_masked_slot_after_a_real_one_is_positioned_at_zero(self) -> None:
        # Holes and right padding, which a left-padded batch never has: the
        # reference fills every masked slot with 0, so ``cumsum - 1`` is not
        # clamped there (a clamp would repeat the last real position instead).
        host = _MaskAndPositionsLM()
        ids = lucid.tensor([[3, 7, 0, 9], [3, 7, 0, 0]]).long()
        mask = lucid.tensor([[1, 1, 0, 1], [1, 1, 0, 0]]).long()
        host.generate(ids, attention_mask=mask, max_new_tokens=2)
        assert host.calls[0]["positions"] == [[0, 1, 0, 2], [0, 1, 0, 0]]
        # The generated token follows the row's real tokens, hole or not.
        assert host.calls[1]["positions"] == [[3], [2]]

    def test_no_mask_hands_forward_neither(self) -> None:
        host = _MaskAndPositionsLM()
        host.generate(lucid.tensor(_LEFT_PADDED_IDS).long(), max_new_tokens=2)
        assert all(c["mask"] is None and c["positions"] is None for c in host.calls)


# ─────────────────────────────────────────────────────────────────────────────
# Refusals
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskRefusals:
    def _host_and_ids(self) -> tuple[_MaskAndPositionsLM, lucid.Tensor]:
        return _MaskAndPositionsLM(), lucid.tensor(_LEFT_PADDED_IDS).long()

    @pytest.mark.parametrize("shape", [(2, 3), (2, 5), (1, 4), (8,), (2, 4, 1)])
    def test_a_mask_that_is_not_the_shape_of_the_prompt_is_a_value_error(
        self, shape: tuple[int, ...]
    ) -> None:
        host, ids = self._host_and_ids()
        with pytest.raises(ValueError, match="shape of input_ids"):
            host.generate(ids, attention_mask=lucid.ones(shape).long(), max_new_tokens=2)
        assert host.calls == []  # refused before any forward ran

    @pytest.mark.parametrize(
        "dtype", [lucid.bool_, lucid.int32, lucid.int64, lucid.float16, lucid.float32]
    )
    def test_masks_of_any_dtype_are_taken(self, dtype: lucid.dtype) -> None:
        host, ids = self._host_and_ids()
        mask = lucid.ones((2, 4), dtype=lucid.int64).to(dtype)
        host.generate(ids, attention_mask=mask, max_new_tokens=1)
        assert len(host.calls) == 1

    def test_a_host_whose_forward_takes_no_mask_refuses_one(self) -> None:
        host = _NoMaskLM()
        ids = lucid.tensor(_LEFT_PADDED_IDS).long()
        with pytest.raises(TypeError, match="takes no attention_mask"):
            host.generate(ids, attention_mask=lucid.ones((2, 4)).long(), max_new_tokens=2)
        assert host.calls == []
        # ... and without one it still generates.
        assert host.generate(ids, max_new_tokens=2).shape == (2, 6)

    def test_the_compiled_decode_step_refuses_a_mask(self) -> None:
        host, ids = self._host_and_ids()
        with pytest.raises(NotImplementedError, match="compiled"):
            host.generate(
                ids,
                attention_mask=lucid.ones((2, 4)).long(),
                max_new_tokens=2,
                cache_implementation="compiled",
            )
        assert host.calls == []

    def test_a_mask_on_another_device_is_a_value_error(self, device_gpu_only: str) -> None:
        host, ids = self._host_and_ids()
        mask = lucid.ones((2, 4), device=device_gpu_only).long()
        with pytest.raises(ValueError, match="is on metal but input_ids is on cpu"):
            host.generate(ids, attention_mask=mask, max_new_tokens=2)

    @pytest.mark.parametrize("name", list(_DECODER_ONLY))
    def test_position_ids_must_cover_the_new_tokens(self, name: str) -> None:
        model = _decoder(name, "cpu")
        ids = lucid.tensor([[3, 7, 1, 9]]).long()
        with pytest.raises(ValueError, match="position_ids must have shape"):
            model(ids, position_ids=lucid.arange(3).reshape(1, 3).long())


# ─────────────────────────────────────────────────────────────────────────────
# max_new_tokens
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(_DECODER_ONLY))
class TestDecoderOnlyMaxNewTokens:
    def test_counts_tokens_added_to_the_prompt(self, name: str, device: str) -> None:
        model = _decoder(name, device)
        ids = lucid.tensor([[3, 7, 1, 9], [4, 5, 6, 8]], device=device).long()
        assert tuple(model.generate(ids, max_new_tokens=3).shape) == (2, 7)

    def test_wins_over_max_length_both_ways(self, name: str, device: str) -> None:
        model = _decoder(name, device)
        ids = lucid.tensor([[3, 7, 1, 9]], device=device).long()
        assert tuple(model.generate(ids, max_length=30, max_new_tokens=2).shape) == (1, 6)
        assert tuple(model.generate(ids, max_length=5, max_new_tokens=6).shape) == (1, 10)

    def test_zero_returns_the_prompt(self, name: str, device: str) -> None:
        model = _decoder(name, device)
        ids = lucid.tensor([[3, 7, 1, 9]], device=device).long()
        assert_close(model.generate(ids, max_new_tokens=0), ids, atol=0.0)

    def test_negative_is_a_value_error(self, name: str, device: str) -> None:
        model = _decoder(name, device)
        ids = lucid.tensor([[3, 7, 1, 9]], device=device).long()
        with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
            model.generate(ids, max_new_tokens=-1)

    def test_the_static_buffer_is_sized_by_it(
        self, name: str, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``max_cache_len`` defaults to the length cap -- prompt plus new tokens
        # here, not the (larger) ``max_length``.
        sizes = _spy_static_cache(monkeypatch, "lucid.models._mixins")
        model = _decoder(name, device)
        ids = lucid.tensor([[3, 7, 1, 9]], device=device).long()
        model.generate(ids, max_length=100, max_new_tokens=3, cache_implementation="static")
        assert sizes == [4 + 3]
        model.generate(ids, max_length=9, cache_implementation="static")
        assert sizes == [7, 9]


@pytest.mark.parametrize("name", list(_SEQ2SEQ))
class TestSeq2SeqMaxNewTokens:
    def _model(self, name: str, device: str) -> nn.Module:
        lucid.manual_seed(0)
        return M.create_model(name, **_SEQ2SEQ[name]).to(device).eval()

    @pytest.mark.parametrize(("use_cache", "cache_implementation"), _CACHES, ids=_CACHE_IDS)
    def test_counts_tokens_after_the_start_token(
        self, name: str, device: str, use_cache: bool, cache_implementation: str
    ) -> None:
        model = self._model(name, device)
        src = lucid.tensor([[3, 7, 1, 9, 2], [4, 5, 6, 8, 2]], device=device).long()
        options = {"use_cache": use_cache, "cache_implementation": cache_implementation}
        out = model.generate(src, max_new_tokens=5, **options)
        assert tuple(out.shape) == (2, 6)  # the start token, then five
        # The same cap spelled as a total length is the same decode.
        assert_close(out, model.generate(src, max_length=6, **options), atol=0.0)

    def test_wins_over_max_length_both_ways(self, name: str, device: str) -> None:
        model = self._model(name, device)
        src = lucid.tensor([[3, 7, 1, 9, 2]], device=device).long()
        assert tuple(model.generate(src, max_length=30, max_new_tokens=3).shape) == (1, 4)
        assert tuple(model.generate(src, max_length=2, max_new_tokens=5).shape) == (1, 6)

    def test_zero_is_the_start_token_alone(self, name: str, device: str) -> None:
        model = self._model(name, device)
        src = lucid.tensor([[3, 7, 1, 9, 2]], device=device).long()
        assert tuple(model.generate(src, max_new_tokens=0).shape) == (1, 1)

    def test_negative_is_a_value_error(self, name: str, device: str) -> None:
        model = self._model(name, device)
        src = lucid.tensor([[3, 7, 1, 9, 2]], device=device).long()
        with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
            model.generate(src, max_new_tokens=-1)

    def test_negative_is_refused_before_the_encoder_runs(
        self, name: str, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = self._model(name, device)
        trunk = getattr(model, _SEQ2SEQ_TRUNK[name])
        encoded: list[int] = []
        encode = trunk.encode

        def counting_encode(*args: Any, **kwargs: Any) -> lucid.Tensor:
            encoded.append(1)
            result: lucid.Tensor = encode(*args, **kwargs)
            return result

        monkeypatch.setattr(trunk, "encode", counting_encode)
        src = lucid.tensor([[3, 7, 1, 9, 2]], device=device).long()
        with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
            model.generate(src, max_new_tokens=-1)
        assert encoded == []

    def test_the_static_buffer_is_sized_by_it(
        self, name: str, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``max_cache_len`` defaults to the length cap; with ``max_new_tokens`` that
        # cap is 1 + N, not the (here larger) ``max_length``.
        sizes = _spy_static_cache(monkeypatch, "lucid.models.text._common._seq2seq")
        model = self._model(name, device)
        src = lucid.tensor([[3, 7, 1, 9, 2]], device=device).long()
        static = model.generate(
            src, max_length=100, max_new_tokens=4, cache_implementation="static"
        )
        assert sizes == [1 + 4]
        model.generate(src, max_length=9, cache_implementation="static")
        assert sizes == [5, 9]
        assert_close(static, model.generate(src, max_new_tokens=4), atol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Class-wide guards
# ─────────────────────────────────────────────────────────────────────────────


def _hosts(root: type) -> list[type]:
    """Every zoo class below ``root``, found by walking subclasses."""
    found: list[type] = []
    for sub in root.__subclasses__():
        if sub.__module__.startswith("lucid.models"):
            found.append(sub)
        found.extend(_hosts(sub))
    return found


class TestSignatures:
    def test_every_zoo_causal_lm_takes_a_mask_and_positions_in_forward(self) -> None:
        # A decoder-only family that drops either would turn ``generate(attention_mask=)``
        # into a refusal, or a left-padded row into a shifted one, with no test noticing.
        hosts = _hosts(CausalLMMixin)
        assert hosts, "no CausalLMMixin host found in lucid.models"
        for host in hosts:
            params = inspect.signature(host.forward).parameters
            assert "attention_mask" in params, host.__name__
            assert "position_ids" in params, host.__name__

    @pytest.mark.parametrize(
        ("generate", "names"),
        [
            (CausalLMMixin.generate, ("attention_mask", "max_new_tokens")),
            (T5ForSeq2SeqLM.generate, ("max_new_tokens", "attention_mask")),
            (TransformerForSeq2SeqLM.generate, ("max_new_tokens", "attention_mask")),
        ],
        ids=["decoder-only", "t5", "transformer"],
    )
    def test_generate_takes_both_as_keywords(
        self, generate: Callable[..., object], names: tuple[str, ...]
    ) -> None:
        params = inspect.signature(generate).parameters
        for name in names:
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
            assert params[name].default is None, name
