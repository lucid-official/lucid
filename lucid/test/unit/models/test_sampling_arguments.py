"""``generate()`` refuses the sampling arguments the reference refuses (LCD-1108).

``temperature=0.0`` used to fill the logits with NaN and fall back to the pad id,
``top_k=0`` crashed in ``topk``, ``repetition_penalty=0.0`` was accepted, and an
unknown ``cache_implementation`` was read as ``"dynamic"``.  The arguments are now
checked once, where :class:`_SamplingParams` is built, so the causal loop and the
seq2seq loops (which share it) refuse alike.

What is an error and what is "off" follows the reference's logits processors:

* ``temperature`` must be strictly positive; ``1.0`` is off.
* ``top_k`` ``None`` or ``0`` is off; a negative or non-integer value is an error.
  (``top_k <= 0`` is *not* uniformly "off" -- only ``0`` is.)
* ``top_p`` must lie in ``[0, 1]``; ``1.0`` is off and ``0.0`` keeps just the argmax.
* ``repetition_penalty`` must be strictly positive, greedy or not; ``1.0`` is off.
* The sampling-only knobs are not read under greedy decoding, so they are not
  checked there either.

The messages are the reference's own, so they are compared as literals here and
against the live reference in the ``parity`` class.

Left-padded prompts and the repetition penalty
----------------------------------------------
The penalty runs over the whole running sequence, pad ids included.  The reference
does exactly that -- its processor reads ``input_ids``, pads and all, and never
looks at the attention mask -- so a padded row penalises its pad id like any seen
token.  Lucid keeps that on purpose (parity over a "cleaner" mask-aware penalty):
``TestPadIdsAreSeenTokens`` pins it, and the parity class compares it with the
reference row for row.
"""

import math

import pytest

import lucid
from lucid.models import create_model
from lucid.models._sampling import (
    _apply_repetition_penalty,
    _check_cache_implementation,
    _next_token,
    _SamplingParams,
    _top_k_filter,
    _top_p_filter,
)

_NAN = float("nan")
_INF = float("inf")

_TEMPERATURE = "`temperature` (={}) has to be a strictly positive float, otherwise your next token scores will be invalid."
_GREEDY_HINT = " If you're looking for greedy decoding strategies, set `do_sample=False`."
_TOP_K = "`top_k` has to be a strictly positive integer, but is {}"
_TOP_P = "`top_p` has to be a float > 0 and < 1, but is {}"
_PENALTY = "`penalty` has to be a strictly positive float, but is {}"
_CACHE = "Invalid `cache_implementation` ({}). Choose one of: ('dynamic', 'static', 'compiled')"


def _params(**overrides: object) -> _SamplingParams:
    knobs: dict[str, object] = dict(
        do_sample=True,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        eos_token_id=None,
        pad_token_id=0,
        dev="cpu",
    )
    return _SamplingParams(**{**knobs, **overrides})


# (the keyword, the bad value, the message the reference gives for it)
_REFUSED = [
    pytest.param(
        "temperature", 0.0, _TEMPERATURE.format(0.0) + _GREEDY_HINT, id="temperature-zero"
    ),
    pytest.param(
        "temperature", 0, _TEMPERATURE.format(0) + _GREEDY_HINT, id="temperature-int-zero"
    ),
    pytest.param("temperature", -1.0, _TEMPERATURE.format(-1.0), id="temperature-negative"),
    pytest.param("temperature", _NAN, _TEMPERATURE.format(_NAN), id="temperature-nan"),
    pytest.param("temperature", "0.5", _TEMPERATURE.format("0.5"), id="temperature-str"),
    pytest.param("temperature", True, _TEMPERATURE.format(True), id="temperature-bool"),
    pytest.param("top_k", -1, _TOP_K.format(-1), id="top_k-negative"),
    pytest.param("top_k", 2.5, _TOP_K.format(2.5), id="top_k-float"),
    pytest.param("top_k", True, _TOP_K.format(True), id="top_k-bool"),
    pytest.param("top_p", -0.1, _TOP_P.format(-0.1), id="top_p-negative"),
    pytest.param("top_p", 1.5, _TOP_P.format(1.5), id="top_p-above-one"),
    pytest.param("top_p", _NAN, _TOP_P.format(_NAN), id="top_p-nan"),
    pytest.param("repetition_penalty", 0.0, _PENALTY.format(0.0), id="penalty-zero"),
    pytest.param("repetition_penalty", -1.0, _PENALTY.format(-1.0), id="penalty-negative"),
    pytest.param("repetition_penalty", _NAN, _PENALTY.format(_NAN), id="penalty-nan"),
    pytest.param("repetition_penalty", None, _PENALTY.format(None), id="penalty-none"),
]

_ACCEPTED = [
    pytest.param("temperature", 1.0, id="temperature-one-is-off"),
    pytest.param("temperature", 1, id="temperature-int-one"),
    pytest.param("temperature", 0.3, id="temperature-small"),
    pytest.param("temperature", 2, id="temperature-int"),
    pytest.param("temperature", _INF, id="temperature-inf"),
    pytest.param("top_k", None, id="top_k-none"),
    pytest.param("top_k", 0, id="top_k-zero-is-off"),
    pytest.param("top_k", 1, id="top_k-one"),
    pytest.param("top_k", 10**6, id="top_k-above-vocab"),
    pytest.param("top_p", None, id="top_p-none"),
    pytest.param("top_p", 0.0, id="top_p-zero"),
    pytest.param("top_p", 0, id="top_p-int-zero"),
    pytest.param("top_p", 0.5, id="top_p-half"),
    pytest.param("top_p", 1.0, id="top_p-one-is-off"),
    pytest.param("top_p", 1, id="top_p-int-one"),
    pytest.param("repetition_penalty", 1.0, id="penalty-one-is-off"),
    pytest.param("repetition_penalty", 1, id="penalty-int-one"),
    pytest.param("repetition_penalty", 0.5, id="penalty-reward"),
    pytest.param("repetition_penalty", 2, id="penalty-int"),
    pytest.param("repetition_penalty", _INF, id="penalty-inf"),
]


class TestTheKnobsAreChecked:
    @pytest.mark.parametrize(("name", "value", "message"), _REFUSED)
    def test_a_bad_value_is_a_value_error_with_the_reference_message(
        self, name: str, value: object, message: str
    ) -> None:
        with pytest.raises(ValueError) as caught:
            _params(**{name: value})
        assert str(caught.value) == message

    @pytest.mark.parametrize(("name", "value"), _ACCEPTED)
    def test_a_good_value_builds(self, name: str, value: object) -> None:
        assert getattr(_params(**{name: value}), name) == value

    def test_the_repetition_penalty_is_checked_under_greedy_decoding_too(self) -> None:
        with pytest.raises(ValueError, match="strictly positive float"):
            _params(do_sample=False, repetition_penalty=0.0)

    @pytest.mark.parametrize("name", ["temperature", "top_k", "top_p"])
    def test_the_sampling_only_knobs_are_not_read_so_not_checked_under_greedy(
        self, name: str
    ) -> None:
        bad = {"temperature": -1.0, "top_k": -1, "top_p": 7.0}[name]
        assert getattr(_params(do_sample=False, **{name: bad}), name) == bad

    @pytest.mark.parametrize("name", ["dynamic", "static", "compiled"])
    def test_every_cache_name_the_loops_know_is_accepted(self, name: str) -> None:
        _check_cache_implementation(name)

    @pytest.mark.parametrize("name", ["foo", "", "Static", "paged", 3, None])
    def test_an_unknown_cache_name_is_a_value_error_with_the_reference_message(
        self, name: object
    ) -> None:
        with pytest.raises(ValueError) as caught:
            _check_cache_implementation(name)
        assert str(caught.value) == _CACHE.format(name)


def _next(logits: list[list[float]], **overrides: object) -> list[int]:
    out = _next_token(
        lucid.tensor(logits), [lucid.zeros((len(logits),)).long()], _params(**overrides)
    )
    return [int(t) for t in out.numpy().tolist()]


class TestOffMeansOff:
    """The values the reference reads as "off" change nothing, bit for bit."""

    _LOGITS = [[0.1, 0.9, 0.5, 0.3, 0.7], [0.4, 0.2, 0.8, 0.6, 0.0]]

    @pytest.mark.parametrize("off", [{"top_k": 0}, {"top_p": 1.0}, {"temperature": 1.0}])
    def test_the_draw_is_the_unfiltered_draw(self, off: dict[str, object]) -> None:
        for seed in range(8):
            lucid.manual_seed(seed)
            plain = _next(self._LOGITS)
            lucid.manual_seed(seed)
            assert _next(self._LOGITS, **off) == plain

    def test_top_p_zero_keeps_only_the_most_likely_token(self) -> None:
        kept = _top_p_filter(lucid.tensor(self._LOGITS), 0.0).numpy().tolist()
        assert [[v > -1e8 for v in row] for row in kept] == [
            [False, True, False, False, False],
            [False, False, True, False, False],
        ]

    def test_top_p_zero_samples_the_argmax_every_time(self) -> None:
        # It used to keep nothing, which sampled uniformly over the whole vocabulary.
        for seed in range(16):
            lucid.manual_seed(seed)
            assert _next(self._LOGITS, top_p=0.0) == [1, 2]

    def test_top_k_filter_keeps_the_k_largest(self) -> None:
        kept = _top_k_filter(lucid.tensor(self._LOGITS), 2).numpy().tolist()
        assert [[v > -1e8 for v in row] for row in kept] == [
            [False, True, False, False, True],
            [False, False, True, True, False],
        ]


class TestPadIdsAreSeenTokens:
    """A left-padded row penalises its pad id like any other seen token (the reference does too)."""

    _PAD = 0

    def _rows(self) -> list[lucid.Tensor]:
        # Row 0 is left-padded: [PAD, 3, 4]; row 1 has the same tokens with no padding: [2, 3, 4].
        return [
            lucid.tensor([self._PAD, 2]).long(),
            lucid.tensor([3, 3]).long(),
            lucid.tensor([4, 4]).long(),
        ]

    def test_the_pad_logit_is_penalised_on_the_padded_row_only(self) -> None:
        logits = lucid.tensor([[2.0] * 6, [2.0] * 6])
        out = _apply_repetition_penalty(logits, lucid.stack(self._rows(), dim=1), 2.0)
        padded, plain = out.numpy().tolist()
        assert padded == [1.0, 2.0, 2.0, 1.0, 1.0, 2.0]
        assert plain == [2.0, 2.0, 1.0, 1.0, 1.0, 2.0]

    def test_the_pad_id_loses_the_argmax_on_the_padded_row_only(self) -> None:
        logits = lucid.tensor([[2.0, 1.5, 0.0, 0.0, 0.0, 0.0]] * 2)
        out = _next_token(logits, self._rows(), _params(do_sample=False, repetition_penalty=2.0))
        assert out.numpy().tolist() == [1, 0]


def _tiny_gpt2() -> object:
    return create_model(
        "gpt2_small_lm",
        vocab_size=50,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    ).eval()


def _tiny_t5() -> object:
    return create_model(
        "t5_small_seq2seq",
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        num_hidden_layers=2,
        num_decoder_layers=2,
        intermediate_size=64,
    ).eval()


def _tiny_transformer() -> object:
    return create_model(
        "transformer_base_seq2seq",
        vocab_size=50,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_decoder_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
    ).eval()


_PROMPT = [[3, 4, 5]]
_SOURCE = [[3, 7, 1, 9, 2]]

#: family -> (builder, the keywords a call needs, the input rows)
_FAMILIES = [
    pytest.param(_tiny_gpt2, {"max_new_tokens": 3}, _PROMPT, id="gpt2"),
    pytest.param(_tiny_t5, {"max_length": 5}, _SOURCE, id="t5"),
    pytest.param(_tiny_transformer, {"max_length": 5}, _SOURCE, id="transformer"),
]

# (the generate() keywords, a regex on the ValueError)
_BAD_CALLS = [
    pytest.param(
        {"do_sample": True, "temperature": 0.0}, r"`temperature` \(=0\.0\)", id="temperature-zero"
    ),
    pytest.param(
        {"do_sample": True, "temperature": -1.0},
        r"`temperature` \(=-1\.0\)",
        id="temperature-negative",
    ),
    pytest.param(
        {"do_sample": True, "top_k": -1},
        r"`top_k` has to be a strictly positive integer",
        id="top_k-negative",
    ),
    pytest.param(
        {"do_sample": True, "top_p": 1.5}, r"`top_p` has to be a float", id="top_p-above-one"
    ),
    pytest.param(
        {"repetition_penalty": 0.0},
        r"`penalty` has to be a strictly positive float",
        id="penalty-zero",
    ),
    pytest.param(
        {"do_sample": True, "repetition_penalty": -2.0},
        r"`penalty` has to be a strictly positive float",
        id="penalty-negative-sampled",
    ),
]


class TestThroughGenerate:
    @pytest.mark.parametrize(("build", "base", "rows"), _FAMILIES)
    @pytest.mark.parametrize(("bad", "pattern"), _BAD_CALLS)
    def test_a_bad_argument_is_refused(
        self,
        build: object,
        base: dict[str, int],
        rows: list[list[int]],
        bad: dict[str, object],
        pattern: str,
    ) -> None:
        model = build()
        with pytest.raises(ValueError, match=pattern):
            model.generate(lucid.tensor(rows).long(), **base, **bad)

    @pytest.mark.parametrize(("build", "base", "rows"), _FAMILIES)
    def test_top_k_zero_is_the_filter_off_and_top_p_zero_is_greedy(
        self, build: object, base: dict[str, int], rows: list[list[int]]
    ) -> None:
        model = build()
        ids = lucid.tensor(rows).long()
        generate = model.generate
        lucid.manual_seed(0)
        unfiltered = generate(ids, **base, do_sample=True)
        lucid.manual_seed(0)
        assert (
            generate(ids, **base, do_sample=True, top_k=0).numpy().tolist()
            == unfiltered.numpy().tolist()
        )
        lucid.manual_seed(0)
        assert (
            generate(ids, **base, do_sample=True, top_p=1.0).numpy().tolist()
            == unfiltered.numpy().tolist()
        )
        greedy = generate(ids, **base, do_sample=False)
        lucid.manual_seed(0)
        assert (
            generate(ids, **base, do_sample=True, top_p=0.0).numpy().tolist()
            == greedy.numpy().tolist()
        )

    @pytest.mark.parametrize(("build", "base", "rows"), _FAMILIES)
    def test_greedy_decoding_ignores_the_sampling_only_knobs(
        self, build: object, base: dict[str, int], rows: list[list[int]]
    ) -> None:
        model = build()
        ids = lucid.tensor(rows).long()
        expected = model.generate(ids, **base).numpy().tolist()
        out = model.generate(ids, **base, do_sample=False, temperature=0.0, top_k=-1, top_p=-0.5)
        assert out.numpy().tolist() == expected

    @pytest.mark.parametrize("build", [_tiny_t5, _tiny_transformer], ids=["t5", "transformer"])
    @pytest.mark.parametrize("use_cache", [True, False])
    def test_a_seq2seq_loop_refuses_an_unknown_cache_name(
        self, build: object, use_cache: bool
    ) -> None:
        model = build()
        with pytest.raises(ValueError, match=r"Invalid `cache_implementation` \(foo\)"):
            model.generate(
                lucid.tensor(_SOURCE).long(),
                max_length=5,
                use_cache=use_cache,
                cache_implementation="foo",
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "LCD-1108: the causal loop reads cache_implementation in _setup_causal_cache "
            "(_mixins.py), claimed by LCD-1110; once it lands, add "
            "`_check_cache_implementation(cache_implementation)` as that function's first line "
            "and drop this marker"
        ),
    )
    @pytest.mark.parametrize("use_cache", [True, False])
    def test_the_causal_loop_refuses_an_unknown_cache_name(self, use_cache: bool) -> None:
        model = _tiny_gpt2()
        with pytest.raises(ValueError, match=r"Invalid `cache_implementation` \(foo\)"):
            model.generate(
                lucid.tensor(_PROMPT).long(),
                max_new_tokens=3,
                use_cache=use_cache,
                cache_implementation="foo",
            )


@pytest.mark.parity
class TestAgainstTheReference:
    """The messages, the filters and the penalty are the reference's, checked live."""

    @pytest.fixture
    def processors(self, ref: object) -> object:
        return pytest.importorskip(
            "transformers.generation.logits_process", reason="the reference text-generation library"
        )

    @pytest.mark.parametrize(
        ("name", "value"), [("temperature", 0.0), ("temperature", -1.0), ("temperature", _NAN)]
    )
    def test_the_temperature_message(self, processors: object, name: str, value: float) -> None:
        with pytest.raises(ValueError) as theirs:
            processors.TemperatureLogitsWarper(value)
        with pytest.raises(ValueError) as ours:
            _params(**{name: value})
        assert str(ours.value) == str(theirs.value)

    @pytest.mark.parametrize("value", [-1, 2.5])
    def test_the_top_k_message(self, processors: object, value: float) -> None:
        with pytest.raises(ValueError) as theirs:
            processors.TopKLogitsWarper(top_k=value)
        with pytest.raises(ValueError) as ours:
            _params(top_k=value)
        assert str(ours.value) == str(theirs.value)

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_the_top_p_message(self, processors: object, value: float) -> None:
        with pytest.raises(ValueError) as theirs:
            processors.TopPLogitsWarper(top_p=value)
        with pytest.raises(ValueError) as ours:
            _params(top_p=value)
        assert str(ours.value) == str(theirs.value)

    @pytest.mark.parametrize("value", [0.0, -1.0, _NAN])
    def test_the_repetition_penalty_message(self, processors: object, value: float) -> None:
        with pytest.raises(ValueError) as theirs:
            processors.RepetitionPenaltyLogitsProcessor(penalty=value)
        with pytest.raises(ValueError) as ours:
            _params(repetition_penalty=value)
        assert str(ours.value) == str(theirs.value)

    def test_the_cache_message_is_theirs_with_our_choices(self, ref: object) -> None:
        config = pytest.importorskip("transformers.generation.configuration_utils")
        with pytest.raises(ValueError) as theirs:
            config.GenerationConfig(cache_implementation="foo").validate()
        with pytest.raises(ValueError) as ours:
            _check_cache_implementation("foo")
        stem = "Invalid `cache_implementation` (foo). Choose one of: "
        assert str(theirs.value).startswith(stem)
        assert str(ours.value).startswith(stem)

    @pytest.mark.parametrize("penalty", [0.5, 1.3, 2.0])
    def test_the_penalty_matches_the_processor_pads_included(
        self, ref: object, processors: object, penalty: float
    ) -> None:
        lucid.manual_seed(0)
        logits = lucid.randn(3, 12)
        logits[0, 5] = 0.0  # a logit of exactly 0 is the sign edge both sides must agree on
        # Row 0 is left-padded with id 0, row 1 has a pad run, row 2 is unpadded.
        prefix = [[0, 0, 4, 7, 4], [0, 9, 9, 2, 5], [3, 1, 4, 1, 5]]
        ours = _apply_repetition_penalty(logits, lucid.tensor(prefix).long(), penalty)
        theirs = processors.RepetitionPenaltyLogitsProcessor(penalty=penalty)(
            ref.tensor(prefix, dtype=ref.long),
            ref.tensor(logits.numpy().tolist(), dtype=ref.float32),
        )
        for row_ours, row_theirs in zip(ours.numpy().tolist(), theirs.tolist(), strict=True):
            assert row_ours == pytest.approx(row_theirs, rel=1e-6, abs=1e-7)

    @pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.9])
    def test_top_p_keeps_the_tokens_the_reference_keeps(
        self, ref: object, processors: object, p: float
    ) -> None:
        lucid.manual_seed(1)
        logits = lucid.randn(4, 30) * 2.0
        ours = [[v > -1e8 for v in row] for row in _top_p_filter(logits, p).numpy().tolist()]
        scores = ref.tensor(logits.numpy().tolist(), dtype=ref.float32)
        kept = processors.TopPLogitsWarper(top_p=p)(None, scores)
        assert ours == [[math.isfinite(v) for v in row] for row in kept.tolist()]

    @pytest.mark.parametrize("k", [1, 3, 29])
    def test_top_k_keeps_the_tokens_the_reference_keeps(
        self, ref: object, processors: object, k: int
    ) -> None:
        lucid.manual_seed(2)
        logits = lucid.randn(4, 30)
        ours = [[v > -1e8 for v in row] for row in _top_k_filter(logits, k).numpy().tolist()]
        scores = ref.tensor(logits.numpy().tolist(), dtype=ref.float32)
        kept = processors.TopKLogitsWarper(top_k=k)(None, scores)
        assert ours == [[math.isfinite(v) for v in row] for row in kept.tolist()]
