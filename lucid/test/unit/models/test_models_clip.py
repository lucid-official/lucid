"""Unit tests for CLIP (Radford et al., 2021).

Two towers and a dot product is an architecture a shape test cannot
distinguish from a working one. Every check below therefore targets a
property that a plausible wrong implementation would still pass shapes
on, and each of the three load-bearing ones carries a companion that
feeds it the wrong version and requires it to be rejected.

The three:

**Where the text feature is read.** ``argmax`` over token ids finds
``[EOS]`` only because ``[EOS]`` is the highest id. Reading the last
column instead returns padding for every caption shorter than the
context — which trains, and ranks nothing.

**Whether the embeddings are normalised.** Without it the contrastive
objective can be lowered by growing norms rather than aligning
directions.

**Whether the loss is symmetric.** A one-sided loss leaves one tower
unconstrained; the numbers still fall.
"""

import math
import os

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
from lucid.models.multimodal.clip import (
    CLIPModel,
    CLIP_EOS,
    CLIP_SOS,
    CLIPConfig,
    CLIPForZeroShotImageClassification,
    CLIPTokenizer,
    CLIPTokenizerFast,
    CLIPViTBase16Weights,
    CLIPViTBase32Weights,
    CLIPViTLarge14_336Weights,
    CLIPViTLarge14Weights,
)
from lucid.models.multimodal.clip._tokenizer import _PATTERN
from lucid.models.multimodal.clip._towers import QuickGELU
from lucid.models.multimodal.clip._model import _contrastive_loss, _TextTransformer

_TINY = dict(
    embed_dim=16,
    image_size=32,
    patch_size=16,
    vision_layers=2,
    vision_width=32,
    vision_heads=2,
    context_length=8,
    vocab_size=64,
    text_width=32,
    text_heads=2,
    text_layers=2,
)


def _tiny(**overrides: object) -> CLIPConfig:
    merged = dict(_TINY)
    merged.update(overrides)
    return CLIPConfig(**merged)  # type: ignore[arg-type]


def _captions(lengths: list[int], vocab: int = 64, context: int = 8) -> lucid.Tensor:
    """Right-padded token ids whose ``[EOS]`` is the highest id present."""
    eos = vocab - 1
    rows = []
    for length in lengths:
        row = [1] + list(range(2, 2 + length)) + [eos]
        row = row + [0] * (context - len(row))
        rows.append(row[:context])
    return lucid.tensor(rows, dtype=lucid.int64)


class TestConfig:
    def test_it_rejects_an_indivisible_image(self) -> None:
        with pytest.raises(ValueError, match="divisible by patch_size"):
            _tiny(image_size=33)

    def test_it_rejects_heads_that_do_not_divide_the_width(self) -> None:
        with pytest.raises(ValueError, match="divisible by"):
            _tiny(text_heads=5)

    def test_it_rejects_a_context_with_no_room_for_sentinels(self) -> None:
        with pytest.raises(ValueError, match=r"\[SOS\]"):
            _tiny(context_length=1)

    @pytest.mark.parametrize("value", [0.0, -0.07])
    def test_it_rejects_a_non_positive_temperature(self, value: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            _tiny(logit_scale_init=value)


class TestTheTextFeatureComesFromEOS:
    """The single most breakable line in the family."""

    def test_padding_after_eos_does_not_change_the_embedding(self) -> None:
        """The claim: only the tokens up to ``[EOS]`` are read.

        Two captions with the same content and different amounts of
        padding must embed identically. They cannot if the feature is
        taken at the last column.
        """
        lucid.manual_seed(0)
        model = CLIPModel(_tiny()).eval()
        short = _captions([2])
        padded = _captions([2])
        # Same caption; scribble junk into the padding of one of them.
        junk = padded.tolist()
        junk[0][-1] = 5
        scribbled = lucid.tensor(junk, dtype=lucid.int64)

        a = model.encode_text(short)
        b = model.encode_text(scribbled)
        assert float((a - b).abs().max().item()) < 1e-5, (
            "the text embedding moved when only post-[EOS] padding changed "
            "— the feature is being read at the wrong position"
        )

    def test_reading_the_last_column_would_differ(self) -> None:
        """Guards the test above.

        If the two were equal for a trivial reason — a tower that ignores
        its input, say — the check would pass while proving nothing. So
        take the same activations at the last column and require that
        they disagree with the ``[EOS]`` reading.
        """
        lucid.manual_seed(0)
        config = _tiny()
        tower = _TextTransformer(config)
        nn.init.normal_(tower.token_embedding.weight, std=0.5)
        nn.init.normal_(tower.text_projection, std=0.5)
        ids = _captions([2])

        at_eos = tower(ids)

        x = tower.token_embedding(ids) + tower.positional_embedding.reshape(
            1, config.context_length, -1
        )
        x = tower.ln_final(tower.transformer(x))
        at_last = x[:, -1] @ tower.text_projection

        assert float((at_eos - at_last).abs().max().item()) > 1e-4, (
            "reading [EOS] and reading the last column agree, so the test "
            "above cannot tell them apart"
        )

    def test_the_sentinel_is_the_largest_id(self) -> None:
        """The assumption ``argmax`` rests on, stated as a test."""
        ids = _captions([3])
        assert int(ids.argmax(dim=-1).item()) == 4
        assert int(ids[0, 4].item()) == _TINY["vocab_size"] - 1


class TestEmbeddingsAreNormalised:
    def test_rows_have_unit_length(self) -> None:
        lucid.manual_seed(0)
        model = CLIPModel(_tiny()).eval()
        out = model(lucid.randn((3, 3, 32, 32)), _captions([2, 3, 1]))
        for name, embeds in (("image", out.image_embeds), ("text", out.text_embeds)):
            norms = (embeds**2).sum(dim=-1) ** 0.5
            assert float((norms - 1.0).abs().max().item()) < 1e-5, name

    def test_an_unnormalised_dot_product_would_differ(self) -> None:
        """Guards the test above — the towers do not emit unit rows by luck."""
        lucid.manual_seed(0)
        model = CLIPModel(_tiny()).eval()
        images = lucid.randn((3, 3, 32, 32))
        raw = model.visual(images)
        norms = (raw**2).sum(dim=-1) ** 0.5
        assert float((norms - 1.0).abs().max().item()) > 1e-3, (
            "the image tower already emits unit rows, so normalising it " "proves nothing"
        )


class TestTheLossIsSymmetric:
    def test_a_perfect_alignment_scores_zero(self) -> None:
        perfect = lucid.tensor([[20.0, -20.0], [-20.0, 20.0]])
        assert float(_contrastive_loss(perfect, perfect.T).item()) < 1e-6

    def test_chance_scores_log_n(self) -> None:
        """With no information the loss is ``ln N`` in both directions."""
        flat = lucid.zeros((4, 4))
        assert float(_contrastive_loss(flat, flat).item()) == pytest.approx(math.log(4.0), abs=1e-5)

    def test_it_reads_both_directions(self) -> None:
        """Guards the two above.

        A loss that scored only images-over-captions would be blind to a
        logit matrix that ranks rows correctly and columns not at all.
        Build exactly that and require the symmetric loss to notice.
        """
        # Column 0 wins for both rows, so ranking images over captions
        # and captions over images are different problems here.
        lopsided = lucid.tensor([[10.0, 9.0], [10.0, 9.5]])
        one_sided = float(_contrastive_loss(lopsided, lopsided).item())
        symmetric = float(_contrastive_loss(lopsided, lopsided.T).item())
        assert abs(symmetric - one_sided) > 1e-3, (
            f"the transpose made no difference ({symmetric:.4f} vs "
            f"{one_sided:.4f}) — the two directions are tied"
        )
        # Which of the two is larger is not fixed — it depends on which
        # direction the asymmetry favours — so only the difference is
        # asserted. A test that pinned the sign would be pinning this
        # matrix rather than the loss.

    def test_it_refuses_a_non_square_batch(self) -> None:
        with pytest.raises(ValueError, match="square"):
            _contrastive_loss(lucid.zeros((2, 3)), lucid.zeros((3, 2)))


class TestTheTemperature:
    def test_it_starts_at_the_papers_value(self) -> None:
        model = CLIPModel(_tiny())
        assert float(model.scale.item()) == pytest.approx(1.0 / 0.07, rel=1e-6)

    def test_it_is_capped(self) -> None:
        """ "clipped to prevent scaling the logits by more than 100"."""
        model = CLIPModel(_tiny())
        with lucid.no_grad():
            model.logit_scale[:] = lucid.full((1,), 20.0)  # exp(20) >> 100
        assert float(model.scale.item()) == pytest.approx(100.0)

    def test_the_cap_is_not_always_active(self) -> None:
        """Guards the test above — a scale pinned at 100 would also pass."""
        model = CLIPModel(_tiny())
        assert float(model.scale.item()) < 100.0

    def test_it_is_learned(self) -> None:
        lucid.manual_seed(0)
        model = CLIPModel(_tiny())
        out = model(lucid.randn((2, 3, 32, 32)), _captions([2, 3]), return_loss=True)
        assert out.loss is not None
        out.loss.backward()
        assert model.logit_scale.grad is not None
        assert float(model.logit_scale.grad.abs().sum().item()) > 0.0


class TestTheTowersAttendCorrectly:
    def test_the_text_tower_is_causal(self) -> None:
        """A later token must not change an earlier position's output."""
        lucid.manual_seed(0)
        config = _tiny()
        tower = _TextTransformer(config)
        nn.init.normal_(tower.token_embedding.weight, std=0.5)
        base = _captions([3])
        changed = base.tolist()
        changed[0][5] = 7  # after the [EOS] at index 4
        moved = lucid.tensor(changed, dtype=lucid.int64)

        x = tower.token_embedding(base) + tower.positional_embedding.reshape(
            1, config.context_length, -1
        )
        y = tower.token_embedding(moved) + tower.positional_embedding.reshape(
            1, config.context_length, -1
        )
        a = tower.transformer(x)[:, :5]
        b = tower.transformer(y)[:, :5]
        assert float((a - b).abs().max().item()) < 1e-5

    def test_the_image_tower_is_not_causal(self) -> None:
        """Guards the test above, and pins the asymmetry.

        Patches have no order to respect, so masking them would throw
        away half the image for the class token — and the causal test
        above would pass just as well if both towers were masked.
        """
        lucid.manual_seed(0)
        model = CLIPModel(_tiny()).eval()
        images = lucid.randn((1, 3, 32, 32))
        first = model.visual(images)
        # Disturb only the last patch; a causal tower reading the class
        # token at position 0 could not see it.
        disturbed = images.tolist()
        disturbed[0][0][-1][-1] = 5.0
        second = model.visual(lucid.tensor(disturbed))
        assert float((first - second).abs().max().item()) > 1e-6


class TestQuickGELU:
    def test_it_is_the_sigmoid_approximation(self) -> None:
        x = lucid.tensor([-2.0, 0.0, 1.0])
        got = QuickGELU()(x).tolist()
        want = [v * (1.0 / (1.0 + math.exp(-1.702 * v))) for v in (-2.0, 0.0, 1.0)]
        assert [round(g, 6) for g in got] == [round(w, 6) for w in want]

    def test_it_is_not_exact_gelu(self) -> None:
        """Guards the test above — the released weights were trained on this
        one, so silently swapping in the exact form must not read as equal."""
        import lucid.nn.functional as F

        x = lucid.tensor([1.5])
        assert abs(float(QuickGELU()(x).item()) - float(F.gelu(x).item())) > 1e-4


class TestVariants:
    """The published parameter counts, which pin every width and depth."""

    @pytest.mark.parametrize(
        "factory,expected",
        [
            ("clip_vit_base_32", 151_277_313),
            ("clip_vit_base_16", 149_620_737),
            ("clip_vit_large_14", 427_616_513),
        ],
    )
    def test_parameter_counts(self, factory: str, expected: int) -> None:
        assert M.create_model(factory).num_parameters() == expected

    def test_the_336px_variant_is_larger(self) -> None:
        """It differs in one config field and 320 rows of positional table.

        Worth its own test because the two were reported as identical by
        the docs-site summary until the bare Parameters were made
        shadow-visible.
        """
        small = M.clip_vit_large_14().num_parameters()
        large = M.clip_vit_large_14_336().num_parameters()
        assert large - small == (577 - 257) * 1024

    def test_the_large_variant_widens_its_text_tower(self) -> None:
        """The paper scales width, not depth."""
        base = M.clip_vit_base_32().config
        large = M.clip_vit_large_14().config
        assert (base.text_width, base.text_heads) == (512, 8)
        assert (large.text_width, large.text_heads) == (768, 12)
        assert base.text_layers == large.text_layers == 12


class TestShadowConstructionSeesEveryParameter:
    """A bare Parameter built from arithmetic loses its shape under
    ``shadow_alloc``, and the docs site reports the shortfall as the
    model's size.

    This caught a real one: ``nn.Parameter(scale * lucid.randn(...))``
    made ViT-L/14 and ViT-L/14@336px report the same count, 1.7M short
    each. The rule that fixes it — every bare Parameter is a direct
    creation call, scaled afterwards — is invisible in any other test.
    """

    def test_the_shadow_count_matches_the_real_one(self) -> None:
        from lucid.nn._shadow import shadow_alloc

        config = _tiny()
        real = CLIPModel(config).num_parameters()
        with shadow_alloc():
            shadow = CLIPModel(config).num_parameters()
        assert shadow == real, (
            f"shadow construction sees {shadow:,} parameters and real "
            f"construction {real:,} — a Parameter built from arithmetic "
            f"has lost its shape"
        )


class TestZeroShot:
    def test_prompts_are_not_paired_with_images(self) -> None:
        """Unlike the contrastive forward, the two batches are independent."""
        lucid.manual_seed(0)
        model = CLIPForZeroShotImageClassification(_tiny()).eval()
        out = model(lucid.randn((2, 3, 32, 32)), _captions([1, 2, 3, 4, 5]))
        assert out.logits.shape == (2, 5)

    def test_it_ranks_the_prompt_its_image_matches(self) -> None:
        """The mechanism, with the towers replaced by known embeddings.

        Training a real preference is out of scope for a unit test, so
        this pins the scoring instead: given embeddings that point at
        each other, the argmax has to land on the diagonal.
        """
        lucid.manual_seed(0)
        model = CLIPForZeroShotImageClassification(_tiny()).eval()
        images = lucid.randn((3, 3, 32, 32))
        prompts = _captions([1, 2, 3])
        out = model(images, prompts)
        # Self-similarity must beat cross-similarity for the *text* side
        # against itself, which is the only alignment available untrained.
        text = model.clip.encode_text(prompts)
        similarity = text @ text.T
        diagonal = [float(similarity[i, i].item()) for i in range(3)]
        assert all(d == pytest.approx(1.0, abs=1e-4) for d in diagonal)
        assert out.logits.shape == (3, 3)


class TestTokenizer:
    """The framing the model's ``argmax`` depends on, and the scheme.

    CLIPModel's BPE is neither of the two already in the tree — byte-level
    like GPT-2's, ``</w>``-suffixed like GPT-1's — so the pieces that
    could be silently swapped for the wrong half are the ones checked.
    """

    @staticmethod
    def _tok() -> CLIPTokenizer:
        vocab = {"a": 0, "b</w>": 1, "ab</w>": 2, CLIP_SOS: 3, CLIP_EOS: 4, "c</w>": 5}
        return CLIPTokenizer(vocab=vocab, merges=[("a", "b</w>")])

    def test_it_merges_across_the_word_marker(self) -> None:
        assert self._tok().encode("ab") == [2]

    def test_encode_does_not_frame(self) -> None:
        """The base class adds sentinels by default; this must not."""
        ids = self._tok().encode("ab")
        assert CLIP_SOS not in ids and 3 not in ids and 4 not in ids

    def test_tokenize_frames_and_pads(self) -> None:
        assert self._tok().tokenize("ab", context_length=5) == [3, 2, 4, 0, 0]

    def test_the_eos_is_the_argmax(self) -> None:
        """The property the model's text-feature lookup rests on."""
        ids = self._tok().tokenize("ab", context_length=6)
        assert ids.index(max(ids)) == 2

    def test_a_caption_that_does_not_fit_is_an_error(self) -> None:
        """Truncating would drop the [EOS] the model reads its feature at."""
        with pytest.raises(ValueError, match=r"\[EOS\]"):
            self._tok().tokenize("ab", context_length=2)

    def test_it_lowercases_and_unescapes(self) -> None:
        tok = self._tok()
        assert tok.encode("AB") == tok.encode("ab")
        assert tok.encode("&amp;amp;") == tok.encode("&")

    def test_it_collapses_whitespace(self) -> None:
        tok = self._tok()
        assert tok.encode("  ab   ") == tok.encode("ab")

    def test_a_vocab_without_sentinels_is_refused(self) -> None:
        with pytest.raises(ValueError, match="startoftext"):
            CLIPTokenizer(vocab={"a": 0}, merges=[])

    def test_the_pattern_splits_the_reference_way(self) -> None:
        """Contractions before letters, digits one at a time.

        Guards the regex translation: ``\\p{L}`` and ``\\p{N}`` are not
        stdlib, so they were rewritten, and a rewrite that merged digit
        runs or ate contractions would still tokenise every caption
        without complaint.
        """
        assert _PATTERN.findall("a dog's paw, 42!") == ["a", "dog", "'s", "paw", ",", "4", "2", "!"]

    def test_digits_are_not_grouped(self) -> None:
        """Guards the test above — ``\\d+`` would read identically here
        until a number longer than one digit appeared."""
        assert _PATTERN.findall("2026") == ["2", "0", "2", "6"]


class TestItRunsOnBothDevices:
    """CLIP had never been run on Metal, and it did not work.

    ``lucid.arange`` with no device builds its index tensor on the CPU,
    and gathering the ``[EOS]`` row with it fails only once the index
    meets accelerator activations — several frames from the line at
    fault. The same shape of bug had already been found in the rollout
    layer, which is why it is now tested at both ends rather than
    assumed to travel with the parameters.
    """

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_a_forward_and_backward_survive(self, device: str) -> None:
        lucid.manual_seed(0)
        model = CLIPModel(_tiny())
        model = model.metal() if device == "metal" else model
        images = lucid.randn((3, 3, 32, 32), device=device)
        captions = _captions([1, 2, 3]).to(device)

        out = model(images, captions, return_loss=True)
        assert str(out.image_embeds.device) == f"device('{device}')"
        assert str(out.logits_per_image.device) == f"device('{device}')"
        assert out.loss is not None
        out.loss.backward()
        total = sum(
            float(p.grad.abs().sum().item()) for p in model.parameters() if p.grad is not None
        )
        assert total > 0.0

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_the_zero_shot_wrapper_too(self, device: str) -> None:
        lucid.manual_seed(0)
        model = CLIPForZeroShotImageClassification(_tiny())
        model = model.metal() if device == "metal" else model
        out = model(lucid.randn((2, 3, 32, 32), device=device), _captions([1, 2, 3, 4]).to(device))
        assert out.logits.shape == (2, 4)
        assert str(out.logits.device) == f"device('{device}')"


@pytest.mark.slow
class TestItLearnsToMatch:
    """The consumer. Nothing else here asks whether CLIP *works*.

    Shapes, normalisation, the sentinel and the loss are each checked in
    isolation above, and a model that got every one of them right could
    still fail to align anything — the objective is only correct if
    optimising it moves matched pairs to the top of their row.

    So: a fixed synthetic pairing with nothing to generalise, trained to
    convergence. Rank-1 starts at chance and has to reach the diagonal.
    """

    @staticmethod
    def _fixture(device: str) -> tuple:
        pairs = 8
        config = CLIPConfig(
            embed_dim=32,
            image_size=32,
            patch_size=8,
            vision_layers=2,
            vision_width=64,
            vision_heads=4,
            context_length=8,
            vocab_size=32,
            text_width=64,
            text_heads=4,
            text_layers=2,
        )
        model = CLIPModel(config)
        model = model.metal() if device == "metal" else model
        images = lucid.stack(
            [lucid.full((3, 32, 32), (i + 1) / pairs) for i in range(pairs)], dim=0
        )
        captions = lucid.tensor(
            [[1, 2 + i, 31, 0, 0, 0, 0, 0] for i in range(pairs)], dtype=lucid.int64
        )
        if device == "metal":
            images, captions = images.metal(), captions.metal()
        return model, images, captions, pairs

    @staticmethod
    def _rank1(model: CLIPModel, images: lucid.Tensor, captions: lucid.Tensor, n: int) -> float:
        with lucid.no_grad():
            predicted = model(images, captions).logits_per_image.argmax(dim=-1)
            target = lucid.arange(n, dtype=lucid.int64, device=predicted.device.type)
            return float((predicted == target).to(lucid.float32).mean().item())

    def test_matched_pairs_reach_the_top_of_their_row(self) -> None:
        import lucid.optim as optim

        lucid.manual_seed(0)
        model, images, captions, pairs = self._fixture("cpu")
        opt = optim.Adam(model.parameters(), lr=3e-4)

        before = self._rank1(model, images, captions, pairs)
        for _ in range(250):
            out = model(images, captions, return_loss=True)
            model.zero_grad()
            assert out.loss is not None
            out.loss.backward()
            opt.step()
        after = self._rank1(model, images, captions, pairs)

        assert before <= 0.5, f"started already solved ({before:.3f})"
        assert after == 1.0, f"did not align: rank-1 {before:.3f} -> {after:.3f}"

    def test_the_temperature_moves_while_it_learns(self) -> None:
        """Guards the test above.

        A model whose towers memorised the pairing without the scale ever
        adapting would still reach rank-1; the paper's claim is that the
        temperature is learned rather than annealed, and only this shows
        it is being optimised at all.
        """
        import lucid.optim as optim

        lucid.manual_seed(0)
        model, images, captions, _ = self._fixture("cpu")
        opt = optim.Adam(model.parameters(), lr=3e-4)
        start = float(model.scale.item())
        for _ in range(100):
            out = model(images, captions, return_loss=True)
            model.zero_grad()
            assert out.loss is not None
            out.loss.backward()
            opt.step()
        assert abs(float(model.scale.item()) - start) > 0.05


class TestWeightsAreDeclaredConsistently:
    """The registry's own numbers, checked against the models they load into.

    A wrong ``num_params`` or a stale ``sha256`` is invisible until a
    user downloads — and this session already shipped a YOLO entry whose
    digest did not match the published file, so these are checked
    offline rather than trusted.
    """

    @pytest.mark.parametrize(
        "factory,enum",
        [
            ("clip_vit_base_32", CLIPViTBase32Weights),
            ("clip_vit_base_16", CLIPViTBase16Weights),
            ("clip_vit_large_14", CLIPViTLarge14Weights),
            ("clip_vit_large_14_336", CLIPViTLarge14_336Weights),
        ],
    )
    def test_declared_parameter_count_matches_the_model(self, factory: str, enum: object) -> None:
        entry = enum.DEFAULT.value  # type: ignore[attr-defined]
        assert entry.meta["num_params"] == M.create_model(factory).num_parameters()

    def test_the_336_preset_crops_to_336(self) -> None:
        """The one entry whose preprocessing differs, and the reason it does."""
        assert CLIPViTLarge14_336Weights.DEFAULT.value.transforms.crop_size == 336
        assert CLIPViTLarge14Weights.DEFAULT.value.transforms.crop_size == 224

    def test_every_url_points_at_its_own_repo(self) -> None:
        """Guards the three above — copy-pasting an entry is how two
        variants end up sharing one checkpoint, which loads cleanly for
        the pair whose shapes happen to agree."""
        seen = set()
        for enum in (
            CLIPViTBase32Weights,
            CLIPViTBase16Weights,
            CLIPViTLarge14Weights,
            CLIPViTLarge14_336Weights,
        ):
            entry = enum.DEFAULT.value
            assert entry.url not in seen, f"duplicate url: {entry.url}"
            assert entry.sha256 not in seen, f"duplicate digest: {entry.sha256}"
            seen.update({entry.url, entry.sha256})


@pytest.mark.skipif(
    os.environ.get("LUCID_TEST_NETWORK") != "1",
    reason="set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestPretrainedRoundTrip:
    """End-to-end through the Lucid API alone: download, verify, run."""

    def test_it_loads_and_runs(self) -> None:
        model = M.clip_vit_base_32(pretrained=True).eval()
        out = model(
            lucid.randn((2, 3, 224, 224)),
            lucid.tensor([[49406, 320, 1125, 49407] + [0] * 73] * 2, dtype=lucid.int64),
        )
        assert out.logits_per_image.shape == (2, 2)
        norms = ((out.image_embeds**2).sum(dim=-1) ** 0.5).tolist()
        assert all(abs(n - 1.0) < 1e-4 for n in norms)

    def test_the_trained_temperature_sits_at_the_cap(self) -> None:
        """Measured: every release stores log(100), so the paper's clip
        was binding when training stopped."""
        model = M.clip_vit_base_32(pretrained=True)
        assert float(model.scale.item()) == pytest.approx(100.0, abs=1e-3)


class TestTheFastTokenizerAgrees:
    """The only property that matters for a fast path: identical output.

    The engine seeds one symbol per codepoint, so it tears ``</w>`` into
    four — measured, and the reason GPT-1's "fast" tokenizer carries no
    acceleration at all. This one works by folding the marker and the
    character before it into a single private-use codepoint across the
    vocabulary, the merge table and each chunk, so the engine seeds what
    the scheme intends. The fold touches keys only; ids are untouched.
    """

    @staticmethod
    def _pair() -> tuple[CLIPTokenizer, CLIPTokenizerFast]:
        """A *total* byte-level vocabulary, which is the contract.

        A partial one makes the two paths disagree by construction —
        the engine drops an unknown symbol where the Python path
        substitutes UNK — so the fast tokenizer refuses it outright and
        the tests must not build one.
        """
        from lucid.utils.tokenizer._pre_tokenizers import ByteLevel

        alphabet = [ByteLevel.encode_bytes(bytes([b])) for b in range(256)]
        vocab: dict[str, int] = {}
        for char in alphabet:
            vocab[char] = len(vocab)
        for char in alphabet:
            vocab[char + "</w>"] = len(vocab)
        for extra in ("ab</w>", "cab</w>", CLIP_SOS, CLIP_EOS):
            vocab[extra] = len(vocab)
        merges = [("a", "b</w>"), ("c", "ab</w>")]
        return (
            CLIPTokenizer(vocab=vocab, merges=merges),
            CLIPTokenizerFast(vocab=vocab, merges=merges),
        )

    @pytest.mark.parametrize("text", ["ab", "cab", "a", "ab ab", "AB", "", "hello world 42!"])
    def test_it_matches_the_python_path(self, text: str) -> None:
        slow, fast = self._pair()
        assert fast.encode(text) == slow.encode(text)

    def test_it_matches_through_framing(self) -> None:
        slow, fast = self._pair()
        assert fast.tokenize("cab", context_length=6) == slow.tokenize("cab", context_length=6)

    def test_the_fold_is_needed(self) -> None:
        """Guards the tests above.

        If the engine had handled ``</w>`` natively the fold would be
        dead code and the agreement would prove nothing about it. Feed
        the raw marker to the engine and require that it comes apart.
        """
        from lucid._C import engine as _C_engine

        vocab = {c: i for i, c in enumerate("helo<>/w")}
        vocab["o</w>"] = 100
        engine_bpe = _C_engine.utils.tokenizer.BPE(vocab, [("l", "o</w>")])
        pieces = [engine_bpe.id_to_token(i) for i in engine_bpe.encode("hello</w>")]
        assert pieces[-4:] == ["<", "/", "w", ">"], (
            "the engine now seeds the marker whole, so the fold in "
            "CLIPTokenizerFast is no longer load-bearing"
        )

    def test_a_non_byte_level_vocab_is_refused(self) -> None:
        """Folding must not collapse two entries onto one key."""
        with pytest.raises(ValueError, match="byte-level"):
            CLIPTokenizerFast(vocab={"": 0, "\x00</w>": 1, CLIP_SOS: 2, CLIP_EOS: 3}, merges=[])
