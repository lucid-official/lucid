"""Unit tests for the vector-quantisation stack.

Covers all three layers: ``F.straight_through`` and ``F.nearest_codebook``
(primitives), ``F.vector_quantize`` (the functional composite), and
``nn.VectorQuantizer`` (the module that adds the codebook parameter and
the two training terms).

The layer's whole reason to exist is gradient routing across a hard
``argmin``: the producer of the input must train as if quantisation were
the identity, while the codebook must train only from its own term.  Both
properties are invisible in a shape check and silent when broken.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# straight_through
# ─────────────────────────────────────────────────────────────────────────────


class TestStraightThrough:
    def test_forward_value_is_the_hard_argument(self) -> None:
        soft = lucid.tensor([0.3, 0.7])
        hard = lucid.tensor([0.0, 1.0])
        assert F.straight_through(hard, soft).tolist() == [0.0, 1.0]

    def test_gradient_passes_to_the_soft_argument_untouched(self) -> None:
        soft = lucid.tensor([0.3, 0.7], requires_grad=True)
        hard = lucid.tensor([0.0, 1.0])
        F.straight_through(hard, soft).sum().backward()

        assert soft.grad is not None
        assert soft.grad.tolist() == [1.0, 1.0]

    def test_hard_argument_receives_no_gradient(self) -> None:
        soft = lucid.tensor([0.3, 0.7], requires_grad=True)
        hard = lucid.tensor([0.0, 1.0], requires_grad=True)
        F.straight_through(hard, soft).sum().backward()

        assert hard.grad is None

    def test_gumbel_softmax_hard_still_routes_gradients(self) -> None:
        # gumbel_softmax(hard=True) is the other caller; it must keep
        # behaving after being rewritten on top of this helper.
        logits = lucid.randn((4, 5), requires_grad=True)
        out = F.gumbel_softmax(logits, tau=0.5, hard=True)

        assert out.shape == (4, 5)
        # Exactly one 1.0 per row.
        assert [float(out[i].sum().item()) for i in range(4)] == [1.0] * 4
        out.sum().backward()
        assert logits.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# nearest_codebook / vector_quantize
# ─────────────────────────────────────────────────────────────────────────────


class TestFunctionalQuantize:
    def test_nearest_codebook_picks_the_closest_entry(self) -> None:
        codebook = lucid.tensor([[0.0, 0.0], [1.0, 1.0]])
        x = lucid.tensor([[0.9, 1.1], [0.1, 0.0]])
        assert F.nearest_codebook(x, codebook).tolist() == [1, 0]

    @pytest.mark.parametrize("shape", [(4, 3), (2, 4, 3), (2, 2, 4, 3)])
    def test_nearest_codebook_drops_only_the_feature_axis(self, shape: tuple[int, ...]) -> None:
        codebook = lucid.randn((8, 3))
        assert F.nearest_codebook(lucid.randn(shape), codebook).shape == shape[:-1]

    def test_nearest_codebook_rejects_a_non_2d_codebook(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            F.nearest_codebook(lucid.randn((4, 3)), lucid.randn((2, 8, 3)))

    def test_nearest_codebook_rejects_a_dim_mismatch(self) -> None:
        with pytest.raises(ValueError, match="trailing axis"):
            F.nearest_codebook(lucid.randn((4, 5)), lucid.randn((8, 3)))

    def test_vector_quantize_returns_the_selected_entries(self) -> None:
        codebook = lucid.tensor([[0.0, 0.0], [1.0, 1.0]])
        quantized, indices = F.vector_quantize(lucid.tensor([[0.9, 1.1]]), codebook)

        assert quantized.tolist() == [[1.0, 1.0]]
        assert indices.tolist() == [1]

    def test_vector_quantize_routes_gradient_to_the_input(self) -> None:
        codebook = lucid.randn((8, 3))
        x = lucid.randn((4, 3), requires_grad=True)
        F.vector_quantize(x, codebook)[0].sum().backward()

        assert x.grad is not None
        assert float(abs(x.grad).sum()) > 0.0

    def test_vector_quantize_agrees_with_the_module(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        x = lucid.randn((5, 4))
        quantized, indices = F.vector_quantize(x, vq.weight)
        out = vq(x)

        assert indices.tolist() == out.indices.tolist()
        assert float((quantized - out.quantized).abs().max().item()) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# VectorQuantizer
# ─────────────────────────────────────────────────────────────────────────────


class TestVectorQuantizerConstruction:
    def test_codebook_shape_and_init_range(self) -> None:
        vq = nn.VectorQuantizer(num_embeddings=64, embedding_dim=8)
        assert vq.weight.shape == (64, 8)
        # Uniform(-1/K, 1/K).
        assert float(vq.weight.abs().max().item()) <= 1.0 / 64.0

    def test_defaults_to_the_paper_commitment_cost(self) -> None:
        assert nn.VectorQuantizer(8, 4).commitment_cost == 0.25

    def test_repr(self) -> None:
        assert "commitment_cost=0.25" in repr(nn.VectorQuantizer(8, 4))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_embeddings": 0, "embedding_dim": 4}, "num_embeddings"),
            ({"num_embeddings": 4, "embedding_dim": 0}, "embedding_dim"),
            (
                {"num_embeddings": 4, "embedding_dim": 4, "commitment_cost": -1.0},
                "commitment_cost",
            ),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            nn.VectorQuantizer(**kwargs)  # type: ignore[arg-type]

    def test_codebook_is_a_registered_parameter(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        assert [n for n, _ in vq.named_parameters()] == ["weight"]


class TestVectorQuantizerForward:
    @pytest.mark.parametrize("shape", [(7, 8), (2, 5, 8), (2, 3, 4, 8)])
    def test_acts_on_the_trailing_axis(self, shape: tuple[int, ...]) -> None:
        vq = nn.VectorQuantizer(64, 8)
        out = vq(lucid.randn(shape))

        assert out.quantized.shape == shape
        assert out.indices.shape == shape[:-1]

    def test_rejects_a_mismatched_trailing_axis(self) -> None:
        vq = nn.VectorQuantizer(64, 8)
        with pytest.raises(ValueError, match="trailing axis"):
            vq(lucid.randn((2, 5)))

    def test_output_unpacks_as_a_tuple(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        quantized, indices, codebook_loss, commitment_loss, perplexity = vq(lucid.randn((3, 4)))
        assert quantized.shape == (3, 4)
        assert indices.shape == (3,)
        for term in (codebook_loss, commitment_loss, perplexity):
            assert term.shape == ()

    def test_indices_are_in_range(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        out = vq(lucid.randn((20, 4)))

        assert out.indices.dtype == lucid.int64
        assert int(out.indices.min().item()) >= 0
        assert int(out.indices.max().item()) < 16

    def test_quantized_equals_the_selected_entries(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        out = vq(lucid.randn((6, 4)))
        assert float((out.quantized - vq.lookup(out.indices)).abs().max().item()) < 1e-6

    def test_codebook_entries_select_themselves(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        picked = vq.assign(vq.weight)
        assert [int(picked[i].item()) for i in range(16)] == list(range(16))

    def test_perplexity_bounds(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        perplexity = float(vq(lucid.randn((64, 4))).perplexity.item())
        assert 1.0 <= perplexity <= 16.0 + 1e-4

    def test_perplexity_is_one_when_every_position_picks_one_entry(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        # Feed the same vector everywhere: one live entry, perplexity 1.
        row = vq.weight[3].reshape(1, 4)
        repeated = lucid.cat([row] * 32, dim=0)
        out = vq(repeated)

        assert abs(float(out.perplexity.item()) - 1.0) < 1e-3
        assert int(out.indices.max().item()) == 3

    def test_loss_applies_the_commitment_cost(self) -> None:
        vq = nn.VectorQuantizer(16, 4, commitment_cost=0.5)
        out = vq(lucid.randn((6, 4)))
        expected = float(out.codebook_loss.item()) + 0.5 * float(out.commitment_loss.item())
        assert abs(float(vq.loss(out).item()) - expected) < 1e-6


class TestVectorQuantizerRoundTrip:
    def test_lookup_inverts_assign(self) -> None:
        vq = nn.VectorQuantizer(32, 6)
        x = lucid.randn((2, 5, 6))
        codes = vq.lookup(vq.assign(x))

        assert codes.shape == (2, 5, 6)
        assert float((codes - vq(x).quantized).abs().max().item()) < 1e-6

    def test_assign_matches_forward_indices(self) -> None:
        vq = nn.VectorQuantizer(32, 6)
        x = lucid.randn((4, 6))
        assert vq.assign(x).tolist() == vq(x).indices.tolist()


class TestVectorQuantizerGradients:
    def test_input_trains_through_the_quantiser(self) -> None:
        # argmin has zero gradient everywhere; without the straight-through
        # estimator this would be None.
        vq = nn.VectorQuantizer(16, 4)
        x = lucid.randn((6, 4), requires_grad=True)
        vq(x).quantized.sum().backward()

        assert x.grad is not None
        assert float(abs(x.grad).sum()) > 0.0

    def test_quantised_output_does_not_train_the_codebook(self) -> None:
        # The straight-through path routes gradient *past* the codebook —
        # which is exactly why the codebook term has to exist.
        vq = nn.VectorQuantizer(16, 4)
        vq(lucid.randn((6, 4), requires_grad=True)).quantized.sum().backward()

        assert vq.weight.grad is None

    def test_codebook_term_trains_the_codebook_only(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        x = lucid.randn((6, 4), requires_grad=True)
        vq(x).codebook_loss.backward()

        assert vq.weight.grad is not None
        assert float(abs(vq.weight.grad).sum().item()) > 0.0
        assert x.grad is None

    def test_commitment_term_trains_the_input_only(self) -> None:
        vq = nn.VectorQuantizer(16, 4)
        x = lucid.randn((6, 4), requires_grad=True)
        vq(x).commitment_loss.backward()

        assert x.grad is not None
        assert float(abs(x.grad).sum()) > 0.0
        assert vq.weight.grad is None
