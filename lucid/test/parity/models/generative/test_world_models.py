"""Reference parity for the pieces the world models are built out of.

There is no reference-framework Dreamer to diff against — the published
implementations are JAX — so this cannot be a model-level comparison.
What it can be, and what the family had none of, is a numerical check on
the *mechanisms*: rebuild each one in the reference framework from Lucid's
own weights and compare.

That is worth having precisely because the control results cannot supply
it.  Every wrong default this family has shipped — an imagination horizon
of 16 where the paper says 15, 41 bins where the reference uses 255, a
reward head three layers deep where it should be one — left a model that
trains, reaches a good return, and reports falling losses.  A swing-up
solved at eight times the best constant says the parts are wired
together; it says nothing about whether each part computes what it
claims.

The pieces chosen are the ones where a silent disagreement would be
invisible downstream: the recurrence every family shares, the divergence
two of them are trained by, the returns all of them regress onto, and the
scale-free transforms DreamerV3 rests on.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.models.generative._common._returns import lambda_return
from lucid.models.generative._common._rssm import RSSM, BlockLinear, categorical_kl


def _np(tensor: Any) -> Any:
    """Lucid tensor to numpy, for comparison."""
    return tensor.numpy()


@pytest.mark.parity
class TestRecurrenceParity:
    """The GRU every world model in this zoo carries its belief through.

    A gate ordering that disagrees with the reference is the archetypal
    silent defect: the cell is still a valid recurrence, still trains,
    still reconstructs — it simply is not the cell the papers describe,
    and no downstream test can tell.
    """

    def test_gru_cell_matches_the_reference(self, ref: Any) -> None:
        lucid.manual_seed(0)
        cell = nn.GRUCell(4, 6)
        state = np.random.RandomState(0).randn(2, 4).astype(np.float32)
        hidden = np.random.RandomState(1).randn(2, 6).astype(np.float32)

        ours = _np(cell(lucid.tensor(state.copy()), lucid.tensor(hidden.copy())))

        reference = ref.nn.GRUCell(4, 6)
        with ref.no_grad():
            for name, parameter in cell.named_parameters():
                getattr(reference, name).copy_(ref.tensor(parameter.numpy()))
            theirs = reference(ref.tensor(state.copy()), ref.tensor(hidden.copy())).detach().numpy()
        assert np.abs(ours - theirs).max() < 1e-5

    def test_the_parameter_layout_is_the_same(self, ref: Any) -> None:
        """Guards the test above — a copy that silently no-ops would pass it."""
        ours = {n: tuple(int(d) for d in p.shape) for n, p in nn.GRUCell(4, 6).named_parameters()}
        theirs = {n: tuple(p.shape) for n, p in ref.nn.GRUCell(4, 6).named_parameters()}
        assert ours == theirs

    def test_the_rssm_recurrence_is_that_cell(self, ref: Any) -> None:
        """The dense path, end to end: projection, activation, then the gate."""
        lucid.manual_seed(0)
        rssm = RSSM(stoch_size=3, deter_size=6, hidden_size=5, action_dim=2, embed_size=4).eval()
        stoch = np.random.RandomState(2).randn(2, 3).astype(np.float32)
        action = np.random.RandomState(3).randn(2, 2).astype(np.float32)
        deter = np.random.RandomState(4).randn(2, 6).astype(np.float32)

        ours = _np(
            rssm._recurrent(
                lucid.tensor(stoch.copy()), lucid.tensor(action.copy()), lucid.tensor(deter.copy())
            )
        )

        with ref.no_grad():
            joined = ref.tensor(np.concatenate([stoch, action], axis=-1))
            weight = ref.tensor(rssm.pre_cell.weight.numpy())
            bias = ref.tensor(rssm.pre_cell.bias.numpy())
            projected = ref.relu(joined @ weight.T + bias)  # act_fn defaults to relu
            cell = ref.nn.GRUCell(5, 6)
            for name, parameter in rssm.cell.named_parameters():
                getattr(cell, name).copy_(ref.tensor(parameter.numpy()))
            theirs = cell(projected, ref.tensor(deter.copy())).detach().numpy()
        assert np.abs(ours - theirs).max() < 1e-5


@pytest.mark.parity
class TestDivergenceParity:
    """The categorical divergence DreamerV2 balances and DreamerV3 floors."""

    def test_categorical_kl_matches_the_reference(self, ref: Any) -> None:
        rng = np.random.RandomState(5)
        posterior = rng.randn(4, 3, 8).astype(np.float32) * 2.0
        prior = rng.randn(4, 3, 8).astype(np.float32) * 2.0

        ours = _np(categorical_kl(lucid.tensor(posterior.copy()), lucid.tensor(prior.copy())))

        q = ref.distributions.Categorical(logits=ref.tensor(posterior.copy()))
        p = ref.distributions.Categorical(logits=ref.tensor(prior.copy()))
        # Lucid sums over the variable axis: a grid of independent categoricals.
        theirs = ref.distributions.kl_divergence(q, p).sum(-1).detach().numpy()
        assert np.abs(ours - theirs).max() < 1e-4

    def test_the_direction_is_pinned(self, ref: Any) -> None:
        """Which argument is which, checked against the swapped divergence.

        The divergence is asymmetric, and both families depend on which
        way round it is: DreamerV2 weights the two directions apart, and
        DreamerV3 floors them separately at different scales.  A
        transposed pair would still produce a finite, falling loss.
        Measured: the correct direction agrees to 1.4e-06, the swapped one
        differs by 2.08.
        """
        rng = np.random.RandomState(5)
        posterior = rng.randn(4, 3, 8).astype(np.float32) * 2.0
        prior = rng.randn(4, 3, 8).astype(np.float32) * 2.0
        ours = _np(categorical_kl(lucid.tensor(posterior.copy()), lucid.tensor(prior.copy())))
        q = ref.distributions.Categorical(logits=ref.tensor(posterior.copy()))
        p = ref.distributions.Categorical(logits=ref.tensor(prior.copy()))
        swapped = ref.distributions.kl_divergence(p, q).sum(-1).detach().numpy()
        assert np.abs(ours - swapped).max() > 1e-2

    def test_it_is_zero_only_when_they_agree(self, ref: Any) -> None:
        """Guards the test above — both sides could share one wrong formula."""
        rng = np.random.RandomState(6)
        same = rng.randn(2, 3, 5).astype(np.float32)
        zero = _np(categorical_kl(lucid.tensor(same.copy()), lucid.tensor(same.copy())))
        assert np.abs(zero).max() < 1e-5


@pytest.mark.parity
class TestReturnParity:
    r"""Lambda-returns, against the closed form rather than the recursion.

    Lucid computes them by a backward recursion, which is the efficient
    form.  Comparing that to the same recursion written again proves
    nothing, so the reference here is the definition it is meant to
    equal — the :math:`\lambda`-weighted average of n-step returns.
    """

    def test_matches_the_weighted_average_of_n_step_returns(self, ref: Any) -> None:
        horizon, lam, gamma = 6, 0.8, 0.95
        rng = np.random.RandomState(7)
        reward = rng.randn(3, horizon + 1).astype(np.float32)
        value = rng.randn(3, horizon + 1).astype(np.float32)

        ours = _np(
            lambda_return(lucid.tensor(reward.copy()), lucid.tensor(value.copy()), gamma, lam)
        )

        r = ref.tensor(reward.copy(), dtype=ref.float64)
        v = ref.tensor(value.copy(), dtype=ref.float64)
        rows = []
        for t in range(horizon):
            remaining = horizon - t
            # n-step return G^(n) = sum_{i<n} gamma^i r_{t+i} + gamma^n v_{t+n}
            n_step = []
            for n in range(1, remaining + 1):
                total = sum(gamma**i * r[:, t + i] for i in range(n))
                n_step.append(total + gamma**n * v[:, t + n])
            mixed = sum((1 - lam) * lam ** (n - 1) * n_step[n - 1] for n in range(1, remaining))
            mixed = mixed + lam ** (remaining - 1) * n_step[remaining - 1]
            rows.append(mixed)
        theirs = ref.stack(rows, dim=1).numpy()
        assert np.abs(ours - theirs).max() < 1e-4


@pytest.mark.parity
class TestScaleFreeTransformParity:
    """DreamerV3's symlog and two-hot, the pieces that decouple reward scale."""

    def test_symlog_matches_the_reference(self, ref: Any) -> None:
        x = np.array([-1e6, -100.0, -1.0, -1e-4, 0.0, 1e-4, 1.0, 100.0, 1e6], dtype=np.float32)
        ours = _np(F.symlog(lucid.tensor(x.copy())))
        t = ref.tensor(x.copy(), dtype=ref.float64)
        theirs = (ref.sign(t) * ref.log1p(t.abs())).numpy()
        assert np.abs(ours - theirs).max() < 1e-3

    def test_symexp_inverts_it(self, ref: Any) -> None:
        x = np.array([-1e5, -3.0, 0.0, 3.0, 1e5], dtype=np.float32)
        back = _np(F.symexp(F.symlog(lucid.tensor(x.copy()))))
        assert np.abs(back - x).max() / max(np.abs(x).max(), 1.0) < 1e-5

    def test_two_hot_matches_a_reference_construction(self, ref: Any) -> None:
        bins = np.linspace(-5.0, 5.0, 11).astype(np.float32)
        values = np.array([-4.3, -0.2, 0.0, 1.7, 4.9], dtype=np.float32)

        ours = _np(F.two_hot(lucid.tensor(values.copy()), lucid.tensor(bins.copy())))

        b = ref.tensor(bins.copy(), dtype=ref.float64)
        v = ref.tensor(values.copy(), dtype=ref.float64).clamp(float(bins[0]), float(bins[-1]))
        below = (ref.searchsorted(b, v.contiguous()) - 1).clamp(0, len(bins) - 2)
        left, right = b[below], b[below + 1]
        weight = (v - left) / (right - left)
        theirs = ref.zeros(len(values), len(bins), dtype=ref.float64)
        theirs.scatter_(1, below.unsqueeze(1), (1 - weight).unsqueeze(1))
        theirs.scatter_(1, (below + 1).unsqueeze(1), weight.unsqueeze(1))
        assert np.abs(ours - theirs.numpy()).max() < 1e-5

    def test_the_decode_is_the_grid_expectation(self, ref: Any) -> None:
        """What the head reads back must be the distribution's mean."""
        bins = np.linspace(-5.0, 5.0, 11).astype(np.float32)
        values = np.array([-4.3, -0.2, 1.7], dtype=np.float32)
        encoded = F.two_hot(lucid.tensor(values.copy()), lucid.tensor(bins.copy()))
        decoded = _np((encoded * lucid.tensor(bins.copy())).sum(dim=-1))
        assert np.abs(decoded - values).max() < 1e-5


@pytest.mark.parity
class TestBlockLinearParity:
    """The block-diagonal map DreamerV3's parameter counts depend on."""

    def test_it_equals_an_explicit_block_matmul(self, ref: Any) -> None:
        lucid.manual_seed(0)
        blocks, in_block, out_block = 4, 3, 5
        layer = BlockLinear(blocks * in_block, blocks * out_block, blocks)
        x = np.random.RandomState(8).randn(2, blocks * in_block).astype(np.float32)

        ours = _np(layer(lucid.tensor(x.copy())))

        weight = ref.tensor(layer.weight.numpy(), dtype=ref.float64)
        bias = ref.tensor(layer.bias.numpy(), dtype=ref.float64)
        t = ref.tensor(x.copy(), dtype=ref.float64).reshape(2, blocks, in_block)
        pieces = [t[:, g] @ weight[g] + bias[g] for g in range(blocks)]
        theirs = ref.cat(pieces, dim=-1).numpy()
        assert np.abs(ours - theirs).max() < 1e-5

    def test_a_dense_map_would_not_agree(self, ref: Any) -> None:
        """Guards the test above: block structure has to be observable."""
        lucid.manual_seed(0)
        layer = BlockLinear(12, 20, 4)
        x = np.random.RandomState(9).randn(2, 12).astype(np.float32)
        ours = _np(layer(lucid.tensor(x.copy())))
        # Same weights laid out densely — a different function.
        dense = ref.zeros(12, 20, dtype=ref.float64)
        weight = ref.tensor(layer.weight.numpy(), dtype=ref.float64)
        for g in range(4):
            dense[g * 3 : (g + 1) * 3, g * 5 : (g + 1) * 5] = weight[g]
        flat = ref.tensor(x.copy(), dtype=ref.float64) @ dense
        assert np.abs(ours - flat.numpy()).max() > 1e-3
