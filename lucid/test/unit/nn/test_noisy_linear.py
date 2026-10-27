"""Parametric exploration noise — Fortunato et al. (2018).

The layer is easy to write in a way that runs, trains and explores
nothing.  Three properties carry it and none is visible in a shape check.

The noise has to be **factorised**: one draw per input and one per
output, combined as an outer product.  Drawing one per weight is the
paper's other variant, gives the same shapes, and costs ``p * q`` random
numbers instead of ``p + q`` — the reason the paper uses factorised noise
for single-threaded agents at all.

The sample has to be **held**.  Redrawing inside ``forward`` makes two
calls on one input two different networks, which breaks the Monte-Carlo
gradient the paper derives and any agent that needs a fixed policy for an
episode.

And ``sigma`` has to **receive gradient**, because the whole claim is
that the scale of exploration is learned rather than annealed.
"""

import math

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


class TestScaledNoise:
    def test_it_is_the_papers_transform(self) -> None:
        r"""``f(x) = sgn(x) * sqrt(|x|)``, stated in equation 10's text."""
        x = lucid.tensor([-9.0, -4.0, 0.0, 4.0, 9.0])
        got = F.scaled_noise(x).tolist()
        assert [round(v, 5) for v in got] == [-3.0, -2.0, 0.0, 2.0, 3.0]

    def test_it_keeps_the_sign(self) -> None:
        x = lucid.randn((256,))
        signs = (F.scaled_noise(x) * x >= 0.0).to(lucid.float32)
        assert float(signs.mean().item()) == 1.0

    def test_it_compresses_magnitude(self) -> None:
        """Large draws are pulled in; that is what the square root is for."""
        x = lucid.tensor([100.0])
        assert float(F.scaled_noise(x).item()) == pytest.approx(10.0)


class TestNoisyLinearFunctional:
    def test_shapes(self) -> None:
        out = F.noisy_linear(lucid.zeros((2, 4)), lucid.zeros((3, 4)), lucid.ones((3, 4)))
        assert out.shape == (2, 3)

    def test_zero_sigma_is_an_ordinary_linear(self) -> None:
        """With no noise scale the layer must be exactly its mean."""
        lucid.manual_seed(0)
        x = lucid.randn((5, 4))
        mu = lucid.randn((3, 4))
        bias = lucid.randn((3,))
        noisy = F.noisy_linear(x, mu, lucid.zeros((3, 4)), bias, lucid.zeros((3,)))
        plain = F.linear(x, mu, bias)
        assert float((noisy - plain).abs().max().item()) < 1e-6

    def test_the_perturbation_is_the_outer_product(self) -> None:
        """Factorised, not per-weight — rebuilt from the two noise vectors."""
        lucid.manual_seed(0)
        x = lucid.randn((2, 4))
        mu, sigma = lucid.zeros((3, 4)), lucid.randn((3, 4))
        eps_in, eps_out = lucid.randn((4,)), lucid.randn((3,))

        got = F.noisy_linear(x, mu, sigma, None, None, eps_in, eps_out)

        f_in, f_out = F.scaled_noise(eps_in), F.scaled_noise(eps_out)
        weight = mu + sigma * (f_out.reshape(3, 1) * f_in.reshape(1, 4))
        assert float((got - F.linear(x, weight)).abs().max().item()) < 1e-5

    def test_a_per_weight_draw_would_differ(self) -> None:
        """Guards the test above — both variants produce the same shapes."""
        lucid.manual_seed(0)
        x = lucid.randn((2, 4))
        mu, sigma = lucid.zeros((3, 4)), lucid.ones((3, 4))
        eps_in, eps_out = lucid.randn((4,)), lucid.randn((3,))
        factorised = F.noisy_linear(x, mu, sigma, None, None, eps_in, eps_out)
        independent = F.linear(x, mu + sigma * lucid.randn((3, 4)))
        assert float((factorised - independent).abs().max().item()) > 1e-3

    def test_a_lone_bias_argument_is_refused(self) -> None:
        with pytest.raises(ValueError):
            F.noisy_linear(
                lucid.zeros((2, 4)),
                lucid.zeros((3, 4)),
                lucid.ones((3, 4)),
                lucid.zeros((3,)),
                None,
            )


class TestNoisyLinearModule:
    def test_the_initialisation_is_the_papers(self) -> None:
        """``mu ~ U[-1/sqrt(p), 1/sqrt(p)]``, ``sigma = sigma_0/sqrt(p)``."""
        layer = nn.NoisyLinear(16, 5, sigma_zero=0.5)
        bound = 1.0 / math.sqrt(16)
        assert float(layer.weight_mu.abs().max().item()) <= bound + 1e-6
        assert float(layer.weight_sigma[0, 0].item()) == pytest.approx(0.5 / math.sqrt(16))

    def test_the_sample_is_held_until_asked(self) -> None:
        """Two forwards must be the same network."""
        lucid.manual_seed(0)
        layer = nn.NoisyLinear(4, 3)
        layer.train()
        x = lucid.ones((1, 4))
        assert float((layer(x) - layer(x)).abs().max().item()) == 0.0

    def test_resample_changes_it(self) -> None:
        """Guards the test above — a layer with no noise would also pass."""
        lucid.manual_seed(0)
        layer = nn.NoisyLinear(4, 3)
        layer.train()
        x = lucid.ones((1, 4))
        before = layer(x)
        layer.resample()
        assert float((layer(x) - before).abs().max().item()) > 1e-6

    def test_eval_is_the_mean_network(self) -> None:
        """A deterministic policy is the layer without its noise."""
        lucid.manual_seed(0)
        layer = nn.NoisyLinear(4, 3)
        layer.eval()
        x = lucid.randn((2, 4))
        assert (
            float((layer(x) - F.linear(x, layer.weight_mu, layer.bias_mu)).abs().max().item())
            < 1e-6
        )

    def test_sigma_learns(self) -> None:
        """The claim: exploration scale is trained, not scheduled."""
        lucid.manual_seed(0)
        layer = nn.NoisyLinear(4, 3)
        layer.train()
        layer(lucid.randn((8, 4))).sum().backward()
        assert layer.weight_sigma.grad is not None
        assert float(layer.weight_sigma.grad.abs().sum().item()) > 0.0
        assert layer.bias_sigma is not None and layer.bias_sigma.grad is not None

    def test_the_noise_is_not_a_parameter(self) -> None:
        """It is sampled, so an optimiser must never see it."""
        layer = nn.NoisyLinear(4, 3)
        names = {n for n, _ in layer.named_parameters()}
        assert names == {"weight_mu", "weight_sigma", "bias_mu", "bias_sigma"}
        assert {n for n, _ in layer.named_buffers()} == {"epsilon_in", "epsilon_out"}

    def test_it_costs_p_plus_q_draws_not_p_times_q(self) -> None:
        """The reason the paper factorises at all."""
        layer = nn.NoisyLinear(64, 32)
        assert layer.noise_in.shape == (64,)
        assert layer.noise_out.shape == (32,)

    def test_without_bias(self) -> None:
        layer = nn.NoisyLinear(4, 3, bias=False)
        assert layer.bias_mu is None and layer.bias_sigma is None
        assert layer(lucid.zeros((2, 4))).shape == (2, 3)

    @pytest.mark.parametrize("sigma_zero", [0.0, -1.0])
    def test_it_rejects_a_non_positive_scale(self, sigma_zero: float) -> None:
        with pytest.raises(ValueError):
            nn.NoisyLinear(4, 3, sigma_zero=sigma_zero)

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_it_runs_on_device(self, device: str) -> None:
        layer = nn.NoisyLinear(4, 3).to(device)
        layer.train()
        out = layer(lucid.zeros((2, 4), device=device))
        assert str(out.device) == f"device('{device}')"
        layer.resample()
        assert str(layer.noise_in.device) == f"device('{device}')"
