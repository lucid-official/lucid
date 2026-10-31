"""The Gamma rejection sampler's retry rounds are independent (LCD-784).

The sampler used to carry an "accepted so far" mask from round to round; the
rounds are now independent and the first accepting one is picked by a ``where``
chain.  These pin the draw of both code paths -- eager (stops early once every
cell accepted) and traced (no host reads, always every round) -- to the values
the carried-mask sampler produced, and the cell no round accepts to ``d``.
"""

import math

import pytest

import lucid
import lucid.distributions.gamma as gamma_mod
from lucid.compile._core.trace_context import recording

_ALPHA = (0.5, 1.0, 2.5, 7.0)

# First four log-draws of ``_sample_standard_log_gamma(alpha, shape)`` under
# ``manual_seed(3)``, as the carried-mask sampler produced them.
_PINNED: dict[tuple[str, tuple[int, ...]], list[float]] = {
    ("eager", ()): [-5.91094, 0.28342, 1.07524, 2.10481],
    ("traced", ()): [-1.10495, 0.28342, 1.07524, 2.10481],
    ("eager", (3,)): [-2.6353, 0.28342, 1.07524, 2.10481],
    ("traced", (3,)): [-1.03632, 0.28342, 1.07524, 2.10481],
    ("eager", (2, 3)): [-3.62301, 0.28342, 1.07524, 2.10481],
    ("traced", (2, 3)): [-3.43374, 0.28342, 1.07524, 2.10481],
}


@pytest.mark.parametrize("mode, shape", list(_PINNED))
def test_draw_matches_the_carried_mask_sampler(mode: str, shape: tuple[int, ...]) -> None:
    alpha = lucid.tensor(list(_ALPHA))
    lucid.manual_seed(3)
    if mode == "traced":
        with recording():
            out = gamma_mod._sample_standard_log_gamma(alpha, shape)
    else:
        out = gamma_mod._sample_standard_log_gamma(alpha, shape)
    assert tuple(out.shape) == shape + (4,)
    got = out.reshape(-1).tolist()[:4]
    assert got == pytest.approx(_PINNED[(mode, shape)], abs=2e-5)


@pytest.mark.parametrize("mode", ["eager", "traced"])
def test_cell_no_round_accepts_takes_the_mode(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """With one round, the ~5 % of cells that reject fall back to ``d = alpha - 1/3``."""
    monkeypatch.setattr(gamma_mod, "_MAX_GAMMA_RETRIES", 1)
    alpha = lucid.tensor([2.5] * 400)
    lucid.manual_seed(5)
    if mode == "traced":
        with recording():
            out = gamma_mod._sample_standard_log_gamma(alpha, ())
    else:
        out = gamma_mod._sample_standard_log_gamma(alpha, ())
    values = out.tolist()
    fallback = math.log(2.5 - 1.0 / 3.0)
    n_fallback = sum(1 for v in values if abs(v - fallback) < 1e-6)
    assert 0 < n_fallback < 80
    assert all(math.isfinite(v) for v in values)
