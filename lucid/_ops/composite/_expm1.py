"""``expm1``'s value: Kahan's ratio, kept apart from its gradient (LCD-772)."""

import lucid
from lucid._tensor.tensor import Tensor


def _expm1(input: Tensor) -> Tensor:
    """:math:`e^x - 1` to full relative precision at every magnitude.

    The value is Kahan's form; the gradient is ``exp(x)``'s.
    """
    # Kahan's form: with u = exp(x), (u - 1) * x / log(u) cancels the rounding
    # of u against the same rounding in log(u), so it keeps full relative
    # precision near 0.  The ratio x / log(u) is grouped first: it is ~1 at
    # every magnitude, whereas (u - 1) * x leaves the dtype's range while the
    # answer is still finite (float16 at x = 10, float32 at 85, float64 at 705
    # all came back inf).  The guards take u == 1 (x underflowed), u == 0 and
    # u == inf, where the ratio is 0/0 or inf/inf.
    with lucid.no_grad():
        u = lucid.exp(input)
        accurate = lucid.where(u == 1.0, input, (u - 1.0) * (input / lucid.log(u)))
        # Kahan's second guard: once u - 1 rounds to -1, -1 is the answer.  The
        # ratio would take log(u) of a subnormal u, whose rounding is no longer
        # relative — expm1(-100) in float32 came out -1.000169.
        accurate = lucid.where(u - 1.0 == -1.0, u - 1.0, accurate)
        accurate = lucid.where((u == 0.0) | ~lucid.isfinite(u), u - 1.0, accurate)
    # The value is the accurate one; the gradient is exp(x)'s, carried by the
    # naive form under it, so the guards' 0/0 branches never reach backward.
    naive = lucid.exp(input) - 1.0
    with lucid.no_grad():
        correction = lucid.where(
            lucid.isfinite(naive), accurate - naive, lucid.zeros_like(accurate)
        )
    # Where exp(x) rounds to 1, x + 0.5 * c * (c - c.detach()) with clamped c
    # equals x bit-for-bit (preserving expm1(-0) = -0 and keeping ±inf finite
    # in MulBackward) while carrying both the unit slope and unit curvature of
    # exp(x) at the origin for higher-order autograd.
    c = lucid.clamp(input, -1.0, 1.0)
    small_branch = input + 0.5 * c * (c - c.detach())
    return lucid.where(u == 1.0, small_branch, naive + correction)
