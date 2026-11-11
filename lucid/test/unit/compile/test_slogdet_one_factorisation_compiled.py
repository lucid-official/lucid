"""A compiled ``slogdet`` step takes its gradient from the matrix its value came from (LCD-954).

The graph's elimination has no fused multiply-add, so a row-duplicated matrix can
end on an exact zero pivot where eager's LAPACK leaves rounding noise (and the
reverse): the compiled value and eager's differ, and both are right.  What the
compiled step may not do is disagree with *itself*.  Its backward used to invert
``A`` -- a different matrix from the ``Aᵀ`` the forward eliminated, so a value the
forward answered finite could meet an exact zero in the gradient.  It now inverts
``Aᵀ``, the one the pivots came from: a finite value has a finite gradient and a
``-inf`` one has the ``cof / det`` infinities.
"""

import math
import warnings

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning

_SIZES = [3, 4, 8, 16]
_SEEDS = range(6)


class _LogAbsDet(nn.Module):
    """``log|det A|`` with ``A`` the one parameter, so ``A``'s gradient is the answer."""

    def __init__(self, a: lucid.Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(a.clone())

    def forward(self, w: lucid.Tensor) -> lucid.Tensor:
        sign, logabs = lucid.linalg.slogdet(self.A)
        return (logabs * w).sum() + sign.sum() * 0.0


def _row_duplicated(n: int, seed: int, device: str) -> lucid.Tensor:
    lucid.manual_seed(seed)
    a = lucid.randn(n, n, device=device)
    a[n - 1] = a[1]
    return a


def _compiled_step(a: lucid.Tensor) -> tuple[float, list[float]]:
    """The loss and ``dA`` of a compiled step, which must not have fallen back."""
    w = lucid.ones((1,), device=a.device)
    dummy = lucid.zeros((1,), device=a.device)
    net = _LogAbsDet(a).to(a.device)
    step = make_step(net, lambda y, _t: y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        # The first call runs eagerly and records; the second is the compiled run.
        step(w, dummy).backward()
        net.A.grad = None
        loss = step(w, dummy)
        loss.backward()
    assert net.A.grad is not None
    return loss.item(), net.A.grad.to("cpu").reshape(-1).tolist()


def test_a_compiled_step_through_a_row_duplicated_matrix_agrees_with_its_own_value(
    device_gpu_only: str,
) -> None:
    classes: set[bool] = set()
    for n in _SIZES:
        for seed in _SEEDS:
            loss, grad = _compiled_step(_row_duplicated(n, seed, device_gpu_only))
            finite_value = math.isfinite(loss)
            classes.add(finite_value)
            assert all(math.isfinite(g) for g in grad) == finite_value, (n, seed, loss)
            if not finite_value:
                # cof / det: ``±inf`` along the cofactor, NaN where it is zero.
                assert not any(math.isfinite(g) for g in grad), (n, seed)
    # Both outcomes occur across the spread, so neither half of the check is vacuous.
    assert classes == {True, False}
