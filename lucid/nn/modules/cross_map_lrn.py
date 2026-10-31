"""
Cross-channel local response normalisation, in the form the reference ships.
"""

from typing import final, override

import lucid as _lucid
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.nn.functional.sampling import pad
from lucid.nn.module import Module


def _window_sum(t: Tensor, low: int, high: int) -> Tensor:
    """``out[:, c] = sum(t[:, c + d] for d in low..high)``, channels outside ``t`` counting as zero."""
    channels = int(t.shape[1])
    left, right = max(0, -low), max(0, high)
    padded = pad(t, [0, 0, 0, 0, left, right])
    shifted = [padded[:, d + left : d + left + channels] for d in range(low, high + 1)]
    return _lucid.stack(shifted, 0).sum(dim=0)


def _scale(sq: Tensor, size: int, alpha: float, k: float) -> Tensor:
    """The normalisation base ``k + alpha / size * (window sum of x**2)``.

    The window is the reference's, quirks included: ``2 * pre - 1`` channels
    centred on ``c`` (``pre = (size - 1) // 2 + 1``) and, from ``c = pre``
    on, channel ``0`` as well — the reference slides the window by adding the
    channel that enters and removing the one that leaves, and the removal
    starts one step late, so channel ``0`` is never removed.
    """
    pre = (size - 1) // 2 + 1
    total = _window_sum(sq, -(pre - 1), pre - 1)
    channels = int(sq.shape[1])
    past_first = (_lucid.arange(channels, device=sq.device) >= pre).to(sq.dtype)
    total = total + sq[:, :1] * past_first.reshape(1, channels, 1, 1)
    return total * (alpha / size) + k


@final
class _CrossMapLRN2d(Function):
    """The forward above with the reference's hand-written backward.

    That backward is not the derivative of its own forward (it was written
    for a window without the quirk and for the opposite channel offset), so
    differentiating the forward would give gradients the reference does not.
    """

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx, input: Tensor, size: int, alpha: float, beta: float, k: float
    ) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"CrossMapLRN2d: Expected input to be 4D, got {input.ndim}D instead.")
        ctx.size, ctx.alpha, ctx.beta = size, alpha, beta
        x = input.detach()
        scale = _scale(x * x, size, alpha, k)
        output = x * scale ** (-beta)
        ctx.save_for_backward(x, scale, output)
        return output

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None, None, None, None]:
        size, alpha, beta = ctx.size, ctx.alpha, ctx.beta
        if size == 1:
            raise RuntimeError("CrossMapLRN2d: the backward is not defined for size=1")
        x, scale, output = ctx.saved_tensors
        # The reference pads the ratio by ``size - 1`` and reads ``size``
        # channels starting at ``c``, from an offset of ``inverse_pre``.
        inverse_pre = int(size - (size - 1) / 2)
        accum = _window_sum(grad * output / scale, -inverse_pre, size - 1 - inverse_pre)
        coefficient = 2.0 * alpha * beta / size
        return grad * scale ** (-beta) - coefficient * x * accum, None, None, None, None


class CrossMapLRN2d(Module):
    r"""Local response normalisation across channels, as the reference's legacy layer.

    Each activation is divided by a power of a sum of squares over
    neighbouring channels:

    .. math::

        y_c = \frac{x_c}{\left(k + \frac{\alpha}{n} \sum_{c'} x_{c'}^2\right)^{\beta}},
        \qquad n = \text{size}.

    This is the original Lua-era layer, reproduced to the reference's values *and*
    gradients, which makes it differ from :class:`LocalResponseNorm` in two
    ways.  The window is :math:`2\lfloor (n-1)/2 \rfloor + 1` channels wide
    (``n`` for odd ``n``, ``n - 1`` for even), and from channel
    :math:`\lfloor (n-1)/2 \rfloor + 1` on it also includes channel ``0``
    (a sliding-window bookkeeping slip the reference has always had).  The
    gradient is the reference's hand-written one, not the derivative of the
    forward; it is not differentiable a second time.

    Parameters
    ----------
    size : int
        Channels the normalisation spans.
    alpha : float, optional
        Multiplicative factor.  Default: ``1e-4``.
    beta : float, optional
        Exponent.  Default: ``0.75``.
    k : float, optional
        Additive constant.  Default: ``1``.

    Shape
    -----
    * **Input**: ``(N, C, H, W)``.
    * **Output**: the same shape.

    Raises
    ------
    ValueError
        If the input is not 4-D.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> lrn = nn.CrossMapLRN2d(size=5)
    >>> lrn(lucid.randn(2, 8, 4, 4)).shape
    (2, 8, 4, 4)

    See Also
    --------
    LocalResponseNorm : The averaging form, with the symmetric window.
    """

    size: int
    alpha: float
    beta: float
    k: float

    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1) -> None:
        """Initialise the module. See the class docstring for parameter semantics."""
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Normalise ``input`` across its channels."""
        out = _CrossMapLRN2d.apply(input, self.size, self.alpha, self.beta, self.k)
        assert isinstance(out, Tensor)
        return out

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"{self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}"
