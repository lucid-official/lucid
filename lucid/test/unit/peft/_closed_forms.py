"""The forward pass of the tiny MLP under OFT / BOFT / HRA / FourierFT / LN tuning, from the reference's formulas.

``test_tuner_contract`` loads adapter files written by hand in the reference's
layout; these functions say what the loaded model must compute, written out
entry by entry from the file's tensors with plain loops and trigonometry, never
through the layers' own code (only the FourierFT *locations* come from the
package, and ``test_fourierft.py`` checks them against the reference framework).
The MLP is ``Linear(6, 8) -> ReLU -> Linear(8, 4)``, adapted at ``0`` and ``2``.
"""

import math
from collections.abc import Callable

import lucid
import lucid.nn as nn
from lucid.peft.tuners.fourierft._locations import _entry_indices

_PREFIX = "base_model.model."
_SIZES = {0: (8, 6), 2: (4, 8)}  # layer index -> (out, in)


def _mlp() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))


def _cayley_2x2(q: float) -> list[list[float]]:
    """``(I - Q)(I + Q)^-1`` for ``Q = [[0, q], [-q, 0]]``, expanded by hand."""
    d = 1 + q * q
    return [[(1 - q * q) / d, -2 * q / d], [2 * q / d, (1 - q * q) / d]]


def _blockdiag(blocks: list[list[list[float]]]) -> lucid.Tensor:
    n = 2 * len(blocks)
    rows = [[0.0] * n for _ in range(n)]
    for k, block in enumerate(blocks):
        for i in range(2):
            for j in range(2):
                rows[2 * k + i][2 * k + j] = block[i][j]
    return lucid.tensor(rows)


def _run(weights: list[lucid.Tensor], biases: list[lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    hidden = nn.functional.relu(x @ weights[0].mT + biases[0])
    return hidden @ weights[1].mT + biases[1]


def _oft(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Block size 2, exact Cayley: each layer sees ``x R`` with ``R`` block-diagonal."""
    net = _mlp()
    h = x
    for index in (0, 2):
        q = t[f"{_PREFIX}{index}.oft_R.weight"].reshape(-1).tolist()
        rotation = _blockdiag([_cayley_2x2(v) for v in q])
        h = (h @ rotation) @ net[index].weight.mT + net[index].bias
        if index == 0:
            h = nn.functional.relu(h)
    return h


def _boft(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Block size 2, one factor: ``W' = (W M^T) * s``, ``M`` block-diagonal Cayley blocks."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        raw = t[f"{_PREFIX}{index}.boft_R"].reshape(-1, 2, 2).tolist()
        blocks = [_cayley_2x2(0.5 * (m[0][1] - m[1][0])) for m in raw]
        rotation = _blockdiag(blocks)
        scale = t[f"{_PREFIX}{index}.boft_s"].reshape(-1, 1)
        weights.append((net[index].weight @ rotation.mT) * scale)
    return _run(weights, [net[0].bias, net[2].bias], x)


def _hra(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Rank 2: ``W' = W H_1 H_2``, ``H_i = I - 2 u_i u_i^T / |u_i|^2``."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        u = t[f"{_PREFIX}{index}.hra_u"]
        n = u.shape[0]
        product = lucid.eye(n)
        for i in range(u.shape[1]):
            column = u[:, i : i + 1]
            norm2 = float((column * column).sum().item())
            product = product @ (lucid.eye(n) - 2.0 * (column @ column.mT) / norm2)
        weights.append(net[index].weight @ product)
    return _run(weights, [net[0].bias, net[2].bias], x)


def _fourierft(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """5 coefficients, scale 2: ``dW[m, n] = s / (M N) * sum_j c_j cos(2 pi (u_j m / M + v_j n / N))``."""
    net = _mlp()
    weights = []
    for index, (out, inn) in _SIZES.items():
        spectrum = t[f"{_PREFIX}{index}.fourierft_spectrum"].tolist()
        where = _entry_indices(out, inn, len(spectrum), 777, "cpu").tolist()
        delta = [
            [
                2.0
                / (out * inn)
                * sum(
                    c * math.cos(2 * math.pi * (u * m / out + v * n / inn))
                    for c, u, v in zip(spectrum, where[0], where[1], strict=True)
                )
                for n in range(inn)
            ]
            for m in range(out)
        ]
        weights.append(net[index].weight + lucid.tensor(delta))
    return _run(weights, [net[0].bias, net[2].bias], x)


def _ln_tuning(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """The tuned copies replace the layers."""
    weights = [t[f"{_PREFIX}{i}.ln_tuning_layers.weight"] for i in (0, 2)]
    biases = [t[f"{_PREFIX}{i}.ln_tuning_layers.bias"] for i in (0, 2)]
    return _run(weights, biases, x)


#: Method name -> the MLP's output for the file's tensors ``t`` and the input ``x``.
CLOSED_FORMS: dict[str, Callable[[dict[str, lucid.Tensor], lucid.Tensor], lucid.Tensor]] = {
    "oft": _oft,
    "boft": _boft,
    "hra": _hra,
    "fourierft": _fourierft,
    "ln_tuning": _ln_tuning,
}
