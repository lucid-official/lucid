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


def _miss(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Shard width 2, balance variant: ``dW[o, c] = S[c mod 2, o]``."""
    net = _mlp()
    weights = []
    for index, (out, inn) in _SIZES.items():
        block = t[f"{_PREFIX}{index}.miss_block"].tolist()
        delta = [[block[c % 2][o] for c in range(inn)] for o in range(out)]
        weights.append(net[index].weight + lucid.tensor(delta))
    return _run(weights, [net[0].bias, net[2].bias], x)


def _randlora(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Rank 2, alpha 4: ``(alpha / r) * sum_{i,k} B[:, k, i] (lambda[i, k] A[i, :min] gamma[k, :])`` summed over the pairs the reference lines up.

    The reference flattens ``A``'s ``(r, n)`` axes rank-major and ``B``'s ``(n, r)``
    axes basis-major before the product, so the sum pairs ``A``'s ``(i, k)`` row with
    ``B``'s column ``(k' , i')`` at the same flat position.
    """
    net = _mlp()
    basis_a, basis_b = t["base_model.randlora_A"], t["base_model.randlora_B"]
    r = basis_a.shape[0]
    weights = []
    for index, (out, inn) in _SIZES.items():
        lam = t[f"{_PREFIX}{index}.randlora_lambda"].tolist()
        gamma = t[f"{_PREFIX}{index}.randlora_gamma"].tolist()
        low, high = min(out, inn), max(out, inn)
        bases = len(lam[0])
        # row (i * bases + k) of the scaled A, column (k * r + i) of B.
        scaled = [
            [lam[i][k] * basis_a[i, 0, c].item() * gamma[k][c] for c in range(low)]
            for i in range(r)
            for k in range(bases)
        ]
        columns = [
            [basis_b[row, k, i].item() for row in range(high)]
            for k in range(bases)
            for i in range(r)
        ]
        a_rows, b_cols = lucid.tensor(scaled), lucid.tensor(columns)  # (r n, low), (n r, high)
        product = b_cols.mT @ a_rows  # (high, low): the update with its larger side first
        delta = product if low == inn else product.mT
        weights.append(net[index].weight + delta * (4 / 2))
    return _run(weights, [net[0].bias, net[2].bias], x)


def _vblora(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Bank of 5 vectors of length 2, top-2: each length-2 piece of ``A`` and ``B`` is a softmax mix of two bank rows."""
    net = _mlp()
    bank = t["base_model.vblora_vector_bank"].tolist()

    def piece(logits: list[float]) -> list[float]:
        order = sorted(range(len(logits)), key=lambda k: -logits[k])[:2]
        top = [logits[k] for k in order]
        exps = [math.exp(v - max(top)) for v in top]
        weights = [e / sum(exps) for e in exps]
        return [sum(w * bank[k][d] for w, k in zip(weights, order, strict=True)) for d in range(2)]

    weights = []
    for index, (out, inn) in _SIZES.items():
        logits_a = t[f"{_PREFIX}{index}.vblora_logits_A"].tolist()  # (r, in/2, 5)
        logits_b = t[f"{_PREFIX}{index}.vblora_logits_B"].tolist()  # (out/2, r, 5)
        r = len(logits_a)
        factor_a = [[v for tile in logits_a[i] for v in piece(tile)] for i in range(r)]  # (r, in)
        factor_b = [
            [piece(logits_b[o][i])[d] for i in range(r)] for o in range(out // 2) for d in range(2)
        ]  # (out, r)
        weights.append(net[index].weight + lucid.tensor(factor_b) @ lucid.tensor(factor_a))
    return _run(weights, [net[0].bias, net[2].bias], x)


def _shira(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """The values sit at the file's indices; everything else of the weight is the base's."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        values = t[f"{_PREFIX}{index}.shira_weight"].tolist()
        rows, cols = t[f"{_PREFIX}{index}.shira_indices"].tolist()
        weight = net[index].weight.tolist()
        for value, i, j in zip(values, rows, cols, strict=True):
            weight[int(i)][int(j)] += value
        weights.append(lucid.tensor(weight))
    return _run(weights, [net[0].bias, net[2].bias], x)


def _c3a(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Block size 2: block ``(i, j)`` of ``dW`` is the circulant ``[[w0, w1], [w1, w0]]`` of kernel ``w``, over ``d_in``."""
    net = _mlp()
    weights = []
    for index, (out, inn) in _SIZES.items():
        kernel = t[f"{_PREFIX}{index}.c3a_kernel"].tolist()  # (out/2, in/2, 2)
        delta = [
            [kernel[o // 2][c // 2][(c % 2 - o % 2) % 2] / inn for c in range(inn)]
            for o in range(out)
        ]
        weights.append(net[index].weight + lucid.tensor(delta))
    return _run(weights, [net[0].bias, net[2].bias], x)


#: Method name -> the MLP's output for the file's tensors ``t`` and the input ``x``.
CLOSED_FORMS: dict[str, Callable[[dict[str, lucid.Tensor], lucid.Tensor], lucid.Tensor]] = {
    "oft": _oft,
    "boft": _boft,
    "hra": _hra,
    "fourierft": _fourierft,
    "ln_tuning": _ln_tuning,
    "miss": _miss,
    "randlora": _randlora,
    "vblora": _vblora,
    "shira": _shira,
    "c3a": _c3a,
}
