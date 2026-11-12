"""Deliberate defects to put into a Lucid model, to prove a parity test can fail.

A parity comparison is only worth having if a wrong model does not pass it.  Each kind
below edits *one site* of a model that has already been aligned with its reference --
one weight tensor, one norm epsilon, one activation -- and returns the callable that
puts the site back, so a sweep can visit every site of one model without rebuilding it.

The sites are found by module type, so they exist in any model built from the
standard layers; a kind that finds none raises, rather than letting a test that
mutates nothing pass.
"""

from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

Undo = Callable[[], None]
Mutation = Callable[[], Undo]
MutationKind = Literal["weight", "norm_eps", "activation"]
Where = Literal["first", "middle", "last"]

#: Standard deviation of the noise added to one weight tensor.  Weights of the models
#: measured here are 0.02-0.3 in magnitude, so this is a 0.3%-5% perturbation of one layer.
#: It is noise, not a constant: a constant shift adds ``c * sum(x)`` to every output, and
#: ``sum(x)`` is zero for the output of a LayerNorm, so a pre-norm transformer cannot see it.
WEIGHT_NOISE: float = 1e-3

#: Each activation module and what a careless port would use in its place.
_ACTIVATION_SWAPS: dict[type[nn.Module], type[nn.Module]] = {
    nn.ReLU6: nn.ReLU,
    nn.ReLU: nn.ReLU6,
    nn.Hardswish: nn.ReLU,
}

_Weighted = nn.Conv1d | nn.Conv2d | nn.Linear
_Normed = nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d | nn.LayerNorm


class _ReluMLP(nn.Module):
    """A transformer MLP that applies ReLU where the architecture applies GELU.

    The MLP calls ``F.gelu`` rather than holding a GELU module, so there is no module to
    replace: this wraps the real one and reuses its layers.
    """

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.inner = mlp

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        mlp = self.inner
        return mlp.drop(mlp.fc2(mlp.drop(F.relu(mlp.fc1(x)))))


def _perturb_weight(layer: _Weighted) -> Undo:
    saved = layer.weight.numpy().copy()
    noise = np.random.default_rng(0).standard_normal(saved.shape).astype(saved.dtype)
    layer.weight.data.copy_(lucid.from_numpy(saved + WEIGHT_NOISE * noise))

    def undo() -> None:
        layer.weight.data.copy_(lucid.from_numpy(saved))

    return undo


def _flip_eps(norm: _Normed) -> Undo:
    """1e-3 where the norm holds a smaller epsilon, 1e-5 where it holds 1e-3 or more.

    The two conventions the zoo's families split on, so the mutation is the realistic
    one: a port that took the other framework's default.
    """
    saved = norm.eps
    norm.eps = 1e-3 if saved < 1e-3 else 1e-5

    def undo() -> None:
        norm.eps = saved

    return undo


def _replace(parent: nn.Module, name: str, child: nn.Module, replacement: nn.Module) -> Undo:
    setattr(parent, name, replacement)

    def undo() -> None:
        setattr(parent, name, child)

    return undo


def _activation_sites(model: nn.Module) -> list[Mutation]:
    found: list[Mutation] = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if type(child) in _ACTIVATION_SWAPS:
                replacement: nn.Module = _ACTIVATION_SWAPS[type(child)]()
            elif type(child).__name__ == "_MLP":
                replacement = _ReluMLP(child)
            else:
                continue
            found.append(partial(_replace, parent, name, child, replacement))
    return found


def sites(model: nn.Module, kind: MutationKind) -> list[Mutation]:
    """Every place ``kind`` of defect can be put into ``model``, in module order."""
    found: list[Mutation] = []
    if kind == "activation":
        found = _activation_sites(model)
    for module in model.modules():
        if kind == "weight" and isinstance(module, _Weighted):
            found.append(partial(_perturb_weight, module))
        elif kind == "norm_eps" and isinstance(module, _Normed):
            found.append(partial(_flip_eps, module))
    if not found:
        raise LookupError(f"no {kind} site in {type(model).__name__}")
    return found


def at(kind: MutationKind, where: Where) -> Callable[[nn.Module], None]:
    """A ``mutate`` for ``_run_parity``: put ``kind`` of defect into one site, by position."""

    def mutate(target: nn.Module) -> None:
        chosen = sites(target, kind)
        chosen[{"first": 0, "middle": len(chosen) // 2, "last": -1}[where]]()

    return mutate
