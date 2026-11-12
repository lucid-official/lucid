r"""The OSF layers: a linear module whose low singular directions train, with their gradients kept out of the high ones.

OSF (Nayak et al., 2025, arXiv:2504.07097, section 3) splits the weight by its SVD, freezes the top
``effective_rank`` triplets and trains the rest as three tensors, :math:`U_{low}`, :math:`\Sigma_{low}`
and :math:`V_{low}` (section 3.2).  The layer computes with the weight
:math:`W' = W + U_{low}\Sigma_{low}V_{low}^\top - U^0_{low}\Sigma^0_{low}V^{0\top}_{low}` where the
superscript marks the frozen copies made when the adapter was, i.e.
:math:`W' = U_{high}\Sigma_{high}V_{high}^\top + U_{low}\Sigma_{low}V_{low}^\top`: no parameter is added to the
model, the trainable ones are the weight's own low part.  The gradients of :math:`U_{low}` and
:math:`V_{low}` are projected onto the complement of the high subspace as they flow (section 3.5, Eq. 4),
so training cannot move the weight along the directions that were frozen.

The update is a difference of two products that are equal at the start, so it is exactly zero then, and
merging adds it to the weight.  ``OSFLayer`` is the family root.  An adapter holds ``osf_svd_params`` (the
three trained tensors, ``U_low``, ``S_low``, ``V_low``, the reference's keys), the frozen buffers
``_osf_U_low_init``, ``_osf_S_low_init``, ``_osf_V_low_init`` and, where the low factors cannot stand in for
them, ``_osf_U_high`` and ``_osf_V_high``; the buffers are not in an adapter file, they are made again from
the base weight.
"""

from functools import partial
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._support import (
    _adapter_dtype,
    _ConfigLayer,
    _LinearSized,
    _ResidualUpdate,
    _SpecField,
)
from lucid.peft.tuners.osf._config import OSFConfig
from lucid.peft.tuners.osf._split import _Complement, _resolve_rank, _Split


class _OsfSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the frozen rank shapes the graph.
    """

    effective_rank: int


class OSFLayer(_ConfigLayer[OSFConfig], _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the split weight shared by the OSF layers.

    Per adapter: ``osf_svd_params``, the frozen buffers described above and the read-only mapping
    ``effective_rank``.  ``get_delta_weight`` is :math:`U_{low}\Sigma_{low}V_{low}^\top` minus the same of
    the frozen copies; the forward pass applies it to the input as two products of the factors.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` whose weight is split; its parameters stay shared.
    adapter_name : str
        Name of the first adapter.
    config : OSFConfig
        ``effective_rank`` (a count, a fraction of the smaller side, or ``None`` for half of it) and ``init_weights``.

    Raises
    ------
    TypeError
        ``base_layer`` is some other module type.
    ValueError
        ``base_layer`` has parameters that only its first input will create, or the rank is not positive.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("osf_svd_params",)
    _repr_prefix = "osf."
    osf_svd_params: nn.ModuleDict[nn.Module]
    _osf_U_low_init: _BufferDict
    _osf_S_low_init: _BufferDict
    _osf_V_low_init: _BufferDict
    _osf_U_high: _BufferDict
    _osf_V_high: _BufferDict

    effective_rank = _SpecField[int]("Frozen rank of each adapter, by name (read-only).")

    @override
    def _new_table(self) -> nn.ModuleDict[nn.Module]:
        return nn.ModuleDict()

    @override
    def _prepare(self, config: OSFConfig) -> None:
        super()._prepare(config)
        for name in ("_osf_U_low_init", "_osf_S_low_init", "_osf_V_low_init"):
            setattr(self, name, _BufferDict(persistent=False))
        self._osf_U_high = _BufferDict(persistent=False)
        self._osf_V_high = _BufferDict(persistent=False)

    @override
    def _enter(self, adapter_name: str, config: OSFConfig) -> None:
        anchor = self.weight
        rows, columns = self.out_features, self.in_features
        frozen = _resolve_rank(config.effective_rank, min(rows, columns))
        if frozen <= 0:
            raise ValueError(
                f"`effective_rank` should be a positive integer value but the value passed is {frozen}"
            )
        split = _Split.of(anchor, frozen, _adapter_dtype(anchor))
        self._osf_U_low_init[adapter_name] = split.u_low.clone()
        self._osf_S_low_init[adapter_name] = split.s_low.clone()
        self._osf_V_low_init[adapter_name] = split.v_low.clone()
        # A high factor is kept only where the low one cannot stand in for it (see ``_Complement``).
        if rows > columns:
            self._osf_U_high[adapter_name] = split.u_high
        if columns > rows:
            self._osf_V_high[adapter_name] = split.v_high
        trained = {"U_low": split.u_low, "S_low": split.s_low, "V_low": split.v_low}
        if config.init_weights is False:
            # A random start, for tests; the frozen copies stay the weight's own.
            trained = {
                name: lucid.randn(*part.shape, device=part.device, dtype=part.dtype)
                for name, part in trained.items()
            }
        self._specs[adapter_name] = _OsfSpec(frozen)
        self.osf_svd_params[adapter_name] = nn.ParameterDict(
            {name: Parameter(part) for name, part in trained.items()}
        )

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its trained tensors, frozen copies and recorded settings."""
        super().delete_adapter(adapter_name)
        for table in (
            self._osf_U_low_init,
            self._osf_S_low_init,
            self._osf_V_low_init,
            self._osf_U_high,
            self._osf_V_high,
        ):
            if adapter_name in table:
                del table[adapter_name]

    # ── the update ────────────────────────────────────────────────────────

    def _trained(self, adapter: str) -> tuple[Parameter, Parameter, Parameter]:
        """``(U_low, S_low, V_low)`` of ``adapter``, the tensors that train."""
        table = self.osf_svd_params[adapter]
        assert isinstance(table, nn.ParameterDict)
        return (
            _parameter_at(table, "U_low"),
            _parameter_at(table, "S_low"),
            _parameter_at(table, "V_low"),
        )

    @staticmethod
    def _kept(table: _BufferDict, adapter: str) -> Tensor | None:
        """The high factor ``table`` keeps for ``adapter``, or ``None`` where the low one stands in for it."""
        try:
            return table[adapter]
        except KeyError:
            return None

    def _guarded(self, adapter: str) -> tuple[Tensor, Tensor, Tensor]:
        """``(U_low, S_low, V_low)`` as the forward pass reads them: ``U`` and ``V`` seen through the gradient projections.

        When a backward pass may follow, ``U_low`` and ``V_low`` are read through an alias that carries
        a hook: the gradient that reaches the parameter is the projection (section 3.5, Eq. 4) of the
        one that reached the alias, as the reference's hooks on the parameters make it.  The hook belongs
        to the graph of this forward pass, not to the parameter, so it is there after ``.to()`` and after a
        copy of the model, neither of which a hook on the parameter survives.

        The hook holds the frozen factors it projects through and not this layer: the engine keeps it where
        the garbage collector cannot see, and a layer it held would be held by the graph as well.
        """
        left, singular, right = self._trained(adapter)
        if not lucid.is_grad_enabled():
            return left, singular, right
        seen_left: Tensor = left
        seen_right: Tensor = right
        if left.requires_grad:
            u_init = self._osf_U_low_init[adapter]
            u_high = self._kept(self._osf_U_high, adapter)
            seen_left = left.view(*left.shape)
            seen_left.register_hook(partial(_Complement.left, u_init=u_init, u_high=u_high))
        if right.requires_grad:
            v_init = self._osf_V_low_init[adapter]
            v_high = self._kept(self._osf_V_high, adapter)
            seen_right = right.view(*right.shape)
            seen_right.register_hook(partial(_Complement.right, v_init=v_init, v_high=v_high))
        return seen_left, singular, seen_right

    @staticmethod
    def _product(x: Tensor, u: Tensor, s: Tensor, v: Tensor) -> Tensor:
        r""":math:`x\,(U\,\mathrm{diag}(s)\,V)^\top`, as two products of the factors."""
        return ((x.to(u.dtype) @ v.mT) * s) @ u.mT

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`U_{low}\Sigma_{low}V_{low}^\top - U^0_{low}\Sigma^0_{low}V^{0\top}_{low}`, the change ``adapter`` makes to the weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the trained tensors' dtype.
        """
        u, s, v = self._trained(adapter)
        initial = (
            self._osf_U_low_init[adapter] * self._osf_S_low_init[adapter]
        ) @ self._osf_V_low_init[adapter]
        delta: Tensor = (u * s) @ v - initial
        return delta

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        u, s, v = self._guarded(adapter)
        now = self._product(x, u, s, v)
        before = self._product(
            x,
            self._osf_U_low_init[adapter],
            self._osf_S_low_init[adapter],
            self._osf_V_low_init[adapter],
        )
        return now - before


class Linear(OSFLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` whose low singular directions train: :math:`y = (W + U\Sigma V^\top - U^0\Sigma^0 V^{0\top})\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`OSFLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import OSFConfig
    >>> from lucid.peft.tuners.osf import Linear
    >>> base = nn.Linear(8, 6)
    >>> layer = Linear(base, "default", OSFConfig(effective_rank=2, target_modules=["x"]))
    >>> {name: tuple(p.shape) for name, p in layer.osf_svd_params["default"].items()}
    {'U_low': (6, 4), 'S_low': (4,), 'V_low': (4, 8)}
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # the trained part starts as the weight's own
    True
    """

    base_layer: nn.Linear
