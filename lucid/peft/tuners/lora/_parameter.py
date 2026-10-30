"""LoRA on one raw parameter of a module: ``target_parameters``."""

from typing import override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners.lora._layer import _Factors, _Place
from lucid.peft.tuners.lora._weight_space import _WeightSpace


# Adapts a parameter, not a module of some type, so it joins no ``base_type`` table:
# the tuner builds it for a parameter target.
class ParamWrapper(_WeightSpace[object]):
    r"""A LoRA update on one ``nn.Parameter`` of a module, for ``target_parameters``.

    The wrapped module is any module that reads the parameter in its own
    forward; the wrapper stands where the module was and calls it with
    :math:`W + s\,BA` read in place of :math:`W` (see ``_WeightSpace``).
    Reading the parameter directly (``wrapper.weight``) gives the base
    value, the update not applied.  The module's other attributes read
    through to it.

    A parameter is ``(out, in)`` — one matrix — or ``(experts, out, in)``,
    a stack with an adapter pair per expert.  A stack whose module sets
    ``is_transposed = True`` is read as ``(experts, in, out)``.  Each
    wrapper adapts one parameter; a module with several adapted parameters
    is wrapped once per parameter, nested, the outermost standing in the
    model.  An adapter file keys the inner wrappers under
    ``....base_layer.lora_A.weight``.

    Parameters
    ----------
    base_layer : nn.Module
        The module holding the parameter (or the wrapper around it, for a
        second parameter); the parameter stays the base model's own.
    adapter_name : str
        Name of the adapter, the key in ``lora_A`` / ``lora_B``.
    parameter_name : str
        Attribute name of the parameter on the module.
    r, lora_alpha, init_lora_weights, use_rslora, use_dora
        The adapter's hyper-parameters, as ``LoraLayer.update_layer`` takes
        them; ``r`` must be positive.
    lora_dropout : float, default=0.0
        Must be 0: the update is folded into the weight, so there is no
        input to drop.

    Raises
    ------
    ValueError
        ``r`` is not positive, ``lora_dropout`` is not 0, the parameter is
        missing or is not 2-D or 3-D.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import ParamWrapper
    >>> class Gate(nn.Module):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.w = nn.Parameter(lucid.randn(4, 6))
    ...     def forward(self, x):
    ...         return x @ self.w.mT
    >>> gate = Gate()
    >>> layer = ParamWrapper(gate, "default", "w", r=2, lora_alpha=4)
    >>> layer.get_base_layer() is gate, tuple(layer.lora_A["default"].weight.shape)
    (True, (2, 6))
    >>> x = lucid.randn(3, 6)
    >>> bool((layer(x) == gate(x)).all().item())   # B starts at zero
    True
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        parameter_name: str,
        r: int = 0,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
        # Plain attributes may be set before ``Module.__init__``; ``_setup`` reads this one.
        self.parameter_name = parameter_name
        super().__init__(
            base_layer,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            use_rslora,
            use_dora,
        )

    @override
    def get_base_layer(self) -> nn.Module:
        """Return the module holding the parameter, through any wrappers nested around it."""
        base = self.base_layer
        return base.get_base_layer() if isinstance(base, ParamWrapper) else base

    @override
    def _base_path(self) -> str:
        base = self.base_layer
        inner = f".{base._base_path()}" if isinstance(base, ParamWrapper) else ""
        return f"base_layer{inner}"

    @override
    def _unwrap(self, keep_trained: bool) -> nn.Module:
        base = self.base_layer
        if isinstance(base, ParamWrapper):
            if self.merged and not keep_trained:
                self.unmerge()
            return base._unwrap(keep_trained)
        return super()._unwrap(keep_trained)

    @override
    def extra_repr(self) -> str:
        return f"parameter_name={self.parameter_name!r}, " + super().extra_repr()

    def _parameter(self) -> Parameter:
        found = getattr(self.get_base_layer(), self.parameter_name, None)
        if isinstance(found, Parameter):
            return found
        raise ValueError(
            f"{type(self.get_base_layer()).__name__} has no parameter "
            f"{self.parameter_name!r} to adapt"
        )

    @override
    def _setup(self) -> None:
        param = self._parameter()
        if param.ndim not in (2, 3):
            raise ValueError(
                f"lora.ParamWrapper adapts a 2-D weight or a 3-D stack of them, but "
                f"{self.parameter_name!r} is {param.ndim}-D"
            )
        # A 3-D stack is (experts, out, in), or (experts, in, out) if its module says so.
        transposed = bool(getattr(self.get_base_layer(), "is_transposed", False))
        self._param_ndim = param.ndim
        self.num_experts = 1 if param.ndim == 2 else param.shape[0]
        self._out_first = param.ndim == 2 or not transposed
        rows, cols = param.shape[-2], param.shape[-1]
        self.out_features, self.in_features = (rows, cols) if self._out_first else (cols, rows)
        self.lora_A: nn.ModuleDict[nn.Linear] = nn.ModuleDict()
        self.lora_B: nn.ModuleDict[nn.Linear] = nn.ModuleDict()

    @override
    def _check_dropout(self, p: float) -> None:
        if p > 0:
            raise ValueError("lora.ParamWrapper does not work with lora_dropout != 0.")

    @override
    def _weights(self) -> dict[str, Parameter]:
        return {self.parameter_name: self._parameter()}

    @override
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        rank = r * self.num_experts
        return (place.linear(self.in_features, rank), place.linear(rank, self.out_features))

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,BA` for every expert, shaped like the parameter.

        With experts the factors are the reference's: ``lora_A`` holds the
        experts' :math:`A_e` stacked along its rows (rank-major within an
        expert), ``lora_B`` holds the :math:`B_e` along its columns (expert
        fastest).

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            The parameter's shape, in the adapter's dtype.
        """
        lora_a, lora_b = self.lora_A[adapter].weight, self.lora_B[adapter].weight
        experts = self.num_experts
        a = lora_a.reshape(experts, -1, self.in_features)  # (E, r, in)
        b = lora_b.reshape(self.out_features, -1, experts).permute([2, 0, 1])  # (E, out, r)
        stack = b @ a  # (E, out, in)
        scaling = self._specs[adapter].scaling
        if self._param_ndim == 2:
            return stack[0] * scaling
        return (stack if self._out_first else stack.mT) * scaling

    @override
    def _delta_for(self, name: str, adapter: str) -> Tensor:
        del name  # the one parameter this wrapper adapts
        return self.get_delta_weight(adapter)
