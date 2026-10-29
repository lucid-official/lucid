"""The LoRA layer: a frozen ``nn.Linear`` with a low-rank update beside it."""

import math
import warnings
from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import _check_dropout, _check_init, _check_number, _check_rank

_HALF = (lucid.float16, lucid.bfloat16)


def _check_adapter_name(adapter_name: object) -> str:
    if not isinstance(adapter_name, str):
        raise TypeError(f"adapter_name must be a str, got {adapter_name!r}")
    if not adapter_name or "." in adapter_name:
        raise ValueError(f"adapter_name must be a non-empty name without '.', got {adapter_name!r}")
    return adapter_name


class LoraLayer(nn.Module):
    r"""Adapter bookkeeping shared by the LoRA layers.

    Holds the wrapped layer as ``base_layer`` — the very module, so its
    ``weight`` and ``bias`` stay the same ``Parameter`` objects (an
    optimizer, a tied head and Core ML's identity-keyed weight sharing
    all keep seeing them) — and, per adapter name, the rank ``r``,
    ``lora_alpha``, ``scaling`` and the ``lora_A`` / ``lora_B`` /
    ``lora_dropout`` modules in ``ModuleDict``\ s, so a state dict reads
    ``....lora_A.default.weight`` as the reference's does.  v1 holds one
    adapter per layer.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt.

    Raises
    ------
    TypeError
        ``base_layer`` is not an ``nn.Linear``.
    """

    def __init__(self, base_layer: nn.Linear) -> None:
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"LoRA adapts nn.Linear in lucid.peft v1, got {type(base_layer).__name__}"
            )
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r: dict[str, int] = {}
        self.lora_alpha: dict[str, float] = {}
        self.scaling: dict[str, float] = {}
        self.lora_dropout: nn.ModuleDict[nn.Dropout | nn.Identity] = nn.ModuleDict()
        self.lora_A: nn.ModuleDict[nn.Linear] = nn.ModuleDict()
        self.lora_B: nn.ModuleDict[nn.Linear] = nn.ModuleDict()
        self.merged_adapters: list[str] = []
        self._active_adapter = ""
        self._disable_adapters = False

    @property
    def weight(self) -> Parameter:
        """The base layer's weight (unchanged unless an adapter is merged into it)."""
        return self.base_layer.weight

    @property
    def bias(self) -> Parameter | None:
        """The base layer's bias, or ``None``."""
        return self.base_layer.bias

    @property
    def active_adapter(self) -> str:
        """Name of the adapter the forward pass applies."""
        return self._active_adapter

    @property
    def disable_adapters(self) -> bool:
        """Whether the forward pass runs the base layer alone."""
        return self._disable_adapters

    @property
    def merged(self) -> bool:
        """Whether an adapter is currently folded into the base weight."""
        return bool(self.merged_adapters)

    def get_base_layer(self) -> nn.Linear:
        """Return the wrapped ``nn.Linear``, whose parameters are the base model's own."""
        return self.base_layer

    def enable_adapters(self, enabled: bool) -> None:
        """Apply the adapter in the forward pass (``True``) or run the base alone (``False``).

        Parameters
        ----------
        enabled : bool
            Whether the adapter takes part.  Gradient flags are left as
            they are: a frozen adapter stays frozen when it is re-enabled.
        """
        self._disable_adapters = not enabled

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: bool | str,
        use_rslora: bool,
        use_dora: bool = False,
    ) -> None:
        r"""Create adapter ``adapter_name``: ``lora_A``, ``lora_B``, dropout and scale.

        The adapter lives on the base weight's device.  Its dtype is the
        base's, except that a float16 / bfloat16 base gets a float32
        adapter — the reference's default, which keeps the low-rank update
        from underflowing in training.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each ``ModuleDict``.
        r : int
            Rank; positive.
        lora_alpha : float
            Numerator of the scale.
        lora_dropout : float
            Dropout on the update's input, in ``[0, 1)``.
        init_lora_weights : bool or str
            ``True``, ``False`` or ``"gaussian"`` (see ``LoraConfig``).
        use_rslora : bool
            Scale by :math:`\alpha/\sqrt r` instead of :math:`\alpha/r`.
        use_dora : bool, default=False
            DoRA; ``True`` raises ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            ``use_dora=True``, or the layer already holds an adapter.
        """
        adapter_name = _check_adapter_name(adapter_name)
        r = _check_rank("r", r)
        lora_alpha = _check_number("lora_alpha", lora_alpha)
        init_lora_weights = _check_init(init_lora_weights)
        p = _check_dropout(lora_dropout)
        if use_dora:
            raise NotImplementedError("use_dora=True: DoRA is not implemented in lucid.peft v1")
        if len(self.lora_A):
            raise NotImplementedError(
                f"this layer already holds adapter {self._active_adapter!r}; "
                "lucid.peft v1 keeps one adapter per layer"
            )
        weight = self.base_layer.weight
        dtype = lucid.float32 if weight.dtype in _HALF else weight.dtype
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.lora_dropout[adapter_name] = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.lora_A[adapter_name] = nn.Linear(
            self.in_features, r, bias=False, device=weight.device, dtype=dtype
        )
        self.lora_B[adapter_name] = nn.Linear(
            r, self.out_features, bias=False, device=weight.device, dtype=dtype
        )
        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._active_adapter = adapter_name

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: bool | str) -> None:
        r"""Initialise adapter ``adapter_name`` the way ``init_lora_weights`` asks.

        ``True``: :math:`A` Kaiming-uniform with ``a = sqrt(5)`` (what
        ``nn.Linear`` uses) and :math:`B = 0`, so the update starts at
        zero.  ``"gaussian"``: :math:`A \sim \mathcal N(0, 1/r^2)`,
        :math:`B = 0`.  ``False``: leave both as ``nn.Linear`` made them.

        Parameters
        ----------
        adapter_name : str
            The adapter to initialise.
        init_lora_weights : bool or str
            ``True``, ``False`` or ``"gaussian"``.
        """
        init_lora_weights = _check_init(init_lora_weights)
        if init_lora_weights is False:
            return
        lora_a = self.lora_A[adapter_name].weight
        if init_lora_weights == "gaussian":
            init.normal_(lora_a, std=1 / self.r[adapter_name])
        else:
            init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        init.zeros_(self.lora_B[adapter_name].weight)


class Linear(LoraLayer):
    r"""An ``nn.Linear`` with a LoRA update: :math:`y = W x + b + s\,B A\,\mathrm{drop}(x)`.

    The base layer runs as it is and stays frozen; the update is computed
    in the adapter's dtype and added in the wider of the two dtypes before
    the result is cast back to the base output's dtype.  ``merge``
    folds :math:`s B A` into :math:`W` in place, so a merged layer costs
    what the base costs; ``unmerge`` takes it out again.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in ``lora_A`` / ``lora_B``.
    r : int, default=0
        Rank.  The default is the reference's and is refused: pass a
        positive rank.
    lora_alpha : float, default=1
        Numerator of the scale.
    lora_dropout : float, default=0.0
        Dropout on the update's input.
    fan_in_fan_out : bool, default=False
        Must be ``False``: ``nn.Linear`` stores its weight ``(out, in)``.
    init_lora_weights : bool or str, default=True
        See ``LoraConfig``.
    use_rslora : bool, default=False
        Scale by :math:`\alpha/\sqrt r`.
    use_dora : bool, default=False
        DoRA; ``True`` raises.

    Raises
    ------
    ValueError
        ``r`` is not positive or ``fan_in_fan_out`` is ``True``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", r=2, lora_alpha=4)
    >>> layer.base_layer is base, layer.scaling["default"]
    (True, 2.0)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        r: int = 0,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
        if fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); nn.Linear stores (out, in)"
            )
        super().__init__(base_layer)
        self.update_layer(
            adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora
        )

    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,B A`, the update ``adapter`` adds to the weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(out_features, in_features)``, in the adapter's dtype.
        """
        return (self.lora_B[adapter].weight @ self.lora_A[adapter].weight) * self.scaling[adapter]

    def _shifted_weight(self, adapter: str, sign: float) -> Tensor:
        """The base weight plus ``sign`` times ``adapter``'s update, in the weight's dtype.

        Summed in the wider dtype, so a bfloat16 weight is rounded once.
        """
        weight = self.base_layer.weight
        shifted = weight + self.get_delta_weight(adapter) * sign
        return shifted if shifted.dtype == weight.dtype else shifted.to(weight.dtype)

    def _check_finite_merge(self, adapter: str) -> None:
        with lucid.no_grad():
            finite = bool(lucid.isfinite(self._shifted_weight(adapter, 1.0)).all().item())
        if not finite:
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter {adapter} seems to be broken"
            )

    def _pending_merges(self, adapter_names: list[str] | None) -> list[str]:
        """The requested adapters (the active one by default) not merged yet."""
        names = [self._active_adapter] if adapter_names is None else list(adapter_names)
        unknown = [name for name in names if name not in self.lora_A]
        if unknown:
            raise ValueError(f"unknown adapter(s) {unknown}; this layer holds {list(self.lora_A)}")
        return [name for name in names if name not in self.merged_adapters]

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        r"""Fold the update into the base weight, in place.

        The weight keeps its identity — it is written, not replaced — so
        whatever holds it (an optimizer, a tied embedding) sees the merged
        values.  That is also why a weight shared with another module must
        not be merged into; ``LoraModel.merge_adapter`` and
        ``merge_and_unload`` refuse it, as this layer alone cannot see the
        rest of the model.

        Parameters
        ----------
        safe_merge : bool, default=False
            Check the merged weight for NaN / inf first and leave the base
            untouched if there are any.
        adapter_names : list of str, optional
            Adapters to merge; the active one by default.

        Raises
        ------
        ValueError
            An unknown adapter name, or ``safe_merge`` found a non-finite
            merged weight.
        """
        pending = self._pending_merges(adapter_names)
        if not pending:
            warnings.warn(f"Already merged: {self.merged_adapters}. Nothing to do.", stacklevel=2)
        if safe_merge:
            for adapter in pending:
                self._check_finite_merge(adapter)
        with lucid.no_grad():
            for adapter in pending:
                self.base_layer.weight.copy_(self._shifted_weight(adapter, 1.0))
                self.merged_adapters.append(adapter)

    def unmerge(self) -> None:
        """Take every merged update back out of the base weight, in place.

        The result equals the original weight up to the rounding of the
        round trip.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.", stacklevel=2)
            return
        with lucid.no_grad():
            while self.merged_adapters:
                adapter = self.merged_adapters.pop()
                self.base_layer.weight.copy_(self._shifted_weight(adapter, -1.0))

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self._disable_adapters and self.merged:
            self.unmerge()
        result = self.base_layer(x)
        if self._disable_adapters or self.merged:
            return result
        adapter = self._active_adapter
        lora_a = self.lora_A[adapter]
        h = x if x.dtype == lora_a.weight.dtype else x.to(lora_a.weight.dtype)
        update = self.lora_B[adapter](lora_a(self.lora_dropout[adapter](h))) * self.scaling[adapter]
        out = result + update
        return out if out.dtype == result.dtype else out.to(result.dtype)

    @override
    def __repr__(self) -> str:
        return "lora." + super().__repr__()
