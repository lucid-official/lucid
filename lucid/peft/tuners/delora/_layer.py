r"""The DeLoRA layers: a frozen base module beside a normalised low-rank update whose norm is bounded.

DeLoRA (Bini et al., 2025, arXiv:2503.18225, section 2.2) splits the strength of a LoRA update
from its direction.  Writing :math:`BA = \sum_i b_i a_i^\top`, it divides every rank-one component
by :math:`\lVert b_i\rVert\,\lVert a_i\rVert` (Eq. 5), so each has unit Frobenius norm, and scales
the sum by a learned boundary :math:`\lambda / r` (Eq. 7) and by the norm of the pretrained weight
(Eq. 8, 15):

.. math:: \Delta W = \frac{\lambda}{r}\,B\,\Xi\,A \cdot \mathrm{diag}(n), \qquad
          \Xi_{ii} = \frac{1}{\lVert b_i\rVert\,\lVert a_i\rVert},

with :math:`n` the column norms of :math:`W` at the time the adapter was made (``delora_w_norm``,
fixed from then on).  Then :math:`\lVert \Delta W\,\mathrm{diag}(n)^{-1}\rVert_F \le \lambda`: the
update cannot leave a ball whose radius is the learned :math:`\lambda`.  The factors start as
the reference library starts them (:math:`B = 0`), which makes the update zero without the
frozen-copy subtraction of Eq. 9.

``DeloraLayer`` is the family root; an adapter holds ``delora_A`` (``(r, in)``), ``delora_B``
(``(out, r)``), ``delora_lambda`` (``(1,)``) and the buffer ``delora_w_norm`` (``(in,)``), the
reference's keys.
"""

import math
from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.linalg import vector_norm
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._delta import _DeltaLayer
from lucid.peft.tuners._layer import _dotted, _parameter_at
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._support import _LinearSized, _ResidualUpdate, _SpecField
from lucid.peft.tuners.delora._config import DeloraConfig

#: The smallest norm a factor row or column is divided by, so a zero factor (``B`` starts at zero) gives a zero update, not NaN.
_FLOOR = 1e-4
_W_NORM = "delora_w_norm"


class _DeloraSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it.
    """

    r: int


class DeloraLayer(_DeltaLayer[DeloraConfig], _LinearSized, _ResidualUpdate, family=True):
    r"""Adapter bookkeeping and the normalised update shared by the DeLoRA layers.

    Per adapter: ``delora_A``, ``delora_B``, ``delora_lambda``, the pretrained weight's column
    norms ``delora_w_norm`` and a dropout of the input, and the read-only mapping ``r``.
    ``get_delta_weight`` is :math:`\Delta W`; the forward pass applies it to the input without
    forming it.

    Parameters
    ----------
    base_layer : nn.Module
        The ``nn.Linear`` to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : DeloraConfig
        ``r``, ``delora_lambda``, ``module_dropout`` and ``init_weights`` (the per-module patterns
        are the tuner's to resolve).

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``base_layer`` builds its parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("delora_A", "delora_B", "delora_lambda")
    _dropout_table = "delora_dropout"
    _repr_prefix = "delora."
    delora_A: nn.ParameterDict
    delora_B: nn.ParameterDict
    delora_lambda: nn.ParameterDict
    delora_w_norm: _BufferDict
    delora_dropout: nn.ModuleDict[nn.Module]

    r = _SpecField[int]("Rank of each adapter, by name (read-only).")

    @override
    def _prepare(self, config: DeloraConfig) -> None:
        super()._prepare(config)
        self.delora_w_norm = _BufferDict(persistent=True)

    @override
    def _enter(self, adapter_name: str, config: DeloraConfig) -> None:
        home = self._home()
        a = home.zeros(config.r, self.in_features)
        init.kaiming_uniform_(a, a=math.sqrt(5))
        b = home.zeros(self.out_features, config.r)
        if not config.init_weights:  # random start, for tests
            init.kaiming_uniform_(b, a=math.sqrt(5))
        boundary = home.full((1,), float(config.delora_lambda))
        with lucid.no_grad():
            norm = vector_norm(self.weight.detach().to(home.dtype), dim=0)
        self._specs[adapter_name] = _DeloraSpec(config.r)
        self.delora_A[adapter_name] = Parameter(a)
        self.delora_B[adapter_name] = Parameter(b)
        self.delora_lambda[adapter_name] = Parameter(boundary)
        self.delora_w_norm[adapter_name] = norm
        self._enter_dropout(adapter_name, config.module_dropout)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its factors, boundary, weight norms and dropout."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.delora_w_norm:
            del self.delora_w_norm[adapter_name]

    @override
    def _adapter_keys(self, adapter_name: str) -> dict[str, str]:
        keys = super()._adapter_keys(adapter_name)
        if adapter_name in self.delora_w_norm:  # a buffer table: the file holds it as the reference's does
            keys[_dotted(_W_NORM, adapter_name)] = _W_NORM
        return keys

    def _scale(self, adapter: str) -> Tensor:
        r""":math:`\lambda / (r\,\lVert a_i\rVert\,\lVert b_i\rVert)`, one per rank-one component (a norm is floored)."""
        a = _parameter_at(self.delora_A, adapter)
        b = _parameter_at(self.delora_B, adapter)
        boundary = _parameter_at(self.delora_lambda, adapter)
        spec = self._specs[adapter]
        assert isinstance(spec, _DeloraSpec)
        a_norm = lucid.clamp(vector_norm(a, dim=1), min=_FLOOR)
        b_norm = lucid.clamp(vector_norm(b, dim=0), min=_FLOOR)
        scale: Tensor = boundary / spec.r / (a_norm * b_norm)
        return scale

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`\Delta W = \frac{\lambda}{r} B\,\Xi\,A\,\mathrm{diag}(n)`.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight, in the factors' dtype.
        """
        a = _parameter_at(self.delora_A, adapter)
        b = _parameter_at(self.delora_B, adapter)
        delta: Tensor = (b * self._scale(adapter)) @ a * self.delora_w_norm[adapter]
        return delta

    @override
    def _contribution(self, adapter: str, x: Tensor) -> Tensor:
        a = _parameter_at(self.delora_A, adapter)
        b = _parameter_at(self.delora_B, adapter)
        dropped = self._dropped(adapter, x).to(a.dtype)
        hidden = ((dropped * self.delora_w_norm[adapter]) @ a.mT) * self._scale(adapter)
        out: Tensor = hidden @ b.mT
        return out


class Linear(DeloraLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` with a DeLoRA update: :math:`y = (W + \Delta W)\,x + b`.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; the other parameters are those of :class:`DeloraLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import DeloraConfig
    >>> from lucid.peft.tuners.delora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", DeloraConfig(r=2, target_modules=["x"]))
    >>> tuple(layer.delora_A["default"].shape), tuple(layer.delora_B["default"].shape)
    ((2, 8), (4, 2))
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    base_layer: nn.Linear
