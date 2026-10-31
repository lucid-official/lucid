"""LoRA on ``nn.Embedding``: the update is a low-rank product of two lookup tables."""

import warnings
from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.functional import embedding
from lucid.nn.parameter import Parameter
from lucid.peft.tuners.lora._layer import _Extras, _Factors, _Place, _Residual
from lucid.peft.tuners.lora._variants import MAGNITUDE_TABLE


class Embedding(_Residual, base_type=nn.Embedding):
    r"""An ``nn.Embedding`` with a LoRA update: :math:`y = E[x] + s\,(A^\top[x])\,B^\top`.

    For an embedding table :math:`E \in \mathbb{R}^{V \times d}` the factors
    are :math:`A \in \mathbb{R}^{r \times V}` and
    :math:`B \in \mathbb{R}^{d \times r}`, kept as bare parameters in
    ``lora_embedding_A`` / ``lora_embedding_B`` (``....lora_embedding_A.default``
    in a state dict, as in the reference).  The update looks the token up in
    :math:`A^\top` — with the base layer's ``padding_idx``, ``max_norm``,
    ``norm_type`` and ``scale_grad_by_freq`` — and projects it with
    :math:`B^\top`.  Merging adds :math:`s\,(BA)^\top` to the table.  With
    ``use_dora`` each embedding dimension carries a magnitude (a vector of
    ``embedding_dim``, normed over the vocabulary); ``lora_bias`` is not
    available on a lookup.

    The default initialisation is the reference's: :math:`A = 0` and
    :math:`B \sim \mathcal N(0, 1)`, so the update starts at zero;
    ``init_lora_weights=False`` leaves both normal.  ``lora_dropout`` has no
    place in a lookup and is not applied (a warning says so).

    Parameters
    ----------
    base_layer : nn.Embedding
        The layer to adapt; kept as ``base_layer``, its table shared.
    adapter_name : str
        Name of the adapter, the key in ``lora_embedding_A`` / ``lora_embedding_B``.
    r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
        The adapter's hyper-parameters, as ``LoraLayer.update_layer`` takes
        them; ``r`` must be positive.  ``lora_dropout`` has no activations to
        act on in a lookup and is not applied (a nonzero value warns).  Only
        ``init_lora_weights`` of ``True``, ``False`` and ``"gaussian"`` apply
        (the last has no meaning other than the first here).

    Raises
    ------
    ValueError
        ``r`` is not positive, ``lora_bias`` is set, or ``init_lora_weights``
        factorises the weight (PiSSA, OLoRA).

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import Embedding
    >>> base = nn.Embedding(10, 4)
    >>> layer = Embedding(base, "default", r=2, lora_alpha=4)
    >>> tuple(layer.lora_embedding_A["default"].shape), tuple(layer.lora_embedding_B["default"].shape)
    ((2, 10), (4, 2))
    >>> ids = lucid.tensor([[1, 3], [7, 0]])
    >>> bool((layer(ids) == base(ids)).all().item())   # A starts at zero
    True
    """

    adapter_layer_names = ("lora_embedding_A", "lora_embedding_B", MAGNITUDE_TABLE)
    base_layer: nn.Embedding

    @override
    def _setup(self) -> None:
        self.in_features = self.base_layer.num_embeddings
        self.out_features = self.base_layer.embedding_dim
        self.lora_embedding_A = nn.ParameterDict()
        self.lora_embedding_B = nn.ParameterDict()

    @override
    def _anchor(self) -> Tensor:
        return self.base_layer.weight

    @override
    def _check_dropout(self, p: float) -> None:
        if p > 0:
            warnings.warn(
                "lora_dropout is not applied by lora.Embedding: a lookup has no activations to drop",
                stacklevel=2,
            )

    @override
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        return (
            Parameter(lucid.randn(r, self.in_features, device=place.device, dtype=place.dtype)),
            Parameter(lucid.randn(self.out_features, r, device=place.device, dtype=place.dtype)),
        )

    def _pair(self, adapter: str) -> tuple[Parameter, Parameter]:
        lora_a, lora_b = self.lora_embedding_A[adapter], self.lora_embedding_B[adapter]
        if lora_a is None or lora_b is None:
            raise KeyError(f"adapter {adapter!r} holds no embedding factors")
        return lora_a, lora_b

    @override
    def _reset_factors(self, adapter: str, gaussian: bool) -> None:
        # ``True`` and ``"gaussian"`` alike: A = 0 and B ~ N(0, 1), the way ``nn.Embedding`` initialises.
        del gaussian
        lora_a, lora_b = self._pair(adapter)
        init.zeros_(lora_a)
        init.normal_(lora_b)

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,(BA)^\top`, the update ``adapter`` adds to the table.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(num_embeddings, embedding_dim)``, in the adapter's dtype.
        """
        lora_a, lora_b = self._pair(adapter)
        return (lora_b @ lora_a).mT * self._specs[adapter].scaling

    @override
    def _dropped(self, adapter: str, x: Tensor) -> Tensor:
        return x  # a lookup has no activations to drop

    @override
    def _applies_dropout(self, adapter: str) -> bool:
        return False

    @override
    def _rows(self, weight: Tensor) -> Tensor:
        # The table is (vocabulary, features); a feature is an output unit, so it is a row of the transpose.
        return weight.mT

    @override
    def _lowrank(self, adapter: str, dropped: Tensor, extras: _Extras) -> Tensor:
        del extras  # a lookup takes no arguments beside its indices
        lora_a, lora_b = self._pair(adapter)
        base = self.base_layer
        after_a = embedding(
            dropped,
            lora_a.mT,
            padding_idx=base.padding_idx,
            max_norm=base.max_norm,
            norm_type=base.norm_type,
            scale_grad_by_freq=base.scale_grad_by_freq,
            sparse=base.sparse,
        )
        return (after_a @ lora_b.mT) * self._specs[adapter].scaling
