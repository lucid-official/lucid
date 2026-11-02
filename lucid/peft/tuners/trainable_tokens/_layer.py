r"""The trainable-tokens layers: a frozen embedding table beside a small copy of some of its rows.

``TrainableTokensLayer`` is the family root.  An adapter holds ``trainable_tokens_delta``
(``(K, dim)``, a ``ParameterDict``: the rows being trained, starting as the table's) and
remembers the table's own rows in ``trainable_tokens_original`` (a buffer table that is not
saved).  A state dict reads ``....trainable_tokens_delta.default`` and an adapter file
``....trainable_tokens_delta``, the reference's key; the token ids are in the config.

The layer computes with the table whose rows at the adapter's ids are replaced by the trained
ones — values, not differences, so a table resized and re-initialised in between cannot
turn the trained rows into garbage — and merging writes those rows into the base table.
"""

from typing import ClassVar, NamedTuple, override

import lucid
import lucid.nn.functional as F
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._shared import _BufferDict
from lucid.peft.tuners._support import _config_as, _ConfigLayer, _SpecField, _WeightRewrite
from lucid.peft.tuners.trainable_tokens._config import TrainableTokensConfig


class _TokensSpec(NamedTuple):
    """What a layer remembers of one adapter besides its tensors.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes it: the ids pick
    the rows the graph rewrites.
    """

    token_indices: tuple[int, ...]


class TrainableTokensLayer(_ConfigLayer, _WeightRewrite, family=True):
    r"""Adapter bookkeeping and the row replacement shared by the trainable-tokens layers.

    Per adapter: ``trainable_tokens_delta`` (the trained rows), ``trainable_tokens_original``
    (the table's rows when the adapter was made) and the read-only mapping ``token_indices``.
    Several adapters on one layer must name disjoint tokens.

    Parameters
    ----------
    base_layer : nn.Module
        The embedding (or tied output head) to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : TrainableTokensConfig
        The adapter's config; its ``token_indices`` are this layer's.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``token_indices`` is empty or names a row the table lacks, or ``base_layer`` builds
        parameters on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("trainable_tokens_delta",)
    _repr_prefix = "trainable_tokens."
    trainable_tokens_delta: nn.ParameterDict
    trainable_tokens_original: _BufferDict

    token_indices = _SpecField[tuple[int, ...]](
        "The token ids each adapter trains, by name (read-only)."
    )

    @override
    def _prepare(self, config: _TargetedConfig) -> None:
        del config  # nothing here depends on it
        self.trainable_tokens_original = _BufferDict(persistent=False)

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(TrainableTokensConfig, config)
        rows = int(self.weight.shape[0])
        if not cfg.token_indices:
            raise ValueError("`token_indices` is empty: name the tokens to train")
        if max(cfg.token_indices) >= rows:
            raise ValueError(
                f"`token_indices` names token {max(cfg.token_indices)}, but the table has {rows} rows"
            )
        weight = self.weight
        index = lucid.tensor(cfg.token_indices, dtype=lucid.int64, device=weight.device)
        originals = weight[index].detach().clone()
        values = originals.clone() if cfg.init_weights else lucid.randn_like(originals)
        self._specs[adapter_name] = _TokensSpec(tuple(cfg.token_indices))
        self.trainable_tokens_delta[adapter_name] = Parameter(values)
        self.trainable_tokens_original[adapter_name] = originals

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove ``adapter_name``: its trained rows and the originals it kept."""
        super().delete_adapter(adapter_name)
        if adapter_name in self.trainable_tokens_original:
            del self.trainable_tokens_original[adapter_name]

    # ── the rows an adapter replaces ──────────────────────────────────────

    def _rows(self, adapter: str) -> Tensor:
        spec = self._specs[adapter]
        assert isinstance(spec, _TokensSpec)
        return lucid.tensor(spec.token_indices, dtype=lucid.int64, device=self.weight.device)

    def _check_overlap(self, adapters: list[str]) -> None:
        """Raise ``ValueError`` if two of ``adapters`` (and the merged ones) name a token in common."""
        seen: set[int] = set()
        for name in dict.fromkeys([*self.merged_adapters, *adapters]):
            spec = self._specs[name]
            assert isinstance(spec, _TokensSpec)
            ids = set(spec.token_indices)
            if seen & ids:
                raise ValueError(
                    f"Token indices of adapter {name} are already defined and would result in "
                    "undefined merging behavior. Only disjunct token indices are currently "
                    "supported."
                )
            seen |= ids

    def _table_with(self, adapters: list[str]) -> Tensor:
        """The base table with each of ``adapters``' rows replaced by the trained ones."""
        table: Tensor = self.weight
        for name in adapters:
            delta = _parameter_at(self.trainable_tokens_delta, name)
            table = table.index_copy(0, self._rows(name), delta.to(table.dtype))
        return table

    @override
    def _rewritten(self, weight: Tensor, adapter: str) -> Tensor:
        delta = _parameter_at(self.trainable_tokens_delta, adapter)
        return weight.index_copy(0, self._rows(adapter), delta.to(weight.dtype))

    @override
    def _reverted(self, weight: Tensor, adapter: str) -> Tensor:
        original = self.trainable_tokens_original[adapter]
        return weight.index_copy(0, self._rows(adapter), original.to(weight.dtype))

    @override
    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """Write the active adapters' rows into the table.

        Parameters
        ----------
        safe_merge : bool, default=False
            Check the merged table for NaN / inf first and leave the base
            untouched if there are any.
        adapter_names : list of str, optional
            Adapters to merge; the active ones this layer holds by default.

        Raises
        ------
        ValueError
            Two of the adapters (or one already merged) train the same token,
            or as for the base layer's ``merge``.
        """
        self._check_overlap(self._pending_merges(adapter_names))
        super().merge(safe_merge, adapter_names)

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return the change merging ``adapter`` makes to the table: trained minus original rows, zero elsewhere.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            Shaped like the base weight.
        """
        delta = _parameter_at(self.trainable_tokens_delta, adapter)
        change = delta - self.trainable_tokens_original[adapter].to(delta.dtype)
        return lucid.zeros_like(self.weight).index_copy(
            0, self._rows(adapter), change.to(self.weight.dtype)
        )

    # ── the forward pass ──────────────────────────────────────────────────

    def _compute(self, x: Tensor, table: Tensor) -> Tensor:
        """The base module's computation with ``table`` as its weight."""
        raise NotImplementedError(
            f"{type(self).__name__} wraps a module with no known forward pass"
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Run the layer on the table with the active adapters' rows in place.

        The original table when adapters are disabled, merged or none applies; two adapters
        that train the same token raise ``ValueError``.
        """
        adapters = [name for name in self.active_adapters if name in self._specs]
        if self.disable_adapters or self.merged or not adapters:
            result: Tensor = self.base_layer(x)
            return result
        self._check_overlap(adapters)
        return self._compute(x, self._table_with(adapters))


class Embedding(TrainableTokensLayer, base_type=nn.Embedding):
    r"""An ``nn.Embedding`` whose chosen rows are trained: ``F.embedding(x, E')``.

    Parameters
    ----------
    base_layer : nn.Embedding
        The table to adapt; the other parameters are those of :class:`TrainableTokensLayer`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.trainable_tokens import Embedding, TrainableTokensConfig
    >>> table = nn.Embedding(10, 4)
    >>> layer = Embedding(table, "default", TrainableTokensConfig(token_indices=[2, 5], target_modules=["x"]))
    >>> tuple(layer.trainable_tokens_delta["default"].shape)
    (2, 4)
    >>> ids = lucid.tensor([[1, 2, 5]])
    >>> bool((layer(ids) == table(ids)).all().item())   # the rows start as the table's
    True
    """

    base_layer: nn.Embedding

    @override
    def _compute(self, x: Tensor, table: Tensor) -> Tensor:
        base = self.base_layer
        return F.embedding(
            x,
            table,
            padding_idx=base.padding_idx,
            max_norm=base.max_norm,
            norm_type=base.norm_type,
            scale_grad_by_freq=base.scale_grad_by_freq,
            sparse=base.sparse,
        )


class Linear(TrainableTokensLayer, base_type=nn.Linear):
    r"""An ``nn.Linear`` whose weight rows are trained: an output head tied to the embedding, ``F.linear(x, E')``.

    Parameters
    ----------
    base_layer : nn.Linear
        The head to adapt; the other parameters are those of :class:`TrainableTokensLayer`.
    """

    base_layer: nn.Linear

    @override
    def _compute(self, x: Tensor, table: Tensor) -> Tensor:
        return F.linear(x, table, self.bias)
