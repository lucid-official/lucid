"""Cross-task shared utilities used by multiple model families.

These helpers are task-agnostic and may be imported by any model
regardless of its output task (classification, detection, segmentation …).
"""

import random
from collections.abc import Mapping, Sized
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import lucid
import lucid.nn as nn
import lucid.optim as optim
import lucid.nn.functional as F

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from lucid._tensor.tensor import Tensor


def apply_overrides[C: DataclassInstance](cfg: C, overrides: Mapping[str, object]) -> C:
    """Merge a factory's ``**overrides`` into its frozen default config.

    Every registered factory takes untyped ``**overrides`` and folds them
    into a frozen dataclass config.  This is the one place that merge
    happens, so no factory has to launder the keyword mapping's type.

    Parameters
    ----------
    cfg : dataclass instance
        The factory's default config.
    overrides : Mapping[str, object]
        Field replacements.  An unknown key raises ``TypeError`` from
        :func:`dataclasses.replace`.

    Returns
    -------
    dataclass instance
        ``cfg`` itself when ``overrides`` is empty, otherwise a new config
        of the same type with the fields replaced.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from lucid.models._utils._common import apply_overrides
    >>> @dataclass(frozen=True)
    ... class Cfg:
    ...     width: int = 8
    >>> base = Cfg()
    >>> apply_overrides(base, {}) is base
    True
    >>> apply_overrides(base, {"width": 16})
    Cfg(width=16)
    """
    return replace(cfg, **overrides) if overrides else cfg


def check_stage_lengths(owner: str, **per_stage: Sized) -> None:
    """Refuse per-stage config tuples that disagree on the stage count.

    A builder walks these fields together, one entry per stage.  Checked in
    the config, a short or long tuple is named where it was written rather
    than surfacing as a ``zip`` length error, or a head built for a stage
    that never ran, inside the model.

    Parameters
    ----------
    owner : str
        The config class, named in the message.
    **per_stage : Sized
        Each per-stage field by its config name.

    Raises
    ------
    ValueError
        When the fields do not all have the same length.

    Examples
    --------
    >>> from lucid.models._utils._common import check_stage_lengths
    >>> check_stage_lengths("Cfg", depths=(1, 1), dims=(8, 16))
    >>> check_stage_lengths("Cfg", depths=(1, 1), dims=(8, 16, 32))
    Traceback (most recent call last):
        ...
    ValueError: Cfg: depths, dims hold one entry per stage and must be the same length; got depths=2, dims=3
    """
    lengths = {name: len(value) for name, value in per_stage.items()}
    if len(set(lengths.values())) > 1:
        got = ", ".join(f"{name}={n}" for name, n in lengths.items())
        raise ValueError(
            f"{owner}: {', '.join(lengths)} hold one entry per stage and must be "
            f"the same length; got {got}"
        )


def make_divisible(
    v: float,
    divisor: int = 8,
    min_value: int | None = None,
) -> int:
    """Round ``v`` to the nearest multiple of ``divisor``.

    The result is at least ``min_value`` (falls back to ``divisor`` when
    *None*).  The 0.9 × v guard prevents the value from being rounded
    down excessively — if the adjusted value is more than 10 % below ``v``
    an extra ``divisor`` is added.

    This is the canonical implementation used by MobileNet, EfficientNet,
    SE-ResNet, SK-ResNet, ResNeSt and any other family that needs channel
    counts aligned to a power-of-two-friendly grid.

    Parameters
    ----------
    v : float
        Raw (possibly non-integer) channel count to round.
    divisor : int, optional
        Alignment granularity.  Default ``8``.
    min_value : int, optional
        Hard lower bound on the result.  When ``None`` (default), falls
        back to ``divisor``.

    Returns
    -------
    int
        Channel count rounded to a multiple of ``divisor``, never below
        ``min_value`` / ``divisor``.

    Examples
    --------
    >>> from lucid.models._utils._common import make_divisible
    >>> make_divisible(33, divisor=8)
    32
    >>> make_divisible(35, divisor=8)
    32
    >>> make_divisible(3, divisor=8, min_value=8)
    8
    """
    min_val: int = min_value if min_value is not None else divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def transform_input_imagenet_to_tf(x: Tensor) -> Tensor:
    r"""Re-normalise ImageNet-standardised input into the ``(x-0.5)/0.5`` range.

    The Inception-v3 and GoogLeNet ImageNet checkpoints are ports of the
    original TensorFlow weights, which were trained on inputs scaled to
    :math:`[-1, 1]`.  Callers, however, feed ImageNet-standardised images —
    that is what the families' bundled preprocessing presets emit — so the
    channel statistics have to be converted inside the model:

    .. math::

        x'_c = x_c \cdot \frac{\sigma_c}{0.5} + \frac{\mu_c - 0.5}{0.5}

    with the usual ImageNet :math:`\mu = (0.485, 0.456, 0.406)` and
    :math:`\sigma = (0.229, 0.224, 0.225)`.  Skipping it leaves every channel
    off by a factor of ~2.2 in scale plus an offset, a large distribution
    shift for a BatchNorm network and a silent drop in top-1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import (
    ...     transform_input_imagenet_to_tf,
    ... )
    >>> transform_input_imagenet_to_tf(lucid.zeros(1, 3, 4, 4)).shape
    (1, 3, 4, 4)

    Same shape, different normalisation: weights converted from a
    TensorFlow checkpoint expect inputs in [-1, 1] rather than the
    ImageNet mean and standard deviation the rest of the zoo uses.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    chans = [
        x[:, c : c + 1] * (std[c] / 0.5) + ((mean[c] - 0.5) / 0.5) for c in range(3)
    ]
    return lucid.cat(chans, dim=1)


def init_cnn_fan_out(model: nn.Module, *, linear_std: float | None = None) -> None:
    r"""Reference CNN initialisation: He/MSRA fan-out convs, unit BN, small FC.

    ``kaiming_normal_(mode="fan_out", nonlinearity="relu")`` on every
    convolution, ones/zeros on every norm layer, and — when ``linear_std`` is
    given — ``N(0, linear_std²)`` with zero bias on every ``Linear`` (the
    MobileNet convention; leave it *None* to keep the framework default for
    fully-connected layers).

    Lucid's own ``Conv2d.reset_parameters`` falls back to
    ``kaiming_uniform(a=sqrt(5))``, which is a *different distribution with a
    different gain* — it is the generic framework default, not the one the
    ResNet/VGG/MobileNet line of papers trains with.  Starting from it changes
    the early-training trajectory of any from-scratch run.

    Parameters
    ----------
    model : nn.Module
        Module to initialise in place.  Every submodule is visited.
    linear_std : float or None, optional, keyword-only, default=None
        Standard deviation for ``Linear`` weights.  ``None`` leaves fully
        connected layers at the framework default; pass a value (0.01 in
        MobileNet) to draw them from ``N(0, linear_std^2)`` with zero bias.

    Returns
    -------
    None
        ``model`` is modified in place.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._common import init_cnn_fan_out
    >>> layer = nn.Linear(4, 4)
    >>> init_cnn_fan_out(layer) is None
    True

    Rewrites the module's parameters where they sit and returns nothing,
    so it is called for its effect rather than assigned.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
        ):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear) and linear_std is not None:
            nn.init.normal_(m.weight, mean=0.0, std=linear_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def zero_init_last_bn(
    model: nn.Module, block_types: tuple[type, ...], attr: str = "bn3"
) -> None:
    r"""Zero the final BatchNorm gamma of every residual block.

    "Zero-initialize the last BN in each residual branch, so the block starts
    as an identity" (He et al., *Bag of Tricks*, and the reference ResNet's
    ``zero_init_residual``).  Improves early-training stability and is what the
    ``zero_init_residual`` config flag is supposed to switch on.

    Parameters
    ----------
    model : nn.Module
        Module to initialise in place.
    block_types : tuple of type
        Residual-block classes to match; a submodule is treated as a block
        when it is an instance of any of them.
    attr : str, optional, default="bn3"
        Attribute name of the block's final normalisation layer.  Blocks
        that do not carry it, or whose layer has no weight, are skipped.

    Returns
    -------
    None
        ``model`` is modified in place.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._common import zero_init_last_bn
    >>> zero_init_last_bn(nn.Linear(4, 4), ()) is None
    True

    Zeroing the last norm of each residual block makes the block start
    as the identity, so a deep network begins as a shallow one and grows
    into its depth instead of fighting it.
    """
    for m in model.modules():
        if isinstance(m, block_types):
            bn = getattr(m, attr, None)
            if bn is not None and getattr(bn, "weight", None) is not None:
                nn.init.zeros_(bn.weight)


def init_transformer_trunc_normal(
    model: nn.Module,
    std: float = 0.02,
    *,
    init_tokens: bool = True,
    zero_head: str | None = None,
) -> None:
    r"""Reference ViT-lineage initialisation: ``trunc_normal_(std)`` + zero bias.

    Every ``Linear`` (and ``Conv2d`` acting as a patch embedding) is drawn from
    a truncated normal at ``std``, biases are zeroed, and LayerNorms are left at
    unit weight / zero bias.  ViT, Swin, ConvNeXt, CvT, PVT, InceptionNeXt,
    EfficientFormer and MaxViT all specify this; without it they start from
    Lucid's generic ``kaiming_uniform(a=sqrt(5))``, whose scale is far larger
    and which the papers' LR schedules are not tuned for.

    Parameters
    ----------
    model : nn.Module
        Module to initialise in place.
    std : float, optional, default=0.02
        Standard deviation of the truncated normal.
    init_tokens : bool, optional, keyword-only, default=True
        Also draw bare ``Parameter`` tokens — class tokens, positional
        tables, distillation tokens — from the same distribution.  They are
        not owned by any layer, so the framework default never touches them
        and they would otherwise start at exact zeros.
    zero_head : str or None, optional, keyword-only, default=None
        Attribute name of a classification head to zero after
        initialisation, matching the papers' fine-tuning recipe.  ``None``
        leaves every head at its drawn value.

    Returns
    -------
    None
        ``model`` is modified in place.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._common import (
    ...     init_transformer_trunc_normal,
    ... )
    >>> init_transformer_trunc_normal(nn.Linear(4, 4)) is None
    True

    Truncated rather than plain normal: the tail of an untruncated draw
    puts a few weights far enough out to dominate the first steps, which
    a transformer's residual path then carries all the way up.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    if init_tokens:
        # Class tokens and positional tables are bare Parameters, so the module
        # sweep above never reaches them.  Left at exact zeros (a common
        # oversight) a CLS token carries no signal to break symmetry with, and
        # every reference draws these from the same truncated normal.
        marks = (
            "cls_token",
            "class_token",
            "pos_embed",
            "position_embed",
            "dist_token",
        )
        for name, prm in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if any(mk in leaf for mk in marks):
                nn.init.trunc_normal_(prm, std=std)

    if zero_head is not None:
        # ViT and its descendants zero the classification head so training
        # starts from a uniform posterior.
        for name, prm in model.named_parameters():
            if name.startswith(zero_head):
                nn.init.zeros_(prm)


def reject_unavailable_pretrained(factory_name: str, *, alternative: str = "") -> None:
    """Refuse ``pretrained=True`` where no weights exist for that factory.

    ``PretrainedModel.from_pretrained`` resolves a registered name by calling
    ``factory(pretrained=True)`` and returns whatever comes back — it never
    checks that weights were actually loaded.  A factory that accepts the flag
    and ignores it therefore hands back a randomly initialised model that the
    caller believes is pretrained, which is a worse failure than refusing:
    it is silent, and it surfaces as mysteriously poor accuracy rather than
    as an error.

    Parameters
    ----------
    factory_name : str
        Name used in the message, so the caller knows which call to change.
    alternative : str, optional
        A sibling factory that *does* publish weights, mentioned in the
        message when given.

    Examples
    --------
    >>> from lucid.models._utils._common import reject_unavailable_pretrained
    >>> reject_unavailable_pretrained("foo_backbone", alternative="foo_cls")
    Traceback (most recent call last):
        ...
    NotImplementedError: No pretrained weights are published for ``foo_backbone``; ``foo_backbone(pretrained=True)`` cannot be honoured.  Use ``foo_cls`` for the checkpointed variant.

    Refusing beats handing back a randomly initialised model that claims
    to be pretrained, and naming the alternative saves the caller from
    guessing which entry does have weights.
    """
    hint = (
        f"  Use ``{alternative}`` for the checkpointed variant." if alternative else ""
    )
    raise NotImplementedError(
        f"No pretrained weights are published for ``{factory_name}``; "
        f"``{factory_name}(pretrained=True)`` cannot be honoured.{hint}"
    )


# ── host reads ───────────────────────────────────────────────────────────────
#
# A model loop that decides something on the host (a score threshold, a
# timestep, a pyramid level) reads the tensor back *once* through these, never
# one element at a time: on Metal every ``t[i].item()`` is a device sync.


def _host_numbers(values: Tensor) -> list[int | float]:
    """Every element of ``values``, flattened, in one host transfer."""
    flat = values.reshape(-1).tolist()
    out: list[int | float] = []
    for value in flat if isinstance(flat, list) else [flat]:
        if not isinstance(value, (int, float)):
            raise TypeError(f"a {type(value).__name__} element is not a number")
        out.append(value)
    return out


def host_floats(values: Tensor) -> list[float]:
    """Every element of ``values``, flattened, in one host transfer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import host_floats
    >>> host_floats(lucid.tensor([[0.5, 1.0], [2.0, 4.0]]))
    [0.5, 1.0, 2.0, 4.0]

    The values are the tensor's own, widened to Python floats exactly as
    ``float(t[i].item())`` would widen them.
    """
    return [float(value) for value in _host_numbers(values)]


def host_ints(values: Tensor) -> list[int]:
    """Every element of ``values``, flattened, as Python ints in one transfer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import host_ints
    >>> host_ints(lucid.tensor([999, 499, 0]))
    [999, 499, 0]

    An integer tensor keeps every bit; a float one truncates toward zero, as
    ``int(t[i].item())`` does.
    """
    return [int(value) for value in _host_numbers(values)]


def host_rows(matrix: Tensor) -> list[list[float]]:
    """The rows of a 2-D ``matrix`` as Python floats, in one host transfer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import host_rows
    >>> host_rows(lucid.tensor([[0.25, 0.75], [1.0, 0.0]]))
    [[0.25, 0.75], [1.0, 0.0]]

    Detection heads threshold every class column of the same score matrix;
    reading the rows once serves all of them.
    """
    if matrix.ndim != 2:
        raise ValueError(f"host_rows: expected a 2-D matrix, got {matrix.ndim}-D")
    count, width = int(matrix.shape[0]), int(matrix.shape[1])
    flat = host_floats(matrix)
    return [flat[row * width : (row + 1) * width] for row in range(count)]


def host_random() -> random.Random:
    """A host-side random stream seeded by one draw from Lucid's generator.

    Mask samplers make many small scalar draws, and how many depends on the
    draws themselves.  Drawing each from Lucid reads a tensor back per
    scalar — a sync when the default device is Metal.  One 63-bit draw on
    the CPU seeds a :class:`random.Random` instead, so
    :func:`lucid.manual_seed` still fixes every mask.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import host_random
    >>> lucid.manual_seed(0)
    >>> first = host_random().random()
    >>> lucid.manual_seed(0)
    >>> host_random().random() == first
    True
    """
    seed = lucid.randint(0, (1 << 63) - 1, (1,), device="cpu")
    return random.Random(int(seed.item()))


def truncated_svd_linear(layer: nn.Linear, rank: int) -> nn.Sequential:
    r"""Factorise a trained ``Linear`` into two smaller ones (Fast R-CNN §3.1).

    A layer :math:`W \in \mathbb{R}^{u \times v}` is replaced by its rank-``t``
    truncated SVD, :math:`W \approx U_t \Sigma_t V_t^{\top}`, split across two
    layers: the first applies :math:`\Sigma_t V_t^{\top}` with no bias, the
    second applies :math:`U_t` and carries the original bias.  Parameter
    count goes from :math:`uv` to :math:`t(u + v)`, so any
    :math:`t < uv / (u + v)` is a saving.

    §3.1 uses this on ``fc6``/``fc7`` to cut detection time — for a detector
    that spends most of its forward pass in the per-RoI fully-connected
    layers, "compressing them with truncated SVD reduces detection time by
    more than 30% with only a small drop in mAP".

    This is a **post-training** transformation: it approximates weights that
    already exist and is not something to train through.  It also changes
    the parameter layout, so the result no longer accepts the original
    checkpoint.

    Args:
        layer: A trained :class:`~lucid.nn.Linear`.
        rank:  ``t``.  Must lie in ``[1, min(u, v)]``; the upper end
            reproduces the layer exactly and saves nothing.

    Returns:
        A :class:`~lucid.nn.Sequential` of two ``Linear`` layers computing
        the same function up to the truncation error.

    Raises:
        ValueError: If ``rank`` is outside the representable range, since a
            silently clamped rank would return a layer that is either
            unexpectedly exact or unexpectedly degenerate.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._common import truncated_svd_linear
    >>> compressed = truncated_svd_linear(nn.Linear(64, 32), rank=8)
    >>> len(compressed)
    2
    """
    out_features = int(layer.weight.shape[0])
    in_features = int(layer.weight.shape[1])
    limit = min(out_features, in_features)
    if not 1 <= rank <= limit:
        raise ValueError(
            f"rank must lie in [1, {limit}] for a {out_features}x{in_features} "
            f"layer; got {rank}.  Above {limit} the factorisation has no more "
            "singular values to keep, and the compression would silently be "
            "an exact copy."
        )

    with lucid.no_grad():
        u, s, vh = lucid.linalg.svd(layer.weight, full_matrices=False)

    first = nn.Linear(in_features, rank, bias=False)
    second = nn.Linear(rank, out_features, bias=layer.bias is not None)
    with lucid.no_grad():
        # Sigma folds into the first factor, so the second is a plain
        # projection and the bias stays where it was.
        first.weight[:] = vh[:rank] * s[:rank].reshape(-1, 1)
        second.weight[:] = u[:, :rank]
        if layer.bias is not None and second.bias is not None:
            second.bias[:] = layer.bias

    return nn.Sequential(first, second)


def fit_linear_svm(
    features: Tensor,
    labels: Tensor,
    *,
    c: float = 1e-3,
    steps: int = 200,
    lr: float = 0.01,
) -> tuple[Tensor, Tensor]:
    r"""Fit one binary linear SVM by minimising hinge loss with L2.

    Solves

    .. math::

        \min_{w, b}\;\; \tfrac{1}{2}\lVert w \rVert^2
        + C \sum_i \max(0,\, 1 - y_i (w^{\top} x_i + b))

    by gradient descent.  R-CNN §2.3 trains one of these per class on
    features cached from a frozen network, taking positives from
    ground-truth boxes only and negatives from proposals below IoU 0.3.

    This is the optimisation, not the pipeline.  The paper's accuracy comes
    substantially from **hard negative mining** — repeatedly rescoring the
    negative pool and refitting on the highest-scoring mistakes — which
    needs the dataset, not just one feature matrix.  A caller reproducing
    R-CNN drives that loop and calls this at each round.

    Args:
        features: ``(N, D)`` feature matrix, one row per example.
        labels:   ``(N,)`` with ``+1`` for positives and ``-1`` for
            negatives.  Zeros are rejected rather than silently treated as
            negatives.
        c:        Penalty on the hinge term, the ``C`` above.
        steps:    Gradient steps.
        lr:       Learning rate.

    Returns:
        ``(w, b)`` — ``w`` of shape ``(D,)`` and ``b`` a scalar tensor.
        Score a new feature with ``x @ w + b``; positive means the class.

    Raises:
        ValueError: If ``labels`` holds anything other than +1 / -1, or if
            it contains only one sign — a single-class fit has no margin to
            maximise and would return an arbitrary separator.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import fit_linear_svm
    >>> x = lucid.tensor([[2.0, 0.0], [-2.0, 0.0]])
    >>> w, b = fit_linear_svm(x, lucid.tensor([1.0, -1.0]))
    >>> w.shape                       # one weight per feature, not a matrix
    (2,)
    >>> margins = (x * w).sum(dim=-1) + b
    >>> bool(margins.tolist()[0] > 0), bool(margins.tolist()[1] < 0)
    (True, True)
    """
    values = cast(list[float], labels.reshape(-1).tolist())
    if any(v not in (1.0, -1.0) for v in values):
        raise ValueError(
            "fit_linear_svm expects labels of exactly +1 and -1; got values "
            f"{sorted(set(values))[:5]}.  Zero-or-one labels are a common "
            "mix-up and would train against the wrong margin."
        )
    if len(set(values)) < 2:
        raise ValueError(
            "fit_linear_svm needs both positive and negative examples; the "
            f"labels are all {values[0]:+.0f}.  With one class there is no "
            "margin to maximise and the returned separator would be arbitrary."
        )

    d = int(features.shape[1])
    dev = features.device.type
    # Kept 2-D internally because ``matmul`` requires it; the caller gets the
    # flat ``(D,)`` vector they would write ``x @ w`` with.
    w = nn.Parameter(lucid.zeros(d, 1, device=dev))
    b = nn.Parameter(lucid.zeros(1, device=dev))
    opt = optim.SGD([w, b], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        scores = (features @ w).reshape(-1) + b
        hinge = F.relu(1.0 - labels * scores).sum()
        loss = 0.5 * (w * w).sum() + c * hinge
        loss.backward()
        opt.step()

    return w.detach().reshape(-1), b.detach().reshape(())


def mine_hard_negatives(
    positives: Tensor,
    negative_pool: Tensor,
    *,
    rounds: int = 5,
    keep_per_round: int = 1000,
    c: float = 1e-3,
    steps: int = 200,
    lr: float = 0.01,
) -> tuple[Tensor, Tensor, list[int]]:
    r"""Train one linear SVM by R-CNN's hard-negative mining loop (§2.3).

    Fitting on a random slice of the negatives wastes the budget: almost
    every background window is trivially separable, so the margin is set by
    a handful of near-misses that a random sample is unlikely to contain.
    The paper's answer is to keep the mistakes — score the whole negative
    pool with the current SVM, keep the highest-scoring (most wrong) ones,
    refit, repeat.  The working set only grows, so each round's separator is
    at least as constrained as the last.

    Args:
        positives:     ``(P, D)`` positive features.  R-CNN takes these from
            ground-truth boxes only, not from high-IoU proposals.
        negative_pool: ``(N, D)`` candidate negatives — proposals below the
            paper's IoU 0.3.
        rounds:        Mining iterations.
        keep_per_round: How many of the worst-scoring negatives to add each
            round.
        c:             Hinge penalty, forwarded to :func:`fit_linear_svm`.
        steps:         Gradient steps per refit.
        lr:            Learning rate per refit.

    Returns:
        ``(w, b, history)`` — the final separator and the working-set size
        after each round, which is the cheapest way to see the loop
        converge: it stops growing once no new negative scores above the
        margin.

    Raises:
        ValueError: If either set is empty.  A one-class fit has no margin,
            and returning an arbitrary separator would look like success.

    Notes:
        This is the optimisation loop over *cached* features.  Extracting
        them from a frozen network, and doing so over a whole dataset, is
        the caller's pipeline — the paper's stage 2 runs offline for exactly
        that reason.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._common import mine_hard_negatives
    >>> pos = lucid.randn(8, 4) + 3.0
    >>> neg = lucid.randn(200, 4) - 3.0
    >>> lucid.manual_seed(0)
    >>> w, b, history = mine_hard_negatives(pos, neg, rounds=2)
    >>> history                       # working-set size, one entry per round
    [200]

    Two rounds were allowed and one was run: mining stops early when a
    round turns up no negative the current separator gets wrong, so the
    length of ``history`` reports what happened rather than what was
    asked for.
    """
    n_pos = int(positives.shape[0])
    n_neg = int(negative_pool.shape[0])
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "mine_hard_negatives needs both positives and negatives; got "
            f"{n_pos} and {n_neg}.  With one class there is no margin to "
            "maximise and the returned separator would be arbitrary."
        )

    dev = positives.device.type
    # Seed the working set so round 1 has something to be wrong about.
    working: list[int] = list(range(min(keep_per_round, n_neg)))
    history: list[int] = []
    w = lucid.zeros(int(positives.shape[1]), device=dev)
    b = lucid.zeros((), device=dev)

    for _ in range(rounds):
        idx = lucid.tensor(working, device=dev).long()
        feats = lucid.cat([positives, negative_pool[idx]], dim=0)
        labels = lucid.tensor([1.0] * n_pos + [-1.0] * len(working), device=dev)
        w, b = fit_linear_svm(feats, labels, c=c, steps=steps, lr=lr)
        history.append(len(working))

        # A negative scoring above -1 is inside the margin: the SVM either
        # calls it foreground or is not confident enough about it.
        scores = cast(
            list[float],
            ((negative_pool @ w.reshape(-1, 1)).reshape(-1) + b).tolist(),
        )
        seen = set(working)
        offenders = sorted(
            (i for i, v in enumerate(scores) if v > -1.0 and i not in seen),
            key=lambda i: -scores[i],
        )
        if not offenders:
            break
        working.extend(offenders[:keep_per_round])

    return w, b, history
