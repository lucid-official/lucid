"""Composition transforms (Albumentations-compatible).

Containers that orchestrate child transforms: ``OneOf``, ``SomeOf``,
``Sequential``, ``OneOrOther``.  A selected child is always applied (its
own ``p`` acts as a selection *weight* for ``OneOf``), so children run
with their probability gate bypassed.  Each child handles its own
multi-target dispatch, so composition works on tensors or full samples.
"""

from typing import override

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms._base import (
    Empty,
    Transform,
    TransformLike,
    _NoParams,
)


def _force(tf: TransformLike, inputs: object) -> object:
    """Apply ``tf`` ignoring its probability gate (restore ``p`` after)."""
    old = getattr(tf, "p", 1.0)
    if isinstance(tf, Transform):
        tf.p = 1.0
        try:
            return tf(inputs)
        finally:
            tf.p = old
    return tf(inputs)


class _Container(_NoParams, Transform[Empty]):
    """Base for composition transforms (children do the real work)."""

    def __init__(self, transforms: list[TransformLike], p: float = 1.0) -> None:
        super().__init__(p=p)
        self.transforms = list(transforms)

    @override
    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:  # unused
        del params
        return img

    def _gate_or_pass(self) -> bool:
        """Return True if the container should run this call."""
        return not (self.p < 1.0 and _random.rand() >= self.p)


class OneOf(_Container):
    r"""Apply exactly one child, sampled by child-``p`` weight (Albumentations ``OneOf``).

    On each call (gated by the container's ``p``), draws a single child
    via a categorical distribution whose weights are the children's
    own ``p`` values normalised to sum to 1.  The selected child is
    then forced to run with its own probability gate bypassed — its
    ``p`` acts purely as a *selection weight*, not a per-call
    skip-probability.  Equal weights ⇒ uniform sampling.

    Parameters
    ----------
    transforms : list of TransformLike
        Candidate children.  Each child's ``p`` is used as its
        unnormalised selection weight; raw callables get weight 1.
    p : float, optional, default=0.5
        Container-level gate — with probability ``1 - p`` the entire
        block is skipped and the input passes through unchanged.

    Examples
    --------
    >>> import lucid.utils.transforms as T
    >>> tf = T.OneOf([
    ...     T.HorizontalFlip(p=1.0),
    ...     T.VerticalFlip(p=1.0),
    ... ], p=1.0)
    """

    def __init__(self, transforms: list[TransformLike], p: float = 0.5) -> None:
        super().__init__(transforms, p=p)
        weights = [float(getattr(t, "p", 1.0)) for t in self.transforms]
        total = sum(weights)
        self._weights = [w / total for w in weights] if total > 0 else None

    @override
    def __call__(self, inputs: object) -> object:
        if not self.transforms or not self._gate_or_pass():
            return inputs
        r = _random.rand()
        acc = 0.0
        chosen = self.transforms[-1]
        weights = self._weights or [1.0 / len(self.transforms)] * len(self.transforms)
        for tf, w in zip(self.transforms, weights, strict=True):
            acc += w
            if r <= acc:
                chosen = tf
                break
        return _force(chosen, inputs)

    @override
    def __repr__(self) -> str:
        return f"OneOf({self.transforms}, p={self.p})"


class SomeOf(_Container):
    r"""Apply ``n`` randomly-chosen children (Albumentations ``SomeOf``).

    Parameters
    ----------
    transforms : list of Transform
    n : int
        Number of children to apply (without replacement).
    p : float, optional, default=1.0

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.SomeOf([T.HorizontalFlip(p=1.0), T.VerticalFlip(p=1.0),
    ...                T.Transpose(p=1.0)], n=2, p=1.0)
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 32, 32)

    Picks ``n`` of them each call, so an augmentation pipeline varies
    without listing every combination.
    """

    def __init__(self, transforms: list[TransformLike], n: int = 1, p: float = 1.0) -> None:
        super().__init__(transforms, p=p)
        self.n = n

    @override
    def __call__(self, inputs: object) -> object:
        if not self.transforms or not self._gate_or_pass():
            return inputs
        pool = list(range(len(self.transforms)))
        picks: list[int] = []
        for _ in range(min(self.n, len(pool))):
            k = _random.randint(0, len(pool))
            picks.append(pool.pop(k))
        picks.sort()
        for i in picks:
            inputs = _force(self.transforms[i], inputs)
        return inputs

    @override
    def __repr__(self) -> str:
        return f"SomeOf({self.transforms}, n={self.n}, p={self.p})"


class Sequential(_Container):
    r"""Apply all children in order, gated by ``p`` (Albumentations ``Sequential``).

    The container's ``p`` controls whether the whole block runs at
    all on a given call — when the gate fails, every child is
    skipped and the input passes through unchanged.  When the gate
    passes, each child is invoked left-to-right with its *own*
    probability gate honoured (unlike :class:`OneOf` / :class:`SomeOf`
    which bypass the child gate).

    Parameters
    ----------
    transforms : list of TransformLike
        Children to chain — each receives the previous child's
        output.
    p : float, optional, default=1.0
        Block-level probability — when below 1, the chain is a
        no-op on ``1 - p`` of calls.

    Examples
    --------
    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.Sequential([T.Resize(16, 16), T.HorizontalFlip(p=1.0)])
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 16, 16)

    Applies its members in order, and its own ``p`` decides whether the
    whole run happens — unlike :class:`Compose`, which always runs and
    leaves each member to its own probability.
    """

    def __init__(self, transforms: list[TransformLike], p: float = 1.0) -> None:
        super().__init__(transforms, p=p)

    @override
    def __call__(self, inputs: object) -> object:
        if not self._gate_or_pass():
            return inputs
        for tf in self.transforms:
            inputs = tf(inputs)
        return inputs

    @override
    def __repr__(self) -> str:
        return f"Sequential({self.transforms}, p={self.p})"


class OneOrOther(_Container):
    r"""Apply ``first`` with probability ``p`` else ``second`` (Albumentations ``OneOrOther``).

    A two-way categorical: every call invokes either ``first`` or
    ``second``, never both, never neither.  The selected child is
    forced to run with its own probability gate bypassed — ``p`` here
    is the *switch* between the two branches, not a skip-probability,
    so the container has effective probability 1.

    Parameters
    ----------
    first : TransformLike
        Child invoked when the per-call uniform sample is below ``p``.
    second : TransformLike
        Child invoked otherwise.
    p : float, optional, default=0.5
        Switch probability; ``0.5`` gives the two branches equal odds.

    Examples
    --------
    >>> import lucid.utils.transforms as T
    >>> tf = T.OneOrOther(
    ...     first=T.HorizontalFlip(p=1.0),
    ...     second=T.VerticalFlip(p=1.0),
    ...     p=0.7,
    ... )
    """

    def __init__(self, first: TransformLike, second: TransformLike, p: float = 0.5) -> None:
        super().__init__([first, second], p=1.0)
        self.first = first
        self.second = second
        self.switch = p

    @override
    def __call__(self, inputs: object) -> object:
        chosen = self.first if _random.rand() < self.switch else self.second
        return _force(chosen, inputs)

    @override
    def __repr__(self) -> str:
        return f"OneOrOther(p={self.switch})"


class ReplayCompose(_Container):
    r"""Compose that records applied params for deterministic replay.

    After a call, ``replay_data`` holds ``(transform, params, applied)``
    per child; :meth:`replay` re-applies the *same* params to a new
    sample (e.g. to apply an identical augmentation to a paired input).

    Examples
    --------
    Records what it sampled, so the same random transform can be applied
    again — to a second view, or to a mask that arrived later:

    >>> import lucid, lucid.utils.transforms as T
    >>> tf = T.ReplayCompose([T.RandomCrop(16, 16), T.HorizontalFlip(p=0.5)])
    >>> tuple(tf(T.Image(lucid.rand(3, 32, 32))).data.shape)
    (3, 16, 16)
    """

    def __init__(self, transforms: list[TransformLike], p: float = 1.0) -> None:
        super().__init__(transforms, p=p)
        self.replay_data: list[tuple[TransformLike, object, bool]] = []

    @override
    def __call__(self, inputs: object) -> object:
        from lucid.utils.transforms._base import _find_reference

        self.replay_data = []
        if not self._gate_or_pass():
            return inputs
        out = inputs
        for tf in self.transforms:
            if isinstance(tf, Transform):
                if tf.p < 1.0 and _random.rand() >= tf.p:
                    self.replay_data.append((tf, None, False))
                    continue
                ref = _find_reference(out)
                params = tf.make_params(ref) if ref is not None else None
                out = tf._dispatch(out, params) if params is not None else out
                self.replay_data.append((tf, params, True))
            else:
                out = tf(out)
                self.replay_data.append((tf, None, True))
        return out

    def replay(self, saved: list[tuple[TransformLike, object, bool]], inputs: object) -> object:
        """Re-apply a previously recorded ``replay_data`` to a new sample.

        Walks the saved ``(transform, params, applied)`` log in order
        and dispatches each :class:`Transform` with its *original*
        sampled params (no fresh sampling).  Raw callables are
        re-invoked without parameters.  Records marked ``applied=False``
        are skipped, preserving the original gate decisions.

        Parameters
        ----------
        saved : list of (TransformLike, object, bool)
            Trace produced by an earlier :meth:`__call__`; usually
            ``other_compose.replay_data``.
        inputs : object
            Sample to feed through the replayed pipeline.

        Returns
        -------
        object
            The replayed output, structurally matching ``inputs``.
        """
        out = inputs
        for tf, params, applied in saved:
            if not applied:
                continue
            if isinstance(tf, Transform) and params is not None:
                out = tf._dispatch(out, params)
            else:
                out = tf(out)
        return out

    @override
    def __repr__(self) -> str:
        return f"ReplayCompose({self.transforms}, p={self.p})"
