"""Guard tests for embedding-loss label branches and loss Module __init__ validation (LCD-250).

Covers:
1. ``F.cosine_embedding_loss`` / ``nn.CosineEmbeddingLoss`` for ``y in {1, -1, 0, 2}``
   across ``reduction in {"none", "mean", "sum"}`` and ``device in {"cpu", "metal"}``
   (forward values and backward gradients against closed form and ``ref``).
2. ``F.hinge_embedding_loss`` / ``nn.HingeEmbeddingLoss`` for ``y in {1, -1, 0, 2}``
   across ``reduction in {"none", "mean", "sum"}`` and ``device in {"cpu", "metal"}``
   (forward values and backward gradients against closed form and ``ref``).
3. Construction-time hyperparameter validation on ``nn.MultiMarginLoss``,
   ``nn.TripletMarginLoss``, ``nn.TripletMarginWithDistanceLoss``, and sibling
   loss ``Module`` classes, held to the reference.
"""

import math
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

REDUCTIONS: tuple[str, ...] = ("none", "mean", "sum")
LABELS: tuple[int, ...] = (1, -1, 0, 2)


def _close(got: object, want: object, tol: float = 1e-5) -> bool:
    if isinstance(got, (list, tuple)):
        assert isinstance(want, (list, tuple)) and len(got) == len(want)
        return all(_close(g, w, tol) for g, w in zip(got, want))
    g, w = float(got), float(want)  # type: ignore[arg-type]
    if math.isnan(w):
        return math.isnan(g)
    if math.isinf(w):
        return g == w
    return abs(g - w) <= tol * max(1.0, abs(w))


def _vals(t: lucid.Tensor) -> object:
    return t.detach().tolist()


class TestCosineEmbeddingLossLabelParity:
    """``cosine_embedding_loss`` matches ``y == 1`` and ``y == -1``; other labels yield 0."""

    @pytest.mark.parametrize("label", LABELS)
    @pytest.mark.parametrize("reduction", REDUCTIONS)
    @pytest.mark.parity
    def test_uniform_label_values_and_grads_match_ref(
        self, ref: object, device: str, label: int, reduction: str
    ) -> None:
        R = ref
        v1 = [[1.0, 2.0, -0.5], [0.5, -1.0, 1.5]]
        v2 = [[0.5, 1.0, 0.25], [1.0, -0.5, 1.0]]
        vy = [float(label), float(label)]
        margin = 0.2

        lx1 = lucid.tensor(v1, requires_grad=True, device=device)
        lx2 = lucid.tensor(v2, requires_grad=True, device=device)
        ly = lucid.tensor(vy, device=device)
        lo = F.cosine_embedding_loss(lx1, lx2, ly, margin=margin, reduction=reduction)  # type: ignore[arg-type]
        lo.sum().backward()

        rx1 = R.tensor(v1, requires_grad=True)  # type: ignore[attr-defined]
        rx2 = R.tensor(v2, requires_grad=True)  # type: ignore[attr-defined]
        ry = R.tensor(vy)  # type: ignore[attr-defined]
        ro = R.nn.functional.cosine_embedding_loss(  # type: ignore[attr-defined]
            rx1, rx2, ry, margin=margin, reduction=reduction
        )
        ro.sum().backward()

        assert _close(_vals(lo), ro.tolist())
        assert lx1.grad is not None and lx2.grad is not None
        assert _close(_vals(lx1.grad), rx1.grad.tolist(), tol=1e-4)
        assert _close(_vals(lx2.grad), rx2.grad.tolist(), tol=1e-4)

        if label not in (1, -1):
            assert _close(_vals(lx1.grad), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            assert _close(_vals(lx2.grad), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    @pytest.mark.parity
    def test_mixed_labels_and_module_wrapper_match_ref(
        self, ref: object, device: str, reduction: str
    ) -> None:
        R = ref
        v1 = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 3.0], [0.5, -1.5, 2.0]]
        v2 = [[0.6, 0.8, 0.0], [1.0, 0.5, 0.0], [2.0, 1.0, -1.0], [-0.5, 1.0, 0.5]]
        vy = [1, -1, 0, 2]
        margin = 0.1

        lx1 = lucid.tensor(v1, requires_grad=True, device=device)
        lx2 = lucid.tensor(v2, requires_grad=True, device=device)
        ly = lucid.tensor(vy, device=device)
        mod = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)  # type: ignore[arg-type]
        lo = mod(lx1, lx2, ly)
        lo.sum().backward()

        rx1 = R.tensor(v1, requires_grad=True)  # type: ignore[attr-defined]
        rx2 = R.tensor(v2, requires_grad=True)  # type: ignore[attr-defined]
        ry = R.tensor(vy)  # type: ignore[attr-defined]
        rmod = R.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)  # type: ignore[attr-defined]
        ro = rmod(rx1, rx2, ry)
        ro.sum().backward()

        assert _close(_vals(lo), ro.tolist())
        assert lx1.grad is not None and lx2.grad is not None
        assert _close(_vals(lx1.grad), rx1.grad.tolist(), tol=1e-4)
        assert _close(_vals(lx2.grad), rx2.grad.tolist(), tol=1e-4)

    def test_1d_inputs_with_non_binary_scalar_target_give_zero(self, device: str) -> None:
        x1 = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
        x2 = lucid.tensor([2.0, 1.0, 0.5], requires_grad=True, device=device)
        y = lucid.tensor(0, device=device)
        out = F.cosine_embedding_loss(x1, x2, y, reduction="none")
        assert out.shape == ()
        assert _close(out.item(), 0.0)
        out.backward()
        assert x1.grad is not None and x2.grad is not None
        assert _close(_vals(x1.grad), [0.0, 0.0, 0.0])
        assert _close(_vals(x2.grad), [0.0, 0.0, 0.0])


class TestHingeEmbeddingLossLabelParity:
    """``hinge_embedding_loss`` sums ``y != -1`` and ``y != 1`` contributions."""

    @pytest.mark.parametrize("label", LABELS)
    @pytest.mark.parametrize("reduction", REDUCTIONS)
    @pytest.mark.parity
    def test_uniform_label_values_and_grads_match_ref(
        self, ref: object, device: str, label: int, reduction: str
    ) -> None:
        R = ref
        vx = [0.2, 0.8, 1.5, -0.3]
        vy = [label, label, label, label]
        margin = 1.0

        lx = lucid.tensor(vx, requires_grad=True, device=device)
        ly = lucid.tensor(vy, device=device)
        lo = F.hinge_embedding_loss(lx, ly, margin=margin, reduction=reduction)  # type: ignore[arg-type]
        lo.sum().backward()

        rx = R.tensor(vx, requires_grad=True)  # type: ignore[attr-defined]
        ry = R.tensor(vy)  # type: ignore[attr-defined]
        ro = R.nn.functional.hinge_embedding_loss(  # type: ignore[attr-defined]
            rx, ry, margin=margin, reduction=reduction
        )
        ro.sum().backward()

        assert _close(_vals(lo), ro.tolist())
        assert lx.grad is not None
        assert _close(_vals(lx.grad), rx.grad.tolist(), tol=1e-5)

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    @pytest.mark.parity
    def test_mixed_labels_and_module_wrapper_match_ref(
        self, ref: object, device: str, reduction: str
    ) -> None:
        R = ref
        vx = [[0.2, 0.8], [1.4, 0.5]]
        vy = [[1, -1], [0, 2]]
        margin = 1.0

        lx = lucid.tensor(vx, requires_grad=True, device=device)
        ly = lucid.tensor(vy, device=device)
        mod = nn.HingeEmbeddingLoss(margin=margin, reduction=reduction)  # type: ignore[arg-type]
        lo = mod(lx, ly)
        lo.sum().backward()

        rx = R.tensor(vx, requires_grad=True)  # type: ignore[attr-defined]
        ry = R.tensor(vy)  # type: ignore[attr-defined]
        rmod = R.nn.HingeEmbeddingLoss(margin=margin, reduction=reduction)  # type: ignore[attr-defined]
        ro = rmod(rx, ry)
        ro.sum().backward()

        assert _close(_vals(lo), ro.tolist())
        assert lx.grad is not None
        assert _close(_vals(lx.grad), rx.grad.tolist(), tol=1e-5)

    def test_zero_label_below_margin_cancels_gradient(self, device: str) -> None:
        """Issue #324 reproducer: ``x=[0.2, 0.8], y=[0, 0], margin=1.0`` -> ``[1.0, 1.0]``, grad 0."""
        x = lucid.tensor([0.2, 0.8], requires_grad=True, device=device)
        y = lucid.tensor([0, 0], device=device)
        out = F.hinge_embedding_loss(x, y, margin=1.0, reduction="none")
        assert _close(_vals(out), [1.0, 1.0])
        out.sum().backward()
        assert x.grad is not None
        assert _close(_vals(x.grad), [0.0, 0.0])

    @pytest.mark.parity
    def test_hinge_boundary_at_margin_matches_ref(self, ref: object, device: str) -> None:
        R = ref
        x = lucid.tensor([1.0, 1.0], requires_grad=True, device=device)
        y = lucid.tensor([-1, 0], device=device)
        out = F.hinge_embedding_loss(x, y, margin=1.0, reduction="none")
        assert _close(_vals(out), [0.0, 1.0])
        out.sum().backward()
        assert x.grad is not None
        assert _close(_vals(x.grad), [-1.0, 0.0])

        rx = R.tensor([1.0, 1.0], requires_grad=True)  # type: ignore[attr-defined]
        ry = R.tensor([-1, 0])  # type: ignore[attr-defined]
        ro = R.nn.functional.hinge_embedding_loss(  # type: ignore[attr-defined]
            rx, ry, margin=1.0, reduction="none"
        )
        ro.sum().backward()
        assert _close(_vals(out), ro.tolist())
        assert _close(_vals(x.grad), rx.grad.tolist())


class TestLossModuleConstructionValidation:
    """Loss ``Module.__init__`` validates hyperparameters eagerly like the reference."""

    @pytest.mark.parametrize("bad_p", [0, 3, -1, 1.5, True, False, "1"])
    @pytest.mark.parity
    def test_multi_margin_loss_invalid_p_refused_at_init(self, ref: object, bad_p: object) -> None:
        R = ref
        with pytest.raises((ValueError, TypeError)):
            nn.MultiMarginLoss(p=bad_p)  # type: ignore[arg-type]
        if not isinstance(bad_p, bool):
            with pytest.raises((ValueError, TypeError, RuntimeError)):
                R.nn.MultiMarginLoss(p=bad_p)  # type: ignore[attr-defined]

    @pytest.mark.parity
    def test_class_weight_must_be_1d_at_init(self, ref: object) -> None:
        R = ref
        for cls in (nn.MultiMarginLoss, nn.CrossEntropyLoss, nn.NLLLoss):
            with pytest.raises(ValueError, match="1D|1-D"):
                cls(weight=lucid.ones(2, 2))
        with pytest.raises((ValueError, RuntimeError)):
            R.nn.MultiMarginLoss(weight=R.ones(2, 2))  # type: ignore[attr-defined]

    @pytest.mark.parametrize("bad_margin", [0.0, 0, -1.0, float("nan"), True, "1.0"])
    @pytest.mark.parity
    def test_triplet_margin_losses_refuse_non_positive_margin_at_init(
        self, ref: object, bad_margin: object
    ) -> None:
        R = ref
        for cls in (nn.TripletMarginLoss, nn.TripletMarginWithDistanceLoss):
            with pytest.raises((ValueError, TypeError)):
                cls(margin=bad_margin)  # type: ignore[arg-type]
        if isinstance(bad_margin, (int, float)) and not isinstance(bad_margin, bool):
            if not math.isnan(bad_margin):
                for rcls in (
                    R.nn.TripletMarginLoss,  # type: ignore[attr-defined]
                    R.nn.TripletMarginWithDistanceLoss,  # type: ignore[attr-defined]
                ):
                    with pytest.raises(ValueError):
                        rcls(margin=bad_margin)

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (nn.CrossEntropyLoss, {"label_smoothing": -0.1}),
            (nn.CrossEntropyLoss, {"label_smoothing": 1.1}),
            (nn.CrossEntropyLoss, {"label_smoothing": float("nan")}),
            (nn.CrossEntropyLoss, {"label_smoothing": True}),
            (nn.SmoothL1Loss, {"beta": -0.1}),
            (nn.SmoothL1Loss, {"beta": float("nan")}),
            (nn.SmoothL1Loss, {"beta": True}),
            (nn.HuberLoss, {"delta": 0.0}),
            (nn.HuberLoss, {"delta": -1.0}),
            (nn.HuberLoss, {"delta": float("nan")}),
            (nn.HuberLoss, {"delta": True}),
            (nn.HuberLoss, {"delta": "bad"}),
        ],
    )
    def test_scalar_hyperparameter_refused_at_init(
        self, cls: type[nn.Module], kwargs: dict[str, object]
    ) -> None:
        with pytest.raises((ValueError, TypeError)):
            cls(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "cls",
        [
            nn.MSELoss,
            nn.L1Loss,
            nn.CrossEntropyLoss,
            nn.NLLLoss,
            nn.BCELoss,
            nn.BCEWithLogitsLoss,
            nn.HuberLoss,
            nn.SmoothL1Loss,
            nn.KLDivLoss,
            nn.TripletMarginLoss,
            nn.CosineEmbeddingLoss,
            nn.MarginRankingLoss,
            nn.HingeEmbeddingLoss,
            nn.PoissonNLLLoss,
            nn.GaussianNLLLoss,
            nn.CTCLoss,
            nn.MultiMarginLoss,
            nn.MultilabelMarginLoss,
            nn.MultiLabelMarginLoss,
            nn.SoftMarginLoss,
            nn.MultiLabelSoftMarginLoss,
            nn.TripletMarginWithDistanceLoss,
        ],
    )
    def test_invalid_reduction_refused_at_init(self, cls: type[nn.Module]) -> None:
        with pytest.raises(ValueError, match="reduction must be one of"):
            cls(reduction="invalid")  # type: ignore[call-arg]
        if cls is not nn.KLDivLoss:
            with pytest.raises(ValueError, match="reduction must be one of"):
                cls(reduction="batchmean")  # type: ignore[call-arg]
        else:
            mod = cls(reduction="batchmean")  # type: ignore[call-arg]
            assert mod.reduction == "batchmean"  # type: ignore[attr-defined]
