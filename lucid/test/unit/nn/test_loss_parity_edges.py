"""The edges of the loss functions, held to the reference (CHA-82).

A sweep of ``nn/functional/loss.py`` found the losses right on ordinary
inputs and wrong at the places training actually reaches: a probability
that is exactly 0 or 1, a logit that is exactly 0, a one-hot target, an
ignored index, a batch that is all padding, a half-precision batch longer
than 65504.  One class per issue; each states the case that was wrong,
checks the value written out on CPU and Metal, and — under ``-m parity`` —
the same case against the reference.
"""

import math

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.nn.test_loss_boundary_validation import losses_with, valid_call

_needs_metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")


def _close(got: object, want: object, tol: float = 1e-5) -> bool:
    """Element-wise closeness of two (nested) lists or scalars, NaN equal to NaN."""
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


# ── CHA-83 ─────────────────────────────────────────────────────────────────


class TestBinaryCrossEntropyAtTheBoundary:
    """``binary_cross_entropy`` was NaN at a probability of exactly 0 or 1.

    The input was clamped to ``[1e-12, 1 - 1e-12]``, and ``1 - 1e-12`` is
    1.0 in float32: a sigmoid of a logit above ~17 is exactly 1, so
    ``log(1 - p)`` was -inf and ``0 * -inf`` was NaN.  The log terms are
    now clamped at -100 and the gradient's denominator floored at 1e-12.
    """

    def test_a_saturated_sigmoid_is_finite(self, device: str) -> None:
        z = lucid.tensor([20.0, -20.0, 0.0], requires_grad=True, device=device)
        y = lucid.tensor([1.0, 0.0, 1.0], device=device)
        loss = nn.BCELoss()(lucid.sigmoid(z), y)
        loss.backward()
        assert _close(loss.item(), 0.2310490608215332)
        assert z.grad is not None
        assert _close(_vals(z.grad), [0.0, 6.870512e-10, -0.16666667])

    def test_the_log_terms_are_clamped_at_minus_100(self, device: str) -> None:
        p = lucid.tensor([0.0, 1.0, 0.0, 1.0], device=device)
        y = lucid.tensor([1.0, 0.0, 0.0, 1.0], device=device)
        out = F.binary_cross_entropy(p, y, reduction="none")
        assert _close(_vals(out), [100.0, 100.0, 0.0, 0.0])

    def test_the_gradient_denominator_is_floored(self, device: str) -> None:
        p = lucid.tensor([0.0, 1.0, 0.0, 1.0, 0.5], requires_grad=True, device=device)
        y = lucid.tensor([1.0, 0.0, 0.0, 1.0, 0.3], device=device)
        F.binary_cross_entropy(p, y, reduction="sum").backward()
        assert p.grad is not None
        assert _close(_vals(p.grad), [-1e12, 1e12, 0.0, 0.0, 0.8])

    def test_the_interior_keeps_its_second_derivative(self) -> None:
        p = lucid.tensor([0.3, 0.7], requires_grad=True)
        y = lucid.tensor([1.0, 0.0])
        (g,) = lucid.autograd.grad(F.binary_cross_entropy(p, y), [p], create_graph=True)
        (gg,) = lucid.autograd.grad(g.sum(), [p])
        assert _close(_vals(gg), [1.0 / (2 * 0.09), 1.0 / (2 * 0.09)])

    def test_float16_is_finite_at_the_boundary(self, device: str) -> None:
        p = lucid.tensor([0.0, 1.0, 0.5], dtype=lucid.float16, device=device)
        y = lucid.tensor([1.0, 0.0, 1.0], dtype=lucid.float16, device=device)
        out = F.binary_cross_entropy(p, y, reduction="none")
        assert out.dtype == lucid.float16
        assert _close(_vals(out), [100.0, 100.0, 0.6931], tol=1e-3)

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        vals_p = [0.0, 1.0, 0.0, 1.0, 1e-13, 0.5, 0.999]
        vals_y = [1.0, 0.0, 0.0, 1.0, 1.0, 0.3, 0.2]
        w = [1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0]
        for reduction in ("none", "mean", "sum"):
            lp = lucid.tensor(vals_p, requires_grad=True, device=device)
            lo = F.binary_cross_entropy(
                lp,
                lucid.tensor(vals_y, device=device),
                weight=lucid.tensor(w, device=device),
                reduction=reduction,
            )
            lo.sum().backward()
            rp = R.tensor(vals_p, requires_grad=True)  # type: ignore[attr-defined]
            ro = R.nn.functional.binary_cross_entropy(  # type: ignore[attr-defined]
                rp, R.tensor(vals_y), weight=R.tensor(w), reduction=reduction  # type: ignore[attr-defined]
            )
            ro.sum().backward()
            assert _close(_vals(lo), ro.tolist())
            assert lp.grad is not None
            assert _close(_vals(lp.grad), rp.grad.tolist(), tol=1e-4)


# ── CHA-84 ─────────────────────────────────────────────────────────────────


class TestBinaryCrossEntropyWithLogitsAtZero:
    """The logit form had the wrong gradient at exactly ``x = 0``.

    ``max(x, 0) - x y + log(1 + exp(-|x|))`` has subgradients 1 and 0 at
    the origin, so ``d/dx`` was ``1 - y`` instead of ``1/2 - y``: a
    zero-initialised head got no gradient from its positive labels.
    """

    def test_a_zero_logit_gets_half_minus_the_label(self, device: str) -> None:
        z = lucid.tensor([0.0, 0.0], requires_grad=True, device=device)
        nn.BCEWithLogitsLoss()(z, lucid.tensor([1.0, 0.0], device=device)).backward()
        assert z.grad is not None
        assert _close(_vals(z.grad), [-0.25, 0.25])

    def test_pos_weight_at_zero(self, device: str) -> None:
        z = lucid.tensor([0.0, 0.0], requires_grad=True, device=device)
        y = lucid.tensor([1.0, 0.0], device=device)
        pw = lucid.tensor(3.0, device=device)
        out = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="none")
        out.sum().backward()
        assert _close(_vals(out), [3 * math.log(2.0), math.log(2.0)])
        assert z.grad is not None
        assert _close(_vals(z.grad), [-1.5, 0.5])

    def test_a_zero_initialised_head_learns_from_positives(self, device: str) -> None:
        head = nn.Linear(3, 1).to(device)
        with lucid.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        x = lucid.ones(4, 3, device=device)
        y = lucid.ones(4, 1, device=device)
        nn.BCEWithLogitsLoss()(head(x), y).backward()
        assert head.bias.grad is not None
        assert _close(head.bias.grad.item(), -0.5)

    def test_saturated_logits_stay_finite(self, device: str) -> None:
        z = lucid.tensor([100.0, -100.0, 30.0, -30.0], requires_grad=True, device=device)
        y = lucid.tensor([0.0, 1.0, 1.0, 0.0], device=device)
        out = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        out.sum().backward()
        assert _close(_vals(out), [100.0, 100.0, 9.357623e-14, 9.357623e-14])
        assert z.grad is not None
        assert _close(_vals(z.grad), [1.0, -1.0, -9.357623e-14, 9.357623e-14])

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        vals_z = [0.0, 0.0, 30.0, -30.0, 1.5, -0.7]
        vals_y = [1.0, 0.0, 0.0, 1.0, 0.25, 0.9]
        w = [1.0, 2.0, 0.5, 1.0, 3.0, 1.0]
        for pos_weight in (None, 3.0):
            for reduction in ("none", "mean", "sum"):
                lz = lucid.tensor(vals_z, requires_grad=True, device=device)
                ly = lucid.tensor(vals_y, requires_grad=True, device=device)
                lo = F.binary_cross_entropy_with_logits(
                    lz,
                    ly,
                    weight=lucid.tensor(w, device=device),
                    pos_weight=(
                        None if pos_weight is None else lucid.tensor(pos_weight, device=device)
                    ),
                    reduction=reduction,
                )
                lo.sum().backward()
                rz = R.tensor(vals_z, requires_grad=True)  # type: ignore[attr-defined]
                ry = R.tensor(vals_y, requires_grad=True)  # type: ignore[attr-defined]
                ro = R.nn.functional.binary_cross_entropy_with_logits(  # type: ignore[attr-defined]
                    rz,
                    ry,
                    weight=R.tensor(w),  # type: ignore[attr-defined]
                    pos_weight=None if pos_weight is None else R.tensor(pos_weight),  # type: ignore[attr-defined]
                    reduction=reduction,
                )
                ro.sum().backward()
                assert _close(_vals(lo), ro.tolist())
                assert lz.grad is not None and ly.grad is not None
                assert _close(_vals(lz.grad), rz.grad.tolist())
                assert _close(_vals(ly.grad), ry.grad.tolist())


# ── CHA-85 ─────────────────────────────────────────────────────────────────


class TestKLDivWithZeroTargets:
    """``kl_div`` was NaN wherever the target was 0.

    ``target * (log(target) - x)`` is ``0 * -inf``.  One-hot and sparse
    distillation targets are made of such entries.
    """

    P = [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]]
    Q = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]

    def test_batchmean_of_a_one_hot_target(self, device: str) -> None:
        x = lucid.log(lucid.tensor(self.P, device=device))
        q = lucid.tensor(self.Q, device=device)
        assert _close(F.kl_div(x, q, reduction="batchmean").item(), 0.43375030)

    def test_zero_entries_contribute_zero(self, device: str) -> None:
        x = lucid.log(lucid.tensor(self.P, device=device))
        q = lucid.tensor(self.Q, device=device)
        out = F.kl_div(x, q, reduction="none")
        assert _close(_vals(out), [[0.35667494, 0.0, 0.0], [0.25541281, 0.25541281, 0.0]])

    def test_the_module_and_every_reduction_are_finite(self, device: str) -> None:
        x = lucid.log(lucid.tensor(self.P, device=device)).requires_grad_()
        q = lucid.tensor(self.Q, device=device)
        for reduction in ("mean", "sum", "batchmean"):
            assert math.isfinite(nn.KLDivLoss(reduction=reduction)(x, q).item())
        F.kl_div(x, q, reduction="batchmean").backward()
        assert x.grad is not None
        assert _close(_vals(x.grad), [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.0]])

    @pytest.mark.parity
    @pytest.mark.filterwarnings("ignore::UserWarning")  # the reference on "mean"
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        for reduction in ("none", "mean", "sum", "batchmean"):
            lx = lucid.log(lucid.tensor(self.P, device=device)).requires_grad_()
            lo = F.kl_div(lx, lucid.tensor(self.Q, device=device), reduction=reduction)
            lo.sum().backward()
            rx = R.log(R.tensor(self.P)).requires_grad_()  # type: ignore[attr-defined]
            ro = R.nn.functional.kl_div(rx, R.tensor(self.Q), reduction=reduction)  # type: ignore[attr-defined]
            ro.sum().backward()
            assert _close(_vals(lo), ro.tolist())
            assert lx.grad is not None
            assert _close(_vals(lx.grad), rx.grad.tolist())


# ── CHA-86 ─────────────────────────────────────────────────────────────────


def _kd_case(device: str) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    """A K-d classification case with an ignored position: log-probs
    ``(2, 3, 2, 2)``, targets ``(2, 2, 2)`` and a class weight."""
    logits = [
        [
            [[0.1, -0.4], [1.2, 0.3]],
            [[0.5, 0.9], [-1.0, 0.2]],
            [[2.0, -0.3], [0.0, 0.7]],
        ],
        [
            [[-0.6, 0.4], [0.8, 1.1]],
            [[0.3, -0.2], [0.6, -0.9]],
            [[1.4, 0.5], [-0.1, 0.0]],
        ],
    ]
    target = [[[0, 2], [-100, 1]], [[2, 1], [0, 2]]]
    x = F.log_softmax(lucid.tensor(logits, device=device), dim=1)
    return (
        x,
        lucid.tensor(target, device=device),
        lucid.tensor([1.0, 2.0, 0.5], device=device),
    )


def _kd_expected(reduction: str) -> object:
    """The K-d case written out: ``-w[t] * log_p[t]`` per kept position."""
    logits = [
        [
            [[0.1, -0.4], [1.2, 0.3]],
            [[0.5, 0.9], [-1.0, 0.2]],
            [[2.0, -0.3], [0.0, 0.7]],
        ],
        [
            [[-0.6, 0.4], [0.8, 1.1]],
            [[0.3, -0.2], [0.6, -0.9]],
            [[1.4, 0.5], [-0.1, 0.0]],
        ],
    ]
    target = [[[0, 2], [-100, 1]], [[2, 1], [0, 2]]]
    w = [1.0, 2.0, 0.5]
    per = [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    total = wsum = 0.0
    for n in range(2):
        for i in range(2):
            for j in range(2):
                t = target[n][i][j]
                if t == -100:
                    continue
                col = [logits[n][c][i][j] for c in range(3)]
                lse = math.log(sum(math.exp(v) for v in col))
                per[n][i][j] = -w[t] * (col[t] - lse)
                total += per[n][i][j]
                wsum += w[t]
    if reduction == "none":
        return per
    return total if reduction == "sum" else total / wsum


class TestClassTargetsAreNeverGatheredRaw:
    """Every gather of a class index used it unclamped.

    ``nll_loss`` gathered the log-probabilities at ``ignore_index`` itself
    (an ``IndexError`` on the CPU with the default -100, an out-of-bounds
    read on Metal); both losses gathered the class weight with the raw
    index and a rank-matched ``gather`` that refused a K-d target; and
    ``multi_margin_loss`` gathered a ``(1, C)`` weight with an ``(N, 1)``
    index, which the CPU refused for any ``N > 1``.
    """

    def test_nll_loss_skips_the_default_ignore_index(self, device: str) -> None:
        x = lucid.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.0, -2.0]], device=device)
        t = lucid.tensor([0, -100], device=device)
        assert _close(F.nll_loss(x, t).item(), 1.0)
        assert _close(nn.NLLLoss()(x, t).item(), 1.0)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_a_weighted_k_d_target_with_an_ignored_position(
        self, device: str, reduction: str
    ) -> None:
        x, t, w = _kd_case(device)
        got = F.nll_loss(x, t, weight=w, reduction=reduction)
        assert _close(_vals(got), _kd_expected(reduction))

    def test_cross_entropy_weight_with_an_ignored_target(self, device: str) -> None:
        logits = lucid.tensor([[1.0, 2.0, 0.5], [0.3, 0.1, 0.9]], device=device)
        t = lucid.tensor([1, -100], device=device)
        w = lucid.tensor([1.0, 2.0, 3.0], device=device)
        lse = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(0.5))
        assert _close(F.cross_entropy(logits, t, weight=w).item(), lse - 2.0)

    def test_multi_margin_weight_for_a_batch(self, device: str) -> None:
        x = lucid.tensor([[0.1, 0.2, 0.4], [0.3, 0.1, 0.2]], device=device)
        t = lucid.tensor([2, 0], device=device)
        w = lucid.tensor([1.0, 2.0, 3.0], device=device)
        assert _close(F.multi_margin_loss(x, t, weight=w).item(), 1.0333333)

    @pytest.mark.parametrize("bad", [3, -1, 7])
    def test_an_out_of_range_cpu_target_raises(self, bad: int) -> None:
        # A CPU target is on the host already: checked, as the reference does.
        x = lucid.tensor([[0.1, 0.2, 0.4], [0.3, 0.1, 0.2]])
        t = lucid.tensor([0, bad])
        for fn in (F.nll_loss, F.cross_entropy, F.multi_margin_loss):
            with pytest.raises(IndexError, match=f"Target {bad} is out of bounds"):
                fn(x, t)

    @_needs_metal
    @pytest.mark.parametrize("bad", [3, -1, 7])
    def test_an_out_of_range_metal_target_poisons_the_loss(self, bad: int) -> None:
        # Reading a Metal target back would stall every step; a bad label
        # makes the loss NaN instead of being scored as some other class.
        x = lucid.tensor([[0.1, 0.2, 0.4], [0.3, 0.1, 0.2]], device="metal")
        t = lucid.tensor([0, bad], device="metal")
        for fn in (F.nll_loss, F.cross_entropy, F.multi_margin_loss):
            assert math.isnan(fn(x, t).item())
            per = fn(x, t, reduction="none").tolist()
            assert math.isfinite(per[0]) and math.isnan(per[1])
        # The ignore_index sentinel is out of range too, and is not poison.
        ok = lucid.tensor([0, -100], device="metal")
        assert math.isfinite(F.cross_entropy(x, ok).item())

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        x, t, w = _kd_case(device)
        rx, rt, rw = (R.tensor(v.tolist()) for v in (x, t, w))  # type: ignore[attr-defined]
        for reduction in ("none", "mean", "sum"):
            lx = x.detach().requires_grad_()
            lo = F.nll_loss(lx, t, weight=w, reduction=reduction)
            lo.sum().backward()
            rxx = rx.clone().requires_grad_()
            ro = R.nn.functional.nll_loss(rxx, rt, weight=rw, reduction=reduction)  # type: ignore[attr-defined]
            ro.sum().backward()
            assert _close(_vals(lo), ro.tolist())
            assert lx.grad is not None
            assert _close(_vals(lx.grad), rxx.grad.tolist())
        mm = [[0.1, 0.2, 0.4], [0.3, 0.1, 0.2], [0.9, -0.5, 0.0]]
        mt, mw = [2, 0, 1], [1.0, 2.0, 3.0]
        for p in (1, 2):
            lx = lucid.tensor(mm, requires_grad=True, device=device)
            lo = F.multi_margin_loss(
                lx,
                lucid.tensor(mt, device=device),
                p=p,
                weight=lucid.tensor(mw, device=device),
            )
            lo.backward()
            rxx = R.tensor(mm, requires_grad=True)  # type: ignore[attr-defined]
            ro = R.nn.functional.multi_margin_loss(  # type: ignore[attr-defined]
                rxx, R.tensor(mt), p=p, weight=R.tensor(mw)  # type: ignore[attr-defined]
            )
            ro.backward()
            assert _close(lo.item(), ro.item())
            assert lx.grad is not None
            assert _close(_vals(lx.grad), rxx.grad.tolist())


# ── CHA-87 ─────────────────────────────────────────────────────────────────


class TestABatchWithEveryTargetIgnored:
    """An all-``ignore_index`` batch gave every parameter a NaN gradient.

    The mean divided by a total weight of 0, and the backward sent
    ``inf * 0`` to every masked position.  The value stays NaN, as in the
    reference; the gradient is 0, also as in the reference.
    """

    def test_the_value_is_nan_and_the_gradient_zero(self, device: str) -> None:
        lin = nn.Linear(4, 3).to(device)
        loss = nn.CrossEntropyLoss()(
            lin(lucid.randn(2, 4, device=device)),
            lucid.tensor([-100, -100], device=device),
        )
        loss.backward()
        assert math.isnan(loss.item())
        assert lin.weight.grad is not None and lin.bias.grad is not None
        assert _vals(lin.weight.grad) == [[0.0] * 4] * 3
        assert _vals(lin.bias.grad) == [0.0] * 3

    @pytest.mark.parametrize("option", ["plain", "weight", "smoothing"])
    def test_every_option_keeps_the_gradient_finite(self, device: str, option: str) -> None:
        x = lucid.tensor([[1.0, 2.0, 3.0], [0.5, 0.1, 0.2]], requires_grad=True, device=device)
        t = lucid.tensor([-100, -100], device=device)
        if option == "weight":
            loss = F.cross_entropy(x, t, weight=lucid.tensor([1.0, 2.0, 3.0], device=device))
        elif option == "smoothing":
            loss = F.cross_entropy(x, t, label_smoothing=0.1)
        else:
            loss = F.cross_entropy(x, t)
        loss.backward()
        assert math.isnan(loss.item())
        assert x.grad is not None
        assert _vals(x.grad) == [[0.0] * 3] * 2

    def test_gradient_accumulation_survives_a_padding_batch(self, device: str) -> None:
        lin = nn.Linear(4, 3).to(device)
        x = lucid.randn(2, 4, device=device)
        F.cross_entropy(lin(x), lucid.tensor([0, 2], device=device)).backward()
        assert lin.weight.grad is not None
        before = _vals(lin.weight.grad)
        F.nll_loss(
            F.log_softmax(lin(x), dim=1), lucid.tensor([-100, -100], device=device)
        ).backward()
        assert _vals(lin.weight.grad) == before

    def test_sum_of_an_all_ignored_batch_is_zero(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0, 3.0]], device=device)
        t = lucid.tensor([-100], device=device)
        assert F.nll_loss(x, t, reduction="sum").item() == 0.0

    @pytest.mark.parity
    @pytest.mark.parametrize("option", ["plain", "weight", "smoothing"])
    def test_matches_the_reference(self, ref: object, device: str, option: str) -> None:
        R = ref
        vals = [[1.0, 2.0, 3.0], [0.5, 0.1, 0.2]]
        lx = lucid.tensor(vals, requires_grad=True, device=device)
        lt = lucid.tensor([-100, -100], device=device)
        rx = R.tensor(vals, requires_grad=True)  # type: ignore[attr-defined]
        rt = R.tensor([-100, -100])  # type: ignore[attr-defined]
        rf = R.nn.functional  # type: ignore[attr-defined]
        if option == "weight":
            lo = F.cross_entropy(lx, lt, weight=lucid.tensor([1.0, 2.0, 3.0], device=device))
            ro = rf.cross_entropy(rx, rt, weight=R.tensor([1.0, 2.0, 3.0]))  # type: ignore[attr-defined]
        elif option == "smoothing":
            lo = F.cross_entropy(lx, lt, label_smoothing=0.1)
            ro = rf.cross_entropy(rx, rt, label_smoothing=0.1)
        else:
            lo = F.cross_entropy(lx, lt)
            ro = rf.cross_entropy(rx, rt)
        lo.backward()
        ro.backward()
        assert _close(lo.item(), ro.item())
        assert lx.grad is not None
        assert _close(_vals(lx.grad), rx.grad.tolist())


# ── CHA-35 (found while fixing it in the engine) ───────────────────────────


class TestHalfPrecisionClassLossesSumInFloat32:
    """A float16 ``cross_entropy`` over more than 65504 rows was NaN.

    The count of kept samples and the summed loss were both float16, so
    both were ``inf`` past 65504 and the mean was ``inf / inf``.  They are
    summed in float32 now and the result rounded back to float16.  (The
    reference's CPU kernel keeps the float16 count and is NaN here too, so
    this is held to the float32 answer rather than to the reference.)
    """

    ROWS = 70_000

    def _case(self, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
        x = lucid.zeros(self.ROWS, 4, dtype=lucid.float16, device=device)
        t = lucid.zeros(self.ROWS, dtype=lucid.int64, device=device)
        return x, t

    def test_cross_entropy_mean_is_finite(self, device: str) -> None:
        x, t = self._case(device)
        x.requires_grad_()
        loss = F.cross_entropy(x, t)
        assert loss.dtype == lucid.float16
        assert _close(loss.item(), math.log(4.0), tol=1e-3)
        loss.backward()
        assert x.grad is not None
        row = _vals(x.grad[0])
        assert _close(row, [-0.75 / self.ROWS] + [0.25 / self.ROWS] * 3, tol=1e-2)

    def test_nll_loss_mean_with_a_weight_is_finite(self, device: str) -> None:
        x, t = self._case(device)
        w = lucid.tensor([2.0, 1.0, 1.0, 1.0], dtype=lucid.float16, device=device)
        loss = F.nll_loss(F.log_softmax(x, dim=1), t, weight=w)
        assert loss.dtype == lucid.float16
        assert _close(loss.item(), math.log(4.0), tol=1e-3)

    def test_a_float16_sum_past_65504_is_inf(self, device: str) -> None:
        # The sum itself does not fit: rounded back, it is inf, as in the
        # reference — only the mean's intermediate overflow was the bug.
        x, t = self._case(device)
        assert math.isinf(F.cross_entropy(x, t, reduction="sum").item())


# ── CHA-90 ─────────────────────────────────────────────────────────────────


class TestMultilabelMarginStopsAtTheFirstMinusOne:
    """``multilabel_margin_loss`` counted labels after the first ``-1``.

    Every column was read with ``index >= 0``; the labels of a sample are
    its entries *up to* the first ``-1``, as its own docstring said.
    """

    X = [[0.1, 0.2, 0.4, 0.8]]

    @pytest.mark.parametrize(
        ("target", "want"),
        [
            ([[3, 0, -1, 1]], 0.85),
            ([[-1, 0, 1, 2]], 0.0),
            ([[3, 3, -1, 1]], 0.65),  # a class listed twice counts twice
            ([[0, 1, 2, 3]], 0.0),
        ],
        ids=["after-the-pad", "pad-first", "duplicate", "all-positive"],
    )
    def test_only_the_labels_before_the_pad_count(
        self, device: str, target: list[list[int]], want: float
    ) -> None:
        x = lucid.tensor(self.X, device=device)
        t = lucid.tensor(target, device=device)
        assert _close(F.multilabel_margin_loss(x, t).item(), want)
        assert _close(nn.MultiLabelMarginLoss()(x, t).item(), want)

    def test_a_1d_input_gives_a_0d_loss(self, device: str) -> None:
        x = lucid.tensor(self.X[0], device=device)
        t = lucid.tensor([3, 0, -1, 1], device=device)
        out = F.multilabel_margin_loss(x, t, reduction="none")
        assert out.shape == ()
        assert _close(out.item(), 0.85)

    def test_an_out_of_range_cpu_label_raises(self) -> None:
        x = lucid.tensor(self.X)
        with pytest.raises(IndexError, match="Target 5 is out of bounds"):
            F.multilabel_margin_loss(x, lucid.tensor([[5, -1, 0, 0]]))
        # After the pad, anything goes — it is never read.
        assert _close(F.multilabel_margin_loss(x, lucid.tensor([[3, -1, 9, -7]])).item(), 0.325)

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        xs = [[0.1, 0.2, 0.4, 0.8], [0.9, -0.3, 0.5, 0.0], [0.3, 0.3, 0.1, 0.7]]
        ts = [[3, 0, -1, 1], [2, 2, 0, -1], [-1, 1, 2, 3]]
        for reduction in ("none", "mean", "sum"):
            lx = lucid.tensor(xs, requires_grad=True, device=device)
            lo = F.multilabel_margin_loss(lx, lucid.tensor(ts, device=device), reduction=reduction)
            lo.sum().backward()
            rx = R.tensor(xs, requires_grad=True)  # type: ignore[attr-defined]
            ro = R.nn.functional.multilabel_margin_loss(  # type: ignore[attr-defined]
                rx, R.tensor(ts), reduction=reduction  # type: ignore[attr-defined]
            )
            ro.sum().backward()
            assert _close(_vals(lo), ro.tolist())
            assert lx.grad is not None
            assert _close(_vals(lx.grad), rx.grad.tolist())


# ── CHA-91 ─────────────────────────────────────────────────────────────────


class TestGaussianNLLVariance:
    """``gaussian_nll_loss`` took ``var`` with the wrong shape rules, a
    clamp that stopped the gradient, and no check for a negative value.

    A per-sample ``(N,)`` variance broadcast along the last axis — one
    variance per feature — and failed when ``D != N``; ``maximum(var, eps)``
    gave every variance below ``eps`` a zero gradient; ``var = -1`` gave a
    loss of half a million.
    """

    def test_a_per_sample_variance_is_per_row(self, device: str) -> None:
        out = F.gaussian_nll_loss(
            lucid.zeros(2, 2, device=device),
            lucid.ones(2, 2, device=device),
            lucid.tensor([1.0, 4.0], device=device),
            reduction="none",
        )
        b = 0.5 * (math.log(4.0) + 0.25)
        assert _close(_vals(out), [[0.5, 0.5], [b, b]])

    def test_a_per_sample_variance_against_a_wider_input(self, device: str) -> None:
        loss = F.gaussian_nll_loss(
            lucid.zeros(2, 3, device=device),
            lucid.ones(2, 3, device=device),
            lucid.tensor([1.0, 4.0], device=device),
        )
        assert _close(loss.item(), (0.5 + 0.5 * (math.log(4.0) + 0.25)) / 2)

    @pytest.mark.parametrize(
        ("x_shape", "var_shape"),
        [((4, 3, 5), (4, 1, 5)), ((3,), ()), ((3,), (1,))],
        ids=["one-size-1-dim", "0d-var", "1-elem-var"],
    )
    def test_the_other_accepted_shapes(
        self, device: str, x_shape: tuple[int, ...], var_shape: tuple[int, ...]
    ) -> None:
        loss = F.gaussian_nll_loss(
            lucid.zeros(*x_shape, device=device),
            lucid.ones(*x_shape, device=device),
            lucid.ones(*var_shape, device=device),
        )
        assert _close(loss.item(), 0.5)

    @pytest.mark.parametrize(("x_shape", "var_shape"), [((2, 3), (3, 2)), ((4, 3, 5), (1, 1, 5))])
    def test_a_variance_of_another_shape_is_refused(
        self, device: str, x_shape: tuple[int, ...], var_shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValueError, match="incorrect size"):
            F.gaussian_nll_loss(
                lucid.zeros(*x_shape, device=device),
                lucid.ones(*x_shape, device=device),
                lucid.ones(*var_shape, device=device),
            )

    def test_the_gradient_passes_the_clamp(self, device: str) -> None:
        v = lucid.tensor([0.0, 1.0], requires_grad=True, device=device)
        F.gaussian_nll_loss(
            lucid.zeros(2, device=device), lucid.ones(2, device=device), v
        ).backward()
        assert v.grad is not None
        # 0.5 (1/v - d^2/v^2) at the clamped v = 1e-6, over a mean of two.
        assert _close(_vals(v.grad), [0.25 * (1e6 - 1e12), 0.0], tol=1e-4)

    def test_a_negative_cpu_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            F.gaussian_nll_loss(lucid.zeros(2), lucid.ones(2), lucid.tensor([-1.0, 4.0]))
        with pytest.raises(ValueError, match="negative"):
            F.gaussian_nll_loss(lucid.zeros(2), lucid.ones(2), -1.0)

    @_needs_metal
    def test_a_negative_metal_variance_poisons_the_loss(self) -> None:
        out = F.gaussian_nll_loss(
            lucid.zeros(2, device="metal"),
            lucid.ones(2, device="metal"),
            lucid.tensor([-1.0, 4.0], device="metal"),
            reduction="none",
        )
        got = out.tolist()
        assert math.isnan(got[0]) and math.isfinite(got[1])

    def test_a_float_variance(self, device: str) -> None:
        loss = nn.GaussianNLLLoss()(
            lucid.zeros(2, device=device), lucid.ones(2, device=device), 2.0
        )
        assert _close(loss.item(), 0.5 * (math.log(2.0) + 0.5))

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        cases = [
            ([[0.0, 0.5], [1.0, -1.0]], [[1.0, 0.0], [0.5, 0.5]], [1.0, 4.0]),
            ([[0.0, 0.5, 2.0]], [[1.0, 0.0, 1.0]], [[0.0, 1e-8, 3.0]]),
            (
                [[0.2, 0.5, 2.0], [0.1, 0.1, 0.1]],
                [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.5], [2.0]],
            ),
        ]
        for xs, ys, vs in cases:
            for full in (False, True):
                lv = lucid.tensor(vs, requires_grad=True, device=device)
                lx = lucid.tensor(xs, requires_grad=True, device=device)
                lo = F.gaussian_nll_loss(lx, lucid.tensor(ys, device=device), lv, full=full)
                lo.backward()
                rv = R.tensor(vs, requires_grad=True)  # type: ignore[attr-defined]
                rx = R.tensor(xs, requires_grad=True)  # type: ignore[attr-defined]
                ro = R.nn.functional.gaussian_nll_loss(  # type: ignore[attr-defined]
                    rx, R.tensor(ys), rv, full=full  # type: ignore[attr-defined]
                )
                ro.backward()
                assert _close(lo.item(), ro.item())
                assert lv.grad is not None and lx.grad is not None
                assert _close(_vals(lv.grad), rv.grad.tolist(), tol=1e-4)
                assert _close(_vals(lx.grad), rx.grad.tolist(), tol=1e-4)


# ── CHA-94 (a) ─────────────────────────────────────────────────────────────


class TestAnUnknownReductionIsRefused:
    """Nine losses treated an unknown ``reduction`` as ``"none"``.

    ``reduction="avg"`` returned the unreduced tensor; the reference, and
    Lucid's own ``l1_loss`` and ``binary_cross_entropy``, raise.  The
    losses are found by introspection — every ``nn.functional`` function
    with a ``reduction`` parameter — so a new one is covered the day it is
    exported.  ``test_loss_boundary_validation.py`` holds the full table of
    bad arguments (and checks that no op runs before the refusal).
    """

    @pytest.mark.parametrize("name", losses_with("reduction"))
    def test_raises_value_error(self, name: str, device: str) -> None:
        with pytest.raises(ValueError, match="reduction"):
            getattr(F, name)(**valid_call(name, device), reduction="avg")

    @pytest.mark.parametrize("name", losses_with("reduction"))
    def test_the_known_ones_still_work(self, name: str, device: str) -> None:
        for reduction in ("none", "mean", "sum"):
            getattr(F, name)(**valid_call(name, device), reduction=reduction)

    def test_a_module_refuses_it_through_its_function(self) -> None:
        s1, s2, ones = lucid.randn(3), lucid.randn(3), lucid.ones(3)
        with pytest.raises(ValueError, match="reduction"):
            nn.MarginRankingLoss(reduction="avg")(s1, s2, ones)  # type: ignore[arg-type]


# ── CHA-94 (b) ─────────────────────────────────────────────────────────────


class TestCrossEntropyWithClassProbabilities:
    """A target of the input's shape holds class probabilities.

    The docstring said so; the gather path refused it with a rank mismatch.
    """

    X = [[1.0, 2.0, 3.0], [0.5, 0.1, 0.2]]
    Q = [[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]]

    @staticmethod
    def _by_hand(x: list[list[float]], q: list[list[float]], w: list[float]) -> list[float]:
        out = []
        for row, probs in zip(x, q):
            lse = math.log(sum(math.exp(v) for v in row))
            out.append(-sum(wc * pc * (v - lse) for wc, pc, v in zip(w, probs, row)))
        return out

    def test_the_value_is_the_expected_log_likelihood(self, device: str) -> None:
        x = lucid.tensor(self.X, device=device)
        q = lucid.tensor(self.Q, device=device)
        per = self._by_hand(self.X, self.Q, [1.0, 1.0, 1.0])
        assert _close(_vals(F.cross_entropy(x, q, reduction="none")), per)
        assert _close(F.cross_entropy(x, q).item(), sum(per) / 2)

    def test_a_weighted_mean_divides_by_the_samples(self, device: str) -> None:
        x = lucid.tensor(self.X, device=device)
        q = lucid.tensor(self.Q, device=device)
        w = [1.0, 2.0, 3.0]
        per = self._by_hand(self.X, self.Q, w)
        got = F.cross_entropy(x, q, weight=lucid.tensor(w, device=device))
        assert _close(got.item(), sum(per) / 2)

    def test_a_one_hot_probability_target_is_the_index_target(self, device: str) -> None:
        x = lucid.tensor(self.X, device=device)
        hard = F.cross_entropy(x, lucid.tensor([2, 0], device=device))
        soft = F.cross_entropy(x, lucid.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=device))
        assert _close(soft.item(), hard.item())

    def test_a_k_d_probability_target(self, device: str) -> None:
        x = lucid.randn(2, 3, 4, device=device)
        q = F.softmax(lucid.randn(2, 3, 4, device=device), dim=1)
        assert F.cross_entropy(x, q, reduction="none").shape == (2, 4)

    def test_an_integer_target_of_the_input_shape_is_refused(self, device: str) -> None:
        x = lucid.tensor(self.X, device=device)
        with pytest.raises(TypeError, match="floating point"):
            F.cross_entropy(x, lucid.tensor([[0, 1, 0], [1, 0, 0]], device=device))

    def test_ignore_index_is_refused(self, device: str) -> None:
        x = lucid.tensor(self.X, device=device)
        with pytest.raises(ValueError, match="ignore_index"):
            F.cross_entropy(x, lucid.tensor(self.Q, device=device), ignore_index=1)

    @pytest.mark.parity
    @pytest.mark.parametrize("option", ["plain", "weight", "smoothing", "none", "sum"])
    def test_matches_the_reference(self, ref: object, device: str, option: str) -> None:
        R = ref
        lx = lucid.tensor(self.X, requires_grad=True, device=device)
        lq = lucid.tensor(self.Q, device=device)
        rx = R.tensor(self.X, requires_grad=True)  # type: ignore[attr-defined]
        rq = R.tensor(self.Q)  # type: ignore[attr-defined]
        rf = R.nn.functional  # type: ignore[attr-defined]
        if option == "weight":
            lo = F.cross_entropy(lx, lq, weight=lucid.tensor([1.0, 2.0, 3.0], device=device))
            ro = rf.cross_entropy(rx, rq, weight=R.tensor([1.0, 2.0, 3.0]))  # type: ignore[attr-defined]
        elif option == "smoothing":
            lo = F.cross_entropy(lx, lq, label_smoothing=0.2)
            ro = rf.cross_entropy(rx, rq, label_smoothing=0.2)
        elif option in ("none", "sum"):
            lo = F.cross_entropy(lx, lq, reduction=option)
            ro = rf.cross_entropy(rx, rq, reduction=option)
        else:
            lo = F.cross_entropy(lx, lq)
            ro = rf.cross_entropy(rx, rq)
        lo.sum().backward()
        ro.sum().backward()
        assert _close(_vals(lo), ro.tolist())
        assert lx.grad is not None
        assert _close(_vals(lx.grad), rx.grad.tolist())


# ── CHA-94 (c) ─────────────────────────────────────────────────────────────


class TestUnbatchedInputs:
    """One sample without its batch dimension was refused.

    ``cross_entropy`` on ``(C,)`` and ``multi_margin_loss`` on ``(C,)``
    raised ``IndexError``; ``ctc_loss`` on ``(T, C)`` said "must be 3-D".
    The reference takes all three as a batch of one.
    """

    def test_cross_entropy_on_one_sample(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device)
        lse = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
        out = F.cross_entropy(x, lucid.tensor(1, device=device), reduction="none")
        assert out.shape == ()
        assert _close(out.item(), lse - 2.0)
        soft = F.cross_entropy(x, lucid.tensor([0.0, 1.0, 0.0], device=device))
        assert _close(soft.item(), lse - 2.0)
        assert _close(nn.CrossEntropyLoss()(x, lucid.tensor([1], device=device)).item(), lse - 2.0)

    def test_nll_loss_on_one_sample(self, device: str) -> None:
        x = lucid.tensor([-1.0, -2.0, -3.0], device=device)
        out = F.nll_loss(x, lucid.tensor(1, device=device), reduction="none")
        assert out.shape == () and _close(out.item(), 2.0)
        assert math.isnan(F.nll_loss(x, lucid.tensor(-100, device=device)).item())

    def test_multi_margin_on_one_sample_keeps_the_target_shape(self, device: str) -> None:
        x = lucid.tensor([0.1, 0.2, 0.4], device=device)
        scalar = F.multi_margin_loss(x, lucid.tensor(2, device=device), reduction="none")
        single = F.multi_margin_loss(x, lucid.tensor([2], device=device), reduction="none")
        assert scalar.shape == () and single.shape == (1,)
        assert _close(scalar.item(), 0.5)
        w = lucid.tensor([1.0, 2.0, 3.0], device=device)
        assert _close(F.multi_margin_loss(x, lucid.tensor(2, device=device), weight=w).item(), 1.5)

    def test_ctc_on_one_sequence(self, device: str) -> None:
        lp = F.log_softmax(lucid.randn(5, 4, device=device), dim=1)
        target = lucid.tensor([1, 2], device=device)
        one = F.ctc_loss(lp, target, lucid.tensor(5), lucid.tensor(2), reduction="none")
        batched = F.ctc_loss(lp.unsqueeze(1), target.reshape(1, 2), [5], [2], reduction="none")
        assert one.shape == ()
        assert _close(one.item(), batched.tolist()[0])
        mean = F.ctc_loss(lp, target, lucid.tensor(5), lucid.tensor(2))
        assert _close(mean.item(), one.item() / 2)

    @pytest.mark.parity
    def test_matches_the_reference(self, ref: object, device: str) -> None:
        R = ref
        rf = R.nn.functional  # type: ignore[attr-defined]
        xs = [1.0, 2.0, 3.0]
        for reduction in ("none", "mean", "sum"):
            lo = F.cross_entropy(
                lucid.tensor(xs, device=device),
                lucid.tensor(1, device=device),
                reduction=reduction,
            )
            ro = rf.cross_entropy(R.tensor(xs), R.tensor(1), reduction=reduction)  # type: ignore[attr-defined]
            assert lo.shape == tuple(ro.shape) and _close(lo.item(), ro.item())
            lo = F.multi_margin_loss(
                lucid.tensor(xs, device=device),
                lucid.tensor([2], device=device),
                reduction=reduction,
            )
            ro = rf.multi_margin_loss(R.tensor(xs), R.tensor([2]), reduction=reduction)  # type: ignore[attr-defined]
            assert lo.shape == tuple(ro.shape) and _close(_vals(lo), ro.tolist())
        logits = [
            [0.1, -0.4, 1.2, 0.3],
            [0.5, 0.9, -1.0, 0.2],
            [2.0, -0.3, 0.0, 0.7],
            [-0.6, 0.4, 0.8, 1.1],
            [0.3, -0.2, 0.6, -0.9],
        ]
        lp = F.log_softmax(lucid.tensor(logits, device=device), dim=1)
        rp = R.tensor(logits).log_softmax(1)  # type: ignore[attr-defined]
        for reduction in ("none", "mean", "sum"):
            lo = F.ctc_loss(
                lp,
                lucid.tensor([1, 2], device=device),
                lucid.tensor(5),
                lucid.tensor(2),
                reduction=reduction,
            )
            ro = rf.ctc_loss(rp, R.tensor([1, 2]), R.tensor(5), R.tensor(2), reduction=reduction)  # type: ignore[attr-defined]
            assert lo.shape == tuple(ro.shape) and _close(lo.item(), ro.item(), tol=1e-4)


# ── CHA-88 ─────────────────────────────────────────────────────────────────


def _weighted_modules() -> dict[str, tuple[nn.Module, list[str]]]:
    """Every loss module with a weight tensor, built with one, and the
    ``state_dict`` keys it should have."""
    w3 = lucid.tensor([1.0, 2.0, 3.0])
    return {
        "CrossEntropyLoss": (nn.CrossEntropyLoss(weight=w3), ["weight"]),
        "NLLLoss": (nn.NLLLoss(weight=w3), ["weight"]),
        "BCELoss": (nn.BCELoss(weight=w3), ["weight"]),
        "BCEWithLogitsLoss": (
            nn.BCEWithLogitsLoss(weight=w3, pos_weight=lucid.tensor([3.0, 1.0, 2.0])),
            ["weight", "pos_weight"],
        ),
        "MultiMarginLoss": (nn.MultiMarginLoss(weight=w3), ["weight"]),
        "MultiLabelSoftMarginLoss": (
            nn.MultiLabelSoftMarginLoss(weight=w3),
            ["weight"],
        ),
    }


def _call(name: str, module: nn.Module, device: str) -> lucid.Tensor:
    x = lucid.tensor([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]], device=device)
    if name in ("CrossEntropyLoss", "NLLLoss", "MultiMarginLoss"):
        return module(x, lucid.tensor([2, 0], device=device))
    y = lucid.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device)
    return module(x, y)


class TestLossWeightsAreBuffers:
    """A loss module's ``weight`` / ``pos_weight`` was a plain attribute.

    ``.to("metal")`` left it on the CPU, so the first forward raised
    ``DeviceMismatch``, and ``state_dict`` did not hold it.
    """

    @pytest.mark.parametrize("name", list(_weighted_modules()))
    def test_state_dict_holds_the_weights(self, name: str) -> None:
        module, keys = _weighted_modules()[name]
        assert sorted(module.state_dict()) == sorted(keys)

    @pytest.mark.parametrize("name", list(_weighted_modules()))
    def test_no_weight_is_no_key(self, name: str) -> None:
        module = getattr(nn, name)()
        assert list(module.state_dict()) == []
        assert module.weight is None

    @_needs_metal
    @pytest.mark.parametrize("name", list(_weighted_modules()))
    def test_to_metal_moves_the_weights(self, name: str) -> None:
        module, keys = _weighted_modules()[name]
        cpu_value = _call(name, module, "cpu").item()
        module = module.to("metal")
        for key in keys:
            assert getattr(module, key).device == "metal"
        assert _close(_call(name, module, "metal").item(), cpu_value)

    @pytest.mark.parametrize("name", list(_weighted_modules()))
    def test_a_checkpoint_round_trips(self, name: str) -> None:
        module, keys = _weighted_modules()[name]
        fresh = type(module)(weight=lucid.zeros(3))
        if "pos_weight" in keys:
            fresh = type(module)(weight=lucid.zeros(3), pos_weight=lucid.zeros(3))
        fresh.load_state_dict(module.state_dict())
        for key in keys:
            assert _vals(getattr(fresh, key)) == _vals(getattr(module, key))

    @pytest.mark.parametrize("name", list(_weighted_modules()))
    def test_an_older_checkpoint_without_the_keys_still_loads(self, name: str) -> None:
        # Written before the weights were buffers: no keys, and either no
        # metadata or version 1.  Strict loading keeps the built weights.
        module, keys = _weighted_modules()[name]
        before = {k: _vals(getattr(module, k)) for k in keys}
        from collections import OrderedDict

        v1: OrderedDict[str, lucid.Tensor] = OrderedDict()
        v1._metadata = {"": {"version": 1}}  # type: ignore[attr-defined]
        for old in ({}, v1):
            result = module.load_state_dict(old, strict=True)
            assert list(result.missing_keys) == []  # type: ignore[attr-defined]
        assert {k: _vals(getattr(module, k)) for k in keys} == before

    def test_a_current_checkpoint_without_the_key_is_still_strict(self) -> None:
        from collections import OrderedDict

        module = nn.CrossEntropyLoss(weight=lucid.tensor([1.0, 2.0, 3.0]))
        v2: OrderedDict[str, lucid.Tensor] = OrderedDict()
        v2._metadata = {"": {"version": 2}}  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="Missing key"):
            module.load_state_dict(v2)

    def test_a_loss_inside_a_model_moves_with_it(self) -> None:
        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)
                self.criterion = nn.CrossEntropyLoss(weight=lucid.tensor([1.0, 2.0, 3.0]))

        model = Head()
        assert sorted(model.state_dict()) == [
            "criterion.weight",
            "fc.bias",
            "fc.weight",
        ]
        model.half()
        assert model.criterion.weight.dtype == lucid.float16

    @pytest.mark.parity
    @pytest.mark.parametrize("name", list(_weighted_modules()))
    def test_the_same_keys_as_the_reference(self, ref: object, name: str) -> None:
        R = ref
        module, keys = _weighted_modules()[name]
        kw = {k: R.tensor(_vals(getattr(module, k))) for k in keys}  # type: ignore[attr-defined]
        theirs = getattr(R.nn, name)(**kw)  # type: ignore[attr-defined]
        assert sorted(theirs.state_dict()) == sorted(module.state_dict())
        # And their checkpoint loads into ours, strictly.
        fresh = type(module)(**{k: lucid.zeros(3) for k in keys})
        fresh.load_state_dict({k: lucid.tensor(v.tolist()) for k, v in theirs.state_dict().items()})
        for key in keys:
            assert _vals(getattr(fresh, key)) == _vals(getattr(module, key))


# ── autocast: what the fixes above must survive ───────────────────────────


def _autocast_loss(name: str, pred: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    if name == "binary_cross_entropy":
        return F.binary_cross_entropy(lucid.sigmoid(pred), (target > 0).float())
    if name == "binary_cross_entropy_with_logits":
        return F.binary_cross_entropy_with_logits(pred, (target > 0).float())
    if name == "cross_entropy":
        return F.cross_entropy(pred, target.argmax(dim=1))
    if name == "gaussian_nll_loss":
        return F.gaussian_nll_loss(pred, target, pred.exp())
    if name == "kl_div":
        return F.kl_div(F.log_softmax(pred, dim=1), F.softmax(target, dim=1), reduction="batchmean")
    if name == "multi_margin_loss":
        return F.multi_margin_loss(pred, target.argmax(dim=1))
    raise AssertionError(name)


_AUTOCAST_LOSSES = [
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "gaussian_nll_loss",
    "kl_div",
    "multi_margin_loss",
]


@_needs_metal
class TestLossesUnderAutocast:
    """Under autocast, ``mul`` / ``div`` / comparisons take float16 operands.

    Two of the fixes first broke there: the BCE gradient floor (1e12) is
    inf in float16 and its masked branch sent back ``0 * inf = NaN``, and a
    ``denom == 0.0`` against a float32 denominator aborted the graph
    compiler of a compiled step.  Each loss trains through ``fused_step``
    under autocast with finite, falling losses.
    """

    @pytest.mark.parametrize("name", _AUTOCAST_LOSSES)
    def test_a_compiled_autocast_step_trains(self, name: str) -> None:
        from lucid.compile import fused_step

        lucid.manual_seed(0)
        model = nn.Linear(8, 4).to("metal")
        x = lucid.randn(4, 8).to("metal")
        t = lucid.randn(4, 4).to("metal")
        opt = lucid.optim.SGD(model.parameters(), lr=1e-1)
        step = fused_step(model, lambda p, y: _autocast_loss(name, p, y), opt)
        losses = []
        for _ in range(3):
            with lucid.amp.autocast(dtype=lucid.float16):
                loss = step(x, t)
            losses.append(float(loss.item()))
        assert all(math.isfinite(v) for v in losses), losses
        assert losses[-1] < losses[0], losses

    def test_bce_of_a_saturated_float16_sigmoid(self) -> None:
        # float16 rounds sigmoid(12) to exactly 1: the boundary is reached
        # at ordinary logits there.
        z = lucid.tensor([12.0, -12.0, 0.5], device="metal", requires_grad=True)
        y = lucid.tensor([0.0, 1.0, 1.0], device="metal")
        with lucid.amp.autocast(dtype=lucid.float16):
            loss = F.binary_cross_entropy(lucid.sigmoid(z), y)
        loss.backward()
        assert math.isfinite(loss.item())
        assert z.grad is not None
        assert all(math.isfinite(v) for v in _vals(z.grad))  # type: ignore[union-attr]


def test_kl_div_honours_deprecated_size_average(device: str) -> None:
    log_q = F.log_softmax(lucid.tensor([[2.0, 0.5, 0.1]], device=device), dim=1)
    p = lucid.tensor([[0.8, 0.15, 0.05]], device=device)
    assert _close(
        F.kl_div(log_q, p, size_average=False).item(),
        F.kl_div(log_q, p, reduction="sum").item(),
    )
    assert _close(
        F.kl_div(log_q, p, size_average=True, reduction="sum").item(),
        F.kl_div(log_q, p, reduction="mean").item(),
    )
