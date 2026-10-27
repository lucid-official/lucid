"""Tests for ``lucid.autograd.detect_anomaly`` and ``set_detect_anomaly`` (LCD-52)."""

import pytest

import lucid
from lucid.autograd import (
    Function,
    FunctionCtx,
    detect_anomaly,
    is_anomaly_enabled,
    set_detect_anomaly,
)


def test_detect_anomaly_catches_nan_backward(device: str) -> None:
    x = lucid.tensor([0.0], device=device, requires_grad=True)
    assert not is_anomaly_enabled()
    with pytest.raises(
        RuntimeError, match=r"Function 'SqrtBackward' returned nan values in its 0th output\."
    ):
        with detect_anomaly():
            assert is_anomaly_enabled()
            y = (x.sqrt() * 0.0).sum()
            y.backward()
    assert not is_anomaly_enabled()


def test_detect_anomaly_catches_nan_in_non_zero_output_slot(device: str) -> None:
    a = lucid.tensor([0.0], device=device, requires_grad=True)
    b = lucid.tensor([0.0], device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError, match=r"Function 'DivBackward' returned nan values in its 1th output\."
    ):
        with detect_anomaly():
            (a / b).sum().backward()
    assert not is_anomaly_enabled()


def test_detect_anomaly_allows_finite_and_inf_gradients(device: str) -> None:
    x = lucid.tensor([2.0, 3.0], device=device, requires_grad=True)
    with detect_anomaly():
        (x * x).sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [4.0, 6.0]

    # Inf without NaN does not raise under check_nan=True (matches reference).
    z = lucid.tensor([0.0], device=device, requires_grad=True)
    with detect_anomaly():
        z.sqrt().sum().backward()
    assert z.grad is not None


def test_detect_anomaly_check_nan_false_and_nesting(device: str) -> None:
    x = lucid.tensor([0.0], device=device, requires_grad=True)
    with detect_anomaly(check_nan=False):
        assert not is_anomaly_enabled()
        (x.sqrt() * 0.0).sum().backward()

    with detect_anomaly(check_nan=True):
        assert is_anomaly_enabled()
        with detect_anomaly(check_nan=False):
            assert not is_anomaly_enabled()
            x2 = lucid.tensor([0.0], device=device, requires_grad=True)
            (x2.sqrt() * 0.0).sum().backward()
        assert is_anomaly_enabled()
    assert not is_anomaly_enabled()


def test_set_detect_anomaly_global_toggle(device: str) -> None:
    x = lucid.tensor([0.0], device=device, requires_grad=True)
    try:
        set_detect_anomaly(True)
        assert is_anomaly_enabled()
        with pytest.raises(
            RuntimeError, match=r"Function 'SqrtBackward' returned nan values in its 0th output\."
        ):
            (x.sqrt() * 0.0).sum().backward()

        set_detect_anomaly(True, check_nan=False)
        assert not is_anomaly_enabled()
        x2 = lucid.tensor([0.0], device=device, requires_grad=True)
        (x2.sqrt() * 0.0).sum().backward()
    finally:
        set_detect_anomaly(False)
    assert not is_anomaly_enabled()


def test_detect_anomaly_in_create_graph_and_functional_grad(device: str) -> None:
    x = lucid.tensor([0.0], device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError, match=r"Function 'SqrtBackward' returned nan values in its 0th output\."
    ):
        with detect_anomaly():
            (x.sqrt() * 0.0).sum().backward(create_graph=True)

    x2 = lucid.tensor([0.0], device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError, match=r"Function 'SqrtBackward' returned nan values in its 0th output\."
    ):
        with detect_anomaly():
            lucid.autograd.grad((x2.sqrt() * 0.0).sum(), [x2])

    x3 = lucid.tensor([0.0], device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError, match=r"Function 'SqrtBackward' returned nan values in its 0th output\."
    ):
        with detect_anomaly():
            lucid.autograd.grad((x3.sqrt() * 0.0).sum(), [x3], create_graph=True)


def test_detect_anomaly_custom_function_and_decorator(device: str) -> None:
    class BadGradFn(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
            return a + b

        @staticmethod
        def backward(ctx: FunctionCtx, grad_out: lucid.Tensor) -> tuple[None, lucid.Tensor]:
            nan_t = lucid.full(
                grad_out.shape, float("nan"), dtype=grad_out.dtype, device=grad_out.device
            )
            return None, nan_t

    @detect_anomaly()
    def run() -> None:
        a = lucid.tensor([1.0], device=device, requires_grad=False)
        b = lucid.tensor([1.0], device=device, requires_grad=True)
        BadGradFn.apply(a, b).sum().backward()

    with pytest.raises(
        RuntimeError, match=r"Function 'PythonBackwardNode' returned nan values in its 1th output\."
    ):
        run()
    assert not is_anomaly_enabled()


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["f16", "bf16"])
def test_detect_anomaly_half_precision(device: str, dtype: lucid.dtype) -> None:
    x = lucid.tensor([0.0], dtype=dtype, device=device, requires_grad=True)
    with pytest.raises(
        RuntimeError, match=r"Function 'SqrtBackward' returned nan values in its 0th output\."
    ):
        with detect_anomaly():
            (x.sqrt() * 0.0).sum().backward()


@pytest.mark.parity
def test_detect_anomaly_parity_with_reference(ref: object) -> None:
    # 1. NaN in 0th output of SqrtBackward raises in both.
    rx = ref.tensor([0.0], requires_grad=True)  # type: ignore[attr-defined]
    with pytest.raises(
        RuntimeError, match=r"Function 'SqrtBackward.*' returned nan values in its 0th output\."
    ):
        with ref.autograd.detect_anomaly():  # type: ignore[attr-defined]
            (rx.sqrt() * 0.0).sum().backward()

    lx = lucid.tensor([0.0], requires_grad=True)
    with pytest.raises(
        RuntimeError, match=r"Function 'SqrtBackward.*' returned nan values in its 0th output\."
    ):
        with lucid.autograd.detect_anomaly():
            (lx.sqrt() * 0.0).sum().backward()

    # 2. NaN in 1st output of DivBackward raises in both.
    ra = ref.tensor([0.0], requires_grad=True)  # type: ignore[attr-defined]
    rb = ref.tensor([0.0], requires_grad=True)  # type: ignore[attr-defined]
    with pytest.raises(
        RuntimeError, match=r"Function 'DivBackward.*' returned nan values in its 1th output\."
    ):
        with ref.autograd.detect_anomaly():  # type: ignore[attr-defined]
            (ra / rb).sum().backward()

    la = lucid.tensor([0.0], requires_grad=True)
    lb = lucid.tensor([0.0], requires_grad=True)
    with pytest.raises(
        RuntimeError, match=r"Function 'DivBackward.*' returned nan values in its 1th output\."
    ):
        with lucid.autograd.detect_anomaly():
            (la / lb).sum().backward()

    # 3. Inf without NaN succeeds in both.
    rz = ref.tensor([0.0], requires_grad=True)  # type: ignore[attr-defined]
    with ref.autograd.detect_anomaly():  # type: ignore[attr-defined]
        rz.sqrt().sum().backward()
    lz = lucid.tensor([0.0], requires_grad=True)
    with lucid.autograd.detect_anomaly():
        lz.sqrt().sum().backward()
    assert rz.grad.isinf().all().item() and lz.grad.isinf().all().item()
