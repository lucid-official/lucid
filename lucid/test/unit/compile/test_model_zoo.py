"""Bit-exact compile-vs-eager parity sweep across the model zoo.

Every model listed in :func:`MODEL_CASES` is instantiated, moved to
metal, and called once in eager and once in compile mode.  The two
outputs must be **bit-exact** (``abs diff == 0.0``) — MPSGraph re-uses
the same Metal kernels as the eager path so any deviation indicates a
real bug (gate-order mismatch, weight-layout reshape regression,
unsuppressed dropout in eval, …).

Coverage targets:
  * one classical CNN per architecture family (LeNet / VGG / ResNet /
    DenseNet / MobileNet / EfficientNet)
  * one transformer family (ViT)
  * one recurrent family (custom LSTM)

Detection / generative / huge-scale variants are intentionally
skipped — they take 30+ seconds each to build, which would bloat CI
time without exercising additional emitter paths.
"""

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn

from lucid.test.unit.compile._helpers import assert_compile_parity, metal_tensor

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


class _LstmHead(nn.Module):
    """Single-layer LSTM + Linear classifier — exercises the LSTM
    emit path (single-layer / unidirectional / F32 / no-proj envelope).
    """

    def __init__(self, input_size: int = 16, hidden_size: int = 32, n_classes: int = 10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y, _ = self.lstm(x)
        return self.fc(y[-1])


MODEL_CASES = [
    pytest.param(
        "lenet_5",
        lambda: M.lenet_5(num_classes=10),
        lambda: metal_tensor(1, 1, 32, 32),
        id="lenet_5",
    ),
    pytest.param(
        "resnet_18",
        lambda: M.resnet_18(num_classes=10),
        lambda: metal_tensor(1, 3, 224, 224),
        id="resnet_18",
    ),
    pytest.param(
        "vgg_11",
        lambda: M.vgg_11(num_classes=10),
        lambda: metal_tensor(1, 3, 224, 224),
        id="vgg_11",
    ),
    pytest.param(
        "mobilenet",
        lambda: M.mobilenet(num_classes=10),
        lambda: metal_tensor(1, 3, 224, 224),
        id="mobilenet",
    ),
    pytest.param(
        "efficientnet_b0",
        lambda: M.efficientnet_b0(num_classes=10),
        lambda: metal_tensor(1, 3, 224, 224),
        id="efficientnet_b0",
    ),
    pytest.param(
        "densenet_121",
        lambda: M.densenet_121(num_classes=10),
        lambda: metal_tensor(1, 3, 224, 224),
        id="densenet_121",
    ),
    pytest.param(
        "vit_base_16",
        lambda: M.vit_base_16(image_size=224, num_classes=10),
        lambda: metal_tensor(1, 3, 224, 224),
        id="vit_base_16",
    ),
    pytest.param("lstm_head", _LstmHead, lambda: metal_tensor(8, 4, 16), id="lstm_head"),
]


@pytest.mark.parametrize(("name", "mk_model", "mk_input"), MODEL_CASES)
def test_model_zoo_bit_exact(name: str, mk_model: object, mk_input: object) -> None:
    """Every listed model must produce identical output in eager and compile."""
    model = mk_model()
    x = mk_input()
    assert_compile_parity(model, x)
