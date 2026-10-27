"""Model-zoo compile-coverage guard — these models must COMPILE, not eager-fallback.

``test_model_zoo`` checks compile-vs-eager *parity*, but parity passes even
when a model silently falls back to eager (eager-vs-eager == 0).  That blind
spot let single-layer LSTM compile stay broken indefinitely.  This suite
instead asserts each model builds a real executable with **no** ``eager_only``
signature, so any regression that pushes a representative model back to eager
(an emitter returning false, a tracer change, a new un-emittable op in a
forward) trips a test instead of hiding behind parity.

Coverage spans the families this session brought into the compile envelope:
classic + modern CNNs, a ViT, BERT / RoFormer (interleaved RoPE) / GPT-2, and
a multi-layer bidirectional LSTM.  Inputs are deliberately small (64x64 images,
tiny text configs) to keep CI fast — compile coverage is structural, not
shape-dependent.
"""

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, assert_compiles, metal_tensor, to_metal

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


def _img(c: int = 3, hw: int = 64) -> lucid.Tensor:
    return metal_tensor(2, c, hw, hw)


def _ids() -> lucid.Tensor:
    # token ids in [0, 16) < vocab_size; int64 for the embedding gather.
    return lucid.arange(0, 16, device="metal").reshape(2, 8).to(lucid.int64)


def _text(cfg_cls: type, model_cls: type) -> nn.Module:
    cfg = cfg_cls(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return model_cls(cfg)


class _LSTMHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(16, 32, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(64, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y, _ = self.lstm(x)
        return self.fc(y[-1])


# (id, factory, input factory) — every entry MUST compile with no eager fallback.
_VISION_CASES = [
    pytest.param(
        lambda: M.lenet_5(num_classes=10), lambda: metal_tensor(2, 1, 32, 32), id="lenet_5"
    ),
    pytest.param(lambda: M.resnet_18(num_classes=10), _img, id="resnet_18"),
    pytest.param(lambda: M.vgg_11(num_classes=10), _img, id="vgg_11"),
    pytest.param(lambda: M.mobilenet_v3_small(num_classes=10), _img, id="mobilenet_v3_small"),
    pytest.param(lambda: M.densenet_121(num_classes=10), _img, id="densenet_121"),
    pytest.param(lambda: M.vit_base_16(image_size=64, num_classes=10), _img, id="vit_base_16"),
]

_TEXT_CASES = [
    pytest.param(lambda: _text(M.BERTConfig, M.BERTModel), _ids, id="bert"),
    pytest.param(lambda: _text(M.RoFormerConfig, M.RoFormerModel), _ids, id="roformer"),
    pytest.param(lambda: _text(M.GPT2Config, M.GPT2Model), _ids, id="gpt2"),
]


@pytest.mark.parametrize("mk_model,mk_input", _VISION_CASES)
def test_vision_model_compiles(mk_model: object, mk_input: object) -> None:
    assert_compiles(mk_model(), mk_input())


@pytest.mark.parametrize("mk_model,mk_input", _TEXT_CASES)
def test_text_model_compiles(mk_model: object, mk_input: object) -> None:
    # RoFormer's interleaved RoPE + BERT's masked attention reach compile now.
    assert_compiles(mk_model(), mk_input())


def test_lstm_head_compiles() -> None:
    # Multi-layer bidirectional LSTM — silently eager-fell-back before this
    # session's multi-output / liveness fixes.
    assert_compiles(_LSTMHead(), metal_tensor(8, 4, 16))


def test_embedding_padding_idx_compiles() -> None:
    """Compiled and eager must answer the same at the pad index.

    This test used to assert the pad rows came back zero, which pinned
    the wrong contract. ``padding_idx`` zeroes that row at initialisation
    and stops its gradient; it does not mask the lookup, so a table whose
    pad row holds a trained vector returns that vector. Asserting zero
    made a compiled graph match an eager kernel that was itself wrong,
    and the pair agreed all the way to bert_base answering 4.14 away from
    the implementation its weights came from.

    What the test was for still holds: the two paths have to agree, and
    a freshly built table has the pad row zeroed already, so they agree
    by accident unless it is filled deliberately.
    """
    emb = nn.Embedding(8, 4, padding_idx=0)
    nn.init.normal_(emb.weight, std=1.0)

    # The eager answer, taken before anything moves: ``assert_compiles``
    # leaves the module on the compile device, and once it is there
    # neither side is left on CPU to compare against.
    emb.eval()
    cpu_ids = lucid.tensor([[0, 1, 2, 0]]).long()
    expected = emb(cpu_ids).numpy()

    ids = cpu_ids.to(COMPILE_DEVICE)
    assert_compiles(emb, ids)
    to_metal(emb)
    out = emb(ids).numpy()

    assert (
        abs(float(expected[0, 0].sum())) > 1e-6
    ), "the pad row is zero, so this would pass without comparing anything"
    assert float(abs(out - expected).max()) < 1e-5
