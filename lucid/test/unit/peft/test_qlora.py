"""QLoRA tests (LCD-717): activation gradients through ``quantized_matmul``,
:class:`~lucid.peft.tuners.lora.QuantizedLinear`, ``prepare_model_for_kbit_training``,
and end-to-end 4-bit GPT-2 fine-tuning on Metal.
"""

import copy

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.nn.quantized.dynamic as nnqd
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from lucid.peft.tuners.lora import QuantizedLinear
from lucid.peft.tuners.lora._weight_site import _QuantizedInt8, _QuantizedMLX, _WeightSite
from lucid.quantization._qgemm import dequantize, is_available, quantize, quantized_matmul

pytestmark = pytest.mark.skipif(not is_available(), reason="MLX quantized kernels not built")


class TestQuantizedMatmulBackward:
    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("transpose", [True, False])
    @pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16, lucid.bfloat16])
    def test_dx_matches_dequantized_reference(
        self, bits: int, transpose: bool, dtype: lucid.dtype
    ) -> None:
        lucid.manual_seed(0)
        w_shape = (32, 64) if transpose else (64, 64)
        out_dim = 32 if transpose else 64
        w = lucid.randn(*w_shape, device="metal")
        packed, scales, biases = quantize(w, group_size=64, bits=bits)
        w_deq = dequantize(packed, scales, biases, group_size=64, bits=bits).to(dtype)

        x0 = lucid.randn(2, 5, 64, dtype=dtype, device="metal")
        grad_out = lucid.randn(2, 5, out_dim, dtype=dtype, device="metal")

        x_q = x0.clone().detach().requires_grad_(True)
        y_q = quantized_matmul(
            x_q, packed, scales, biases, transpose=transpose, group_size=64, bits=bits
        )
        assert y_q.dtype == dtype
        (y_q * grad_out).sum().backward()

        x_ref = x0.clone().detach().requires_grad_(True)
        y_ref = x_ref @ (w_deq.mT if transpose else w_deq)
        (y_ref * grad_out).sum().backward()

        assert x_q.grad is not None and x_ref.grad is not None
        assert x_q.grad.dtype == dtype
        diff = float((x_q.grad.to(lucid.float32) - x_ref.grad.to(lucid.float32)).abs().max().item())
        tol = 5e-2 if dtype in (lucid.float16, lucid.bfloat16) else 1e-4
        assert diff < tol

    def test_double_backward_create_graph(self) -> None:
        lucid.manual_seed(1)
        w = lucid.randn(32, 64, device="metal")
        packed, scales, biases = quantize(w, group_size=64, bits=4)
        x = lucid.randn(4, 64, device="metal", requires_grad=True)
        y = quantized_matmul(x, packed, scales, biases, transpose=True, group_size=64, bits=4)
        (gx,) = lucid.autograd.grad(y.pow(2).sum(), (x,), create_graph=True)
        loss2 = gx.pow(2).sum()
        loss2.backward()
        assert x.grad is not None and bool(lucid.isfinite(x.grad).all().item())


class TestQuantizedLinearLayer:
    @pytest.mark.parametrize("bits", [4, 8])
    def test_mlx_quantized_linear_forward_backward_and_merge(self, bits: int) -> None:
        lucid.manual_seed(2)
        base = nnq.QuantizedLinearMLX.from_float(nn.Linear(64, 32).to("metal"), bits=bits)
        assert type(_WeightSite.of(base)) is _QuantizedMLX
        layer = QuantizedLinear(base, "default", r=4, lora_alpha=8)

        assert layer.lora_A["default"].weight.device.type == "metal"
        assert layer.lora_A["default"].weight.dtype == lucid.float32

        x = lucid.randn(3, 64, device="metal", requires_grad=True)
        out = layer(x)
        assert bool(lucid.allclose(out, base(x), atol=1e-6))

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.lora_A["default"].weight.grad is not None
        assert layer.lora_B["default"].weight.grad is not None

        with lucid.no_grad():
            layer.lora_B["default"].weight.copy_(lucid.randn(32, 4, device="metal") * 0.1)

        packed_before = base.packed_weight.clone()
        scales_before = base.scales.clone()
        biases_before = base.biases.clone()

        y_unmerged = layer(x.detach())
        layer.merge()
        assert layer.merged
        y_merged = layer(x.detach())
        # Re-quantizing W + delta introduces at most one quantization grid step of error
        tol = 0.25 if bits == 4 else 0.05
        assert float((y_merged - y_unmerged).abs().max().item()) < tol

        layer.unmerge()
        assert not layer.merged
        assert bool((base.packed_weight == packed_before).all().item())
        assert bool((base.scales == scales_before).all().item())
        assert bool((base.biases == biases_before).all().item())
        assert bool(lucid.allclose(layer(x.detach()), y_unmerged, atol=1e-6))

    def test_int8_dynamic_linear_lora_and_merge(self) -> None:
        lucid.manual_seed(3)
        base = nnqd.Linear.from_float(nn.Linear(32, 16))
        assert type(_WeightSite.of(base)) is _QuantizedInt8
        net = nn.Sequential(base)
        peft = get_peft_model(net, LoraConfig(r=2, lora_alpha=4, target_modules="all-linear"))
        assert isinstance(net[0], QuantizedLinear)

        x = lucid.randn(4, 32, requires_grad=True)
        y = peft(x)
        y.sum().backward()
        assert net[0].lora_B["default"].weight.grad is not None

        with lucid.no_grad():
            net[0].lora_B["default"].weight.copy_(lucid.randn(16, 2) * 0.05)
        w_before = base.weight_int8.clone()
        peft.merge_adapter()
        assert net[0].merged
        peft.unmerge_adapter()
        assert not net[0].merged
        assert bool((base.weight_int8 == w_before).all().item())


def _quantize_gpt2_4bit(model: nn.Module) -> nn.Module:
    """Replace every non-head ``nn.Linear`` in ``model`` with a 4-bit ``QuantizedLinearMLX``."""
    for name, child in list(model.named_children()):
        if name == "lm_head":
            continue
        if isinstance(child, nn.Linear) and child.in_features % 64 == 0:
            setattr(model, name, nnq.QuantizedLinearMLX.from_float(child, bits=4, group_size=64))
        else:
            _quantize_gpt2_4bit(child)
    return model


def _model_storage_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for t in (*model.parameters(), *model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


class TestQLoRAGPT2EndToEnd:
    def test_4bit_gpt2_qlora_reduces_loss_and_uses_less_memory_than_bf16_lora(self) -> None:
        lucid.manual_seed(42)
        base_model = create_model(
            "gpt2_small_lm",
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            max_position_embeddings=16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        ).to("metal")

        bf16_model = copy.deepcopy(base_model).to(lucid.bfloat16)
        bf16_peft = get_peft_model(
            bf16_model, LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn", "c_proj", "c_fc"])
        )
        bf16_bytes = _model_storage_bytes(bf16_peft)

        q_model = _quantize_gpt2_4bit(copy.deepcopy(base_model).to(lucid.bfloat16))
        q_model = prepare_model_for_kbit_training(q_model)
        qlora_peft = get_peft_model(
            q_model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear")
        )
        qlora_bytes = _model_storage_bytes(qlora_peft)

        # 4-bit quantized blocks + fp32 norms/embeddings/adapters use measurably less storage
        assert qlora_bytes < bf16_bytes

        trainable = [p for p in qlora_peft.parameters() if p.requires_grad]
        assert trainable and all(p.dtype == lucid.float32 for p in trainable)

        input_ids = lucid.randint(0, 128, (4, 8), device="metal")
        labels = input_ids.clone()

        opt = optim.AdamW(trainable, lr=1e-2)
        losses: list[float] = []
        for _ in range(20):
            opt.zero_grad()
            out = qlora_peft(input_ids, labels=labels)
            assert out.loss is not None
            losses.append(float(out.loss.item()))
            out.loss.backward()
            opt.step()

        assert losses[-1] < losses[0]
