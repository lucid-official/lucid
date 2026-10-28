"""
benchmarks/bench_models.py — representative architecture & StaticCache decode suite.

Covers:
  1. Vision & sequence forward + training steps (ResNet-18, ViT-Tiny,
     TransformerEncoderLayer, GPT-2, T5).
  2. Compiled training step speedup (TransformerEncoderLayer fused_step vs eager).
  3. Autoregressive StaticCache decode throughput (tok/s) for GPT-2:
     eager StaticCache vs compiled_decode_step.
"""

import os
import sys
import time
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.benchmarks._core import BenchResult, fmt_table, metal_available
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.models.text.gpt2 import GPT2Config, GPT2LMHeadModel
from lucid.models.text.t5 import T5Config, T5ForSeq2SeqLM
from lucid.models.vision.resnet import resnet_18_cls
from lucid.models.vision.vit import ViTConfig, ViTForImageClassification
from lucid.utils.cache import StaticCache


def _sync(tensor: lucid.Tensor) -> None:
    if tensor.is_metal:
        from lucid._C import engine as _C_engine

        _C_engine.eval_gpu(tensor._impl)


def _time_tensor_fn(fn: Callable[[], lucid.Tensor], *, warmup: int, iters: int) -> BenchResult:
    for _ in range(warmup):
        _sync(fn())
    times: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _sync(fn())
        times.append(time.perf_counter_ns() - t0)
    return BenchResult("model_fn", times)


def _bench_arch_pair(
    name: str,
    device: str,
    fwd_fn: Callable[[], lucid.Tensor],
    step_fn: Callable[[], lucid.Tensor],
    *,
    units_per_step: int,
    unit_key: str,
    iters: int,
) -> dict[str, object]:
    with lucid.no_grad():
        fwd_r = _time_tensor_fn(fwd_fn, warmup=2, iters=iters)
    step_r = _time_tensor_fn(step_fn, warmup=2, iters=iters)
    fwd_ms = fwd_r.mean_us / 1_000.0
    step_ms = step_r.mean_us / 1_000.0
    rate = units_per_step / (step_r.mean_us * 1e-6)
    return {
        "name": name,
        "device": device,
        "fwd_mean_ms": round(fwd_ms, 3),
        "step_mean_ms": round(step_ms, 3),
        "step_p95_ms": round(step_r.p95_us / 1_000.0, 3),
        "unit_key": unit_key,
        "rate": round(rate, 1),
    }


def _bench_vision_and_transformer(device: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # 1. ResNet-18 (B=2, 3x64x64)
    rn = resnet_18_cls(num_classes=10).to(device)
    rn_opt = optim.AdamW(rn.parameters(), lr=1e-3)
    rn_x = lucid.randn(2, 3, 64, 64, device=device)
    rn_y = lucid.randint(0, 10, (2,), device=device)

    def _rn_step() -> lucid.Tensor:
        rn_opt.zero_grad()
        loss = F.cross_entropy(rn.forward(rn_x).logits, rn_y)
        loss.backward()
        rn_opt.step()
        return loss

    rows.append(
        _bench_arch_pair(
            "resnet18_b2_64",
            device,
            lambda: rn.forward(rn_x).logits,
            _rn_step,
            units_per_step=2,
            unit_key="samples_sec",
            iters=6,
        )
    )

    # 2. ViT-Tiny (B=4, 3x64x64, patch=16, dim=64, depth=2)
    vit_cfg = ViTConfig(
        image_size=64, patch_size=16, num_classes=10, dim=64, depth=2, num_heads=4, dropout=0.0
    )
    vit = ViTForImageClassification(vit_cfg).to(device).eval()
    vit_opt = optim.AdamW(vit.parameters(), lr=1e-3)
    vit_x = lucid.randn(4, 3, 64, 64, device=device)
    vit_y = lucid.randint(0, 10, (4,), device=device)

    def _vit_step() -> lucid.Tensor:
        vit_opt.zero_grad()
        loss = F.cross_entropy(vit.forward(vit_x).logits, vit_y)
        loss.backward()
        vit_opt.step()
        return loss

    rows.append(
        _bench_arch_pair(
            "vit_tiny_b4_64",
            device,
            lambda: vit.forward(vit_x).logits,
            _vit_step,
            units_per_step=4,
            unit_key="samples_sec",
            iters=15,
        )
    )

    # 3. TransformerEncoderLayer (B=4, T=16, D=64)
    enc = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True
    ).to(device)
    enc_opt = optim.AdamW(enc.parameters(), lr=1e-3)
    enc_x = lucid.randn(4, 16, 64, device=device)
    enc_t = lucid.randn(4, 16, 64, device=device)

    def _enc_step() -> lucid.Tensor:
        enc_opt.zero_grad()
        loss = F.mse_loss(enc.forward(enc_x), enc_t)
        loss.backward()
        enc_opt.step()
        return loss

    rows.append(
        _bench_arch_pair(
            "transformer_enc_b4_t16",
            device,
            lambda: enc.forward(enc_x),
            _enc_step,
            units_per_step=64,
            unit_key="tok_s",
            iters=30,
        )
    )
    return rows


def _bench_lm_suite(device: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # 1. GPT-2 (B=2, T=16, D=64, L=2)
    gpt_cfg = GPT2Config(
        vocab_size=256,
        max_position_embeddings=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    gpt = GPT2LMHeadModel(gpt_cfg).to(device).eval()
    gpt_opt = optim.AdamW(gpt.parameters(), lr=1e-3)
    ids = lucid.randint(0, 256, (2, 16), device=device)

    def _gpt_step() -> lucid.Tensor:
        gpt_opt.zero_grad()
        out = gpt.forward(ids, labels=ids)
        assert out.loss is not None
        out.loss.backward()
        gpt_opt.step()
        return out.loss

    rows.append(
        _bench_arch_pair(
            "gpt2_small_b2_t16",
            device,
            lambda: gpt.forward(ids).logits,
            _gpt_step,
            units_per_step=32,
            unit_key="tok_s",
            iters=20,
        )
    )

    # 2. T5 (B=2, T=16, D=64, L=2)
    t5_cfg = T5Config(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_decoder_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        head_dim=16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    t5 = T5ForSeq2SeqLM(t5_cfg).to(device).eval()
    t5_opt = optim.AdamW(t5.parameters(), lr=1e-3)

    def _t5_step() -> lucid.Tensor:
        t5_opt.zero_grad()
        out = t5.forward(input_ids=ids, labels=ids)
        assert out.loss is not None
        out.loss.backward()
        t5_opt.step()
        return out.loss

    rows.append(
        _bench_arch_pair(
            "t5_small_b2_t16",
            device,
            lambda: t5.forward(input_ids=ids, decoder_input_ids=ids).logits,
            _t5_step,
            units_per_step=32,
            unit_key="tok_s",
            iters=15,
        )
    )
    return rows


def _bench_gpt2_decode(device: str, *, compiled: bool) -> dict[str, object]:
    cfg = GPT2Config(
        vocab_size=256,
        max_position_embeddings=64,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=128,
    )
    gpt = GPT2LMHeadModel(cfg).to(device).eval()
    prompt = lucid.randint(0, 256, (1, 4), device=device).long()
    tok = lucid.tensor([[11]], device=device).long()
    cache = StaticCache(64)

    with lucid.no_grad():
        cp0 = lucid.arange(0, 4, device=device).long()
        _sync(gpt.forward(prompt, past_key_values=cache, use_cache=True, cache_position=cp0).logits)

        def _fwd(x: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
            return gpt.forward(x, past_key_values=cache, use_cache=True, cache_position=cp).logits

        step = compiled_decode_step(_fwd, cache) if compiled else _fwd
        pos_list = [lucid.tensor([4 + (i % 48)], device=device).long() for i in range(40)]

        for i in range(4):
            _sync(step(tok, pos_list[i]))
        times: list[int] = []
        for i in range(4, 24):
            t0 = time.perf_counter_ns()
            _sync(step(tok, pos_list[i]))
            times.append(time.perf_counter_ns() - t0)

    br = BenchResult("gpt2_decode", times)
    mean_ms = br.mean_us / 1_000.0
    tok_s = 1.0 / (br.mean_us * 1e-6)
    mode = "compiled" if compiled else "eager"
    return {
        "name": f"gpt2_static_decode_{mode}",
        "device": device,
        "mean_ms": round(mean_ms, 3),
        "p95_ms": round(br.p95_us / 1_000.0, 3),
        "tok_s": round(tok_s, 1),
    }


def _record_arch_row(out: dict[str, object], r: dict[str, object]) -> tuple[str, ...]:
    key = f"models/{r['name']}/{r['device']}"
    out[f"{key}/fwd_mean_ms"] = r["fwd_mean_ms"]
    out[f"{key}/step_mean_ms"] = r["step_mean_ms"]
    out[f"{key}/step_p95_ms"] = r["step_p95_ms"]
    unit_key = str(r["unit_key"])
    out[f"{key}/step_{unit_key}"] = r["rate"]
    unit_lbl = "samples/s" if unit_key == "samples_sec" else "tok/s"
    return (
        str(r["name"]),
        str(r["device"]),
        f"{r['fwd_mean_ms']:.2f}",
        f"{r['step_mean_ms']:.2f}",
        f"{r['rate']:.1f} {unit_lbl}",
    )


def run(verbose: bool = True) -> dict[str, object]:
    if verbose:
        print("\n── Representative architectures & StaticCache decode ────────────")

    has_gpu = metal_available()
    devices = ["cpu"] + (["metal"] if has_gpu else [])
    out: dict[str, object] = {}

    arch_rows: list[tuple[str, ...]] = [("workload", "device", "fwd ms", "step ms", "throughput")]
    for dev in devices:
        arch_rows.extend(
            _record_arch_row(out, r)
            for r in _bench_vision_and_transformer(dev) + _bench_lm_suite(dev)
        )

    decode_rows: list[tuple[str, ...]] = [("decode mode", "device", "ms/token", "p95 ms", "tok/s")]
    for dev in devices:
        modes = [False, True] if dev == "metal" else [False]
        for comp in modes:
            dr = _bench_gpt2_decode(dev, compiled=comp)
            dkey = f"models/{dr['name']}/{dev}"
            out[f"{dkey}/mean_ms"] = dr["mean_ms"]
            out[f"{dkey}/p95_ms"] = dr["p95_ms"]
            out[f"{dkey}/tok_s"] = dr["tok_s"]
            decode_rows.append(
                (
                    str(dr["name"]),
                    dev,
                    f"{dr['mean_ms']:.2f}",
                    f"{dr['p95_ms']:.2f}",
                    f"{dr['tok_s']:.1f}",
                )
            )

    if verbose:
        print(fmt_table(arch_rows))
        print("\nStaticCache single-token decode:")
        print(fmt_table(decode_rows))
    return out


if __name__ == "__main__":
    run()
