"""Zoo paths that built a tensor without the model's device.

Every case here worked on the CPU and raised ``DeviceMismatch`` (or simply
ran on the wrong device) once the model was moved to Metal, because some
helper tensor was created with no ``device=`` and so defaulted to the CPU:

* YOLO v1-v4 ``postprocess`` — the NMS class ids, selection indices and the
  empty result (CHA-195); v1/v2 training — the loss targets (CHA-195).
* ``yolo_v3_tiny`` returned a private class with no ``postprocess``
  (CHA-201), and v1/v2 read the score matrix back one ``.item()`` per
  anchor and class.
* I-JEPA / V-JEPA ``gather_tokens`` broadcast helper (CHA-196).
* DiT ``generate`` final-step ``alpha_bar_prev`` (CHA-197).
* ``DiffusionMixin.generate`` (DDPM) and ``RSSM.initial`` defaulted
  ``device`` to ``"cpu"`` instead of following the model (CHA-202).

Each test runs on every available device through the ``device`` fixture, and
every model is a real factory shrunk by config overrides.
"""

import math
from typing import Any

import pytest

import lucid
import lucid.models as M
from lucid._tensor.tensor import Tensor
from lucid.models._registry import _registry_lookup


def _on(t: Tensor, device: str) -> bool:
    return t.device.type == device


# ─────────────────────────────────────────────────────────────────────────────
# YOLO (CHA-195, CHA-201)
# ─────────────────────────────────────────────────────────────────────────────

_SIDE = 64

# (factory, overrides, postprocess call style)
_YOLO_CASES = [
    ("yolo_tiny", {"split_size": 2, "num_classes": 3}, "v1"),
    ("yolo_v2", {"num_classes": 3}, "v2"),
    ("yolo_v3", {"num_classes": 3}, "v3"),
    ("yolo_v3_tiny", {"num_classes": 3}, "v3"),
    ("yolo_v4", {"num_classes": 3}, "v3"),
]
_YOLO_IDS = [name for name, _, _ in _YOLO_CASES]


def _yolo(name: str, device: str, **overrides: object) -> Any:
    lucid.manual_seed(0)
    return M.create_model(name, **overrides).to(device).eval()


def _yolo_detect(model: Any, style: str, device: str) -> list[dict[str, Tensor]]:
    lucid.manual_seed(0)
    x = lucid.randn((2, 3, _SIDE, _SIDE), device=device)
    with lucid.no_grad():
        if style == "v1":
            out = model(x, image_size=(_SIDE, _SIDE))
            return model.postprocess(out, image_size=(_SIDE, _SIDE))
        out = model(x)
        if style == "v2":
            return model.postprocess(out, image_size=(_SIDE, _SIDE))
        return model.postprocess(out, image_sizes=[(_SIDE, _SIDE)] * 2)


class TestYOLOPostprocess:
    @pytest.mark.parametrize("name,overrides,style", _YOLO_CASES, ids=_YOLO_IDS)
    def test_detections_stay_on_the_model_device(
        self, name: str, overrides: dict[str, object], style: str, device: str
    ) -> None:
        # score_thresh=0 forces detections, so NMS actually runs — that is
        # the path whose class-id tensor was built on the CPU.
        model = _yolo(name, device, score_thresh=0.0, **overrides)
        results = _yolo_detect(model, style, device)
        assert len(results) == 2
        for res in results:
            n = int(res["boxes"].shape[0])
            assert n > 0, name
            assert tuple(res["boxes"].shape) == (n, 4)
            assert int(res["scores"].shape[0]) == n
            assert int(res["labels"].shape[0]) == n
            for key in ("boxes", "scores", "labels"):
                assert _on(res[key], device), (name, key)

    @pytest.mark.parametrize("name,overrides,style", _YOLO_CASES, ids=_YOLO_IDS)
    def test_the_empty_result_is_on_the_model_device(
        self, name: str, overrides: dict[str, object], style: str, device: str
    ) -> None:
        # No score can reach 2, so every image takes the empty branch.
        model = _yolo(name, device, score_thresh=2.0, **overrides)
        for res in _yolo_detect(model, style, device):
            assert tuple(res["boxes"].shape) == (0, 4)
            assert tuple(res["scores"].shape) == (0,)
            assert tuple(res["labels"].shape) == (0,)
            for key in ("boxes", "scores", "labels"):
                assert _on(res[key], device), (name, key)

    @pytest.mark.parametrize("name,overrides,style", _YOLO_CASES, ids=_YOLO_IDS)
    def test_thresholding_reads_the_scores_back_once(
        self,
        name: str,
        overrides: dict[str, object],
        style: str,
        device: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The old v1/v2 (and v4) scan called .item() once per anchor and
        # class — 67,600 device round-trips for v2 at 416px.  With nothing
        # clearing the threshold, no scalar read-back is needed at all.
        model = _yolo(name, device, score_thresh=2.0, **overrides)
        lucid.manual_seed(0)
        x = lucid.randn((1, 3, _SIDE, _SIDE), device=device)
        with lucid.no_grad():
            out = model(x, image_size=(_SIDE, _SIDE)) if style == "v1" else model(x)

        calls = 0
        original = Tensor.item

        def counting_item(self: Tensor) -> object:
            nonlocal calls
            calls += 1
            return original(self)

        monkeypatch.setattr(Tensor, "item", counting_item)
        if style == "v3":
            model.postprocess(out, image_sizes=[(_SIDE, _SIDE)])
        else:
            model.postprocess(out, image_size=(_SIDE, _SIDE))
        assert calls == 0, f"{name}: {calls} .item() calls"


class TestYOLOV3TinyIsPublic:
    def test_the_factory_returns_the_exported_class(self) -> None:
        from lucid.models.vision import yolo

        model = M.create_model("yolo_v3_tiny", num_classes=3)
        assert type(model) is yolo.YOLOV3TinyForObjectDetection
        assert "YOLOV3TinyForObjectDetection" in yolo.__all__
        assert callable(getattr(model, "postprocess", None))

    def test_the_registry_names_the_public_class(self) -> None:
        from lucid.models.vision.yolo import YOLOV3TinyForObjectDetection

        entry = _registry_lookup("yolo_v3_tiny", task="object-detection")
        assert entry.model_class is YOLOV3TinyForObjectDetection
        assert not entry.model_class.__name__.startswith("_")


def _yolo_targets(device: str) -> list[dict[str, Tensor]]:
    return [
        {
            "boxes": lucid.tensor([[0.1, 0.1, 0.5, 0.5], [0.4, 0.3, 0.9, 0.8]], device=device),
            "labels": lucid.tensor([1, 2], device=device).long(),
        },
        {
            "boxes": lucid.tensor([[0.2, 0.2, 0.6, 0.7]], device=device),
            "labels": lucid.tensor([0], device=device).long(),
        },
    ]


class TestYOLOTrainsOnDevice:
    @pytest.mark.parametrize(
        "name,overrides,style",
        [_YOLO_CASES[0], _YOLO_CASES[1]],
        ids=_YOLO_IDS[:2],
    )
    def test_one_step(
        self, name: str, overrides: dict[str, object], style: str, device: str
    ) -> None:
        # v1 and v2 built every loss target from a bare lucid.tensor(...).
        lucid.manual_seed(0)
        model = M.create_model(name, **overrides).to(device).train()
        optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
        x = lucid.randn((2, 3, _SIDE, _SIDE), device=device)

        if style == "v1":
            out = model(x, targets=_yolo_targets(device), image_size=(_SIDE, _SIDE))
        else:
            out = model(x, targets=_yolo_targets(device))
        assert out.loss is not None
        assert _on(out.loss, device)
        assert math.isfinite(float(out.loss.item()))

        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, name
        assert all(_on(g, device) for g in grads), name


# ─────────────────────────────────────────────────────────────────────────────
# I-JEPA / V-JEPA (CHA-196)
# ─────────────────────────────────────────────────────────────────────────────

_IJEPA_SMALL: dict[str, object] = {
    "image_size": 32,
    "patch_size": 8,
    "dim": 16,
    "depth": 1,
    "num_heads": 2,
    "predictor_dim": 8,
    "predictor_depth": 1,
    "min_keep": 1,
    "target_scale": (0.1, 0.2),
    "context_scale": (0.8, 1.0),
}

_VJEPA_SMALL: dict[str, object] = {
    "image_size": 32,
    "patch_size": 8,
    "tubelet_size": 2,
    "num_frames": 4,
    "dim": 24,
    "depth": 1,
    "num_heads": 2,
    "predictor_dim": 12,
    "predictor_depth": 1,
    "short_range_blocks": 2,
    "long_range_blocks": 1,
}


def test_gather_tokens_on_device(device: str) -> None:
    from lucid.models.vision._common._transformers import gather_tokens

    tokens = lucid.arange(24, device=device).float().reshape(2, 4, 3)
    indices = lucid.tensor([[3, 0], [1, 1]], device=device).long()
    picked = gather_tokens(tokens, indices)
    assert _on(picked, device)
    assert picked.to("cpu").tolist() == [
        [[9.0, 10.0, 11.0], [0.0, 1.0, 2.0]],
        [[15.0, 16.0, 17.0], [15.0, 16.0, 17.0]],
    ]


@pytest.mark.parametrize(
    "name,overrides,shape",
    [
        ("ijepa_base_16", _IJEPA_SMALL, (2, 3, 32, 32)),
        ("vjepa_large_16", _VJEPA_SMALL, (2, 4, 3, 32, 32)),
    ],
    ids=["ijepa", "vjepa"],
)
def test_jepa_trains_one_step_on_device(
    name: str, overrides: dict[str, object], shape: tuple[int, ...], device: str
) -> None:
    lucid.manual_seed(0)
    model = M.create_model(name, **overrides).to(device).train()
    optimizer = lucid.optim.Adam(model.trainable_parameters(), lr=1e-3)

    out = model(lucid.rand(*shape, device=device))
    assert _on(out.loss, device)
    assert math.isfinite(float(out.loss.item()))

    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    model.update_target(model.momentum(0, 10))

    grads = [p.grad for p in model.trainable_parameters() if p.grad is not None]
    assert grads, name
    assert all(_on(g, device) for g in grads), name


_VJEPA2_SMALL: dict[str, object] = {
    "image_size": 16,
    "patch_size": 8,
    "tubelet_size": 2,
    "num_frames": 4,
    "dim": 24,
    "depth": 1,
    "num_heads": 2,
    "predictor_dim": 12,
    "predictor_depth": 1,
    "predictor_heads": 2,
    "predictor_num_mask_tokens": 2,
}


@pytest.mark.parametrize("use_rope", [True, False], ids=["rope", "sincos"])
def test_vjepa2_trains_and_encodes_on_device(use_rope: bool, device: str) -> None:
    """LCD-639: without RoPE the encoder and the predictor add a sine/cosine
    table, which was built on the CPU in float64 — a ``DeviceMismatch`` on
    Metal.  Only the rotary path ran there."""
    lucid.manual_seed(0)
    model = M.create_model("vjepa2_vit_large", use_rope=use_rope, **_VJEPA2_SMALL)
    model = model.to(device).train()
    optimizer = lucid.optim.Adam(model.trainable_parameters(), lr=1e-3)
    # A 2-frame-deep 2x2 grid: eight tokens, the last two held out.
    context = lucid.arange(6, dtype=lucid.int32, device=device).reshape(1, 6)
    target = lucid.arange(6, 8, dtype=lucid.int32, device=device).reshape(1, 2)

    out = model(lucid.rand(1, 4, 3, 16, 16, device=device), context, target)
    assert _on(out.loss, device)
    assert math.isfinite(float(out.loss.item()))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    model.update_target(model.momentum(0, 10))

    grads = [p.grad for p in model.trainable_parameters() if p.grad is not None]
    assert grads and all(_on(g, device) for g in grads)
    with lucid.no_grad():
        features = model.eval().encode(lucid.rand(1, 4, 3, 16, 16, device=device))
    assert _on(features, device) and tuple(features.shape) == (1, 24)


# ─────────────────────────────────────────────────────────────────────────────
# DiT (CHA-197)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_dit_generates_on_device(device: str, eta: float) -> None:
    # The last reverse step used a bare CPU alpha_bar_prev = 1, so every
    # run reached it and raised.
    lucid.manual_seed(0)
    model = M.create_model("dit_small_2_gen", sample_size=8, hidden_size=32, depth=1, num_heads=4)
    model = model.to(device).eval()
    samples = model.generate(1, steps=2, eta=eta).samples
    assert tuple(samples.shape) == (1, 4, 8, 8)
    assert _on(samples, device)
    assert not any(math.isnan(v) for v in samples.to("cpu").reshape(-1).tolist())


# ─────────────────────────────────────────────────────────────────────────────
# DDPM / RSSM default device (CHA-202)
# ─────────────────────────────────────────────────────────────────────────────

_DDPM_SMALL: dict[str, object] = {
    "sample_size": 16,
    "base_channels": 16,
    "channel_mult": (1, 2),
    "num_res_blocks": 1,
    "attention_resolutions": (8,),
    "num_heads": 2,
    "resnet_groups": 8,
    "num_train_timesteps": 20,
}

_PLANET_SMALL: dict[str, object] = {
    "action_dim": 2,
    "stoch_size": 4,
    "deter_size": 8,
    "hidden_size": 8,
    "cnn_depth": 4,
    "reward_hidden": 8,
}


def test_ddpm_generate_follows_the_model_device(device: str) -> None:
    from lucid.models.generative import DDPMScheduler

    lucid.manual_seed(0)
    model = M.create_model("ddpm_cifar_gen", **_DDPM_SMALL).to(device).eval()
    out = model.generate(DDPMScheduler(num_train_timesteps=20), n_samples=1, num_inference_steps=3)
    assert tuple(out.samples.shape) == (1, 3, 16, 16)
    assert _on(out.samples, device)


def test_ddpm_generate_honours_an_explicit_device() -> None:
    from lucid.models.generative import DDPMScheduler

    lucid.manual_seed(0)
    model = M.create_model("ddpm_cifar_gen", **_DDPM_SMALL).eval()
    out = model.generate(
        DDPMScheduler(num_train_timesteps=20),
        n_samples=1,
        num_inference_steps=2,
        device="cpu",
    )
    assert _on(out.samples, "cpu")


def test_rssm_initial_follows_the_model_device(device: str) -> None:
    lucid.manual_seed(0)
    model = M.create_model("planet", **_PLANET_SMALL).to(device).eval()
    start = model.rssm.initial(2)
    for part in (start.deter, start.stoch):
        assert _on(part, device)

    # The default start state must feed the recurrence without a move.
    imagined = model.imagine(start, lucid.rand((2, 3, 2), device=device))
    assert _on(imagined.stoch, device)
    assert tuple(imagined.stoch.shape) == (2, 3, 4)


def test_rssm_initial_honours_an_explicit_device(device: str) -> None:
    model = M.create_model("planet", **_PLANET_SMALL).to(device)
    start = model.rssm.initial(1, device="cpu")
    assert _on(start.deter, "cpu")
    assert _on(start.stoch, "cpu")
