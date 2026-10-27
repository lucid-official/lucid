"""Invalid outputs and oversized models cannot earn a parity verdict."""

import numpy as np
import pytest
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

from tools import check_pretrained_parity as parity


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_outputs_are_errors(bad: float) -> None:
    wanted = np.array([[2.0, 1.0]])
    got = np.array([[bad, 1.0]])
    assert "error" in parity._compare_arrays(wanted, got)
    assert "error" in parity._compare_arrays(got, wanted)


def test_broadcastable_shape_mismatch_is_an_error() -> None:
    assert "error" in parity._compare_arrays(np.ones((2, 3)), np.ones((1, 3)))


def test_empty_and_zero_outputs_do_not_count_as_evidence() -> None:
    for values in (np.zeros((1, 3)), np.empty((0, 3))):
        assert "degenerate" in parity._compare_arrays(values, values)
    assert "degenerate" in parity._compare_fields([])


def test_real_disagreement_remains_measurable() -> None:
    result = parity._compare_arrays(np.array([[2.0, 1.0]]), np.array([[1.0, 3.0]]))
    assert result["max_diff"] == 2.0
    assert result["top1_agrees"] is False


def test_semantic_class_axis_is_channels_not_image_width() -> None:
    wanted = np.array([[[[3.0, 3.000001]], [[1.0, 1.000001]]]], dtype=np.float32)
    got = wanted[..., ::-1].copy()
    result = parity._compare_fields([("semantic_logits", wanted, got)])
    assert result["top1_agrees"] is True
    assert result["max_diff"] < result["tolerance"]
    assert (
        parity._compare_fields([("semantic_logits", wanted, wanted[:, ::-1])])["top1_agrees"]
        is False
    )
    assert parity._compare_fields([("masks_queries_logits", wanted, got)])["top1_agrees"] is True


def test_parameter_limit_applies_to_vision_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("a model beyond the memory bound must not be loaded")

    monkeypatch.setattr(parity, "create_model", forbidden)
    result = parity._compare("unused", "timm/unused", (1, 3, 224, 224), 100, 99)
    assert "unreachable" in result


def test_default_selection_and_partial_evidence_survive_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.json"
    wrong = SimpleNamespace(value=SimpleNamespace(meta={"source": "timm/wrong", "num_params": 1}))
    correct = SimpleNamespace(
        value=SimpleNamespace(meta={"source": "timm/default", "num_params": 1})
    )
    enum = SimpleNamespace(
        DEFAULT=correct,
        __members__={"OLD": wrong, "CURRENT": correct, "DEFAULT": correct},
    )
    monkeypatch.setattr(parity, "_WEIGHTS_BY_MODEL", {"first": enum, "second": enum})
    monkeypatch.setattr(parity, "_factory_default", lambda name: correct)
    monkeypatch.setattr(parity.sys, "argv", ["parity", "--json", str(path)])

    def compare(name, source, *args):
        assert source == "timm/default"
        assert parity.os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"
        assert parity.socket.getdefaulttimeout() == 60.0
        if name == "second":
            report = json.loads(path.read_text())
            assert report["processed"] == report["compared"] == 1
            assert report["finished"] is report["complete"] is False
            raise KeyboardInterrupt
        return {"max_diff": 0.0, "scale": 1.0, "top1_agrees": True}

    monkeypatch.setattr(parity, "_compare", compare)
    with pytest.raises(KeyboardInterrupt):
        parity.main()
    assert json.loads(path.read_text())["evidence"][0]["model"] == "first"


@pytest.mark.parametrize("timeout,previous", [(None, None), (17.0, "0"), (90.0, "1")])
@pytest.mark.parametrize("interrupted", [False, True])
def test_reference_io_is_read_only_bounded_and_restores_caller_settings(
    monkeypatch,
    timeout,
    previous,
    interrupted,
) -> None:
    import socket

    current = [timeout]
    monkeypatch.setattr(socket, "getdefaulttimeout", lambda: current[0])
    monkeypatch.setattr(socket, "setdefaulttimeout", lambda value: current.__setitem__(0, value))
    if previous is None:
        monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    else:
        monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", previous)
    with pytest.raises(KeyboardInterrupt) if interrupted else nullcontext():
        with parity._reference_io():
            assert parity.os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"
            assert socket.getdefaulttimeout() == (60.0 if timeout is None else min(timeout, 60.0))
            if interrupted:
                raise KeyboardInterrupt
    assert socket.getdefaulttimeout() == timeout
    assert parity.os.environ.get("DISABLE_SAFETENSORS_CONVERSION") == previous


@pytest.mark.parametrize(
    "source,kind,identifier",
    [
        ("diffusers/google/ddpm-cifar10-32", "diffusers", "google/ddpm-cifar10-32"),
        (
            "facebook/maskformer-resnet50-ade",
            "transformers",
            "facebook/maskformer-resnet50-ade",
        ),
        (
            "facebook/mask2former-swin-tiny-ade-semantic",
            "transformers",
            "facebook/mask2former-swin-tiny-ade-semantic",
        ),
    ],
)
def test_existing_checkpoint_sources_select_their_actual_adapter(source, kind, identifier) -> None:
    assert parity._reference_for(source) == (kind, identifier)


def test_denoising_adapter_uses_identical_samples_and_fixed_timestep(
    monkeypatch,
) -> None:
    import lucid

    seen = []

    class Model:
        config = SimpleNamespace(sample_size=4, in_channels=3)

        def eval(self):
            return self

        def __call__(self, values, timestep):
            seen.append(
                (
                    values.numpy(),
                    (int(timestep.item()) if isinstance(timestep, lucid.Tensor) else timestep),
                )
            )
            return SimpleNamespace(sample=values)

    model = Model()
    monkeypatch.setattr(
        parity,
        "_REGISTRY",
        {"ddpm": SimpleNamespace(model_class=SimpleNamespace(__name__="DDPMModel"))},
    )
    monkeypatch.setattr(parity, "create_model", lambda *a, **k: model)
    monkeypatch.setattr(
        parity,
        "ref_module",
        lambda: SimpleNamespace(from_numpy=lucid.from_numpy, no_grad=nullcontext),
    )
    original_import = parity.importlib.import_module
    monkeypatch.setattr(
        parity.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(UNet2DModel=SimpleNamespace(from_pretrained=lambda repo: model))
            if name == "diffusers"
            else original_import(name)
        ),
    )
    result = parity._compare("ddpm", "diffusers/google/ddpm-cifar10-32", (1, 3, 4, 4), 1)
    assert result["max_diff"] == 0
    assert result["fields"] == ["denoising_sample_t500"]
    assert seen[0][1] == seen[1][1] == 500
    np.testing.assert_array_equal(seen[0][0], seen[1][0])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-params", "nan"],
        ["--max-params", "inf"],
        ["--max-params", "0"],
        ["--limit", "0"],
        ["--limit", "-1"],
    ],
)
def test_invalid_resource_bounds_fail_before_loading(monkeypatch, arguments) -> None:
    monkeypatch.setattr(parity.sys, "argv", ["parity", *arguments])
    with pytest.raises(SystemExit) as raised:
        parity.main()
    assert raised.value.code == 2


def test_cleanup_process_owns_caches_without_changing_parent(monkeypatch) -> None:
    import os

    monkeypatch.setenv("HF_XET_CACHE", "/outside/custom-xet-cache")
    monkeypatch.setenv("HF_ASSETS_CACHE", "/outside/custom-assets-cache")
    before = os.environ.copy()
    directories = []

    def run(command, *, env, check):
        root = Path(command[-1])
        directories.append(root)
        assert root.is_dir()
        assert command[-2] == "--_owned-cache"
        for key in (
            "LUCID_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_XET_CACHE",
            "HF_ASSETS_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
        ):
            assert Path(env[key]).is_relative_to(root)
        assert check is False
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(parity.subprocess, "run", run)
    assert parity._isolated_run(["--clean-downloads", "--model", "ddpm_cifar"]) == 7
    assert not directories[0].exists()
    assert os.environ == before


@pytest.mark.parametrize("explicit", [False, True])
def test_private_download_cache_keeps_client_auth_location_without_reading_token(
    monkeypatch, explicit: bool
) -> None:
    import os
    import sys
    from types import ModuleType

    provider = ModuleType("huggingface_hub")
    provider.constants = SimpleNamespace(HF_TOKEN_PATH="/client-config/token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", provider)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    if explicit:
        monkeypatch.setenv("HF_TOKEN_PATH", "/explicit-config/token")
    before = os.environ.copy()

    def run(command, *, env, check):
        expected = "/explicit-config/token" if explicit else "/client-config/token"
        assert env["HF_TOKEN_PATH"] == expected
        assert env.get("HF_TOKEN") == before.get("HF_TOKEN")
        assert not Path(expected).exists()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(parity.subprocess, "run", run)
    assert parity._isolated_run(["--clean-downloads"]) == 0
    assert os.environ == before


def test_factory_default_can_differ_from_shared_enum_default() -> None:
    from lucid.models.generative.diamond._weights import DIAMONDWeights
    import lucid.weights as weights

    original_resolver = weights.resolve_weights
    assert parity._factory_default("diamond_csgo") is DIAMONDWeights.CSGO
    assert parity._factory_default("diamond") is DIAMONDWeights.DEFAULT
    assert weights.resolve_weights is original_resolver
    assert parity._params_of("diamond_csgo", DIAMONDWeights.CSGO) == 381_642_502


def test_stable_diffusion_wrapper_weights_are_discoverable() -> None:
    from lucid.models.generative.stable_diffusion import StableDiffusionWeights
    from lucid.weights import list_pretrained, weights_for

    assert weights_for("stable_diffusion_gen") is StableDiffusionWeights
    assert list_pretrained("stable_diffusion_gen") == list_pretrained("stable_diffusion")
    assert parity._factory_default("stable_diffusion_gen") is StableDiffusionWeights.DEFAULT


@pytest.mark.parametrize(
    "base",
    [
        "clip_vit_base_32",
        "clip_vit_base_16",
        "clip_vit_large_14",
        "clip_vit_large_14_336",
    ],
)
def test_clip_zero_shot_weights_are_discoverable(base: str) -> None:
    from lucid.weights import list_pretrained, weights_for

    wrapper = base + "_zero_shot"
    assert weights_for(wrapper) is weights_for(base)
    assert list_pretrained(wrapper) == list_pretrained(base)
    assert parity._factory_default(wrapper) is parity._factory_default(base)
    assert parity._HF_HEADS["CLIPForZeroShotImageClassification"] == (
        "CLIPModel",
        ("image_embeds", "text_embeds", "logits"),
    )


def test_missing_darknet_reader_is_unverified_not_a_conversion_failure(
    monkeypatch,
) -> None:
    original_import = parity.importlib.import_module
    monkeypatch.setattr(
        parity.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(dnn=SimpleNamespace()) if name == "cv2" else original_import(name)
        ),
    )
    result = parity._compare_darknet("yolo_v3_tiny", (1, 3, 224, 224))
    assert "unreachable" in result
    assert "error" not in result


def test_darknet_epsilon_is_adapted_before_import_without_editing_checkpoint(
    tmp_path,
) -> None:
    import struct

    path = tmp_path / "original.weights"
    payload = np.arange(13, dtype=np.float32)
    original = struct.pack("<iiiQ", 0, 2, 0, 0) + payload.tobytes()
    path.write_bytes(original)
    got = parity._darknet_oracle_blob(path, [(2, 3, True), (1, 1, False)])
    result = np.frombuffer(got, dtype="<f4", offset=20)
    np.testing.assert_allclose(result[6:8] + 1e-6, payload[6:8] + 1e-5, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result[:6], payload[:6])
    np.testing.assert_array_equal(result[8:], payload[8:])
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="full payload"):
        parity._darknet_oracle_blob(path, [(2, 3, True)])


def test_original_diamond_modules_are_isolated_and_removed_on_failure(
    monkeypatch,
) -> None:
    import io
    import sys

    prefix = "_lucid_parity_diamond_reference"
    seen = []

    def fetch(url, *, timeout):
        seen.append(url)
        assert timeout == 30
        source = b"VALUE = 17\n" if url.endswith("blocks.py") else b"from ..blocks import VALUE\n"
        return io.BytesIO(source)

    monkeypatch.setattr(parity.urllib.request, "urlopen", fetch)
    with pytest.raises(RuntimeError, match="comparison failed"):
        with parity._diamond_reference("pinned-revision") as module:
            assert module.VALUE == 17
            assert prefix in sys.modules
            raise RuntimeError("comparison failed")
    assert not any(name.startswith(prefix) for name in sys.modules)
    assert len(seen) == 2
    assert all("/pinned-revision/" in url for url in seen)


def test_darknet_download_has_a_read_timeout_and_verifies_bytes(monkeypatch, tmp_path) -> None:
    import hashlib
    import importlib
    import io

    # The YOLO converter imports only Lucid, NumPy and the standard library,
    # so fetching and verifying bytes needs no oracle and runs in every tier.
    converter = importlib.import_module("tools.convert_weights.yolo")
    data = b"checkpoint bytes"

    def fetch(request, *, timeout):
        assert timeout == 30
        return io.BytesIO(data)

    monkeypatch.setenv("LUCID_HOME", str(tmp_path))
    monkeypatch.setattr(converter.urllib.request, "urlopen", fetch)
    path = converter._fetch(
        "https://example.invalid/model.weights", hashlib.sha256(data).hexdigest()
    )
    assert path.read_bytes() == data
    with pytest.raises(RuntimeError, match="digest"):
        converter._fetch("https://example.invalid/wrong.weights", "0" * 64)
    assert not (tmp_path / "darknet-src" / "wrong.weights").exists()


def test_dit_original_frequency_contract_applies_to_every_block() -> None:
    projections = [SimpleNamespace(downscale_freq_shift=1) for _ in range(3)]
    model = SimpleNamespace(
        transformer_blocks=[
            SimpleNamespace(norm1=SimpleNamespace(emb=SimpleNamespace(time_proj=p)))
            for p in projections
        ]
    )
    parity._dit_original_frequencies(model)
    assert all(p.downscale_freq_shift == 0 for p in projections)
    projections[0].downscale_freq_shift = 7
    with pytest.raises(ValueError, match="frequency convention"):
        parity._dit_original_frequencies(model)
