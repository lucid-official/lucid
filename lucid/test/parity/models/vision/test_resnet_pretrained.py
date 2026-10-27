"""Pretrained-weights integration parity — ResNet-18 IMAGENET1K_V1.

Exercises the full :mod:`lucid.weights` path end to end: a
``resnet_18_cls(pretrained=True)`` call downloads the hosted checkpoint
from the Hub, verifies its SHA-256, and loads it — then we assert the
resulting logits match the upstream vision source the weights were
converted from.

Network- and source-dependent, so it auto-skips when the reference vision
package is absent or the download is unreachable.  Marked ``parity`` +
``slow``.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.ref_framework import require_ref_vision

pytestmark = [pytest.mark.parity, pytest.mark.slow]

# H5: only the ref_framework fixture may name the reference framework;
# every other test receives it through that indirection.
_tv = require_ref_vision(module_level=True)


def _load_lucid_pretrained() -> object:
    """Build resnet_18_cls(pretrained=True), skipping on network failure."""
    from lucid.models.vision.resnet import resnet_18_cls

    try:
        model = resnet_18_cls(pretrained=True)
    except OSError as exc:  # network down / Hub unreachable; model failures must fail
        pytest.skip(f"pretrained download unavailable: {exc}")
    model.eval()
    return model


def test_resnet18_pretrained_matches_source(ref) -> None:
    """Lucid pretrained logits match the reference ResNet18 source."""
    resnet18 = _tv.models.resnet18
    ResNet18_Weights = _tv.models.ResNet18_Weights

    lucid_model = _load_lucid_pretrained()

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 224, 224)).astype("float32")

    ref_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    with ref.no_grad():
        ref_logits = ref_model(ref.from_numpy(x)).numpy()

    out_logits = lucid_model(lucid.tensor(x)).logits.numpy()

    assert out_logits.shape == ref_logits.shape == (1, 1000)
    assert int(out_logits.argmax()) == int(ref_logits.argmax())
    np.testing.assert_allclose(out_logits, ref_logits, atol=1e-4)


def test_resnet18_pretrained_stage_activations_match(ref) -> None:
    """Localise a conversion mismatch before pooling and the classifier hide it."""
    ours = _load_lucid_pretrained()
    source = _tv.models.resnet18(weights=_tv.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    wanted, got = {}, {}
    handles = []

    def record(destination, name):
        def hook(module, inputs, output):
            destination[name] = output.detach().numpy().copy()

        return hook

    pairs = [("stem", ours.stem, source.relu)] + [
        (name, getattr(ours, name), getattr(source, name))
        for name in ("layer1", "layer2", "layer3", "layer4", "avgpool")
    ]
    try:
        for name, own_layer, ref_layer in pairs:
            handles.append(own_layer.register_forward_hook(record(got, name)))
            handles.append(ref_layer.register_forward_hook(record(wanted, name)))
        values = np.random.default_rng(17).standard_normal((1, 3, 224, 224)).astype(np.float32)
        with lucid.no_grad(), ref.no_grad():
            ours(lucid.tensor(values))
            source(ref.from_numpy(values))
    finally:
        for handle in handles:
            handle.remove()
    assert set(got) == set(wanted) == {name for name, _, _ in pairs}
    for name in got:
        np.testing.assert_allclose(
            got[name],
            wanted[name],
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"first differing stage: {name}",
        )


def test_model_construction_failure_is_not_reported_as_a_download_skip(
    monkeypatch,
) -> None:
    import lucid.models.vision.resnet as resnet

    def broken(**kwargs):
        raise RuntimeError("checkpoint shape mismatch")

    monkeypatch.setattr(resnet, "resnet_18_cls", broken)
    with pytest.raises(RuntimeError, match="checkpoint shape mismatch"):
        _load_lucid_pretrained()


def test_resnet18_pretrained_sha_pins_artifact() -> None:
    """The enum's SHA-256 matches the hosted file (download verifies it)."""
    from lucid.models.vision.resnet import ResNet18Weights

    # Loading would have raised on a SHA mismatch; here we just assert the
    # entry is fully specified (non-empty pinned hash + Hub URL).
    w = ResNet18Weights.IMAGENET1K_V1
    assert len(w.sha256) == 64
    assert w.url.startswith("https://huggingface.co/lucid-dl/resnet-18/")
    _load_lucid_pretrained()  # exercises download + SHA verify (or skips)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "parity or slow"])
