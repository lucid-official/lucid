"""``lucid.peft`` models in Core ML: the adapter exports, merged or not, and the base ships once.

Three claims, each against a real package:

* **Predictions.**  A ``PeftModel`` exported as it is (the LoRA branch
  traced as two small linears and an add), and the plain model
  ``merge_and_unload()`` returns, both predict as eager does — 1e-4 of
  the output's scale in float32, 1e-2 in float16.
* **One copy of the base.**  A LoRA layer keeps the base layer's own
  ``Parameter`` objects, and ``export_functions`` writes a parameter once
  per identity.  So a package of ``base`` + ``adapted`` weighs what the
  adapted model alone does, and a second adapter over the same base
  costs its own adapter bytes and nothing more.  The weights file is
  read entry by entry (its layout is in ``BlobWriter.h``), so the
  assertions are exact rather than "about".
* **What breaks it.**  Sharing is by identity: a deep copy of the base
  ships its weights again, and a compressed weight is not shared at all
  (LCD-688, a strict xfail below).

Measured on the 64-64-32 net used here (float32 weights):

    base alone                                  25,280 B  (weights file)
    + one rank-4 adapter, unmerged              30,144 B   (+4,864)
    merged (``merge_and_unload``)               25,280 B   (the base again)
    base + adapted, one package                 30,144 B   (== adapted alone)
    base + adapter A (r=4) + adapter B (r=8)    38,592 B   (A + B - base)
    the same three as separate packages         89,152 B

An adapter costs its parameters plus a zero bias per bias-free linear and
a four-byte scale each; at rank 4 that is about 12 % over the parameter
count (the zero bias of ``B`` is one row in ``r``), and falls as the rank
grows.
"""

import copy
import os
import struct
from collections.abc import Callable

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.peft import LoraConfig, PeftModel, get_peft_model

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

#: Output gap, as a fraction of the output's largest magnitude.
TOL = 1e-4
TOL_FLOAT16 = 1e-2

#: ``weights/weight.bin``: a 64-byte file header, then per entry a 64-byte
#: header (``0xDEADBEEF``, dtype, size, data offset) and the payload,
#: padded to 64.
_ENTRY_MAGIC = 0xDEADBEEF


class _Net(nn.Module):
    """Two linear layers around a tanh — the two LoRA targets."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).tanh())


class _AdapterOff(nn.Module):
    """The adapted model with its adapter off — the base function of a shared package."""

    def __init__(self, peft: PeftModel) -> None:
        super().__init__()
        self.peft = peft
        self.eval()  # export refuses a model in training mode

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        with self.peft.disable_adapter():
            return self.peft(x)


def _config(r: int = 4, lora_dropout: float = 0.0) -> LoraConfig:
    # ``B`` random: the default zero ``B`` would make the adapter a no-op to export.
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=["fc1", "fc2"],
        lora_dropout=lora_dropout,
        init_lora_weights=False,
    )


def _example() -> lucid.Tensor:
    lucid.manual_seed(7)
    return lucid.randn(2, 64)


def _base() -> _Net:
    lucid.manual_seed(0)
    return _Net().eval()


def _peft(r: int = 4, lora_dropout: float = 0.0) -> PeftModel:
    return get_peft_model(_base(), _config(r, lora_dropout)).eval()


def _sharing_the_base_of(peft: PeftModel) -> _Net:
    """A second ``_Net`` holding the very ``Parameter`` objects of ``peft``'s base."""
    other = _Net()
    for name, param in peft.get_base_model().named_parameters():
        if "lora_" in name:
            continue
        owner, _, leaf = name.replace("base_layer.", "").rpartition(".")
        setattr(other.get_submodule(owner), leaf, param)
    return other


def _scale(t: lucid.Tensor) -> float:
    return max(float(t.abs().max().item()), 1e-6)


def _assert_predicts(got: lucid.Tensor, want: lucid.Tensor, tol: float = TOL) -> None:
    assert got.shape == want.shape
    gap = float((got - want).abs().max().item())
    assert gap <= tol * _scale(want), f"gap {gap:.3e} over {tol} of scale {_scale(want):.3e}"


def _eager(model: nn.Module, x: lucid.Tensor) -> lucid.Tensor:
    with lucid.no_grad():
        return model(x)


def _weights_file(package: str) -> str:
    return os.path.join(package, "Data", "com.apple.CoreML", "weights", "weight.bin")


def _entries(package: str) -> list[int]:
    """Payload size of every entry in the package's weights file, in file order."""
    with open(_weights_file(package), "rb") as handle:
        raw = handle.read()
    sizes: list[int] = []
    at = 64
    while at < len(raw):
        magic, _dtype, size, offset = struct.unpack_from("<IIQQ", raw, at)
        assert magic == _ENTRY_MAGIC, f"no entry header at {at}"
        sizes.append(size)
        at = (offset + size + 63) // 64 * 64
    return sizes


def _weight_bytes(package: str) -> int:
    return os.path.getsize(_weights_file(package))


def _package_bytes(package: str) -> int:
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, names in os.walk(package)
        for name in names
    )


def _floats(model: nn.Module, picks: Callable[[str], bool]) -> int:
    return sum(p.numel() for name, p in model.named_parameters() if picks(name))


def _close(*handles: cml.CoreMLModel) -> None:
    for handle in handles:
        handle.close()


# ── predictions ───────────────────────────────────────────────────


class TestPredictions:
    def test_an_unmerged_peft_model_predicts_as_eager(self, tmp_path: object) -> None:
        peft = _peft()
        x = _example()
        want = _eager(peft, x)
        with peft.disable_adapter():
            base_only = _eager(peft, x)
        assert float((want - base_only).abs().max().item()) > 1e-2, "the adapter changes nothing"
        exported = cml.export(peft, x, f"{tmp_path}/unmerged.mlpackage")
        try:
            _assert_predicts(exported.predict(x), want)
            fresh = lucid.randn(2, 64) * 3
            _assert_predicts(exported.predict(fresh), _eager(peft, fresh))
        finally:
            exported.close()

    def test_the_merged_model_predicts_as_the_adapted_one(self, tmp_path: object) -> None:
        peft = _peft()
        x = _example()
        want = _eager(peft, x)
        plain = peft.merge_and_unload().eval()
        assert type(plain) is _Net
        exported = cml.export(plain, x, f"{tmp_path}/merged.mlpackage")
        try:
            _assert_predicts(exported.predict(x), want)
        finally:
            exported.close()

    def test_the_adapter_is_what_the_unmerged_package_adds(self, tmp_path: object) -> None:
        """Exported as the base alone, the same net predicts differently."""
        peft = _peft()
        x = _example()
        adapted = cml.export(peft, x, f"{tmp_path}/adapted.mlpackage")
        base = cml.export(_AdapterOff(peft), x, f"{tmp_path}/off.mlpackage")
        try:
            with peft.disable_adapter():
                _assert_predicts(base.predict(x), _eager(peft, x))
            apart = float((adapted.predict(x) - base.predict(x)).abs().max().item())
            assert apart > 1e-2 * _scale(adapted.predict(x))
        finally:
            _close(adapted, base)

    def test_a_loaded_adapter_exports(self, tmp_path: object) -> None:
        """``from_pretrained`` gives a frozen adapter in eval mode; it exports like a trained one."""
        trained = _peft()
        x = _example()
        want = _eager(trained, x)
        trained.save_pretrained(f"{tmp_path}/adapter")
        loaded = PeftModel.from_pretrained(_base(), f"{tmp_path}/adapter")
        exported = cml.export(loaded, x, f"{tmp_path}/loaded.mlpackage")
        try:
            _assert_predicts(exported.predict(x), want)
        finally:
            exported.close()

    def test_a_dropout_layer_is_a_no_op_in_an_eval_export(self, tmp_path: object) -> None:
        peft = _peft(lora_dropout=0.25)
        x = _example()
        exported = cml.export(peft.eval(), x, f"{tmp_path}/dropout.mlpackage")
        try:
            _assert_predicts(exported.predict(x), _eager(peft, x))
        finally:
            exported.close()

    def test_a_float16_package_with_an_adapter_predicts_to_half_precision(
        self, tmp_path: object
    ) -> None:
        peft = _peft()
        x = _example()
        exported = cml.export(
            peft, x, f"{tmp_path}/half.mlpackage", precision=cml.Precision.FLOAT16
        )
        try:
            _assert_predicts(exported.predict(x), _eager(peft, x), TOL_FLOAT16)
        finally:
            exported.close()

    def test_a_merged_package_has_no_adapter_branch(self, tmp_path: object) -> None:
        """Merging is the deployment path: the package is the base's, entry for entry."""
        base_package = f"{tmp_path}/base.mlpackage"
        x = _example()
        cml.export(_base(), x, base_package).close()

        peft = _peft()
        plain = peft.merge_and_unload().eval()
        merged_package = f"{tmp_path}/merged.mlpackage"
        cml.export(plain, x, merged_package).close()

        assert _entries(merged_package) == _entries(base_package)
        assert _weight_bytes(merged_package) == _weight_bytes(base_package)


# ── one copy of the base ──────────────────────────────────────────


class TestOneCopyOfTheBase:
    def test_an_unmerged_package_is_the_base_plus_the_adapter(self, tmp_path: object) -> None:
        x = _example()
        base_package = f"{tmp_path}/base.mlpackage"
        cml.export(_base(), x, base_package).close()

        peft = _peft()
        unmerged_package = f"{tmp_path}/unmerged.mlpackage"
        cml.export(peft, x, unmerged_package).close()

        base_entries, unmerged_entries = _entries(base_package), _entries(unmerged_package)
        adapter_bytes = 4 * _floats(peft, lambda name: "lora_" in name)
        # Every base entry is there once; what is added is the adapter and
        # its small constants — nowhere near a second copy of the base.
        for size in base_entries:
            unmerged_entries.remove(size)
        extra = sum(unmerged_entries)
        assert adapter_bytes <= extra <= 1.5 * adapter_bytes, (adapter_bytes, extra)
        assert extra < 0.5 * sum(base_entries)

    def test_the_base_function_adds_nothing_to_the_adapted_one(self, tmp_path: object) -> None:
        peft = _peft()
        x = _example()
        alone = f"{tmp_path}/alone.mlpackage"
        cml.export(peft, x, alone).close()

        together = f"{tmp_path}/together.mlpackage"
        handles = cml.export_functions(
            {"base": (_AdapterOff(peft), x), "adapted": (peft, x)}, together, default="adapted"
        )
        try:
            assert sorted(_entries(together)) == sorted(_entries(alone))
            assert _weight_bytes(together) == _weight_bytes(alone)
            with peft.disable_adapter():
                _assert_predicts(handles["base"].predict(x), _eager(peft, x))
            _assert_predicts(handles["adapted"].predict(x), _eager(peft, x))
        finally:
            _close(*handles.values())

    def test_a_second_adapter_costs_its_own_bytes_only(self, tmp_path: object) -> None:
        """Two adapters over one base: the base is written once for all three functions."""
        x = _example()
        peft_a = _peft(4)
        peft_b = get_peft_model(_sharing_the_base_of(peft_a), _config(8)).eval()
        base_param = peft_a.get_base_model().fc1.base_layer.weight
        assert peft_b.get_base_model().fc1.base_layer.weight is base_param

        base_package = f"{tmp_path}/base.mlpackage"
        cml.export(_AdapterOff(peft_a), x, base_package).close()
        a_package = f"{tmp_path}/a.mlpackage"
        cml.export(peft_a, x, a_package).close()
        b_package = f"{tmp_path}/b.mlpackage"
        cml.export(peft_b, x, b_package).close()

        shared = f"{tmp_path}/shared.mlpackage"
        handles = cml.export_functions(
            {"base": (_AdapterOff(peft_a), x), "a": (peft_a, x), "b": (peft_b, x)}, shared
        )
        try:
            alone = (
                _weight_bytes(base_package),
                _weight_bytes(a_package),
                _weight_bytes(b_package),
            )
            # A + (B - base): the base is in A already, B brings its adapter.
            assert _weight_bytes(shared) == alone[1] + alone[2] - alone[0]
            assert _weight_bytes(shared) < sum(alone)
            assert _package_bytes(shared) < sum(
                _package_bytes(p) for p in (base_package, a_package, b_package)
            )
            for name, model in (("a", peft_a), ("b", peft_b)):
                _assert_predicts(handles[name].predict(x), _eager(model, x))
            with peft_a.disable_adapter():
                _assert_predicts(handles["base"].predict(x), _eager(peft_a, x))
        finally:
            _close(*handles.values())

    def test_the_adapters_of_two_functions_differ(self, tmp_path: object) -> None:
        """Sharing the base must not share the adapters: each function keeps its own."""
        x = _example()
        peft_a = _peft(4)
        peft_b = get_peft_model(_sharing_the_base_of(peft_a), _config(8)).eval()
        handles = cml.export_functions(
            {"a": (peft_a, x), "b": (peft_b, x)}, f"{tmp_path}/two.mlpackage"
        )
        try:
            got_a, got_b = handles["a"].predict(x), handles["b"].predict(x)
            assert float((got_a - got_b).abs().max().item()) > 1e-3 * _scale(got_a)
            _assert_predicts(got_a, _eager(peft_a, x))
            _assert_predicts(got_b, _eager(peft_b, x))
        finally:
            _close(*handles.values())

    def test_a_deep_copy_of_the_base_ships_its_weights_again(self, tmp_path: object) -> None:
        """Sharing is by ``Parameter`` identity, so a copy is a second set of weights."""
        x = _example()
        peft = _peft()
        twin = get_peft_model(copy.deepcopy(_base()), _config()).eval()
        together = f"{tmp_path}/copies.mlpackage"
        handles = cml.export_functions({"one": (peft, x), "two": (twin, x)}, together)
        alone = f"{tmp_path}/one.mlpackage"
        cml.export(peft, x, alone).close()
        try:
            base_bytes = 4 * _floats(peft, lambda name: "lora_" not in name)
            assert _weight_bytes(together) >= 2 * _weight_bytes(alone) - 128
            assert _weight_bytes(together) - _weight_bytes(alone) >= base_bytes
        finally:
            _close(*handles.values())

    @pytest.mark.xfail(
        strict=True,
        reason="LCD-688: a compressed (int8) weight is written per function; only float "
        "weights are shared by identity",
    )
    def test_an_int8_base_is_shared_between_functions(self, tmp_path: object) -> None:
        peft = _peft()
        x = _example()
        weights = cml.WeightPrecision.INT8
        alone = f"{tmp_path}/alone.mlpackage"
        cml.export(peft, x, alone, weights=weights).close()
        together = f"{tmp_path}/together.mlpackage"
        handles = cml.export_functions(
            {"base": (_AdapterOff(peft), x), "adapted": (peft, x)}, together, weights=weights
        )
        _close(*handles.values())
        assert _weight_bytes(together) == _weight_bytes(alone)
