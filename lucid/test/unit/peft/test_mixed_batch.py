"""``model(x, adapter_names=[...])``: each batch row through the adapter it is named for.

The reference for every case is the model run one row at a time with that
row's adapter set — a mixed batch must equal the rows computed alone — on
CPU and Metal, for linear, embedding, convolution and ``modules_to_save``
slots.  The refusals are the reference's, and a refused or failed call must
leave every slot as it found it.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    IA3Config,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    get_peft_model,
)
from lucid.peft.tuners.lora import Linear


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _config(targets: list[str], **kwargs: object) -> LoraConfig:
    return LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=targets,
        init_lora_weights=False,
        **kwargs,  # type: ignore[arg-type]
    )


def _mlp(device: str) -> PeftModel:
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)
    peft = get_peft_model(net, _config(["0", "2"]), adapter_name="a")
    peft.add_adapter("b", _config(["0", "2"]))
    peft.eval()
    return peft


def _alone(peft: PeftModel, x: lucid.Tensor, names: list[str]) -> lucid.Tensor:
    """The rows of ``x`` computed one at a time, each with its adapter set (or none)."""
    rows = []
    for i, name in enumerate(names):
        row = x[i : i + 1]
        if name == "__base__":
            with peft.disable_adapter():
                rows.append(peft(row))
        else:
            peft.set_adapter(name)
            rows.append(peft(row))
    return lucid.cat(rows)


class TestRoutedRows:
    def test_each_row_equals_that_row_alone(self, device: str) -> None:
        peft = _mlp(device)
        names = ["a", "b", "__base__", "a", "b", "b"]
        x = lucid.randn(6, 6, device=device)
        with lucid.no_grad():
            mixed = peft(x, adapter_names=names)
            expected = _alone(peft, x, names)
        assert _close(mixed, expected)

    def test_the_adapters_in_use_are_ignored(self, device: str) -> None:
        peft = _mlp(device)
        x = lucid.randn(3, 6, device=device)
        with lucid.no_grad():
            peft.set_adapter("a")
            mixed = peft(x, adapter_names=["b", "b", "b"])
            peft.set_adapter("b")
            assert _close(mixed, peft(x))

    def test_a_layer_without_the_adapter_runs_its_base(self, device: str) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 3)).to(device)
        peft = get_peft_model(net, _config(["0", "1"]), adapter_name="a")
        peft.add_adapter("only_first", _config(["0"]))
        peft.eval()
        x = lucid.randn(4, 6, device=device)
        names = ["only_first", "a", "only_first", "__base__"]
        with lucid.no_grad():
            assert _close(peft(x, adapter_names=names), _alone(peft, x, names))

    def test_embedding_and_convolution_slots(self, device: str) -> None:
        lucid.manual_seed(0)
        table = nn.Sequential(nn.Embedding(10, 4)).to(device)
        peft = get_peft_model(table, _config(["0"]), adapter_name="a")
        peft.add_adapter("b", _config(["0"]))
        peft.eval()
        tokens = lucid.tensor([[1, 3], [7, 0], [9, 2]], device=device)
        names = ["b", "__base__", "a"]
        with lucid.no_grad():
            assert _close(peft(tokens, adapter_names=names), _alone(peft, tokens, names))

        conv = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1)).to(device)
        peft = get_peft_model(conv, _config(["0"]), adapter_name="a")
        peft.add_adapter("b", _config(["0"]))
        peft.eval()
        image = lucid.randn(3, 2, 5, 5, device=device)
        names = ["a", "b", "b"]
        with lucid.no_grad():
            assert _close(peft(image, adapter_names=names), _alone(peft, image, names))

    def test_modules_to_save_run_the_named_adapters_copy(self, device: str) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = nn.Linear(6, 6)
                self.head = nn.Linear(6, 2)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.head(self.body(x))

        lucid.manual_seed(0)
        peft = get_peft_model(
            Net().to(device), _config(["body"], modules_to_save=["head"]), adapter_name="a"
        )
        peft.add_adapter("b", _config(["body"], modules_to_save=["head"]))
        with lucid.no_grad():  # the copies start equal; make them differ
            for name in ("a", "b"):
                wrapper = peft.get_base_model().head
                wrapper.modules_to_save[name].weight.copy_(lucid.randn(2, 6, device=device))
        peft.eval()
        x = lucid.randn(4, 6, device=device)
        names = ["a", "b", "__base__", "b"]
        with lucid.no_grad():
            assert _close(peft(x, adapter_names=names), _alone(peft, x, names))


class TestSlotsAreLeftAlone:
    def test_the_flags_and_hooks_are_restored(self, device: str) -> None:
        peft = _mlp(device)
        layers = [m for m in peft.modules() if isinstance(m, Linear)]
        x = lucid.randn(2, 6, device=device)
        with lucid.no_grad():
            before = peft(x)
            peft(x, adapter_names=["a", "b"])
            assert _close(peft(x), before)
        assert all(not layer.disable_adapters and not layer._forward_hooks for layer in layers)

    def test_a_failed_call_restores_them_too(self, device: str) -> None:
        peft = _mlp(device)
        layers = [m for m in peft.modules() if isinstance(m, Linear)]
        with pytest.raises(ValueError, match="Length of `adapter_names`"):
            peft(lucid.randn(3, 6, device=device), adapter_names=["a", "b"])
        assert all(not layer.disable_adapters and not layer._forward_hooks for layer in layers)

    def test_inside_disable_adapter_the_base_runs(self, device: str) -> None:
        peft = _mlp(device)
        x = lucid.randn(3, 6, device=device)
        with lucid.no_grad(), peft.disable_adapter():
            base = peft(x)
            assert _close(peft(x, adapter_names=["a", "b", "a"]), base)
        assert all(not m.disable_adapters for m in peft.modules() if isinstance(m, Linear))


class TestRefusals:
    def test_training_mode(self, device: str) -> None:
        peft = _mlp(device)
        peft.train()
        with pytest.raises(ValueError, match="training mode"):
            peft(lucid.randn(2, 6, device=device), adapter_names=["a", "b"])

    def test_an_unknown_adapter_is_named(self, device: str) -> None:
        peft = _mlp(device)
        with pytest.raises(ValueError, match="non-existing adapter\\(s\\): ghost, phantom"):
            peft(lucid.randn(3, 6, device=device), adapter_names=["a", "ghost", "phantom"])

    def test_the_row_count_must_match(self, device: str) -> None:
        peft = _mlp(device)
        with pytest.raises(ValueError, match="got 1 and 3 respectively"):
            peft(lucid.randn(3, 6, device=device), adapter_names=["a"])

    def test_merged_adapters(self, device: str) -> None:
        peft = _mlp(device)
        peft.base_model.merge_adapter(["a"])
        with pytest.raises(ValueError, match="merged adapters.*unmerge_adapter"):
            peft(lucid.randn(2, 6, device=device), adapter_names=["a", "b"])
        peft.base_model.unmerge_adapter()
        with lucid.no_grad():
            peft(lucid.randn(2, 6, device=device), adapter_names=["a", "b"])

    def test_dora_adapters_refuse_but_others_beside_them_do_not(self, device: str) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 8)).to(device)
        peft = get_peft_model(net, _config(["0"], use_dora=True), adapter_name="dora")
        peft.add_adapter("plain", _config(["0"]))
        peft.eval()
        x = lucid.randn(2, 6, device=device)
        with pytest.raises(ValueError, match="DoRA is enabled"):
            peft(x, adapter_names=["dora", "plain"])
        with lucid.no_grad():
            assert _close(
                peft(x, adapter_names=["plain", "__base__"]),
                _alone(peft, x, ["plain", "__base__"]),
            )

    def test_attention_layers_cannot_be_routed(self, device: str) -> None:
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.attn(x, x, x)[0]

        peft = get_peft_model(Block().to(device), _config(["attn"]))
        peft.eval()
        with pytest.raises(NotImplementedError, match="MultiheadAttention"):
            peft(lucid.randn(2, 3, 8, device=device), adapter_names=["default", "__base__"])

    def test_another_method_cannot_route(self, device: str) -> None:
        net = nn.Sequential(nn.Linear(6, 4)).to(device)
        peft = get_peft_model(net, IA3Config(target_modules=["0"], feedforward_modules=[]))
        with pytest.raises(NotImplementedError, match="IA3Model cannot send the rows"):
            peft(lucid.randn(2, 6, device=device), adapter_names=["default", "default"])


class _Decoder(nn.Module):
    """A model with an encoder, a decoder and a ``generate`` that repeats the batch per beam."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)

    def get_encoder(self) -> nn.Module:
        return self.enc

    def forward(self, x: lucid.Tensor, beams: int = 1) -> lucid.Tensor:
        memory = self.enc(x)
        # Each row repeated per beam in turn, as beam search expands a batch.
        memory = lucid.stack([memory] * beams, dim=1).flatten(0, 1) if beams > 1 else memory
        return self.dec(memory)

    def generate(self, x: lucid.Tensor, num_beams: int = 1) -> lucid.Tensor:
        return self(x, beams=num_beams)


class TestGenerate:
    def _peft(self, device: str) -> PeftModelForCausalLM:
        lucid.manual_seed(0)
        config = _config(["enc", "dec"], task_type="CAUSAL_LM")
        peft = get_peft_model(_Decoder().to(device), config, adapter_name="a")
        peft.add_adapter("b", _config(["enc", "dec"], task_type="CAUSAL_LM"))
        peft.eval()
        assert isinstance(peft, PeftModelForCausalLM)
        return peft

    def test_generate_routes_each_prompt(self, device: str) -> None:
        peft = self._peft(device)
        x = lucid.randn(3, 4, device=device)
        names = ["a", "b", "__base__"]
        with lucid.no_grad():
            routed = peft.generate(x, adapter_names=names)
            assert _close(routed, _alone(peft, x, names))

    def test_beams_repeat_the_names_but_not_the_encoders(self, device: str) -> None:
        peft = self._peft(device)
        x = lucid.randn(2, 4, device=device)
        names = ["a", "b"]
        with lucid.no_grad():
            routed = peft.generate(x, adapter_names=names, num_beams=2)
            # Encoder: rows (a, b); decoder: rows (a, a, b, b), each prompt's beams together.
            single = _alone(peft, x, names)
            assert _close(routed, lucid.stack([single] * 2, dim=1).flatten(0, 1))

    def test_without_names_generate_is_unchanged(self, device: str) -> None:
        peft = self._peft(device)
        x = lucid.randn(2, 4, device=device)
        with lucid.no_grad():
            assert _close(peft.generate(x), peft(x))
