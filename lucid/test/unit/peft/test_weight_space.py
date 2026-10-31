"""``lora.MultiheadAttention`` and ``lora.ParamWrapper``: the layers that shift the weights a module reads.

Their closed form is the base module run on ``W + s B A``: the reference is
built here by writing that sum into a copy of the module and calling it, so a
layer that shifted the wrong weight, the wrong way, or left a shadow behind
shows as a difference, on CPU and Metal.
"""

import copy
import warnings

import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core import signature
from lucid.peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import lora


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


# ── MultiheadAttention ───────────────────────────────────────────────────────

E = 8


def _attention(
    device: str, **kwargs: object
) -> tuple[nn.MultiheadAttention, lora.MultiheadAttention, lucid.Tensor]:
    lucid.manual_seed(0)
    base = nn.MultiheadAttention(E, 2, batch_first=True).to(device)
    options: dict[str, object] = {"r": 2, "lora_alpha": 6, "init_lora_weights": False} | kwargs
    layer = lora.MultiheadAttention(base, "default", **options)  # type: ignore[arg-type]
    return base, layer, lucid.randn(2, 5, E, device=device)


def _shifted_copy(
    base: nn.MultiheadAttention, layer: lora.MultiheadAttention, adapter: str = "default"
) -> nn.MultiheadAttention:
    """A copy of the attention with the adapter's update written into its two projections."""
    ref = copy.deepcopy(base)
    with lucid.no_grad():
        ref.in_proj_weight.add_(layer._delta_for("in_proj_weight", adapter))
        ref.out_proj_weight.add_(layer._delta_for("out_proj_weight", adapter))
    return ref


class TestAttention:
    def test_the_factors_are_the_fused_input_and_the_output_projection_s(self, device: str) -> None:
        base, layer, _ = _attention(device)
        assert layer.get_base_layer() is base
        shapes = {
            name: tuple(getattr(layer, name)["default"].weight.shape)
            for name in ("lora_A", "lora_B", "out_proj_lora_A", "out_proj_lora_B")
        }
        assert shapes == {
            "lora_A": (2, E),
            "lora_B": (3 * E, 2),
            "out_proj_lora_A": (2, E),
            "out_proj_lora_B": (E, 2),
        }
        assert layer.scaling["default"] == 3.0
        assert repr(layer).startswith("lora.MultiheadAttention(")

    def test_the_attention_s_attributes_read_through(self, device: str) -> None:
        base, layer, _ = _attention(device)
        assert (layer.embed_dim, layer.num_heads, layer.batch_first) == (E, 2, True)
        assert layer.in_proj_weight is base.in_proj_weight
        assert layer.out_proj_weight is base.out_proj_weight
        with pytest.raises(AttributeError):
            layer.no_such_attribute  # noqa: B018

    def test_a_zero_update_is_the_base_bit_for_bit(self, device: str) -> None:
        base, layer, x = _attention(device, init_lora_weights=True)
        out, weights = layer(x, x, x)
        expected, expected_weights = base(x, x, x)
        assert _same(out, expected) and _same(weights, expected_weights)

    def test_the_output_is_the_base_on_the_shifted_weights(self, device: str) -> None:
        base, layer, x = _attention(device)
        ref = _shifted_copy(base, layer)
        out, weights = layer(x, x, x)
        expected, expected_weights = ref(x, x, x)
        assert _close(out, expected) and _close(weights, expected_weights)
        plain, _ = base(x, x, x)
        assert not _close(out, plain, atol=1e-3)

    def test_the_delta_is_scale_times_b_a_for_each_projection(self, device: str) -> None:
        _, layer, _ = _attention(device)
        b, a = layer.lora_B["default"].weight, layer.lora_A["default"].weight
        assert _close(layer.get_delta_weight("default"), (b @ a) * 3.0)
        ob = layer.out_proj_lora_B["default"].weight
        oa = layer.out_proj_lora_A["default"].weight
        assert _close(layer._delta_for("out_proj_weight", "default"), (ob @ oa) * 3.0)

    def test_the_registered_parameters_are_untouched_by_a_forward(self, device: str) -> None:
        base, layer, x = _attention(device)
        weights = (base.in_proj_weight, base.out_proj_weight)
        values = [w.clone() for w in weights]
        names = list(dict(base.named_parameters()))
        layer(x, x, x)
        assert (base.in_proj_weight, base.out_proj_weight) == weights
        assert all(_same(w, v) for w, v in zip(weights, values, strict=True))
        assert list(dict(base.named_parameters())) == names
        assert "in_proj_weight" not in vars(base) and "out_proj_weight" not in vars(base)

    def test_a_failing_forward_leaves_no_shadow_behind(self, device: str) -> None:
        base, layer, x = _attention(device)
        with pytest.raises(ValueError, match="to be 3-D"):
            layer(x, lucid.randn(4, E, device=device), x)
        assert "in_proj_weight" not in vars(base) and "out_proj_weight" not in vars(base)

    def test_a_forward_does_not_move_the_compile_key(self, device: str) -> None:
        _, layer, x = _attention(device)
        before = signature._module_state(layer)
        layer(x, x, x)
        assert signature._module_state(layer) == before

    def test_gradients_reach_the_factors_and_not_the_frozen_weights(self, device: str) -> None:
        base, layer, x = _attention(device)
        base.requires_grad_(False)
        layer(x, x, x)[0].sum().backward()
        for name in ("lora_A", "lora_B", "out_proj_lora_A", "out_proj_lora_B"):
            grad = getattr(layer, name)["default"].weight.grad
            assert grad is not None and float(grad.abs().max().item()) > 0, name
        assert base.in_proj_weight.grad is None and base.out_proj_weight.grad is None

    def test_merging_writes_both_projections_in_place_and_unmerging_restores_them(
        self, device: str
    ) -> None:
        base, layer, x = _attention(device)
        ref = _shifted_copy(base, layer)
        original = (base.in_proj_weight.clone(), base.out_proj_weight.clone())
        weights = (base.in_proj_weight, base.out_proj_weight)
        with lucid.no_grad():
            adapted = layer(x, x, x)[0]
            layer.merge()
            assert (base.in_proj_weight, base.out_proj_weight) == weights
            assert _close(base.in_proj_weight, ref.in_proj_weight, atol=1e-6)
            assert _close(base.out_proj_weight, ref.out_proj_weight, atol=1e-6)
            assert _close(layer(x, x, x)[0], adapted)
            layer.unmerge()
            assert _close(base.in_proj_weight, original[0], atol=1e-5)
            assert _close(base.out_proj_weight, original[1], atol=1e-5)

    def test_cross_attention_shifts_the_same_weights(self, device: str) -> None:
        base, layer, x = _attention(device)
        memory = lucid.randn(2, 7, E, device=device)
        ref = _shifted_copy(base, layer)
        assert _close(layer(x, memory, memory)[0], ref(x, memory, memory)[0])

    def test_a_bfloat16_attention_gets_a_float32_adapter(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.MultiheadAttention(E, 2, batch_first=True).to(device).bfloat16()
        layer = lora.MultiheadAttention(base, "default", r=2, lora_alpha=2, init_lora_weights=False)
        assert layer.lora_A["default"].weight.dtype == lucid.float32
        x = lucid.randn(2, 5, E, device=device).bfloat16()
        assert layer(x, x, x)[0].dtype == lucid.bfloat16

    def test_dropout_is_not_applied_and_says_so(self, device: str) -> None:
        with pytest.warns(UserWarning, match="not applied by lora.MultiheadAttention"):
            _, layer, x = _attention(device, lora_dropout=0.3)
        layer.train()
        assert _same(layer(x, x, x)[0], layer(x, x, x)[0])

    def test_two_adapters_shift_the_weights_by_their_sum_when_both_are_active(
        self, device: str
    ) -> None:
        base, layer, x = _attention(device)
        layer.update_layer("other", 2, 4, 0.0, False, False)
        layer.set_adapter(["default", "other"])
        ref = copy.deepcopy(base)
        with lucid.no_grad():
            for name in ("in_proj_weight", "out_proj_weight"):
                for adapter in ("default", "other"):
                    getattr(ref, name).add_(layer._delta_for(name, adapter))
        assert _close(layer(x, x, x)[0], ref(x, x, x)[0])

    def test_the_tuner_wraps_the_attention_and_keys_the_file_as_the_reference_does(
        self, device: str
    ) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.MultiheadAttention(E, 2)).to(device)
        peft = get_peft_model(model, LoraConfig(r=2, target_modules=["0"]))
        assert isinstance(model[0], lora.MultiheadAttention)
        assert sorted(get_peft_model_state_dict(peft)) == [
            "base_model.model.0.base_layer.out_proj.lora_A.weight",
            "base_model.model.0.base_layer.out_proj.lora_B.weight",
            "base_model.model.0.lora_A.weight",
            "base_model.model.0.lora_B.weight",
        ]


class TestSeparateProjections:
    """An attention with its own query, key and value weights: a cross-attention over a wider
    or narrower context (``kdim`` / ``vdim``), or grouped key/value heads."""

    #: ``(kwargs of the attention, key width, value width)``
    _CASES = {
        "context_width": (dict(kdim=6, vdim=6), 6, 6),
        "key_value_widths_apart": (dict(kdim=6, vdim=10), 6, 10),
        "grouped_heads": (dict(num_kv_heads=2), E, E),
    }
    _WEIGHTS = ("q_proj_weight", "k_proj_weight", "v_proj_weight", "out_proj_weight")

    @staticmethod
    def _built(
        device: str, case: str, **options: object
    ) -> tuple[nn.MultiheadAttention, lora.MultiheadAttention, tuple[lucid.Tensor, ...]]:
        lucid.manual_seed(0)
        extra, key_width, value_width = TestSeparateProjections._CASES[case]
        heads = 4 if "num_kv_heads" in extra else 2
        base = nn.MultiheadAttention(E, heads, batch_first=True, **extra).to(device)  # type: ignore[arg-type]
        merged: dict[str, object] = {"r": 2, "lora_alpha": 6, "init_lora_weights": False} | options
        layer = lora.MultiheadAttention(base, "default", **merged)  # type: ignore[arg-type]
        inputs = (
            lucid.randn(2, 5, E, device=device),
            lucid.randn(2, 7, key_width, device=device),
            lucid.randn(2, 7, value_width, device=device),
        )
        return base, layer, inputs

    @staticmethod
    def _shifted(
        base: nn.MultiheadAttention, layer: lora.MultiheadAttention
    ) -> nn.MultiheadAttention:
        ref = copy.deepcopy(base)
        with lucid.no_grad():
            for name in TestSeparateProjections._WEIGHTS:
                getattr(ref, name).add_(layer._delta_for(name, "default"))
        return ref

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_each_weight_gets_a_pair_shaped_by_its_rows_and_columns(
        self, device: str, case: str
    ) -> None:
        base, layer, _ = self._built(device, case)
        assert type(layer).__name__ == "_SeparateMultiheadAttention"
        assert isinstance(layer, lora.MultiheadAttention)
        assert layer.get_base_layer() is base and repr(layer).startswith("lora.")
        for name in self._WEIGHTS:
            table_a, table_b = layer._SLOTS[name]
            out, inn = base.get_parameter(name).shape
            assert tuple(getattr(layer, table_a)["default"].weight.shape) == (2, inn)
            assert tuple(getattr(layer, table_b)["default"].weight.shape) == (out, 2)
        assert layer.in_proj_weight is None

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_the_output_is_the_base_on_the_shifted_weights(self, device: str, case: str) -> None:
        base, layer, inputs = self._built(device, case)
        ref = self._shifted(base, layer)
        out, _ = layer(*inputs)
        expected, _ = ref(*inputs)
        assert _close(out, expected)
        assert not _close(out, base(*inputs)[0], atol=1e-3)

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_a_zero_update_is_the_base_bit_for_bit(self, device: str, case: str) -> None:
        base, layer, inputs = self._built(device, case, init_lora_weights=True)
        assert _same(layer(*inputs)[0], base(*inputs)[0])

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_gradients_reach_every_pair(self, device: str, case: str) -> None:
        base, layer, inputs = self._built(device, case)
        base.requires_grad_(False)
        layer(*inputs)[0].sum().backward()
        for table in layer.adapter_layer_names:
            grad = getattr(layer, table)["default"].weight.grad
            assert grad is not None and float(grad.abs().max().item()) > 0, table

    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_merging_writes_all_four_weights_and_unmerging_restores_them(
        self, device: str, case: str
    ) -> None:
        base, layer, inputs = self._built(device, case)
        ref = self._shifted(base, layer)
        original = {n: base.get_parameter(n).clone() for n in self._WEIGHTS}
        with lucid.no_grad():
            adapted = layer(*inputs)[0]
            layer.merge()
            for name in self._WEIGHTS:
                assert _close(base.get_parameter(name), ref.get_parameter(name), atol=1e-6)
            assert _close(layer(*inputs)[0], adapted)
            layer.unmerge()
            for name in self._WEIGHTS:
                assert _close(base.get_parameter(name), original[name], atol=1e-5)

    def test_the_tuner_wraps_it_and_keys_every_table_under_its_projection(
        self, device: str
    ) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.MultiheadAttention(E, 2, kdim=6, vdim=6)).to(device)
        peft = get_peft_model(model, LoraConfig(r=2, target_modules=["0"]))
        assert isinstance(model[0], lora.MultiheadAttention)
        keys = sorted(get_peft_model_state_dict(peft))
        assert keys == [
            f"base_model.model.0.base_layer.{stem}.lora_{ab}.weight"
            for stem in ("k_proj", "out_proj", "q_proj", "v_proj")
            for ab in ("A", "B")
        ]

    def test_a_layout_that_does_not_hold_the_attention_is_refused(self, device: str) -> None:
        from lucid.peft.tuners.lora._attention import _SeparateMultiheadAttention

        fused = nn.MultiheadAttention(E, 2).to(device)
        with pytest.raises(ValueError, match="does not adapt"):
            _SeparateMultiheadAttention(fused, "default", r=2)


# ── ParamWrapper ─────────────────────────────────────────────────────────────


class _Holder(nn.Module):
    """Two raw parameters read in the forward: a matrix and a stack of three."""

    def __init__(self, transposed: bool = False) -> None:
        super().__init__()
        self.is_transposed = transposed
        self.w = nn.Parameter(lucid.randn(5, 6))
        self.v = nn.Parameter(lucid.randn(3, 6, 5) if transposed else lucid.randn(3, 5, 6))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        v = self.v if self.is_transposed else self.v.mT  # (E, in, out)
        return x @ self.w.mT + (x.unsqueeze(0) @ v).sum(0)


class _Net(nn.Module):
    def __init__(self, transposed: bool = False) -> None:
        super().__init__()
        self.holder = _Holder(transposed)
        self.head = nn.Linear(5, 2)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.holder(x))


def _wrapper(
    device: str, parameter: str = "w", transposed: bool = False, **kwargs: object
) -> tuple[_Holder, lora.ParamWrapper, lucid.Tensor]:
    lucid.manual_seed(0)
    holder = _Holder(transposed).to(device)
    options: dict[str, object] = {"r": 2, "lora_alpha": 6, "init_lora_weights": False} | kwargs
    layer = lora.ParamWrapper(holder, "default", parameter, **options)  # type: ignore[arg-type]
    return holder, layer, lucid.randn(4, 6, device=device)


def _run_with(holder: _Holder, x: lucid.Tensor, **shifts: lucid.Tensor) -> lucid.Tensor:
    ref = copy.deepcopy(holder)
    with lucid.no_grad():
        for name, delta in shifts.items():
            getattr(ref, name).add_(delta)
    return ref(x)


class TestParameter:
    def test_a_matrix_gets_one_pair_shaped_by_its_rows_and_columns(self, device: str) -> None:
        holder, layer, _ = _wrapper(device)
        assert layer.get_base_layer() is holder and layer.base_layer is holder
        assert tuple(layer.lora_A["default"].weight.shape) == (2, 6)  # (r, in)
        assert tuple(layer.lora_B["default"].weight.shape) == (5, 2)  # (out, r)
        assert sorted(layer.state_dict()) == [
            "base_layer.v",
            "base_layer.w",
            "lora_A.default.weight",
            "lora_B.default.weight",
        ]
        assert "parameter_name='w'" in repr(layer) and repr(layer).startswith("lora.ParamWrapper(")

    def test_the_wrapper_answers_for_the_module_it_wraps(self, device: str) -> None:
        holder, layer, _ = _wrapper(device)
        # The parameter read directly is the base's, the update not applied.
        assert layer.w is holder.w and layer.is_transposed is False

    def test_the_output_is_the_module_on_the_shifted_parameter(self, device: str) -> None:
        holder, layer, x = _wrapper(device)
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == (5, 6)
        b, a = layer.lora_B["default"].weight, layer.lora_A["default"].weight
        assert _close(delta, (b @ a) * 3.0)
        assert _close(layer(x), _run_with(holder, x, w=delta))
        assert "w" not in vars(holder)

    def test_a_zero_update_is_the_base_bit_for_bit(self, device: str) -> None:
        holder, layer, x = _wrapper(device, init_lora_weights=True)
        assert _same(layer(x), holder(x))

    @pytest.mark.parametrize("transposed", [False, True])
    def test_a_stack_has_a_pair_per_expert(self, device: str, transposed: bool) -> None:
        holder, layer, x = _wrapper(device, "v", transposed)
        assert tuple(layer.lora_A["default"].weight.shape) == (2 * 3, 6)  # (r * experts, in)
        assert tuple(layer.lora_B["default"].weight.shape) == (5, 2 * 3)  # (out, r * experts)
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == tuple(holder.v.shape)
        # The reference's reading: A as (experts, r, in), B as (out, r, experts).
        a = layer.lora_A["default"].weight.reshape(3, 2, 6)
        b = layer.lora_B["default"].weight.reshape(5, 2, 3)
        for expert in range(3):
            per_expert = (b[:, :, expert] @ a[expert]) * 3.0  # (out, in)
            got = delta[expert].mT if transposed else delta[expert]
            assert _close(got, per_expert)
        assert _close(layer(x), _run_with(holder, x, v=delta))

    def test_merging_writes_the_parameter_in_place(self, device: str) -> None:
        holder, layer, x = _wrapper(device)
        weight, original = holder.w, holder.w.clone()
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert holder.w is weight and layer.merged
            assert _close(holder.w, original + layer.get_delta_weight("default"), atol=1e-6)
            assert _close(layer(x), adapted)
            assert _close(holder(x), adapted)
            layer.unmerge()
            assert _close(holder.w, original, atol=1e-5)

    def test_gradients_reach_the_factors_and_not_the_frozen_parameter(self, device: str) -> None:
        holder, layer, x = _wrapper(device, "v")
        holder.requires_grad_(False)
        layer(x).sum().backward()
        assert holder.v.grad is None
        for table in (layer.lora_A, layer.lora_B):
            grad = table["default"].weight.grad
            assert grad is not None and float(grad.abs().max().item()) > 0

    @pytest.mark.parametrize(
        ("options", "error", "match"),
        [
            ({"lora_dropout": 0.1}, ValueError, "lora_dropout != 0"),
            ({"r": 0}, ValueError, "positive integer"),
        ],
    )
    def test_what_the_layer_cannot_honour_is_refused(
        self, device: str, options: dict[str, object], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            _wrapper(device, **options)

    def test_the_parameter_must_exist_and_be_a_matrix_or_a_stack(self, device: str) -> None:
        holder = _Holder().to(device)
        with pytest.raises(ValueError, match="no parameter 'nope'"):
            lora.ParamWrapper(holder, "default", "nope", r=2)
        flat = nn.Linear(3, 3).to(device)
        with pytest.raises(ValueError, match="1-D"):
            lora.ParamWrapper(flat, "default", "bias", r=2)

    def test_a_second_parameter_nests_around_the_first(self, device: str) -> None:
        holder, inner, x = _wrapper(device, "w")
        outer = lora.ParamWrapper(inner, "default", "v", r=2, lora_alpha=6, init_lora_weights=False)
        assert outer.get_base_layer() is holder and inner.get_base_layer() is holder
        assert outer._base_path() == "base_layer.base_layer"
        assert inner._base_path() == "base_layer"
        expected = _run_with(
            holder, x, w=inner.get_delta_weight("default"), v=outer.get_delta_weight("default")
        )
        assert _close(outer(x), expected)
        assert sorted(outer.state_dict()) == [
            "base_layer.base_layer.v",
            "base_layer.base_layer.w",
            "base_layer.lora_A.default.weight",
            "base_layer.lora_B.default.weight",
            "lora_A.default.weight",
            "lora_B.default.weight",
        ]

    def test_a_wrapper_is_a_slot_that_unwraps_to_the_module(self, device: str) -> None:
        holder, inner, _ = _wrapper(device, "w")
        outer = lora.ParamWrapper(inner, "default", "v", r=2, lora_alpha=6, init_lora_weights=False)
        original = holder.v.clone()
        outer.merge()
        merged = holder.v.clone()
        assert not _close(merged, original, atol=1e-6)
        assert outer._unwrap(keep_trained=True) is holder
        assert outer.merged and _same(
            holder.v, merged
        )  # keeping the trained result keeps the merge
        assert outer._unwrap(keep_trained=False) is holder
        assert not outer.merged and _close(holder.v, original, atol=1e-5)


class _Tied(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.holder = _Holder()
        self.other = nn.Module()
        self.other.w = self.holder.w

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.holder(x)


def _net(device: str, transposed: bool = False) -> _Net:
    lucid.manual_seed(0)
    return _Net(transposed).to(device)


class TestParameterTargets:
    def test_a_parameter_is_found_by_its_dotted_name_or_a_suffix_of_it(self, device: str) -> None:
        for names in (["holder.w"], ["w"], ["holder.w", "w"]):
            net = _net(device)
            peft = get_peft_model(net, LoraConfig(r=2, target_parameters=names))
            assert peft.base_model.targeted_parameter_names == ["holder.w"], names
            assert peft.base_model.targeted_module_names == []
            assert isinstance(net.holder, lora.ParamWrapper) and type(net.head) is nn.Linear

    def test_two_parameters_of_a_module_nest_in_declaration_order(self, device: str) -> None:
        net = _net(device)
        peft = get_peft_model(net, LoraConfig(r=2, target_parameters=["w", "v"]))
        assert peft.base_model.targeted_parameter_names == ["holder.w", "holder.v"]
        outer = net.holder
        assert isinstance(outer, lora.ParamWrapper) and outer.parameter_name == "v"
        assert isinstance(outer.base_layer, lora.ParamWrapper)
        assert outer.base_layer.parameter_name == "w"

    def test_parameters_and_modules_are_adapted_together(self, device: str) -> None:
        net = _net(device)
        peft = get_peft_model(
            net, LoraConfig(r=2, target_modules=["head"], target_parameters=["w"])
        )
        assert peft.base_model.targeted_module_names == ["head"]
        assert peft.base_model.targeted_parameter_names == ["holder.w"]
        assert isinstance(net.head, lora.Linear) and isinstance(net.holder, lora.ParamWrapper)

    def test_only_the_adapters_train(self, device: str) -> None:
        net = _net(device)
        peft = get_peft_model(net, LoraConfig(r=2, target_parameters=["w", "v"]))
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trained and all(".default." in n for n in trained)
        assert not net.holder.base_layer.base_layer.w.requires_grad

    def test_the_config_roundtrips_and_the_adapter_file_nests_as_the_reference_does(
        self, device: str, tmp_path: object
    ) -> None:
        net = _net(device)
        config = LoraConfig(r=2, target_parameters=["w", "v"])
        assert LoraConfig.from_dict(config.to_dict()) == config
        peft = get_peft_model(net, config)
        assert sorted(get_peft_model_state_dict(peft)) == [
            "base_model.model.holder.base_layer.lora_A.weight",
            "base_model.model.holder.base_layer.lora_B.weight",
            "base_model.model.holder.lora_A.weight",
            "base_model.model.holder.lora_B.weight",
        ]
        assert isinstance(peft, PeftModel)

    def test_a_second_adapter_joins_the_wrapper_of_a_parameter_it_shares(self, device: str) -> None:
        net = _net(device)
        peft = get_peft_model(
            net, LoraConfig(r=2, target_parameters=["w"], init_lora_weights=False)
        )
        peft.add_adapter("other", LoraConfig(r=3, target_parameters=["w"], init_lora_weights=False))
        assert isinstance(net.holder, lora.ParamWrapper)
        assert not isinstance(net.holder.base_layer, lora.ParamWrapper)  # joined, not nested
        assert sorted(net.holder.r.items()) == [("default", 2), ("other", 3)]
        assert peft.base_model.targeted_parameter_names == ["holder.w"]

    def test_a_second_adapter_on_another_parameter_wraps_around(self, device: str) -> None:
        net = _net(device)
        x = lucid.randn(4, 6, device=device)
        peft = get_peft_model(
            net, LoraConfig(r=2, target_parameters=["w"], init_lora_weights=False)
        )
        first = peft(x)
        peft.add_adapter("other", LoraConfig(r=2, target_parameters=["v"], init_lora_weights=False))
        outer = net.holder
        assert isinstance(outer, lora.ParamWrapper) and outer.parameter_name == "v"
        assert outer._adapter_names() == ["other"]
        assert _same(peft(x), first)  # added, not active: the first adapter still answers
        peft.set_adapter("other")
        assert not _same(peft(x), first)
        peft.set_adapter(["default", "other"])
        both = peft(x)
        assert not _same(both, first)
        peft.delete_adapter("other")
        peft.set_adapter("default")
        assert _same(peft(x), first)

    def test_merge_and_unload_gives_the_plain_model_back(self, device: str) -> None:
        net = _net(device)
        ids = {n: id(p) for n, p in net.named_parameters()}
        x = lucid.randn(4, 6, device=device)
        peft = get_peft_model(
            net, LoraConfig(r=2, target_parameters=["w", "v"], init_lora_weights=False)
        )
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and type(net.holder) is _Holder
            assert {n: id(p) for n, p in net.named_parameters()} == ids
            assert _close(net(x), adapted, atol=1e-4)
        assert all(p.requires_grad for p in net.parameters())

    def test_unload_undoes_a_merge_of_the_nested_wrappers(self, device: str) -> None:
        net = _net(device)
        x = lucid.randn(4, 6, device=device)
        plain = {n: p.clone() for n, p in net.named_parameters()}
        expected = net(x)
        peft = get_peft_model(
            net, LoraConfig(r=2, target_parameters=["w", "v"], init_lora_weights=False)
        )
        peft.base_model.merge_adapter()
        peft.unload()
        assert type(net.holder) is _Holder
        assert all(_close(p, plain[n], atol=1e-5) for n, p in net.named_parameters())
        assert _close(net(x), expected, atol=1e-5)

    def test_a_parameter_the_model_ties_cannot_be_merged(self, device: str) -> None:
        lucid.manual_seed(0)
        tied = _Tied().to(device)
        peft = get_peft_model(
            tied, LoraConfig(r=2, target_parameters=["holder.w"], init_lora_weights=False)
        )
        with pytest.raises(RuntimeError, match="other.w"):
            peft.merge_and_unload()

    def test_the_bias_of_a_wrapped_linear_trains_once_for_lora_only(self, device: str) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 4)).to(device)
        peft = get_peft_model(
            model, LoraConfig(r=2, target_parameters=["weight"], bias="lora_only")
        )
        trained = sorted(n for n, p in peft.named_parameters() if p.requires_grad)
        assert "base_model.model.0.base_layer.bias" in trained
        assert not model[0].base_layer.weight.requires_grad

    def test_the_compile_key_follows_the_active_adapter(self, device: str) -> None:
        net = _net(device)
        peft = get_peft_model(net, LoraConfig(r=2, target_parameters=["w", "v"]))
        peft.add_adapter("other", LoraConfig(r=2, target_parameters=["w"]))
        peft.set_adapter("default")
        before = signature._module_state(peft)
        peft.set_adapter("other")
        assert signature._module_state(peft) != before


class TestParameterRefusals:
    def test_a_parameter_that_matches_nothing_is_an_error_naming_it(self, device: str) -> None:
        net = _net(device)
        with pytest.raises(ValueError, match=r"Target parameters \['nope'\] not found"):
            get_peft_model(net, LoraConfig(r=2, target_parameters=["nope"]))
        assert type(net.holder) is _Holder and all(p.requires_grad for p in net.parameters())

    def test_missing_parameters_beside_matching_modules_only_warn(self, device: str) -> None:
        net = _net(device)
        with pytest.warns(RuntimeWarning, match="no parameter was matched"):
            get_peft_model(
                net, LoraConfig(r=2, target_modules=["head"], target_parameters=["nope"])
            )
        assert isinstance(net.head, lora.Linear)

    def test_a_parameter_of_the_top_level_module_is_refused(self, device: str) -> None:
        holder = _Holder().to(device)
        with pytest.raises(ValueError, match="top-level module"):
            get_peft_model(holder, LoraConfig(r=2, target_parameters=["w"]))

    def test_a_parameter_of_a_module_adapted_as_a_module_is_refused(self, device: str) -> None:
        net = _net(device)
        with pytest.raises(ValueError, match="also adapted as a module"):
            get_peft_model(
                net, LoraConfig(r=2, target_modules=["head"], target_parameters=["weight"])
            )
        assert type(net.head) is nn.Linear

    def test_dropout_cannot_apply_to_a_parameter_and_leaves_the_model_whole(
        self, device: str
    ) -> None:
        net = _net(device)
        with pytest.raises(ValueError, match="lora_dropout != 0"):
            get_peft_model(net, LoraConfig(r=2, lora_dropout=0.1, target_parameters=["w"]))
        assert type(net.holder) is _Holder and all(p.requires_grad for p in net.parameters())

    def test_a_late_failure_withdraws_the_whole_adapter(self, device: str) -> None:
        net = _net(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ``bias`` is 1-D: the second parameter cannot be adapted, after the first was built.
            net.holder.b = nn.Parameter(lucid.zeros(5, device=device))
            with pytest.raises(ValueError, match="1-D"):
                get_peft_model(net, LoraConfig(r=2, target_parameters=["w", "b"]))
        assert type(net.holder) is _Holder and "peft_config" not in net.__dict__

    def test_weighted_adapters_refuse_a_layer_that_is_not_linear(self, device: str) -> None:
        net = _net(device)
        peft = get_peft_model(
            net, LoraConfig(r=2, target_modules=["head"], target_parameters=["w"])
        )
        peft.add_adapter("other", LoraConfig(r=2, target_modules=["head"]))
        # ``default`` also adapts a parameter, which only a linear layer's factors can be averaged for.
        with pytest.raises(NotImplementedError, match="nn.Linear adapters only"):
            peft.add_weighted_adapter(["default", "other"], [0.5, 0.5], "mix")
