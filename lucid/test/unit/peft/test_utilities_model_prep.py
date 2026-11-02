"""``cast_mixed_precision_params``, ``shift_tokens_right``, ``prepare_model_for_kbit_training``
and ``find_kappa_target_modules``."""

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    LoraConfig,
    cast_mixed_precision_params,
    find_kappa_target_modules,
    get_peft_model,
    prepare_model_for_kbit_training,
    shift_tokens_right,
)


def _diag(*values: float) -> lucid.Tensor:
    """A square matrix whose singular values are ``values``."""
    n = len(values)
    return lucid.tensor([[values[i] if i == j else 0.0 for j in range(n)] for i in range(n)])


def _linear(*singular_values: float) -> nn.Linear:
    layer = nn.Linear(len(singular_values), len(singular_values), bias=False)
    layer.weight.data = _diag(*singular_values)
    return layer


class TestCastMixedPrecisionParams:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
    def test_frozen_to_dtype_and_trainable_to_float32(self, dtype: lucid.dtype) -> None:
        peft = get_peft_model(
            nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)), LoraConfig(r=2, target_modules=["0"])
        )
        cast_mixed_precision_params(peft, dtype)
        for name, param in peft.named_parameters():
            assert param.dtype == (lucid.float32 if param.requires_grad else dtype), name
        assert any(p.requires_grad for p in peft.parameters())
        assert any(not p.requires_grad for p in peft.parameters())

    def test_a_trainable_half_parameter_is_raised_to_float32(self) -> None:
        net = nn.Sequential(nn.Linear(2, 2)).to(lucid.float16)
        cast_mixed_precision_params(net, lucid.float16)
        assert {p.dtype for p in net.parameters()} == {lucid.float32}

    def test_values_survive_the_cast(self) -> None:
        net = nn.Sequential(nn.Linear(2, 2, bias=False))
        net[0].weight.data = lucid.tensor([[0.5, 1.0], [2.0, -4.0]])
        net.requires_grad_(False)
        cast_mixed_precision_params(net, lucid.bfloat16)
        assert net[0].weight.dtype == lucid.bfloat16
        assert net[0].weight.tolist() == [[0.5, 1.0], [2.0, -4.0]]

    def test_a_shared_parameter_is_cast_once(self) -> None:
        a = nn.Linear(2, 2)
        net = nn.Sequential(a, a)
        net.requires_grad_(False)
        cast_mixed_precision_params(net, lucid.float16)
        assert net[0].weight is net[1].weight and net[0].weight.dtype == lucid.float16


class TestShiftTokensRight:
    def test_first_column_is_the_start_token_and_the_rest_shift(self) -> None:
        ids = lucid.tensor([[5, 6, 7], [8, 9, 10]])
        out = shift_tokens_right(ids, pad_token_id=0, decoder_start_token_id=2)
        assert out.tolist() == [[2, 5, 6], [2, 8, 9]]

    def test_ignore_index_becomes_the_pad_token(self) -> None:
        ids = lucid.tensor([[5, -100, -100, 7]])
        out = shift_tokens_right(ids, pad_token_id=1, decoder_start_token_id=2)
        assert out.tolist() == [[2, 5, 1, 1]]

    def test_the_start_token_is_not_rewritten_even_if_it_is_the_ignore_index(self) -> None:
        out = shift_tokens_right(
            lucid.tensor([[4, 5]]), pad_token_id=9, decoder_start_token_id=-100
        )
        assert out.tolist() == [[9, 4]]

    def test_input_is_untouched_and_dtype_and_shape_kept(self) -> None:
        ids = lucid.tensor([[1, -100, 3]])
        out = shift_tokens_right(ids, 0, 2)
        assert ids.tolist() == [[1, -100, 3]]
        assert out.dtype == ids.dtype and out.shape == ids.shape

    def test_a_single_column_is_all_start_tokens(self) -> None:
        assert shift_tokens_right(lucid.tensor([[7], [8]]), 0, 2).tolist() == [[2], [2]]

    def test_pad_token_is_required(self) -> None:
        with pytest.raises(ValueError, match="pad_token_id"):
            shift_tokens_right(lucid.tensor([[1, 2]]), None, 2)

    def test_rank_must_be_two(self) -> None:
        with pytest.raises(ValueError, match="batch_size, sequence_length"):
            shift_tokens_right(lucid.tensor([1, 2, 3]), 0, 2)

    def test_on_metal(self) -> None:
        ids = lucid.tensor([[5, 6, -100]], device="metal")
        out = shift_tokens_right(ids, 0, 2)
        assert out.device.type == "metal" and out.tolist() == [[2, 5, 6]]


class _Half(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 4)
        self.register_buffer("scale", lucid.ones(4).to(lucid.float16))


class TestPrepareModelForKbitTraining:
    def test_it_freezes_every_parameter(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        prepare_model_for_kbit_training(net)
        assert not any(p.requires_grad for p in net.parameters())

    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
    def test_half_precision_parameters_become_float32_with_their_values(
        self, dtype: lucid.dtype
    ) -> None:
        net = _Half().to(dtype)
        before = net.fc.weight.tolist()
        prepare_model_for_kbit_training(net)
        assert {p.dtype for p in net.parameters()} == {lucid.float32}
        assert net.fc.weight.tolist() == before

    def test_other_dtypes_are_left_alone(self) -> None:
        net = nn.Sequential(nn.Linear(2, 2))
        net[0].weight.data = net[0].weight.data.to(lucid.float64)
        prepare_model_for_kbit_training(net)
        assert net[0].weight.dtype == lucid.float64

    def test_it_returns_the_model(self) -> None:
        net = nn.Sequential(nn.Linear(2, 2))
        assert prepare_model_for_kbit_training(net) is net

    def test_the_options_are_accepted_and_change_nothing(self) -> None:
        net = nn.Sequential(nn.Linear(2, 2))
        out = prepare_model_for_kbit_training(
            net,
            use_gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            auto_clear_cache=False,
        )
        assert out is net and not any(p.requires_grad for p in net.parameters())

    def test_an_adapter_added_afterwards_trains(self) -> None:
        net = prepare_model_for_kbit_training(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)))
        peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"]))
        trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in n for n in trainable)

    @pytest.mark.parametrize(
        ("attribute", "value", "reported"),
        [
            ("is_loaded_in_8bit", True, "8-bit"),
            ("is_loaded_in_4bit", True, "4-bit"),
            ("quantization_method", "gptq", "gptq"),
            ("quantization_method", "hqq", "hqq"),
            ("hqq_quantized", True, "hqq"),
        ],
    )
    def test_a_model_that_reports_quantization_is_refused_and_left_alone(
        self, attribute: str, value: object, reported: str
    ) -> None:
        net = nn.Sequential(nn.Linear(2, 2))
        setattr(net, attribute, value)
        with pytest.raises(NotImplementedError, match=reported):
            prepare_model_for_kbit_training(net)
        assert all(p.requires_grad for p in net.parameters())

    def test_an_unknown_quantization_method_is_not_a_quantized_model(self) -> None:
        net = nn.Sequential(nn.Linear(2, 2))
        net.quantization_method = "none"
        prepare_model_for_kbit_training(net)
        assert not any(p.requires_grad for p in net.parameters())


class TestFindKappaTargetModules:
    def test_lowest_condition_numbers_first(self) -> None:
        net = nn.Sequential(
            _linear(1.0, 1.0), _linear(8.0, 1.0), _linear(2.0, 1.0), _linear(4.0, 1.0)
        )
        picked = find_kappa_target_modules(net, top_p=0.5)
        assert picked == {"target_modules": ["0", "2"], "target_parameters": None}

    def test_the_condition_number_is_largest_over_smallest(self) -> None:
        # kappas: "0" = 3, "1" = 2; the better-conditioned one is picked alone
        net = nn.Sequential(_linear(6.0, 2.0), _linear(5.0, 4.0, 2.5))
        assert find_kappa_target_modules(net, top_p=0.5)["target_modules"] == ["1"]

    def test_the_fraction_is_floored_but_at_least_one(self) -> None:
        net = nn.Sequential(*[_linear(float(i + 1), 1.0) for i in range(5)])
        assert len(find_kappa_target_modules(net, top_p=0.5)["target_modules"] or []) == 2
        assert find_kappa_target_modules(net, top_p=0.0)["target_modules"] == ["0"]
        assert find_kappa_target_modules(net, top_p=1.0)["target_modules"] == [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]

    def test_a_singular_matrix_has_a_finite_number(self) -> None:
        net = nn.Sequential(_linear(1.0, 0.0), _linear(3.0, 1.0))
        assert find_kappa_target_modules(net, top_p=0.5)["target_modules"] == ["1"]

    def test_names_are_dotted_paths_and_only_linear_modules_count(self) -> None:
        net = nn.Sequential(nn.Sequential(_linear(2.0, 1.0)), nn.LayerNorm(2), nn.ReLU())
        assert find_kappa_target_modules(net, top_p=1.0)["target_modules"] == ["0.0"]

    def test_a_model_without_linear_modules(self) -> None:
        assert find_kappa_target_modules(nn.Sequential(nn.ReLU())) == {
            "target_modules": [],
            "target_parameters": None,
        }

    def test_large_weights_are_skipped(self) -> None:
        net = nn.Sequential(_linear(2.0, 1.0, 1.0), _linear(1.0, 1.0))
        picked = find_kappa_target_modules(net, top_p=1.0, max_dim_size_to_analyze=2)
        assert picked["target_modules"] == ["1"]

    def test_half_precision_weights_are_analysed(self) -> None:
        net = nn.Sequential(_linear(2.0, 1.0), _linear(4.0, 1.0)).to(lucid.float16)
        assert find_kappa_target_modules(net, top_p=0.5)["target_modules"] == ["0"]

    def test_top_p_outside_the_unit_interval_is_refused(self) -> None:
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="top_p"):
                find_kappa_target_modules(nn.Sequential(_linear(1.0, 1.0)), top_p=bad)

    def test_show_progress_changes_nothing(self) -> None:
        net = nn.Sequential(_linear(2.0, 1.0), _linear(4.0, 1.0))
        assert find_kappa_target_modules(net, show_progress=False) == find_kappa_target_modules(net)


class _Experts(nn.Module):
    def __init__(self, gate_up: lucid.Tensor, down: lucid.Tensor) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up)
        self.down_proj = nn.Parameter(down)


class _MoE(nn.Module):
    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        self.router = _linear(1.0, 1.0)
        self.experts = experts


def _stack(*mats: lucid.Tensor) -> lucid.Tensor:
    return lucid.stack(list(mats), dim=0)


class TestFindKappaFusedExperts:
    def test_three_dimensional_parameters_are_returned_as_target_parameters(self) -> None:
        # gate_up: experts with kappa 4 and 2 -> mean 3; down: kappa 1 and 1 -> mean 1
        experts = _Experts(
            _stack(_diag(4.0, 1.0), _diag(2.0, 1.0)), _stack(_diag(1.0, 1.0), _diag(1.0, 1.0))
        )
        picked = find_kappa_target_modules(_MoE(experts), top_p=0.5)
        assert picked["target_parameters"] == ["experts.down_proj"]
        assert picked["target_modules"] == ["router"]

    def test_the_fraction_applies_to_parameters_too(self) -> None:
        experts = _Experts(_stack(_diag(4.0, 1.0)), _stack(_diag(2.0, 1.0)))
        picked = find_kappa_target_modules(_MoE(experts), top_p=1.0)
        assert picked["target_parameters"] == ["experts.down_proj", "experts.gate_up_proj"]

    def test_custom_suffixes_replace_the_defaults(self) -> None:
        experts = _Experts(_stack(_diag(4.0, 1.0)), _stack(_diag(2.0, 1.0)))
        picked = find_kappa_target_modules(
            _MoE(experts), top_p=1.0, moe_param_suffixes=(".gate_up_proj",)
        )
        assert picked["target_parameters"] == ["experts.gate_up_proj"]

    def test_an_unmatched_suffix_gives_none(self) -> None:
        experts = _Experts(_stack(_diag(4.0, 1.0)), _stack(_diag(2.0, 1.0)))
        picked = find_kappa_target_modules(_MoE(experts), moe_param_suffixes=(".nothing",))
        assert picked["target_parameters"] is None

    def test_two_dimensional_parameters_with_those_names_are_not_experts(self) -> None:
        experts = _Experts(_diag(4.0, 1.0), _diag(2.0, 1.0))
        assert find_kappa_target_modules(_MoE(experts))["target_parameters"] is None

    def test_oversized_experts_are_skipped(self) -> None:
        experts = _Experts(_stack(_diag(4.0, 1.0, 1.0)), _stack(_diag(2.0, 1.0)))
        picked = find_kappa_target_modules(_MoE(experts), top_p=1.0, max_dim_size_to_analyze=2)
        assert picked["target_parameters"] == ["experts.down_proj"]

    def test_the_result_feeds_a_lora_config(self) -> None:
        net = nn.Sequential(_linear(1.0, 1.0), _linear(5.0, 1.0))
        picked = find_kappa_target_modules(net, top_p=0.5)
        peft = get_peft_model(net, LoraConfig(r=2, target_modules=picked["target_modules"]))
        assert peft.base_model.targeted_module_names == ["0"]
