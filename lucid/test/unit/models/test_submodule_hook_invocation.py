"""Regression tests for submodule hook invocation across ``lucid/models/`` (LCD-312).

Calling ``self.<submodule>.forward(...)`` directly bypasses ``Module.__call__``,
skipping all forward pre-hooks, forward hooks, and backward hooks registered on
that submodule.  These tests verify both runtime hook invocation on the primary
models from issue #385 and an AST invariant across ``lucid/models/``.
"""

import ast
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.models import BERTConfig, BERTForCausalLM, BERTModel
from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
from lucid.models.generative.dit import DiTConfig, DiTForImageGeneration
from lucid.models.generative.mean_flow import (
    MeanFlowConfig,
    MeanFlowForImageGeneration,
)

_MODELS_ROOT = Path(__file__).resolve().parents[3] / "models"
_DEFERRED_DETECTION_FILES = frozenset(
    {
        "vision/faster_rcnn/_model.py",
        "vision/mask_rcnn/_model.py",
    }
)


def _find_direct_forward_calls(py_file: Path) -> list[str]:
    """Return ``path:lineno: expr`` for every non-``super().forward(...)`` call."""
    rel = py_file.relative_to(_MODELS_ROOT).as_posix()
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    hits: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward"
        ):
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == "super"
            ):
                continue
            hits.append(f"{rel}:{node.lineno}: {ast.unparse(node.func)}")
    return hits


class TestBERTSubmoduleHooks:
    def _tiny_bert_config(self) -> BERTConfig:
        return BERTConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            type_vocab_size=2,
            num_labels=3,
        )

    @pytest.mark.parametrize(
        "submodule_path",
        [
            "embeddings",
            "encoder",
            "encoder.layer.0",
            "encoder.layer.0.attention",
            "encoder.layer.0.attention.self",
        ],
    )
    def test_bert_model_submodule_forward_hook_fires(self, submodule_path: str) -> None:
        model = BERTModel(self._tiny_bert_config()).eval()
        target: nn.Module = model
        for part in submodule_path.split("."):
            target = target[int(part)] if part.isdigit() else getattr(target, part)  # type: ignore[index]

        calls: list[int] = []
        handle = target.register_forward_hook(lambda *_args: calls.append(1))
        try:
            input_ids = lucid.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=lucid.int64)
            out = model(input_ids)
        finally:
            handle.remove()

        assert out.last_hidden_state.shape == (2, 4, 32)
        assert len(calls) >= 1

    def test_bert_causal_lm_fires_encoder_hook(self) -> None:
        model = BERTForCausalLM(self._tiny_bert_config()).eval()
        calls: list[int] = []
        handle = model.bert.encoder.register_forward_hook(lambda *_args: calls.append(1))
        try:
            input_ids = lucid.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=lucid.int64)
            out = model(input_ids)
        finally:
            handle.remove()

        assert out.logits.shape == (2, 4, 64)
        assert len(calls) == 1


class TestMeanFlowSubmoduleHooks:
    def _tiny_config(self, **overrides: object) -> MeanFlowConfig:
        base: dict[str, object] = {
            "sample_size": 8,
            "patch_size": 2,
            "hidden_size": 32,
            "depth": 2,
            "num_heads": 4,
            "num_classes": 10,
        }
        base.update(overrides)
        return MeanFlowConfig(**base)  # type: ignore[arg-type]

    def test_mean_flow_forward_and_cfg_target_fire_backbone_hook(self) -> None:
        model = MeanFlowForImageGeneration(
            self._tiny_config(
                guidance_scale=2.0,
                guidance_mix=0.5,
                guidance_interval=(0.0, 1.0),
            )
        ).eval()
        calls: list[int] = []
        handle = model.mean_flow.register_forward_hook(lambda *_args: calls.append(1))
        try:
            x = lucid.randn((2, 4, 8, 8))
            labels = lucid.tensor([1, 2], dtype=lucid.int64)
            out = model(x, labels)
        finally:
            handle.remove()

        assert out.loss is not None
        assert len(calls) >= 3

    def test_mean_flow_generate_fires_backbone_hook(self) -> None:
        model = MeanFlowForImageGeneration(self._tiny_config()).eval()
        calls: list[int] = []
        handle = model.mean_flow.register_forward_hook(lambda *_args: calls.append(1))
        try:
            labels = lucid.tensor([1, 2], dtype=lucid.int64)
            gen = model.generate(
                n_samples=2,
                steps=2,
                labels=labels,
            )
        finally:
            handle.remove()

        assert gen.samples.shape == (2, 4, 8, 8)
        assert len(calls) == 2


class TestDiTSubmoduleHooks:
    def _tiny_config(self) -> DiTConfig:
        return DiTConfig(
            sample_size=8,
            patch_size=2,
            hidden_size=32,
            depth=2,
            num_heads=4,
            num_classes=10,
            num_train_timesteps=10,
        )

    def test_dit_forward_and_generate_fire_backbone_hook(self) -> None:
        model = DiTForImageGeneration(self._tiny_config()).eval()
        calls: list[int] = []
        handle = model.dit.register_forward_hook(lambda *_args: calls.append(1))
        try:
            x = lucid.randn((2, 4, 8, 8))
            labels = lucid.tensor([1, 2], dtype=lucid.int64)
            out = model(x, labels)
            assert out.loss is not None
            assert len(calls) == 1

            calls.clear()
            gen = model.generate(
                n_samples=2,
                steps=2,
                labels=labels,
            )
            assert gen.samples.shape == (2, 4, 8, 8)
            assert len(calls) == 2
        finally:
            handle.remove()


class TestDIAMONDSubmoduleHooksAndGuards:
    def _tiny_config(self, *, with_agent: bool = True) -> DIAMONDConfig:
        return DIAMONDConfig(
            sample_size=16,
            unet_channels=(8, 8),
            unet_layers=(1, 1),
            reward_channels=(8, 8),
            reward_layers=(1, 1),
            actor_channels=(8, 8),
            actor_layers=(1, 1),
            cond_dim=16,
            reward_cond_dim=8,
            reward_lstm_dim=16,
            actor_lstm_dim=16,
            num_actions=4,
            horizon=3,
            with_agent=with_agent,
        )

    def test_optional_agent_submodules_raise_when_disabled(self) -> None:
        model = DIAMONDModel(self._tiny_config(with_agent=False)).eval()
        obs = lucid.randn((2, 3, 16, 16))
        next_obs = lucid.randn((2, 3, 16, 16))
        act = lucid.zeros((2,), dtype=lucid.int64)

        with pytest.raises(ValueError, match="with_agent=False"):
            model.step_reward_end(obs, next_obs, act)

        with pytest.raises(ValueError, match="with_agent=False"):
            model.step_actor_critic(obs)

    def test_agent_submodule_hooks_fire_when_enabled(self) -> None:
        model = DIAMONDModel(self._tiny_config(with_agent=True)).eval()
        assert model.reward_end is not None
        assert model.actor_critic is not None

        reward_calls: list[int] = []
        actor_calls: list[int] = []
        h1 = model.reward_end.register_forward_hook(lambda *_args: reward_calls.append(1))
        h2 = model.actor_critic.register_forward_hook(lambda *_args: actor_calls.append(1))
        try:
            obs = lucid.randn((2, 3, 16, 16))
            next_obs = lucid.randn((2, 3, 16, 16))
            act = lucid.zeros((2,), dtype=lucid.int64)
            model.step_reward_end(obs, next_obs, act)
            model.step_actor_critic(obs)
        finally:
            h1.remove()
            h2.remove()

        assert len(reward_calls) == 1
        assert len(actor_calls) == 1


class TestNoDirectSubmoduleForwardCallsAST:
    def test_no_direct_submodule_forward_calls_in_models(self) -> None:
        violations: list[str] = []
        for py_file in sorted(_MODELS_ROOT.rglob("*.py")):
            rel = py_file.relative_to(_MODELS_ROOT).as_posix()
            if rel in _DEFERRED_DETECTION_FILES:
                continue
            violations.extend(_find_direct_forward_calls(py_file))
        assert (
            violations == []
        ), "Direct .forward(...) calls bypass Module.__call__ and skip hooks:\n" + "\n".join(
            violations
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "LCD-312: faster_rcnn/_model.py and mask_rcnn/_model.py had uncommitted "
            "changes on main during LCD-312 and were out of scope"
        ),
    )
    def test_no_direct_submodule_forward_calls_in_deferred_detection_models(
        self,
    ) -> None:
        violations: list[str] = []
        for rel in sorted(_DEFERRED_DETECTION_FILES):
            violations.extend(_find_direct_forward_calls(_MODELS_ROOT / rel))
        assert violations == []
