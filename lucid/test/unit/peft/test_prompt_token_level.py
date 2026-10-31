"""Prompt learning for token classification and question answering (BERT, RoFormer; CPU and Metal).

A tagger or a span reader has a row of logits per input position, so a prompt in front of the
embeddings must not show in its outputs: the head's input is cut back to the input's length, and
the model's own loss then reads labels and answer positions in the caller's coordinates.  A prefix
leaves the sequence as long as it was and needs no cut.  The reference library does *not* cut
(its logits keep a row per virtual token, so its own loss fails on labels); what it computes for
the real positions is the oracle here.
"""

import dataclasses
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import (
    PeftModel,
    PeftModelForQuestionAnswering,
    PeftModelForTokenClassification,
    get_peft_model,
)
from lucid.peft.tuners._prompt._sites import _HeadSite
from lucid.serialization import load_safetensors
from lucid.test.unit.peft._parity import adapter_library
from lucid.test.unit.peft.test_prompt_learning import _METHODS, _TEXT, _V, _ids

_BATCH, _LENGTH, _LABELS = 2, 5, 3


@dataclasses.dataclass(frozen=True)
class _Row:
    """One zoo family under one token-level task."""

    factory: str
    task: str
    head: str

    def build(self, device: str) -> nn.Module:
        lucid.manual_seed(0)
        extra: dict[str, object] = {"num_labels": _LABELS} if self.task == "TOKEN_CLS" else {}
        return create_model(self.factory, **_TEXT, **extra).eval().to(device)

    @property
    def wrapper(self) -> type[PeftModel]:
        if self.task == "TOKEN_CLS":
            return PeftModelForTokenClassification
        return PeftModelForQuestionAnswering


_ROWS = {
    "bert-token": _Row("bert_base_token_cls", "TOKEN_CLS", "classifier"),
    "roformer-token": _Row("roformer_token_cls", "TOKEN_CLS", "classifier"),
    "bert-qa": _Row("bert_base_qa", "QUESTION_ANS", "qa_outputs"),
    "roformer-qa": _Row("roformer_qa", "QUESTION_ANS", "qa_outputs"),
}
_TOKEN_ROWS = [name for name, row in _ROWS.items() if row.task == "TOKEN_CLS"]
_QA_ROWS = [name for name, row in _ROWS.items() if row.task == "QUESTION_ANS"]
_IN_FRONT = ["prompt", "p-tuning-mlp", "p-tuning-lstm"]
_ALL = list(_METHODS)


def _scores(row: _Row, out: Any) -> list[lucid.Tensor]:
    """The per-position outputs of a call: the logits, or the start and end logits."""
    if row.task == "TOKEN_CLS":
        return [out.logits]
    return [out.start_logits, out.end_logits]


def _adapted(row: _Row, method: str, device: str) -> PeftModel:
    return get_peft_model(row.build(device), _METHODS[method](row.task))


class TestOutputsAreTheInputsLength:
    @pytest.mark.parametrize("method", _ALL)
    @pytest.mark.parametrize("name", _ROWS)
    def test_no_row_per_virtual_token_reaches_the_output(
        self, name: str, method: str, device: str
    ) -> None:
        row = _ROWS[name]
        peft = _adapted(row, method, device)
        assert type(peft) is row.wrapper
        out = peft(_ids(device, (_BATCH, _LENGTH)))
        tail = (_LABELS,) if row.task == "TOKEN_CLS" else ()
        for scores in _scores(row, out):
            assert tuple(scores.shape) == (_BATCH, _LENGTH, *tail)

    @pytest.mark.parametrize("method", _IN_FRONT)
    @pytest.mark.parametrize("name", _ROWS)
    def test_the_cut_leaves_the_trunk_run_over_prompt_then_input(
        self, name: str, method: str, device: str
    ) -> None:
        # The head reads exactly the input's positions of the trunk's output for [prompt; input].
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        base = row.build(device)
        peft = _adapted(row, method, device)
        prompt = peft.base_model.prompt_encoder["default"].prompts(_BATCH, None)
        trunk = getattr(base, base.base_model_prefix)
        joined = lucid.cat([prompt, base.get_input_embeddings()(ids)], dim=1)
        hidden = trunk(inputs_embeds=joined).last_hidden_state[:, _V:]
        head = getattr(base, row.head)
        want = head(hidden)
        got = _scores(row, peft(ids))
        if row.task == "TOKEN_CLS":
            assert bool(lucid.allclose(got[0], want, atol=1e-5))
        else:
            assert bool(lucid.allclose(got[0], want[..., 0], atol=1e-5))
            assert bool(lucid.allclose(got[1], want[..., 1], atol=1e-5))

    @pytest.mark.parametrize("method", _ALL)
    @pytest.mark.parametrize("name", _ROWS)
    def test_a_disabled_adapter_is_the_base_model(
        self, name: str, method: str, device: str
    ) -> None:
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        base, peft = row.build(device), _adapted(row, method, device)
        with peft.disable_adapter():
            for got, want in zip(_scores(row, peft(ids)), _scores(row, base(ids)), strict=True):
                assert bool(lucid.allclose(got, want, atol=1e-6))


class TestLabelsAndPositions:
    @pytest.mark.parametrize("method", _ALL)
    @pytest.mark.parametrize("name", _TOKEN_ROWS)
    def test_labels_are_the_inputs_not_widened(self, name: str, method: str, device: str) -> None:
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        labels = lucid.tensor([[0, 1, 2, -100, 1], [2, 2, 0, 1, -100]], device=device).long()
        out = _adapted(row, method, device)(ids, labels=labels)
        logits = out.logits.reshape(-1, _LABELS)
        want = nn.functional.cross_entropy(logits, labels.reshape(-1), ignore_index=-100)
        assert abs(float(out.loss.item()) - float(want.item())) < 1e-5

    @pytest.mark.parametrize("method", _ALL)
    @pytest.mark.parametrize("name", _QA_ROWS)
    def test_answer_positions_are_in_the_inputs_coordinates(
        self, name: str, method: str, device: str
    ) -> None:
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        starts = lucid.tensor([1, 3], device=device).long()
        ends = lucid.tensor([2, 4], device=device).long()
        out = _adapted(row, method, device)(ids, start_positions=starts, end_positions=ends)
        start = nn.functional.cross_entropy(out.start_logits, starts)
        end = nn.functional.cross_entropy(out.end_logits, ends)
        assert abs(float(out.loss.item()) - float((start + end).item() / 2)) < 1e-5

    @pytest.mark.parametrize("name", _QA_ROWS)
    def test_a_position_past_the_input_is_skipped_as_it_is_without_a_prompt(
        self, name: str, device: str
    ) -> None:
        # The model's own rule: a position at or past the sequence length counts for nothing.
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        out = _adapted(row, "prompt", device)(
            ids,
            start_positions=lucid.tensor([_LENGTH, _LENGTH + 4], device=device).long(),
            end_positions=lucid.tensor([_LENGTH, _LENGTH], device=device).long(),
        )
        assert abs(float(out.loss.item())) < 1e-6


class TestMasksAndTypes:
    @pytest.mark.parametrize("method", _ALL)
    @pytest.mark.parametrize("name", _ROWS)
    def test_the_attention_mask_covers_the_prompt(
        self, name: str, method: str, device: str
    ) -> None:
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        peft = _adapted(row, method, device)
        plain = _scores(row, peft(ids))
        full = lucid.ones(_BATCH, _LENGTH, device=device).long()
        for got, want in zip(_scores(row, peft(ids, attention_mask=full)), plain, strict=True):
            assert bool(lucid.allclose(got, want, atol=1e-5))
        padded = lucid.ones(_BATCH, _LENGTH, device=device).long()
        padded[:, 0] = 0
        masked = _scores(row, peft(ids, attention_mask=padded))
        assert not bool(lucid.allclose(masked[0][:, 1:], plain[0][:, 1:], atol=1e-5))

    @pytest.mark.parametrize("name", ["bert-token", "bert-qa"])
    def test_token_types_follow_the_prompt(self, name: str, device: str) -> None:
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        peft = _adapted(row, "prompt", device)
        zeros = lucid.zeros(_BATCH, _LENGTH, device=device).long()
        for got, want in zip(
            _scores(row, peft(ids, token_type_ids=zeros)), _scores(row, peft(ids)), strict=True
        ):
            assert bool(lucid.allclose(got, want, atol=1e-5))
        ones = lucid.ones(_BATCH, _LENGTH, device=device).long()
        different = _scores(row, peft(ids, token_type_ids=ones))
        assert not bool(lucid.allclose(different[0], _scores(row, peft(ids))[0], atol=1e-5))


class TestTraining:
    @pytest.mark.parametrize("method", ["prompt", "p-tuning-mlp", "prefix"])
    @pytest.mark.parametrize("name", _ROWS)
    def test_the_prompt_and_the_head_train_and_nothing_else(
        self, name: str, method: str, device: str
    ) -> None:
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        peft = _adapted(row, method, device)
        assert peft.peft_config["default"].modules_to_save == [row.head]
        if row.task == "TOKEN_CLS":
            loss = peft(ids, labels=lucid.zeros(_BATCH, _LENGTH, device=device).long()).loss
        else:
            positions = lucid.tensor([1, 2], device=device).long()
            loss = peft(ids, start_positions=positions, end_positions=positions).loss
        loss.backward()
        named = [n for n, p in peft.named_parameters() if p.requires_grad]
        assert any("prompt_encoder" in n for n in named)
        assert any(f"{row.head}.modules_to_save" in n for n in named)
        assert all("prompt_encoder" in n or f"{row.head}.modules_to_save" in n for n in named)
        assert {n for n, p in peft.named_parameters() if p.grad is not None} == set(named)

    @pytest.mark.parametrize("name", ["bert-token", "roformer-token"])
    def test_a_tagger_learns_its_labels(self, name: str, device: str) -> None:
        row = _ROWS[name]
        ids = _ids(device, (4, _LENGTH))
        labels = (ids % _LABELS).long()
        peft = _adapted(row, "prompt", device)
        optimizer = optim.Adam([p for p in peft.parameters() if p.requires_grad], lr=5e-2)
        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            loss = peft(ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0]


class TestFile:
    @pytest.mark.parametrize("method", ["prompt", "prefix"])
    @pytest.mark.parametrize("name", ["bert-token", "roformer-qa"])
    def test_a_saved_adapter_loads_to_the_same_outputs(
        self, name: str, method: str, device: str, tmp_path: Path
    ) -> None:
        row = _ROWS[name]
        ids = _ids(device, (_BATCH, _LENGTH))
        peft = _adapted(row, method, device)
        for p in peft.parameters():
            if p.requires_grad:
                with lucid.no_grad():
                    p.copy_(p + 0.25)
        peft.save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        assert set(saved) == {
            "prompt_embeddings",
            f"base_model.{row.head}.weight",
            f"base_model.{row.head}.bias",
        }
        loaded = PeftModel.from_pretrained(row.build(device), tmp_path)
        assert type(loaded) is row.wrapper
        for got, want in zip(_scores(row, loaded(ids)), _scores(row, peft(ids)), strict=True):
            assert bool(lucid.allclose(got, want, atol=1e-5))


class TestTheHeadSite:
    def test_a_head_is_cut_only_when_the_prompt_is_in_front(self, device: str) -> None:
        row = _ROWS["bert-token"]
        cut = {}
        for method in ("prompt", "prefix"):
            head = _adapted(row, method, device).get_base_model().classifier
            cut[method] = any(isinstance(h, _HeadSite) for h in head._forward_pre_hooks.values())
        assert cut == {"prompt": True, "prefix": False}

    def test_a_head_called_with_more_than_one_tensor_is_refused(self, device: str) -> None:
        peft = _adapted(_ROWS["bert-token"], "prompt", device)
        head = peft.get_base_model().classifier
        with pytest.raises(TypeError, match="one"):
            head(x=lucid.zeros(_BATCH, 6, 16, device=device))
        with pytest.raises(TypeError, match="one"):
            head(lucid.zeros(_BATCH, 16, device=device))

    def test_a_head_reading_no_more_than_the_prompt_is_refused(self, device: str) -> None:
        peft = _adapted(_ROWS["bert-token"], "prompt", device)
        with pytest.raises(ValueError, match="virtual tokens"):
            peft.get_base_model().classifier(lucid.zeros(_BATCH, _V, 16, device=device))

    def test_unloading_takes_the_cut_out(self, device: str) -> None:
        row = _ROWS["bert-qa"]
        ids = _ids(device, (_BATCH, _LENGTH))
        base = row.build(device)
        peft = _adapted(row, "prompt", device)
        model = peft.unload()
        for got, want in zip(_scores(row, model(ids)), _scores(row, base(ids)), strict=True):
            assert bool(lucid.allclose(got, want, atol=1e-6))
        assert not any(m._forward_pre_hooks for m in model.modules())


# ── the reference library ────────────────────────────────────────────────────

_SIZES: dict[str, int] = {
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "vocab_size": 50,
    "max_position_embeddings": 64,
}
_REFERENCE: dict[str, tuple[str, str, str]] = {
    # row -> (family of the reference model, its class for the task, the config's own extras)
    "bert-token": ("Bert", "BertForTokenClassification", ""),
    "roformer-token": ("RoFormer", "RoFormerForTokenClassification", "embedding_size"),
    "bert-qa": ("Bert", "BertForQuestionAnswering", ""),
    "roformer-qa": ("RoFormer", "RoFormerForQuestionAnswering", "embedding_size"),
}


def _reference_model(name: str, ref: ModuleType, labels: int) -> Any:
    transformers = pytest.importorskip("transformers")
    family, cls, extra = _REFERENCE[name]
    ref.manual_seed(0)
    config = getattr(transformers, f"{family}Config")(
        **_SIZES,
        **({extra: _SIZES["hidden_size"]} if extra else {}),
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=labels,
    )
    return getattr(transformers, cls)(config).eval()


def _same_weights(mine: nn.Module, reference: Any) -> None:
    """Copy the reference model's weights; only a pooler the reference leaves out may stay unset."""
    weights = {k: v.detach().numpy() for k, v in reference.state_dict().items()}
    own = set(mine.state_dict())
    assert {k for k in own if k not in weights} <= {
        "bert.pooler.dense.weight",
        "bert.pooler.dense.bias",
    }
    mine.load_state_dict({k: lucid.tensor(weights[k]) for k in own if k in weights}, strict=False)


@pytest.mark.parity
class TestAgainstTheReference:
    """What the reference computes for the input's positions is what the cut head returns."""

    @pytest.mark.parametrize("method", ["prompt", "prefix"])
    @pytest.mark.parametrize("name", _ROWS)
    def test_the_outputs_are_the_references_for_the_real_positions(
        self, name: str, method: str, ref: ModuleType
    ) -> None:
        library = adapter_library()
        row = _ROWS[name]
        labels = _LABELS if row.task == "TOKEN_CLS" else 2
        reference = _reference_model(name, ref, labels)
        mine = create_model(row.factory, num_labels=labels, **_TEXT).eval()
        _same_weights(mine, reference)
        task = getattr(library.TaskType, row.task)
        theirs = library.get_peft_model(
            reference,
            (
                library.PromptTuningConfig(task_type=task, num_virtual_tokens=_V)
                if method == "prompt"
                else library.PrefixTuningConfig(task_type=task, num_virtual_tokens=_V)
            ),
        ).eval()
        ours = get_peft_model(mine, _METHODS[method](row.task)).eval()
        with lucid.no_grad():
            ours.base_model.prompt_encoder["default"].embedding.weight.copy_(
                lucid.tensor(theirs.prompt_encoder["default"].embedding.weight.detach().numpy())
            )
        ids = np.random.default_rng(0).integers(2, 50, (_BATCH, _LENGTH))
        mask = np.ones((_BATCH, _LENGTH), dtype=np.int64)
        mask[1, -1] = 0
        with ref.no_grad():
            want = theirs(input_ids=ref.tensor(ids), attention_mask=ref.tensor(mask))
        got = ours(lucid.tensor(ids).long(), attention_mask=lucid.tensor(mask).long())
        names = ["logits"] if row.task == "TOKEN_CLS" else ["start_logits", "end_logits"]
        front = _V if method == "prompt" else 0
        for key in names:
            theirs_rows = getattr(want, key).detach().numpy()
            assert theirs_rows.shape[1] == front + _LENGTH  # the reference keeps the virtual rows
            mine_rows = getattr(got, key).numpy()
            assert mine_rows.shape[1] == _LENGTH
            assert np.abs(mine_rows - theirs_rows[:, front:]).max() < 1e-5
