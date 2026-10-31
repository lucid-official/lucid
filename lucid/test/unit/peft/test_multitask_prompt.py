"""Multitask prompt tuning: a shared prompt scaled by a low-rank factor pair per task (CPU and Metal).

The prompt of a batch row is ``embedding(ids) * (cols[task] @ rows[task])``, the task of each row
given by ``task_ids`` on the call.  What these tests pin: the closed form, that a mixed batch is
the rows run alone, the adapter file in the reference's layout (``prompt_embeddings``,
``prefix_task_cols``, ``prefix_task_rows``), the start from a source adapter, and the reference's
numbers.
"""

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    LoraConfig,
    MultitaskPromptEmbedding,
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftType,
    PromptTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners.multitask_prompt_tuning import MultitaskPromptTuningModel
from lucid.serialization import load_safetensors
from lucid.test.unit.peft._parity import adapter_library
from lucid.test.unit.peft.test_prompt_learning import _T5, _TEXT, _V, _gpt2, _ids

_TASKS, _RANKS = 3, 2
_BATCH, _LENGTH = 4, 5


def _config(task: str | None = "SEQ_CLS", **fields: Any) -> MultitaskPromptTuningConfig:
    fields = {"num_tasks": _TASKS, "num_ranks": _RANKS, **fields}
    return MultitaskPromptTuningConfig(num_virtual_tokens=_V, task_type=task, **fields)


def _classifier(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return create_model("bert_base_cls", num_labels=3, **_TEXT).eval().to(device)


def _peft(device: str, config: PeftConfig | None = None) -> PeftModel:
    return get_peft_model(_classifier(device), config or _config())


def _spread(peft: PeftModel) -> None:
    """Give every task its own, clearly different factors (they start near zero)."""
    encoder = peft.base_model.prompt_encoder["default"]
    with lucid.no_grad():
        encoder.prefix_task_cols.normal_(0.0, 1.0)
        encoder.prefix_task_rows.normal_(0.0, 1.0)


def _tasks(device: str, *ids: int) -> lucid.Tensor:
    return lucid.tensor(list(ids), device=device).long()


_MIXED = (0, 2, 1, 2)


class TestConfig:
    def test_the_fields_are_the_references(self) -> None:
        config = MultitaskPromptTuningConfig(num_virtual_tokens=4)
        assert config.peft_type == PeftType.MULTITASK_PROMPT_TUNING
        assert config.prompt_tuning_init == MultitaskPromptTuningInit.RANDOM
        assert (config.num_tasks, config.num_ranks, config.prompt_tuning_init_task) == (1, 1, 0)
        assert config.prompt_tuning_init_state_dict_path is None
        assert {"num_tasks", "num_ranks", "prompt_tuning_init_task"} <= set(config.to_dict())
        assert {"prompt_tuning_init_state_dict_path", "num_virtual_tokens"} <= set(config.to_dict())

    def test_a_config_round_trips_and_is_found_by_its_type(self) -> None:
        config = _config(prompt_tuning_init="EXACT_SOURCE_TASK", prompt_tuning_init_task=2)
        again = PeftConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        assert type(again) is MultitaskPromptTuningConfig and again == config

    @pytest.mark.parametrize("field", ["num_tasks", "num_ranks"])
    def test_a_count_must_be_positive(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            _config(**{field: 0})
        with pytest.raises(TypeError, match=field):
            _config(**{field: 1.5})

    def test_the_source_task_may_be_the_first_but_not_negative(self) -> None:
        assert _config(prompt_tuning_init_task=0).prompt_tuning_init_task == 0
        with pytest.raises(ValueError, match="prompt_tuning_init_task"):
            _config(prompt_tuning_init_task=-1)

    def test_prompt_tunings_own_rules_still_hold(self) -> None:
        with pytest.raises(ValueError, match="tokenizer_name_or_path"):
            _config(prompt_tuning_init="TEXT")
        with pytest.raises(ValueError, match="prompt_tuning_init"):
            _config(prompt_tuning_init="SAMPLE_VOCAB")


class TestPrompt:
    def test_a_rows_prompt_is_the_shared_prompt_scaled_by_its_tasks_factors(
        self, device: str
    ) -> None:
        peft = _peft(device)
        _spread(peft)
        encoder = peft.base_model.prompt_encoder["default"]
        prompts = encoder.prompts(_BATCH, _tasks(device, *_MIXED))
        assert tuple(prompts.shape) == (_BATCH, _V, 16)
        shared = encoder.embedding.weight
        for row, task in enumerate(_MIXED):
            factors = encoder.prefix_task_cols[task] @ encoder.prefix_task_rows[task]
            assert bool(lucid.allclose(prompts[row], shared * factors, atol=1e-5))

    def test_a_mixed_batch_is_its_rows_run_alone(self, device: str) -> None:
        peft = _peft(device)
        _spread(peft)
        ids = _ids(device, (_BATCH, _LENGTH))
        mixed = peft(ids, task_ids=_tasks(device, *_MIXED)).logits
        for row, task in enumerate(_MIXED):
            alone = peft(ids[row : row + 1], task_ids=_tasks(device, task)).logits
            assert bool(lucid.allclose(mixed[row : row + 1], alone, atol=1e-5))
        assert not bool(
            lucid.allclose(
                mixed[1], peft(ids[1:2], task_ids=_tasks(device, 0)).logits[0], atol=1e-5
            )
        )

    def test_the_prompt_is_in_front_of_the_input_as_prompt_tuning_puts_it(
        self, device: str
    ) -> None:
        peft = _peft(device)
        _spread(peft)
        base = _classifier(device)
        ids = _ids(device, (_BATCH, _LENGTH))
        tasks = _tasks(device, *_MIXED)
        prompt = peft.base_model.prompt_encoder["default"].prompts(_BATCH, tasks)
        joined = lucid.cat([prompt, base.get_input_embeddings()(ids)], dim=1)
        want = base.classifier(base.bert(inputs_embeds=joined).pooler_output)
        assert bool(lucid.allclose(peft(ids, task_ids=tasks).logits, want, atol=1e-5))

    def test_a_disabled_adapter_runs_the_base(self, device: str) -> None:
        peft, base = _peft(device), _classifier(device)
        ids = _ids(device, (_BATCH, _LENGTH))
        with peft.disable_adapter():
            got = peft(ids, task_ids=_tasks(device, *_MIXED)).logits
        assert bool(lucid.allclose(got, base(ids).logits, atol=1e-6))


class TestTaskIds:
    def test_they_are_required(self, device: str) -> None:
        with pytest.raises(ValueError, match="task_ids cannot be None"):
            _peft(device)(_ids(device, (_BATCH, _LENGTH)))

    def test_one_task_per_row(self, device: str) -> None:
        peft, ids = _peft(device), _ids(device, (_BATCH, _LENGTH))
        with pytest.raises(ValueError, match="one task per row"):
            peft(ids, task_ids=_tasks(device, 0, 1))
        with pytest.raises(ValueError, match="one task per row"):
            peft(ids, task_ids=lucid.zeros(_BATCH, 1, device=device).long())

    def test_a_task_outside_the_table_is_an_index_error_on_every_device(self, device: str) -> None:
        peft, ids = _peft(device), _ids(device, (_BATCH, _LENGTH))
        with pytest.raises(IndexError, match=r"\[0, 3\)"):
            peft(ids, task_ids=_tasks(device, 0, 1, 2, _TASKS))
        with pytest.raises(IndexError, match=r"\[0, 3\)"):
            peft(ids, task_ids=_tasks(device, 0, 1, 2, -1))

    def test_they_are_integers(self, device: str) -> None:
        peft, ids = _peft(device), _ids(device, (_BATCH, _LENGTH))
        with pytest.raises(ValueError, match="integers"):
            peft(ids, task_ids=lucid.zeros(_BATCH, device=device))

    def test_the_scope_ends_with_the_call(self, device: str) -> None:
        peft, ids = _peft(device), _ids(device, (_BATCH, _LENGTH))
        peft(ids, task_ids=_tasks(device, *_MIXED))
        assert peft.base_model._task_ids is None
        with pytest.raises(IndexError):
            peft(ids, task_ids=_tasks(device, 0, 1, 2, 9))
        assert peft.base_model._task_ids is None
        with pytest.raises(ValueError, match="task_ids cannot be None"):
            peft(ids)

    def test_an_adapter_with_one_prompt_refuses_them(self, device: str) -> None:
        ids, tasks = _ids(device, (_BATCH, _LENGTH)), _tasks(device, *_MIXED)
        plain = get_peft_model(_classifier(device), PromptTuningConfig(num_virtual_tokens=_V))
        with pytest.raises(ValueError, match="task_ids choose a prompt per task"):
            plain(ids, task_ids=tasks)
        lora = get_peft_model(
            _classifier(device), LoraConfig(r=2, target_modules=["query"], task_type="SEQ_CLS")
        )
        with pytest.raises(ValueError, match="task_ids given to an adapter whose prompt is not"):
            lora(ids, task_ids=tasks)


class TestTraining:
    def test_the_shared_prompt_the_factors_and_the_head_train(self, device: str) -> None:
        peft = _peft(device)
        _spread(peft)
        ids = _ids(device, (_BATCH, _LENGTH))
        labels = _tasks(device, 0, 1, 2, 1)
        peft(ids, task_ids=_tasks(device, 0, 0, 2, 2), labels=labels).loss.backward()
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert {
            "base_model.prompt_encoder.default.embedding.weight",
            "base_model.prompt_encoder.default.prefix_task_cols",
            "base_model.prompt_encoder.default.prefix_task_rows",
        } <= trained
        assert all("prompt_encoder" in n or "classifier.modules_to_save" in n for n in trained)
        assert {n for n, p in peft.named_parameters() if p.grad is not None} == trained

    def test_only_the_tasks_in_the_batch_get_a_gradient(self, device: str) -> None:
        peft = _peft(device)
        _spread(peft)
        ids = _ids(device, (2, _LENGTH))
        peft(ids, task_ids=_tasks(device, 0, 2), labels=_tasks(device, 0, 1)).loss.backward()
        cols = peft.base_model.prompt_encoder["default"].prefix_task_cols.grad
        assert cols is not None
        used = [float(cols[task].abs().sum().item()) for task in range(_TASKS)]
        assert used[0] > 0 and used[2] > 0 and used[1] == 0

    def test_a_loss_goes_down(self, device: str) -> None:
        import lucid.optim as optim

        peft = _peft(device)
        ids = _ids(device, (_BATCH, _LENGTH))
        labels, tasks = _tasks(device, 0, 1, 2, 1), _tasks(device, *_MIXED)
        optimizer = optim.Adam([p for p in peft.parameters() if p.requires_grad], lr=5e-2)
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            loss = peft(ids, task_ids=tasks, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0]


class TestFile:
    def test_the_file_holds_the_references_keys(self, device: str, tmp_path: Path) -> None:
        peft = _peft(device)
        assert set(get_peft_model_state_dict(peft)) == {
            "prompt_embeddings",
            "prefix_task_cols",
            "prefix_task_rows",
            "base_model.classifier.weight",
            "base_model.classifier.bias",
        }
        peft.save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        encoder = peft.base_model.prompt_encoder["default"]
        assert tuple(saved["prefix_task_cols"].shape) == (_TASKS, _V, _RANKS)
        assert tuple(saved["prefix_task_rows"].shape) == (_TASKS, _RANKS, 16)
        assert bool(lucid.allclose(saved["prompt_embeddings"], encoder.embedding.weight.cpu()))
        written = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
        assert written["peft_type"] == "MULTITASK_PROMPT_TUNING"
        assert (written["num_tasks"], written["num_ranks"]) == (_TASKS, _RANKS)

    def test_a_saved_adapter_loads_to_the_same_outputs(self, device: str, tmp_path: Path) -> None:
        peft = _peft(device)
        _spread(peft)
        ids, tasks = _ids(device, (_BATCH, _LENGTH)), _tasks(device, *_MIXED)
        peft.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_classifier(device), tmp_path)
        assert type(loaded) is PeftModelForSequenceClassification
        assert type(loaded.base_model) is MultitaskPromptTuningModel
        assert loaded.peft_config["default"].inference_mode
        assert bool(
            lucid.allclose(
                loaded(ids, task_ids=tasks).logits, peft(ids, task_ids=tasks).logits, atol=1e-5
            )
        )
        assert not [n for n, p in loaded.named_parameters() if p.requires_grad]

    def test_a_file_loads_into_a_second_adapter_slot(self, device: str, tmp_path: Path) -> None:
        peft = _peft(device)
        _spread(peft)
        peft.save_pretrained(tmp_path)
        other = _peft(device)
        other.load_adapter(tmp_path, "from-file")
        other.set_adapter("from-file")
        ids, tasks = _ids(device, (_BATCH, _LENGTH)), _tasks(device, *_MIXED)
        assert bool(
            lucid.allclose(
                other(ids, task_ids=tasks).logits, peft(ids, task_ids=tasks).logits, atol=1e-5
            )
        )


# ── starting from a source adapter ───────────────────────────────────────────


def _source(device: str, directory: Path) -> tuple[str, dict[str, lucid.Tensor]]:
    """A 3-task adapter, trained apart (its factors are spread), saved; the path and its tensors."""
    peft = _peft(device)
    _spread(peft)
    peft.save_pretrained(directory)
    return str(directory / "adapter_model.safetensors"), load_safetensors(
        str(directory / "adapter_model.safetensors")
    )


def _start(device: str, path: str | None, init: str, **fields: Any) -> MultitaskPromptEmbedding:
    peft = _peft(
        device, _config(prompt_tuning_init=init, prompt_tuning_init_state_dict_path=path, **fields)
    )
    return peft.base_model.prompt_encoder["default"]


class TestStartFromASourceAdapter:
    def test_only_source_shared_takes_the_prompt_alone(self, device: str, tmp_path: Path) -> None:
        path, source = _source(device, tmp_path)
        encoder = _start(device, path, "ONLY_SOURCE_SHARED")
        assert bool(lucid.allclose(encoder.embedding.weight.cpu(), source["prompt_embeddings"]))
        assert not bool(lucid.allclose(encoder.prefix_task_cols.cpu(), source["prefix_task_cols"]))
        assert float(encoder.prefix_task_cols.abs().max().item()) < 0.2  # drawn, std 0.02

    def test_the_exact_source_task_takes_one_tasks_factors(
        self, device: str, tmp_path: Path
    ) -> None:
        path, source = _source(device, tmp_path)
        encoder = _start(device, path, "EXACT_SOURCE_TASK", num_tasks=1, prompt_tuning_init_task=1)
        assert tuple(encoder.prefix_task_cols.shape) == (1, _V, _RANKS)
        assert bool(lucid.allclose(encoder.embedding.weight.cpu(), source["prompt_embeddings"]))
        assert bool(
            lucid.allclose(encoder.prefix_task_cols[0].cpu(), source["prefix_task_cols"][1])
        )
        assert bool(
            lucid.allclose(encoder.prefix_task_rows[0].cpu(), source["prefix_task_rows"][1])
        )

    def test_the_average_takes_the_mean_over_the_source_tasks(
        self, device: str, tmp_path: Path
    ) -> None:
        path, source = _source(device, tmp_path)
        encoder = _start(device, path, "AVERAGE_SOURCE_TASKS", num_tasks=1)
        for name in ("prefix_task_cols", "prefix_task_rows"):
            mean = source[name].mean(dim=0)
            assert bool(lucid.allclose(getattr(encoder, name)[0].cpu(), mean, atol=1e-6))

    def test_a_target_with_more_tasks_than_the_derived_factors_is_refused(
        self, device: str, tmp_path: Path
    ) -> None:
        path, _ = _source(device, tmp_path)
        with pytest.raises(ValueError, match="needs"):
            _start(device, path, "AVERAGE_SOURCE_TASKS", num_tasks=2)

    def test_the_source_must_be_named_and_readable(self, device: str, tmp_path: Path) -> None:
        path, _ = _source(device, tmp_path)
        for init in ("AVERAGE_SOURCE_TASKS", "EXACT_SOURCE_TASK", "ONLY_SOURCE_SHARED"):
            with pytest.raises(ValueError, match="needs to be specified"):
                _start(device, None, init, num_tasks=1)
        with pytest.raises(ValueError, match="safetensors"):
            _start(device, str(tmp_path / "state.bin"), "ONLY_SOURCE_SHARED")
        with pytest.raises(IndexError, match="prompt_tuning_init_task=5"):
            _start(device, path, "EXACT_SOURCE_TASK", num_tasks=1, prompt_tuning_init_task=5)

    def test_a_source_without_the_factors_or_of_another_width_is_refused(
        self, device: str, tmp_path: Path
    ) -> None:
        from lucid.serialization import save_safetensors

        _, source = _source(device, tmp_path)
        bare = tmp_path / "bare.safetensors"
        save_safetensors({"prompt_embeddings": source["prompt_embeddings"]}, str(bare))
        with pytest.raises(ValueError, match="prefix_task_cols"):
            _start(device, str(bare), "EXACT_SOURCE_TASK", num_tasks=1)
        assert _start(device, str(bare), "ONLY_SOURCE_SHARED") is not None
        wide = tmp_path / "wide.safetensors"
        save_safetensors({"prompt_embeddings": lucid.zeros(_V + 1, 16)}, str(wide))
        with pytest.raises(ValueError, match="needs"):
            _start(device, str(wide), "ONLY_SOURCE_SHARED")

    def test_an_adapter_built_from_a_source_loads_without_it(
        self, device: str, tmp_path: Path
    ) -> None:
        path, _ = _source(device, tmp_path / "source")
        peft = _peft(
            device,
            _config(
                prompt_tuning_init="EXACT_SOURCE_TASK",
                prompt_tuning_init_state_dict_path=path,
                prompt_tuning_init_task=2,
                num_tasks=1,
            ),
        )
        peft.save_pretrained(tmp_path / "target")
        (tmp_path / "source" / "adapter_model.safetensors").unlink()
        loaded = PeftModel.from_pretrained(_classifier(device), tmp_path / "target")
        ids, tasks = _ids(device, (2, _LENGTH)), _tasks(device, 0, 0)
        assert bool(
            lucid.allclose(
                loaded(ids, task_ids=tasks).logits, peft(ids, task_ids=tasks).logits, atol=1e-5
            )
        )

    def test_text_initialisation_is_prompt_tunings(self, device: str) -> None:
        class Chars:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 50 for c in text]

        model = _classifier(device)
        config = _config(
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="ab",
            tokenizer_name_or_path="chars",
            token_dim=16,
            num_transformer_submodules=1,
        )
        encoder = MultitaskPromptEmbedding(config, model.get_input_embeddings(), tokenizer=Chars())
        table = model.get_input_embeddings().weight
        want = lucid.stack([table[ord("a") % 50], table[ord("b") % 50], table[ord("a") % 50]])
        assert bool(lucid.allclose(encoder.embedding.weight, want))


class TestTasks:
    def test_a_causal_model_takes_task_ids_in_forward_and_generate(self, device: str) -> None:
        lucid.manual_seed(0)
        peft = get_peft_model(_gpt2(device), _config("CAUSAL_LM"))
        _spread(peft)
        assert type(peft) is PeftModelForCausalLM
        ids, tasks = _ids(device, (3, 4)), _tasks(device, 0, 1, 2)
        assert peft(ids, task_ids=tasks).logits.shape == (3, 4 + _V, 50)
        generated = peft.generate(ids, max_new_tokens=4, task_ids=tasks)
        assert generated.shape == (3, 8)
        for row, task in enumerate((0, 1, 2)):
            alone = peft.generate(
                ids[row : row + 1], max_new_tokens=4, task_ids=_tasks(device, task)
            )
            assert bool((generated[row : row + 1] == alone).all().item())

    def test_an_encoder_decoder_reads_the_first_block_of_rows(self, device: str) -> None:
        lucid.manual_seed(0)
        model = create_model("t5_small_seq2seq", **_T5).eval().to(device)
        peft = get_peft_model(model, _config("SEQ_2_SEQ_LM"))
        _spread(peft)
        assert type(peft) is PeftModelForSeq2SeqLM
        assert peft.base_model.prompt_encoder["default"].num_embeddings == 2 * _V
        src, labels = _ids(device, (3, 5)), _ids(device, (3, 4), seed=3)
        out = peft(src, labels=labels, task_ids=_tasks(device, 0, 1, 2))
        assert out.encoder_last_hidden_state.shape[1] == 5 + _V
        one = peft(src[1:2], labels=labels[1:2], task_ids=_tasks(device, 1))
        assert bool(lucid.allclose(out.logits[1:2], one.logits, atol=1e-5))


# ── the reference library ────────────────────────────────────────────────────

_SIZES: dict[str, int] = {
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "vocab_size": 50,
    "max_position_embeddings": 64,
}


def _pair(ref: ModuleType, library: ModuleType, init: dict[str, Any]) -> tuple[Any, PeftModel]:
    """The reference's adapted BERT classifier and ours, with the same weights and the same ``init``."""
    transformers = pytest.importorskip("transformers")
    ref.manual_seed(0)
    config = transformers.BertConfig(
        **_SIZES, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, num_labels=3
    )
    reference = transformers.BertForSequenceClassification(config).eval()
    mine = create_model("bert_base_cls", num_labels=3, **_TEXT).eval()
    weights = {k: v.detach().numpy() for k, v in reference.state_dict().items()}
    mine.load_state_dict(
        {k: lucid.tensor(weights[k]) for k in mine.state_dict() if k in weights}, strict=False
    )
    shared: dict[str, Any] = {"task_type": "SEQ_CLS", "num_virtual_tokens": _V, **init}
    theirs = library.get_peft_model(
        reference, library.MultitaskPromptTuningConfig(num_tasks=_TASKS, num_ranks=_RANKS, **shared)
    ).eval()
    ours = get_peft_model(
        mine, MultitaskPromptTuningConfig(num_tasks=_TASKS, num_ranks=_RANKS, **shared)
    ).eval()
    return theirs, ours


def _bring_across(theirs: Any, ours: PeftModel) -> None:
    """Make the prompt the reference built the prompt ours has."""
    source = theirs.prompt_encoder["default"]
    with lucid.no_grad():
        target = ours.base_model.prompt_encoder["default"]
        target.embedding.weight.copy_(lucid.tensor(source.embedding.weight.detach().numpy()))
        target.prefix_task_cols.copy_(lucid.tensor(source.prefix_task_cols.detach().numpy()))
        target.prefix_task_rows.copy_(lucid.tensor(source.prefix_task_rows.detach().numpy()))


@pytest.mark.parity
class TestAgainstTheReference:
    def test_the_outputs_and_the_file_are_the_references(self, ref: ModuleType) -> None:
        library = adapter_library()
        theirs, ours = _pair(ref, library, {})
        _bring_across(theirs, ours)
        ids = np.random.default_rng(0).integers(2, 50, (_BATCH, _LENGTH))
        tasks = np.array(_MIXED)
        mask = np.ones((_BATCH, _LENGTH), dtype=np.int64)
        mask[0, -1] = 0
        with ref.no_grad():
            want = theirs(
                input_ids=ref.tensor(ids),
                attention_mask=ref.tensor(mask),
                task_ids=ref.tensor(tasks),
            ).logits.numpy()
        got = ours(
            lucid.tensor(ids).long(),
            attention_mask=lucid.tensor(mask).long(),
            task_ids=lucid.tensor(tasks).long(),
        ).logits.numpy()
        assert np.abs(got - want).max() < 1e-5
        theirs_file, ours_file = library.get_peft_model_state_dict(
            theirs
        ), get_peft_model_state_dict(ours)
        assert set(theirs_file) == set(ours_file)
        for key in ("prompt_embeddings", "prefix_task_cols", "prefix_task_rows"):
            assert np.abs(ours_file[key].numpy() - theirs_file[key].detach().numpy()).max() == 0.0

    @pytest.mark.parametrize(
        ("init", "fields"),
        [
            ("ONLY_SOURCE_SHARED", {}),
            ("EXACT_SOURCE_TASK", {"prompt_tuning_init_task": 1}),
            ("AVERAGE_SOURCE_TASKS", {}),
        ],
    )
    def test_a_source_start_gives_the_references_tensors(
        self, init: str, fields: dict[str, Any], ref: ModuleType, tmp_path: Path
    ) -> None:
        library = adapter_library()
        path, _ = _source("cpu", tmp_path)
        counts = {} if init == "ONLY_SOURCE_SHARED" else {"num_tasks": 1}
        shared = {
            "task_type": "SEQ_CLS",
            "num_virtual_tokens": _V,
            "num_ranks": _RANKS,
            "prompt_tuning_init": init,
            "prompt_tuning_init_state_dict_path": path,
            **fields,
        }
        transformers = pytest.importorskip("transformers")
        reference = transformers.BertForSequenceClassification(
            transformers.BertConfig(**_SIZES, num_labels=3)
        ).eval()
        theirs = library.get_peft_model(
            reference,
            library.MultitaskPromptTuningConfig(**{"num_tasks": _TASKS, **counts}, **shared),
        )
        ours = _peft(
            "cpu", MultitaskPromptTuningConfig(**{"num_tasks": _TASKS, **counts}, **shared)
        )
        reference_encoder = theirs.prompt_encoder["default"]
        encoder = ours.base_model.prompt_encoder["default"]
        for name in ("embedding.weight", "prefix_task_cols", "prefix_task_rows"):
            if init == "ONLY_SOURCE_SHARED" and name != "embedding.weight":
                continue  # drawn at random on each side
            got = encoder.embedding.weight if name == "embedding.weight" else getattr(encoder, name)
            want = (
                reference_encoder.embedding.weight
                if name == "embedding.weight"
                else getattr(reference_encoder, name)
            )
            assert np.abs(got.numpy() - want.detach().numpy()).max() < 1e-7, name
