"""Regression tests for LCD-193, LCD-198, LCD-199, LCD-200, LCD-212, and LCD-215."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models._base import ModelConfig, PretrainedModel
from lucid.models.text.roformer import (
    RoFormerConfig,
    RoFormerForMaskedLM,
    RoFormerForMultipleChoice,
    RoFormerForQuestionAnswering,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerModel,
)
from lucid.models.vision.convnext import ConvNeXtConfig, ConvNeXtForImageClassification
from lucid.models.vision.crossvit import CrossViTConfig, CrossViTForImageClassification
from lucid.models.vision.cvt import CvTConfig, CvTForImageClassification
from lucid.models.vision.efficientformer import (
    EfficientFormerConfig,
    EfficientFormerForImageClassification,
)
from lucid.models.vision.inception_next import (
    InceptionNeXtConfig,
    InceptionNeXtForImageClassification,
)
from lucid.models.vision.maxvit import MaxViTConfig, MaxViTForImageClassification
from lucid.models.vision.pvt import PVTConfig, PVTForImageClassification
from lucid.models.vision.resnet import ResNetConfig, ResNetForImageClassification
from lucid.models.vision.swin import (
    SwinConfig,
    SwinTransformerForImageClassification,
)
from lucid.models.vision.vgg import VGGConfig, VGGForImageClassification
from lucid.utils.transforms import Image, Mask, MaskDropout, functional as F_tx

# ── LCD-193: No .numpy() in compute paths & AST gate ─────────────────────────


def test_lcd193_no_numpy_in_transforms_compute_paths() -> None:
    """Transforms functional.adjust_hue and MaskDropout run without .numpy()."""
    img = lucid.rand(3, 8, 8)
    out = F_tx.adjust_hue(img, 0.25)
    assert out.shape == (3, 8, 8)

    mask = Mask(lucid.ones((1, 8, 8), dtype=lucid.int32))
    dropout = MaskDropout(max_objects=1, p=1.0)
    out_sample = dropout({"image": Image(img), "mask": mask})
    assert isinstance(out_sample, dict)
    assert lucid.allclose(out_sample["mask"].data, lucid.zeros_like(mask.data))


def test_lcd193_check_numpy_h4_ast_gate(tmp_path: Path) -> None:
    """tools/check_numpy_h4.py flags .numpy() calls and passes on the repository."""
    import tools.check_numpy_h4 as h4

    bad_file = tmp_path / "bad.py"
    bad_file.write_text("x = tensor.numpy().tolist()\n", encoding="utf-8")
    hits = h4._scan_file(bad_file)
    assert len(hits) == 1
    assert ".numpy()" in hits[0].statement
    assert "compile/_optim/compiler.py" not in h4.SANCTIONED

    compiler_rel = Path("compile/_optim/compiler.py")
    compiler_file = tmp_path / "compiler.py"
    compiler_file.write_text(
        "class _CompiledOptimizer:\n"
        "    def state_dict(self):\n"
        "        return self.buf.numpy()\n"
        "    def _step_entry(self, i):\n"
        "        return self.step.numpy()\n"
        "    def _update(self, p):\n"
        "        return p.numpy()\n",
        encoding="utf-8",
    )
    scoped_hits = h4._scan_file(compiler_file, rel_path=compiler_rel)
    assert [v.lineno for v in scoped_hits] == [7]
    assert h4.scan(h4.LUCID_ROOT) == []


# ── LCD-198: ModelConfig.from_dict preserves tuple fields ────────────────────


@dataclass(frozen=True)
class _NestedTupleConfig(ModelConfig):
    model_type: ClassVar[str] = "_nested_tuple_test"
    depths: tuple[int, ...] = (2, 2, 6, 2)
    window_size: tuple[int, int] = (7, 7)
    nested: tuple[tuple[int, int], ...] = ((1, 2), (3, 4))
    optional_tuple: Optional[tuple[str, ...]] = None


class _DummyPretrainedModel(PretrainedModel):
    config_class = _NestedTupleConfig

    def __init__(self, config: _NestedTupleConfig) -> None:
        super().__init__(config)
        self.proj = nn.Linear(4, 2)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.proj(x)


def test_lcd198_model_config_json_roundtrip_preserves_tuples(tmp_path: Path) -> None:
    """ModelConfig.from_dict and save_pretrained/from_pretrained preserve tuple fields."""
    cfg = _NestedTupleConfig(
        depths=(3, 4, 6, 3),
        window_size=(8, 8),
        nested=((2, 4), (6, 8)),
        optional_tuple=("stage1", "stage2"),
    )
    raw = cfg.to_dict()
    # Simulate JSON serialization turning tuples into lists
    raw["depths"] = [3, 4, 6, 3]
    raw["window_size"] = [8, 8]
    raw["nested"] = [[2, 4], [6, 8]]
    raw["optional_tuple"] = ["stage1", "stage2"]

    restored = _NestedTupleConfig.from_dict(raw)
    assert isinstance(restored.depths, tuple) and restored.depths == (3, 4, 6, 3)
    assert isinstance(restored.window_size, tuple) and restored.window_size == (8, 8)
    assert isinstance(restored.nested, tuple) and all(isinstance(x, tuple) for x in restored.nested)
    assert restored.nested == ((2, 4), (6, 8))
    assert isinstance(restored.optional_tuple, tuple) and restored.optional_tuple == (
        "stage1",
        "stage2",
    )

    # Real vision config round-trip through JSON file
    swin_cfg = SwinConfig(depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16))
    json_path = str(tmp_path / "swin_config.json")
    swin_cfg.save(json_path)
    swin_loaded = SwinConfig.load(json_path)
    assert isinstance(swin_loaded.depths, tuple)
    assert isinstance(swin_loaded.num_heads, tuple)

    # PretrainedModel save_pretrained / from_pretrained round-trip
    model = _DummyPretrainedModel(cfg)
    save_dir = str(tmp_path / "ckpt")
    model.save_pretrained(save_dir)
    loaded_model = _DummyPretrainedModel.from_pretrained(save_dir)
    assert isinstance(loaded_model.config.depths, tuple)
    assert isinstance(loaded_model.config.nested, tuple)
    assert isinstance(loaded_model.config.nested[0], tuple)


# ── LCD-199: Vision *ForImageClassification weight initialization ────────────


@pytest.mark.parametrize(
    "cls_fn",
    [
        lambda: ConvNeXtForImageClassification(
            ConvNeXtConfig(depths=(1, 1, 1, 1), dims=(16, 32, 64, 128), num_classes=5)
        ),
        lambda: SwinTransformerForImageClassification(
            SwinConfig(
                image_size=32,
                embed_dim=16,
                depths=(2, 2, 2, 2),
                num_heads=(2, 4, 4, 8),
                window_size=4,
                num_classes=5,
            )
        ),
        lambda: PVTForImageClassification(
            PVTConfig(
                embed_dims=(16, 32, 64, 128),
                num_heads=(1, 2, 4, 8),
                mlp_ratios=(2.0, 2.0, 2.0, 2.0),
                depths=(1, 1, 1, 1),
                sr_ratios=(8, 4, 2, 1),
                num_classes=5,
            )
        ),
        lambda: CvTForImageClassification(
            CvTConfig(
                dims=(16, 32, 64),
                num_heads=(1, 2, 4),
                depths=(1, 1, 1),
                num_classes=5,
            )
        ),
        lambda: CrossViTForImageClassification(
            CrossViTConfig(
                image_size=240,
                patch_sizes=(12, 16),
                embed_dims=(16, 32),
                depths=((1, 1, 0),),
                num_heads=(2, 2),
                mlp_ratio=(2.0, 2.0, 1.0),
                num_classes=5,
            )
        ),
        lambda: MaxViTForImageClassification(
            MaxViTConfig(
                depths=(1, 1, 1, 1),
                dims=(16, 32, 64, 128),
                window_size=7,
                head_dim=16,
                num_classes=5,
            )
        ),
        lambda: EfficientFormerForImageClassification(
            EfficientFormerConfig(
                depths=(1, 1, 1, 1),
                embed_dims=(16, 32, 64, 128),
                num_vit=1,
                num_classes=5,
            )
        ),
        lambda: InceptionNeXtForImageClassification(
            InceptionNeXtConfig(
                depths=(1, 1, 1, 1),
                dims=(16, 32, 64, 128),
                num_classes=5,
            )
        ),
    ],
)
def test_lcd199_vision_transformer_for_image_classification_inits_weights(
    cls_fn,
) -> None:
    """All 8 transformer/hybrid *ForImageClassification wrappers run init_transformer_trunc_normal."""
    model = cls_fn()
    linear_biases = [
        m.bias for m in model.modules() if isinstance(m, nn.Linear) and m.bias is not None
    ]
    assert len(linear_biases) >= 1
    for b in linear_biases:
        assert lucid.allclose(b, lucid.zeros_like(b))


def test_lcd199_vgg_and_resnet_for_image_classification_cnn_init() -> None:
    """VGGForImageClassification and ResNetForImageClassification run init_cnn_fan_out."""
    vgg = VGGForImageClassification(VGGConfig(arch=(1, 1, 1, 1, 1), num_classes=5))
    conv_biases = [m.bias for m in vgg.modules() if isinstance(m, nn.Conv2d) and m.bias is not None]
    assert len(conv_biases) == 5
    for b in conv_biases:
        assert lucid.allclose(b, lucid.zeros_like(b))

    resnet = ResNetForImageClassification(
        ResNetConfig(
            block_type="basic",
            layers=(1, 1, 1, 1),
            stem_channels=64,
            hidden_sizes=(64, 128, 256, 512),
            num_classes=5,
            zero_init_residual=True,
        )
    )
    first_conv = next(m for m in resnet.modules() if isinstance(m, nn.Conv2d))
    conv1_std = float(lucid.std(first_conv.weight).item())
    assert 0.018 < conv1_std < 0.033
    block = resnet.layer1[0]
    assert lucid.allclose(block.bn2.weight, lucid.zeros_like(block.bn2.weight))


# ── LCD-200: RoFormer weight initialization ──────────────────────────────────


@pytest.mark.parametrize(
    "model_cls",
    [
        RoFormerModel,
        RoFormerForMaskedLM,
        RoFormerForSequenceClassification,
        RoFormerForTokenClassification,
        RoFormerForMultipleChoice,
        RoFormerForQuestionAnswering,
    ],
)
def test_lcd200_roformer_weight_init(model_cls) -> None:
    """RoFormer models implement _init_weights and initialize Linear/Embedding/LayerNorm."""
    cfg = RoFormerConfig(
        vocab_size=128,
        embedding_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        pad_token_id=0,
        initializer_range=0.02,
    )
    model = model_cls(cfg)
    assert hasattr(model, "_init_weights")

    for m in model.modules():
        if isinstance(m, nn.Linear):
            std = float(lucid.std(m.weight).item())
            assert 0.005 < std < 0.05
            if m.bias is not None:
                assert lucid.allclose(m.bias, lucid.zeros_like(m.bias))
        elif isinstance(m, nn.Embedding) and m.weight.shape[0] == cfg.vocab_size:
            pad_row = m.weight[cfg.pad_token_id]
            assert lucid.allclose(pad_row, lucid.zeros_like(pad_row))
        elif isinstance(m, nn.LayerNorm):
            assert lucid.allclose(m.weight, lucid.ones_like(m.weight))
            assert lucid.allclose(m.bias, lucid.zeros_like(m.bias))


# ── LCD-212: SparseAdam state_dict / load_state_dict ─────────────────────────


def test_lcd212_sparse_adam_state_dict_roundtrip() -> None:
    """SparseAdam exports and restores step, exp_avg, and exp_avg_sq via state_dict."""
    p1 = nn.Parameter(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
    opt1 = optim.SparseAdam([p1], lr=1e-2, betas=(0.9, 0.99))
    p1.grad = lucid.tensor([[0.5, -0.2], [0.0, 0.0]])
    opt1.step()

    sd = opt1.state_dict()
    assert 0 in sd["state"]
    st = sd["state"][0]
    assert st["step"] == 1
    assert "exp_avg" in st and "exp_avg_sq" in st

    # Restore into a fresh SparseAdam and verify next step is identical
    p2 = nn.Parameter(p1.detach().clone())
    opt2 = optim.SparseAdam([p2], lr=1e-2, betas=(0.9, 0.99))
    opt2.load_state_dict(sd)

    g_next = lucid.tensor([[0.1, 0.3], [-0.4, 0.2]])
    p1.grad = g_next.clone()
    p2.grad = g_next.clone()
    opt1.step()
    opt2.step()
    assert lucid.allclose(p1, p2)


# ── LCD-215: Optimizer param_groups keys & NAdam options ─────────────────────


@pytest.mark.parametrize(
    "opt_cls",
    [
        optim.Adam,
        optim.AdamW,
        optim.Adamax,
        optim.NAdam,
        optim.RAdam,
        optim.SparseAdam,
    ],
)
def test_lcd215_adam_family_betas_in_param_groups_and_mutable(opt_cls) -> None:
    """Adam-family optimizers store 'betas' in param_groups and honor mid-run edits."""
    p = nn.Parameter(lucid.tensor([1.0, -2.0, 3.0]))
    opt = opt_cls([p], lr=1e-2, betas=(0.9, 0.999))
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)

    # Mutate group["betas"] before step and compare against fresh optimizer with new betas
    opt.param_groups[0]["betas"] = (0.5, 0.8)
    p_ref = nn.Parameter(lucid.tensor([1.0, -2.0, 3.0]))
    opt_ref = opt_cls([p_ref], lr=1e-2, betas=(0.5, 0.8))

    for _ in range(2):
        g = lucid.tensor([0.3, -0.4, 0.5])
        p.grad = g.clone()
        p_ref.grad = g.clone()
        opt.step()
        opt_ref.step()

    assert lucid.allclose(p, p_ref)


def test_lcd215_rprop_etas_and_step_sizes_in_param_groups_and_mutable() -> None:
    """Rprop stores 'etas' and 'step_sizes' in param_groups and honors mid-run edits."""
    p = nn.Parameter(lucid.tensor([1.0, -2.0]))
    opt = optim.Rprop([p], lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0))
    assert opt.param_groups[0]["etas"] == (0.5, 1.2)
    assert opt.param_groups[0]["step_sizes"] == (1e-6, 50.0)

    opt.param_groups[0]["etas"] = (0.3, 1.5)
    opt.param_groups[0]["step_sizes"] = (1e-3, 0.005)

    p_ref = nn.Parameter(lucid.tensor([1.0, -2.0]))
    opt_ref = optim.Rprop([p_ref], lr=1e-2, etas=(0.3, 1.5), step_sizes=(1e-3, 0.005))

    for g_val in ([0.4, -0.2], [0.4, -0.2]):
        p.grad = lucid.tensor(g_val)
        p_ref.grad = lucid.tensor(g_val)
        opt.step()
        opt_ref.step()

    assert lucid.allclose(p, p_ref)


def test_lcd215_nadam_momentum_decay_and_decoupled_weight_decay() -> None:
    """NAdam supports momentum_decay and decoupled_weight_decay in __init__ and step."""
    p_coupled = nn.Parameter(lucid.tensor([1.5, -2.5]))
    p_decoupled = nn.Parameter(lucid.tensor([1.5, -2.5]))
    p_fast_md = nn.Parameter(lucid.tensor([1.5, -2.5]))

    opt_coupled = optim.NAdam(
        [p_coupled],
        lr=1e-2,
        weight_decay=0.1,
        momentum_decay=4e-3,
        decoupled_weight_decay=False,
    )
    opt_decoupled = optim.NAdam(
        [p_decoupled],
        lr=1e-2,
        weight_decay=0.1,
        momentum_decay=4e-3,
        decoupled_weight_decay=True,
    )
    opt_fast_md = optim.NAdam(
        [p_fast_md],
        lr=1e-2,
        weight_decay=0.1,
        momentum_decay=0.2,
        decoupled_weight_decay=False,
    )

    assert opt_decoupled.param_groups[0]["momentum_decay"] == 4e-3
    assert opt_decoupled.param_groups[0]["decoupled_weight_decay"] is True

    for _ in range(3):
        g = lucid.tensor([0.4, -0.6])
        p_coupled.grad = g.clone()
        p_decoupled.grad = g.clone()
        p_fast_md.grad = g.clone()
        opt_coupled.step()
        opt_decoupled.step()
        opt_fast_md.step()

    assert not lucid.allclose(p_coupled, p_decoupled)
    assert not lucid.allclose(p_coupled, p_fast_md)
