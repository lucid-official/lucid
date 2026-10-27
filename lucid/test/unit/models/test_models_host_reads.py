"""LCD-411: model code reads a device tensor back to the host once, not one
element at a time.

On Metal every ``.item()`` / ``.tolist()`` is a sync.  A read per detection,
per sampling step or per drawn mask scalar stalls the device that many times.
The runtime checks count the reads on the paths that had them and require the
count not to grow with detections, steps or masks.  The static check keeps the
per-element pattern out of the whole zoo.
"""

import ast
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import lucid
from lucid.models import create_model
from lucid.models.generative.ddpm import DDPMConfig, DDPMForImageGeneration
from lucid.models.vision.ijepa import IJEPAConfig
from lucid.models.vision.ijepa._model import _sample_masks
from lucid.models.vision.vjepa import VJEPAConfig
from lucid.models.vision.vjepa._model import _sample_collection

type HostReads = Callable[[Callable[[], object]], list[str]]

_MODELS = Path(lucid.__file__).parent / "models"

_DETECTORS = {
    "mask_rcnn_resnet50_fpn": dict(
        roi_representation=64, mask_hidden_channels=16, mask_num_convs=1, mask_predictor_hidden=16
    ),
    "faster_rcnn_resnet50_fpn": dict(roi_representation_size=64),
}


def _detections_and_reads(host_reads: HostReads, name: str, proposals: int) -> tuple[int, int]:
    lucid.manual_seed(0)
    model = create_model(
        name,
        num_classes=3,
        backbone_layers=(1, 1, 1, 1),
        fpn_out_channels=32,
        rpn_post_nms_top_n=proposals,
        score_thresh=0.0,
        nms_thresh=1.0,
        **_DETECTORS[name],
    ).eval()
    output = model(lucid.randn((1, 3, 64, 64)))
    found: list[list[dict[str, lucid.Tensor]]] = []
    reads = host_reads(lambda: found.append(model.postprocess(output, image_sizes=[(64, 64)])))
    return int(found[0][0]["boxes"].shape[0]), len(reads)


@pytest.mark.parametrize("name", sorted(_DETECTORS))
def test_postprocess_reads_do_not_grow_with_detections(name: str, host_reads: HostReads) -> None:
    few = _detections_and_reads(host_reads, name, 5)
    many = _detections_and_reads(host_reads, name, 30)
    assert many[0] > few[0] > 0, (few, many)
    assert many[1] == few[1], f"{few[1]} reads for {few[0]} boxes, {many[1]} for {many[0]}"


def _generate_reads(host_reads: HostReads, steps: int) -> int:
    lucid.manual_seed(0)
    config = DDPMConfig(
        sample_size=8,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        resnet_groups=4,
        num_train_timesteps=20,
        attention_resolutions=(),
    )
    model = DDPMForImageGeneration(config).eval()
    scheduler = model.make_scheduler()
    return len(
        host_reads(lambda: model.generate(scheduler, n_samples=1, num_inference_steps=steps))
    )


def test_diffusion_generate_reads_do_not_grow_with_steps(host_reads: HostReads) -> None:
    """Schedulers keep their timesteps on device (LCD-383), so a read per
    step is a sync per step."""
    assert _generate_reads(host_reads, 2) == _generate_reads(host_reads, 6) <= 1


def _ijepa(blocks: int) -> IJEPAConfig:
    return IJEPAConfig(
        image_size=32,
        patch_size=8,
        dim=16,
        depth=1,
        num_heads=2,
        predictor_dim=8,
        predictor_depth=1,
        min_keep=1,
        num_target_blocks=blocks,
        target_scale=(0.1, 0.2),
        context_scale=(0.8, 1.0),
    )


def _vjepa() -> VJEPAConfig:
    return VJEPAConfig(
        image_size=32,
        patch_size=8,
        tubelet_size=2,
        num_frames=4,
        dim=24,
        depth=1,
        num_heads=2,
        predictor_dim=12,
        predictor_depth=1,
    )


@pytest.mark.parametrize("batch,blocks", [(1, 1), (6, 4)])
def test_ijepa_masks_read_once(host_reads: HostReads, batch: int, blocks: int) -> None:
    reads = host_reads(lambda: _sample_masks(_ijepa(blocks), batch, "cpu"))
    assert len(reads) == 1, reads


@pytest.mark.parametrize("batch,blocks", [(1, 1), (6, 8)])
def test_vjepa_masks_read_once(host_reads: HostReads, batch: int, blocks: int) -> None:
    config = _vjepa()
    reads = host_reads(
        lambda: _sample_collection(config, batch, blocks, config.short_range_scale, "cpu")
    )
    assert len(reads) == 1, reads


def test_masks_still_follow_the_seed() -> None:
    """The draws moved to a host stream seeded from Lucid's generator, so
    :func:`lucid.manual_seed` still fixes every mask."""
    config = _vjepa()
    drawn = []
    for _ in range(2):
        lucid.manual_seed(7)
        image = _sample_masks(_ijepa(4), 3, "cpu")
        video = _sample_collection(config, 3, 8, config.short_range_scale, "cpu")
        drawn.append([t.tolist() for t in (*image, *video)])
    assert drawn[0] == drawn[1]
    lucid.manual_seed(8)
    assert _sample_masks(_ijepa(4), 3, "cpu")[1].tolist() != drawn[0][1]


# ── static: no element-at-a-time reads anywhere in the zoo ─────────────────


def _bound_names(target: ast.expr) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _is_scalar_draw(call: ast.expr) -> bool:
    """``lucid.rand(...)`` and its siblings, called directly."""
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "lucid"
        and call.func.attr in {"rand", "randn", "randint", "randperm", "multinomial"}
    )


def _per_element_reads(source: str) -> Iterator[int]:
    """Lines that read a tensor back one element per loop iteration, or read
    back a freshly drawn random scalar.

    The first is ``t[i].item()`` (or ``.tolist()``) where ``i`` is bound by an
    enclosing ``for`` or comprehension; the second is ``lucid.rand(1).item()``.
    A whole draw read once — ``lucid.randperm(n).tolist()`` — is fine.
    """

    def visit(node: ast.AST, loop_names: frozenset[str]) -> Iterator[int]:
        if isinstance(node, (ast.For, ast.AsyncFor)):
            yield from visit(node.iter, loop_names)
            inner = loop_names | _bound_names(node.target)
            for child in (*node.body, *node.orelse):
                yield from visit(child, inner)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = loop_names.union(*(_bound_names(gen.target) for gen in node.generators))
            for child in ast.iter_child_nodes(node):
                yield from visit(child, inner)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"item", "tolist"}
        ):
            receiver = node.func.value
            indexed_by_loop = isinstance(receiver, ast.Subscript) and bool(
                _bound_names(receiver.slice) & loop_names
            )
            draw = node.func.attr == "item" and _is_scalar_draw(receiver)
            if indexed_by_loop or draw:
                yield node.lineno
        for child in ast.iter_child_nodes(node):
            yield from visit(child, loop_names)

    yield from visit(ast.parse(source), frozenset())


def test_no_model_reads_a_tensor_one_element_at_a_time() -> None:
    offenders = [
        f"{path.relative_to(_MODELS.parent)}:{line}"
        for path in sorted(_MODELS.rglob("*.py"))
        for line in _per_element_reads(path.read_text())
    ]
    assert not offenders, (
        "read the tensor once with lucid.models._utils._common.host_floats / "
        "host_ints / host_rows (or host_random for draws) instead:\n" + "\n".join(offenders)
    )


def test_the_static_check_can_fail() -> None:
    source = (
        "def f(t, n):\n"
        "    keep = [i for i in range(n) if float(t[i].item()) > 0]\n"
        "    for j in range(n):\n"
        "        x = int(t[j].item())\n"
        "    y = t[0].item()\n"
        "    z = float(lucid.rand(1).item())\n"
        "    w = [float(v) for v in t.tolist()]\n"
        "    p = lucid.randperm(n).tolist()\n"
    )
    assert list(_per_element_reads(source)) == [2, 4, 6]
