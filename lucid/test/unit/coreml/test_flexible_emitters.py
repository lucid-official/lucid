"""A flexible export never answers at the traced batch.

Every emitter that writes a shape into the program — a ``reshape``
target, a ``tile`` count, the size of a ``zeros`` — used to write the one
it was traced at. A package exported at batch 1 with a batch range then
loaded, accepted batch 3 and answered with one row, or aborted inside
Core ML (LCD-362). Each case here is exported over a batch range at batch
1 and run at batch 3: it either matches the eager model in shape and
values, or the export refuses it by name with ``ShapeNotFlexible``.
``exact`` records which, so a case that starts being refused — or
starts working — is noticed rather than absorbed.

Everything runs on ``CPU_ONLY``: with every compute unit, the mismatched
packages this guards against abort the process from inside the GPU
compiler, and take the rest of the run with them.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.coreml as cml
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

_BATCH = 3


class _Fn(nn.Module):
    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)


class _ClassToken(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Parameter(lucid.randn(1, 1, 8))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        spread = self.token.expand(x.shape[0], -1, -1)
        return lucid.cat([spread, x], dim=1).sum(dim=2)


class _Bilinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Bilinear(8, 8, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.layer(x, x * 2.0)


def _theta(x: lucid.Tensor) -> lucid.Tensor:
    return F.affine_grid(x.reshape(x.shape[0], 2, 3), (x.shape[0], 1, 4, 4))


_SEQ = (1, 5, 8)
_IMAGE = (1, 2, 4, 4)
_VOLUME = (1, 1, 2, 2, 2)
_REFUSED = False


def _case(
    name: str,
    fn: Callable[[lucid.Tensor], lucid.Tensor] | type[nn.Module],
    shape: tuple[int, ...] = _SEQ,
    exact: bool = True,
) -> object:
    """One row: the model, its example shape at batch 1, and the verdict."""

    def make() -> nn.Module:
        return fn() if isinstance(fn, type) else _Fn(fn)

    return pytest.param(make, shape, exact, id=name)


def _up(mode: str) -> Callable[[lucid.Tensor], lucid.Tensor]:
    return lambda x: F.interpolate(x, scale_factor=2.0, mode=mode)


_CASES = [
    _case("reshape", lambda x: x.reshape(x.shape[0], -1)),
    _case("reshape-merge", lambda x: x.reshape(-1, 8) * 2.0),
    _case("squeeze", lambda x: x.unsqueeze(1).squeeze(1) + 1.0),
    _case("unsqueeze", lambda x: x.unsqueeze(2) * 2.0),
    _case("broadcast_to", _ClassToken),
    _case("broadcast_to-batch-operand", lambda x: x[:, :1].expand(-1, 5, -1) + x),
    _case("zeros", lambda x: lucid.zeros(x.shape[0], 5, 8) + x),
    _case("ones", lambda x: lucid.ones(x.shape[0], 1, 8) * x),
    _case("full", lambda x: lucid.full((x.shape[0], 5, 1), 2.0) * x),
    # The tracer hands ``arange``'s consumers the tensor it made, as a
    # buffer: a constant at the traced length, so it is refused.
    _case(
        "arange",
        lambda x: x + lucid.arange(x.shape[0]).reshape(-1, 1, 1).to(lucid.float32),
        exact=_REFUSED,
    ),
    _case("masked_fill", lambda x: x.masked_fill(x > 0, -1.0)),
    _case("group_norm", lambda x: F.group_norm(x, 2), (1, 4, 3, 3)),
    _case("repeat", lambda x: x.repeat_interleave(2, dim=1)),
    _case("affine_grid", _theta, (1, 6)),
    _case("bilinear", _Bilinear, (1, 8)),
    _case("lp_normalize", lambda x: F.normalize(x, dim=2)),
    _case("lp_normalize-p3", lambda x: F.normalize(x, p=3.0, dim=2)),
    _case("var", lambda x: x.var(dim=2)),
    _case("var-over-batch", lambda x: x.var(dim=0), exact=_REFUSED),
    _case("log_softmax", lambda x: F.log_softmax(x, dim=2)),
    _case("rms_norm", lambda x: F.rms_norm(x, (8,))),
    _case("roll", lambda x: lucid.roll(x, [1], [1])),
    _case("unfold_dim", lambda x: x.unfold(1, 2, 1)),
    _case("tril", lambda x: lucid.tril(x), (1, 5, 5)),
    _case("erfinv", lambda x: lucid.erfinv(x.clip(-0.9, 0.9))),
    _case("unbind", lambda x: lucid.cat(list(lucid.unbind(x, 1)), 1), (1, 3, 4)),
    _case("split", lambda x: lucid.split(x, [2, 3], 1)[1] * 2.0),
    _case("topk", lambda x: lucid.topk(x, 3, 2)[0]),
    _case(
        "sdpa-causal",
        lambda x: F.scaled_dot_product_attention(x, x, x, is_causal=True),
        (1, 2, 5, 8),
    ),
    _case("interpolate_bilinear", _up("bilinear"), _IMAGE),
    _case("zero-d-output", lambda x: x.sum()),
    _case("inner", lambda x: x.inner(x), (1, 8), _REFUSED),
    _case("diagonal", lambda x: x.diagonal(0, 1, 2), (1, 4, 4), _REFUSED),
    _case("unfold", lambda x: F.unfold(x, 2), _IMAGE, _REFUSED),
    _case("fold", lambda x: F.fold(x, (3, 3), 2), (1, 8, 4), _REFUSED),
    _case("interpolate_nearest_3d", _up("nearest"), _VOLUME, _REFUSED),
    _case("interpolate_trilinear", _up("trilinear"), _VOLUME, _REFUSED),
]


def _agrees(got: lucid.Tensor, reference: lucid.Tensor) -> None:
    assert tuple(got.shape) == tuple(reference.shape)
    scale = max(float(reference.abs().max().item()), 1.0)
    assert float((got - reference).abs().max().item()) / scale < 1e-4


@pytest.mark.parametrize(("make", "shape", "exact"), _CASES)
def test_a_batch_range_is_exact_or_refused(
    make: Callable[[], nn.Module], shape: tuple[int, ...], exact: bool, tmp_path: object
) -> None:
    lucid.manual_seed(0)
    model = make().eval()
    try:
        exported = cml.export(
            model,
            lucid.randn(*shape),
            str(tmp_path / "flex.mlpackage"),
            shape_range={0: (1, 4)},
            compute_units=cml.ComputeUnits.CPU_ONLY,
        )
    except cml.ShapeNotFlexible as refusal:
        assert not exact, f"refused a case with an exact flexible form: {refusal}"
        assert refusal.op_name
        return
    try:
        assert exact, "exported a case this table says is refused — update it"
        for batch in (1, _BATCH):
            x = lucid.randn(batch, *shape[1:])
            _agrees(exported.predict(x), model(x))
    finally:
        exported.close()


class _Strided(nn.Module):
    """A stride-32 height and width that are 1 then 4 — as the batch is,
    when the batch and the resolution move together."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 32, stride=32)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.conv(x).unsqueeze(1) * 2.0


class _Grouped(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return F.group_norm(self.conv(x), 2).unsqueeze(1).squeeze(1)


class _Flattened(nn.Module):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x.reshape(x.shape[0], x.shape[1], -1).unsqueeze(2) + 1.0


_BATCH_AND_PLANE = {0: (1, 4), 2: (32, 128), 3: (32, 128)}
#: Inside every range below, on its corners and off them.
_PLANES = [
    (1, 3, 32, 32),
    (1, 3, 64, 64),
    (2, 3, 32, 32),
    (1, 3, 128, 128),
    (2, 3, 96, 32),
    (4, 3, 128, 64),
    (1, 3, 96, 32),
    (1, 3, 32, 96),
]


@pytest.mark.parametrize(
    ("make", "ranges"),
    [
        pytest.param(_Strided, _BATCH_AND_PLANE, id="stride-32-coincidence"),
        pytest.param(_Grouped, _BATCH_AND_PLANE, id="group_norm-batch-and-plane"),
        pytest.param(_Flattened, _BATCH_AND_PLANE, id="reshape-batch-and-plane"),
        pytest.param(_Grouped, {2: (32, 128), 3: (32, 128)}, id="square-plane"),
        pytest.param(_Grouped, {2: (32, 128)}, id="height-alone"),
        pytest.param(_Flattened, {3: (32, 128)}, id="width-alone"),
    ],
)
def test_several_ranged_axes_are_told_apart(
    make: Callable[[], nn.Module], ranges: dict[int, tuple[int, int]], tmp_path: object
) -> None:
    """Ranged axes that moved together at every probe made a stride-32
    height look like the batch, and the package answered ``(1, 1, 8, 1, 1)``
    for a 64x64 image. Every admitted shape here, corner or not, either
    matches eager or the export is refused."""
    lucid.manual_seed(0)
    model = make().eval()
    try:
        exported = cml.export(
            model,
            lucid.randn(1, 3, 32, 32),
            str(tmp_path / "plane.mlpackage"),
            shape_range=ranges,
            compute_units=cml.ComputeUnits.CPU_ONLY,
        )
    except cml.ShapeNotFlexible as refusal:
        pytest.fail(f"refused a model with an exact flexible form: {refusal}")
    default = (1, 3, 32, 32)
    admitted = [
        shape
        for shape in _PLANES
        if all(
            (
                ranges[axis][0] <= size <= ranges[axis][1]
                if axis in ranges
                else size == default[axis]
            )
            for axis, size in enumerate(shape)
        )
    ]
    assert len(admitted) >= 2
    try:
        for shape in admitted:
            x = lucid.randn(*shape)
            _agrees(exported.predict(x), model(x))
    finally:
        exported.close()


def test_a_range_moves_one_axis_at_a_time() -> None:
    """Two probes per ranged axis and the example: ``2k + 1`` traces."""
    from lucid.coreml._flex import plan_probes

    probes, ordered, bounds = plan_probes(
        (2, 3, 32, 32), 1, None, {0: (1, 4), 2: (16, 64), 3: (16, 64)}
    )
    assert probes == [
        (2, 3, 32, 32),
        (1, 3, 32, 32),
        (4, 3, 32, 32),
        (2, 3, 16, 32),
        (2, 3, 64, 32),
        (2, 3, 32, 16),
        (2, 3, 32, 64),
    ]
    assert ordered == []
    assert bounds == [(1, 4), (3, 3), (16, 64), (16, 64)]


def test_a_first_call_cache_is_named_in_the_refusal() -> None:
    """Swin fills a mask cache on its first call at each shape. When that
    is still the difference after a second call, saying "the operations
    differ" alone sent people looking at the wrong thing."""
    from lucid.coreml._flex import Record, compare

    def record(names: list[str], unsettled: bool) -> Record:
        return Record(
            [(n, {}, [(1,)]) for n in names],
            list(range(len(names))),
            -1,
            [(n, (i,)) for i, n in enumerate(names)],
            {},
            unsettled,
        )

    with pytest.raises(cml.ShapeNotFlexible, match="cache the model fills"):
        compare([record(["add"], False), record(["add", "mul"], True)], [(1,), (2,)], frozenset())


def test_enumerated_shapes_follow_the_batch_too(tmp_path: object) -> None:
    """The reproduction from LCD-362: ``(1, 6)`` for a batch of two."""
    model = _ClassToken().eval()
    exported = cml.export(
        model,
        lucid.randn(1, 5, 8),
        str(tmp_path / "token.mlpackage"),
        shapes=[(1, 5, 8), (2, 5, 8)],
        compute_units=cml.ComputeUnits.CPU_ONLY,
    )
    try:
        for batch in (1, 2):
            x = lucid.randn(batch, 5, 8)
            _agrees(exported.predict(x), model(x))
    finally:
        exported.close()


def test_an_integer_index_broadcast_keeps_its_type(tmp_path: object) -> None:
    """LCD-371: advanced indexing broadcasts an int64 index, and the
    ``full`` it is merged with used to go out as a float constant, so the
    ``tile`` after it was refused at parse time for changing type."""

    class Pick(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            rows = (x[:4] > 0).to(lucid.int64)
            cols = (x[4:8] > 0).to(lucid.int64)
            return x.reshape(4, 4)[rows, cols]

    model = Pick().eval()
    exported = cml.export(
        model,
        lucid.randn(16),
        str(tmp_path / "pick.mlpackage"),
        compute_units=cml.ComputeUnits.CPU_ONLY,
    )
    try:
        for _ in range(3):
            x = lucid.randn(16)
            _agrees(exported.predict(x), model(x))
    finally:
        exported.close()


def test_a_model_that_caches_on_its_first_call(tmp_path: object) -> None:
    """LCD-364: Swin fills its attention-mask cache on the first call, so
    the export's own trace was longer than every probe's, and the open
    axes found on the probes were applied to other values by id."""
    lucid.manual_seed(0)
    model = M.create_model(
        "swin_tiny_cls",
        image_size=56,
        embed_dim=16,
        depths=(2, 2),
        num_heads=(1, 2),
        num_classes=10,
    ).eval()
    exported = cml.export(
        model,
        lucid.randn(1, 3, 56, 56),
        str(tmp_path / "swin.mlpackage"),
        shape_range={0: (1, 4)},
        compute_units=cml.ComputeUnits.CPU_ONLY,
    )
    try:
        x = lucid.randn(_BATCH, 3, 56, 56)
        _agrees(exported.predict(x), model(x).logits)
    finally:
        exported.close()


#: Zoo families the scout found answering at the traced batch or failing
#: at prediction, at sizes small enough to export in seconds; ResNet is
#: the control that always worked.
_PICTURE = (1, 3, 64, 64)
_ZOO = [
    pytest.param("resnet_18_cls", {"layers": (1, 1, 1, 1)}, _PICTURE, id="resnet"),
    pytest.param(
        "vit_base_16_cls",
        {"image_size": 32, "dim": 64, "depth": 2, "num_heads": 2},
        (1, 3, 32, 32),
        id="vit",
    ),
    pytest.param(
        "cvt_13_cls",
        {"dims": (16, 32, 64), "depths": (1, 1, 1), "num_heads": (1, 1, 2)},
        _PICTURE,
        id="cvt",
    ),
    pytest.param(
        "convnext_tiny_cls",
        {"depths": (1, 1, 1, 1), "dims": (8, 16, 32, 64)},
        _PICTURE,
        id="convnext",
    ),
    pytest.param("sk_resnet_18_cls", {"layers": (1, 1, 1, 1)}, _PICTURE, id="sk"),
]


@pytest.mark.parametrize(("name", "overrides", "shape"), _ZOO)
def test_zoo_classifiers_take_a_batch_range(
    name: str, overrides: dict[str, object], shape: tuple[int, ...], tmp_path: object
) -> None:
    lucid.manual_seed(0)
    model = M.create_model(name, num_classes=10, **overrides).eval()
    with lucid.no_grad():
        # A zero-initialised head answers zero for every batch, which
        # would agree whatever the package did with the other rows.
        for parameter in model.parameters():
            if float(parameter.abs().max().item()) == 0.0:
                parameter.copy_(lucid.randn(*parameter.shape) * 0.02)
    exported = cml.export(
        model,
        lucid.randn(*shape),
        str(tmp_path / f"{name}.mlpackage"),
        shape_range={0: (1, 4)},
        compute_units=cml.ComputeUnits.CPU_ONLY,
    )
    try:
        x = lucid.randn(_BATCH, *shape[1:])
        _agrees(exported.predict(x), model(x).logits)
    finally:
        exported.close()


def test_a_language_model_takes_a_batch_range(tmp_path: object) -> None:
    lucid.manual_seed(0)
    model = M.create_model(
        "gpt_lm",
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
    ).eval()
    exported = cml.export(
        model,
        lucid.randint(0, 100, (1, 8)),
        str(tmp_path / "gpt.mlpackage"),
        shape_range={0: (1, 4)},
        compute_units=cml.ComputeUnits.CPU_ONLY,
    )
    try:
        tokens = lucid.randint(0, 100, (_BATCH, 8))
        got = exported.predict(tokens)
        logits = got["logits"] if isinstance(got, dict) else got
        _agrees(logits, model(tokens).logits)
    finally:
        exported.close()


@pytest.mark.parametrize("source", ["operand", "result"])
def test_the_guard_is_the_builder_not_the_emitters(
    source: str, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An emitter written tomorrow that bakes a size is refused by name.

    The cases above cover the emitters that exist. This one stands in for
    the next: it reads its operand's — or its own result's — shape and
    writes it into a constant, and the read itself is what refuses.
    """
    from lucid.coreml import _emit
    from lucid.coreml._build import Builder

    def baking(b: Builder, op: _emit.TracedOp, ins: list[str]) -> _emit.EmitResult:
        shape = b.shape_of(ins[0]) if source == "operand" else op.outputs[0].shape
        return "reshape", [("x", ins[0]), ("shape", b.const_ints(list(shape)))]

    monkeypatch.setitem(_emit.EMITTERS, "relu", baking)
    with pytest.raises(cml.ShapeNotFlexible) as refusal:
        cml.export(
            _Fn(F.relu).eval(),
            lucid.randn(1, 5, 8),
            str(tmp_path / "baked.mlpackage"),
            shape_range={0: (1, 4)},
            compute_units=cml.ComputeUnits.CPU_ONLY,
        )
    assert refusal.value.op_name == "relu"
