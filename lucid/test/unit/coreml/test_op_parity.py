"""Per-op agreement between an exported package and the eager model.

An end-to-end model test proves the ops that model happens to use. It
says nothing about the rest of the table, and a wrong emitter in an
unused corner waits there until someone exports a model that reaches it.
These run one op at a time.

Every case compares values, never shapes: an emitter bound to the wrong
MIL op — ``negative_slope`` read where Lucid writes ``slope``, ``axis``
where it writes ``dim`` — produces the right shape and the wrong numbers.
"""

import types
from typing import Any

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class _Apply(nn.Module):
    """One traced callable, so an op can be exported on its own."""

    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


def _check(fn: object, x: lucid.Tensor, tmp_path: object, tol: float = 1e-5) -> None:
    model = _Apply(fn).eval()
    reference = model(x)
    exported = cml.export(model, x, f"{tmp_path}/op.mlpackage")
    try:
        got = exported.predict(x)
        assert got.shape == reference.shape
        scale = float(reference.abs().max().item()) or 1.0
        assert float((got - reference).abs().max().item()) / scale < tol
    finally:
        exported.close()


# ── the tables ───────────────────────────────────────────────────────────────
#
# ``positive`` marks the ops whose domain excludes the negatives that
# ``randn`` would otherwise hand them.

_UNARY = [
    ("abs", lucid.abs, False),
    ("arccos", lambda x: lucid.arccos(lucid.tanh(x)), False),
    ("arcsin", lambda x: lucid.arcsin(lucid.tanh(x)), False),
    ("arctan", lucid.arctan, False),
    ("ceil", lucid.ceil, False),
    ("cos", lucid.cos, False),
    ("cosh", lucid.cosh, False),
    ("erf", lucid.erf, False),
    ("exp", lucid.exp, False),
    ("floor", lucid.floor, False),
    ("trunc", lucid.trunc, False),
    ("log", lucid.log, True),
    ("log2", lucid.log2, True),
    ("neg", lambda x: -x, False),
    ("reciprocal", lucid.reciprocal, True),
    ("round", lucid.round, False),
    ("rsqrt", lucid.rsqrt, True),
    ("sign", lucid.sign, False),
    ("sin", lucid.sin, False),
    ("sinh", lucid.sinh, False),
    ("sqrt", lucid.sqrt, True),
    ("square", lucid.square, False),
    ("tan", lucid.tan, False),
]

_ACTIVATION = [
    ("elu", F.elu),
    ("mish", F.mish),
    ("selu", F.selu),
    ("softplus", F.softplus),
    ("log_softmax", lambda x: F.log_softmax(x, dim=1)),
]

_ACTIVATION += [("hard_sigmoid", F.hardsigmoid), ("hard_swish", F.hardswish)]

_SPATIAL = [
    ("conv1d", nn.Conv1d(4, 2, 3).eval(), (1, 4, 10)),
    ("conv2d", nn.Conv2d(4, 2, 3).eval(), (1, 4, 6, 6)),
    ("conv3d", nn.Conv3d(4, 2, 3).eval(), (1, 4, 6, 6, 6)),
    # A sliding window along one axis.  MIL has the operation and puts
    # the window axis directly after the one it slid along, where Lucid
    # appends it last — so the emitter is that op plus a transpose, and
    # a wrong reconciliation shows up here as values rather than shapes.
    ("unfold_dim", lambda x: x.unfold(2, 3, 2), (1, 2, 8, 4)),
    # ``x ** c`` and ``c ** x`` with the constant on the trace.  Reached
    # through LP pooling and the p-norms.
    ("pow_scalar", lambda x: F.lp_pool2d(x.abs() + 0.5, 2.0, 2), (1, 2, 8, 8)),
    # MIL's ``conv_transpose`` takes the weight as (C_in, C_out / groups,
    # *K) — the layout Lucid already stores — so grouping needs no
    # relayout on this path.  The cases are here because nothing else
    # would notice if that stopped being true.
    ("conv_transpose1d", nn.ConvTranspose1d(4, 2, 3, stride=2).eval(), (1, 4, 10)),
    ("conv_transpose2d", nn.ConvTranspose2d(4, 2, 3, stride=2).eval(), (1, 4, 6, 6)),
    ("conv_transpose3d", nn.ConvTranspose3d(4, 2, 3, stride=2).eval(), (1, 4, 6, 6, 6)),
    (
        "conv_transpose1d_grouped",
        nn.ConvTranspose1d(4, 4, 3, stride=2, groups=2).eval(),
        (1, 4, 10),
    ),
    (
        "conv_transpose2d_grouped",
        nn.ConvTranspose2d(4, 4, 3, stride=2, groups=2).eval(),
        (1, 4, 6, 6),
    ),
    (
        "conv_transpose2d_depthwise",
        nn.ConvTranspose2d(4, 4, 3, stride=2, groups=4).eval(),
        (1, 4, 6, 6),
    ),
    (
        "conv_transpose3d_grouped",
        nn.ConvTranspose3d(4, 4, 3, stride=2, groups=2).eval(),
        (1, 4, 6, 6, 6),
    ),
    (
        "conv_transpose1d_dilated",
        nn.ConvTranspose1d(4, 2, 3, stride=2, dilation=2).eval(),
        (1, 4, 10),
    ),
    (
        "conv_transpose2d_dilated",
        nn.ConvTranspose2d(4, 2, 3, stride=2, dilation=2).eval(),
        (1, 4, 6, 6),
    ),
    ("max_pool1d", lambda x: F.max_pool1d(x, 2), (1, 4, 10)),
    ("max_pool2d", lambda x: F.max_pool2d(x, 2), (1, 4, 6, 6)),
    ("max_pool3d", lambda x: F.max_pool3d(x, 2), (1, 4, 6, 6, 6)),
    ("avg_pool1d", lambda x: F.avg_pool1d(x, 2), (1, 4, 10)),
    ("avg_pool2d", lambda x: F.avg_pool2d(x, 2), (1, 4, 6, 6)),
    ("avg_pool3d", lambda x: F.avg_pool3d(x, 2), (1, 4, 6, 6, 6)),
    ("group_norm", lambda x: F.group_norm(x, 2), (1, 4, 6, 6)),
    ("rms_norm", lambda x: F.rms_norm(x, (6,)), (1, 4, 6, 6)),
]

_OTHER = [
    ("clip", lambda x: lucid.clip(x, -0.5, 0.5), False),
    ("diagonal", lucid.diagonal, False),
    ("norm", lambda x: lucid.linalg.norm(x), False),
    ("repeat", lambda x: lucid.repeat(x, 2, dim=1), False),
    ("split", lambda x: lucid.split(x, 2, dim=1)[0], False),
    ("flip", lambda x: lucid.flip(x, dims=1), False),
    ("masked_fill", lambda x: lucid.masked_fill(x, x > 0, 0.0), False),
    ("maximum", lambda x: lucid.maximum(x, x * 2), False),
    ("minimum", lambda x: lucid.minimum(x, x * 2), False),
    ("pad", lambda x: F.pad(x, (1, 1, 1, 1)), False),
    ("pow", lambda x: lucid.pow(x, 2.0), True),
    ("prod", lambda x: lucid.prod(x, dim=1), True),
    ("sum", lambda x: lucid.sum(x, dim=1), False),
    ("tile", lambda x: lucid.tile(x, (1, 1, 2, 2)), False),
    # Added when exporting families the smoke test did not reach: VQ-VAE
    # needs the first three to look its codebook up, DETR needs cumsum
    # for its positional encoding.
    ("equal", lambda x: lucid.eq(x, x * 0.0).to(lucid.float32), False),
    ("argmax", lambda x: lucid.argmax(x, dim=1).to(lucid.float32), False),
    ("argmin", lambda x: lucid.argmin(x, dim=1).to(lucid.float32), False),
    ("argmax_flat", lambda x: lucid.argmax(x).to(lucid.float32), False),
    ("argmax_keepdim", lambda x: lucid.argmax(x, dim=1, keepdim=True).to(lucid.float32), False),
    (
        "one_hot",
        lambda x: F.one_hot(lucid.argmax(x, dim=1), num_classes=4).to(lucid.float32),
        False,
    ),
    ("cumsum", lambda x: lucid.cumsum(x, dim=1), False),
    ("cumsum_last", lambda x: lucid.cumsum(x, dim=-1), False),
]


@pytest.mark.parametrize(("name", "fn", "positive"), _UNARY, ids=[c[0] for c in _UNARY])
def test_a_unary_op_matches(name: str, fn: object, positive: bool, tmp_path: object) -> None:
    x = lucid.rand(1, 4, 6, 6) + 0.5 if positive else lucid.randn(1, 4, 6, 6)
    _check(fn, x, tmp_path)


@pytest.mark.parametrize(("name", "fn"), _ACTIVATION, ids=[c[0] for c in _ACTIVATION])
def test_an_activation_matches(name: str, fn: object, tmp_path: object) -> None:
    _check(fn, lucid.randn(1, 4, 6, 6), tmp_path)


@pytest.mark.parametrize(("name", "fn", "positive"), _OTHER, ids=[c[0] for c in _OTHER])
def test_a_structural_op_matches(name: str, fn: object, positive: bool, tmp_path: object) -> None:
    x = lucid.rand(1, 4, 6, 6) + 0.5 if positive else lucid.randn(1, 4, 6, 6)
    _check(fn, x, tmp_path)


class TestCoreMLDoesNotPromote:
    """Lucid promotes mixed operands; Core ML refuses them.

    And it refuses at *parse* time, naming an internal value and saying
    only that ``x``, ``y`` and the output must share a type — which is
    several steps from the operation that mixed them. CLIP builds its
    position indices this way and could not be exported at all.

    The cast goes to the result's declared type, not to a promotion
    guessed from the operands: ``int64 * 3`` declares an integer result,
    and promoting its operands to float instead produces float operands
    feeding an int output — the same refusal, one step later.
    """

    @pytest.mark.parametrize(
        ("name", "fn"),
        [
            (
                "int-times-scalar",
                lambda x: (lucid.arange(x.shape[-1]).to(lucid.int64) * 3).to(lucid.float32) + x,
            ),
            ("float-times-int-tensor", lambda x: x * lucid.arange(x.shape[-1]).to(lucid.float32)),
            (
                "int-compared-to-scalar",
                lambda x: (lucid.arange(x.shape[-1]).to(lucid.int64) > 1).to(lucid.float32) + x,
            ),
            (
                "float-compared-to-float",
                lambda x: (x > lucid.arange(x.shape[-1]).to(lucid.float32)).to(lucid.float32),
            ),
            (
                # ``arange(n)`` is int64 and reaches the package as an
                # integer buffer, which was recorded as float: the
                # computed side was cast to float to "match" it.
                "int-buffer-compared-to-computed-int",
                lambda x: x * (lucid.arange(x.shape[0]) == x.argmax(-1)).to(lucid.float32)[:, None],
            ),
        ],
        ids=["int-scalar", "float-int", "int-compare", "float-compare", "int-buffer-compare"],
    )
    def test_a_mixed_operand_pair_is_reconciled(
        self, name: str, fn: object, tmp_path: object
    ) -> None:
        lucid.manual_seed(0)
        _check(fn, lucid.randn(2, 6), tmp_path)


class TestRankSixIsStagedRatherThanRefused:
    """Upsampling by stuffing zeros wants rank 6 for two axes at once.

    Splitting each axis into ``(d, 1)``, padding the new axis to
    ``(d, 2)`` and merging back is how a diffusion decoder doubles its
    resolution before a convolution. Doing both spatial axes together
    needs six axes, and Core ML programs stop at five.

    One axis at a time leaves the rank where it started, so the same
    arithmetic goes out in three operations per axis instead of being
    refused. The check is against eager, since a wrong staging would
    still produce a tensor of the right shape.
    """

    def test_a_zero_stuffing_upsample_exports(self, tmp_path: object) -> None:
        class _Upsample(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                n, c, h, w = (int(d) for d in x.shape)
                wide = x.reshape(n, c, h, 1, w, 1)
                wide = F.pad(wide, (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
                return wide.reshape(n, c, h * 2, w * 2)

        lucid.manual_seed(0)
        model = _Upsample().eval()
        x = lucid.randn(1, 3, 4, 4)
        reference = model(x)

        exported = cml.export(model, x, f"{tmp_path}/stuff.mlpackage")
        try:
            got = exported.predict(x)
            assert tuple(got.shape) == tuple(reference.shape)
            scale = max(float(reference.abs().max().item()), 1e-9)
            assert float((got - reference).abs().max().item()) / scale < 1e-6
        finally:
            exported.close()

    def test_a_rank_six_tensor_nothing_stages_is_still_refused(self, tmp_path: object) -> None:
        """The cap is a format limit, and saying so beats a parse error.

        A rank-6 value that is read for real — not consumed by a pattern
        that can be staged — has no Core ML spelling. Refusing it by name
        is what keeps the failure near the operation that caused it.
        """

        class _Tall(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x.reshape(1, 1, 2, 1, 3, 4) * 2.0

        with pytest.raises(Exception, match="rank"):
            cml.export(_Tall().eval(), lucid.randn(2, 12), f"{tmp_path}/tall.mlpackage")


class TestWorkThatDoesNotDependOnTheInput:
    """Computed once, at export, instead of on every prediction.

    A windowed transformer spends a third of its graph on arithmetic
    that has nothing to do with its input: Swin builds the index into
    its relative-position table out of ``arange`` and rebuilds it every
    time. Those operations land on the CPU, between operations that land
    on the Neural Engine, so they cost more than themselves.

    Written down instead: ``swin_tiny`` goes from 1249 operations at 50%
    on the accelerator to 543 at 95%, agreeing with eager to 5e-07.

    What must not happen is folding something that *does* depend on the
    input — a package that answers every input with the first one's
    result. The region is grown from the weights and from operations
    that make a tensor out of nothing, never from an operation whose
    inputs the tracer failed to record; ``test_no_frozen_answers.py`` is
    the standing check.
    """

    def test_index_arithmetic_leaves_the_graph(self, tmp_path: object) -> None:
        class _Indexed(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.table = nn.Parameter(lucid.randn(16, 4))

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                rows = lucid.arange(4).to(lucid.int64)
                columns = lucid.arange(4).to(lucid.int64)
                first, last = lucid.meshgrid(rows, columns)
                flat = (first * 4 + last).reshape(16, 1)
                index = flat + lucid.zeros(16, 4).to(lucid.int64)
                bias = lucid.gather(self.table, 0, index)
                return x + bias

        lucid.manual_seed(0)
        model = _Indexed().eval()
        x = lucid.randn(16, 4)

        exported = cml.export(model, x, f"{tmp_path}/folded.mlpackage")
        try:
            plan = exported.compute_plan()
            # One add is all that is left to compute; everything that
            # built the index was decided here.
            assert plan.total_compute <= 2, plan.compute
            assert exported.verify(model, x, relative=True) < 1e-5
        finally:
            exported.close()

    def test_the_input_side_is_never_folded(self, tmp_path: object) -> None:
        """The same shape of graph, one step from the input.

        Arithmetic that reads the input has to stay, however constant
        the rest of the expression looks.
        """

        class _Reads(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                scale = lucid.arange(int(x.shape[-1])).to(lucid.float32)
                return (x * scale).relu() + scale

        lucid.manual_seed(0)
        model = _Reads().eval()
        first = lucid.randn(3, 5)
        second = lucid.randn(3, 5) * 9 - 4

        exported = cml.export(model, first, f"{tmp_path}/reads.mlpackage")
        try:
            moved = float((exported.predict(first) - exported.predict(second)).abs().max().item())
            assert moved > 1e-6
            assert exported.verify(model, second, relative=True) < 1e-5
        finally:
            exported.close()


class TestContiguousIsNotAnOperation:
    """A MIL value has no layout, so making one contiguous is nothing.

    Lucid's ``contiguous`` copies a strided tensor into packed storage.
    A MIL program has no strides to pack against, so emitting
    ``identity`` for it asks Core ML to schedule a copy that means
    nothing — and Swin calls it after every permute, which put 109 such
    operations in the program, all of them on the CPU. An operation on
    the CPU between two on the Neural Engine costs a round trip whatever
    the operation is: dropping them took Swin from 37% of its
    computation on the accelerator to 50%.
    """

    def test_it_adds_no_operations(self, tmp_path: object) -> None:
        class _Packed(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                y = x.permute(0, 2, 1).contiguous()
                return y.permute(0, 2, 1).contiguous() + x

        class _Plain(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                y = x.permute(0, 2, 1)
                return y.permute(0, 2, 1) + x

        lucid.manual_seed(0)
        x = lucid.randn(1, 4, 6)
        counts = []
        for name, model in (("packed", _Packed()), ("plain", _Plain())):
            exported = cml.export(model.eval(), x, f"{tmp_path}/{name}.mlpackage")
            try:
                counts.append(len(exported.compute_plan().placements))
                assert exported.verify(model.eval(), x, relative=True) < 1e-5
            finally:
                exported.close()
        assert counts[0] == counts[1]


class TestHalfPrecisionKeepsItsIntegers:
    """A float16 body must not drag the integer values with it.

    Two places assumed every value was a float. The driver brackets a
    float16 program with casts so callers still hand over and receive
    float32 — but it cast *every* output, including an integer one, and
    Core ML refuses a cast whose named type is not its output's. And
    ``astype`` named the type the model asked for rather than the type
    the program declares, which differ in a float16 body for a cast to
    float32. VQ-VAE has both: it casts around its codebook and returns
    the codebook indices.

    Neither failed on the float32 path, which is the one the tests ran.
    """

    def test_an_integer_output_survives_a_float16_export(self, tmp_path: object) -> None:
        class _Indices(nn.Module):
            def forward(self, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
                return x * 2.0, lucid.argmax(x, dim=1)

        lucid.manual_seed(0)
        model = _Indices().eval()
        x = lucid.randn(2, 6)
        exported = cml.export(
            model, x, f"{tmp_path}/ints.mlpackage", precision=cml.Precision.FLOAT16
        )
        try:
            got = exported.predict(x)
            values = got if isinstance(got, dict) else {exported.output_names[0]: got}
            indices = values[exported.output_names[1]]
            assert "int" in str(indices.dtype)
            want = lucid.argmax(x, dim=1)
            assert indices.to(lucid.int64).tolist() == want.tolist()
        finally:
            exported.close()

    def test_a_cast_to_float32_inside_a_float16_body(self, tmp_path: object) -> None:
        """The model asks for float32; the program is float16.

        Emitting the requested type produces "Specified dtype of cast
        does not match that of output tensor" — the declared type is
        what the rest of the program agrees with.
        """

        class _Casts(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return (x.to(lucid.float32) * 3.0).to(lucid.float32) + x

        lucid.manual_seed(0)
        model = _Casts().eval()
        x = lucid.randn(2, 6)
        exported = cml.export(
            model, x, f"{tmp_path}/casts.mlpackage", precision=cml.Precision.FLOAT16
        )
        try:
            assert exported.verify(model, x, relative=True) < 1e-2
        finally:
            exported.close()


class TestWhereTheWorkActuallyGoes:
    """A package that exports perfectly can still run nowhere useful.

    The Neural Engine is a float16 device. A program left at the default
    float32 therefore lands entirely on the CPU however the compute
    units were asked for — measured on a ResNet-18: 66 of 66 operations
    on the CPU at float32, 66 of 68 on the Neural Engine at float16.
    Core ML reports no problem either way, and the accelerator is the
    whole reason this subsystem exists.

    So the plan says why, where somebody is already asking where the
    work went.
    """

    def test_float32_lands_on_the_cpu_and_says_so(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()).eval()
        x = lucid.randn(1, 3, 32, 32)

        exported = cml.export(
            model, x, f"{tmp_path}/f32.mlpackage", compute_units=cml.ComputeUnits.CPU_AND_NE
        )
        try:
            plan = exported.compute_plan()
            if plan.total_compute == 0:
                pytest.skip("this macOS reports no compute plan")
            assert plan.ane_fraction == 0.0
            assert "float32" in plan.note
            assert "FLOAT16" in plan.note
        finally:
            exported.close()

    def test_float16_reaches_the_neural_engine_and_stays_quiet(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        # Big enough that Core ML's planner bothers to dispatch: a
        # two-operation model stays on the CPU at any precision, which is
        # the planner's choice rather than an answer about float16.
        model = nn.Sequential(
            *[layer for _ in range(8) for layer in (nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())]
        ).eval()
        x = lucid.randn(1, 64, 32, 32)

        exported = cml.export(
            model,
            x,
            f"{tmp_path}/f16.mlpackage",
            compute_units=cml.ComputeUnits.CPU_AND_NE,
            precision=cml.Precision.FLOAT16,
        )
        try:
            plan = exported.compute_plan()
            if plan.total_compute == 0:
                pytest.skip("this macOS reports no compute plan")
            assert plan.ane_fraction > 0.0
            assert plan.note == ""
        finally:
            exported.close()

    def test_a_cpu_only_request_is_not_second_guessed(self, tmp_path: object) -> None:
        """The note is for a request that did not take, not for one honoured.

        Somebody who asked for the CPU got the CPU, and telling them how
        to reach the Neural Engine would be answering a question they
        did not ask.
        """
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()).eval()
        x = lucid.randn(1, 3, 32, 32)

        exported = cml.export(
            model, x, f"{tmp_path}/cpu.mlpackage", compute_units=cml.ComputeUnits.CPU_ONLY
        )
        try:
            assert exported.compute_plan().note == ""
        finally:
            exported.close()


class TestAnEmptyConstantIsRefused:
    """A shape the trace could not know, named as such.

    Fast R-CNN's region features are sized by how many proposals
    survived, which on an untrained model with random input is often
    none — so the graph holds a constant of shape ``(0, 512, 7, 7)`` and
    the package, if written, would answer "no proposals" for every input.

    The blob writer already refused it, with "the tensor has no host
    storage": true, and no help at all in working out that the model's
    shapes depend on its data. The refusal now says which shape and
    where it came from.
    """

    def test_an_empty_constant_says_where_the_shape_came_from(self, tmp_path: object) -> None:
        class _Empty(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                # A constant with no elements, of the shape a proposal
                # stage produces when nothing survives.
                nothing = lucid.zeros(0, int(x.shape[-1]))
                return lucid.concat([x, nothing], dim=0)

        with pytest.raises(Exception, match="empty tensor"):
            cml.export(_Empty().eval(), lucid.randn(2, 4), f"{tmp_path}/empty.mlpackage")


class TestTheOutputBufferIsReadCorrectly:
    """Core ML does not always hand back a packed array.

    On the paths that allow the Neural Engine, an output whose innermost
    dimension is not a multiple of the alignment comes back padded. Copied
    as if packed, the values interleave with the padding: right shape,
    wrong numbers, no error. ``tile`` on the last axis is the smallest
    case that produces one.
    """

    @pytest.mark.parametrize(
        "units",
        [cml.ComputeUnits.ALL, cml.ComputeUnits.CPU_ONLY, cml.ComputeUnits.CPU_AND_NE],
        ids=["all", "cpu", "cpu_and_ne"],
    )
    def test_a_padded_output_is_not_read_as_packed(self, units: object, tmp_path: object) -> None:
        x = lucid.arange(0, 12).reshape(1, 1, 3, 4) * 1.0
        model = _Apply(lambda t: lucid.tile(t, (1, 1, 1, 2))).eval()
        reference = model(x)
        exported = cml.export(model, x, f"{tmp_path}/strided.mlpackage", compute_units=units)
        try:
            got = exported.predict(x)
            assert float((got - reference).abs().max().item()) == 0.0
        finally:
            exported.close()


@pytest.mark.parametrize(("name", "fn", "shape"), _SPATIAL, ids=[c[0] for c in _SPATIAL])
def test_a_spatial_op_matches(
    name: str, fn: object, shape: tuple[int, ...], tmp_path: object
) -> None:
    _check(fn, lucid.randn(*shape), tmp_path)


class TestTheGapAgainstCompileIsAccountedFor:
    """Every op `lucid.compile` runs either exports, or is listed here.

    The two backends read the same trace, so an op the MPSGraph builder
    accepts is one an export can meet. Leaving that difference unwritten
    is how a model becomes unexportable without anyone noticing which op
    did it; naming each absence makes adding a compile emitter without an
    export emitter a failing test rather than a surprise months later.
    """

    #: Core ML's program dialect has no equivalent and no decomposition.
    IMPOSSIBLE = {
        # No complex dtype.
        "complex",
        "conj",
        "imag",
        "real",
        "view_as_complex",
        "view_as_real",
        # No linear-algebra solver.
        "det",
        "inv",
        "solve",
        "solve_triangular",
        # One float32 unit in the last place is the whole result, and MIL
        # can neither reinterpret a float's bits nor promise the exact
        # ``exp2`` / ``log2`` an arithmetic ulp needs. A float16 package
        # rounds the step away entirely: it would load and return ``a``.
        "nextafter",
    }

    #: Only reachable from a model in training mode, which export refuses.
    TRAINING_ONLY = {
        "alpha_dropout",
        "bce_loss",
        "bce_with_logits",
        "cross_entropy_loss",
        "drop_block",
        "drop_path",
        "dropoutnd",
        "huber_loss",
        "mse_loss",
        "nll_loss",
    }

    #: In the registry, but decomposed into mapped ops before tracing.
    NEVER_TRACED = {
        "batch_norm",
        "batch_norm1d",
        "batch_norm3d",
        "cube",
        "cube_root",
        # ``rpow_scalar`` belongs to the bucket below rather than this
        # one — nothing calls it — but it is kept here beside its
        # sibling because ``pow_scalar`` was listed here and was wrong:
        # LP pooling traces it, which is how ``zfnet`` found it.  Three
        # models cannot tell a list like this from the truth.
        "rpow_scalar",
        "norm",
        # No Python caller anywhere in ``lucid/``: an engine op with no
        # way to reach it from a model.
        "global_response_norm",
        # ``F.apply_rotary_emb`` decomposes into mapped ops; the engine
        # op it is named after never appears in a trace.
        "rotary_pos_embedding",
    }

    #: The trace does not carry enough to translate them.
    TRACER_GAP = {
        # Operands the tracer does not all wire, re-measured 2026-09-04.
        # Translating from what is recorded would put the wrong values in
        # the result, so each emitter refuses by name rather than
        # guessing at the rest.
        #
        #   meshgrid       one node, N outputs. ``OpScopeFull`` mints a
        #                  single output with an empty shape and each of
        #                  the N ``wire_autograd`` calls overwrites the
        #                  same node, so only the last input survives.
        #                  Carrying it needs a multi-output IR, not a
        #                  wiring fix.
        #   embedding_bag  the weight is recorded, the indices and
        #                  offsets are not. ``embedding`` right above it
        #                  patches this up by calling ``on_op_io``
        #                  explicitly; this one does not, and its
        #                  segments are data-dependent besides.
        #
        # ``grid_sample`` was here until its operands reached the trace.
        # It was never a Core ML limit — MIL has had ``resample`` all
        # along; the op returned before ``wire_autograd`` whenever no
        # gradient was wanted, which is every inference pass.
        "meshgrid",
        "embedding_bag",
    }

    #: Expressible and not written (tracked by LCD-444).
    NOT_YET: set[str] = set()

    def test_no_unaccounted_gap(self) -> None:
        from lucid.coreml._emit import EMITTERS

        registry = {s.name for s in _C_engine.op_registry_all() if not s.internal}
        compiled = {
            name for name in registry | set(EMITTERS) if _C_engine.compile.emitter_registered(name)
        }
        accounted = (
            self.IMPOSSIBLE
            | self.TRAINING_ONLY
            | self.NEVER_TRACED
            | self.TRACER_GAP
            | self.NOT_YET
        )
        unaccounted = compiled - set(EMITTERS) - accounted
        assert not unaccounted, (
            f"lucid.compile runs {sorted(unaccounted)} and lucid.coreml does not. "
            "Add an emitter, or add the op to one of the sets above with the "
            "reason it cannot have one."
        )

    def test_nothing_listed_is_already_supported(self) -> None:
        from lucid.coreml._emit import EMITTERS

        stale = (self.IMPOSSIBLE | self.NOT_YET) & set(EMITTERS)
        assert not stale, f"{sorted(stale)} are mapped now — take them off the list"


class TestTheOperationsMilDoesNotHave:
    """Each of these is several MIL operations standing in for one.

    A decomposition is where a translation goes quietly wrong — the shape
    survives and the values do not — so every one is compared against the
    eager result rather than merely exported.
    """

    def test_unfold_matches_exactly(self, tmp_path: object) -> None:
        for kernel, stride, padding in ((2, 1, 0), (3, 2, 1)):
            _check(
                lambda x, k=kernel, s=stride, p=padding: F.unfold(x, k, stride=s, padding=p),
                lucid.randn(1, 3, 8, 8),
                tmp_path,
            )

    def test_fold_matches_exactly(self, tmp_path: object) -> None:
        """``fold`` adds where blocks overlap, which is the hard half."""
        for kernel, stride, padding, side in ((2, 1, 0, 6), (2, 2, 0, 6), (3, 1, 1, 8)):
            _check(
                lambda x, k=kernel, s=stride, p=padding, n=side: F.fold(
                    F.unfold(x, k, stride=s, padding=p), (n, n), k, stride=s, padding=p
                ),
                lucid.randn(1, 2, side, side),
                tmp_path,
            )

    def test_affine_grid_matches(self, tmp_path: object) -> None:
        for corners in (False, True):
            _check(
                lambda x, c=corners: F.affine_grid(
                    x.reshape(1, 2, 3), (1, 4, 6, 6), align_corners=c
                ),
                lucid.randn(1, 6),
                tmp_path,
            )

    def test_bilinear_and_lp_normalize_match(self, tmp_path: object) -> None:
        weight = lucid.randn(3, 4, 4)
        _check(lambda x: F.bilinear(x, x, weight), lucid.randn(1, 4), tmp_path)
        for order in (1, 2):
            _check(lambda x, p=order: F.normalize(x, p=p, dim=1), lucid.randn(1, 4, 6, 6), tmp_path)

    def test_three_dimensional_resampling_matches(self, tmp_path: object) -> None:
        for mode in ("nearest", "trilinear"):
            _check(
                lambda x, m=mode: F.interpolate(x, scale_factor=2, mode=m),
                lucid.randn(1, 2, 4, 4, 4),
                tmp_path,
            )

    def test_erfinv_is_an_approximation_and_says_how_close(self, tmp_path: object) -> None:
        """MIL has no inverse error function; this one is a polynomial.

        Looser than the other ops on purpose, and measured rather than
        hoped: the central region agrees to ~3e-7 and the tails to ~5e-6,
        which is the order of a float32 export's other error.
        """
        _check(
            lambda x: lucid.erfinv(lucid.tanh(x) * 0.9),
            lucid.randn(1, 4, 16, 16),
            tmp_path,
            tol=1e-5,
        )


class TestNanToNumReplacesEachSpecialValue:
    """``nan_to_num`` is three selects; each has to see its own value.

    ``randn`` never makes a NaN or an infinity, so the op tables would only
    ever check the identity half of it. The input here carries all three.
    """

    _SPECIAL = [[1.5, float("nan"), float("inf"), -float("inf"), -2.0]]

    def test_the_given_replacements(self, tmp_path: object) -> None:
        _check(
            lambda x: lucid.nan_to_num(x, nan=0.5, posinf=7.0, neginf=-7.0),
            lucid.tensor(self._SPECIAL),
            tmp_path,
        )

    def test_the_default_replacements_are_float32s_extremes(self, tmp_path: object) -> None:
        _check(lucid.nan_to_num, lucid.tensor(self._SPECIAL), tmp_path)


class TestMeshgridCarriesAllOfItsOperands:
    """It used to record one input however many it was given.

    ``meshgrid`` wires each output for autograd against its own single
    input, and the tracer takes the last write — so the node ended up
    with one operand for a two-dimensional grid, and the emitter refused
    rather than reconstruct the missing one and put wrong values in it.
    The op records the full list now.

    Which axis each input varies along is the whole difference between
    the two indexings, and the output shapes reveal it only when the
    inputs have different lengths — so the trace carries that too. The
    square case is here because it is the one shapes cannot answer.
    """

    @pytest.mark.parametrize("indexing", ["ij", "xy"])
    @pytest.mark.parametrize("square", [False, True], ids=["oblong", "square"])
    def test_both_indexings_match(self, indexing: str, square: bool, tmp_path: object) -> None:
        import lucid.nn as nn

        second = 3 if square else 4

        class UsesMeshgrid(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                rows = lucid.arange(3).to(lucid.float32)
                columns = lucid.arange(second).to(lucid.float32) * 100.0
                first, last = lucid.meshgrid(rows, columns, indexing=indexing)
                return (first + last) * x.sum()

        model = UsesMeshgrid().eval()
        x = lucid.randn(2, 2)
        exported = cml.export(model, x, f"{tmp_path}/mesh_{indexing}_{second}.mlpackage")
        try:
            assert exported.verify(model, x, relative=True) < 1e-5
        finally:
            exported.close()

    def test_a_grid_the_trace_cannot_describe_is_still_refused(self) -> None:
        """The refusal stays, for a trace that really is short of operands.

        Reconstructing a missing input would put the wrong values in the
        grid, which is worse than not exporting — so the emitter says how
        many it got and how many the grid needs.
        """
        from lucid.coreml._build import UnsupportedOp

        message = str(UnsupportedOp("meshgrid", "the trace recorded 1 operand(s)"))
        assert "meshgrid" in message
        assert "_emit.py" not in message


class TestTheSamplingGridIsTheOneTheModelAsked:
    """``align_corners`` decides which source coordinate each sample reads.

    It reaches the exported package only if the trace carries it. It did
    not for the three-dimensional resamplers, so a model built with
    ``align_corners=True`` exported as its ``False`` counterpart: a
    package that loads, runs, and returns a plausible volume whose values
    are 23% off. Both settings are checked here, in both ranks, because
    agreeing on the default proves nothing about the flag.
    """

    @pytest.mark.parametrize("align_corners", [False, True], ids=["centres", "corners"])
    @pytest.mark.parametrize(
        ("size", "shape", "mode"),
        [
            ((8, 8), (1, 2, 4, 4), "bilinear"),
            ((6, 8, 8), (1, 2, 3, 4, 4), "trilinear"),
            ((5, 7, 6), (1, 2, 3, 4, 4), "trilinear"),
            ((3, 8, 8), (1, 2, 3, 4, 4), "trilinear"),
        ],
        ids=["bilinear", "trilinear-grown", "trilinear-fractional", "trilinear-same-depth"],
    )
    def test_both_settings_export_to_what_they_mean(
        self,
        size: tuple[int, ...],
        shape: tuple[int, ...],
        mode: str,
        align_corners: bool,
        tmp_path: object,
    ) -> None:
        _check(
            lambda x: F.interpolate(x, size=size, mode=mode, align_corners=align_corners),
            lucid.randn(*shape),
            tmp_path,
        )


class TestGridSampleReachesTheExport:
    """It was listed as untranslatable, and the reason was in the tracer.

    ``grid_sample`` returned before ``wire_autograd`` whenever no
    gradient was wanted. Every inference pass takes that path, and an
    export traces one, so the node arrived with no operands at all —
    which reads as "Core ML cannot do this" and was never about Core ML.
    MIL has ``resample``: same operands, same ``(x, y)`` order, same
    normalised range.

    ``mode``, ``padding_mode`` and ``align_corners`` each change the
    answer rather than the layout, so all of them are checked.
    """

    @pytest.mark.parametrize("align_corners", [False, True], ids=["edges", "corners"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border"])
    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    def test_every_setting_exports_to_what_it_means(
        self, mode: str, padding_mode: str, align_corners: bool, tmp_path: object
    ) -> None:
        lucid.manual_seed(1)
        # Coordinates spread past +-1 so the padding mode is actually
        # exercised rather than only the interior.
        grid = lucid.rand(1, 4, 4, 2) * 2.6 - 1.3

        _check(
            lambda x: F.grid_sample(
                x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
            ),
            lucid.randn(1, 3, 5, 5),
            tmp_path,
        )

    def test_reflection_exports_as_reflection(self, tmp_path: object) -> None:
        """Reflection reaches the trace as a reflected grid sampled with border.

        ``F.grid_sample`` mirrors the grid in elementwise ops and hands the
        kernel ``border`` padding, so the export carries those ops and
        ``resample`` with ``border`` — and agrees with the eager model,
        which matches the reference's reflection.
        """
        lucid.manual_seed(2)
        grid = lucid.rand(1, 4, 4, 2) * 3.0 - 1.5
        _check(
            lambda x: F.grid_sample(x, grid, padding_mode="reflection"),
            lucid.randn(1, 3, 5, 5),
            tmp_path,
        )


class TestTheOpsetTheWriterDeclares:
    """A payload shaped for an older opset than the one declared.

    ``gather_along_axis`` gained ``validate_indices`` in iOS17, which is
    the opset this writer says it is emitting. Leaving the parameter out
    produced a package that failed to *parse* — "Required param
    'validate_indices' is missing" — so nothing in it ran, and every
    model with a gather went down together. ``coatnet`` was one.

    Rank-1 indices along an axis is the shape that reaches
    ``gather_along_axis`` rather than the row-lookup ``gather``.
    """

    def test_gather_along_an_axis_parses_and_matches(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        index = lucid.tensor([[2, 0, 1, 3], [1, 1, 0, 2]]).to(lucid.int64)
        _check(lambda x: lucid.gather(x, 1, index), lucid.randn(2, 4), tmp_path)

    def test_a_row_lookup_still_parses(self, tmp_path: object) -> None:
        """``embedding`` takes the other branch and already had it."""
        lucid.manual_seed(0)
        table = nn.Embedding(4, 8).eval()
        _check(lambda ids: table(ids), lucid.tensor([[0, 2, 1]]).to(lucid.int64), tmp_path)


class TestTheWindowPartitionGetsPastTheRankCap:
    """Core ML caps tensors at rank 5; window partition wants six.

    ``(B, H, W, C)`` becomes ``(B, H/w, w, W/w, w, C)``, is permuted, and
    collapses back — the tall shape exists for exactly two operations.
    Every windowed-attention transformer is built from it, so refusing
    the rank took Swin and MaxViT (22 of the zoo's factories) with it.

    The triple is now staged instead: a reshape is a reinterpretation of
    a contiguous buffer, so a run of adjacent axes can be grouped into
    one, and each stage views the value as ``(left, a, b, right)`` — rank
    four whatever the logical rank is — and swaps the middle pair. The
    tall shape stays bookkeeping and is never asked for.
    """

    def test_a_window_partition_exports(self, tmp_path: object) -> None:
        def partition(x: lucid.Tensor) -> lucid.Tensor:
            y = x.reshape(1, 4, 2, 4, 2, 3)
            y = lucid.permute(y, (0, 1, 3, 2, 4, 5))
            return y.reshape(16, 2, 2, 3)

        lucid.manual_seed(0)
        _check(partition, lucid.randn(1, 8, 8, 3), tmp_path)

    def test_the_merge_back_exports(self, tmp_path: object) -> None:
        def merge(x: lucid.Tensor) -> lucid.Tensor:
            y = x.reshape(1, 4, 4, 2, 2, 3)
            y = lucid.permute(y, (0, 1, 3, 2, 4, 5))
            return y.reshape(1, 8, 8, 3)

        lucid.manual_seed(0)
        _check(merge, lucid.randn(16, 2, 2, 3), tmp_path)

    def test_a_permutation_that_is_not_one_adjacent_swap(self, tmp_path: object) -> None:
        """MaxViT's grid partition moves three axes, not two.

        Staging by adjacent swaps covers it without special-casing —
        one stage per inversion — which is why the rewrite matches the
        pattern rather than the model.
        """

        def grid(x: lucid.Tensor) -> lucid.Tensor:
            y = x.reshape(1, 2, 4, 2, 4, 3)
            y = lucid.permute(y, (0, 2, 4, 1, 3, 5))
            return y.reshape(16, 2, 2, 3)

        lucid.manual_seed(0)
        _check(grid, lucid.randn(1, 8, 8, 3), tmp_path)

    def test_a_tall_tensor_that_does_not_collapse_is_still_refused(self, tmp_path: object) -> None:
        """The cap is real; only the transient case can be staged.

        A rank-6 value that something other than the triple reads has to
        exist, and Core ML has no way to hold it. Saying so by name beats
        the compiler's parse failure several steps downstream.
        """

        class Tall(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x.reshape(1, 4, 2, 4, 2, 3).sum(dim=5)

        with pytest.raises(cml.UnsupportedRank):
            cml.export(Tall().eval(), lucid.randn(1, 8, 8, 3), f"{tmp_path}/tall.mlpackage")


class TestAHostComputedResultSaysSo:
    """A model that reads values computes on the host, not in the graph.

    A result built from ``.item()`` is a constant by the time the trace
    ends, and nothing in the graph depends on the input.  The compile path
    used to bake that in and answer every later input with the first one's
    result; export refuses instead, which is the safe direction — but
    "output is not part of the trace" did not say why.  (``lucid.histogram``
    was such a model until it became tensor operations, API-05.)
    """

    def test_the_refusal_names_the_cause(self, tmp_path: object) -> None:
        class ReadsValues(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.tensor([x.sum().item()] * 4)

        with pytest.raises(ValueError, match="host"):
            cml.export(ReadsValues().eval(), lucid.randn(16), f"{tmp_path}/host.mlpackage")


class TestJoiningTwoMasks:
    """``&`` between booleans, and the reason it reached nothing until now.

    A bitwise operation has no gradient, so it never goes through the
    wiring that also records a traced operation's operands — the same
    position comparisons are in, and ``Compare.cpp`` pushes them into the
    tracer by hand for exactly this reason. Nothing pushed these. The
    trace held ``bitwise_and`` with no inputs at all, its consumer read
    an identifier nothing had produced, and the export died several
    steps later on a bare ``KeyError: 10`` — no operation named, no
    model, nothing to act on.

    It had gone unseen because no test reached it: of a hundred and
    twenty-nine emitters, fifteen had never run in any test, and this was
    the one among them that was broken. An emitter nothing exercises has
    never been compared against anything.

    ``|`` and ``^`` are here because they were in the same file with the
    same defect and had no emitter either, and a mask joined one way is
    joined the others soon after.
    """

    @pytest.mark.parametrize(
        ("name", "fn"),
        [
            ("and", lambda x: ((x > 0.0) & (x < 0.5)).to(lucid.float32)),
            ("or", lambda x: ((x > 0.0) | (x < -0.5)).to(lucid.float32)),
            ("xor", lambda x: ((x > 0.0) ^ (x < 0.5)).to(lucid.float32)),
        ],
    )
    def test_it_agrees_with_the_eager_model(self, name, fn, tmp_path) -> None:
        lucid.manual_seed(0)
        _check(fn, lucid.randn(3, 6), tmp_path)

    def test_a_joined_mask_selects(self, tmp_path: object) -> None:
        """What the operation is for, rather than the operation alone.

        A padding mask joined to a causal one, then used — which is where
        an attention model meets this.
        """

        def masked(x: lucid.Tensor) -> lucid.Tensor:
            return lucid.where((x > 0.0) & (x < 1.0), x, x * 0.0)

        lucid.manual_seed(0)
        _check(masked, lucid.randn(3, 6), tmp_path)

    def test_it_still_answers_a_different_input(self, tmp_path: object) -> None:
        """The broken trace's other possible ending.

        An operation with no recorded operands is a constant as far as
        the graph is concerned, and the folder is entitled to evaluate it
        and bake the answer in. That would have exported cleanly and
        returned the first input's mask forever.
        """

        def joined(x: lucid.Tensor) -> lucid.Tensor:
            return ((x > 0.0) & (x < 0.5)).to(lucid.float32)

        lucid.manual_seed(0)
        first, second = lucid.randn(3, 6), lucid.randn(3, 6) * 4.0
        model = _Apply(joined).eval()
        exported = cml.export(model, first, f"{tmp_path}/joined.mlpackage")
        try:
            answered = exported.predict(second)
            assert float((answered - model(second)).abs().max().item()) == 0.0
            moved = float((answered - exported.predict(first)).abs().max().item())
            assert moved > 0.0
        finally:
            exported.close()


class TestBitsAreNotBooleans:
    """MIL has ``logical_and`` and nothing below it.

    Lucid's ``&`` also takes integers and combines their bit patterns.
    Emitting ``logical_and`` for that would load, run, and answer — every
    non-zero integer is true, so ``5 & 3`` comes back ``True`` where the
    model wanted ``1``. It is refused by name instead.
    """

    def test_an_integer_pair_is_refused(self, tmp_path: object) -> None:
        def bits(x: lucid.Tensor) -> lucid.Tensor:
            return (x.to(lucid.int32) & 3).to(lucid.float32)

        lucid.manual_seed(0)
        with pytest.raises(cml.UnsupportedOp, match="bit level"):
            cml.export(_Apply(bits).eval(), lucid.randn(3, 6) * 8.0, f"{tmp_path}/bits.mlpackage")

    def test_a_shift_is_refused_as_unmapped(self, tmp_path: object) -> None:
        """Core ML has no shift at all, so this one needs no special case."""

        def shifted(x: lucid.Tensor) -> lucid.Tensor:
            return (x.to(lucid.int32) << 1).to(lucid.float32)

        lucid.manual_seed(0)
        with pytest.raises(cml.UnsupportedOp, match="bitwise_left_shift"):
            cml.export(
                _Apply(shifted).eval(), lucid.randn(3, 6) * 8.0, f"{tmp_path}/shift.mlpackage"
            )


# ── the emitters nothing had reached ─────────────────────────────────────────
#
# Recorded by wrapping the emitter table and running the whole suite:
# of a hundred and twenty-nine, fifteen had never been called. One of
# them, ``bitwise_and``, was broken; the rest were fine and unguarded,
# which is the same thing a week later. They are exercised here so the
# table has no unvisited corner left.

_UNREACHED = [
    ("greater_equal", lambda x: (x >= 0.0).to(lucid.float32)),
    ("less_equal", lambda x: (x <= 0.0).to(lucid.float32)),
    ("invert", lambda x: (~(x > 0.0)).to(lucid.float32)),
    ("isfinite", lambda x: lucid.isfinite(x).to(lucid.float32)),
    ("min", lucid.min),
    ("min_dim", lambda x: lucid.min(x, dim=1).values),
    ("min_keepdim", lambda x: lucid.min(x, dim=1, keepdim=True).values),
    ("triu", lucid.triu),
    ("triu_above_the_diagonal", lambda x: lucid.triu(x, diagonal=1)),
    ("tril_below_the_diagonal", lambda x: lucid.tril(x, diagonal=-1)),
    ("clone", lambda x: x.clone() + 1.0),
    # Factories: the value is a constant, and what is checked is that the
    # constant written into the package is the one Lucid made.
    ("arange", lambda x: x + lucid.arange(5).to(lucid.float32)),
    ("ones", lambda x: x * lucid.ones(3, 5)),
    ("zeros", lambda x: x + lucid.zeros(3, 5)),
    ("full", lambda x: x + lucid.full((3, 5), 2.5)),
]


@pytest.mark.parametrize(("name", "fn"), _UNREACHED, ids=[name for name, _fn in _UNREACHED])
def test_an_emitter_no_test_had_reached(name, fn, tmp_path) -> None:
    lucid.manual_seed(0)
    _check(fn, lucid.randn(3, 5), tmp_path)


class TestArangeEmitter:
    """``arange`` goes out as a constant of the type the trace gave it.

    No export reaches this emitter today: the tracer records the op, but
    what its consumers read is the tensor it made, carried as a buffer.
    It is called directly, so the integer branch is not an unvisited
    corner waiting for the tracer to change.
    """

    @staticmethod
    def _builder(tmp_path: object) -> Any:
        from lucid.coreml import _spec
        from lucid.coreml._build import Builder

        body_mil, body_blob = _spec.body_dtypes(_spec.Precision.FLOAT32)
        cm = _C_engine.coreml
        blob = cm.BlobWriter(f"{tmp_path}/weights.bin")
        return Builder(cm.MilProgram([]), blob, body_mil, body_blob, False)

    @staticmethod
    def _emit(builder: Any, dtype: object, start: float, count: int) -> str:
        from lucid.coreml._emit import EMITTERS

        out = types.SimpleNamespace(id=0, shape=(count,), dtype=dtype)
        op = types.SimpleNamespace(
            name="arange", attrs={"start": start, "step": 1.0}, outputs=[out]
        )
        return str(EMITTERS["arange"](builder, op, []).name)

    def test_an_integer_sequence_is_an_int32_constant(self, tmp_path: object) -> None:
        builder = self._builder(tmp_path)
        name = self._emit(builder, _C_engine.I64, 2.0, 4)
        assert builder.dtype_of(name) == _C_engine.coreml.DTYPE_INT32
        assert builder.shape_of(name) == [4]

    def test_a_float_sequence_is_the_body_float(self, tmp_path: object) -> None:
        builder = self._builder(tmp_path)
        name = self._emit(builder, _C_engine.F32, 0.5, 4)
        assert builder.dtype_of(name) == _C_engine.coreml.DTYPE_FLOAT32

    def test_integers_past_int32_are_refused(self, tmp_path: object) -> None:
        from lucid.coreml._build import UnsupportedOp

        builder = self._builder(tmp_path)
        with pytest.raises(UnsupportedOp):
            self._emit(builder, _C_engine.I64, float(2**31 - 2), 4)


def test_a_grid_of_two_axes(tmp_path: object) -> None:
    """``meshgrid`` records every operand, which it did not always.

    It reached the tracer with one of a two-axis grid's inputs, so the
    other was a constant in the package — right for the example input and
    frozen after. Multiplying the two planes together needs both.
    """

    def grid(x: lucid.Tensor) -> lucid.Tensor:
        rows, columns = lucid.meshgrid(x[0], x[1])
        return rows * columns

    lucid.manual_seed(0)
    _check(grid, lucid.randn(2, 4), tmp_path)


def test_scattering_into_a_row(tmp_path: object) -> None:
    """Repeated indices accumulate, which is the whole of the operation."""

    def scattered(x: lucid.Tensor) -> lucid.Tensor:
        index = lucid.tensor([[0, 1, 0, 1, 0]]).to(lucid.int64)
        return lucid.scatter_add(x, 0, index, x[:1] * 2.0)

    lucid.manual_seed(0)
    _check(scattered, lucid.randn(3, 5), tmp_path)


def test_overwriting_into_a_row(tmp_path: object) -> None:
    """``scatter``, ``x[key] = v`` and ``index_copy`` trace as ``scatter_set``.

    None had a translation; ``index_copy`` was refused by name, and once the
    other two stopped tracing as a gather-subtract-scatter_add (CHA-158) a
    model that wrote into a tensor was too.  Written over NaN, so the
    overwrite cannot pass as an add.
    """

    def overwritten(x: lucid.Tensor) -> lucid.Tensor:
        index = lucid.tensor([[2, 0, 1, 2, 0]]).to(lucid.int64)
        out = lucid.scatter(x * float("nan"), 0, index, x[:1] * 2.0)
        out = out.index_copy(1, lucid.tensor([4, 0]), x[:, 1:3])
        out[:, 2] = 0.5
        return lucid.nan_to_num(out, nan=-7.0)

    lucid.manual_seed(0)
    _check(overwritten, lucid.randn(3, 5), tmp_path)


def test_a_model_that_does_nothing_is_refused(tmp_path: object) -> None:
    """``nn.Identity`` alone leaves no operation to export.

    The refusal is right and worth pinning: a package with an empty
    program would load and return its input, which reads as working.
    """

    class DoesNothing(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.through = nn.Identity()

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.through(x)

    lucid.manual_seed(0)
    with pytest.raises(ValueError, match="never reached an op"):
        cml.export(DoesNothing().eval(), lucid.randn(3, 5), f"{tmp_path}/nothing.mlpackage")


class TestDrawingRandomNumbersIsRefused:
    """The refusal this subsystem works hardest for, and it had no test.

    ``randn`` was reached only sideways — a model in the reloading tests
    uses it to make an export fail on purpose — and ``rand`` was reached
    by nothing. Both fold at build time: the operation's inputs are all
    constants, so the draw happens once, while the package is written,
    and every prediction for the life of the file returns that sample.

    Nothing about it looks wrong. The package loads, the numbers are
    plausible, and a variational encoder exported that way has a latent
    that never moves. Refusing is the only honest answer, so it is worth
    a test that names it rather than one that happens to pass through.
    """

    @pytest.mark.parametrize("draw", [lucid.randn, lucid.rand], ids=["randn", "rand"])
    def test_it_names_the_operation(self, draw, tmp_path) -> None:
        class Samples(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x + draw(3, 5)

        lucid.manual_seed(0)
        with pytest.raises(cml.UnsupportedOp, match="fixed sample"):
            cml.export(Samples().eval(), lucid.randn(3, 5), f"{tmp_path}/sampled.mlpackage")
