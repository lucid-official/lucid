"""Palettization and sparsity: what the package stores, and what runs it.

Both settings replace a weight with something smaller and lossy, which
makes them the easiest place in the subsystem to be wrong without
anything saying so — a mis-packed key stream or a table read at the wrong
stride still loads, still predicts, and still returns numbers of about
the right size.

So the tests here do not compare the compressed model against the
original. They compare it against **its own reconstruction**: the exact
values the tables and keys say the weight should now hold. That
separates the two questions. Whether four bits is enough for a given
model is the caller's trade to make and ``verify`` measures it; whether
the package computes what it claims to store is this file's business,
and the answer has to be exact.

That distinction is what caught the one real defect here. Core ML's GPU
path expands ``constexpr_lut_to_dense`` incorrectly for palette sizes 4,
16 and 256 — silently, with no error and plausible-looking output. A
comparison against the original model reads that as "four-bit
palettization is lossy", which it also is; only a comparison against the
package's own tables separates a wrong answer from an approximate one.
``lucid.coreml`` now keeps palettized models off that path, and the last
test here is what notices if that stops working.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.coreml as cml
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.coreml import _build

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)

#: Every palette width the operation takes.  Three and six do not divide
#: a byte, so their keys straddle byte boundaries; eight is the only one
#: that is not a sub-byte type at all. All three cases have been wrong at
#: some point, which is why the list is exhaustive rather than a sample.
WIDTHS = [1, 2, 3, 4, 6, 8]


class _Stack(nn.Module):
    """Enough convolutions that Core ML plans this across accelerators.

    A one-operation model stays on the CPU whatever units are asked for,
    and the CPU expands every palette correctly — so a smaller model
    cannot see the defect the last test covers. Eight layers of 128
    channels is the smallest shape that reproduced it.
    """

    def __init__(self, layers: int = 8, channels: int = 128) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                for _ in range(layers)
            ]
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


def _weight_of(conv: nn.Module) -> lucid.Tensor:
    weight = conv.weight
    return weight.data if hasattr(weight, "data") else weight


def _reconstruct(weight: lucid.Tensor, bits: int) -> lucid.Tensor:
    """What the tables and keys say the weight now holds.

    Deliberately built from the same helpers the builder uses: the claim
    under test is that the *package* reproduces this, not that the
    clustering is any good.
    """
    count = 1 << bits
    groups = _build._palette_groups(int(weight.shape[0]), count, int(weight.numel()))
    rows = weight.reshape(groups, -1)
    palettes = _build._palettes_for(rows, count)
    keys = _build._assign(rows, _build._edge_table(palettes), count)
    return lucid.gather(palettes, 1, keys).reshape(*weight.shape)


def _run_stack(model: _Stack, x: lucid.Tensor, bits: int) -> lucid.Tensor:
    out = x
    for conv in model.convs:
        out = F.conv2d(out, _reconstruct(_weight_of(conv), bits), padding=1)
    return out


class TestTheKeysSurviveTheTrip:
    @pytest.mark.parametrize("bits", WIDTHS)
    def test_packing_a_key_stream_round_trips(self, bits: int) -> None:
        """The keys are a dense little-endian bit stream, no padding.

        Not one key per byte and not one row per byte: 1000 six-bit keys
        occupy 750 bytes and most of them straddle a boundary. The
        decoder here is the specification, written out longhand.
        """
        lucid.manual_seed(0)
        count = 1 << bits
        keys = (lucid.rand(1000) * (count - 0.001)).to(lucid.int64)
        packed = _C_engine.coreml.pack_bits(_unwrap(keys), bits)
        assert len(packed) == -(-1000 * bits // 8)

        got: list[int] = []
        accumulator = held = 0
        for byte in packed:
            accumulator |= byte << held
            held += 8
            while held >= bits and len(got) < 1000:
                got.append(accumulator & (count - 1))
                accumulator >>= bits
                held -= bits
        assert got == keys.tolist()

    @pytest.mark.parametrize("count", [2, 4, 16, 64, 256])
    def test_the_bisection_agrees_with_a_linear_scan(self, count: int) -> None:
        """``_assign`` replaced ``bucketize`` and has to answer the same.

        ``bucketize`` walks the whole edge list per element, which at 256
        entries is 255 passes over the weight and most of an export. The
        bisection does it in ``log2(count)`` — and gives each row its own
        edges, which ``bucketize`` cannot do at all.
        """
        lucid.manual_seed(0)
        groups, width = 5, 200
        palettes = lucid.sort(lucid.randn(groups, count), dim=-1).values
        values = lucid.randn(groups, width) * 2
        got = _build._assign(values, _build._edge_table(palettes), count).tolist()

        table, sample = palettes.tolist(), values.tolist()
        for row in range(groups):
            edges = [(table[row][i] + table[row][i + 1]) / 2 for i in range(count - 1)]
            for column in range(width):
                want = sum(1 for edge in edges if edge < sample[row][column])
                assert got[row][column] == want


class TestThePackageComputesWhatItStores:
    @pytest.mark.parametrize("bits", WIDTHS)
    def test_a_palettized_export_reproduces_its_own_tables(
        self, bits: int, tmp_path: object
    ) -> None:
        """Exact, not close.

        Expanding a table is a copy, not an arithmetic — every element
        equals one of the stored entries. Anything but a bit-for-bit
        match means the keys, the tables or the grouping were read
        differently than they were written.
        """
        lucid.manual_seed(0)
        model = _Stack().eval()
        x = lucid.randn(1, 128, 32, 32)
        want = _run_stack(model, x, bits)

        exported = cml.export(
            model, x, f"{tmp_path}/p{bits}.mlpackage", weights=cml.Palettize(bits=bits)
        )
        try:
            got = exported.predict(x)
            scale = max(float(want.abs().max().item()), 1e-9)
            assert float((got - want).abs().max().item()) / scale < 1e-6
        finally:
            exported.close()

    def test_a_sparse_export_reproduces_its_own_mask(self, tmp_path: object) -> None:
        """The survivors and the mask, against the weight they describe."""
        lucid.manual_seed(0)
        model = _Stack(layers=4).eval()
        x = lucid.randn(1, 128, 16, 16)

        want = x
        for conv in model.convs:
            weight = _weight_of(conv)
            flat = weight.reshape(-1)
            threshold = float(lucid.quantile(flat.abs(), 0.5).item())
            kept = lucid.where(flat.abs() > threshold, flat, lucid.zeros_like(flat))
            want = F.conv2d(want, kept.reshape(*weight.shape), padding=1)

        exported = cml.export(
            model, x, f"{tmp_path}/s.mlpackage", weights=cml.Sparsify(ratio=0.5)
        )
        try:
            got = exported.predict(x)
            scale = max(float(want.abs().max().item()), 1e-9)
            assert float((got - want).abs().max().item()) / scale < 1e-6
        finally:
            exported.close()


class TestWhatIsLeftAlone:
    @pytest.mark.parametrize(
        ("name", "weights"),
        [
            ("palettize", cml.Palettize(bits=4)),
            ("sparsify", cml.Sparsify(ratio=0.5)),
        ],
    )
    def test_a_value_with_no_channel_axis_is_stored_whole(
        self, name: str, weights: object
    ) -> None:
        """Biases and norm parameters are not weights to approximate.

        A rank-1 value has no output channel to group or threshold
        along, and setting half a batch norm's scales to zero removes
        those channels from the network entirely. The int8 path has
        refused rank-1 from the start; these two now refuse it on the
        same grounds.
        """
        lucid.manual_seed(0)
        vector = lucid.randn(4096)
        if isinstance(weights, cml.Palettize):
            assert _build._palettize_weight(vector, weights.bits) is None
        else:
            assert _build._sparsify_weight(vector, weights.ratio) is None

    def test_a_constant_with_nothing_above_its_median_stays_dense(self) -> None:
        """A uniform tensor has no element above its own median.

        Folding a batch norm leaves whole zero vectors behind, and
        thresholding one keeps nothing at all. That would be a sparse
        constant with an empty payload, which is not a thing to write —
        it goes out dense instead of failing.
        """
        assert _build._sparsify_weight(lucid.zeros(64, 64), 0.5) is None
        assert _build._sparsify_weight(lucid.ones(64, 64) * 3.0, 0.5) is None

    @pytest.mark.parametrize("bits", [0, 5, 7, 9, 16])
    def test_a_width_the_operation_cannot_carry_is_refused(self, bits: int) -> None:
        with pytest.raises(ValueError, match="palettization takes"):
            cml.Palettize(bits=bits)

    @pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
    def test_a_ratio_outside_the_range_is_refused(self, ratio: float) -> None:
        with pytest.raises(ValueError):
            cml.Sparsify(ratio=ratio)


class TestAStochasticForwardIsRefused:
    """Sampling inside ``forward`` is a boundary, not a missing emitter.

    MIL has ``random_normal`` and emitting it produces a package that
    loads and runs — and returns the same numbers on every call, from a
    fresh load, and from a second export made under a different seed.
    Its inputs are all constants, so the compiler folds it and the
    sample is drawn once at build time. That is a deterministic model
    wearing a sampler's shape.

    The refusal is what this test protects: an emitter added later
    without measuring the fold would turn a clear error into a
    variational encoder whose latent never moves.
    """

    def test_a_model_that_samples_says_what_is_wrong(self, tmp_path: object) -> None:
        class _Sampler(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x + lucid.randn(2, 3)

        with pytest.raises(NotImplementedError, match="one fixed sample"):
            cml.export(_Sampler().eval(), lucid.zeros(2, 3), f"{tmp_path}/s.mlpackage")

    def test_an_unmapped_op_still_reads_as_a_gap(self) -> None:
        """The other message has to stay the other message.

        A refusal with a reason and an operation nobody has mapped want
        opposite things from the reader: one says do not do this, the
        other says here is where to add it.
        """
        from lucid.coreml._build import UnsupportedOp

        assert "no Core ML translation" in str(UnsupportedOp("nonesuch"))
        assert "_emit.py" in str(UnsupportedOp("nonesuch"))
        assert "_emit.py" not in str(UnsupportedOp("nonesuch", "because measured"))


class TestPalettizedModelsAvoidTheGpu:
    """The defect this file exists for.

    Core ML's GPU path expands the table incorrectly for palette sizes 4,
    16 and 256 — measured on macOS 26 against the package's own tables,
    at 15% error for the model above, while the CPU and the Neural Engine
    are exact. It reports nothing. ``ALL`` is the default, so without
    this guard the common case would be the wrong one.
    """

    def test_the_default_units_exclude_the_gpu(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        model = _Stack(layers=2, channels=64).eval()
        x = lucid.randn(1, 64, 8, 8)
        exported = cml.export(
            model, x, f"{tmp_path}/p.mlpackage", weights=cml.Palettize(bits=4)
        )
        try:
            assert exported.palettized
            assert exported.compute_units is cml.ComputeUnits.CPU_AND_NE
        finally:
            exported.close()

    def test_asking_for_the_gpu_says_why_it_cannot(self, tmp_path: object) -> None:
        """Refused rather than redirected.

        A caller who named the GPU is owed an answer about the GPU, and
        the answer is that this package cannot be run there correctly.
        """
        lucid.manual_seed(0)
        model = _Stack(layers=2, channels=64).eval()
        x = lucid.randn(1, 64, 8, 8)
        path = f"{tmp_path}/g.mlpackage"
        cml.export(model, x, path, weights=cml.Palettize(bits=4)).close()

        with pytest.raises(ValueError, match="GPU"):
            cml.load(path, compute_units=cml.ComputeUnits.CPU_AND_GPU)

    def test_an_uncompressed_model_keeps_the_units_it_was_given(
        self, tmp_path: object
    ) -> None:
        """The guard reads the package, not the request.

        Nothing about an ordinary export changes, including its right to
        the GPU — the restriction has to be as narrow as the defect.
        """
        lucid.manual_seed(0)
        model = _Stack(layers=2, channels=64).eval()
        x = lucid.randn(1, 64, 8, 8)
        exported = cml.export(model, x, f"{tmp_path}/d.mlpackage")
        try:
            assert not exported.palettized
            assert exported.compute_units is cml.ComputeUnits.ALL
        finally:
            exported.close()


class TestTheSizeActuallyFalls:
    @pytest.mark.parametrize("bits", [1, 2, 4, 6])
    def test_a_narrower_palette_writes_a_smaller_package(
        self, bits: int, tmp_path: object
    ) -> None:
        """Compression that does not compress is a bug, not a trade.

        The tables are stored too, so the saving is not exactly the ratio
        of bit widths — the check is that each width lands inside the
        band its keys and tables allow.
        """
        import os

        def _size(root: str) -> int:
            return sum(
                os.path.getsize(os.path.join(walked, name))
                for walked, _dirs, names in os.walk(root)
                for name in names
            )

        lucid.manual_seed(0)
        model = _Stack(layers=4).eval()
        x = lucid.randn(1, 128, 16, 16)

        dense = cml.export(model, x, f"{tmp_path}/dense.mlpackage")
        packed = cml.export(
            model, x, f"{tmp_path}/p{bits}.mlpackage", weights=cml.Palettize(bits=bits)
        )
        try:
            ratio = _size(dense.path) / _size(packed.path)
            # Keys alone would give 32/bits against a float32 body; the
            # tables take some of that back, so the floor is generous.
            assert ratio > 32 / bits * 0.5
        finally:
            dense.close()
            packed.close()
