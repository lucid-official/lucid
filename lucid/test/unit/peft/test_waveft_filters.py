"""The wavelet filters of WaveFT, checked against the equations that define them, and the inverse transform built from them.

No table is compared with another table here.  A scaling filter ``h`` of an orthogonal wavelet is
pinned by its definition: ``sum h = sqrt 2``, the double-shift orthonormality, and a number of vanishing
moments (``N`` for ``dbN`` and ``symN``, ``2N`` for ``coifN`` along with the scaling function's own
``2N - 1``); for the Daubechies and symlet families the squared response is the same closed form of
``N``.  The other three filters follow from ``h`` by the quadrature-mirror relations.  The
transform is checked by perfect reconstruction (analysis, then inverse, gives the matrix back) and
against the synthesis sums written out in ``_wavelet_forms``.  ``parity`` tests compare the tables
with the reference library's where that library's are right.
"""

import math
from types import ModuleType

import pytest

import lucid
from lucid.peft.tuners.waveft._filters import _SCALING, WAVELET_FAMILIES, _filter_bank
from lucid.peft.tuners.waveft._idwt import _idwt2
from lucid.test.unit.peft import _wavelet_forms as forms
from lucid.test.unit.peft._parity import adapter_library

_ORTHOGONAL = [name for name in WAVELET_FAMILIES if not name.startswith("coif")]
_COIFLETS = [name for name in WAVELET_FAMILIES if name.startswith("coif")]


def _order(name: str) -> int:
    """``N`` of ``dbN`` / ``symN`` / ``coifN``."""
    return int(name.lstrip("dbsymcoif"))


def _vanishing_moments(name: str) -> int:
    """How many moments of the wavelet vanish: ``N``, or ``2N`` for a coiflet."""
    return 2 * _order(name) if name.startswith("coif") else _order(name)


def _length(name: str) -> int:
    return (6 if name.startswith("coif") else 2) * _order(name)


def test_the_families_are_the_ones_the_reference_names_have() -> None:
    assert WAVELET_FAMILIES == (
        *(f"db{n}" for n in range(1, 11)),
        *(f"sym{n}" for n in range(2, 11)),
        *(f"coif{n}" for n in range(1, 6)),
    )


class TestScalingFilters:
    @pytest.mark.parametrize("name", WAVELET_FAMILIES)
    def test_the_length_is_two_n_or_six_n(self, name: str) -> None:
        assert len(_SCALING[name]) == _length(name)

    @pytest.mark.parametrize("name", WAVELET_FAMILIES)
    def test_the_taps_sum_to_root_two_and_are_orthonormal_under_double_shifts(
        self, name: str
    ) -> None:
        h = _SCALING[name]
        assert abs(sum(h) - math.sqrt(2)) <= 1e-14
        assert forms.orthonormality_gap(h) <= 1e-14

    @pytest.mark.parametrize("name", WAVELET_FAMILIES)
    def test_the_wavelet_has_its_vanishing_moments_and_not_one_more(self, name: str) -> None:
        h = _SCALING[name]
        for order in range(_vanishing_moments(name)):
            scale = sum(abs(k**order * c) for k, c in enumerate(h))
            assert abs(forms.wavelet_moment(h, order)) <= 1e-12 * scale, order
        # Not vacuous: the next moment is far from zero.
        assert abs(forms.wavelet_moment(h, _vanishing_moments(name))) > 0.1

    @pytest.mark.parametrize("name", _COIFLETS)
    def test_a_coiflets_scaling_function_has_vanishing_moments_about_its_centre(
        self, name: str
    ) -> None:
        h, n = _SCALING[name], _order(name)
        for order in range(1, 2 * n):
            scale = sum(abs((k - 2 * n) ** order * c) for k, c in enumerate(h))
            assert abs(forms.scaling_moment(h, order, 2 * n)) <= 1e-9 * scale, order

    @pytest.mark.parametrize("name", _ORTHOGONAL)
    def test_the_squared_response_is_the_daubechies_closed_form(self, name: str) -> None:
        h, n = _SCALING[name], _order(name)
        for omega in (0.1, 0.7, 1.3, 2.0, 2.6, 3.0):
            assert abs(forms.response_energy(h, omega) - forms.daubechies_energy(n, omega)) <= 1e-12

    @pytest.mark.parametrize("n", range(1, 11))
    def test_a_daubechies_filter_puts_its_energy_in_front_of_the_mirror_image(self, n: int) -> None:
        """Extremal phase: every partial energy sum is at least the mirror image's."""
        h = _SCALING[f"db{n}"]
        mirror = h[::-1]
        for count in range(1, len(h)):
            assert forms.front_energy(h, count) >= forms.front_energy(mirror, count) - 1e-12

    def test_the_symlets_of_order_two_and_three_are_the_daubechies_filters(self) -> None:
        for n in (2, 3):
            assert (
                max(
                    abs(a - b) for a, b in zip(_SCALING[f"sym{n}"], _SCALING[f"db{n}"], strict=True)
                )
                < 1e-12
            )

    def test_a_symlet_of_order_four_or_more_is_not_the_daubechies_filter_and_is_more_symmetric(
        self,
    ) -> None:
        for n in (4, 5, 6):
            sym, daub = _SCALING[f"sym{n}"], _SCALING[f"db{n}"]
            assert max(abs(a - b) for a, b in zip(sym, daub, strict=True)) > 0.01
            assert forms.phase_deviation(sym) < forms.phase_deviation(daub)


class TestFilterBank:
    @pytest.mark.parametrize("name", WAVELET_FAMILIES)
    def test_the_other_filters_are_the_quadrature_mirror_and_the_reversals(self, name: str) -> None:
        bank = _filter_bank(name)
        h, taps = _SCALING[name], len(_SCALING[name])
        assert bank.rec_lo == h and bank.dec_lo == h[::-1] and bank.length == taps
        assert bank.rec_hi == tuple((-1) ** k * h[taps - 1 - k] for k in range(taps))
        assert bank.dec_hi == bank.rec_hi[::-1]

    @pytest.mark.parametrize("name", WAVELET_FAMILIES)
    def test_the_high_pass_filter_is_orthogonal_to_the_low_pass_at_every_double_shift(
        self, name: str
    ) -> None:
        bank = _filter_bank(name)
        taps = bank.length
        for shift in range(-(taps // 2) + 1, taps // 2):
            product = sum(
                bank.rec_lo[k] * bank.rec_hi[k + 2 * shift]
                for k in range(taps)
                if 0 <= k + 2 * shift < taps
            )
            assert abs(product) <= 1e-14

    def test_an_unknown_wavelet_is_refused_with_the_names_there_are(self) -> None:
        with pytest.raises(ValueError, match="No filters for wavelet 'db11'.*db10"):
            _filter_bank("db11")


class TestInverseTransform:
    @pytest.mark.parametrize("name", ["db1", "db2", "db4", "sym4", "sym7", "coif1", "coif2"])
    def test_the_inverse_of_the_analysis_is_the_matrix(self, name: str) -> None:
        bank = _filter_bank(name)
        lucid.manual_seed(0)
        side = 2 * bank.length  # even, and a few filter lengths wide
        x = lucid.randn(side, side + 2).tolist()
        bands = forms.analysis_2d(x, bank.dec_lo, bank.dec_hi)
        back = _idwt2(*(lucid.tensor(b) for b in bands), name)
        assert tuple(back.shape) == (side, side + 2)
        gap = max(
            abs(a - b)
            for ra, rb in zip(back.tolist(), x, strict=True)
            for a, b in zip(ra, rb, strict=True)
        )
        assert gap <= 1e-5

    @pytest.mark.parametrize("name", ["db1", "db3", "sym2", "sym5", "coif1"])
    def test_the_inverse_is_the_synthesis_sums(self, name: str) -> None:
        bank = _filter_bank(name)
        taps = bank.length
        lucid.manual_seed(1)
        height, width = taps + 1, taps + 3
        bands = tuple(lucid.randn(height, width) for _ in range(4))
        mine = _idwt2(*bands, name)
        expected = forms.synthesis_2d(
            tuple(b.tolist() for b in bands),
            bank.rec_lo,
            bank.rec_hi,  # type: ignore[arg-type]
        )
        assert tuple(mine.shape) == (2 * height - taps + 2, 2 * width - taps + 2)
        gap = max(
            abs(a - b)
            for ra, rb in zip(mine.tolist(), expected, strict=True)
            for a, b in zip(ra, rb, strict=True)
        )
        assert gap <= 1e-5

    @pytest.mark.parametrize("name", ["db2", "sym4", "coif1"])
    def test_the_inverse_is_linear_in_the_bands_and_differentiable(self, name: str) -> None:
        lucid.manual_seed(2)
        taps = _filter_bank(name).length
        bands = [lucid.randn(taps, taps, requires_grad=True) for _ in range(4)]
        out = _idwt2(*bands, name)
        out.sum().backward()
        for band in bands:
            assert band.grad is not None and bool(band.grad.any().item())
        zero = lucid.zeros(taps, taps)
        assert not bool(_idwt2(zero, zero, zero, zero, name).any().item())

    @pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
    def test_the_dtype_follows_the_bands(self, dtype: lucid.dtype) -> None:
        bands = [lucid.ones(6, 6, dtype=dtype) for _ in range(4)]
        assert _idwt2(*bands, "db2").dtype == dtype

    def test_the_inverse_runs_on_metal_as_on_the_cpu(self, device: str) -> None:
        lucid.manual_seed(3)
        bands = [lucid.randn(7, 7) for _ in range(4)]
        on_cpu = _idwt2(*bands, "sym4")
        there = _idwt2(*(b.to(device) for b in bands), "sym4")
        assert float((on_cpu - there.to("cpu")).abs().max().item()) <= 1e-5


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.usefixtures("ref")
def test_the_tables_match_the_reference_where_its_tables_are_right() -> None:
    """Every family but ``coif4`` and ``coif5``, whose reference tables break the orthonormality they must satisfy."""
    adapter_library()
    from peft.tuners.waveft.wavelet import Wavelet  # the reference's own table, read for comparison

    for name in WAVELET_FAMILIES:
        if name in ("coif4", "coif5"):
            continue
        theirs = Wavelet(name)
        bank = _filter_bank(name)
        for ours, other in (
            (bank.dec_lo, theirs.dec_lo),
            (bank.dec_hi, theirs.dec_hi),
            (bank.rec_lo, theirs.rec_lo),
            (bank.rec_hi, theirs.rec_hi),
        ):
            assert len(ours) == len(other), name
            assert max(abs(a - b) for a, b in zip(ours, other, strict=True)) <= 1e-10, name


@pytest.mark.parity
@pytest.mark.usefixtures("ref")
def test_the_reference_tables_that_differ_are_the_ones_that_violate_the_definitions() -> None:
    adapter_library()
    from peft.tuners.waveft.wavelet import Wavelet

    for name in ("coif4", "coif5"):
        theirs = tuple(Wavelet(name).rec_lo)
        assert abs(sum(theirs) - math.sqrt(2)) > 1e-3 or len(theirs) != _length(name)
        assert abs(sum(_SCALING[name]) - math.sqrt(2)) <= 1e-14


@pytest.mark.parity
@pytest.mark.parametrize("name", ["db1", "db2", "db4", "sym4", "sym8", "coif1", "coif3"])
def test_the_inverse_transform_is_the_references(ref: ModuleType, name: str) -> None:
    adapter_library()
    from peft.tuners.waveft.waverec2d import waverec2d

    lucid.manual_seed(4)
    taps = _filter_bank(name).length
    bands = [lucid.randn(taps + 3, taps + 5) for _ in range(4)]
    theirs = waverec2d(
        (ref.tensor(bands[0].tolist()), tuple(ref.tensor(b.tolist()) for b in bands[1:])), name
    )
    mine = _idwt2(*bands, name)
    assert tuple(mine.shape) == tuple(theirs.shape)
    assert float((ref.tensor(mine.tolist()) - theirs).abs().max()) <= 1e-6
