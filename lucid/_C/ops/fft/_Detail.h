// lucid/_C/ops/fft/_Detail.h
//
// Internal helpers shared across all FFT ops.  Mirrors
// ``ops/linalg/_Detail.h``: header-only inline helpers, MLX interop,
// double-precision Accelerate vDSP FFT kernels, output-shape arithmetic,
// and dtype guards.  Nothing in this header is part of the public API.
//
// Architecture note (carve-out)
// -----------------------------
// Single-precision / half FFTs bypass ``IBackend`` dispatch and call
// ``mlx::core::fft`` directly from the ops layer, mirroring the linalg
// pattern (``as_mlx_array_gpu`` + ``kMlxLinalgStream``).  For FFT
// specifically, MLX wraps Apple Accelerate vDSP on the CPU stream, so
// dispatching MLX from both ``Device::CPU`` and ``Device::GPU`` paths is
// consistent with the H3 spirit ("CPU = Accelerate, GPU = MLX") —
// Accelerate is reached transitively through MLX.
//
// MLX's FFT is single-precision only (``F16``/``BF16``/``F32``/``C64``),
// so ``F64`` and ``C128`` on ``Device::CPU`` call Apple Accelerate vDSP's
// double-precision FFT routines (``vDSP_fft_zopD`` / ``vDSP_fft_zipD``)
// directly (LCD-458).

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <Accelerate/Accelerate.h>
#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/fft.h>
#include <mlx/ops.h>

#include "../../autograd/Helpers.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/PhantomMode.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../core/fwd.h"
#include "../utils/Promote.h"

namespace lucid::fft_detail {

using ::lucid::helpers::fresh;

// MLX stream constant used for single-precision FFT dispatch.
//
// Notes
// -----
// MLX has no Metal FFT kernel on Apple Silicon; it routes through
// Accelerate vDSP on the CPU stream.  GPU tensors continue to hold
// data in GPU-accessible memory — only the dispatch lane is CPU, and
// MLX handles the host/device transfer transparently.
inline const ::mlx::core::Device kMlxFftStream{::mlx::core::Device::cpu};

// Resolve the output dtype for ``fftn`` / ``ifftn``.
//
// Returns
// -------
// Dtype
//     ``Dtype::C128`` for ``F64`` / ``C128`` on CPU, ``Dtype::C64`` for
//     ``F16`` / ``BF16`` / ``F32`` / ``C64``.
//
// Raises
// ------
// LucidError
//     If ``in`` is not a floating-point or complex dtype, or if a double-
//     precision dtype is requested on ``Device::GPU``.
inline Dtype dtype_for_complex_fft(Dtype in, Device dev) {
    switch (in) {
    case Dtype::C64:
    case Dtype::F32:
    case Dtype::F16:
        return Dtype::C64;
    case Dtype::C128:
    case Dtype::F64:
        if (dev == Device::GPU)
            ErrorBuilder("fft").not_implemented(
                "fftn/ifftn does not support F64/C128 on Metal (GPU)");
        return Dtype::C128;
    case Dtype::BF16:
    default:
        ErrorBuilder("fft").not_implemented(
            "fftn/ifftn requires a floating-point or complex input, got " +
            std::string(dtype_name(in)));
        return Dtype::C64;
    }
}

// Resolve the output dtype for ``rfftn``.
//
// Returns
// -------
// Dtype
//     ``Dtype::C128`` for ``F64`` on CPU, ``Dtype::C64`` for
//     ``F16`` / ``BF16`` / ``F32``.
//
// Raises
// ------
// LucidError
//     If ``in`` is not a real floating-point dtype, or if ``F64`` is
//     requested on ``Device::GPU``.
inline Dtype dtype_for_rfft(Dtype in, Device dev) {
    switch (in) {
    case Dtype::F32:
    case Dtype::F16:
        return Dtype::C64;
    case Dtype::F64:
        if (dev == Device::GPU)
            ErrorBuilder("rfftn").not_implemented("rfftn does not support F64 on Metal (GPU)");
        return Dtype::C128;
    case Dtype::BF16:
    default:
        ErrorBuilder("rfftn").not_implemented("rfftn requires a real floating-point input, got " +
                                              std::string(dtype_name(in)));
        return Dtype::C64;
    }
}

// Resolve the output dtype for ``irfftn``.
//
// Returns
// -------
// Dtype
//     ``Dtype::F64`` for ``C128`` on CPU, ``Dtype::F32`` for ``C64``.
//
// Raises
// ------
// LucidError
//     If ``in`` is not ``C64`` or ``C128``, or if ``C128`` is requested
//     on ``Device::GPU``.
inline Dtype dtype_for_irfft(Dtype in, Device dev) {
    switch (in) {
    case Dtype::C64:
        return Dtype::F32;
    case Dtype::C128:
        if (dev == Device::GPU)
            ErrorBuilder("irfftn").not_implemented("irfftn does not support C128 on Metal (GPU)");
        return Dtype::F64;
    default:
        ErrorBuilder("irfftn").not_implemented("irfftn requires C64/C128 input, got " +
                                               std::string(dtype_name(in)));
        return Dtype::F32;
    }
}

// Normalise a possibly-negative axis list to non-negative form, in-place,
// through the one dim rule (:func:`distinct_dims`).
//
// Parameters
// ----------
// axes : vector<int>&
//     Axes to rewrite.  Each negative axis $a$ becomes $a + r$ where
//     $r$ is the input rank.
// rank : int
//     Input rank used as the modulus.
// op : const char*
//     Op name embedded into the error message on failure.
//
// Raises
// ------
// IndexError
//     If any axis falls outside $[-r, r)$.
// LucidError
//     If two axes name the same dimension.  MLX would refuse it with its
//     own ``ValueError``.
inline void normalise_axes(std::vector<int>& axes, int rank, const char* op) {
    axes = distinct_dims(axes, rank, op);
}

// Populate ``axes`` with ``[0, 1, ..., rank-1]`` when empty.
//
// Notes
// -----
// MLX's ``fftn`` accepts an empty axis list with the same meaning, but
// Lucid pre-fills so subsequent shape arithmetic (output-shape helpers
// below) can index ``axes`` directly without branching.
inline void default_axes_all(std::vector<int>& axes, int rank) {
    if (!axes.empty())
        return;
    axes.reserve(rank);
    for (int i = 0; i < rank; ++i)
        axes.push_back(i);
}

// Compute the output shape for a complex FFT (``fftn`` / ``ifftn``).
//
// Parameters
// ----------
// in_shape : Shape
//     Input shape.
// n : vector<int64_t>
//     Per-axis transform lengths.  Empty means "preserve input shape".
// axes : vector<int>
//     Already normalised, non-empty (when ``n`` is non-empty).
// op : const char*
//     Op name embedded into the error message.
//
// Returns
// -------
// Shape
//     ``in_shape`` with each ``axes[i]`` overwritten by ``n[i]``, or
//     ``in_shape`` unchanged when ``n`` is empty.
//
// Raises
// ------
// LucidError
//     If ``len(n) != len(axes)`` when ``n`` is non-empty.
inline std::vector<std::int64_t> resolve_fft_lengths(const Shape& in_shape,
                                                     const std::vector<std::int64_t>& n,
                                                     const std::vector<int>& axes,
                                                     bool is_irfft,
                                                     const char* op) {
    if (!n.empty() && n.size() != axes.size())
        ErrorBuilder(op).fail("len(s) must match len(axes)");
    std::vector<std::int64_t> lengths(axes.size());
    for (std::size_t i = 0; i < axes.size(); ++i) {
        const auto ax = static_cast<std::size_t>(axes[i]);
        std::int64_t len = n.empty() ? in_shape[ax] : n[i];
        if (n.empty() && is_irfft && i + 1 == axes.size())
            len = (in_shape[ax] - 1) * 2;
        if (len <= 0) {
            ErrorBuilder(op).invalid_argument("Invalid number of data points (" +
                                              std::to_string(len) + ") specified");
        }
        lengths[i] = len;
    }
    return lengths;
}

inline Shape complex_fft_out_shape(const Shape& in_shape,
                                   const std::vector<std::int64_t>& n,
                                   const std::vector<int>& axes,
                                   const char* op) {
    const auto lengths = resolve_fft_lengths(in_shape, n, axes, false, op);
    Shape out = in_shape;
    for (std::size_t i = 0; i < axes.size(); ++i)
        out[static_cast<std::size_t>(axes[i])] = lengths[i];
    return out;
}

inline Shape rfft_out_shape(const Shape& in_shape,
                            const std::vector<std::int64_t>& n,
                            const std::vector<int>& axes,
                            const char* op) {
    if (axes.empty())
        ErrorBuilder(op).fail("rfftn requires at least one axis");
    const auto lengths = resolve_fft_lengths(in_shape, n, axes, false, op);
    Shape out = in_shape;
    for (std::size_t i = 0; i + 1 < axes.size(); ++i)
        out[static_cast<std::size_t>(axes[i])] = lengths[i];
    const int last = axes.back();
    out[static_cast<std::size_t>(last)] = lengths.back() / 2 + 1;
    return out;
}

inline Shape irfft_out_shape(const Shape& in_shape,
                             const std::vector<std::int64_t>& n,
                             const std::vector<int>& axes,
                             const char* op) {
    if (axes.empty())
        ErrorBuilder(op).fail("irfftn requires at least one axis");
    const auto lengths = resolve_fft_lengths(in_shape, n, axes, true, op);
    Shape out = in_shape;
    for (std::size_t i = 0; i < axes.size(); ++i)
        out[static_cast<std::size_t>(axes[i])] = lengths[i];
    return out;
}

inline ::mlx::core::Shape mlx_n_from_lucid(const std::vector<std::int64_t>& n,
                                           const char* op = "fft") {
    ::mlx::core::Shape out;
    out.reserve(n.size());
    for (auto v : n) {
        if (v <= 0) {
            ErrorBuilder(op).invalid_argument("Invalid number of data points (" +
                                              std::to_string(v) + ") specified");
        }
        if (v > std::numeric_limits<std::int32_t>::max())
            ErrorBuilder(op).fail("FFT length exceeds INT32_MAX");
        out.push_back(static_cast<std::int32_t>(v));
    }
    return out;
}

inline FFTSetupD cached_fft_setup_d(vDSP_Length log2n) {
    static std::mutex mu;
    static FFTSetupD setups[32] = {};
    static vDSP_Length max_log2n = 0;
    std::lock_guard<std::mutex> lock(mu);
    if (log2n == 0)
        return nullptr;
    if (setups[log2n] == nullptr) {
        if (log2n > max_log2n) {
            FFTSetupD prev = max_log2n > 0 ? setups[max_log2n] : nullptr;
            setups[log2n] = vDSP_create_fftsetupD(log2n, kFFTRadix2);
            if (prev == nullptr)
                max_log2n = log2n;
        } else {
            setups[log2n] = setups[max_log2n];
        }
    }
    return setups[log2n];
}

struct Fft1DPlanD {
    std::int64_t n = 0;
    bool inverse = false;
    bool is_pow2 = false;
    vDSP_Length log2n = 0;
    vDSP_Length log2m = 0;
    std::int64_t m = 0;
    FFTSetupD setup = nullptr;
    std::vector<double> chirp_re;
    std::vector<double> chirp_im;
    std::vector<double> b_hat_re;
    std::vector<double> b_hat_im;

    Fft1DPlanD(std::int64_t length, bool inv) : n(length), inverse(inv) {
        if (n <= 1) {
            is_pow2 = true;
            return;
        }
        if ((n & (n - 1)) == 0) {
            is_pow2 = true;
            while ((std::int64_t{1} << log2n) < n)
                ++log2n;
            setup = cached_fft_setup_d(log2n);
            return;
        }
        is_pow2 = false;
        m = 1;
        log2m = 0;
        while (m < 2 * n - 1) {
            m <<= 1;
            ++log2m;
        }
        setup = cached_fft_setup_d(log2m);
        chirp_re.resize(static_cast<std::size_t>(n));
        chirp_im.resize(static_cast<std::size_t>(n));
        b_hat_re.assign(static_cast<std::size_t>(m), 0.0);
        b_hat_im.assign(static_cast<std::size_t>(m), 0.0);

        const double sign = inverse ? 1.0 : -1.0;
        const double pi_over_n = 3.141592653589793238462643383279502884 / static_cast<double>(n);
        const std::uint64_t two_n = static_cast<std::uint64_t>(n) * 2ULL;
        for (std::int64_t j = 0; j < n; ++j) {
            const std::uint64_t j_mod = static_cast<std::uint64_t>(j) % two_n;
            const std::uint64_t j2_mod = (j_mod * j_mod) % two_n;
            const double angle = sign * pi_over_n * static_cast<double>(j2_mod);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            const auto uj = static_cast<std::size_t>(j);
            chirp_re[uj] = c;
            chirp_im[uj] = s;
            b_hat_re[uj] = c;
            b_hat_im[uj] = -s;
            if (j > 0) {
                const auto wrap = static_cast<std::size_t>(m - j);
                b_hat_re[wrap] = c;
                b_hat_im[wrap] = -s;
            }
        }
        DSPDoubleSplitComplex b_split{b_hat_re.data(), b_hat_im.data()};
        vDSP_fft_zipD(setup, &b_split, 1, log2m, kFFTDirection_Forward);
    }

    void execute(const std::complex<double>* in,
                 std::int64_t in_len,
                 std::int64_t in_stride,
                 std::complex<double>* out,
                 std::int64_t out_stride,
                 std::vector<double>& work_re,
                 std::vector<double>& work_im) const {
        const std::int64_t copy_len = std::min(in_len, n);
        if (n == 1) {
            out[0] = copy_len > 0 ? in[0] : std::complex<double>(0.0, 0.0);
            return;
        }
        if (is_pow2) {
            for (std::int64_t j = 0; j < copy_len; ++j) {
                const auto v = in[j * in_stride];
                work_re[static_cast<std::size_t>(j)] = v.real();
                work_im[static_cast<std::size_t>(j)] = v.imag();
            }
            for (std::int64_t j = copy_len; j < n; ++j) {
                work_re[static_cast<std::size_t>(j)] = 0.0;
                work_im[static_cast<std::size_t>(j)] = 0.0;
            }
            DSPDoubleSplitComplex split{work_re.data(), work_im.data()};
            vDSP_fft_zipD(setup, &split, 1, log2n,
                          inverse ? kFFTDirection_Inverse : kFFTDirection_Forward);
            const double scale = inverse ? (1.0 / static_cast<double>(n)) : 1.0;
            for (std::int64_t k = 0; k < n; ++k) {
                out[k * out_stride] =
                    std::complex<double>(work_re[static_cast<std::size_t>(k)] * scale,
                                         work_im[static_cast<std::size_t>(k)] * scale);
            }
            return;
        }

        for (std::int64_t j = 0; j < copy_len; ++j) {
            const auto v = in[j * in_stride];
            const auto uj = static_cast<std::size_t>(j);
            const double cr = chirp_re[uj];
            const double ci = chirp_im[uj];
            work_re[uj] = v.real() * cr - v.imag() * ci;
            work_im[uj] = v.real() * ci + v.imag() * cr;
        }
        std::fill(work_re.begin() + copy_len, work_re.begin() + m, 0.0);
        std::fill(work_im.begin() + copy_len, work_im.begin() + m, 0.0);

        DSPDoubleSplitComplex a_split{work_re.data(), work_im.data()};
        vDSP_fft_zipD(setup, &a_split, 1, log2m, kFFTDirection_Forward);

        for (std::int64_t k = 0; k < m; ++k) {
            const auto uk = static_cast<std::size_t>(k);
            const double ar = work_re[uk];
            const double ai = work_im[uk];
            const double br = b_hat_re[uk];
            const double bi = b_hat_im[uk];
            work_re[uk] = ar * br - ai * bi;
            work_im[uk] = ar * bi + ai * br;
        }
        vDSP_fft_zipD(setup, &a_split, 1, log2m, kFFTDirection_Inverse);

        const double scale = inverse ? (1.0 / (static_cast<double>(m) * static_cast<double>(n)))
                                     : (1.0 / static_cast<double>(m));
        for (std::int64_t k = 0; k < n; ++k) {
            const auto uk = static_cast<std::size_t>(k);
            const double cr = work_re[uk] * scale;
            const double ci = work_im[uk] * scale;
            const double wr = chirp_re[uk];
            const double wi = chirp_im[uk];
            out[k * out_stride] = std::complex<double>(cr * wr - ci * wi, cr * wi + ci * wr);
        }
    }
};

inline void fft_1d_axis_c128(const std::complex<double>* in_ptr,
                             const Shape& in_shape,
                             std::complex<double>* out_ptr,
                             const Shape& out_shape,
                             int axis,
                             bool inverse) {
    const auto ax = static_cast<std::size_t>(axis);
    const std::int64_t in_len = in_shape[ax];
    const std::int64_t out_len = out_shape[ax];
    std::int64_t outer = 1;
    for (std::size_t i = 0; i < ax; ++i)
        outer *= in_shape[i];
    std::int64_t inner = 1;
    for (std::size_t i = ax + 1; i < in_shape.size(); ++i)
        inner *= in_shape[i];

    Fft1DPlanD plan(out_len, inverse);
    const std::size_t scratch_len = static_cast<std::size_t>(plan.is_pow2 ? out_len : plan.m);
    std::vector<double> work_re(scratch_len);
    std::vector<double> work_im(scratch_len);

    for (std::int64_t o = 0; o < outer; ++o) {
        const std::complex<double>* in_outer = in_ptr + o * in_len * inner;
        std::complex<double>* out_outer = out_ptr + o * out_len * inner;
        for (std::int64_t i = 0; i < inner; ++i) {
            plan.execute(in_outer + i, in_len, inner, out_outer + i, inner, work_re, work_im);
        }
    }
}

inline std::vector<std::complex<double>> copy_f64_or_c128_input(const TensorImplPtr& a,
                                                                const char* op) {
    const auto& cur_cs = std::get<::lucid::CpuStorage>(a->storage());
    const std::size_t cur_n = shape_numel(a->shape());
    std::vector<std::complex<double>> buf(cur_n);
    const Dtype dt = a->dtype();
    if (dt == Dtype::F64) {
        const auto* src = reinterpret_cast<const double*>(cur_cs.ptr.get());
        for (std::size_t i = 0; i < cur_n; ++i)
            buf[i] = std::complex<double>(src[i], 0.0);
    } else if (dt == Dtype::C128) {
        const auto* src = reinterpret_cast<const std::complex<double>*>(cur_cs.ptr.get());
        if (cur_n > 0)
            std::copy(src, src + cur_n, buf.begin());
    } else {
        ErrorBuilder(op).not_implemented("CPU double-precision FFT requires float64 or complex128");
    }
    return buf;
}

inline void run_fft_axes_c128(std::vector<std::complex<double>>& buf_a,
                              Shape& cur_shape,
                              const std::vector<std::int64_t>& lengths,
                              const std::vector<int>& axes,
                              std::size_t num_axes,
                              bool inverse) {
    std::vector<std::complex<double>> buf_b;
    for (std::size_t idx = 0; idx < num_axes; ++idx) {
        const int ax = axes[idx];
        Shape next_shape = cur_shape;
        next_shape[static_cast<std::size_t>(ax)] = lengths[idx];
        buf_b.resize(shape_numel(next_shape));
        if (!buf_a.empty() && !buf_b.empty()) {
            fft_1d_axis_c128(buf_a.data(), cur_shape, buf_b.data(), next_shape, ax, inverse);
        } else {
            std::fill(buf_b.begin(), buf_b.end(), std::complex<double>(0.0, 0.0));
        }
        buf_a.swap(buf_b);
        cur_shape = std::move(next_shape);
    }
}

inline Storage cpu_complex_fftn_f64(const TensorImplPtr& a,
                                    const std::vector<std::int64_t>& n,
                                    const std::vector<int>& axes,
                                    const Shape& out_shape,
                                    bool inverse) {
    const char* op = inverse ? "ifftn" : "fftn";
    const auto lengths = resolve_fft_lengths(a->shape(), n, axes, false, op);
    for (std::int64_t len : lengths) {
        if (len <= 0) {
            ErrorBuilder(op).invalid_argument("Invalid number of data points (" +
                                              std::to_string(len) + ") specified");
        }
    }
    auto buf_a = copy_f64_or_c128_input(a, op);
    Shape cur_shape = a->shape();
    run_fft_axes_c128(buf_a, cur_shape, lengths, axes, axes.size(), inverse);

    auto out_cs = ::lucid::helpers::allocate_cpu(out_shape, Dtype::C128);
    if (!buf_a.empty()) {
        std::copy(buf_a.begin(), buf_a.end(),
                  reinterpret_cast<std::complex<double>*>(out_cs.ptr.get()));
    }
    return Storage{std::move(out_cs)};
}

inline Storage cpu_rfftn_f64(const TensorImplPtr& a,
                             const std::vector<std::int64_t>& n,
                             const std::vector<int>& axes,
                             const Shape& out_shape) {
    const auto lengths = resolve_fft_lengths(a->shape(), n, axes, false, "rfftn");
    Shape full_shape = a->shape();
    for (std::size_t i = 0; i < axes.size(); ++i)
        full_shape[static_cast<std::size_t>(axes[i])] = lengths[i];

    Storage full_st = cpu_complex_fftn_f64(a, lengths, axes, full_shape, /*inverse=*/false);
    const auto& full_cs = std::get<::lucid::CpuStorage>(full_st);
    const auto* full_ptr = reinterpret_cast<const std::complex<double>*>(full_cs.ptr.get());

    const auto last_ax = static_cast<std::size_t>(axes.back());
    const std::int64_t full_last = full_shape[last_ax];
    const std::int64_t half_last = out_shape[last_ax];
    std::int64_t outer = 1;
    for (std::size_t i = 0; i < last_ax; ++i)
        outer *= out_shape[i];
    std::int64_t inner = 1;
    for (std::size_t i = last_ax + 1; i < out_shape.size(); ++i)
        inner *= out_shape[i];

    auto out_cs = ::lucid::helpers::allocate_cpu(out_shape, Dtype::C128);
    auto* out_ptr = reinterpret_cast<std::complex<double>*>(out_cs.ptr.get());
    if (out_ptr != nullptr && full_ptr != nullptr) {
        for (std::int64_t o = 0; o < outer; ++o) {
            const auto* src_o = full_ptr + o * full_last * inner;
            auto* dst_o = out_ptr + o * half_last * inner;
            std::copy(src_o, src_o + half_last * inner, dst_o);
        }
    }
    return Storage{std::move(out_cs)};
}

inline void expand_hermitian_slice_c128(const std::complex<double>* src_o,
                                        std::complex<double>* dst_o,
                                        std::int64_t n_last,
                                        std::int64_t copy_bins,
                                        std::int64_t inner) {
    for (std::int64_t k = 0; k < copy_bins; ++k) {
        std::copy(src_o + k * inner, src_o + (k + 1) * inner, dst_o + k * inner);
    }
    if (copy_bins > 0) {
        for (std::int64_t i = 0; i < inner; ++i)
            dst_o[i] = std::complex<double>(dst_o[i].real(), 0.0);
    }
    if (n_last % 2 == 0 && (n_last / 2) < copy_bins) {
        const std::int64_t nyq = n_last / 2;
        for (std::int64_t i = 0; i < inner; ++i)
            dst_o[nyq * inner + i] = std::complex<double>(dst_o[nyq * inner + i].real(), 0.0);
    }
    const std::int64_t mirror_bins = std::min(copy_bins, (n_last + 1) / 2);
    for (std::int64_t k = 1; k < mirror_bins; ++k) {
        const std::int64_t dst_k = n_last - k;
        for (std::int64_t i = 0; i < inner; ++i)
            dst_o[dst_k * inner + i] = std::conj(dst_o[k * inner + i]);
    }
}

inline Storage cpu_irfftn_c128(const TensorImplPtr& a,
                               const std::vector<std::int64_t>& n,
                               const std::vector<int>& axes,
                               const Shape& out_shape) {
    const auto lengths = resolve_fft_lengths(a->shape(), n, axes, true, "irfftn");
    auto buf_a = copy_f64_or_c128_input(a, "irfftn");
    Shape cur_shape = a->shape();
    run_fft_axes_c128(buf_a, cur_shape, lengths, axes, axes.size() - 1, /*inverse=*/true);

    const int last_ax_int = axes.back();
    const auto last_ax = static_cast<std::size_t>(last_ax_int);
    const std::int64_t n_last = lengths.back();
    const std::int64_t in_last = cur_shape[last_ax];
    const std::int64_t copy_bins = std::min(in_last, n_last / 2 + 1);
    std::int64_t outer = 1;
    for (std::size_t i = 0; i < last_ax; ++i)
        outer *= cur_shape[i];
    std::int64_t inner = 1;
    for (std::size_t i = last_ax + 1; i < cur_shape.size(); ++i)
        inner *= cur_shape[i];

    Shape full_last_shape = cur_shape;
    full_last_shape[last_ax] = n_last;
    std::vector<std::complex<double>> buf_b(shape_numel(full_last_shape),
                                            std::complex<double>(0.0, 0.0));
    for (std::int64_t o = 0; o < outer; ++o) {
        expand_hermitian_slice_c128(buf_a.data() + o * in_last * inner,
                                    buf_b.data() + o * n_last * inner, n_last, copy_bins, inner);
    }

    buf_a.resize(shape_numel(out_shape));
    if (!buf_b.empty() && !buf_a.empty()) {
        fft_1d_axis_c128(buf_b.data(), full_last_shape, buf_a.data(), out_shape, last_ax_int,
                         /*inverse=*/true);
    }

    auto out_cs = ::lucid::helpers::allocate_cpu(out_shape, Dtype::F64);
    auto* out_ptr = reinterpret_cast<double*>(out_cs.ptr.get());
    for (std::size_t i = 0; i < buf_a.size(); ++i)
        out_ptr[i] = buf_a[i].real();
    return Storage{std::move(out_cs)};
}

// Wrap an MLX FFT result as a Lucid ``Storage`` on the target device.
//
// Parameters
// ----------
// out : mlx::core::array&&
//     The freshly produced FFT result (will be made contiguous before
//     wrapping).
// out_dtype : Dtype
//     Lucid-side dtype tag.
// out_shape : Shape
//     Logical Lucid shape for the result.
// dev : Device
//     Target device.  ``GPU`` wraps in-place; ``CPU`` downloads to host.
//
// Returns
// -------
// Storage
//     A ``GpuStorage`` wrapping the MLX array, or a ``CpuStorage``
//     containing a downloaded copy.
inline Storage
finalise_result(::mlx::core::array&& out, Dtype out_dtype, const Shape& out_shape, Device dev) {
    if (dev == Device::GPU)
        return Storage{::lucid::gpu::wrap_mlx_array(::mlx::core::contiguous(out), out_dtype)};
    auto wrapped = ::lucid::gpu::wrap_mlx_array(::mlx::core::contiguous(out), out_dtype);
    return Storage{::lucid::gpu::download_gpu_to_cpu(wrapped, out_shape)};
}

// Materialise a Lucid input tensor as an ``mlx::core::array``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (CPU or GPU).
//
// Returns
// -------
// mlx::core::array
//     For GPU tensors the underlying MLX array is returned by value
//     (cheap shared handle).  For CPU tensors the buffer is uploaded
//     via ``MlxBridge`` (one host-to-device copy).
inline ::mlx::core::array as_mlx_input(const TensorImplPtr& a) {
    if (a->device() == Device::GPU) {
        const auto& g = std::get<::lucid::GpuStorage>(a->storage());
        return *g.arr;
    }
    const auto& cs = std::get<::lucid::CpuStorage>(a->storage());
    auto gs = ::lucid::gpu::upload_cpu_to_gpu(cs, a->shape());
    return *gs.arr;
}

template <typename DtypeFn, typename ShapeFn, typename DoubleFn, typename MlxFn>
inline TensorImplPtr run_nd_fft(const char* op_name,
                                const char* input_label,
                                const TensorImplPtr& a_in,
                                const std::vector<std::int64_t>& n,
                                const std::vector<int>& axes_in,
                                Dtype double_out_dtype,
                                DtypeFn&& resolve_dtype,
                                ShapeFn&& resolve_shape,
                                DoubleFn&& run_double_cpu,
                                MlxFn&& run_mlx) {
    Validator::input(a_in, input_label).non_null();
    const AmpScope amp{op_name, a_in->device()};
    const TensorImplPtr a = amp.cast(a_in);
    OpScopeFull scope{op_name, a->device(), a->dtype(), a->shape()};

    const Dtype out_dtype = resolve_dtype(a->dtype(), a->device());
    const int rank = static_cast<int>(a->shape().size());
    if (rank == 0)
        ErrorBuilder(op_name).invalid_argument("input must be at least 1-D");

    std::vector<int> axes = axes_in;
    default_axes_all(axes, rank);
    normalise_axes(axes, rank, op_name);

    const Shape out_shape = resolve_shape(a->shape(), n, axes, op_name);

    Storage out;
    if (PhantomMode::is_enabled()) {
        out = Storage{
            PhantomStorage{shape_numel(out_shape) * dtype_size(out_dtype), out_dtype, a->device()}};
    } else if (out_dtype == double_out_dtype) {
        out = run_double_cpu(a, axes, out_shape);
    } else {
        auto in_arr = as_mlx_input(a);
        ::mlx::core::array out_arr = run_mlx(in_arr, axes);
        out = finalise_result(std::move(out_arr), out_dtype, out_shape, a->device());
    }
    auto result = fresh(std::move(out), out_shape, out_dtype, a->device());
    // Record the trace I/O explicitly.  Without it the node keeps the
    // empty input list ``on_op_enter`` seeded and its result never
    // becomes a traced tensor, so the builder drops the node as dead and
    // the *consumer* sees the result as a fresh external feed — bound
    // once, at trace time.  The compiled model then reported success and
    // returned that first answer for every later input.
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a}, result);
    return result;
}

}  // namespace lucid::fft_detail
