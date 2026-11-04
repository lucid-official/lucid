// lucid/_C/ops/fft/Fftn.cpp
//
// Forward N-dimensional DFT via mlx::core::fft::fftn.  Routes both Device::CPU
// and Device::GPU through MLX (FFT carve-out — see _Detail.h).  Autograd is
// not wired here; the Python lucid.fft module wraps this op in a
// lucid.autograd.Function whose backward calls ifftn_op.

#include "Fftn.h"

#include "_Detail.h"

namespace lucid {

TensorImplPtr fftn_op(const TensorImplPtr& a,
                      const std::vector<std::int64_t>& n,
                      const std::vector<int>& axes_in) {
    using ::mlx::core::fft::FFTNorm;
    return fft_detail::run_nd_fft(
        "fftn", "fftn.a", a, n, axes_in, fft_detail::dtype_for_complex_fft,
        fft_detail::complex_fft_out_shape,
        [&](const TensorImplPtr& x, const std::vector<int>& axes, const Shape& out_shape,
            Dtype out_dtype) {
            return out_dtype == Dtype::C128
                       ? fft_detail::cpu_complex_fftn_f64(x, n, axes, out_shape, /*inverse=*/false)
                       : fft_detail::cpu_complex_fftn_f32(x, n, axes, out_shape, /*inverse=*/false);
        },
        [&](const ::mlx::core::array& in_arr, const std::vector<int>& axes) {
            return n.empty()
                       ? ::mlx::core::fft::fftn(in_arr, axes, FFTNorm::Backward,
                                                fft_detail::kMlxFftStream)
                       : ::mlx::core::fft::fftn(in_arr, fft_detail::mlx_n_from_lucid(n, "fftn"),
                                                axes, FFTNorm::Backward, fft_detail::kMlxFftStream);
        });
}

}  // namespace lucid
