// lucid/_C/ops/fft/Ifftn.cpp
//
// Forward N-dimensional inverse DFT via mlx::core::fft::ifftn.

#include "Ifftn.h"

#include "_Detail.h"

namespace lucid {

TensorImplPtr ifftn_op(const TensorImplPtr& a,
                       const std::vector<std::int64_t>& n,
                       const std::vector<int>& axes_in) {
    using ::mlx::core::fft::FFTNorm;
    return fft_detail::run_nd_fft(
        "ifftn", "ifftn.a", a, n, axes_in, fft_detail::dtype_for_complex_fft,
        fft_detail::complex_fft_out_shape,
        [&](const TensorImplPtr& x, const std::vector<int>& axes, const Shape& out_shape,
            Dtype out_dtype) {
            return out_dtype == Dtype::C128
                       ? fft_detail::cpu_complex_fftn_f64(x, n, axes, out_shape, /*inverse=*/true)
                       : fft_detail::cpu_complex_fftn_f32(x, n, axes, out_shape, /*inverse=*/true);
        },
        [&](const ::mlx::core::array& in_arr, const std::vector<int>& axes) {
            return n.empty()
                       ? ::mlx::core::fft::ifftn(in_arr, axes, FFTNorm::Backward,
                                                 fft_detail::kMlxFftStream)
                       : ::mlx::core::fft::ifftn(in_arr, fft_detail::mlx_n_from_lucid(n, "ifftn"),
                                                 axes, FFTNorm::Backward,
                                                 fft_detail::kMlxFftStream);
        });
}

}  // namespace lucid
