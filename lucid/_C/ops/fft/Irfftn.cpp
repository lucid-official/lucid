// lucid/_C/ops/fft/Irfftn.cpp
//
// Forward N-dimensional inverse real DFT via mlx::core::fft::irfftn.

#include "Irfftn.h"

#include "_Detail.h"

namespace lucid {

TensorImplPtr irfftn_op(const TensorImplPtr& a,
                        const std::vector<std::int64_t>& n,
                        const std::vector<int>& axes_in) {
    using ::mlx::core::fft::FFTNorm;
    return fft_detail::run_nd_fft(
        "irfftn", "irfftn.a", a, n, axes_in, fft_detail::dtype_for_irfft,
        fft_detail::irfft_out_shape,
        [&](const TensorImplPtr& x, const std::vector<int>& axes, const Shape& out_shape,
            Dtype out_dtype) {
            return out_dtype == Dtype::F64 ? fft_detail::cpu_irfftn_c128(x, n, axes, out_shape)
                                           : fft_detail::cpu_irfftn_c64(x, n, axes, out_shape);
        },
        [&](const ::mlx::core::array& in_arr, const std::vector<int>& axes) {
            return n.empty()
                       ? ::mlx::core::fft::irfftn(in_arr, axes, FFTNorm::Backward,
                                                  fft_detail::kMlxFftStream)
                       : ::mlx::core::fft::irfftn(in_arr, fft_detail::mlx_n_from_lucid(n, "irfftn"),
                                                  axes, FFTNorm::Backward,
                                                  fft_detail::kMlxFftStream);
        });
}

}  // namespace lucid
