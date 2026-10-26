// lucid/_C/backend/gpu/methods/Unary.cpp
// Out-of-line GpuBackend method definitions for the Unary family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::exp(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::exp(x); });
}

Storage GpuBackend::log(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::log(x); });
}

Storage GpuBackend::sqrt(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sqrt(x); });
}

Storage GpuBackend::rsqrt(const Storage& a, const Shape& shape, Dtype dt) {
    if (dt == Dtype::F32) {
        return mlx_unary_fused(a, dt, [](const ::mlx::core::array& x) {
            return ::mlx::core::reciprocal(::mlx::core::sqrt(x));
        });
    }
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::rsqrt(x); });
}

Storage GpuBackend::abs(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::abs(x); });
}

Storage GpuBackend::neg(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::negative(x); });
}

Storage GpuBackend::sign(const Storage& a, const Shape& shape, Dtype dt) {
    if (is_complex(dt))
        ErrorBuilder("gpu_backend::sign").not_implemented("complex not supported");
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sign(x); });
}

Storage GpuBackend::floor(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::floor(x); });
}

Storage GpuBackend::ceil(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::ceil(x); });
}

Storage GpuBackend::round(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::round(x); });
}

Storage GpuBackend::sin(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sin(x); });
}

Storage GpuBackend::cos(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::cos(x); });
}

Storage GpuBackend::tanh(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::tanh(x); });
}

Storage GpuBackend::log2(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::log2(x); });
}

Storage GpuBackend::erf(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::erf(x); });
}

Storage GpuBackend::erfinv(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::erfinv(x); });
}

Storage GpuBackend::reciprocal(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::reciprocal(x); });
}

Storage GpuBackend::square(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::square(x); });
}

Storage GpuBackend::cube(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt,
                     [](auto& x) { return ::mlx::core::multiply(::mlx::core::square(x), x); });
}

Storage GpuBackend::cube_root(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [dt](auto& x) {
        ::mlx::core::array exponent(1.0 / 3.0, gpu::to_mlx_dtype(dt));
        return ::mlx::core::multiply(::mlx::core::sign(x),
                                     ::mlx::core::power(::mlx::core::abs(x), exponent));
    });
}

Storage GpuBackend::tan(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::tan(x); });
}

Storage GpuBackend::asin(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::arcsin(x); });
}

Storage GpuBackend::acos(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::arccos(x); });
}

Storage GpuBackend::atan(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::arctan(x); });
}

Storage GpuBackend::sinh(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::sinh(x); });
}

Storage GpuBackend::cosh(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [](auto& x) { return ::mlx::core::cosh(x); });
}

Storage GpuBackend::invert(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [dt](auto& x) -> ::mlx::core::array {
        if (dt == Dtype::Bool)
            return ::mlx::core::logical_not(x);
        return ::mlx::core::bitwise_invert(x);
    });
}

Storage GpuBackend::complex_real(const Storage& a, const Shape&) {
    const auto& gs = std::get<GpuStorage>(a);
    auto out = ::mlx::core::contiguous(::mlx::core::real(*gs.arr));
    return Storage{gpu::wrap_mlx_array(std::move(out), Dtype::F32)};
}

Storage GpuBackend::complex_imag(const Storage& a, const Shape&) {
    const auto& gs = std::get<GpuStorage>(a);
    auto out = ::mlx::core::contiguous(::mlx::core::imag(*gs.arr));
    return Storage{gpu::wrap_mlx_array(std::move(out), Dtype::F32)};
}

Storage GpuBackend::complex_combine(const Storage& re, const Storage& im, const Shape&) {
    const auto& re_g = std::get<GpuStorage>(re);
    const auto& im_g = std::get<GpuStorage>(im);
    // Laid out, not computed (``complex_from_parts``): built as
    // ``re + 1j·im``, ``complex(inf, inf)`` and ``polar`` of an infinite
    // radius gave a NaN real part on Metal.
    auto re_f = ::mlx::core::astype(*re_g.arr, ::mlx::core::float32);
    auto im_f = ::mlx::core::astype(*im_g.arr, ::mlx::core::float32);
    return Storage{
        gpu::wrap_mlx_array(::mlx::core::contiguous(complex_from_parts(re_f, im_f)), Dtype::C64)};
}

Storage GpuBackend::complex_conj(const Storage& a, const Shape&, Dtype dt) {
    if (dt == Dtype::C128)
        ErrorBuilder("conj").not_implemented(
            "complex128 is not supported on GPU (MLX has no complex128)");
    if (!is_complex(dt)) {
        // For real dtypes the conjugate is the identity — return the
        // input directly without an MLX call.  Testing "not C64" here
        // would have meant a complex128 tensor silently came back
        // unchanged; it is refused above instead.
        return a;
    }
    const auto& gs = std::get<GpuStorage>(a);
    auto out = ::mlx::core::contiguous(::mlx::core::conjugate(*gs.arr));
    return Storage{gpu::wrap_mlx_array(std::move(out), Dtype::C64)};
}

Storage GpuBackend::any(const Storage& a, const Shape&, Dtype dt) {
    return mlx_unary(a, {}, Dtype::Bool,
                     [](auto& x) { return ::mlx::core::any(truth_operand(x)); });
}

Storage GpuBackend::all(const Storage& a, const Shape&, Dtype dt) {
    return mlx_unary(a, {}, Dtype::Bool,
                     [](auto& x) { return ::mlx::core::all(truth_operand(x)); });
}

Storage GpuBackend::isinf(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, Dtype::Bool, [](auto& x) {
        if (is_complex_array(x))
            return ::mlx::core::logical_or(::mlx::core::isinf(::mlx::core::real(x)),
                                           ::mlx::core::isinf(::mlx::core::imag(x)));
        return ::mlx::core::isinf(x);
    });
}

Storage GpuBackend::isnan(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, Dtype::Bool, [](auto& x) { return ::mlx::core::isnan(x); });
}

Storage GpuBackend::isfinite(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, Dtype::Bool, [](auto& x) {
        if (is_complex_array(x))
            return ::mlx::core::logical_and(::mlx::core::isfinite(::mlx::core::real(x)),
                                            ::mlx::core::isfinite(::mlx::core::imag(x)));
        return ::mlx::core::isfinite(x);
    });
}

Storage GpuBackend::nan_to_num(const Storage& a,
                               const Shape& shape,
                               Dtype dt,
                               double nan_val,
                               double posinf_val,
                               double neginf_val) {
    return mlx_unary(a, shape, dt, [dt, nan_val, posinf_val, neginf_val](auto& x) {
        // An integer or bool tensor holds no NaN or infinity, so this is
        // the identity — and the replacement values, a float's range by
        // default, are not representable in it: building them as integer
        // arrays was undefined (the UBSan build halted here).
        if (dt == Dtype::Bool || dt == Dtype::I8 || dt == Dtype::I16 || dt == Dtype::I32 ||
            dt == Dtype::I64)
            return x;
        // A complex number has each part replaced on its own, as the
        // reference does: (1 + inf j) becomes (1 + max j).  MLX compared
        // the complex value as a whole and left it as it was.
        if (is_complex_array(x))
            return complex_from_parts(
                replace_nonfinite(::mlx::core::real(x), nan_val, posinf_val, neginf_val),
                replace_nonfinite(::mlx::core::imag(x), nan_val, posinf_val, neginf_val));
        return replace_nonfinite(x, nan_val, posinf_val, neginf_val);
    });
}

}  // namespace backend
}  // namespace lucid
