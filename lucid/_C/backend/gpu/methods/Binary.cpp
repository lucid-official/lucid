// lucid/_C/backend/gpu/methods/Binary.cpp
// Out-of-line GpuBackend method definitions for the Binary family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt, [](auto& x, auto& y) { return ::mlx::core::add(x, y); });
}

Storage GpuBackend::sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt,
                      [](auto& x, auto& y) { return ::mlx::core::subtract(x, y); });
}

Storage GpuBackend::mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt,
                      [](auto& x, auto& y) { return ::mlx::core::multiply(x, y); });
}

Storage GpuBackend::div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt, [](auto& x, auto& y) { return ::mlx::core::divide(x, y); });
}

Storage GpuBackend::pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt, [](auto& x, auto& y) { return gpu_pow(x, y); });
}

Storage GpuBackend::bitwise_binary(
    const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) {
    return mlx_binary(a, b, shape, dt, [op](auto& x, auto& y) {
        if (op == 0)
            return ::mlx::core::bitwise_and(x, y);
        if (op == 1)
            return ::mlx::core::bitwise_or(x, y);
        if (op == 2)
            return ::mlx::core::bitwise_xor(x, y);
        if (op == 3 || op == 4)
            return gpu_shift(x, y, op == 3);
        ErrorBuilder("gpu_backend::bitwise_binary").fail("unknown op");
    });
}

Storage
GpuBackend::compare_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype, int op) {
    return mlx_binary(a, b, shape, Dtype::Bool, [op](auto& x, auto& y) {
        if (op == 0)
            return ::mlx::core::equal(x, y);
        if (op == 1)
            return ::mlx::core::not_equal(x, y);
        if (op == 2)
            return ::mlx::core::greater(x, y);
        if (op == 3)
            return ::mlx::core::greater_equal(x, y);
        if (op == 4)
            return ::mlx::core::less(x, y);
        if (op == 5)
            return ::mlx::core::less_equal(x, y);
        ErrorBuilder("gpu_backend::compare_binary").fail("unknown op");
    });
}

Storage GpuBackend::maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt, [](auto& x, auto& y) { return ::mlx::core::maximum(x, y); });
}

Storage GpuBackend::minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt, [](auto& x, auto& y) { return ::mlx::core::minimum(x, y); });
}

Storage GpuBackend::where_branch(
    const Storage& grad, const Storage& cond, const Shape& shape, Dtype dt, bool true_branch) {
    const auto& gg = std::get<GpuStorage>(grad);
    const auto& gc = std::get<GpuStorage>(cond);
    auto zero = ::mlx::core::zeros(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
    auto result = true_branch ? ::mlx::core::where(*gc.arr, *gg.arr, zero)
                              : ::mlx::core::where(*gc.arr, zero, *gg.arr);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::masked_fill(
    const Storage& a, const Storage& mask, const Shape&, Dtype dt, double value) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gm = std::get<GpuStorage>(mask);
    ::mlx::core::array v(static_cast<float>(value), gpu::to_mlx_dtype(dt));
    auto result = ::mlx::core::where(*gm.arr, v, *ga.arr);
    return Storage{gpu::wrap_mlx_array(std::move(result), dt)};
}

Storage GpuBackend::pow_scalar(const Storage& a, const Shape&, Dtype dt, double exp) {
    const auto& gs = std::get<GpuStorage>(a);
    auto result = gpu_pow(*gs.arr, exp);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::rpow_scalar(const Storage& a, const Shape&, Dtype dt, double base) {
    const auto& gs = std::get<GpuStorage>(a);
    auto result = gpu_pow(gpu::mlx_scalar(base, gpu::to_mlx_dtype(dt)), *gs.arr);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::clip(const Storage& a, const Shape&, Dtype dt, double min_v, double max_v) {
    const auto& gs = std::get<GpuStorage>(a);
    auto lo = gpu::mlx_scalar(min_v, gpu::to_mlx_dtype(dt));
    auto hi = gpu::mlx_scalar(max_v, gpu::to_mlx_dtype(dt));
    auto result = ::mlx::core::clip(*gs.arr, lo, hi);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

std::pair<Storage, Storage> GpuBackend::expand_and_multiply(const Storage& mask,
                                                            const Storage& x,
                                                            const Shape& mask_shape,
                                                            const Shape& x_shape,
                                                            Dtype dt) {
    (void)mask_shape;
    const auto& gm = std::get<GpuStorage>(mask);
    const auto& gx = std::get<GpuStorage>(x);
    auto full_mask = ::mlx::core::broadcast_to(*gm.arr, gpu::to_mlx_shape(x_shape));
    auto out = ::mlx::core::multiply(full_mask, *gx.arr);
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(full_mask), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)}};
}

Storage GpuBackend::ge_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt, [dt](auto& x, auto& y) {
        return ::mlx::core::astype(::mlx::core::greater_equal(x, y), gpu::to_mlx_dtype(dt));
    });
}

Storage GpuBackend::lt_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
    return mlx_binary(a, b, shape, dt, [dt](auto& x, auto& y) {
        return ::mlx::core::astype(::mlx::core::less(x, y), gpu::to_mlx_dtype(dt));
    });
}

Storage GpuBackend::add_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) {
    return mlx_unary(a, shape, dt, [scalar, dt](auto& x) {
        ::mlx::core::array c(scalar, gpu::to_mlx_dtype(dt));
        return ::mlx::core::add(x, c);
    });
}

Storage GpuBackend::mul_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) {
    return mlx_unary(a, shape, dt, [scalar, dt](auto& x) {
        ::mlx::core::array c(scalar, gpu::to_mlx_dtype(dt));
        return ::mlx::core::multiply(x, c);
    });
}

Storage GpuBackend::in_range_mask(
    const Storage& a, const Shape& shape, Dtype dt, double lo, double hi, bool inclusive) {
    return mlx_unary(a, shape, dt, [lo, hi, dt, inclusive](auto& x) {
        ::mlx::core::array lo_arr(lo, gpu::to_mlx_dtype(dt));
        ::mlx::core::array hi_arr(hi, gpu::to_mlx_dtype(dt));
        using namespace ::mlx::core;
        auto above = inclusive ? greater_equal(x, lo_arr) : greater(x, lo_arr);
        auto below = inclusive ? less_equal(x, hi_arr) : less(x, hi_arr);
        auto in_range = logical_and(above, below);
        return ::mlx::core::astype(in_range, gpu::to_mlx_dtype(dt));
    });
}

Storage GpuBackend::leaky_mask(const Storage& a, const Shape& shape, Dtype dt, double slope) {
    return mlx_unary(a, shape, dt, [slope, dt](auto& x) {
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array one(1.0, gpu::to_mlx_dtype(dt));
        ::mlx::core::array slope_arr(slope, gpu::to_mlx_dtype(dt));
        return ::mlx::core::where(::mlx::core::greater(x, zero), one, slope_arr);
    });
}

Storage GpuBackend::positive_mask(const Storage& a, const Shape& shape, Dtype dt) {
    return mlx_unary(a, shape, dt, [dt](auto& x) {
        ::mlx::core::array zero(0.0, gpu::to_mlx_dtype(dt));
        return ::mlx::core::astype(::mlx::core::greater(x, zero), gpu::to_mlx_dtype(dt));
    });
}

Storage GpuBackend::floordiv(const Storage& a, const Storage& b, const Shape&, Dtype dt) {
    return Storage{gpu::wrap_mlx_array(
        gpu_divmod(DivMod::FloorDiv, *std::get<GpuStorage>(a).arr, *std::get<GpuStorage>(b).arr),
        dt)};
}

Storage GpuBackend::where_op(
    const Storage& cond, const Storage& x, const Storage& y, const Shape&, Dtype dt) {
    const auto& gc = std::get<GpuStorage>(cond);
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gy = std::get<GpuStorage>(y);
    auto out = ::mlx::core::where(*gc.arr, *gx.arr, *gy.arr);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

}  // namespace backend
}  // namespace lucid
