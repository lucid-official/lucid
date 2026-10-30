// lucid/_C/backend/gpu/Shift.h
//
// ``x << y`` and ``x >> y`` on Metal with the reference's shift amounts.
//
// Metal's native shift masks ``y`` modulo the bit width (``1 << 32`` is 1,
// ``1 << -1`` is ``1 << 31``), where an amount outside ``[0, width)`` shifts
// every bit out: 0 for a left shift, the sign fill for a right shift.  That
// is the rule ``CpuBackend::bitwise_binary`` applies, and so the rule here.
//
// An out-of-range amount is replaced before the native shift: by 0 for a
// left shift, whose result is then masked to 0, and by ``width - 1`` for a
// right shift, which leaves exactly the sign fill.  All of it is lazy array
// arithmetic, so no host sync.
#pragma once

#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/ops.h>

namespace lucid::backend {

// ``x`` and ``y`` share a signed integer dtype (the shift ops refuse bool).
template <bool Left>
std::vector<::mlx::core::array> shift_steps(const std::vector<::mlx::core::array>& in) {
    namespace mx = ::mlx::core;
    const mx::array& x = in[0];
    const mx::array& y = in[1];
    const int width = static_cast<int>(x.itemsize()) * 8;
    const mx::array zero(0, y.dtype());
    const mx::array in_range =
        mx::logical_and(mx::greater_equal(y, zero), mx::less(y, mx::array(width, y.dtype())));
    if (Left)
        return {mx::where(in_range, mx::left_shift(x, mx::where(in_range, y, zero)),
                          mx::array(0, x.dtype()))};
    return {mx::right_shift(x, mx::where(in_range, y, mx::array(width - 1, y.dtype())))};
}

// One fused kernel per call, not a pass per step.
inline ::mlx::core::array
gpu_shift(const ::mlx::core::array& x, const ::mlx::core::array& y, bool left) {
    namespace mx = ::mlx::core;
    static const auto shift_left = mx::compile(&shift_steps<true>, /*shapeless=*/true);
    static const auto shift_right = mx::compile(&shift_steps<false>, /*shapeless=*/true);
    return (left ? shift_left : shift_right)({x, y})[0];
}

}  // namespace lucid::backend
