// lucid/_C/backend/gpu/ShapelessFused.h
//
// The one place a GPU composite is compiled with ``shapeless=true``.
//
// A shapeless trace is replayed on shapes it never saw, so MLX recomputes the
// output extents from the inputs it is handed, and its broadcast rule takes the
// larger extent where the eager rule takes the 0 of a 1 against a 0.  Compiled
// shapeless, ``maximum`` of a ``(0,)`` array and a ``(1,)`` array comes back
// ``(1,)`` with one element in its buffer (MLX 0.32):
//
//     f = mx.compile(lambda a, b: mx.maximum(a, b), shapeless=True)
//     f(mx.zeros((0,)), mx.ones((1,))).shape        # (1,)    eager: (0,)
//     f(mx.zeros((2, 0)), mx.ones((2, 1))).shape    # (2, 1)  eager: (2, 0)
//
// ``wrap_mlx_array`` swaps an array with no elements for a fresh empty one, but
// this array is not empty by its own shape, so a result labelled ``(0,)`` kept a
// four-byte buffer (LCD-1062).  An in-place op then failed ``copy_from`` with
// ``nbytes mismatch (GPU)`` (the shifts) or left its receiver on the wrong
// buffer (``maximum_``), and an out-of-place result was a tensor whose storage
// disagreed with its shape.
//
// A broadcast comes out empty only when an operand has no elements, and such a
// call has nothing to fuse, so it runs the steps eagerly.  The shape and dtype of
// every output are then MLX's own whatever the steps compute — several outputs of
// different dtypes, a reduction — where substituting ``zeros(broadcast_shapes)``
// would restate each output's shape and dtype here.
//
// Every shapeless ``compile`` in the engine goes through :class:`ShapelessFused`
// (``test_shapeless_compile_routing.py`` scans the sources to keep it so): a site
// that calls ``mlx::core::compile(..., /*shapeless=*/true)`` itself brings the
// defect back.
#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>

namespace lucid {
namespace gpu {

using FusedArrays = std::vector<::mlx::core::array>;

// Capture-less on purpose: MLX identifies a compiled function by its address, so
// the steps have to be a function (or a lambda that converts to one) for the
// cache to find the trace again.
using FusedSteps = FusedArrays (*)(const FusedArrays&);

// Whether any operand holds no elements — metadata only, nothing is evaluated.
inline bool has_empty_operand(const FusedArrays& ins) {
    return std::any_of(ins.begin(), ins.end(),
                       [](const ::mlx::core::array& a) { return a.size() == 0; });
}

// ``steps`` traced once and replayed for every shape and dtype, except when an
// operand is empty, which runs ``steps`` itself.
class ShapelessFused {
public:
    explicit ShapelessFused(FusedSteps steps)
        : steps_(steps), compiled_(::mlx::core::compile(steps, /*shapeless=*/true)) {}

    FusedArrays operator()(const FusedArrays& ins) const {
        if (has_empty_operand(ins))
            return steps_(ins);
        return compiled_(ins);
    }

private:
    FusedSteps steps_;
    std::function<FusedArrays(const FusedArrays&)> compiled_;
};

// ``Steps{}`` fused over ``ins``, one :class:`ShapelessFused` per ``Steps`` type.
// ``Steps`` must be a default-constructible, stateless callable (a capture-less
// lambda) taking and returning :type:`FusedArrays`.
template <class Steps>
FusedArrays shapeless_fused(const FusedArrays& ins, Steps) {
    static const ShapelessFused fused(
        [](const FusedArrays& in) -> FusedArrays { return Steps{}(in); });
    return fused(ins);
}

// One-input element-wise ``Fn{}`` — ``Fn`` takes an array and returns one.
template <class Fn>
::mlx::core::array fused_unary(const ::mlx::core::array& x, Fn) {
    FusedArrays out =
        shapeless_fused({x}, [](const FusedArrays& in) -> FusedArrays { return {Fn{}(in[0])}; });
    return std::move(out[0]);
}

// Two-input element-wise ``Fn{}`` — ``Fn`` takes two arrays and returns one.
template <class Fn>
::mlx::core::array fused_binary(const ::mlx::core::array& x, const ::mlx::core::array& y, Fn) {
    FusedArrays out = shapeless_fused(
        {x, y}, [](const FusedArrays& in) -> FusedArrays { return {Fn{}(in[0], in[1])}; });
    return std::move(out[0]);
}

}  // namespace gpu
}  // namespace lucid
