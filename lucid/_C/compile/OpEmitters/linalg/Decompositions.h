// lucid/_C/compile/OpEmitters/linalg/Decompositions.h
//
// Shared MPSGraph helpers: the triangular solve used by both
// :class:`SolveTriangularEmitter` (forward / VJP region emission) and
// :class:`SolveTriangularVjp` (standalone reverse-mode VJP emission), and the
// partial-pivoting LU used by ``lu_factor`` and ``det``.

#pragma once

#include <cstdint>

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace lucid::compile {

// The unrolled elimination runs one serial chain of small kernels per step, about
// 0.3 ms each on an M4 Max (n = 64: 21 ms, n = 128: 44 ms, n = 256: 89 ms), against
// about 1 ms for the whole eager LU at any of those sizes.  Past ~32 the compiled
// call costs more than the eager fallback it replaces, so the emitter declines
// there — the same bound the other unrolled decompositions use (kMaxDirectDim).
constexpr std::int64_t kMaxLuDim = 32;

// ``getrf`` of a float32 ``(*, m, n)`` tensor: the packed factor, the one-based
// pivots (int32, ``(*, min(m, n))``), ``info`` (int32, ``(*)``), and the
// determinant it implies (``(*)``, meaningful for a square matrix).  ``lu`` is
// nil for a tensor the unrolled elimination does not take.
struct LuGraph {
    MPSGraphTensor* lu = nil;
    MPSGraphTensor* pivots = nil;
    MPSGraphTensor* info = nil;
    MPSGraphTensor* det = nil;
};

LuGraph emit_lu_mps(MPSGraph* g, MPSGraphTensor* a);

MPSGraphTensor* emit_solve_triangular_mps(
    MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, bool upper, bool unitriangular);

}  // namespace lucid::compile
