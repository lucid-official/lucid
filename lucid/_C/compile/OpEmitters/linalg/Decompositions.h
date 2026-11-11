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

// The inverse and the cofactor chain three to six of those eliminations through one
// training step (slogdet: the factor, the inverse, the singular branch's cofactor).
// Measured on an M1 Pro in a conv net whose compiled step saves about 12 ms over the
// eager one, a slogdet step at n = 12 / 16 / 20 / 24 / 28 / 32 costs the compiled step
// 3 / 4 / 6 / 10 / 11 / 15 ms more, so it still beats the eager fallback up to 24 and
// loses from about 30.  Past this the step falls back, as it did before.
constexpr std::int64_t kMaxCofactorDim = 24;

// ``getrf`` of a float32 ``(*, m, n)`` tensor: the packed factor, the one-based
// pivots (int32, ``(*, min(m, n))``), ``info`` (int32, ``(*)``), and the
// determinant it implies (``(*)``, meaningful for a square matrix) and the sign
// of the row permutation (``(*)``, +-1).  ``lu`` is nil for a tensor the unrolled
// elimination does not take.
struct LuGraph {
    MPSGraphTensor* lu = nil;
    MPSGraphTensor* pivots = nil;
    MPSGraphTensor* info = nil;
    MPSGraphTensor* det = nil;
    MPSGraphTensor* sign = nil;
};

LuGraph emit_lu_mps(MPSGraph* g, MPSGraphTensor* a);

// ``det`` of a float32 ``(*, n, n)`` tensor as eager forms it: the pivot product of
// the LU of the transpose, so a singular matrix ends on the same signed zero.  Nil
// for a tensor the elimination does not take.
MPSGraphTensor* emit_det_mps(MPSGraph* g, MPSGraphTensor* a);

// The inverse of a float32 ``(*, n, n)`` tensor, ``n <= kMaxCofactorDim``: one
// elimination carries the identity along (``L^-1 P``) and a back substitution
// finishes it.  A singular matrix divides by its zero pivot, so the result is
// inf / nan there -- eager's ``inv`` raises, which a compiled call cannot.
// Nil for a tensor the elimination does not take.
MPSGraphTensor* emit_inverse_mps(MPSGraph* g, MPSGraphTensor* a);

// ``cof(A)``, the adjugate's transpose, of a float32 ``(*, n, n)`` tensor,
// ``n <= kMaxCofactorDim``.  Valid at every rank: ``adj(A) = sgn(P) adj(U) L^-1 P``
// with ``adj(U)`` built from products of U's entries and never a division by a
// pivot, so a zero pivot (an exactly singular matrix, the rank n-1 case whose
// cofactor is the only nonzero gradient) costs nothing.  ``det(A) A^-T`` is
// wrong there.  Nil for a tensor the elimination does not take.
MPSGraphTensor* emit_cofactor_mps(MPSGraph* g, MPSGraphTensor* a);

MPSGraphTensor* emit_solve_triangular_mps(
    MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, bool upper, bool unitriangular);

}  // namespace lucid::compile
