// lucid/_C/compile/OpEmitters/linalg/Decompositions.h
//
// Shared MPSGraph triangular-solve helper used by both
// :class:`SolveTriangularEmitter` (forward / VJP region emission) and
// :class:`SolveTriangularVjp` (standalone reverse-mode VJP emission).

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace lucid::compile {

MPSGraphTensor* emit_solve_triangular_mps(
    MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, bool upper, bool unitriangular);

}  // namespace lucid::compile
