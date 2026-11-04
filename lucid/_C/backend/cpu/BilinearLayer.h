// lucid/_C/backend/cpu/BilinearLayer.h
//
// Bilinear layer backward for the CPU backend, written once over the scalar
// type for float32 and float64.  ``y[b, k] = x1[b] . W[k] . x2[b] + bias[k]``.
//
// Layout: x1 is (B, D1), x2 is (B, D2), W is (Dout, D1, D2), grad is
// (B, Dout).  Every gradient pointer may be null: that gradient is not
// computed, and the operands only it reads may be null too.

#pragma once

#include <cstddef>

#include "../GradMask.h"

namespace lucid::backend::cpu {

template <typename T>
void bilinear_layer_backward_typed(const T* x1p,
                                   const T* x2p,
                                   const T* wp,
                                   const T* gyp,
                                   T* dx1p,
                                   T* dx2p,
                                   T* dwp,
                                   T* dbp,
                                   std::size_t B,
                                   std::size_t D1,
                                   std::size_t D2,
                                   std::size_t Dout) {
    zero_fill(dx1p, B * D1);
    zero_fill(dx2p, B * D2);
    zero_fill(dwp, Dout * D1 * D2);
    zero_fill(dbp, Dout);
    for (std::size_t bi = 0; bi < B; ++bi) {
        const T* x1b = offset_or_null(x1p, bi * D1);
        const T* x2b = offset_or_null(x2p, bi * D2);
        const T* gb = gyp + bi * Dout;
        T* dx1b = offset_or_null(dx1p, bi * D1);
        T* dx2b = offset_or_null(dx2p, bi * D2);
        for (std::size_t k = 0; k < Dout; ++k) {
            const T g = gb[k];
            if (dbp)
                dbp[k] += g;
            if (none_set(dx1b, dx2b, dwp))
                continue;
            const T* Wk = offset_or_null(wp, k * D1 * D2);
            T* dWk = offset_or_null(dwp, k * D1 * D2);
            for (std::size_t i = 0; i < D1; ++i) {
                T row = T{0};
                for (std::size_t j = 0; j < D2; ++j) {
                    if (dx1b)
                        row += Wk[i * D2 + j] * x2b[j];
                    if (dx2b)
                        dx2b[j] += g * x1b[i] * Wk[i * D2 + j];
                    if (dWk)
                        dWk[i * D2 + j] += g * x1b[i] * x2b[j];
                }
                if (dx1b)
                    dx1b[i] += g * row;
            }
        }
    }
}

}  // namespace lucid::backend::cpu
