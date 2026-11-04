// lucid/_C/backend/cpu/GlobalResponseNorm.h
//
// Global Response Normalization kernels for the CPU backend,
// ``y = gamma * (x * Nx) + beta * x`` with ``Nx = G / (mean_c(G) + eps)``
// and ``G`` the spatial 2-norm of each (batch, channel) plane.  Written once
// over the scalar type for float32 and float64; the 16-bit floats reach
// these through float32 — :class:`CpuBackend` widens them at the door, so a
// half ``x^2`` never overflows (LCD-487).
//
// Layout: x, y, grad and dx are (batch, channels, spatial) row-major, Nx is
// (batch, channels), and gamma, beta and their gradients are (channels,).

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "../GradMask.h"

namespace lucid::backend::cpu {

// G of every plane into ``norms`` (batch * channels) and its channel mean
// into ``means`` (batch).  The forward and the backward both need them.
template <typename T>
void grn_global_norms(const T* x,
                      int batch,
                      int channels,
                      int spatial,
                      std::vector<T>& norms,
                      std::vector<T>& means) {
    norms.assign(static_cast<std::size_t>(batch) * channels, T{0});
    means.assign(static_cast<std::size_t>(batch), T{0});
    for (int b = 0; b < batch; ++b) {
        T sum = T{0};
        for (int c = 0; c < channels; ++c) {
            const std::size_t bc = static_cast<std::size_t>(b) * channels + c;
            const T* xr = x + bc * spatial;
            T acc = T{0};
            for (int s = 0; s < spatial; ++s)
                acc += xr[s] * xr[s];
            norms[bc] = static_cast<T>(std::sqrt(static_cast<double>(acc)));
            sum += norms[bc];
        }
        means[b] = sum / static_cast<T>(channels);
    }
}

template <typename T>
void global_response_norm_forward_typed(const T* x,
                                        const T* gamma,
                                        const T* beta,
                                        T* y,
                                        T* nx,
                                        int batch,
                                        int channels,
                                        int spatial,
                                        double eps) {
    std::vector<T> norms, means;
    grn_global_norms(x, batch, channels, spatial, norms, means);
    for (int b = 0; b < batch; ++b) {
        const T denom = means[b] + static_cast<T>(eps);
        for (int c = 0; c < channels; ++c) {
            const std::size_t bc = static_cast<std::size_t>(b) * channels + c;
            const T n = norms[bc] / denom;
            nx[bc] = n;
            const T* xr = x + bc * spatial;
            T* yr = y + bc * spatial;
            for (int s = 0; s < spatial; ++s)
                yr[s] = gamma[c] * (xr[s] * n) + beta[c] * xr[s];
        }
    }
}

// Per (batch, channel) plane: ``sgx = sum(grad * x)``, which is the whole
// of dgamma (times Nx), of dbeta, and of the term ``a`` the dx path starts
// from.  ``a``, ``dgamma`` and ``dbeta`` are each null when not wanted.
template <typename T>
void grn_backward_planes(const T* x,
                         const T* gamma,
                         const T* nx,
                         const T* grad,
                         T* a,
                         T* dgamma,
                         T* dbeta,
                         int batch,
                         int channels,
                         int spatial) {
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < channels; ++c) {
            const std::size_t bc = static_cast<std::size_t>(b) * channels + c;
            const T* xr = x + bc * spatial;
            const T* gr = grad + bc * spatial;
            T sgx = T{0};
            for (int s = 0; s < spatial; ++s)
                sgx += gr[s] * xr[s];
            if (a)
                a[bc] = sgx * gamma[c];
            if (dgamma)
                dgamma[c] += sgx * nx[bc];
            if (dbeta)
                dbeta[c] += sgx;
        }
}

// dG is the gradient reaching each plane's norm G, through both Nx = G / d
// and the channel mean inside d; dx then adds the path through G = |x|.
template <typename T>
void grn_backward_dx(const T* x,
                     const T* gamma,
                     const T* beta,
                     const T* nx,
                     const T* grad,
                     const std::vector<T>& a,
                     const std::vector<T>& norms,
                     const std::vector<T>& means,
                     T* dx,
                     int batch,
                     int channels,
                     int spatial,
                     double eps) {
    for (int b = 0; b < batch; ++b) {
        const std::size_t row = static_cast<std::size_t>(b) * channels;
        const T denom = means[b] + static_cast<T>(eps);
        T sum_ag = T{0};
        for (int c = 0; c < channels; ++c)
            sum_ag += a[row + c] * norms[row + c];
        const T common = -sum_ag / (denom * denom) / static_cast<T>(channels);
        for (int c = 0; c < channels; ++c) {
            const std::size_t bc = row + c;
            const T dg = a[bc] / denom + common;
            const T inv_norm = (norms[bc] > T{0}) ? T{1} / norms[bc] : T{0};
            const T* xr = x + bc * spatial;
            const T* gr = grad + bc * spatial;
            T* dxr = dx + bc * spatial;
            for (int s = 0; s < spatial; ++s)
                dxr[s] = gamma[c] * nx[bc] * gr[s] + beta[c] * gr[s] + dg * xr[s] * inv_norm;
        }
    }
}

template <typename T>
void global_response_norm_backward_typed(const T* x,
                                         const T* gamma,
                                         const T* beta,
                                         const T* nx,
                                         const T* grad,
                                         T* dx,
                                         T* dgamma,
                                         T* dbeta,
                                         int batch,
                                         int channels,
                                         int spatial,
                                         double eps) {
    if (none_set(dx, dgamma, dbeta))
        return;
    std::vector<T> norms, means;
    if (dx)
        grn_global_norms(x, batch, channels, spatial, norms, means);
    zero_fill(dgamma, static_cast<std::size_t>(channels));
    zero_fill(dbeta, static_cast<std::size_t>(channels));
    std::vector<T> a(dx ? norms.size() : 0, T{0});
    grn_backward_planes(x, gamma, nx, grad, dx ? a.data() : nullptr, dgamma, dbeta, batch, channels,
                        spatial);
    if (dx)
        grn_backward_dx(x, gamma, beta, nx, grad, a, norms, means, dx, batch, channels, spatial,
                        eps);
}

}  // namespace lucid::backend::cpu
