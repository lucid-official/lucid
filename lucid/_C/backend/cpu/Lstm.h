// lucid/_C/backend/cpu/Lstm.h
//
// Single-layer, single-direction LSTM kernels for the CPU backend, written
// once over the scalar type so float32 and float64 share every line.  The
// two 16-bit float formats have no Accelerate kernels and reach these through
// float32 — :class:`CpuBackend` widens them at the door.
//
// Layouts (row-major, sequence-first):
//   input (T, B, I), h0 / hn (B, Hrec), c0 / cn (B, H), output (T, B, Hrec),
//   gates (T, B, 4H) holding the post-activation [i, f, g, o],
//   cells (T + 1, B, H) holding c_0 .. c_T.
// Hrec is proj_size when the projected variant is on, otherwise H.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Storage.h"
#include "../IBackend.h"
#include "Blas.h"
#include "Vforce.h"

namespace lucid::backend::cpu::lstm {

struct Dims {
    int T, B, I, H, P;

    int rec() const noexcept { return P > 0 ? P : H; }
    int gates() const noexcept { return 4 * H; }
};

inline Dims dims_of(const IBackend::LstmOpts& o) noexcept {
    return {o.seq_len, o.batch_size, o.input_size, o.hidden_size, o.proj_size};
}

inline void gemm(bool ta,
                 bool tb,
                 int M,
                 int N,
                 int K,
                 const float* A,
                 int lda,
                 const float* Bm,
                 int ldb,
                 float beta,
                 float* C,
                 int ldc) {
    sgemm(ta, tb, M, N, K, 1.0f, A, lda, Bm, ldb, beta, C, ldc);
}

inline void gemm(bool ta,
                 bool tb,
                 int M,
                 int N,
                 int K,
                 const double* A,
                 int lda,
                 const double* Bm,
                 int ldb,
                 double beta,
                 double* C,
                 int ldc) {
    dgemm(ta, tb, M, N, K, 1.0, A, lda, Bm, ldb, beta, C, ldc);
}

template <class S>
const S* in(const Storage& s) {
    return reinterpret_cast<const S*>(std::get<CpuStorage>(s).ptr.get());
}

template <class S>
S* out(CpuStorage& s) {
    return reinterpret_cast<S*>(s.ptr.get());
}

inline CpuStorage alloc(std::size_t numel, Dtype dt) {
    const std::size_t nb = count_nbytes(numel, dtype_size(dt));
    return CpuStorage{allocate_aligned_bytes(nb, Device::CPU), nb, dt};
}

inline std::size_t n(int a, int b, int c = 1) {
    return static_cast<std::size_t>(a) * static_cast<std::size_t>(b) * static_cast<std::size_t>(c);
}

// Rebuild the unprojected hidden vector h_raw = o * tanh(c) for one step.
template <class S>
void rebuild_h_raw(const Dims& d, const S* o_gate, const S* c_step, S* h_raw) {
    const int B = d.B, H = d.H, fH = d.gates();
    vtanh(c_step, h_raw, n(B, H));
    for (int b = 0; b < B; ++b)
        for (int k = 0; k < H; ++k)
            h_raw[n(b, H) + k] *= o_gate[n(b, fH) + k];
}

// Turn one step's pre-activations into gates and advance the cell:
// c_t = f * c_{t-1} + i * g, h_raw = o * tanh(c_t).
template <class S>
void activate_step(const Dims& d, const S* raw, const S* c_prev, S* gates, S* c_next, S* h_raw) {
    const int B = d.B, H = d.H, fH = d.gates();
    const std::size_t total_g = n(B, fH);
    for (std::size_t j = 0; j < total_g; ++j)
        gates[j] = -raw[j];
    vexp(gates, gates, total_g);
    for (int b = 0; b < B; ++b) {
        const S* rb = raw + n(b, fH);
        S* gb = gates + n(b, fH);
        vtanh(rb + 2 * H, gb + 2 * H, static_cast<std::size_t>(H));
        for (int k = 0; k < H; ++k) {
            gb[k] = S(1) / (S(1) + gb[k]);
            gb[H + k] = S(1) / (S(1) + gb[H + k]);
            gb[3 * H + k] = S(1) / (S(1) + gb[3 * H + k]);
            c_next[n(b, H) + k] = gb[H + k] * c_prev[n(b, H) + k] + gb[k] * gb[2 * H + k];
        }
    }
    rebuild_h_raw(d, gates + 3 * H, c_next, h_raw);
}

// Returns {output, hn, cn, gates, cells}; see the file header for layouts.
template <class S>
std::vector<Storage> forward_train(const Storage& input,
                                   const Storage& h0,
                                   const Storage& c0,
                                   const std::vector<Storage>& w,
                                   const IBackend::LstmOpts& opts,
                                   Dtype dt) {
    const Dims d = dims_of(opts);
    const int B = d.B, I = d.I, H = d.H, P = d.P, Hr = d.rec(), fH = d.gates();

    CpuStorage out_s = alloc(n(d.T, B, Hr), dt), hn_s = alloc(n(B, Hr), dt);
    CpuStorage cn_s = alloc(n(B, H), dt), gates_s = alloc(n(d.T, B, fH), dt);
    CpuStorage cells_s = alloc(n(d.T + 1, B, H), dt);

    const S* X = in<S>(input);
    const S* Wih = in<S>(w[0]);
    const S* Whh = in<S>(w[1]);
    const S* Bih = in<S>(w[2]);
    const S* Bhh = in<S>(w[3]);
    const S* Whr = P > 0 ? in<S>(w[4]) : nullptr;
    S* Y = out<S>(out_s);
    S* G = out<S>(gates_s);
    S* C = out<S>(cells_s);

    std::copy_n(in<S>(c0), n(B, H), C);
    std::vector<S> bias(static_cast<std::size_t>(fH));
    for (int k = 0; k < fH; ++k)
        bias[static_cast<std::size_t>(k)] = Bih[k] + Bhh[k];

    std::vector<S> raw(n(B, fH)), h_raw(n(B, H));
    const S* h_prev = in<S>(h0);
    for (int t = 0; t < d.T; ++t) {
        for (int b = 0; b < B; ++b)
            std::copy_n(bias.data(), n(fH, 1), raw.data() + n(b, fH));
        gemm(false, true, B, fH, I, X + n(t, B, I), I, Wih, I, S(1), raw.data(), fH);
        gemm(false, true, B, fH, Hr, h_prev, Hr, Whh, Hr, S(1), raw.data(), fH);
        activate_step(d, raw.data(), C + n(t, B, H), G + n(t, B, fH), C + n(t + 1, B, H),
                      h_raw.data());
        S* yt = Y + n(t, B, Hr);
        if (P > 0)
            gemm(false, true, B, P, H, h_raw.data(), H, Whr, H, S(0), yt, P);
        else
            std::copy_n(h_raw.data(), n(B, H), yt);
        h_prev = yt;
    }

    std::copy_n(h_prev, n(B, Hr), out<S>(hn_s));
    std::copy_n(C + n(d.T, B, H), n(B, H), out<S>(cn_s));
    return {Storage{std::move(out_s)}, Storage{std::move(hn_s)}, Storage{std::move(cn_s)},
            Storage{std::move(gates_s)}, Storage{std::move(cells_s)}};
}

// The recurrent input of step t: h0 at t == 0, otherwise the (possibly
// projected) o_{t-1} * tanh(c_{t-1}) rebuilt from the saved trajectory.
template <class S>
void recurrent_input(const Dims& d,
                     int t,
                     const S* G,
                     const S* C,
                     const S* H0,
                     const S* Whr,
                     S* scratch,
                     S* h_prev) {
    const int B = d.B, H = d.H, fH = d.gates(), Hr = d.rec();
    if (B == 0)
        return;
    if (t == 0) {
        std::copy_n(H0, n(B, Hr), h_prev);
        return;
    }
    S* h_raw = d.P > 0 ? scratch : h_prev;
    rebuild_h_raw(d, G + n(t - 1, B, fH) + 3 * H, C + n(t, B, H), h_raw);
    if (d.P > 0)
        gemm(false, true, B, d.P, H, h_raw, H, Whr, H, S(0), h_prev, d.P);
}

// Chain one step's hidden gradient through the gates.  On return ``dG``
// holds d(pre-activation) for [i, f, g, o] and ``dc`` holds dL/dc_{t-1}.
template <class S>
void gate_grads(const Dims& d,
                const S* gt,
                const S* ct,
                const S* ct_prev,
                const S* dh,
                S* tc_buf,
                S* dc,
                S* dG) {
    const int H = d.H, fH = d.gates();
    vtanh(ct, tc_buf, n(d.B, H));
    for (int b = 0; b < d.B; ++b) {
        const S* gb = gt + n(b, fH);
        S* dg = dG + n(b, fH);
        for (int k = 0; k < H; ++k) {
            const std::size_t j = n(b, H) + static_cast<std::size_t>(k);
            const S i = gb[k], f = gb[H + k], g = gb[2 * H + k], o = gb[3 * H + k];
            const S tc = tc_buf[j];
            const S dc_k = dh[j] * o * (S(1) - tc * tc) + dc[j];
            dg[k] = dc_k * g * i * (S(1) - i);
            dg[H + k] = dc_k * ct_prev[j] * f * (S(1) - f);
            dg[2 * H + k] = dc_k * i * (S(1) - g * g);
            dg[3 * H + k] = dh[j] * tc * o * (S(1) - o);
            dc[j] = dc_k * f;
        }
    }
}

// The hidden gradient of step t in the gates' (size-H) basis: dY_t plus the
// gradient carried back from step t + 1, pulled through W_hr when projected
// (which is also where dW_hr picks up this step's share).
template <class S>
void hidden_grad(const Dims& d,
                 int t,
                 const S* dY,
                 const S* dh_next,
                 const S* G,
                 const S* C,
                 const S* Whr,
                 S* dh_step,
                 S* h_raw,
                 S* dWhr,
                 S* dh_raw) {
    const int B = d.B, H = d.H, fH = d.gates(), Hr = d.rec();
    if (B == 0)
        return;
    for (std::size_t j = 0; j < n(B, Hr); ++j)
        dh_step[j] = dY[n(t, B, Hr) + j] + dh_next[j];
    if (d.P == 0) {
        std::copy_n(dh_step, n(B, H), dh_raw);
        return;
    }
    rebuild_h_raw(d, G + n(t, B, fH) + 3 * H, C + n(t + 1, B, H), h_raw);
    gemm(true, false, d.P, H, B, dh_step, d.P, h_raw, H, S(1), dWhr, H);
    gemm(false, false, B, H, d.P, dh_step, d.P, Whr, H, S(0), dh_raw, H);
}

// BPTT.  Returns {dX, dh0, dc0, dW_ih, dW_hh, db_ih, db_hh[, dW_hr]}.
template <class S>
std::vector<Storage> backward(const Storage& grad_output,
                              const Storage& grad_hn,
                              const Storage& grad_cn,
                              const Storage& input,
                              const Storage& h0,
                              const std::vector<Storage>& w,
                              const Storage& gates_all,
                              const Storage& cells_all,
                              const IBackend::LstmOpts& opts,
                              Dtype dt) {
    const Dims d = dims_of(opts);
    const int B = d.B, I = d.I, H = d.H, P = d.P, Hr = d.rec(), fH = d.gates();

    CpuStorage dX_s = alloc(n(d.T, B, I), dt), dH0_s = alloc(n(B, Hr), dt);
    CpuStorage dC0_s = alloc(n(B, H), dt), dWih_s = alloc(n(fH, I), dt);
    CpuStorage dWhh_s = alloc(n(fH, Hr), dt), dB_s = alloc(n(fH, 1), dt);
    CpuStorage dWhr_s = alloc(n(P, H), dt);
    std::fill_n(out<S>(dWih_s), n(fH, I), S(0));
    std::fill_n(out<S>(dWhh_s), n(fH, Hr), S(0));
    std::fill_n(out<S>(dB_s), n(fH, 1), S(0));
    std::fill_n(out<S>(dWhr_s), n(P, H), S(0));

    const S* X = in<S>(input);
    const S* Wih = in<S>(w[0]);
    const S* Whh = in<S>(w[1]);
    const S* Whr = P > 0 ? in<S>(w[4]) : nullptr;
    const S* G = in<S>(gates_all);
    const S* C = in<S>(cells_all);
    S* dB = out<S>(dB_s);

    std::vector<S> dh(in<S>(grad_hn), in<S>(grad_hn) + n(B, Hr));
    std::vector<S> dc(in<S>(grad_cn), in<S>(grad_cn) + n(B, H));
    std::vector<S> dh_step(n(B, Hr)), dh_raw(n(B, H)), h_raw(n(B, H)), h_prev(n(B, Hr));
    std::vector<S> dG(n(B, fH));

    for (int t = d.T - 1; t >= 0; --t) {
        hidden_grad(d, t, in<S>(grad_output), dh.data(), G, C, Whr, dh_step.data(), h_raw.data(),
                    out<S>(dWhr_s), dh_raw.data());
        gate_grads(d, G + n(t, B, fH), C + n(t + 1, B, H), C + n(t, B, H), dh_raw.data(),
                   h_raw.data(), dc.data(), dG.data());
        recurrent_input(d, t, G, C, in<S>(h0), Whr, h_raw.data(), h_prev.data());

        gemm(false, false, B, I, fH, dG.data(), fH, Wih, I, S(0), out<S>(dX_s) + n(t, B, I), I);
        gemm(false, false, B, Hr, fH, dG.data(), fH, Whh, Hr, S(0), dh.data(), Hr);
        gemm(true, false, fH, I, B, dG.data(), fH, X + n(t, B, I), I, S(1), out<S>(dWih_s), I);
        gemm(true, false, fH, Hr, B, dG.data(), fH, h_prev.data(), Hr, S(1), out<S>(dWhh_s), Hr);
        for (int b = 0; b < B; ++b)
            for (int k = 0; k < fH; ++k)
                dB[k] += dG[n(b, fH) + static_cast<std::size_t>(k)];
    }

    std::copy_n(dh.data(), n(B, Hr), out<S>(dH0_s));
    std::copy_n(dc.data(), n(B, H), out<S>(dC0_s));

    // b_ih and b_hh enter the pre-activation as one sum, so they share a
    // gradient — but each edge gets its own buffer, since the engine adds
    // into what it is handed.
    CpuStorage dBhh_s = alloc(n(fH, 1), dt);
    std::copy_n(out<S>(dB_s), n(fH, 1), out<S>(dBhh_s));
    std::vector<Storage> result{Storage{std::move(dX_s)},   Storage{std::move(dH0_s)},
                                Storage{std::move(dC0_s)},  Storage{std::move(dWih_s)},
                                Storage{std::move(dWhh_s)}, Storage{std::move(dB_s)},
                                Storage{std::move(dBhh_s)}};
    if (P > 0)
        result.emplace_back(Storage{std::move(dWhr_s)});
    return result;
}

}  // namespace lucid::backend::cpu::lstm
